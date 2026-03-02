# RUN CELLPOSE
import os
import glob
import sys
import cv2
from math import ceil
import numpy as np
import pip
import tifffile
import argparse
import tqdm
import logging
models_logger = logging.getLogger(__name__)
import cv2
from skimage.morphology import remove_small_objects
from skimage.segmentation import find_boundaries
from typing import Tuple, List 


from cellbin2.image.augmentation import f_ij_16_to_8_v2 as f_ij_16_to_8
from cellbin2.image.augmentation import f_rgb2gray
from cellbin2.contrib.cellpose_segmentor import f_instance2semantics_max, poolingOverlap
from cellbin2.image.mask import f_instance2semantics
from cellbin2.image import cbimread, cbimwrite
#from cellbin2.dnn.segmentor.postprocess import f_postprocess_cellpose
from cellbin2.contrib.cell_segmentor import CellSegParam
from cellbin2.utils import clog



def split_image_into_patches(
    image: np.ndarray, 
    patch_size: int = 2000, 
    overlap: int = 48
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:

    
    h, w = image.shape[:2]
    stride = patch_size - overlap  
    patches = []
    positions = []

    y_steps = max(1, (h - overlap) // stride + 1)
    x_steps = max(1, (w - overlap) // stride + 1)
    
    for y_idx in range(y_steps):
        for x_idx in range(x_steps):
            y_start = y_idx * stride
            x_start = x_idx * stride
            
            # edge patches process
            if y_idx == y_steps - 1:
                y_start = h - patch_size
            if x_idx == x_steps - 1:
                x_start = w - patch_size

            y_start = max(0, y_start)
            x_start = max(0, x_start)
            y_end = min(h, y_start + patch_size)
            x_end = min(w, x_start + patch_size)

            patch = image[y_start:y_end, x_start:x_end, :]
            
            # padding
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                pad_h = patch_size - patch.shape[0]
                pad_w = patch_size - patch.shape[1]
                patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
            
            patches.append(patch)
            positions.append((y_start, x_start, y_end, x_end))
    
    return patches, positions



def merge_masks_with_or(
    masks: List[np.ndarray], 
    positions: List[Tuple[int, int, int, int]], 
    original_shape: Tuple[int, int]
) -> np.ndarray:

    h, w = original_shape
    full_mask = np.zeros((h, w), dtype=np.uint8)
    
    for mask, (y_start, x_start, y_end, x_end) in zip(masks, positions):
        patch_h = y_end - y_start
        patch_w = x_end - x_start
        
        valid_mask = mask[:patch_h, :patch_w]
        
        # process overlap area with logic or 
        full_mask[y_start:y_end, x_start:x_end] = np.logical_or(
            full_mask[y_start:y_end, x_start:x_start+patch_w],
            valid_mask
        ).astype(np.uint8)
    
    return full_mask


def omnipose_pred_3c(
    img_path: str, 
    use_gpu, 
    model_dir = "",
    patch_size: int = 4096,
    overlap: int = 24,
    output_path = None
) -> np.ndarray:

    '''try:
        import cellpose
    except ImportError:
        pip.main(['install', 'git+https://www.github.com/mouseland/cellpose.git'])
    if not cellpose.version.startswith('4.0.'):
        pip.main(['install', 'git+https://www.github.com/mouseland/cellpose.git'])
    import cellpose
    import logging'''

    '''current_dir = os.path.dirname(os.path.abspath(__file__))
    cellpose_omni_path = os.path.join(current_dir, 'omnipose', 'src')
    if cellpose_omni_path not in sys.path:
        sys.path.insert(0, cellpose_omni_path)'''

    from cellpose_omni import models
    #import omnipose
    from cellpose_omni import io, transforms 

    '''from omnipose.gpu import use_gpu
    device, use_GPU = use_gpu()'''

    logging.getLogger('cellpose').setLevel(logging.WARNING)
    img = io.imread(img_path)

    if img.ndim == 2:  # gray
        img = np.stack([img, img, img], axis=-1)
        chan = [0, 0]
    elif img.ndim == 3 and img.shape[2] == 3:
        chan = [1, 3]  # RGB
    elif img.ndim == 3 and img.shape[2] != 3: # rgb C H W
        img = np.transpose(img, (1, 2, 0))
        chan = [1, 3]  # RGB
    # patches

    patches, positions = split_image_into_patches(img, patch_size, overlap)

    # mark overlap area
    overlap_mask = np.zeros(img.shape[:2], dtype=bool)
    h, w = img.shape[:2]

    stride = patch_size - overlap

    for x in range(stride, w, stride):
        x_start = max(0, x - overlap)
        x_end = min(w, x + overlap)
        if x_start < x_end:
            overlap_mask[:, x_start:x_end] = True

    for y in range(stride, h, stride):
        y_start = max(0, y - overlap)
        y_end = min(h, y + overlap)
        if y_start < y_end:
            overlap_mask[y_start:y_end, :] = True
    
    # patch segmentation
    print(chan)
    model = models.CellposeModel(gpu=use_gpu, model_type="cyto2_omni")
    masks = []
    for i, patch in enumerate(tqdm.tqdm(patches, desc='Segment cells with [Cellpose]')):
        mask = model.eval(patch,channels=chan)[0]
        mask = f_instance2semantics_max(mask)
        '''num_cells, instance_mask = cv2.connectedComponents(
            (mask > 0).astype(np.uint8), 
            connectivity=4
        )
        
        sizes = []
        for i in range(1, num_cells):
            sizes.append(np.sum(instance_mask == i))
        
        if not sizes:  
            masks.append(mask)
            continue
        avg_size = np.mean(sizes)
        
        new_mask = np.zeros_like(mask, dtype=np.uint8)
        
        for i in range(1, num_cells):
            cell_size = np.sum(instance_mask == i)
            
            if cell_size <= avg_size * 5:
                new_mask[instance_mask == i] = 1'''
        masks.append(mask)
    
    # merge mask patches
    full_mask = merge_masks_with_or(masks, positions, img.shape[:2])
    #full_mask = f_postprocess_cellpose(full_mask, overlap_mask)
    #full_mask = watershed(full_mask)
    if output_path:
        name = os.path.splitext(os.path.basename(img_path))[0]
        c_mask_path = os.path.join(output_path, f"{name}_omnipose_mask.tif")
        cbimwrite(output_path=c_mask_path, files=full_mask, compression=True)

    return full_mask



demo = """
python cellposesam.py \
-i
"xxx/B02512C5_after_tc_regist.tif"
-o
xxx/tmp
-m
xxx/models
-n
cyto2
-g
0
"""

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(usage=f"{demo}")
    parser.add_argument('-i', "--input", help="the input img path")
    parser.add_argument('-o', "--output", help="the output file")
    parser.add_argument("-m", "--model_path", help="model path")
    parser.add_argument("-g", "--gpu", help="use gpu (1) or not (0)", default=0)

    args = parser.parse_args()
    img_path = args.input
    output_path = args.output
    gpu = args.gpu
    model_path = args.model_path
    use_gpu = True
    if gpu == 0:
        use_gpu = False

    mask =  omnipose_pred_3c(img_path, 
                    use_gpu = use_gpu, 
                    model_dir = model_path,
                    output_path=output_path)
