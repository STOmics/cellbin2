# RUN CELLPOSE
import os
import glob
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
from cellbin2.contrib.cellpose_segmentor import instance2semantics, f_instance2semantics_max, poolingOverlap
from cellbin2.image.mask import f_instance2semantics
from cellbin2.image import cbimread, cbimwrite
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


def cellposesam_pred_3c(
    img_path: str, 
    use_gpu, 
    model_dir,
    patch_size: int = 2000,
    overlap: int = 48,
    output_path = None
) -> np.ndarray:

    try:
        import cellpose
    except ImportError:
        pip.main(['install', 'git+https://www.github.com/mouseland/cellpose.git'])
    if cellpose.version != '4.0.8':
        pip.main(['install', 'git+https://www.github.com/mouseland/cellpose.git'])
    import cellpose
    import logging

    from cellpose import models, core, io, plot

    logging.getLogger('cellpose').setLevel(logging.WARNING)
    img = io.imread(img_path)

    if img.ndim == 2:  # gray
        img = np.stack([img, img, img], axis=-1)
        chan = [0, 0]
    elif img.ndim == 3 and img.shape[2] == 3:
        chan = [2, 0]  # RGB
    elif img.ndim == 3 and img.shape[2] != 3: # rgb C H W
        img = np.transpose(img, (1, 2, 0))
        chan = [2, 0]  # RGB
    # patches
    patches, positions = split_image_into_patches(img, patch_size, overlap)
    
    # patch segmentation
    model = models.CellposeModel(gpu = use_gpu, pretrained_model=model_dir)
    masks = []
    for i, patch in enumerate(tqdm.tqdm(patches, desc='Segment cells with [Cellpose]')):
        mask = model.eval(patch, diameter=None, channels=chan)[0]
        boundaries = find_boundaries(mask, mode='inner')
        
        mask[boundaries] = 0
        mask = f_instance2semantics(mask)
        masks.append(mask)
    
    # merge mask patches
    full_mask = merge_masks_with_or(masks, positions, img.shape[:2])
    if output_path:
        name = os.path.splitext(os.path.basename(img_path))[0]
        c_mask_path = os.path.join(output_path, f"{name}_cpsam_mask.tif")
        cbimwrite(output_path=c_mask_path, files=full_mask, compression=True)

    return full_mask


def cellposesam_pred(img_path, 
                   use_gpu, 
                   model_dir,
                   cfg=None, 
                   output_path=None,
                   photo_size=2048,
                   photo_step=2000,):
    try:
        import cellpose
    except ImportError:
        pip.main(['install', 'git+https://www.github.com/mouseland/cellpose.git'])
    if cellpose.version != '4.0.8':
        pip.main(['install', 'git+https://www.github.com/mouseland/cellpose.git'])
    import cellpose

    from cellpose import models, core, io, plot
    try:
        import patchify
    except ImportError:
        pip.main(['install', 'patchify==0.2.3'])
    import patchify
    use_gpu = False
    overlap = photo_size - photo_step
    if (overlap % 2) == 1:
        overlap = overlap + 1
    act_step = ceil(overlap / 2)
    logging.getLogger('cellpose.models').setLevel(logging.WARNING)
    model = models.CellposeModel(gpu = use_gpu, pretrained_model=model_dir)
    img = io.imread(img_path)
    img = f_ij_16_to_8(img)
    img = f_rgb2gray(img, True)

    res_image = np.pad(img, ((act_step, act_step), (act_step, act_step)), 'constant')
    res_a = res_image.shape[0]
    res_b = res_image.shape[1]
    re_length = ceil((res_a - (photo_size - photo_step)) / photo_step) * photo_step + (
            photo_size - photo_step)
    re_width = ceil((res_b - (photo_size - photo_step)) / photo_step) * photo_step + (
            photo_size - photo_step)
    regray_image = np.pad(res_image, ((0, re_length - res_a), (0, re_width - res_b)), 'constant')
    patches = patchify.patchify(regray_image, (photo_size, photo_size), step=photo_step)
    wid = patches.shape[0]
    high = patches.shape[1]
    a_patches = np.full((wid, high, (photo_size - overlap), (photo_size - overlap)), 255, dtype=np.uint8)

    for i in tqdm.tqdm(range(wid), desc='Segment cells with [Cellpose]'):
        for j in range(high):
            img_data = patches[i, j, :, :]
            masks = model.eval(img_data, diameter=None, channels=[0, 0])[0]
            masks = f_instance2semantics_max(masks)
            a_patches[i, j, :, :] = masks[act_step:(photo_size - act_step),
                                    act_step:(photo_size - act_step)]

    patch_nor = patchify.unpatchify(a_patches,
                                    ((wid) * (photo_size - overlap), (high) * (photo_size - overlap)))
    nor_imgdata = np.array(patch_nor)
    after_wid = patch_nor.shape[0]
    after_high = patch_nor.shape[1]
    cropped_1 = nor_imgdata[0:(after_wid - (re_length - res_a)), 0:(after_high - (re_width - res_b))]
    cropped_1 = np.uint8(remove_small_objects(cropped_1 > 0, min_size=2))
    if output_path is not None:
        name = os.path.splitext(os.path.basename(img_path))[0]
        c_mask_path = os.path.join(output_path, f"{name}_v3_mask.tif")
        cbimwrite(output_path=c_mask_path, files=cropped_1, compression=True)
    return cropped_1

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

    mask =  cellposesam_pred_3c(img_path, 
                    use_gpu = use_gpu, 
                    model_dir = model_path,
                    output_path=output_path)
