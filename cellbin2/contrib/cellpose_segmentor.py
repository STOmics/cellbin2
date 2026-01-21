import os
import sys
import cv2
from math import ceil
import pip
import tqdm
import numpy.typing as npt
import numpy as np
from skimage.morphology import remove_small_objects

from cellbin2.image.augmentation import f_ij_16_to_8_v2 as f_ij_16_to_8
from cellbin2.image.augmentation import f_rgb2gray
from cellbin2.image.mask import f_instance2semantics
from cellbin2.image import cbimread, cbimwrite
from cellbin2.dnn.segmentor.postprocess import watershed_segmentation,f_postprocess_rna
from cellbin2.contrib.cell_segmentor import CellSegParam
from cellbin2.utils import clog

from typing import Tuple, List 



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

            patch = image[y_start:y_end, x_start:x_end]
            
            # padding
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                pad_h = patch_size - patch.shape[0]
                pad_w = patch_size - patch.shape[1]
                patch = np.pad(patch, ((0, pad_h), (0, pad_w)), mode='constant')
            
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


def asStride(arr, sub_shape, stride):
    """
    Get a strided sub-matrices view of an ndarray.

    This function is similar to `skimage.util.shape.view_as_windows()`.

    Args:
        arr (ndarray): The input ndarray.
        sub_shape (tuple): The shape of the sub-matrices.
        stride (tuple): The step size along each axis.

    Returns:
        ndarray: A view of strided sub-matrices.
    """
    s0, s1 = arr.strides[:2]
    m1, n1 = arr.shape[:2]
    m2, n2 = sub_shape
    view_shape = (1 + (m1 - m2) // stride[0], 1 + (n1 - n2) // stride[1], m2, n2) + arr.shape[2:]
    strides = (stride[0] * s0, stride[1] * s1, s0, s1) + arr.strides[2:]
    subs = np.lib.stride_tricks.as_strided(arr, view_shape, strides=strides)
    return subs


def poolingOverlap(mat, ksize, stride=None, method='max', pad=False):
    """
    Perform overlapping pooling on 2D or 3D data.

    Args:
        mat (ndarray): The input array to pool.
        ksize (tuple of 2): Kernel size in (ky, kx).
        stride (tuple of 2, optional): Stride of the pooling window. If None, it defaults to the kernel size (non - overlapping pooling).
        method (str, optional): Pooling method, 'max' for max - pooling, 'mean' for mean - pooling.
        pad (bool, optional): Whether to pad the input matrix or not. If not padded, the output size will be (n - f)//s+1, where n is the matrix size, f is the kernel size, and s is the stride. If padded, the output size will be ceil(n/s).

    Returns:
        ndarray: The pooled matrix.
    """

    m, n = mat.shape[:2]
    ky, kx = ksize
    if stride is None:
        stride = (ky, kx)
    sy, sx = stride

    _ceil = lambda x, y: int(np.ceil(x / float(y)))

    # Replace zeros with NaNs to handle them in max and mean calculations
    mat = np.where(mat == 0, np.nan, mat)

    if pad:
        # Calculate the padded size
        ny = _ceil(m, sy)
        nx = _ceil(n, sx)
        size = ((ny - 1) * sy + ky, (nx - 1) * sx + kx) + mat.shape[2:]
        mat_pad = np.full(size, np.nan)
        mat_pad[:m, :n, ...] = mat
    else:
        # Ensure the matrix is large enough for the kernel if not padding
        mat_pad = mat[:(m - ky) // sy * sy + ky, :(n - kx) // sx * sx + kx, ...]

    # Create a view of the matrix with the specified stride
    view = asStride(mat_pad, ksize, stride)
    if method == 'max':
        # Perform max-pooling and convert NaNs back to zeros
        result = np.nanmax(view, axis=(2, 3))
    else:
        # Perform mean-pooling and convert NaNs back to zeros
        result = np.nanmean(view, axis=(2, 3))
    result = np.nan_to_num(result)
    return result


def f_instance2semantics_max(ins):
    """
    Processes an instance segmentation mask to remove small objects and converts it to a semantic segmentation mask.

    Args:
        ins (numpy.ndarray): The instance segmentation mask.

    Returns:
        numpy.ndarray: The semantic segmentation mask.
    """
    ins_m = poolingOverlap(ins, ksize=(2, 2), stride=(1, 1), pad=True, method='mean')
    mask = np.uint8(np.subtract(np.float64(ins), ins_m))
    ins[mask != 0] = 0
    ins = f_instance2semantics(ins)
    return ins


def main(
    file_path: str, 
    gpu,
    model_dir: str,
    stain_type= None,
    output_path=None,
    patch_size: int = 2000,
    overlap: int = 15
) -> np.ndarray:

    try:
        import cellpose
    except ImportError:
        pip.main(['install', 'cellpose==3.1.1.2'])
    if cellpose.version != '3.1.1.2':
        pip.main(['install', 'cellpose==3.1.1.2'])
    import cellpose
    try:
        import patchify
    except ImportError:
        pip.main(['install', 'patchify==0.2.3'])
    from cellpose import models,io
    import patchify
    import logging
    logging.getLogger('cellpose.models').setLevel(logging.WARNING)
    img = io.imread(file_path)

    # patches
    patches, positions = split_image_into_patches(img, patch_size, overlap)
    
    # patch segmentation
    model = models.CellposeModel(gpu = gpu, pretrained_model=model_dir)
    masks = []
    for i, patch in enumerate(tqdm.tqdm(patches, desc='Segment cells with [Cellpose]')):
        mask = model.eval(patch, diameter=None, channels=[0, 0])[0]
        mask = f_instance2semantics_max(mask)
        num_cells, instance_mask = cv2.connectedComponents(
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
                new_mask[instance_mask == i] = 1
        masks.append(new_mask)
    
    # merge mask patches
    full_mask = merge_masks_with_or(masks, positions, img.shape[:2])
    #full_mask = apply_watershed(full_mask)
    full_mask = f_postprocess_rna(full_mask)

    if output_path:
        name = os.path.splitext(os.path.basename(file_path))[0]
        c_mask_path = os.path.join(output_path, f"{name}_cellpose_mask.tif")
        cbimwrite(output_path=c_mask_path, files=full_mask, compression=True)

    return full_mask

demo = """
python cellpose_segmentor.py \
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


def segment4cell(input_path: str, cfg: CellSegParam, use_gpu: bool, stain_type: str) -> npt.NDArray[np.uint8]:
    model_dir = getattr(cfg, f"{stain_type}_weights_path")
    mask = main(
        file_path=input_path,
        gpu=use_gpu,
        model_dir=model_dir,
        stain_type= stain_type
    )
    return mask


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(usage=f"{demo}")
    parser.add_argument('-i', "--input", help="the input img path")
    parser.add_argument('-o', "--output", help="the output file")
    parser.add_argument("-m", "--model_dir", help="model dir")
    parser.add_argument("-n", "--model_name", help="model name", default="cyto2torch_0")
    parser.add_argument("-g", "--gpu", help="the gpu index", default="-1")

    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    model_name = args.model_name
    gpu = args.gpu
    model_dir = args.model_dir
    model_path = os.path.join(model_dir, model_name)

    main(
        file_path=input_path,
        gpu=gpu,
        model_dir=model_path,
        output_path=output_path
    )
    sys.exit()

    # model = r'E:\03.users\liuhuanlin\01.data\cellbin2\weights'
    # input_path = r'E:\03.users\liuhuanlin\01.data\cellbin2\output\B03624A2_DAPI_10X.tiff'
    # cfg = CellSegParam(**{'IF_weights_path': model, 'GPU': 0})
    # mask = segment4cell(input_path, cfg)
    # cbimwrite(r'E:\03.users\liuhuanlin\01.data\cellbin2\output\res_mask.tiff', mask)
