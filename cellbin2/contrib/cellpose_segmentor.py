import os
import sys

import cv2
from math import ceil
import pip
import tqdm
import numpy.typing as npt
import numpy as np
from cellbin2.image import cbimread, cbimwrite
from cellbin2.dnn.segmentor.postprocess import f_postprocess_cellpose
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

def _make_patch_weight_soft(
    patch_h: int,
    patch_w: int,
    edge_fade: int
) -> np.ndarray:
    if edge_fade <= 0:
        return np.ones((patch_h, patch_w), dtype=np.float32)

    y = np.arange(patch_h, dtype=np.float32)
    x = np.arange(patch_w, dtype=np.float32)

    dist_top = y
    dist_bottom = patch_h - 1 - y
    dist_left = x
    dist_right = patch_w - 1 - x

    dy = np.minimum(dist_top, dist_bottom)
    dx = np.minimum(dist_left, dist_right)
    d = np.minimum(dy[:, None], dx[None, :])

    w = np.clip(d / float(edge_fade), 0.0, 1.0)
    w = 0.05 + 0.95 * w
    return w.astype(np.float32)


def merge_masks_with_overlap_only_max(
    masks: List[np.ndarray],
    positions: List[Tuple[int, int, int, int]],
    original_shape: Tuple[int, int],
    overlap: int = 48,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Semantic-mask merge with strict separation:

    1) non-overlap region: copied directly, never modified again
    2) overlap region only: weighted max fusion

    Args:
        masks: list of semantic masks (0/1)
        positions: list of (y_start, x_start, y_end, x_end)
        original_shape: (H, W)
        overlap: used only for edge-fade weight width
        threshold: threshold for overlap-region weighted max score

    Returns:
        merged semantic mask, uint8
    """
    h, w = original_shape

    # real coverage from actual patch positions
    coverage = np.zeros((h, w), dtype=np.uint16)
    for y_start, x_start, y_end, x_end in positions:
        coverage[y_start:y_end, x_start:x_end] += 1

    non_overlap_global = (coverage == 1)
    overlap_global = (coverage > 1)

    # final output
    full_mask = np.zeros((h, w), dtype=np.uint8)

    # overlap score canvas only
    full_score = np.zeros((h, w), dtype=np.float32)

    edge_fade = max(1, overlap)

    for mask, (y_start, x_start, y_end, x_end) in zip(masks, positions):
        patch_h = y_end - y_start
        patch_w = x_end - x_start

        valid_mask = mask[:patch_h, :patch_w].astype(np.uint8)
        weight = _make_patch_weight_soft(patch_h, patch_w, edge_fade=edge_fade)

        # global masks restricted to this patch window
        non_overlap_roi = non_overlap_global[y_start:y_end, x_start:x_end]
        overlap_roi = overlap_global[y_start:y_end, x_start:x_end]

        # 1) only fill non-overlap pixels here; these pixels are unique-owner pixels
        if np.any(non_overlap_roi):
            dst = full_mask[y_start:y_end, x_start:x_end]
            dst[non_overlap_roi] = valid_mask[non_overlap_roi]
            full_mask[y_start:y_end, x_start:x_end] = dst

        # 2) only accumulate score on overlap pixels
        if np.any(overlap_roi):
            patch_score = valid_mask.astype(np.float32) * weight
            roi_score = full_score[y_start:y_end, x_start:x_end]
            roi_score[overlap_roi] = np.maximum(roi_score[overlap_roi], patch_score[overlap_roi])
            full_score[y_start:y_end, x_start:x_end] = roi_score

    # 3) finalize overlap pixels only
    full_mask[overlap_global] = (full_score[overlap_global] > threshold).astype(np.uint8)

    return full_mask



def cellpose_instance2semantics(
    ins: np.ndarray,
    boundary_expand: int = 0,
    keep_outer_boundary: bool = True
) -> np.ndarray:
    """
    Robustly convert instance mask to semantic mask.

    Principle:
    - Start from foreground mask: ins > 0
    - Detect pixels that touch a DIFFERENT non-zero instance in 8-neighborhood
    - Remove only those inter-instance boundary pixels
    - Optionally dilate the removed boundary slightly

    Args:
        ins:
            Instance mask. 0 = background, >0 = instance id.
        boundary_expand:
            Extra dilation iterations on detected inter-instance boundary.
            0 means only remove the direct touching boundary.
            1 is sometimes useful if merge later tends to reconnect thin gaps.
        keep_outer_boundary:
            If True, only remove boundaries between different non-zero instances.
            Foreground-background outer contour is kept.
            This is usually what you want for semantic mask.

    Returns:
        Semantic binary mask, uint8, values in {0, 1}.
    """
    ins = np.asarray(ins)
    if ins.ndim != 2:
        raise ValueError(f"`ins` must be 2D, got shape={ins.shape}")

    h, w = ins.shape
    fg = ins > 0
    boundary = np.zeros((h, w), dtype=bool)

    # 8-neighborhood
    shifts = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1),
    ]

    for dy, dx in shifts:
        # source window on current image
        y1 = max(0, -dy)
        y2 = min(h, h - dy)
        x1 = max(0, -dx)
        x2 = min(w, w - dx)

        # shifted neighbor window
        yy1 = max(0, dy)
        yy2 = min(h, h + dy)
        xx1 = max(0, dx)
        xx2 = min(w, w + dx)

        center = ins[y1:y2, x1:x2]
        neigh = ins[yy1:yy2, xx1:xx2]

        if keep_outer_boundary:
            # only detect boundary between DIFFERENT non-zero instances
            diff = (center > 0) & (neigh > 0) & (center != neigh)
        else:
            # also treat fg-bg transitions as removable boundary
            diff = (center != neigh) & ((center > 0) | (neigh > 0))

        boundary[y1:y2, x1:x2] |= diff

    if boundary_expand > 0:
        kernel = np.ones((3, 3), np.uint8)
        boundary = cv2.dilate(
            boundary.astype(np.uint8),
            kernel,
            iterations=boundary_expand
        ).astype(bool)

    sem = fg.astype(np.uint8)
    sem[boundary] = 0
    return sem

def build_overlap_mask(
    positions: List[Tuple[int, int, int, int]],
    original_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Build real overlap mask from actual patch positions.

    A pixel is marked True if it is covered by more than one patch.

    Args:
        positions: list of (y_start, x_start, y_end, x_end)
        original_shape: (H, W)

    Returns:
        overlap_mask: bool ndarray, shape (H, W)
    """
    h, w = original_shape
    coverage = np.zeros((h, w), dtype=np.uint16)

    for y_start, x_start, y_end, x_end in positions:
        coverage[y_start:y_end, x_start:x_end] += 1
    overlap_mask = coverage > 1
    return overlap_mask

def main(
    file_path: str, 
    gpu,
    model_dir: str,
    stain_type= None,
    output_path=None,
    patch_size: int = 4096,
    overlap: int = 48
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

    # mark overlap area
    overlap_mask = build_overlap_mask(positions, img.shape[:2])
    
    # patch segmentation
    model = models.CellposeModel(gpu = gpu, pretrained_model=model_dir)
    masks = []
    for i, patch in enumerate(tqdm.tqdm(patches, desc='Segment cells with [Cellpose]')):
        if "cyto3" in model_dir:
            mask = model.eval(patch, diameter=None, channels=[0, 0],cellprob_threshold=-2.0, flow_threshold=0)[0]
        else:
            mask = model.eval(patch, diameter=None, channels=[0, 0],cellprob_threshold=-2.0, flow_threshold=0)[0]
        mask = cellpose_instance2semantics(mask)
        masks.append(mask)
    
    # merge mask patches
    full_mask = merge_masks_with_overlap_only_max(
        masks,
        positions,
        img.shape[:2],
        overlap=overlap,
        threshold=0.5
    )
    #full_mask = apply_watershed(full_mask)
    full_mask = f_postprocess_cellpose(full_mask, overlap_mask)

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        name = os.path.splitext(os.path.basename(file_path))[0]
        c_mask_path = os.path.join(output_path, f"{name}_cellpose_mask.tif")
        cbimwrite(output_path=c_mask_path, files=full_mask, compression=True)

    return (full_mask > 0).astype(np.uint8)

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
