import os
from os.path import join
from typing import Final, NamedTuple, TypedDict, Tuple
import argparse
from skimage.measure import label
import numpy as np
from scipy import ndimage
import cv2

import sys
# sys.path.append(r"D:\git\tmp_branch\cellbin2") # add cellbin2 to path
from cellbin2.contrib.cellpose_segmentor import f_instance2semantics
from cellbin2.image import cbimread, cbimwrite
from cellbin2.utils.pro_monitor import process_decorator

MAX_INPUT_LABEL_VALUE: Final[int] = np.iinfo(np.uint32).max


# @process_decorator('GiB')
def unique_nonzero_pairs_numpy(masks):
    """Compute the unique pairs between to labeled masks with nonzero labels using numpy.

    Args:
        masks (tuple[np.ndarray, np.ndarray]): The masks to compare and
            generated unique pairings.

    Returns:
        np.ndarray[tuple[int, int], np.intp]: A matrix of shape `(p, 2)`
            containing the `p` unique pairs.
        np.ndarray[tuple[int, int], int]: An array of shape `(p,)` of counts
            specifying how many times each pair occurred.
    """
    mask0 = masks[0].ravel()
    mask1 = masks[1].ravel()
    # Find pairs where both labels are non-zero
    valid_indices = (mask0 > 0) & (mask1 > 0)
    valid_mask0 = mask0[valid_indices]
    valid_mask1 = mask1[valid_indices]
    # Combine valid pairs
    combined = np.column_stack((valid_mask0, valid_mask1))
    # Find unique pairs and their counts
    unique_pairs, counts = np.unique(combined, axis=0, return_counts=True)
    return unique_pairs, counts


# @process_decorator('GiB')
def pair_map_by_largest_overlap(masks):
    """Create mappings between two masks, using the largest overlap to pair.

    Args:
        masks (tuple[LabeledMask,LabeledMask]): The masks to compare and
            generated unique pairings.

    Returns:
        np.ndarray[tuple[int], np.dtype[np.uint32]]: A map from the 1st mask to
            the 1st.
        np.ndarray[tuple[int], np.dtype[np.uint32]]: A map from the 2nd mask to
            the 2nd.
    """
    nz_paired_labels, nz_counts = unique_nonzero_pairs_numpy(masks)

    # assign each cell the nuclei with the most overlap
    count_sort_ix = np.argsort(nz_counts, kind="stable")

    mask_a, mask_b = masks

    a_to_b = np.zeros(np.max(mask_a) + 1, dtype=np.uint32)
    a_to_b[nz_paired_labels[count_sort_ix, 0]] = nz_paired_labels[count_sort_ix, 1]
    b_to_a = np.zeros(np.max(mask_b) + 1, dtype=np.uint32)
    b_to_a[nz_paired_labels[count_sort_ix, 1]] = nz_paired_labels[count_sort_ix, 0] #maska 和 maskb的对应关系，细胞核属于哪个膜，膜包含哪个核

    return a_to_b, b_to_a


# @process_decorator('GiB')
def make_mask_consecutive(
        mask,
        start_from: int = 1,
):
    """Given a mask of integers, reassign the labels to be consecutive.

    Args:
        mask (np.ndarray[tuple[int, int], np.uint32]): a mask of positive integers,
        with 0 meaning background, which might not be consecutive

    Returns:
        mask: a new mask where the labels are consecutive integers
    """
    unique_input_labels = np.unique(mask) 
    unique_input_labels = unique_input_labels[unique_input_labels > 0] 
    if unique_input_labels.shape[0] == 0: 
        assert np.all(mask == 0)
        return mask

    num_labels = unique_input_labels.shape[0] 
    max_label = np.max(unique_input_labels)
    assert (
            max_label < MAX_INPUT_LABEL_VALUE 
    ), "Input labels out of range for relabeling procedure"
    label_remapper = np.zeros(max_label + 1, np.uint32) 
    label_remapper[unique_input_labels] = np.arange(start_from, num_labels + start_from)

    return label_remapper[mask]


# @process_decorator('GiB')
def overlap_fractions(
        cell_mask,
        nucleus_mask,
        cells_to_nuclei,
        c=False
):
    """Compute the fraction of overlap area of a nucleus and the cell its assigned to.

    This function assumes:
        - `cells_to_nuclei` is a map from cell index to its assigned nucleus,
          which covers more of the cell than any other nucleus
        - `cell_mask` is labled consecutively and the ith label corresponds to
          the ith index in `cells_to_nuclei`.

    Args:
        cell_mask (LabeledMask): The labeled cell mask.
        nucleus_mask (LabeledMask): The labeled nucleus mask.
        cells_to_nuclei (ndarray): A 1D array mapping cells to their assigned
            nucleus.

    Returns:
        ndarray: 1D array containing the fraction of nucleus area that overlaps
            the cell for each cell.
    """
    cell_labels = np.arange(len(cells_to_nuclei))
    nz_assignments = np.nonzero(cells_to_nuclei)
    nz_cell_labels = cell_labels[nz_assignments]
    nz_cells_to_nuclei = cells_to_nuclei[nz_assignments]

    def _max_overlap(val):
        labels, counts = np.unique(val, return_counts=True)
        nonzero = np.nonzero(labels)
        labels = labels[nonzero]
        counts = counts[nonzero]

        if len(counts) == 0:
            return 0

        return np.max(counts)

    if len(nz_cell_labels) == 0:
        return np.zeros(nz_cell_labels.shape, dtype=np.float64)

    # Gives the counts of the label occurring the most times over each cell
    max_counts = ndimage.labeled_comprehension(
        nucleus_mask, cell_mask, nz_cell_labels, _max_overlap, int, 0
    )
    if c:
        areas = ndimage.labeled_comprehension(
            cell_mask, cell_mask, nz_cell_labels, lambda val: val.shape[0], int, 0
        )
    else:
        # Gives the area of each nucleus in pixels
        areas = ndimage.labeled_comprehension(
            nucleus_mask, nucleus_mask, nz_cells_to_nuclei, lambda val: val.shape[0], int, 0
        ) 

    overlap_frac = np.zeros(cells_to_nuclei.shape, dtype=np.float64)
    overlap_frac[nz_assignments] = max_counts / areas 

    return overlap_frac


class MaskTile(NamedTuple):
    """A class that defines a mask tile with overlap region on the left and above."""

    # the row in the mask where the tile starts
    row_start: int
    # the col in the mask where the tile starts
    col_start: int
    # the end row (exclusive) of the tile
    row_end: int
    # the end col (exclusive) of the tile
    col_end: int
    # the row (exclusive) where overlap region ends
    # equivalently, this is the first row of the unique part of the tile
    unique_row_start: int
    # the col (exclusive) where overlap region ends
    # equivalently, this is the first col of the unique part of the tile
    unique_col_start: int
    
    
def num_n_area(mask):
    if len(np.unique(mask)) > 2:
        _, mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)

    # 计算连通域及其统计信息
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)

    return num_labels - 1


def count_2mask_overlap(nuclei_mask_path, cell_mask_path,save_path):
    sem = 2
    connectivity = 8
    nuclei_mask_raw = cbimread(nuclei_mask_path, only_np=True)
    cell_mask_raw = cbimread(cell_mask_path, only_np=True)
    num_nuclei = num_n_area(nuclei_mask_raw)
    num_cell = num_n_area(cell_mask_raw)
    print(f"DAPI图像总共圈出 {num_nuclei}个细胞核")
    print(f"RNA总共圈出 {num_cell}个细胞")

    if len(np.unique(nuclei_mask_raw)) <= sem:
        _, nuclei_mask = cv2.connectedComponents(nuclei_mask_raw, connectivity=connectivity)
    else:
        nuclei_mask_sem = f_instance2semantics(nuclei_mask_raw)
        _, nuclei_mask = cv2.connectedComponents(nuclei_mask_sem, connectivity=connectivity)
        cbimwrite(join(save_path, f"nuclei_mask_ori.tif"), nuclei_mask_sem * 255)
    if len(np.unique(cell_mask_raw)) <= sem:
        _, cell_mask = cv2.connectedComponents(cell_mask_raw, connectivity=connectivity)
    else:
        cell_mask_sem = f_instance2semantics(cell_mask_raw)
        _, cell_mask = cv2.connectedComponents(cell_mask_sem, connectivity=connectivity)
        cbimwrite(join(save_path, f"cell_mask_ori.tif"), cell_mask_sem * 255)

    cell_mask[:] = make_mask_consecutive(cell_mask)
    nuclei_mask[:] = make_mask_consecutive(nuclei_mask)
    
    overlap_threshold = 0.5
    # Generate mappings between cells and nuclei and vis versa.

    cell_to_interior, interior_to_cell = pair_map_by_largest_overlap(
        (cell_mask, nuclei_mask)
    )
    # interior mask与cell mask的关系处理 细胞质与细胞膜
    # interior & cell overlap > 0.5, 认为interior和cell表示的为一个细胞，interior的去除
    # 0 < interior & cell overlap <= 0.5，认为表示的为两个细胞，interior保留独有的那部分，这里目前可能出现一个细胞分成两个
    # interior & cell overlap = 0，这个比较简单，独有的细胞
    overlap_fracs_interior_to_cell = overlap_fractions(nuclei_mask, cell_mask, interior_to_cell, c=True)#细胞质区域对应的膜
    interior_to_cell_overlap_upper_threshold = overlap_fracs_interior_to_cell >= overlap_threshold
    summ = np.sum(interior_to_cell_overlap_upper_threshold)
    print(f"有{summ}个细胞核与RNA圈细胞重叠（>0.5）")
    print(f"percentage:{summ/num_nuclei}")
    print(f'即有{num_nuclei - summ}个DAPI圈细胞重叠')

    cell_to_interior, interior_to_cell = pair_map_by_largest_overlap(
        (nuclei_mask, cell_mask)
    )

    overlap_fracs_interior_to_cell = overlap_fractions(cell_mask, nuclei_mask, interior_to_cell, c=True)  # 细胞质区域对应的膜
    interior_to_cell_overlap_upper_threshold = overlap_fracs_interior_to_cell >= 0.1
    summ = np.sum(interior_to_cell_overlap_upper_threshold)
    print(f"有{summ}个RNA细胞核与DAPI圈细胞重叠(>0.1)[因为RNA细胞普遍大于DAPI圈细胞，且包含多个DAPI细胞在里面]")
    print(f"percentage:{summ / num_cell}")
    

if __name__ == '__main__':
    #main
    parser = argparse.ArgumentParser(description="Count overlap between two masks")
    
    parser.add_argument('-n', required=True, help='Path to nuclei mask image')
    parser.add_argument('-c', required=True, help='Path to cell mask image')
    parser.add_argument('-o', required=True, help='Directory to save output')

    args = parser.parse_args()
    count_2mask_overlap(args.n, args.c, args.o)
