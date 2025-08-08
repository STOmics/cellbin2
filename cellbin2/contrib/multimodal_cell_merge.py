import os
from os.path import join
from typing import Final, NamedTuple, TypedDict, Tuple

from skimage.measure import label
import numpy as np
from scipy import ndimage
import cv2

from cellbin2.contrib.cellpose_segmentor import f_instance2semantics
from cellbin2.image import cbimread, cbimwrite
from cellbin2.utils.pro_monitor import process_decorator

MAX_INPUT_LABEL_VALUE: Final[int] = np.iinfo(np.uint32).max


# def unique_nonzero_pairs(masks):
#     """Compute the unique pairs between to labeled masks with nonzero labels.
#
#     Args:
#         masks (tuple[LabeledMask,LabeledMask]): The masks to compare and
#             generated unique pairings.
#
#     Returns:
#         np.ndarray[tuple[int,int], np.intp]: A matrix of shape `(p, 2)`
#             containing the `p` unique pairs.
#         np.ndarray[tuple[int,int], int]: An array of shape `(p,)` of counts
#             specifying how many times each pair occured.
#     """
#     # a (p, 2) matrix of pairs of values from cell and nuclei masks.
#     # paired_counts has the number of times that pair is seen
#     paired_labels, paired_counts = np.unique(
#         np.column_stack((masks[0].ravel(), masks[1].ravel())),
#         axis=0,
#         return_counts=True,
#     )
#     # Pairs of (cell-label, nucleus-label), where both the cell and nucleus ID
#     # are non-zero (i.e., not background)
#     nz_pairs = (paired_labels[:, 0] > 0) & (paired_labels[:, 1] > 0)
#     nz_counts = paired_counts[nz_pairs]
#     nz_paired_labels = paired_labels[nz_pairs, :]
#     del paired_labels, paired_counts, nz_pairs
#
#     return nz_paired_labels, nz_counts

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
    b_to_a[nz_paired_labels[count_sort_ix, 1]] = nz_paired_labels[count_sort_ix, 0] 

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
def instance2semantics(ins):
    """
    instance to semantics
    Args:
        ins(ndarray):labeled instance

    Returns(ndarray):mask
    """
    ins[np.where(ins > 0)] = 1
    return np.array(ins, dtype=np.uint8)

# @process_decorator('GiB')
def interior_cell_merge(cell_mask_raw, interior_mask_raw, overlap_threshold=0.5, save_path=""):
    sem = 2
    connectivity = 8
    interior_mask_raw = interior_mask_raw.astype(np.uint8)
    cell_mask_raw = cell_mask_raw.astype(np.uint8)
    
    if len(np.unique(cell_mask_raw)) <= sem:
        _, cell_mask = cv2.connectedComponents(cell_mask_raw, connectivity=connectivity)
    else:
        cell_mask_sem = instance2semantics(cell_mask_raw)
        _, cell_mask = cv2.connectedComponents(cell_mask_sem, connectivity=connectivity)
        cbimwrite(join(save_path, f"cell_mask_ori.tif"), cell_mask_sem * 255)
    if len(np.unique(interior_mask_raw)) <= sem:
        _, interior_mask = cv2.connectedComponents(interior_mask_raw, connectivity=connectivity)
    else:
        interior_mask_sem = instance2semantics(interior_mask_raw)
        _, interior_mask = cv2.connectedComponents(interior_mask_sem, connectivity=connectivity)
        cbimwrite(join(save_path, f"interior_mask_ori.tif"), interior_mask_sem * 255)
    # to instance segmentation and relable all cells
    cell_mask[:] = make_mask_consecutive(cell_mask)
    interior_mask[:] = make_mask_consecutive(interior_mask)

    lower_right_overlap_mask = np.zeros(cell_mask.shape, dtype=bool)
    top_left_bounds_cells = np.ones(cell_mask.shape, dtype=bool) 

    edge_filter_cells = np.logical_and(
        top_left_bounds_cells, np.logical_not(lower_right_overlap_mask)
    )
    num_cells = np.count_nonzero(np.unique(cell_mask[edge_filter_cells]))

    # Generate mappings between cells and nuclei and vis versa.
    cell_to_interior, interior_to_cell = pair_map_by_largest_overlap( 
        (cell_mask, interior_mask)
    )

    # process overlap between interior mask and cell mask
    # if interior & cell overlap > 0.5, consider interior and cell as same cell, remove the interior
    # if 0 < interior & cell overlap <= 0.5，consider interior and cell as two cells，keep the non-overlapping part of interior mask
    # interior & cell overlap = 0，keep both 
    overlap_fracs_interior_to_cell = overlap_fractions(interior_mask, cell_mask, interior_to_cell, c=True)
    interior_to_cell_no_overlap = overlap_fracs_interior_to_cell == 0
    interior_to_cell_overlap_below_threshold = overlap_fracs_interior_to_cell <= overlap_threshold
    interior_keep_idx = np.logical_or(interior_to_cell_no_overlap, interior_to_cell_overlap_below_threshold)
    interior_keep_idx[0] = False  
    interior_keep_bool_mask = interior_keep_idx[interior_mask]
    interior_mask_unique_bool_mask = interior_keep_bool_mask * np.logical_not(cell_mask)
    interior_mask_unique = interior_mask_unique_bool_mask * interior_mask
    # some connected components may be split into separate fragments, relabel
    interior_mask_unique_relabel = label(interior_mask_unique)
    # if interior is split into multiple parts, retain only the largest fragment
    interior_to_interior_unique, interior_unique_to_interior \
        = pair_map_by_largest_overlap((interior_mask, interior_mask_unique_relabel))
    interior_splits_remove_bool_mask = interior_to_interior_unique[interior_mask] == interior_mask_unique_relabel
    interior_mask_final = interior_mask_unique_relabel * interior_splits_remove_bool_mask
    interior_mask_final = make_mask_consecutive(
        interior_mask_final, start_from=np.max(cell_mask) + 1
    )
    print(f"interior index from {np.max(cell_mask) + 1} to {np.max(interior_mask_final)}")
    int_interior_mask_final = instance2semantics(interior_mask_final)
    contours, _ = cv2.findContours(int_interior_mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    interior_boundary = np.zeros_like(int_interior_mask_final)
    cv2.drawContours(interior_boundary, contours, -1, 1, 1)
    cell_mask_add_interior = np.add(cell_mask, interior_mask_final)
    
    cell_mask = cell_mask_add_interior
    save_cell_mask = instance2semantics(cell_mask_add_interior)
    save_cell_mask = np.where(interior_boundary > 0, 0, save_cell_mask)
    if save_path != "":
        cbimwrite(join(save_path, f"cell_mask_add_interior.tif"), instance2semantics(save_cell_mask) * 255)
        cbimwrite(join(save_path, f"interior_mask_final.tif"), instance2semantics(interior_mask_final) * 255)
    #cell_mask = cbimread(join(save_path, f"cell_mask_add_interior.tif"), only_np=True)
    save_cell_mask = save_cell_mask.astype(np.uint8)
    return save_cell_mask
        # partial_metrics,
    

def nuclei_cell_merge(nuclei_mask_raw, cell_mask_raw, overlap_threshold=0.5, save_path=""):
    """
    integrate nuclei to cell mask
    overlap between cell mask and nuclei mask:
        1. nuc has more than 0.5 overlap with cell, save cell only
        2. nuc has less than 0.5 overlap with cell, save cell only
        3. nuc has 0 overlap with cell, save both nuc and cell

    final mask: cell mask +  processed nuclei mask
    """
    sem = 2
    connectivity = 8
    nuclei_mask_raw = nuclei_mask_raw.astype(np.uint8)
    cell_mask_raw = cell_mask_raw.astype(np.uint8)
    if len(np.unique(nuclei_mask_raw)) <= sem: 
        _, nuclei_mask = cv2.connectedComponents(nuclei_mask_raw, connectivity=connectivity)
    else:
        nuclei_mask_sem = instance2semantics(nuclei_mask_raw)
        _, nuclei_mask = cv2.connectedComponents(nuclei_mask_sem, connectivity=connectivity)
        cbimwrite(join(save_path, f"nuclei_mask_ori.tif"), nuclei_mask_sem * 255)
    if len(np.unique(cell_mask_raw)) <= sem:
        _, cell_mask = cv2.connectedComponents(cell_mask_raw, connectivity=connectivity)
    else:
        cell_mask_sem = instance2semantics(cell_mask_raw)
        _, cell_mask = cv2.connectedComponents(cell_mask_sem, connectivity=connectivity)
        cbimwrite(join(save_path, f"cell_mask_ori.tif"), cell_mask_sem * 255)
    
    
    cell_mask_sem = instance2semantics(cell_mask)
    nuclei_mask[:] = make_mask_consecutive(nuclei_mask)
    cell_to_nucleus, nucleus_to_cell = pair_map_by_largest_overlap(
        (cell_mask, nuclei_mask)
    )
    
    
    lower_right_overlap_mask = np.zeros(cell_mask.shape, dtype=bool)
    top_left_bounds_cells = np.ones(cell_mask.shape, dtype=bool) 
    top_left_bounds_nuclei = np.ones(nuclei_mask.shape, dtype=bool)

    # Get the locations where the overlap is less than the threshold. For those
    # cases, set the assignment to zero.
    overlap_fracs = overlap_fractions(cell_mask, nuclei_mask, cell_to_nucleus)
    cell_to_nucleus[np.where(overlap_fracs < overlap_threshold)] = 0 
    del overlap_fracs

    # the output mask for nuclei
    output_nuclei_mask = np.zeros_like(nuclei_mask) 

    # for nuclei that are assigned, trim the part outside the cell
    # matching_mask is a mask over the matrix for the spots where
    # the cell matches the nuclei it was assigned to.
    # we can use this to "trim" the input nuclei mask just to the
    # portion matching the cell.
    matching_ix = np.where((cell_to_nucleus[cell_mask] == nuclei_mask) & (cell_to_nucleus[cell_mask] != 0))
    output_nuclei_mask[matching_ix] = cell_mask[matching_ix]
    del matching_ix

    # indices of cells without a nuclei
    no_nuc = cell_to_nucleus == 0 
    no_nuc[0] = 0


    # get a mask covering cells without a nucleus
    no_nuc_cell_mask = no_nuc[cell_mask] 
    # for cells without a nuclei, set their nucleus equal to cell boundary
    output_nuclei_mask[no_nuc_cell_mask] = cell_mask[no_nuc_cell_mask]

    # remove from the mask duplicate portions of the tile
    # so we can accurately quantify unique no-nuc cells metric
    no_nuc_cell_mask[lower_right_overlap_mask] = 0
    no_nuc_cell_mask[np.logical_not(top_left_bounds_cells)] = 0
    num_cells_no_nuc = np.unique(cell_mask[no_nuc_cell_mask]).shape[0]
    del no_nuc_cell_mask

    # Indices of nuclei without any overlap with a cell. This eliminates cases
    # were a nucleus could be assigned to some cell, but isn't because there is
    # another larger nucleus.
    not_assigned = nucleus_to_cell == 0 
    # not_assigned_by_interior = nucleus_to_interior == 0
    # not_assigned = np.logical_and(not_assigned_by_cell, not_assigned_by_interior)
    not_assigned[0] = False

    # boolean mask for spots in the nuclei mask that are nuclei unassigned to a cell
    unassigned_nuclei_mask = not_assigned[nuclei_mask]

    # indices in the nuclei mask that are nuclei unassigned to a cell
    unassigned_nuclei_indices = np.nonzero(unassigned_nuclei_mask)

    # boudary of the nuclei
    int_unassigned_nuclei_mask = unassigned_nuclei_mask.astype(np.uint8)
    contours, _ = cv2.findContours(int_unassigned_nuclei_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary = np.zeros_like(int_unassigned_nuclei_mask)
    cv2.drawContours(boundary, contours, -1, 1, 1)  

    # mutate the unassigned_nuclei_mask to help quantify accurately
    # our metric for num_nuc_no_cell
    unassigned_nuclei_mask[lower_right_overlap_mask] = False
    unassigned_nuclei_mask[np.logical_not(top_left_bounds_nuclei)] = False 
    # `unassigned_nuc_mask` now holds the mask of nuclei where is there no assigned
    # cell, which is not in the lower-right overlap. Each label will be repeated the
    # number of times it appears in the mask. Taking the unique values gives us
    # the labels of each nucleus without a corresponding cell.
    num_nuc_no_cell = np.count_nonzero(np.unique(nuclei_mask[unassigned_nuclei_mask]))
    del unassigned_nuclei_mask

    # for every nucleus not assigned a cell,
    # make it its own cell with same boundaries and new consecutive
    # mask labels starting from the current biggest cell mask label
    unassigned_nuc_values = nuclei_mask[unassigned_nuclei_indices]
    del nuclei_mask

    start_from = np.max(cell_mask_sem) + 1
    # assert start_from == np.max(output_nuclei_mask) + 1
    if unassigned_nuc_values.shape[0] > 0:
        consecutive_unassigned_nuc = make_mask_consecutive(
            unassigned_nuc_values, start_from=start_from
        )
        print(f"nuc index from {start_from} to {np.max(consecutive_unassigned_nuc)}")
        cell_mask_sem[unassigned_nuclei_indices] = consecutive_unassigned_nuc
        cell_mask[unassigned_nuclei_indices] = consecutive_unassigned_nuc
        output_nuclei_mask[unassigned_nuclei_indices] = consecutive_unassigned_nuc
    cell_mask_add_interior_sem = instance2semantics(cell_mask_sem)
    output_nuclei_mask = output_nuclei_mask - cell_mask_sem
    output_nuclei_mask = np.where(output_nuclei_mask < 0, 0, output_nuclei_mask) 
    # cell_mask_add_interior_sem[unassigned_nuclei_indices] = 255
    # cell_mask_add_interior_sem[np.nonzero(np.logical_and(cell_mask_add_interior_sem, interior_keep_mask_))] = 125
    # cell_mask_add_interior_sem[cell_mask_add_interior_sem == 1] = 50
    save_cell_mask_add_interior_sem = np.where(boundary > 0, 0, cell_mask_add_interior_sem)
    print(f"# of cells: {np.max(cell_mask) + 1}")
    if save_path != "":
        cbimwrite(join(save_path, f"output_nuclei_mask.tif"), instance2semantics(output_nuclei_mask) * 255)
        cbimwrite(join(save_path, f"cell_mask_add_interior_add_nuclei_sem.tif"), instance2semantics(save_cell_mask_add_interior_sem) * 255)
    return output_nuclei_mask, save_cell_mask_add_interior_sem
        # partial_metrics,
    #



# @process_decorator('GiB')
def multimodal_merge(nuclei_mask_path, cell_mask_path, interior_mask_path, overlap_threshold=0.5, save_path=""):
    """
    assume input instance mask
    overlap between cell mask and interior mask:
    1. overlap == 0, keep both mask
    2. overlap > 0.5, keep cell mask only
    3. 0 < overlap < 0.5, keep cell mask and the non-overlap area of interior mask

    cell mask: cell mask +  processed interior mask
    nuc mask:
        1. nuc has less than 0.5 overlap with cell, save cell only
        2. nuc has 0 overlap with cell, save both nuc and cell
        3. nuc has more than 0.5 overlap with cell, save cell only
    """
    layer = 1
    sem = 2
    connectivity = 8
    nuclei_mask_raw = cbimread(nuclei_mask_path, only_np=True)
    cell_mask_raw = cbimread(cell_mask_path, only_np=True)
    interior_mask_raw = cbimread(interior_mask_path, only_np=True)

    cell_add_interior = interior_cell_merge(cell_mask_raw, interior_mask_raw, overlap_threshold=0.5, save_path=save_path)
    #cell_add_interior_path = join(save_path, f"cell_mask_add_interior.tif")
    #cell_add_interior = cbimread(cell_add_interior_path, only_np=True)
    #interior_cell_merge(nuclei_mask_path=nuclei_mask_path, cell_mask_path=cell_mask_path, interior_mask_path=interior_mask_path, overlap_threshold=0.5, save_path=save_path)
    #-----------------------------start merging nuclei---------------------------------------------------------
    
    output_nuclei_mask, final_mask = nuclei_cell_merge(nuclei_mask_raw, cell_add_interior, overlap_threshold=0.5, save_path=save_path)
    #output_nuclei_mask, cell_mask = nuclei_cell_merge(nuclei_mask_path = nuclei_mask_path, cell_mask_path = cell_mask_path , overlap_threshold=0.5, save_path=save_path)
    #cell_add_interior_path = join(save_path, f"cell_mask_add_interior_add_nuclei_sem.tif")
    #final_mask = cbimread(cell_add_interior_path, only_np=True)
    final_mask = final_mask.astype(np.uint8)
    return final_mask
    #


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


if __name__ == '__main__':
    save_path = r"/storeData/USER/data/01.CellBin/00.user/wangaoli/data/result/时空多蛋白数据/chip/Q00148CA_test/multimodal"
    nuclei_mask_path = r"/storeData/USER/data/01.CellBin/00.user/wangaoli/data/result/时空多蛋白数据/chip/Q00148CA_test/Q00148CA_DAPI_mask_raw.tif"
    cell_mask_path = r"/storeData/USER/data/01.CellBin/00.user/wangaoli/data/result/时空多蛋白数据/chip/Q00148CA_test/Q00148CA_CY5_IF_mask_raw.tif"
    interior_mask_path = r"/storeData/USER/data/01.CellBin/00.user/wangaoli/data/result/时空多蛋白数据/chip/Q00148CA_test/Q00148CA_TRITC_IF_mask_raw.tif"
    nuclei_mask_raw = cbimread(nuclei_mask_path, only_np=True)
    cell_mask_raw = cbimread(cell_mask_path, only_np=True)
    interior_mask_raw = cbimread(interior_mask_path, only_np=True)
    
    multimodal_merge(
        nuclei_mask_path=nuclei_mask_path,
        cell_mask_path=cell_mask_path,
        interior_mask_path=interior_mask_path,
        save_path=save_path
    )