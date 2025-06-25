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
    mask0 = masks[0].ravel()#二维展一维
    mask1 = masks[1].ravel()
    # Find pairs where both labels are non-zero
    valid_indices = (mask0 > 0) & (mask1 > 0)#找出重叠区域
    valid_mask0 = mask0[valid_indices]#mask中的重叠区域
    valid_mask1 = mask1[valid_indices]#mask中的重叠区域
    # Combine valid pairs
    combined = np.column_stack((valid_mask0, valid_mask1))#重叠对
    # Find unique pairs and their counts
    unique_pairs, counts = np.unique(combined, axis=0, return_counts=True)#配对及出现次数（像素数）
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
    nz_paired_labels, nz_counts = unique_nonzero_pairs_numpy(masks)#配对及出现次数（像素数）

    # assign each cell the nuclei with the most overlap
    count_sort_ix = np.argsort(nz_counts, kind="stable")#排序

    mask_a, mask_b = masks

    a_to_b = np.zeros(np.max(mask_a) + 1, dtype=np.uint32)
    a_to_b[nz_paired_labels[count_sort_ix, 0]] = nz_paired_labels[count_sort_ix, 1]
    b_to_a = np.zeros(np.max(mask_b) + 1, dtype=np.uint32)
    b_to_a[nz_paired_labels[count_sort_ix, 1]] = nz_paired_labels[count_sort_ix, 0] #maska 和 maskb的对应关系，细胞核属于哪个膜，膜包含哪个核

    return a_to_b, b_to_a


# @process_decorator('GiB')
def make_mask_consecutive( #用连续整数标签所有细胞
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
    unique_input_labels = np.unique(mask) #unique 标签
    unique_input_labels = unique_input_labels[unique_input_labels > 0] #去除背景标签0
    if unique_input_labels.shape[0] == 0: #mask全是背景，直接return
        assert np.all(mask == 0)
        return mask

    num_labels = unique_input_labels.shape[0] #除背景外的标签数量 （语义分割）
    max_label = np.max(unique_input_labels)
    assert (
            max_label < MAX_INPUT_LABEL_VALUE #防止标签溢出 unit32
    ), "Input labels out of range for relabeling procedure"
    label_remapper = np.zeros(max_label + 1, np.uint32) #全0数字
    label_remapper[unique_input_labels] = np.arange(start_from, num_labels + start_from)#转为连续整数标签（转实例分割）
    #这儿是不是可以加一个防溢出？

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
    nz_cell_labels = cell_labels[nz_assignments]#找出需要筛选的细胞
    nz_cells_to_nuclei = cells_to_nuclei[nz_assignments]#找出有效配对

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
    overlap_frac[nz_assignments] = max_counts / areas #重叠区域像素数量/核面积（像素数量）

    return overlap_frac


# @process_decorator('GiB')
def multimodal_merge(nuclei_mask_path, cell_mask_path, interior_mask_path, overlap_threshold=0.5, save_path=""):
    """
    这里现在假设输入为instance mask
    overlap between cell mask and nuc mask:
    1. overlap == 0
    2. overlap > 0.5
    3. 0 < overlap < 0.5

    cell mask: cell mask +  (overlap == 0)的nuc mask
    nuc mask:
        1. cell对nuc overlap为0，保留cell
        2. nuc对cell overlap为0，保留nuc
        3. 若overlap > 0.5, nuc只保留overlap部分
        4. 以每个cell为基准，与每个cell overlap最大的nuc若为(0,0.5)之间，该nuc被去除，将cell的结果复制过来

    """
    layer = 1
    sem = 2
    connectivity = 8
    nuclei_mask_raw = cbimread(nuclei_mask_path, only_np=True)
    cell_mask_raw = cbimread(cell_mask_path, only_np=True)
    interior_mask_raw = cbimread(interior_mask_path, only_np=True)
    #if len(nuclei_mask_raw.shape) > layer:
       # nuclei_mask_raw = nuclei_mask_raw[:, :, layer]
    #if len(cell_mask_raw.shape) > layer:
        #cell_mask_raw = cell_mask_raw[:, :, layer]
    #if len(interior_mask_raw.shape) > layer:
        #interior_mask_raw = interior_mask_raw[:, :, layer]
    print(len(np.unique(nuclei_mask_raw)))
    print(len(np.unique(cell_mask_raw)))
    print(len(np.unique(interior_mask_raw)))
    '''    
    if len(np.unique(nuclei_mask_raw)) <= sem:
        _, nuclei_mask = cv2.connectedComponents(nuclei_mask_raw, connectivity=connectivity)
    else:
        nuclei_mask_sem = f_instance2semantics(nuclei_mask_raw)#细胞核mask转语义
        _, nuclei_mask = cv2.connectedComponents(nuclei_mask_sem, connectivity=connectivity)
        # nuclei_mask = nuclei_mask_raw
    if len(np.unique(cell_mask_raw)) <= sem:
        _, cell_mask = cv2.connectedComponents(cell_mask_raw, connectivity=connectivity)
    else:
        cell_mask_sem = f_instance2semantics(cell_mask_raw)#细胞膜mask转语义
        _, cell_mask = cv2.connectedComponents(cell_mask_sem, connectivity=connectivity)
    if len(np.unique(interior_mask_raw)) <= sem:
        _, interior_mask = cv2.connectedComponents(interior_mask_raw, connectivity=connectivity)
    else:
        interior_mask_sem = f_instance2semantics(interior_mask_raw)#细胞质mask转语义
        _, interior_mask = cv2.connectedComponents(interior_mask_sem, connectivity=connectivity)

    if save_path != "": #存储语义mask
        cbimwrite(join(save_path, f"nuclei_mask_ori.tif"), nuclei_mask_sem * 255)
        cbimwrite(join(save_path, f"cell_mask_ori.tif"), cell_mask_sem * 255)
        cbimwrite(join(save_path, f"interior_mask_ori.tif"), interior_mask_sem * 255)
    '''


    if len(np.unique(nuclei_mask_raw)) <= sem: #输入的是语义分割
        _, nuclei_mask = cv2.connectedComponents(nuclei_mask_raw, connectivity=connectivity)
    else:
        nuclei_mask_sem = f_instance2semantics(nuclei_mask_raw)#细胞核mask转语义
        _, nuclei_mask = cv2.connectedComponents(nuclei_mask_sem, connectivity=connectivity)
        cbimwrite(join(save_path, f"nuclei_mask_ori.tif"), nuclei_mask_sem * 255)
    if len(np.unique(cell_mask_raw)) <= sem:
        _, cell_mask = cv2.connectedComponents(cell_mask_raw, connectivity=connectivity)
    else:
        cell_mask_sem = f_instance2semantics(cell_mask_raw)#细胞膜mask转语义
        _, cell_mask = cv2.connectedComponents(cell_mask_sem, connectivity=connectivity)
        cbimwrite(join(save_path, f"cell_mask_ori.tif"), cell_mask_sem * 255)
    if len(np.unique(interior_mask_raw)) <= sem:
        _, interior_mask = cv2.connectedComponents(interior_mask_raw, connectivity=connectivity)
    else:
        interior_mask_sem = f_instance2semantics(interior_mask_raw)#细胞质mask转语义
        _, interior_mask = cv2.connectedComponents(interior_mask_sem, connectivity=connectivity)
        cbimwrite(join(save_path, f"interior_mask_ori.tif"), interior_mask_sem * 255)




    #全部转实例分割
    cell_mask[:] = make_mask_consecutive(cell_mask)
    nuclei_mask[:] = make_mask_consecutive(nuclei_mask)
    interior_mask[:] = make_mask_consecutive(interior_mask)

    lower_right_overlap_mask = np.zeros(cell_mask.shape, dtype=bool)
    top_left_bounds_cells = np.ones(cell_mask.shape, dtype=bool) #标记膜边界
    top_left_bounds_nuclei = np.ones(nuclei_mask.shape, dtype=bool)#标记核边界

    edge_filter_nuclei = np.logical_and(
        top_left_bounds_nuclei, np.logical_not(lower_right_overlap_mask)
    )
    edge_filter_cells = np.logical_and(
        top_left_bounds_cells, np.logical_not(lower_right_overlap_mask)
    )
    num_cells = np.count_nonzero(np.unique(cell_mask[edge_filter_cells]))
    num_nuclei = np.count_nonzero(np.unique(nuclei_mask[edge_filter_nuclei]))

    # Generate mappings between cells and nuclei and vis versa.

    cell_to_interior, interior_to_cell = pair_map_by_largest_overlap( #细胞质与细胞膜的对应关系
        (cell_mask, interior_mask)
    )

    # interior mask与cell mask的关系处理 细胞质与细胞膜
    # interior & cell overlap > 0.5, 认为interior和cell表示的为一个细胞，interior的去除
    # 0 < interior & cell overlap <= 0.5，认为表示的为两个细胞，interior保留独有的那部分，这里目前可能出现一个细胞分成两个
    # interior & cell overlap = 0，这个比较简单，独有的细胞
    overlap_fracs_interior_to_cell = overlap_fractions(interior_mask, cell_mask, interior_to_cell, c=True)#细胞质区域对应的膜
    interior_to_cell_no_overlap = overlap_fracs_interior_to_cell == 0
    interior_to_cell_overlap_below_threshold = overlap_fracs_interior_to_cell <= overlap_threshold
    interior_keep_idx = np.logical_or(interior_to_cell_no_overlap, interior_to_cell_overlap_below_threshold)
    interior_keep_idx[0] = False  # 背景
    # 保留interior独有的部分
    interior_keep_bool_mask = interior_keep_idx[interior_mask]
    interior_mask_unique_bool_mask = interior_keep_bool_mask * np.logical_not(cell_mask)
    interior_mask_unique = interior_mask_unique_bool_mask * interior_mask
    # 此时部分连通域可能被切成独立的几块，需要重新label
    interior_mask_unique_relabel = label(interior_mask_unique)
    # 原interior mask与独有的interior mask进行最大overlap计算，为了去除切割后的小部分，仅保留最大的那一块
    interior_to_interior_unique, interior_unique_to_interior \
        = pair_map_by_largest_overlap((interior_mask, interior_mask_unique_relabel))
    interior_splits_remove_bool_mask = interior_to_interior_unique[interior_mask] == interior_mask_unique_relabel
    interior_mask_final = interior_mask_unique_relabel * interior_splits_remove_bool_mask
    interior_mask_final = make_mask_consecutive(
        interior_mask_final, start_from=np.max(cell_mask) + 1
    )

    cell_mask_add_interior = np.add(cell_mask, interior_mask_final)
    print(f"interior index from {np.max(cell_mask) + 1} to {np.max(interior_mask_final)}")
    cell_mask = cell_mask_add_interior
    if save_path != "":
        cbimwrite(join(save_path, f"cell_mask_add_interior.tif"), f_instance2semantics(cell_mask_add_interior) * 255)
        cbimwrite(join(save_path, f"interior_mask_final.tif"), f_instance2semantics(interior_mask_final) * 255)


    #-----------------------------以下合并细胞核与细胞质膜---------------------------------------------------------
    cell_to_nucleus, nucleus_to_cell = pair_map_by_largest_overlap(#细胞核与细胞质膜的对应关系
        (cell_mask, nuclei_mask)
    )
    # Get the locations where the overlap is less than the threshold. For those
    # cases, set the assignment to zero.
    overlap_fracs = overlap_fractions(cell_mask, nuclei_mask, cell_to_nucleus)
    cell_to_nucleus[np.where(overlap_fracs < overlap_threshold)] = 0 #重叠区域小于0.5的标记为0（不对应任何细胞核，该细胞被当作无核细胞）
    del overlap_fracs

    # the output mask for nuclei
    output_nuclei_mask = np.zeros_like(nuclei_mask) #生成一个全0数字，作为output基础（类似空画板）

    # for nuclei that are assigned, trim the part outside the cell
    # matching_mask is a mask over the matrix for the spots where
    # the cell matches the nuclei it was assigned to.
    # we can use this to "trim" the input nuclei mask just to the
    # portion matching the cell.
    matching_ix = np.where((cell_to_nucleus[cell_mask] == nuclei_mask) & (cell_to_nucleus[cell_mask] != 0))
    output_nuclei_mask[matching_ix] = cell_mask[matching_ix]#先记上有match的细胞，细胞核多余部分被裁切
    del matching_ix

    # indices of cells without a nuclei
    no_nuc = cell_to_nucleus == 0 #无核细胞
    no_nuc[0] = 0


    # get a mask covering cells without a nucleus
    no_nuc_cell_mask = no_nuc[cell_mask] #先记上无核细胞
    # for cells without a nuclei, set their nucleus equal to cell boundary
    # cell without nuclei说明在这个位置，cell有分割，但nuclei可能没分到
    # （为什么说可能呢，因为将cell与nuclei overlap小于threshold的都算进来了）
    # 所以把cell的分割结果也给到nuclei上
    output_nuclei_mask[no_nuc_cell_mask] = cell_mask[no_nuc_cell_mask]#记上所有无核、几乎无核细胞

    # remove from the mask duplicate portions of the tile
    # so we can accurately quantify unique no-nuc cells metric
    no_nuc_cell_mask[lower_right_overlap_mask] = 0
    no_nuc_cell_mask[np.logical_not(top_left_bounds_cells)] = 0#去重，去边界
    num_cells_no_nuc = np.unique(cell_mask[no_nuc_cell_mask]).shape[0]
    del no_nuc_cell_mask

    # Indices of nuclei without any overlap with a cell. This eliminates cases
    # were a nucleus could be assigned to some cell, but isn't because there is
    # another larger nucleus.
    not_assigned = nucleus_to_cell == 0 #没有细胞的核
    # not_assigned_by_interior = nucleus_to_interior == 0
    # not_assigned = np.logical_and(not_assigned_by_cell, not_assigned_by_interior)
    not_assigned[0] = False

    # boolean mask for spots in the nuclei mask that are nuclei unassigned to a cell
    unassigned_nuclei_mask = not_assigned[nuclei_mask]#核是否有对应细胞，布尔值

    # indices in the nuclei mask that are nuclei unassigned to a cell
    unassigned_nuclei_indices = np.nonzero(unassigned_nuclei_mask)#细胞核没有对应膜的部分

    # mutate the unassigned_nuclei_mask to help quantify accurately
    # our metric for num_nuc_no_cell
    unassigned_nuclei_mask[lower_right_overlap_mask] = False
    unassigned_nuclei_mask[np.logical_not(top_left_bounds_nuclei)] = False #去边界，帮助准确量化
    # `unassigned_nuc_mask` now holds the mask of nuclei where is there no assigned
    # cell, which is not in the lower-right overlap. Each label will be repeated the
    # number of times it appears in the mask. Taking the unique values gives us
    # the labels of each nucleus without a corresponding cell.
    num_nuc_no_cell = np.count_nonzero(np.unique(nuclei_mask[unassigned_nuclei_mask]))#有多少核不完全有细胞
    del unassigned_nuclei_mask

    # for every nucleus not assigned a cell,
    # make it its own cell with same boundaries and new consecutive
    # mask labels starting from the current biggest cell mask label
    unassigned_nuc_values = nuclei_mask[unassigned_nuclei_indices]
    del nuclei_mask

    start_from = np.max(cell_mask_add_interior) + 1
    # assert start_from == np.max(output_nuclei_mask) + 1
    if unassigned_nuc_values.shape[0] > 0:
        consecutive_unassigned_nuc = make_mask_consecutive(
            unassigned_nuc_values, start_from=start_from
        )
        print(f"nuc index from {start_from} to {np.max(consecutive_unassigned_nuc)}")
        cell_mask_add_interior[unassigned_nuclei_indices] = consecutive_unassigned_nuc
        cell_mask[unassigned_nuclei_indices] = consecutive_unassigned_nuc
        output_nuclei_mask[unassigned_nuclei_indices] = consecutive_unassigned_nuc
    cell_mask_add_interior_sem = f_instance2semantics(cell_mask_add_interior)
    # cell_mask_add_interior_sem[unassigned_nuclei_indices] = 255
    # cell_mask_add_interior_sem[np.nonzero(np.logical_and(cell_mask_add_interior_sem, interior_keep_mask_))] = 125
    # cell_mask_add_interior_sem[cell_mask_add_interior_sem == 1] = 50
    print(f"# of cells: {np.max(cell_mask) + 1}")
    if save_path != "":
        cbimwrite(join(save_path, f"output_nuclei_mask.tif"), f_instance2semantics(output_nuclei_mask) * 255)
        cbimwrite(join(save_path, f"cell_mask_add_interior_add_nuclei_sem.tif"), cell_mask_add_interior_sem * 255)
    min_nuclei_label = 0 if np.sum(output_nuclei_mask == 0) > 0 else 1

    no_cells = cell_mask == 0
    assert np.sum(output_nuclei_mask[no_cells]) == 0

    min_cell_label = 0 if np.sum(no_cells) > 0 else 1
    del no_cells  # no_cells takes O(mask) memory

    assert np.all(
        np.unique(output_nuclei_mask)
        == np.arange(min_nuclei_label, np.max(cell_mask) + 1)
    )

    # assert cell mask equals the nuclei mask everywhere where nuclei mask is nonzero
    nuclei_nz = output_nuclei_mask > 0
    assert np.all(cell_mask[nuclei_nz] == output_nuclei_mask[nuclei_nz])
    del nuclei_nz  # nuclei_nz takes O(mask) memory

    # assert every cell and every nuclei is somewhere in the mask
    assert np.all(
        np.unique(cell_mask) == np.arange(min_cell_label, np.max(cell_mask) + 1)
    )

    return (
        output_nuclei_mask,
        cell_mask,
        # partial_metrics,
    )
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
    save_path = r"/storeData/USER/data/01.CellBin/00.user/wangaoli/data/result/时空多蛋白数据/chip/Q00148CA/multimodal_adjusted2"
    nuclei_mask_path = r"/storeData/USER/data/01.CellBin/00.user/wangaoli/data/result/时空多蛋白数据/chip/Q00148CA/Q00148CA_DAPI_transform_mask_raw.tif"
    cell_mask_path = r"/storeData/USER/data/01.CellBin/00.user/wangaoli/data/result/时空多蛋白数据/chip/Q00148CA/Q00148CA_CY5_IF_transform_mask_adjusted2.tif"
    interior_mask_path = r"/storeData/USER/data/01.CellBin/00.user/wangaoli/data/result/时空多蛋白数据/chip/Q00148CA/Q00148CA_TRITC_IF_transform_mask_raw_adjusted.tif"
    multimodal_merge(
        nuclei_mask_path=nuclei_mask_path,
        cell_mask_path=cell_mask_path,
        interior_mask_path=interior_mask_path,
        save_path=save_path
    )