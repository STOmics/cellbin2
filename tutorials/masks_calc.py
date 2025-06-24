import os

import numpy as np

from cellbin2.image import cbimread, cbimwrite
from cellbin2.contrib.cellpose_segmentor import f_instance2semantics
from cellbin2.utils import clog
from skimage.measure import label


def calc(a, b, overlap_threshold=0.5):
    """
    overlap(a, b) / a
    """
    a_tag = os.path.basename(a)
    b_tag = os.path.basename(b)
    a_raw = cbimread(a, only_np=True)
    b_raw = cbimread(b, only_np=True)

    # a_raw_sem = f_instance2semantics(a_raw)
    # b_raw_sem = f_instance2semantics(b_raw)

    a_mask = label(a_raw, connectivity=2)
    b_mask = label(b_raw, connectivity=2)

    a_mask_cell_counts = len(np.unique(a_mask))
    b_mask_cell_counts = len(np.unique(b_mask))

    clog.info(f"{a_tag} cell counts: {a_mask_cell_counts - 1}")
    clog.info(f"{b_tag} cell counts: {b_mask_cell_counts - 1}")

    # overlap between a and b
    overlap = np.logical_and(a_mask, b_mask)

    # get only overlap from a
    a_mask_overlap = a_mask[overlap]

    # labels and corresponding area of overlap in a
    a_mask_overlap_labels, a_mask_overlap_counts = np.unique(
        a_mask_overlap,
        return_counts=True,
    )

    # get all labels and corresponding area in a
    a_mask_labels, a_mask_counts = np.unique(
        a_mask,
        return_counts=True,
    )

    # get counts of overlap labels
    a_mask_counts_subset = a_mask_counts[a_mask_overlap_labels]

    # overlap labels: overlap counts / whole area
    overlap_over_a = a_mask_overlap_counts / a_mask_counts_subset

    # only consider overlap above a threshold
    the_chosen = overlap_over_a > overlap_threshold

    # for debug
    a_mask_chosen = a_mask_overlap_labels[the_chosen]
    keep_mask = np.isin(a_mask, a_mask_chosen)
    a_mask_keep = np.where(keep_mask, a_mask, 0)
    a_mask_keep[a_mask_keep != 0] = 255
    a_mask_keep = a_mask_keep.astype('uint8')
    a_vis_path = os.path.splitext(a)[0] + "_keep" + os.path.splitext(a)[1]
    cbimwrite(a_vis_path, a_mask_keep)

    keep_mask = np.isin(a_mask, a_mask_chosen, invert=True)
    a_mask_remove = np.where(keep_mask, a_mask, 0)
    a_mask_remove[a_mask_remove != 0] = 255
    a_mask_remove = a_mask_remove.astype('uint8')
    a_vis_path = os.path.splitext(a)[0] + "_remove" + os.path.splitext(a)[1]
    cbimwrite(a_vis_path, a_mask_remove)
    # 重叠部分大于阈值/a mask的细胞数
    overlap_counts_over_a = np.sum(the_chosen) / a_mask_cell_counts

    # 重叠部分大于阈值/a mask的细胞数
    overlap_counts_over_b = np.sum(the_chosen) / b_mask_cell_counts

    clog.info(f"overlap / {a_tag}: {round(overlap_counts_over_a, 2)}")
    clog.info(f"overlap / {b_tag}: {round(overlap_counts_over_b, 2)}")
    print()


if __name__ == '__main__':
    nuclei_mask_path = "/media/Data1/user/dengzhonghan/data/GB_FFPE/C05073E4/C05073E4_ssDNA_regist_cp_mask_after_tc.tif"
    cell_mask_path = "/media/Data1/user/dengzhonghan/data/GB_FFPE/C05073E4/C05073E4_gene_cp_mask.tif"
    calc(a=nuclei_mask_path, b=cell_mask_path)
