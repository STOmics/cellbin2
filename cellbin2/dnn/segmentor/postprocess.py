from typing import Union

import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.filters import sobel
from skimage.measure import label, regionprops
import cv2

from cellbin2.image.mask import f_instance2semantics
from cellbin2.image.morphology import f_deep_watershed
from cellbin2.dnn.segmentor.utils import SUPPORTED_MODELS
from cellbin2.utils.common import TechType
from cellbin2.utils import clog


def f_postpocess(pred):
    pred = pred[0, :, :, 0]

    # pred[pred > 0] = 1
    # pred = np.uint8(pred)

    pred = f_instance2semantics(pred)
    return pred


def f_postprocess_v2(pred):
    if isinstance(pred, list):
        pred = pred[0]
    pred = np.expand_dims(pred, axis=(0, -1))
    # pred = np.uint64(np.multiply(np.around(pred, decimals=2), 100))
    # pred = np.uint8(normalize_to_0_255(pred))

    pred = f_deep_watershed([pred],
                            maxima_threshold=round(0.1 * 255),
                            maxima_smooth=0,
                            interior_threshold=round(0.2 * 255),
                            interior_smooth=0,
                            fill_holes_threshold=15,
                            small_objects_threshold=0,
                            radius=2,
                            watershed_line=0,
                            maxima_algorithm='h_maxima')
    pred = f_postpocess(pred)
    return pred


def f_watershed(mask):
    tmp = mask.copy()
    tmp[tmp > 0] = 255
    tmp = np.uint8(tmp)
    contours, _ = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(tmp, [contours[0]], -1, 255, -1)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(tmp, cv2.MORPH_OPEN, open_kernel, iterations=3)
    sure_bg = cv2.dilate(opening, open_kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    thr_lst = np.round(np.arange(1, 10) * 0.1, 1)
    cpnt_lst = []
    mtmp_lst = []
    for thr in thr_lst:
        _, sure_fg = cv2.threshold(dist_transform, thr * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        sure_fg = cv2.erode(sure_fg, open_kernel)
        unknown = cv2.subtract(sure_bg, sure_fg)
        unknown = cv2.morphologyEx(unknown, cv2.MORPH_CLOSE, open_kernel, iterations=2)
        sure_fg[unknown > 0] = 0
        count, markers = cv2.connectedComponents(sure_fg, connectivity=8)
        cpnt_lst.append(count)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(cv2.cvtColor(opening, cv2.COLOR_GRAY2RGB), markers)
        map_tmp = np.zeros(markers.shape, np.uint8)
        map_tmp[markers == -1] = 255
        cv2.drawContours(map_tmp, [contours[0]], -1, 0, 1)
        mtmp_lst.append(map_tmp)
    map_final = np.zeros(mask.shape)
    max_cpnt = max(cpnt_lst)
    for i in range(len(mtmp_lst)):
        if cpnt_lst[i] == max_cpnt:
            map_tmp = mtmp_lst[i]
            stack_tmp = map_tmp / 255 + map_final / 255
            stack_tmp[stack_tmp <= 1] = 0
            if stack_tmp.sum() > 0:
                _, lbl = cv2.connectedComponents(map_tmp, connectivity=8)
                for j in range(1, lbl.max()):
                    if stack_tmp[lbl == j].sum() > 0:
                        map_tmp[lbl == j] = 0

            map_final[map_tmp > 0] = 255
    opening[map_final > 0] = 0
    opening = cv2.erode(opening, np.ones((3, 3)))
    opening = cv2.dilate(opening, open_kernel)
    return opening, tmp

def watershed_segmentation(binary_image, sigma=3.5):
    tmp = binary_image.copy()
    binary_mask = binary_image > 0
    
    distance = ndi.distance_transform_edt(binary_mask)
    local_min = distance < 1.5 
    
    blurred_distance = ndi.gaussian_filter(distance, sigma=sigma)
    
    # peak_local
    fp = np.ones((3,) * binary_mask.ndim)
    coords = peak_local_max(blurred_distance, footprint=fp, labels=binary_mask)
    
    # markers
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers = ndi.label(mask)[0]
    
    # watershed
    labels = watershed(-blurred_distance, markers, mask=binary_mask)
    
    edges_labels = sobel(labels)
    edges_binary = sobel(binary_mask.astype(float))
    edges = np.logical_xor(edges_labels != 0, edges_binary != 0)
    
    # postprocess
    result = np.logical_not(edges) * binary_mask
    result = ndi.binary_opening(result)
    result[local_min] = 0
    labels_cut, _ = ndi.label(result)
    for _ in range(2):  
        border = ndi.binary_dilation(labels_cut > 0) & (labels_cut == 0)
        y, x = np.where(border)
        for yi, xi in zip(y, x):
            neighbors = labels_cut[max(yi-1,0):yi+2, max(xi-1,0):xi+2]
            unique_neighbors = np.unique(neighbors[neighbors > 0])
            if len(unique_neighbors) == 1:
                labels_cut[yi, xi] = unique_neighbors[0]
    labels_cut = np.where(labels_cut > 0, 1, 0).astype(np.uint8)
    return labels_cut, tmp

def f_postprocess_rna(mask):
    from skimage.morphology import remove_small_objects
    clog.info(f"Start rna post processing")
    label_mask = label(mask, connectivity=2)
    props = regionprops(label_mask, label_mask)
    for idx, obj in enumerate(props):
        bbox = obj['bbox']
        label_mask_temp = label_mask[bbox[0]: bbox[2], bbox[1]: bbox[3]].copy()
        tmp_mask = label_mask_temp.copy()
        tmp_mask[tmp_mask != obj['label']] = 0
        tmp_mask, tmp_area = watershed_segmentation(tmp_mask)
        tmp_mask = np.uint32(tmp_mask)
        tmp_mask[tmp_mask > 0] = obj['label']
        label_mask_temp[tmp_area > 0] = tmp_mask[tmp_area > 0]
        label_mask[bbox[0]: bbox[2], bbox[1]: bbox[3]][tmp_area > 0] = label_mask_temp[tmp_area > 0]
    label_mask = np.where(label_mask > 0, 1, 0).astype(np.uint8)
    pred = remove_small_objects(label_mask.astype(np.bool8), min_size=80, connectivity=2).astype(np.uint8)
    #post_mask=watershed_segmentation(mask)
    return pred


def f_postprocess_cellpose(mask, overlap_mask=None):
    """
    Only apply watershed for cells overlaped with the patches overlap area.
    To prevent over split.
    """
    clog.info(f"Start post processing")
    label_mask = label(mask, connectivity=2)
    props = regionprops(label_mask, label_mask)
    for idx, obj in enumerate(props):
        bbox = obj['bbox']
        need_watershed = False
        if overlap_mask is None:
            need_watershed = True
        else:
            cell_bbox_region = label_mask[bbox[0]: bbox[2], bbox[1]: bbox[3]] == obj['label']
            overlap_bbox_region = overlap_mask[bbox[0]: bbox[2], bbox[1]: bbox[3]]
            if np.any(cell_bbox_region & overlap_bbox_region):
                need_watershed = True
        
        if need_watershed:
            label_mask_temp = label_mask[bbox[0]: bbox[2], bbox[1]: bbox[3]].copy()
            tmp_mask = label_mask_temp.copy()
            tmp_mask[tmp_mask != obj['label']] = 0
            tmp_mask, tmp_area = watershed_segmentation(tmp_mask)
            tmp_mask = np.uint32(tmp_mask)
            tmp_mask[tmp_mask > 0] = obj['label']
            label_mask_temp[tmp_area > 0] = tmp_mask[tmp_area > 0]
            label_mask[bbox[0]: bbox[2], bbox[1]: bbox[3]][tmp_area > 0] = label_mask_temp[tmp_area > 0]
    label_mask[label_mask > 0] = 1
    #post_mask=watershed_segmentation(mask)
    return np.uint8(label_mask)


model_postprocess = {
    SUPPORTED_MODELS[0]: {
        TechType.ssDNA: f_postprocess_v2,
        TechType.DAPI: f_postprocess_v2,
        TechType.HE: f_postprocess_v2
    },
    SUPPORTED_MODELS[1]: {
        TechType.HE: f_postprocess_v2,
    },
    SUPPORTED_MODELS[2]: {
        TechType.Transcriptomics: f_postprocess_rna
    }
}


class CellSegPostprocess:
    def __init__(self, model_name):
        self.model_name = model_name
        self.m_postprocess = model_postprocess[self.model_name]

    def __call__(self, img: npt.NDArray, stain_type):
        post_func = self.m_postprocess.get(stain_type)
        img = post_func(img)
        return img
