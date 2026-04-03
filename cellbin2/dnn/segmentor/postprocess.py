from typing import Union

import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.filters import sobel
from skimage.measure import label, regionprops
import cv2
from skimage.morphology import remove_small_objects

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


def f_postprocess_cellpose(mask, overlap_mask=None, area_ratio_thresh=5.0):
    """
    Only apply watershed for cells overlaped with the patches overlap area.
    To prevent over split.

    After watershed, remove cells whose area is larger than
    area_ratio_thresh * mean_cell_area.
    """
    clog.info(f"Start post processing")

    label_mask = label(mask, connectivity=2)
    props = regionprops(label_mask, label_mask)

    for obj in props:
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
            ys, ye, xs, xe = bbox[0], bbox[2], bbox[1], bbox[3]
            sub = label_mask[ys:ye, xs:xe].copy()
            sub[tmp_area > 0] = label_mask_temp[tmp_area > 0]
            label_mask[ys:ye, xs:xe] = sub

    # start area filter
    label_mask = label(label_mask > 0, connectivity=2)

    props_after = regionprops(label_mask)
    if len(props_after) > 0:
        areas = np.array([obj.area for obj in props_after], dtype=np.float32)
        ref_area = np.median(areas)
        max_area = ref_area * 5

        for obj in props_after:
            if obj.area > max_area:
                label_mask[label_mask == obj.label] = 0

    label_mask[label_mask > 0] = 1
    label_mask = remove_small_objects(label_mask.astype(np.bool8), min_size=80, connectivity=2)
    
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
