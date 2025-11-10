from typing import Union

import numpy as np
import numpy.typing as npt
from skimage.measure import label, regionprops
import cv2
import tifffile

from cellbin2.image.mask import f_instance2semantics
from cellbin2.image.morphology import f_deep_watershed, vincent_soille_watershed
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

def f_postprocess_transcriptomics(pred):
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
    pred = pred[0, :, :, 0]

    # pred[pred > 0] = 1
    # pred = np.uint8(pred)

    pred = f_instance2semantics(pred)
    img = np.where(pred > 1, 1, 0).astype(np.uint8)


    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))  
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  
    

    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_small)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_large)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.fillPoly(img, [contour], color=1)
    return img
    return pred
    img = np.where(pred > 1, 1, 0).astype(np.uint8)


    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))  
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  
    

    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_small)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_large)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.fillPoly(img, [contour], color=1)
    return img


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


def f_postprocess_rna(mask):
    def f_check_shape(ct):
        farthest = 0
        max_dist = 0
        for i in range(ct.shape[0]):
            d = np.sqrt((ct[i][0][0] - ct[i - 1][0][0]) ** 2 + (ct[i][0][1] - ct[i - 1][0][1]) ** 2)
            if d > max_dist:
                max_dist = d
                farthest = i
        rect = cv2.minAreaRect(ct)
        if rect[1][0] * rect[1][1] == 0:
            return True
        if rect[1][0] / rect[1][1] >= 3 or rect[1][0] * rect[1][1] <= 1 / 3 or max_dist ** 2 > rect[1][0] * rect[1][1]:
            return True
        return False

    def f_contour_interpolate(mask_temp, value=255):
        tmp = mask_temp.copy()
        tmp[tmp == value] = 0
        img = mask_temp.copy()
        img[img != value] = 0
        img[img > 0] = 255
        img = np.uint8(img)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(contours[0])
        img = np.uint16(img)
        img[img > 0] = value
        cv2.drawContours(img, [hull], -1, value, thickness=-1)
        img[img + tmp > value] = 0
        return img

    def f_is_border(img, i, j):
        s_r = [-1, 0, 1]
        s_c = [-1, 0, 1]
        if i == 0:
            s_r = s_r[1:]
        if i == img.shape[0] - 1:
            s_r = s_r[:-1]
        if j == 0:
            s_c = s_c[1:]
        if j == img.shape[1] - 1:
            s_c = s_c[:-1]
        for r in s_r:
            for c in s_c:
                if img[i + r][j + c] != 0 and img[i + r][j + c] != img[i][j]:
                    return 1
        return 0

    def f_border_map(img):
        map = np.zeros(img.shape)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i][j] == 0:
                    continue
                map[i][j] = f_is_border(img, i, j)
        return map
    clog.info(f"Start rna post processing")
    label_mask = label(mask, connectivity=2)
    props = regionprops(label_mask, label_mask)
    for idx, obj in enumerate(props):
        bbox = obj['bbox']
        label_mask_temp = label_mask[bbox[0]: bbox[2], bbox[1]: bbox[3]].copy()
        if obj['filled_area'] < 80:
            # if obj['filled_area'] < 0:
            label_mask_temp[label_mask_temp == obj['label']] = 0
            label_mask[bbox[0]: bbox[2], bbox[1]: bbox[3]] = label_mask_temp
        else:
            tmp_mask = label_mask_temp.copy()
            tmp_mask[tmp_mask != obj['label']] = 0
            tmp_mask, tmp_area = f_watershed(tmp_mask)
            tmp_mask = np.uint32(tmp_mask)
            tmp_mask[tmp_mask > 0] = obj['label']
            label_mask_temp[tmp_area > 0] = tmp_mask[tmp_area > 0]
            label_mask[bbox[0]: bbox[2], bbox[1]: bbox[3]][tmp_area > 0] = label_mask_temp[tmp_area > 0]
    label_mask[label_mask > 0] = 255
    label_mask = label(label_mask, connectivity=2)
    props = regionprops(label_mask, label_mask)
    for idx, obj in enumerate(props):
        bbox = obj['bbox']
        label_mask_temp = label_mask[bbox[0]: bbox[2], bbox[1]: bbox[3]]
        if obj['filled_area'] < 80:
            label_mask_temp[label_mask_temp == obj['label']] = 0
            label_mask[bbox[0]: bbox[2], bbox[1]: bbox[3]] = label_mask_temp
            continue
        tmp_mask = label_mask_temp.copy()
        tmp_mask[tmp_mask != obj['label']] = 0
        tmp_mask[tmp_mask > 0] = 255
        tmp_mask = np.uint8(tmp_mask)
        contours, _ = cv2.findContours(tmp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if 0 not in contours[0] and f_check_shape(contours[0]):
            label_mask_temp[label_mask_temp == obj['label']] = 0
            label_mask[bbox[0]: bbox[2], bbox[1]: bbox[3]] = label_mask_temp
        intp = f_contour_interpolate(label_mask_temp, obj['label'])
        label_mask[bbox[0]: bbox[2], bbox[1]: bbox[3]][intp == obj['label']] = obj['label']
    map = f_border_map(label_mask)
    label_mask[map > 0] = 0
    label_mask[np.where(label_mask > 0)] = 1
    return np.uint8(label_mask)


model_postprocess = {
    SUPPORTED_MODELS[0]: {
        TechType.ssDNA: f_postprocess_v2,
        TechType.DAPI: f_postprocess_v2,
        TechType.HE: f_postprocess_v2,
        TechType.Transcriptomics: f_postprocess_transcriptomics,
        TechType.Protein: f_postprocess_transcriptomics
    },
    SUPPORTED_MODELS[1]: {
        TechType.HE: f_postprocess_v2,
    },
    SUPPORTED_MODELS[2]: {
        TechType.Transcriptomics: f_postprocess_rna,
        TechType.Stereocell: f_postprocess_rna
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
