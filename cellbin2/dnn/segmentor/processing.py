from cellbin2.image.augmentation import f_padding as f_pad

import numpy as np
import cv2
from skimage.exposure import rescale_intensity


def f_preformat(img):
    if img.ndim < 3:
        img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, axis=0)
    return img


def f_preformat_rna(img):
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    return img


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def f_postformat_rna(pred):
    if isinstance(pred, list):
        pred = pred[0]
    pred = np.squeeze(pred)
    pred = sigmoid(pred)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    pred = np.array(pred, dtype="uint8")
    return pred


def normalize_to_0_255(arr):
    v_max = np.max(arr)
    v_min = np.min(arr)
    if v_max == 0:
        return arr

    # check if there is value in range 0-255 
    if 0 <= v_min <= 255 or 0 <= v_max <= 255 or (v_max > 255 and v_min < 0):
        # if true, multiply the arry values by factor 
        factor = 1000
        np.multiply(arr, factor)

    # normalization 
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    return ((arr - arr_min) * 255) / (arr_max - arr_min)


def f_postformat(pred):
    if not isinstance(pred, list):
        pred = [np.uint8(rescale_intensity(pred, out_range=(0, 255)))]
    else:
        pred = [np.uint8(rescale_intensity(pred[0], out_range=(0, 255)))]
    # p_max = np.max(pred[-1])
    pred = pred[0][0, :, :, 0]
    # pred = f_deep_watershed(pred,
    #                         maxima_threshold=round(0.1 * 255),
    #                         maxima_smooth=0,
    #                         interior_threshold=round(0.2 * 255),
    #                         interior_smooth=0,
    #                         fill_holes_threshold=15,
    #                         small_objects_threshold=0,
    #                         radius=2,
    #                         watershed_line=0)
    return pred


def f_padding(img, shape, mode='constant'):
    h, w = img.shape[:2]
    win_h, win_w = shape[:2]
    img = f_pad(img, 0, abs(win_h - h), 0, abs(win_w - w), mode)
    return img


def f_fusion(img1, img2):
    img1 = cv2.bitwise_or(img1, img2)
    return img1
