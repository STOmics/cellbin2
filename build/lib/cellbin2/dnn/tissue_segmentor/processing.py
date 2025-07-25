#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author  : hedongdong1
# @Time    : 2024/10/17 22:57
# @File    : processing.py
# @annotation    :

import numpy as np
import numpy.typing as npt

def f_preformat(img: npt.NDArray) -> npt.NDArray:
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, axis=0)
    return img


def f_postformat(pred: npt.NDArray) -> npt.NDArray:
    if isinstance(pred, list):
        pred = pred[0]
    pred = np.squeeze(pred)
    return pred