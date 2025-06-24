#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author  : hedongdong1
# @Time    : 2024/10/17 17:08
# @File    : preprocess.py
# @annotation    :

from typing import Union
import cv2
import numpy as np
import numpy.typing as npt
from cellbin2.image.augmentation import f_ij_auto_contrast, f_ij_auto_contrast_v2, f_ij_16_to_8_v2, f_equalize_adapthist_V2
from cellbin2.image.augmentation import f_histogram_normalization
from cellbin2.image.augmentation import f_resize
from cellbin2.utils.common import TechType
from cellbin2.image import cbimread
from cellbin2.utils import clog


def f_pre_ssdna_dapi_SAW_V_7_1(img: npt.NDArray, input_size: tuple, stain_type: TechType) -> npt.NDArray:  #corresponding to segmentation version 230523
    clog.info("preprocessing data type: ssDNA/DAPI")
    clog.info("version: SAW_V_7_1")
    img = f_resize(img, input_size, "BILINEAR")
    img = f_ij_auto_contrast(img)
    if stain_type == TechType.ssDNA:
        img = f_equalize_adapthist_V2(img, 128)

    return img


def f_pre_ssdna_240618(img: npt.NDArray, input_size: tuple, stain_type: TechType) -> npt.NDArray:
    clog.info("preprocessing data type: ssDNA")
    clog.info("version: 240618")
    img = f_ij_auto_contrast_v2(img)
    img = f_resize(img, input_size, "BILINEAR")
    return img


def f_pre_he_240201(img: npt.NDArray, input_size: tuple, stain_type: TechType) -> npt.NDArray:
    clog.info("preprocessing data type: HE")
    clog.info("version: 240201")
    img = f_resize(img, input_size, "BILINEAR")
    img = img[:, :, 1]
    img = np.bitwise_not(img)
    img = f_ij_auto_contrast(img)
    return img


def f_pre_he_241018(img: npt.NDArray, input_size: tuple, stain_type: TechType) -> npt.NDArray:
    clog.info("preprocessing data type: HE")
    clog.info("version: 241018")
    img = f_resize(img, input_size)
    img = img.astype(np.float32)
    img = img/255
    return img


def f_pre_transcriptomics_protein_220909(img: npt.NDArray, input_size: tuple, stain_type: TechType) -> npt.NDArray:
    clog.info("preprocessing data type: Transcriptomics/Protein")
    clog.info("version: 220909")
    img[img > 0] = 255
    img = np.array(img).astype(np.uint8)
    img = f_resize(img, input_size, "BILINEAR")
    img = f_ij_auto_contrast(img)
    return img


def f_pre_if(img: npt.NDArray) -> npt.NDArray:
    return img


class TissueSegPreprocess:
    def __init__(self, model_name, support_model):
        self.model_preprocess = {
            support_model.SUPPORTED_MODELS[0]: {
                TechType.ssDNA: f_pre_ssdna_dapi_SAW_V_7_1,
                TechType.DAPI: f_pre_ssdna_dapi_SAW_V_7_1,
            },
            support_model.SUPPORTED_MODELS[1]: {
                TechType.ssDNA: f_pre_ssdna_240618,
            },
            support_model.SUPPORTED_MODELS[2]: {
                TechType.HE: f_pre_he_240201,
            },
            support_model.SUPPORTED_MODELS[3]: {
                TechType.HE: f_pre_he_241018,
            },
            support_model.SUPPORTED_MODELS[4]: {
                TechType.Transcriptomics: f_pre_transcriptomics_protein_220909,
                TechType.Protein: f_pre_transcriptomics_protein_220909
            },
            support_model.SUPPORTED_MODELS[5]: {
                TechType.IF: f_pre_if
            }
        }
        self.model_name = model_name
        self.m_preprocess: dict= self.model_preprocess[self.model_name]

    def __call__(self, img: Union[str, npt.NDArray], stain_type: TechType, input_size: tuple):

        # support image reading 
        if isinstance(img, str):
            img = self.im_read(img)

            if len(img.shape) == 3 and stain_type != TechType.HE:
                clog.warning(
                    'the input image is an RGB image, bug the stain type is not HE,convert the RGB image to GRAY image')
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # basic operation
        img = np.squeeze(img)
        if img.dtype != 'uint8' and stain_type != TechType.IF:
            img = f_ij_16_to_8_v2(img)

        # different process for diffrent staining
        pre_func = self.m_preprocess.get(stain_type)

        if stain_type == TechType.IF:
            img = pre_func(img)
            return img
        img = pre_func(img, input_size, stain_type)
        # basic operation
        img = f_histogram_normalization(img)
        if img.dtype != np.float32:
            img = np.array(img).astype(np.float32)
        img = np.ascontiguousarray(img)
        return img

    @staticmethod
    def im_read(im_path: str) -> npt.NDArray:
        img = cbimread(im_path, only_np=True)
        return img
