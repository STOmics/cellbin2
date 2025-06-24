#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author  : hedongdong1
# @Time    : 2024/10/17 17:08
# @File    : preprocess.py
# @annotation    :
import copy
from typing import Union, List
import cv2
import numpy as np
import numpy.typing as npt
from cellbin2.image.augmentation import f_ij_auto_contrast, f_ij_auto_contrast_v2, f_ij_16_to_8_v2, f_equalize_adapthist_V2
from cellbin2.image.augmentation import f_histogram_normalization
from cellbin2.image.augmentation import f_resize
from cellbin2.utils.common import TechType
from cellbin2.image import cbimread
from cellbin2.utils import clog
from cellbin2.dnn.tissue_segmentor.preprocess import f_pre_if


def crop_image(image: npt.NDArray, chip_size: List) -> List:
    crop_img_list = []
    chip_hei, chip_wid = chip_size
    clog.info(f"chip size: w: {chip_wid}, h: {chip_hei}")

    if len(image.shape) == 3:
        img_hei, img_wid, _ = image.shape
    else:
        img_hei, img_wid = image.shape
    clog.info(f"image size: w: {img_wid}, h: {img_hei}")

    crop_wid = img_wid // chip_wid
    crop_hei = img_hei // chip_hei

    for i in range(chip_wid):
        for j in range(chip_hei):
            start_col = i * crop_wid
            if i != chip_wid - 1:
                end_col = (i + 1) * crop_wid
            else:
                end_col = img_wid

            start_row = j * crop_hei
            if j != chip_hei - 1:
                end_row = (j + 1) * crop_hei
            else:
                end_row = img_hei

            crop_img = image[start_row:end_row, start_col:end_col]
            crop_img_list.append(crop_img)
    return crop_img_list


def f_pre_ssdna_dapi_SAW_V_7_1(img: npt.NDArray, input_size: tuple, chip_size: List, stain_type: TechType) -> List:  #对应组织分割230523版本
    clog.info("preprocessing data type: ssDNA/DAPI")
    clog.info("version: SAW_V_7_1")
    img = f_resize(img, input_size, "BILINEAR")
    img = f_ij_auto_contrast(img)
    if stain_type == TechType.ssDNA:
        img = f_equalize_adapthist_V2(img, 128)

    img_list = crop_image(image=img, chip_size=chip_size)
    return img_list


def f_pre_ssdna_240618(img: npt.NDArray, input_size: tuple, chip_size: List, stain_type: TechType) -> List:
    clog.info("preprocessing data type: ssDNA")
    clog.info("version: 240618")
    img = f_ij_auto_contrast_v2(img)
    img = f_resize(img, input_size, "BILINEAR")

    img_list = crop_image(image=img, chip_size=chip_size)
    return img_list


def f_pre_he_240201(img: npt.NDArray, input_size: tuple, chip_size: List, stain_type: TechType) -> List:
    clog.info("preprocessing data type: HE")
    clog.info("version: 240201")
    img = f_resize(img, input_size, "BILINEAR")
    img = img[:, :, 1]
    img = np.bitwise_not(img)
    img = f_ij_auto_contrast(img)

    img_list = crop_image(image=img, chip_size=chip_size)
    return img_list


def f_pre_he_241018(img: npt.NDArray, input_size: tuple, chip_size: List, stain_type: TechType) -> List:
    clog.info("preprocessing data type: HE")
    clog.info("version: 241018")
    img = f_resize(img, input_size)
    img = img.astype(np.float32)
    img = img/255

    img_list = crop_image(image=img, chip_size=chip_size)
    return img_list


def f_pre_transcriptomics_protein_220909(img: npt.NDArray, input_size: tuple, chip_size: List, stain_type: TechType) -> List:
    clog.info("preprocessing data type: Transcriptomics/Protein")
    clog.info("version: 220909")
    img[img > 0] = 255
    img = np.array(img).astype(np.uint8)
    img = f_resize(img, input_size, "BILINEAR")
    img = f_ij_auto_contrast(img)

    img_list = crop_image(image=img, chip_size=chip_size)
    return img_list


class BigChipTissueSegPreprocess:
    def __init__(self, model_name, support_model, chip_size):
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
        self.chip_size = chip_size

    def __call__(self, img: Union[str, npt.NDArray], stain_type: TechType, input_size: tuple):

        # 支持读图
        if isinstance(img, str):
            img = self.im_read(img)

            if len(img.shape) == 3 and stain_type != TechType.HE:
                clog.warning(
                    'the input image is an RGB image, bug the stain type is not HE,convert the RGB image to GRAY image')
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 基操
        img = np.squeeze(img)
        if img.dtype != 'uint8' and stain_type != TechType.IF:
            img = f_ij_16_to_8_v2(img)

        # 不同染色不同操作
        pre_func = self.m_preprocess.get(stain_type)

        if stain_type == TechType.IF:
            img = pre_func(img)
            return img
        img_list = pre_func(img, input_size, self.chip_size, stain_type)
        # 基操
        image_list_ = []
        for img in img_list:
            img = f_histogram_normalization(img)
            if img.dtype != np.float32:
                img = np.array(img).astype(np.float32)
            img = np.ascontiguousarray(img)
            image_list_.append(img)
        return image_list_

    @staticmethod
    def im_read(im_path: str) -> npt.NDArray:
        img = cbimread(im_path, only_np=True)
        return img
