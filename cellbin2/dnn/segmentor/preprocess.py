from typing import Union
import numpy as np
import numpy.typing as npt
from skimage.exposure import rescale_intensity

from cellbin2.dnn.segmentor.utils import SUPPORTED_MODELS
from cellbin2.image.augmentation import f_rgb2gray, f_ij_auto_contrast_v3, f_ij_16_to_8_v2
from cellbin2.image.augmentation import f_percentile_threshold, f_histogram_normalization, f_equalize_adapthist
from cellbin2.image.augmentation import f_clahe_rgb
from cellbin2.utils.common import TechType
from cellbin2.image import cbimread


def f_pre_ssdna(img: npt.NDArray, enhance_times: int) -> npt.NDArray:
    if img.ndim == 3:
        img = f_rgb2gray(img, False)
    img = f_percentile_threshold(img)
    img = f_equalize_adapthist(img, 128, enhance_times)
    img = f_histogram_normalization(img)
    return img


def f_pre_rna(img: npt.NDArray, enhance_times: int) -> npt.NDArray:
    #img = np.where(img<3, 0, img)
    img = f_ij_auto_contrast_v3(img)
    return img


def f_pre_he(img: npt.NDArray, enhance_times: int) -> npt.NDArray:
    for _ in range(enhance_times or 1):
        img = f_clahe_rgb(img)

    if img.dtype != np.float32:
        img = np.array(img).astype(np.float32)
    img = rescale_intensity(img, out_range=(0.0, 1.0))
    return img


def f_pre_he_invert(img: npt.NDArray, enhance_times: int) -> npt.NDArray:
    img = f_rgb2gray(img, True)
    img = f_pre_ssdna(img)
    return img


model_preprocess = {
    SUPPORTED_MODELS[0]: {
        TechType.ssDNA: f_pre_ssdna,
        TechType.DAPI: f_pre_ssdna,
        TechType.HE: f_pre_he_invert
    },
    SUPPORTED_MODELS[1]: {
        TechType.HE: f_pre_he,
    },
    SUPPORTED_MODELS[2]: {
        TechType.Transcriptomics: f_pre_rna
    }
}

import tifffile as tif

class CellSegPreprocess:
    def __init__(self, model_name, enhance_times):
        self.model_name = model_name
        self.m_preprocess: dict = model_preprocess[self.model_name]
        self.enhance_times = enhance_times

    def __call__(self, img: Union[str, npt.NDArray], stain_type):
        # support image reading 
        if isinstance(img, str):
            img = self.im_read(img)

        # basic operation
        img = np.squeeze(img)
        if img.dtype != 'uint8':
            img = f_ij_16_to_8_v2(img)

        # different process for diffrent staining 
        pre_func = self.m_preprocess.get(stain_type)
        img = pre_func(img, self.enhance_times)

        # basic operation
        if img.dtype != np.float32:
            img = np.array(img).astype(np.float32)
        img = np.ascontiguousarray(img)
        # print(img.sum())
        return img

    def im_read(self, im_path: str) -> npt.NDArray:
        img = cbimread(im_path, only_np=True)
        return img
