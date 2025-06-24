from typing import Tuple, Union
import numpy as np
import numpy.typing as npt
from numpy import ndarray
from skimage.exposure import rescale_intensity
from skimage.morphology import remove_small_objects
import cv2
from cellbin2.image.augmentation import f_resize
from cellbin2.image.morphology import f_fill_holes
from cellbin2.image.threshold import f_th_li, f_th_sauvola
from cellbin2.utils.common import TechType
from cellbin2.utils import clog


def transfer_16bit_to_8bit(image_16bit: np.ndarray) -> np.ndarray:
    """
    Transfer the bit deepth of image from 16bit to 8bit
    """
    min_16bit = np.min(image_16bit)
    max_16bit = np.max(image_16bit)
    div = 255 / (max_16bit - min_16bit)

    image_8bit = np.zeros(image_16bit.shape, dtype=np.uint8)
    chunk_size = 10000
    for idx in range(image_16bit.shape[0] // chunk_size + 1):
        s = slice(idx * chunk_size, (idx + 1) * chunk_size)
        image_8bit[s] = np.array(
            np.rint((image_16bit[s, :] - min_16bit) * div), dtype=np.uint8
        )

    return image_8bit


def f_post_ssdna_dapi_SAW_V_7_1(img: npt.NDArray, src_shape: tuple) -> npt.NDArray:
    clog.info("postprocessing data type: ssDNA/DAPI")
    clog.info("version: SAW_V_7_1")
    img = np.uint8(rescale_intensity(img, out_range=(0, 255)))
    img[img < 64] = 0
    img = f_th_sauvola(img, win_size=127, k=0.5, r=128.0)
    img = remove_small_objects(img, min_size=64, connectivity=2)
    img = f_fill_holes(img, size=64, connectivity=2)
    img = np.uint8(img)
    img = f_resize(img, src_shape)
    # img[img > 0] = 1
    return img


def f_post_ssdna_240618(img: npt.NDArray, src_shape: tuple) -> npt.NDArray:
    clog.info("postprocessing data type: ssDNA")
    clog.info("version: 240618")
    img = np.uint8(rescale_intensity(img, out_range=(0, 255)))
    for i in range(5):  # iteratively upscale and apply mean filtering to smooth the edges of the generated mask
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        img = cv2.blur(img, (5, 5))
    img = cv2.resize(img, (src_shape[1], src_shape[0]), interpolation=cv2.INTER_LINEAR)
    img = cv2.blur(img, (5, 5))
    img[img > 0] = 1
    return img


def f_post_he_240201(img: npt.NDArray, src_shape: tuple) -> npt.NDArray:
    clog.info("postprocessing data type: HE")
    clog.info("version: 240201")
    img = np.uint8(rescale_intensity(img, out_range=(0, 255)))
    img = f_th_li(img)
    img = remove_small_objects(img, min_size=64, connectivity=2)
    img = f_fill_holes(img, size=64, connectivity=2)
    img = np.uint8(img)
    img = f_resize(img, src_shape)
    return img


def f_post_he_241018(img: npt.NDArray, src_shape: tuple) -> npt.NDArray:
    clog.info("postprocessing data type: HE")
    clog.info("version: 241018")
    img = np.uint8(rescale_intensity(img, out_range=(0, 255)))
    img = f_resize(img, src_shape)
    img[img>0] = 1
    return img


def f_post_transcriptomics_protein_220909(img: npt.NDArray, src_shape: tuple) -> npt.NDArray:
    clog.info("postprocessing data type: Transcriptomics/Protein")
    clog.info("version: 220909")
    img = np.uint8(rescale_intensity(img, out_range=(0, 255)))
    img = f_th_li(img)
    img = remove_small_objects(img, min_size=64, connectivity=2)
    img = f_fill_holes(img, size=64, connectivity=2)
    img = np.uint8(img)
    img = f_resize(img, src_shape)
    return img


def f_post_if(img: npt.NDArray, threshold_list: Tuple[int, int]) -> Tuple[Tuple[Union[float, int], Union[float, int]], ndarray]:
    clog.info("postprocessing data type: IF")
    clog.info(f"threshold_list: {threshold_list}")
    if threshold_list:
        l, u = threshold_list
        img[img < l] = 0
        img[img > u] = 0
        img[img > 0] = 1
        if img.dtype == 'uint16':
            mask = np.asarray(img, dtype=np.uint8)
        else:
            mask = img

        return threshold_list, mask
    else:
        if len(img.shape) != 2:
            clog.error(f"image shape error,only support gray image, image shape: {img.shape}")
            clog.info('return input image')
            return (-1, -1), img
        img = cv2.GaussianBlur(img, (5, 5), 0)
        threshold, mask = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

        if img.dtype == 'uint16':
            upper_threshold = 65535
        else:
            upper_threshold = 255
        threshold_list = threshold, upper_threshold

        if mask.dtype != "uint8":
            mask = transfer_16bit_to_8bit(mask)
        return threshold_list, mask


class TissueSegPostprocess:
    def __init__(self, model_name, support_model):
        self.model_postprocess = {
            support_model.SUPPORTED_MODELS[0]: {
                TechType.ssDNA: f_post_ssdna_dapi_SAW_V_7_1,
                TechType.DAPI: f_post_ssdna_dapi_SAW_V_7_1,
            },
            support_model.SUPPORTED_MODELS[1]: {
                TechType.ssDNA: f_post_ssdna_240618,
            },
            support_model.SUPPORTED_MODELS[2]: {
                TechType.HE: f_post_he_240201,
            },
            support_model.SUPPORTED_MODELS[3]: {
                TechType.HE: f_post_he_241018,
            },
            support_model.SUPPORTED_MODELS[4]: {
                TechType.Transcriptomics: f_post_transcriptomics_protein_220909,
                TechType.Protein: f_post_transcriptomics_protein_220909
            },
            support_model.SUPPORTED_MODELS[5]: {
                TechType.IF: f_post_if
            }
        }
        self.model_name = model_name
        self.m_postprocess: dict = self.model_postprocess[self.model_name]

    def __call__(self, img: npt.NDArray, stain_type, src_shape, threshold_list = None):
        post_func = self.m_postprocess.get(stain_type)
        if stain_type == TechType.IF:
            threshold, img = post_func(img, threshold_list)
            return threshold, img
        img = post_func(img, src_shape)
        return img
