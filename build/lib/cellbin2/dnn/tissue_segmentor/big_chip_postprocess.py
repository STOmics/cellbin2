import numpy as np
import numpy.typing as npt
from skimage.exposure import rescale_intensity
from skimage.morphology import remove_small_objects
from cellbin2.image.morphology import f_fill_holes
from cellbin2.image.threshold import f_th_li, f_th_sauvola

from cellbin2.utils.common import TechType
from cellbin2.utils import clog
from cellbin2.dnn.tissue_segmentor.postprocess import f_post_if


def f_post_ssdna_dapi_SAW_V_7_1(img: npt.NDArray) -> npt.NDArray:
    clog.info("postprocessing data type: ssDNA/DAPI")
    clog.info("version: SAW_V_7_1")
    img = np.uint8(rescale_intensity(img, out_range=(0, 255)))
    img[img < 64] = 0
    img = f_th_sauvola(img, win_size=127, k=0.5, r=128.0)
    img = remove_small_objects(img, min_size=64, connectivity=2)
    img = f_fill_holes(img, size=64, connectivity=2)
    img = np.uint8(img)
    return img


def f_post_ssdna_240618(img: npt.NDArray) -> npt.NDArray:
    clog.info("postprocessing data type: ssDNA")
    clog.info("version: 240618")
    img = np.uint8(rescale_intensity(img, out_range=(0, 255)))
    img[img > 0] = 1
    return img


def f_post_he_240201(img: npt.NDArray) -> npt.NDArray:
    clog.info("postprocessing data type: HE")
    clog.info("version: 240201")
    img = np.uint8(rescale_intensity(img, out_range=(0, 255)))
    img = f_th_li(img)
    img = remove_small_objects(img, min_size=64, connectivity=2)
    img = f_fill_holes(img, size=64, connectivity=2)
    img = np.uint8(img)
    return img


def f_post_he_241018(img: npt.NDArray) -> npt.NDArray:
    clog.info("postprocessing data type: HE")
    clog.info("version: 241018")
    img = np.uint8(rescale_intensity(img, out_range=(0, 255)))
    img[img>0] = 1
    return img


def f_post_transcriptomics_protein_220909(img: npt.NDArray) -> npt.NDArray:
    clog.info("postprocessing data type: Transcriptomics/Protein")
    clog.info("version: 220909")
    img = np.uint8(rescale_intensity(img, out_range=(0, 255)))
    img = f_th_li(img)
    img = remove_small_objects(img, min_size=64, connectivity=2)
    img = f_fill_holes(img, size=64, connectivity=2)
    img = np.uint8(img)
    return img


class BigChipTissueSegPostprocess:
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

    def __call__(self, img: npt.NDArray, stain_type, threshold_list = None):
        post_func = self.m_postprocess.get(stain_type)
        if stain_type == TechType.IF:
            threshold, img = post_func(img, threshold_list)
            return threshold, img
        img = post_func(img)
        return img
