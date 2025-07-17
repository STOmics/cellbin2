from cellbin2.dnn.onnx_net import OnnxNet
from cellbin2.dnn.tissue_segmentor.preprocess import TissueSegPreprocess
from cellbin2.dnn.tissue_segmentor.postprocess import TissueSegPostprocess
from cellbin2.dnn.tissue_segmentor.big_chip_preprocess import BigChipTissueSegPreprocess
from cellbin2.dnn.tissue_segmentor.big_chip_postprocess import BigChipTissueSegPostprocess
from cellbin2.dnn.tissue_segmentor.processing import f_preformat, f_postformat
from cellbin2.image.augmentation import f_resize
from typing import Optional, Union, List, Tuple
from cellbin2.utils.common import TechType
from cellbin2.utils import clog
from cellbin2.contrib.param import TissueSegOutputInfo
import numpy.typing as npt
import numpy as np


def stitch_mask(mask_list: List[npt.NDArray], chip_size: List) -> npt.NDArray:
    chip_hei, chip_wid = chip_size
    idx = 0
    stitched_output_mask = None
    for i in range(chip_wid):
        vertical_stitch_mask = None
        for j in range(chip_hei):
            if vertical_stitch_mask is None:
                vertical_stitch_mask = mask_list[idx]
            else:
                vertical_stitch_mask = np.concatenate((vertical_stitch_mask, mask_list[idx]), axis=0)

            idx += 1
        if stitched_output_mask is None:
            stitched_output_mask = vertical_stitch_mask
        else:
            stitched_output_mask = np.concatenate((stitched_output_mask, vertical_stitch_mask), axis=1)

    return stitched_output_mask


class TissueSegmentationBcdu(object):
    def __init__(self,
                 input_size: tuple = (512, 512, 1),
                 stain_type: TechType = '',
                 threshold_list: Tuple[int, int] = None,
                 gpu: int = -1,
                 mode: str = "onnx",
                 num_threads: int = 0,
                 preprocess: Union[TissueSegPreprocess, BigChipTissueSegPreprocess] = None,
                 postprocess: Union[TissueSegPostprocess, BigChipTissueSegPostprocess] = None,
                 ):

        self.INPUT_SIZE = input_size

        self.stain_type = stain_type
        self.gpu = gpu
        self.mode = mode
        self.model = None
        self.mask_num = None
        self.num_threads = num_threads
        self.threshold_list = threshold_list

        self.pre_format = f_preformat
        self.post_format = f_postformat

        self.preprocess = preprocess
        self.postprocess = postprocess

    def f_init_model(self, model_path):
        self.model = OnnxNet(model_path=model_path, gpu=self.gpu, num_threads=self.num_threads)
        self.INPUT_SIZE = self.model.f_get_input_shape()

        clog.info(f'model input size:{self.INPUT_SIZE}')

    def f_predict(self, img, binx) -> TissueSegOutputInfo:
        pred_out = TissueSegOutputInfo()
        if self.stain_type == TechType.IF:
            img = self.preprocess(img=img, stain_type=self.stain_type, input_size=None, binx=binx)
            threshold_list, pred = self.postprocess(img=img, stain_type=self.stain_type, src_shape=None, threshold_list=self.threshold_list)
            pred_out.tissue_mask = pred

            pred_out.threshold_list = threshold_list
            return pred_out
        src_shape = img.shape[:2]
        img = self.pre_format(self.preprocess(img=img, stain_type=self.stain_type, input_size=self.INPUT_SIZE))

        pred = self.model.f_predict(img)
        pred = self.postprocess(self.post_format(pred), stain_type=self.stain_type, src_shape=src_shape)

        pred_out.tissue_mask = pred
        return pred_out


    def f_predict_big_chip(self, img, chip_size, binx) -> TissueSegOutputInfo:
        pred_out = TissueSegOutputInfo()
        if self.stain_type == TechType.IF:
            img = self.preprocess(img=img, stain_type=self.stain_type, input_size=None, binx=1)
            threshold_list, pred = self.postprocess(img=img, stain_type=self.stain_type, threshold_list=self.threshold_list)
            pred_out.tissue_mask = pred

            pred_out.threshold_list = threshold_list
            return pred_out
        target_size = (int(chip_size[0] * self.INPUT_SIZE[0]), int(chip_size[1] * self.INPUT_SIZE[1]))  # zoom out image to target size
        src_shape = img.shape[:2]
        img_list = self.preprocess(img=img, stain_type=self.stain_type, input_size=target_size)
        pred_list = []
        for img in img_list:
            img = self.pre_format(img=img)
            tmp_predict = self.model.f_predict(img)
            tmp_predict = self.postprocess(self.post_format(tmp_predict), stain_type=self.stain_type)
            pred_list.append(self.post_format(tmp_predict))
        pred = stitch_mask(mask_list=pred_list, chip_size=chip_size)
        pred = f_resize(pred, src_shape)  # zoom in image to origin shape
        pred_out.tissue_mask = pred
        return pred_out
