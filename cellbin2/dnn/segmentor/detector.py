import sys

from cellbin2.image.wsi_split import SplitWSI
from cellbin2.utils import clog
# from cellbin2.dnn.cseg import CellSegmentation
from cellbin2.dnn.segmentor.predict import CellPredict
from cellbin2.dnn.segmentor.processing import f_preformat, f_postformat, f_padding, f_fusion, f_preformat_rna, \
    f_postformat_rna
from cellbin2.dnn.onnx_net import OnnxNet
from cellbin2.utils.common import TechType
from cellbin2.dnn.segmentor.preprocess import CellSegPreprocess
from cellbin2.dnn.segmentor.postprocess import CellSegPostprocess

from skimage.morphology import remove_small_objects
import numpy as np
import numpy.typing as npt
from typing import Optional, Union


# TensorRT/ONNX
# HE/DAPI/mIF
class Segmentation:

    def __init__(
            self,
            model_path: str = "",
            mode: str = "onnx",
            gpu: int = -1,
            num_threads: int = 0,
            win_size: tuple = (256, 256),
            overlap: Union[float, int] = 16,
            stain_type: TechType = '',
            preprocess: Optional[CellSegPreprocess] = None,
            postprocess: Optional[CellSegPostprocess] = None
    ):
        self.model_path = model_path
        self._win_size = win_size
        self._overlap = overlap
        self.watershed_win_size = (4900, 4900)

        self._gpu = gpu
        self._mode = mode
        self._model: Optional[OnnxNet] = None
        self._sess: Optional[CellPredict] = None
        self._num_threads = num_threads
        self.stain_type = stain_type

        if self.stain_type == TechType.Transcriptomics:
            self.pre_fomat = f_preformat_rna
            self.post_format = f_postformat_rna
        else:
            self.pre_fomat = f_preformat
            self.post_format = f_postformat

        self.preprocess = preprocess
        self.postprocess = postprocess

    def f_init_model(self, model_path):
        """
        init model
        """
        self._model = OnnxNet(model_path, self._gpu, self._num_threads)
        self._sess = CellPredict(self._model, self.pre_fomat, self.post_format)
        pre_str = ''
        for key, val in self.preprocess.m_preprocess.items():
            pre_str += f"({key.name}: {val.__name__})--"
        post_str = ''
        for key, val in self.postprocess.m_postprocess.items():
            post_str += f"({key.name}: {val.__name__})--"
        clog.info(f"Pre-process--> {pre_str}")
        clog.info(f"Post-process--> {post_str}")

    def f_predict(self, img: Union[str, npt.NDArray]) -> npt.NDArray[np.uint8]:
        """

        :param img:CHANGE
        :return: full-size mask image
        2023/09/21 @fxzhao set need_fun_ret as False, this result is not ued in current version 
        """
        # 1. preprocess image
        img = self.preprocess(
            img=img,
            stain_type=self.stain_type
        )
        # 2. predict
        sp_run = SplitWSI(
            img=img,
            win_shape=self._win_size,
            overlap=self._overlap,
            batch_size=100,
            need_fun_ret=False,
            need_combine_ret=True,
            editable=False,
            tar_dtype=np.uint8,
            dst_shape=(img.shape[:2]),
            win_back=True
        )
        sp_run.f_set_run_fun(self._sess.f_predict)
        sp_run.f_set_pre_fun(f_padding, self._win_size)
        # sp_run.f_set_fusion_fun(f_fusion)
        _, _, pred_raw = sp_run.f_split2run()
        # print(pred_raw.sum())

        # 3. postprocess
        if self.stain_type == TechType.Transcriptomics:
            pred = self.postprocess(pred_raw, self.stain_type)
            pred = remove_small_objects(pred.astype(np.bool8), min_size=15, connectivity=2).astype(np.uint8)
        else:
            # post processing
            sp_run2 = SplitWSI(pred_raw, self.watershed_win_size, self._overlap, 1, False, True, False, np.uint8)
            sp_run2.f_set_run_fun(self.postprocess, self.stain_type)
            sp_run2.f_set_pre_fun(f_padding, self.watershed_win_size)
            sp_run2.f_set_fusion_fun(f_fusion)
            _, _, pred = sp_run2.f_split2run()
            pred = remove_small_objects(pred.astype(np.bool8), min_size=15, connectivity=2).astype(np.uint8)
        return pred


if __name__ == '__main__':
    pass
