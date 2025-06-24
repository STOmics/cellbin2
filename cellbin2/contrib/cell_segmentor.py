import os
from typing import Union, Tuple, List
import matplotlib.pyplot as plt
import numpy.typing as npt
import numpy as np
from pydantic import BaseModel, Field

from cellbin2.dnn.segmentor.detector import Segmentation
from cellbin2.dnn.segmentor.cell_trace import get_trace as get_t
from cellbin2.dnn.segmentor.cell_trace import get_trace_v2 as get_t_v2
from cellbin2.dnn.segmentor.cell_trace import cal_area, cal_int, get_partial_res, cell_int_hist
from cellbin2.utils import clog
from cellbin2.utils.common import TechType
from cellbin2.contrib.fast_correct import run_fast_correct
from cellbin2.dnn.segmentor.postprocess import CellSegPostprocess
from cellbin2.dnn.segmentor.preprocess import CellSegPreprocess
from cellbin2.dnn.segmentor.utils import SUPPORTED_STAIN_TYPE_BY_MODEL, TechToWeightName
from cellbin2.utils.pro_monitor import process_decorator
from cellbin2.contrib.base_module import BaseModule
from cellbin2.image import cbimread
from cellbin2.utils.common import fPlaceHolder, iPlaceHolder
from cellbin2.utils.weights_manager import download_by_names


class CellSegParam(BaseModel, BaseModule):
    ssDNA_weights_path: str = Field("cellseg_bcdu_SHDI_221008_tf.onnx", description="name of the model")
    DAPI_weights_path: str = Field("cellseg_bcdu_SHDI_221008_tf.onnx", description="name of the model")
    HE_weights_path: str = Field("cellseg_bcdu_H_250303_tf_R.onnx", description="name of the model")
    IF_weights_path: str = Field("cyto2torch_0", description="name of the model")
    Transcriptomics_weights_path: str = Field("cellseg_unet_RNA_20230606.onnx", description="name of the model")
    Protein_weights_path: str = Field("cellseg_unet_RNA_20230606.onnx", description="name of the model")
    num_threads: int = Field(1, description="name of the model")
    GPU: int = Field(0, description="name of the model")

    # def get_weights_path(self, stain_type):
    #     if stain_type == TechType.ssDNA or stain_type == TechType.DAPI:
    #         p = self.ssDNA_weights_path
    #     elif stain_type == TechType.IF:
    #         p = self.IF_weights_path
    #     elif stain_type == TechType.HE:
    #         p = self.HE_weights_path
    #     elif stain_type == TechType.Transcriptomics or TechType.Protein:
    #         p = self.Transcriptomics_weights_path
    #     else: p = None
    #
    #     return p


class CellSegmentation:
    def __init__(
            self,
            cfg: CellSegParam,
            stain_type: TechType,
            gpu: int = -1,
            num_threads: int = 0,
    ):
        """
        Args:
            cfg (CellSegParam): network param
            stain_type (TechType): image stain type
            gpu (int): gpu index
            num_threads (int): default is 0,When you use the CPU,
        """
        super(CellSegmentation, self).__init__()
        self.cfg = cfg
        self.stain_type = stain_type
        self._model_path = self.cfg.get_weights_path(self.stain_type)
        if not os.path.exists(self._model_path):
            clog.info(f"{self._model_path} does not exists, will download automatically. ")
            download_by_names(
                save_dir=os.path.dirname(self._model_path),
                weight_names=[os.path.basename(self._model_path)]
            )
        # self._model_path = getattr(self.cfg, TechToWeightName[self.stain_type])
        self.model_name, self.mode = os.path.splitext(os.path.basename(self._model_path))
        if self.stain_type not in SUPPORTED_STAIN_TYPE_BY_MODEL[self.model_name]:
            clog.warning(
                f"{self.stain_type.name} not in supported list "
                f"{[i.name for i in SUPPORTED_STAIN_TYPE_BY_MODEL[self.model_name]]}"
            )
            return
        if self.stain_type == TechType.Transcriptomics:
            self._WIN_SIZE = (512, 512)
            self._OVERLAP = 0.1
        else:
            self._WIN_SIZE = (256, 256)
            self._OVERLAP = 16
        self.pre_process = CellSegPreprocess(
            model_name=self.model_name
        )
        self.post_process = CellSegPostprocess(
            model_name=self.model_name
        )
        self._gpu = gpu

        self._num_threads = num_threads

        self._cell_seg = Segmentation(
            mode=self.mode[1:],
            gpu=self._gpu,
            num_threads=self._num_threads,
            win_size=self._WIN_SIZE,
            overlap=self._OVERLAP,
            stain_type=self.stain_type,
            preprocess=self.pre_process,
            postprocess=self.post_process
        )
        clog.info("start loading model weight")
        self._cell_seg.f_init_model(model_path=self._model_path)
        clog.info("end loading model weight")

    @process_decorator('GiB')
    def run(self, img: Union[str, npt.NDArray]) -> npt.NDArray[np.uint8]:
        """
        run cell predict
        Args:
            img (ndarray): img array

        Returns:
            mask (ndarray)

        """
        if not hasattr(self, '_cell_seg'):
            clog.info(f"{self.__class__.__name__} failed to initialize, can not predict")
            mask = np.zeros_like(img, dtype='uint8')
        else:
            clog.info("start cell segmentation")
            mask = self._cell_seg.f_predict(img)
            clog.info("end cell segmentation")
        return mask

    @classmethod
    def run_fast(cls, mask: npt.NDArray, distance: int, process: int) -> npt.NDArray[np.uint8]:
        if distance > 0:
            fast_mask = run_fast_correct(
                mask_path=mask,
                distance=distance,
                n_jobs=process
            )
            return fast_mask
        else:
            clog.info(f"distance is: {distance} which is less than 0, return mask as it is")
            return mask

    @staticmethod
    def get_trace(mask):
        """
        2023/09/20 @fxzhao 对大尺寸图片采用加速版本以降低内存
        """
        if mask.shape[0] > 40000:
            return get_t_v2(mask)
        else:
            return get_t(mask)

    @classmethod
    def get_stats(
            cls,
            c_mask_p,
            cor_mask_p,
            t_mask_p,
            register_img_p,
            keep=5,
            size=1024,
            save_dir=None,
    ) -> Tuple[float, float, float, List[Tuple[npt.NDArray, List[Tuple[int, int, int, int]]]], plt.figure]:
        """
        Args:
            c_mask_p: 细胞分割mask，只接受单通道
            cor_mask_p: 修正mask，只接受单通道
            t_mask_p: 组织分割mask，只接受单通道
            register_img_p: 默认对RGB图做转灰度且反色的处理，若不是H&E染色的RGB图，请转成单通道再传入
            keep: 保留几张图
            size: 切图的尺寸
            save_dir: 存储路径

        Returns:
            Tuple[float, float, float, List[Tuple[npt.NDArray, List[Tuple[int, int, int, int]]]], plt.figure]:
                第一个元素：细胞分割mask与组织分割mask面积比值
                第二个元素：修正mask与组织分割mask面积比值
                第三个元素：细胞分割mask与组织分割mask亮度壁纸
                第四个元素：返回keep个大小为size的图以及位置坐标，位置坐标以y_begin, y_end, x_begin, x_end。选择局部的逻辑，是挑选的细胞分割结果最多的区域
                第五个元素：细胞分割mask的intensity图，是plt.figure类型
        Examples:
            >>> c_mask_p = "/media/Data/dzh/data/tmp/test_cseg_report/A01386A4_DAPI_mask.tif"
            >>> t_mask_p = "/media/Data/dzh/data/tmp/test_cseg_report/A01386A4_DAPI_tissue_cut.tif"
            >>> cor_mask_p = "/media/Data/dzh/data/tmp/test_cseg_report/A01386A4_DAPI_mask_edm_dis_10.tif"
            >>> register_img_p = "/media/Data/dzh/data/tmp/test_cseg_report/A01386A4_DAPI_regist.tif"
            >>> save_dir = "/media/Data/dzh/data/tmp/test_cseg_report"
            >>> area_ratio, area_ratio_cor, int_ratio, cell_with_outline, fig = CellSegmentation.get_stats(c_mask_p=c_mask_p,
            ... cor_mask_p=cor_mask_p,
            ... t_mask_p=t_mask_p,
            ... register_img_p=register_img_p,
            ... save_dir=save_dir)
            >>> assert area_ratio == 0.5619422933118077
            >>> assert area_ratio_cor == 0.8134636910954126
            >>> assert int_ratio == 0.7224478732571403
            >>> fig.savefig(os.path.join(save_dir, f"test.png"))
        """

        @process_decorator('GiB')
        def get_cell_stats():
            from cellbin2.image.augmentation import f_ij_16_to_8_v2
            from cellbin2.image import cbimread, cbimwrite

            # read image
            register_img = cbimread(register_img_p, only_np=True)
            c_mask = cbimread(c_mask_p, only_np=True)
            t_mask = cbimread(t_mask_p, only_np=True)
            if cor_mask_p != "":
                cor_mask = cbimread(cor_mask_p, only_np=True)
            else:
                cor_mask = None

            # 16 to 8
            register_img = f_ij_16_to_8_v2(register_img)

            # calculate area ratio
            area_ratio = cal_area(cell_mask=c_mask, tissue_mask=t_mask)  # 细胞分割mask与tissue mask的面积比值
            clog.info(f"cell mask area / tissue mask area = {area_ratio}")
            if cor_mask is not None:
                area_ratio_cor = cal_area(cell_mask=cor_mask, tissue_mask=t_mask)  # 修正mask与组织分割mask面积比值
            else:
                area_ratio_cor = fPlaceHolder
            clog.info(f"correct mask area / tissue mask area = {area_ratio_cor}")
            # calculate intensity ratio
            int_ratio = cal_int(
                c_mask=c_mask,
                t_mask=t_mask,
                register_img=register_img
            )  # 细胞分割mask与tissue mask的亮度比值
            clog.info(f"cell mask intensity / tissue mask intensity = {int_ratio}")

            # get cell mask int
            fig = cell_int_hist(c_mask=c_mask, register_img=register_img)
            clog.info(f"cell mask intensity calculation finished")

            # get partial vis images
            cell_with_outline = get_partial_res(
                c_mask=c_mask,
                t_mask=t_mask,
                register_img=register_img,
                keep=keep,
                k=size
            )
            if save_dir is not None:
                for i, v in enumerate(cell_with_outline):
                    im, box = v
                    box = [str(i) for i in box]
                    cord_str = "_".join(box)
                    save_path = os.path.join(save_dir, f"{cord_str}.tif")
                    cbimwrite(save_path, im)
            return area_ratio, area_ratio_cor, int_ratio, cell_with_outline, fig

        return get_cell_stats()


def s_main():
    import argparse
    from cellbin2.image import cbimwrite

    parser = argparse.ArgumentParser(description="you should add those parameter")
    parser.add_argument('-i', "--input", required=True, help="the input img path")
    parser.add_argument('-o', "--output", required=True, help="the output directory")
    parser.add_argument("-p", "--model", required=True, help="model directory")
    parser.add_argument("-s", "--stain", required=True, choices=['he', 'ssdna', 'dapi', 'scell'], help="stain type")
    parser.add_argument("-f", "--fast", action='store_true', help="if run fast correct")
    parser.add_argument("-t", "--tissue", action='store_true', help="if run tissue seg")
    parser.add_argument("-m", "--mode", choices=['onnx', 'tf'], help="onnx or tf", default="onnx")
    parser.add_argument("-g", "--gpu", type=int, help="the gpu index", default=0)
    args = parser.parse_args()

    usr_stype_to_inner = {
        'ssdna': TechType.ssDNA,
        'dapi': TechType.DAPI,
        "he": TechType.HE,
        "scell": TechType.Transcriptomics
    }

    input_path = args.input
    output_path = args.output
    model_dir = args.model
    mode = args.mode
    gpu = args.gpu
    user_s_type = args.stain
    fast = args.fast
    tc = args.tissue

    # stain type from user input to inner type
    s_type = usr_stype_to_inner.get(user_s_type)

    # name pattern
    name = os.path.splitext(os.path.basename(input_path))[0]
    os.makedirs(output_path, exist_ok=True)
    c_mask_path = os.path.join(output_path, f"{name}_v3_mask.tif")
    t_mask_path = os.path.join(output_path, f"{name}_tissue_mask.tif")
    f_mask_path = os.path.join(output_path, f"{name}_v3_corr_mask.tif")

    # tissue segmentation
    tm = None
    if tc:
        from cellbin2.utils.config import Config
        from cellbin2.contrib.tissue_segmentor import segment4tissue
        from cellbin2.contrib.tissue_segmentor import TissueSegInputInfo
        from cellbin2 import CB2_DIR
        c_file = os.path.join(CB2_DIR, 'cellbin2/config/cellbin.yaml')
        conf = Config(c_file, weights_root=model_dir)
        ti = TissueSegInputInfo(
            weight_path_cfg=conf.tissue_segmentation,
            input_path=input_path,
            stain_type=s_type,
        )
        to = segment4tissue(ti)
        tm = to.tissue_mask
        cbimwrite(t_mask_path, tm * 255)

    # get model path
    cfg = CellSegParam()
    for p_name in cfg.model_fields:
        default_name = getattr(cfg, p_name)
        if not p_name.endswith('_weights_path'):
            continue
        if mode == 'tf':
            default_name = default_name.replace(".onnx", ".hdf5")
        setattr(cfg, p_name, os.path.join(model_dir, default_name))

    c_mask, f_mask = segment4cell(
        input_path=input_path,
        cfg=cfg,
        s_type=s_type,
        gpu=gpu,
        fast=fast,
    )
    if tm is not None:
        c_mask = tm * c_mask

    # save mask
    cbimwrite(c_mask_path, c_mask * 255)
    if f_mask is not None:
        if tm is not None:
            f_mask = tm * f_mask
        cbimwrite(f_mask_path, f_mask * 255)


def segment4cell(
        input_path: str,
        cfg: CellSegParam,
        s_type: TechType,
        gpu: int,
        fast: bool
) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    # read user input image
    img = cbimread(input_path, only_np=True)

    # initialize cell segmentation model
    cell_seg = CellSegmentation(
        cfg=cfg,
        stain_type=s_type,
        gpu=gpu,
        num_threads=0,
    )

    # run cell segmentation
    mask = cell_seg.run(img=img)

    # fast correct
    fast_mask = None
    if fast:
        fast_mask = CellSegmentation.run_fast(mask=mask, distance=10, process=5)
        return mask, fast_mask

    return mask, fast_mask


if __name__ == '__main__':
    s_main()
