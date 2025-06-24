from typing import Dict, Optional, Union
import os
import numpy as np
# from math import ceil
# from sklearn.cluster import DBSCAN
from pydantic import BaseModel, Field

from cellbin2.dnn.classify.onnx_mobilenet3 import OnnxMobileNet3
from cellbin2.image.augmentation import f_ij_16_to_8
from cellbin2.utils import clog
from cellbin2.utils.common import TechType
from cellbin2.image.augmentation import f_rgb2gray, dapi_enhance, he_enhance
from cellbin2.image import cbimread
from cellbin2.contrib.base_module import BaseModule

SUPPORTED_STAIN_TYPE = (TechType.ssDNA, TechType.DAPI)
weight_name_ext = '_weights_path'
TechToWeightName = {i.value: i.name + weight_name_ext for i in SUPPORTED_STAIN_TYPE}

pt_enhance_method = {
    TechType.ssDNA.value: dapi_enhance,
    TechType.DAPI.value: dapi_enhance,
    TechType.HE.value: he_enhance
}


class ClarityOutput(BaseModel):
    score: float = Field(description="清晰度得分")
    pred: np.ndarray = Field(description="预测结果")
    cut_siz: tuple
    overlap: int

    class Config:
        arbitrary_types_allowed = True


class ClarityParam(BaseModel, BaseModule):
    GPU: int = Field(0, description="推理使用的GPU编号")
    num_threads: int = Field(0, description="推理使用的线程数")
    # DEFAULT_STAIN_TYPE: TechType = TechType.ssDNA
    # SUPPORTED_STAIN_TYPE: tuple = (TechType.ssDNA, TechType.DAPI)
    ssDNA_weights_path: str = Field(
        "clarity_eval_mobilev3small05064_DAPI_20230608_pytorch.onnx", description="ssDNA染色图对应的权重文件名")
    DAPI_weights_path: str = Field(
        "clarity_eval_mobilev3small05064_DAPI_20230608_pytorch.onnx", description="DAPI染色图对应的权重文件名")

    # def update_weights_path(self, weights_path: str):
    #     self.ssDNA_weights_path = os.path.join(weights_path, self.ssDNA_weights_path)
    #     self.DAPI_weights_path = os.path.join(weights_path, self.DAPI_weights_path)
    #
    # def get_weights_path(self, stain_type):
    #     if stain_type == TechType.ssDNA:
    #         p = self.ssDNA_weights_path
    #     elif stain_type == TechType.DAPI:
    #         p = self.DAPI_weights_path
    #     else:
    #         p = None
    #
    #     return p


COLOR_SET = {
    'red': (255, 0, 0),
    'lightsalmon': (255, 160, 122),
    'blue': (0, 0, 255),
    'lightblue': (173, 216, 230),
    'yellow': (255, 255, 0),
    'green': (0, 128, 0),
    'pink': (255, 203, 192),
    'black': (0, 0, 0)
}

REPRESENT_4 = {
    0: 'black',
    1: 'blur',
    2: 'good',
    3: 'over_expo',
    -1: 'uncertain',
}

REPRESENT_6 = {
    0: 'black',
    1: 'good',
    2: 'first_level_blur',
    3: 'second_level_blur',
    4: 'first_level_over_expo',
    5: 'second_level_over_expo',
    -1: 'uncertain',
}

COLOR_4 = {
    'black': 'yellow',
    'blur': 'blue',
    'good': 'green',
    'over_expo': 'red',
    'uncertain': 'pink'
}

COLOR_6 = {
    'black': 'black',
    'first_level_blur': 'blue',
    'second_level_blur': 'blue',
    'good': 'green',
    'first_level_over_expo': 'red',
    'second_level_over_expo': 'red',
    'uncertain': 'pink'
}

WEIGHT_MAP_6 = {
    0: 0,
    1: 1,
    2: 0.2,
    3: 0,
    4: 0.9,
    5: 0
}

WEIGHT_MAP_4 = {
    0: 0,
    1: 0,
    2: 1,
    3: 0,
}


class ClarityQC(object):
    def __init__(
            self,
            cfg: ClarityParam,
            stain_type: TechType
    ) -> None:
        """
            Load clarity model

            Args:
                cfg: ClarityParam
                stain_type: 染色类型
        """
        if stain_type not in SUPPORTED_STAIN_TYPE:
            clog.info(f"Clarity eval only support {[i.name for i in SUPPORTED_STAIN_TYPE]}, fail to initialize")
            return
        self.cfg: ClarityParam = cfg
        self.stain_type = stain_type
        # 模型初始化
        self.cl_classify: Optional[OnnxMobileNet3] = None
        self.cl_classify = OnnxMobileNet3(
            weight_path=getattr(self.cfg, TechToWeightName[self.stain_type.value]),
            batch_size=2000,
            conf_thresh=0.0,
            gpu=self.cfg.GPU,
            num_threads=self.cfg.num_threads,
        )
        self.pre_func = None
        self.num_class = self.cl_classify.output_shape[1]
        if self.num_class == 4:
            self.represent = REPRESENT_4
            self.color = COLOR_4
            self.weight_map = WEIGHT_MAP_4
        elif self.num_class == 6:
            self.represent = REPRESENT_6
            self.color = COLOR_6
            self.weight_map = WEIGHT_MAP_6

        # 结果
        self.counts: Dict[str, int] = {}
        self.score: float = -1.0
        self.preds: np.ndarray = np.array([])  # [[class, prob], ...]
        self.box_lst: list = []  # [[y_begin, y_end, x_begin, x_end], ...]
        self.boxes_: list = []

    def run(self, img: np.ndarray):
        """
        This function will spilit the input image into (64, 64) pieces, then classify each piece into category.
        Category is ['black', 'over_exposure', 'blur', 'good']

        Args:
            img (): stitched image after tissue cut (numpy ndarray)

        Returns:
            self.counts: counts of each category ['black', 'blur', 'good', 'over_expo']
            self.score: clarity score
            self.preds: prediction in
                - shape is ceil(image_height / (64 - _overlap)),  ceil(image_width / (64 - self._overlap), 2)
                - 2 -> 1st: class, 2nd probability
            self.boxes: the pieces coordinate
                - [[y_begin, y_end, x_begin, x_end], ...]
        Examples:
            >>> import tifffile
            >>> weight_path = "/media/Data/dzh/weights/clarity_eval_mobilev3small05064_DAPI_20230608_pytorch.onnx"
            >>> models_config = {'ssDNA_weights_path': weight_path, 'DAPI_weights_path': weight_path,}
            >>> stain_type = TechType.ssDNA
            >>> cfg = ClarityParam(**models_config)
            >>> clarity_qc = ClarityQC(cfg=cfg, stain_type=stain_type)
            >>> img_path = "/media/Data/dzh/data/cellbin2/TRACK_DETECT/A02786D312/stitch/A02786D312_fov_stitched.tif"
            >>> img = tifffile.imread(img_path)
            >>> img = img[2000: 2000 + 5000, 2000: 2000 + 5000]
            >>> clarity_qc.run(img)
            >>> clarity_qc.score
            0.712282309807516

        """
        if not hasattr(self, 'cl_classify'):
            clog.info(f"{self.__class__.__name__} failed to initialize, can not detect")
            return
        clog.info(f"Clarity eval input has {img.ndim} dims, using enhance func {self.pre_func}")
        if not isinstance(img, np.ndarray):
            raise Exception(f"Only accept numpy array as input")
        if img.ndim == 3:
            clog.warning(
                f"Clarity eval only accept single channel image, Your input has more than one channel, Will convert "
                f"to single channel "
            )
            img = f_rgb2gray(img)
        if img.dtype != np.uint8:
            img = f_ij_16_to_8(img)

        counts, preds, box_lst = self.cl_classify.inference(img)
        counts = {self.represent[k]: v for k, v in counts.items()}
        score = self.cl_classify.score_calculator(preds, self.weight_map)
        self.counts = counts
        self.score = score
        self.preds = preds
        self.box_lst = box_lst

        clog.info(f"Clarity eval counts: {self.counts}")
        clog.info(f"Clarity eval score: {self.score}")

    @classmethod
    def post_process(cls, preds: np.ndarray) -> np.ndarray:
        """
        Args:
            preds: 需要从ipr中读取

        Returns:
            show_arr: numpy array
        """
        import cv2
        preds = preds[:, :, 0]
        h, w = preds.shape[:2]
        show_arr = np.zeros((h, w, 3), dtype='uint8')
        for i in range(show_arr.shape[0]):
            for j in range(show_arr.shape[1]):
                cur_class = preds[i, j]
                cur_color = COLOR_SET[COLOR_6[REPRESENT_6[cur_class]]]
                show_arr[i, j] = cur_color
        show_arr = cv2.cvtColor(show_arr, cv2.COLOR_RGB2BGR)
        return show_arr


def run_detect(
        img_file: str,
        cfg: ClarityParam,
        stain_type: TechType,
):
    clarity_qc = ClarityQC(cfg=cfg, stain_type=stain_type)
    img = cbimread(img_file)
    clarity_qc.run(img.image)
    clarity_out = ClarityOutput(
        score=clarity_qc.score,
        pred=clarity_qc.preds,
        cut_siz=clarity_qc.cl_classify.img_size,
        overlap=clarity_qc.cl_classify.overlap
    )

    return clarity_out


if __name__ == '__main__':
    import tifffile
    from cellbin2.image.augmentation import f_gray2bgr

    weight_path = r"E:\03.users\liuhuanlin\01.data\cellbin2\weights\clarity_eval_mobilev3small05064_DAPI_20230608_pytorch.onnx"
    models_config = {'ssDNA_weights_path': weight_path, 'DAPI_weights_path': weight_path}
    stain_type = TechType.ssDNA
    cfg = ClarityParam(**models_config)

    img_path = r"E:\03.users\liuhuanlin\01.data\cellbin2\cellbin2_test\image\A03599D1_DAPI_fov_stitched.tif"
    run_detect(img_file=img_path, cfg=cfg, stain_type=stain_type)
