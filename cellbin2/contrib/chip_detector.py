import cv2
import numpy as np

from typing import List, Tuple, Dict
from cellbin2.utils import clog
from pydantic import BaseModel, Field
from scipy.spatial.distance import cdist

import cellbin2.image as cbi

from cellbin2.image.augmentation import f_ij_16_to_8
from cellbin2.dnn.detector import OBB8Detector, Yolo8Detector
from cellbin2.utils.common import TechType
from typing import Union
from cellbin2.image import CBImage
from cellbin2.contrib.alignment.basic import transform_points, ChipBoxInfo
from cellbin2.contrib.base_module import BaseModule
from cellbin2.utils.plot_funcs import get_view_image

SUPPORTED_STAIN_TYPE = (TechType.ssDNA, TechType.DAPI, TechType.HE)
weight_name_ext = '_weights_path'
TechToWeightName = {i.value: i.name.lower() + weight_name_ext for i in SUPPORTED_STAIN_TYPE}


class ChipParam(BaseModel, BaseModule):
    detect_channel: int = Field(-1, description="若输入图为3通道，需指明检测通道。否则，程序会自动转为单通道图")
    stage1_weights_path: str = Field(
        "chip_detect_obb8n_1024_SDH_stage1_202410_pytorch.onnx", description="ssDNA染色图对应的权重文件名")
    stage2_weights_path: str = Field(
        "chip_detect_yolo8x_1024_SDH_stage2_202410_pytorch.onnx", description="ssDNA染色图对应的权重文件名")
    GPU: int = Field(0, description="推理使用的GPU编号")
    num_threads: int = Field(0, description="推理使用的线程数")

    def get_stage1_weights_path(self, ):
        return self.stage1_weights_path

    def get_stage2_weights_path(self, ):
        return self.stage2_weights_path


class ChipDetector(object):
    """ 图像数据： 芯片区域检测器 """

    PADDING_SIZE = 1000

    def __init__(self,
                 cfg: ChipParam,
                 stain_type: TechType):
        """

        Args:
            cfg:
            stain_type:

        Returns:

        Examples:


        """
        if stain_type not in SUPPORTED_STAIN_TYPE:
            clog.info(f"Track detect only support {[i.name for i in SUPPORTED_STAIN_TYPE]}, fail to initialize")
            return
        # 初始化
        if cfg is not None: self.cfg: ChipParam = cfg
        else: self.cfg = ChipParam()

        self.stain_type = stain_type
        self.chip_actual_size = (None, None)

        # output
        self.left_top: List[float] = [0.0, 0.0]
        self.right_top: List[float] = [0.0, 0.0]
        self.left_bottom: List[float] = [0.0, 0.0]
        self.right_bottom: List[float] = [0.0, 0.0]
        self.scale_x: float = 1.0
        self.scale_y: float = 1.0
        self.rotation: float = 0.0
        self.is_available: bool = True
        self.chip_size: Tuple[float, float] = (0.0, 0.0)

        self.source_image = None

        self.onnx_model_global = self.cfg.get_stage1_weights_path()
        self.onnx_model_local = self.cfg.get_stage2_weights_path()

        self.rough_corner_points = None
        self.finetune_corner_points = None

        self.set_points_flag = False

    def set_corner_points(self, points: np.ndarray):
        """

        Args:
            points:

        Returns:

        """
        if isinstance(points, np.ndarray):
            if points.shape == (4, 2):
                self.finetune_corner_points = points
                self.set_points_flag = True
                clog.info(f"Set corner points done.")

    def control(
            self,
            threshold_length_rate: float = 0.05,
            threshold_rotate: float = 2,
    ):
        """

        Args:
            threshold_length_rate: 长宽比例 0.05的比值误差
            threshold_rotate: 夹角最大误差

        Returns:

        """
        self.is_available = True
        # 长宽比判断
        dist = cdist(self.finetune_corner_points, self.finetune_corner_points)
        dist_list = [dist[i, (i + 1) % dist.shape[0]] for i in range(dist.shape[0])]
        dist_rate = np.matrix([dist_list]) / np.matrix([dist_list]).T
        clog.info(f"Chip detector -> length-to-width ratio == "
                  f"max: {np.round(np.max(dist_rate), 5)}  min: {np.round(np.min(dist_rate), 5)}")

        _dr = (self.chip_actual_size[0] / self.chip_actual_size[1],
               self.chip_actual_size[1] / self.chip_actual_size[0])

        # if np.any(np.abs(dist_rate - 1) > threshold_length_rate):
        if np.abs(np.max(dist_rate) - np.max(_dr)) > threshold_length_rate or \
                np.abs(np.min(dist_rate) - np.min(_dr)) > threshold_length_rate:
            self.is_available = False

        # 内角关系判断
        fcp = self.finetune_corner_points
        k_list = [(fcp[i, 1] - fcp[(i + 1) % fcp.shape[0], 1]) /
                  (fcp[i, 0] - fcp[(i + 1) % fcp.shape[0], 0])
                  for i in range(fcp.shape[0])]
        r_list = list(map(lambda x: np.degrees(np.arctan(x)), k_list))

        included_angle = list()
        for i in range(len(r_list)):
            _r = r_list[(i + 1) % len(r_list)] - r_list[i]
            if _r < 0: _r = 180 + _r
            included_angle.append(_r)
        # included_angle = [np.abs(180 * (i % 2) - (r_list[(i + 1) % len(r_list)] - r_list[i]))
        #                   for i in range(len(r_list))]
        clog.info(f"Chip detector -> included angle == {list(map(lambda x: np.round(x, 5), included_angle))}")

        included_angle = np.abs(np.array(list(map(lambda x: 90 - x, included_angle))))
        if np.any(included_angle > threshold_rotate):
            self.is_available = False
        clog.info(f"Chip detector -> is available == {self.is_available}")

    def detect(self, file_path: str, actual_size: Tuple[int, int]):
        """ 入口函数

        Args:
            file_path:
            actual_size:

        Returns:

        """
        self.parse_image(file_path, actual_size)

        if not self.set_points_flag:
            self.stage_rough()
            self.stage_finetune()

        self.parse_info()
        self.control()

    def parse_image(self, img, actual_size):
        """

        Args:
            img:
            actual_size:

        Returns:

        """
        self.chip_actual_size = actual_size

        # Read the input image using CBI
        self.source_image = cbi.cbimread(img, only_np = True)
        self.source_image = f_ij_16_to_8(self.source_image)

    def parse_info(self):
        """ 回传参数

        Returns:

        """
        self.rotation = self.calculate_rotation_angle(self.finetune_corner_points)
        self.chip_size = (cdist([self.finetune_corner_points[0]], [self.finetune_corner_points[1]])[0][0],
                cdist([self.finetune_corner_points[0]], [self.finetune_corner_points[3]])[0][0])

        clog.info('On image, chip size == {}'.format(self.chip_size))
        _sx = np.max(self.chip_size) / np.max(self.chip_actual_size)
        _sy = np.min(self.chip_size) / np.min(self.chip_actual_size)

        if self.chip_actual_size[1] > self.chip_actual_size[0]:
            self.scale_x, self.scale_y = _sx, _sy
        else:
            self.scale_x, self.scale_y = _sy, _sx
        clog.info('Calculate scale(XY) == ({}, {})'.format(self.scale_x, self.scale_y))

        self.left_top, self.left_bottom, self.right_bottom, self.right_top = \
            [list(i) for i in self.finetune_corner_points]

    def stage_rough(self):
        """ 粗定位框

        Returns:

        """
        obb8_detector = OBB8Detector(self.onnx_model_global, self.source_image)
        self.rough_corner_points = obb8_detector.run()

    def stage_finetune(self):
        """ 微调框

        Returns:

        """
        self.rough_corner_points = self.check_border(self.rough_corner_points)
        rotate = self.calculate_rotation_angle(self.rough_corner_points)

        rotated_image = cbi.cbimread(self.source_image)
        rotated_image = rotated_image.trans_image(rotate=rotate)
        new_corner_points, M = transform_points(
            points=self.rough_corner_points,
            rotation=-rotate,
            src_shape=self.source_image.shape
        )
        rotated_image = rotated_image.image

        rotated_image = self.padding_border(rotated_image, self.PADDING_SIZE)
        new_corner_points += self.PADDING_SIZE

        # detector init
        new_points = list()
        for i, _p in enumerate(new_corner_points):
            x, y = map(lambda k: self.PADDING_SIZE if k < self.PADDING_SIZE else int(k), _p)
            if x > rotated_image.shape[1] - self.PADDING_SIZE: x = rotated_image.shape[1] - self.PADDING_SIZE
            if y > rotated_image.shape[0] - self.PADDING_SIZE: y = rotated_image.shape[0] - self.PADDING_SIZE

            _img = rotated_image[y - self.PADDING_SIZE: y + self.PADDING_SIZE,
                   x - self.PADDING_SIZE: x + self.PADDING_SIZE]

            yolo8_detector = Yolo8Detector(self.onnx_model_local, _img)
            yolo8_detector.set_preprocess_func(self._finetune_preprocess)

            points = yolo8_detector.run()
            points = self.check_border(points)
            new_points.append(points[i] - self.PADDING_SIZE)
            new_corner_points[i, :] = [x, y]

        finetune_points = np.array(new_points) + new_corner_points
        finetune_points -= self.PADDING_SIZE
        self.finetune_corner_points = self._inv_points(M, finetune_points)

        # TODO 后续需要将四个角点变换成矩形

    @staticmethod
    def _finetune_preprocess(img):
        if img.ndim == 3: ei = cv2.equalizeHist(img[:, :, 0])
        else: ei = cv2.equalizeHist(img)

        ei = cv2.cvtColor(ei, cv2.COLOR_GRAY2RGB)

        return ei

    @staticmethod
    def _inv_points(mat, points):
        """

        Args:
            mat:
            points:

        Returns:

        """
        if mat.shape == (2, 3):
            _mat = np.eye(3)
            _mat[:2, :] = mat
        else:
            _mat = mat

        new_points = np.matrix(_mat).I @ np.concatenate(
            [points, np.ones((points.shape[0], 1))], axis = 1
        ).transpose(1, 0)

        return np.array(new_points)[:2, :].transpose()

    @staticmethod
    def calculate_rotation_angle(points):
        """计算左上角和右上角的线段与水平轴的夹角"""
        # 排序点，以获取 y 最小的两个点
        sorted_points = sorted(points, key = lambda p:p[1])

        # 取 y 最小的两个点
        y_min_points = sorted_points[:2]

        # 从 y 最小的两个点中选择 x 最小和 x 最大的点
        p1 = min(y_min_points, key = lambda p:p[0])  # x 最小的点
        p2 = max(y_min_points, key = lambda p:p[0])  # x 最大的点

        # 计算线段 p1p2 的角度
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        # print(dx, dy)  # 输出 dx 和 dy 以检查计算是否正确
        angle = np.arctan2(dy, dx)

        # 将角度转换为度制
        angle_degrees = np.degrees(angle)

        return angle_degrees

    @staticmethod
    def padding_border(img, size):
        return cv2.copyMakeBorder(img, size, size, size, size, cv2.BORDER_CONSTANT, value = (0,))

    @staticmethod
    def check_border(file: np.ndarray):
        """ Check array, default (left-up, left_down, right_down, right_up)

        Args:
            file:

        Returns:

        """
        if not isinstance(file, np.ndarray): return None
        assert file.shape == (4, 2), "Array shape error."

        file = file[np.argsort(np.mean(file, axis = 1)), :]
        if file[1, 0] > file[2, 0]:
            file = file[(0, 2, 1, 3), :]

        file = file[(0, 1, 3, 2), :]

        return file


def detect_chip(file_path: Union[str, np.ndarray],
                cfg: ChipParam,
                stain_type: TechType,
                actual_size: Tuple[int, int],
                is_debug: bool) -> Tuple[ChipBoxInfo, Dict[str, np.ndarray]]:
    """

    Args:
        file_path:
        cfg:
        stain_type: 染色类型 （ssDNA | DAPI | HE | IF）
        actual_size: 芯片原始标准大小 500nm/pixel尺度下 例S1芯片(19992, 19992)
        is_debug: is debug mode or not

    Returns:

    """
    debug_image_dic = {}
    cd = ChipDetector(cfg=cfg, stain_type=stain_type)
    cd.detect(file_path, actual_size=actual_size)
    info = {
        'LeftTop': cd.left_top, 'LeftBottom': cd.left_bottom,
        'RightTop': cd.right_top, 'RightBottom': cd.right_bottom,
        'ScaleX': cd.scale_x, 'ScaleY': cd.scale_y, 'Rotation': cd.rotation,
        'ChipSize': cd.chip_size, 'IsAvailable': cd.is_available
    }
    # if is_debug is False, the returned dic debug_image_dic is empty
    if is_debug:
        points = np.array([cd.left_top, cd.left_bottom, cd.right_bottom, cd.right_top])
        debug_image_dic = get_view_image(image=file_path,
                                         points=points,
                                         is_matrix=False,
                                         radius=50)
    return ChipBoxInfo(**info), debug_image_dic


def main():
    cfg = ChipParam(
        **{"stage1_weights_path":
            r"D:\hedongdong1\Workspace\01.chip_box_detect\algorithm_develop\old_model\chip_detect_obb8n_640_SD_202409_pytorch.onnx",
            "stage2_weights_path":
            r"D:\hedongdong1\Workspace\01.chip_box_detect\algorithm_develop\old_model\chip_detect_yolo8x_1024_SDH_stage2_202410_pytorch.onnx"})

    file_path = r"D:\hedongdong1\Workspace\01.chip_box_detect\show_interface\test_data\C04144G513_ssDNA_stitch.tif"
    info, debug_dic = detect_chip(file_path, cfg=cfg, stain_type=TechType.ssDNA, actual_size=(19992, 19992), is_debug=True)
    print(info.IsAvailable)


if __name__ == '__main__':
    # points = np.loadtxt(r"D:\02.data\temp\temp_cellbin2_test\trans_data_1\B03025E4\B03025E4_DAPI_stitch.txt")
    #
    # cd = ChipDetector(cfg = None, stain_type = "DAPI")
    # cd.set_corner_points(points)
    # cd.detect(r"D:\02.data\temp\temp_cellbin2_test\trans_data_1\A00792D3\label.tif", (19992, 19992))

    main()
