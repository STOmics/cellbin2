import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import cv2
import os
import tqdm
import multiprocessing as mp
from collections import Counter
from pydantic import BaseModel, Field

from cellbin2.image import cbimread, cbimwrite
from cellbin2.dnn.detector.onnx_yolo5_obb import OBB5Detector
from cellbin2.utils import clog
from cellbin2.utils.common import TechType
from cellbin2.image.augmentation import f_rgb2gray, dapi_enhance, he_enhance
from cellbin2.image import CBImage
from cellbin2.image.wsi_split import SplitWSI
from cellbin2.contrib.param import TrackPointsInfo
from cellbin2.contrib.base_module import BaseModule


pt_enhance_method = {
    TechType.ssDNA.value: dapi_enhance,
    TechType.DAPI.value: dapi_enhance,
    TechType.HE.value: he_enhance
}


class TrackPointsParam(BaseModel, BaseModule):
    detect_channel: int = Field(
        -1, description="If the input image is 3-channel, indicate the detection channel. "
                        "Otherwise, the program will automatically switch to a single channel diagram"
    )
    first_level_thr: int = Field(5, description="[th, th2) -> track_point_score = 1")
    second_level_thr: int = Field(20, description="[th2, inf) -> track_point_score = 2")
    good_thresh: int = Field(5, description="Those exceeding the threshold are counted as good_fov")
    process: int = Field(1, description="Point detection inference process number setting")
    # Global template settings only require 0.5 or more points
    conf_filter: float = Field(
        0.5, description="Filter the confidence level of the detection results based on this value. "
                         "If the value is less than 0, no filtering will be performed")
    high_freq_angle_thr: float = Field(0.05, )
    DEFAULT_STAIN_TYPE: TechType = TechType.ssDNA
    SUPPORTED_STAIN_TYPE: Tuple[TechType, TechType, TechType] = (TechType.ssDNA, TechType.DAPI, TechType.HE)
    ssDNA_weights_path: str = Field(
        "points_detect_yolov5obb_SSDNA_20220513_pytorch.onnx",
        description="The weight file name corresponding to the ssDNA staining image")
    DAPI_weights_path: str = Field(
        "points_detect_yolov5obb_SSDNA_20220513_pytorch.onnx",
        description="The weight file name corresponding to the DAPI staining graph")
    HE_weights_path: str = Field(
        "points_detect_yolov5obb_SSDNA_20220513_pytorch.onnx",
        description="The weight file name corresponding to the HE staining graph")
    IF_weights_path: str = Field(
        "points_detect_yolov5obb_SSDNA_20220513_pytorch.onnx",
        description="The weight file name corresponding to the IF staining graph")
    GPU: int = Field(0, description="GPU number used for inference")
    num_threads: int = Field(0, description="The number of threads used for inference")

    # def get_weights_path(self, stain_type: TechType):
    #     if stain_type == TechType.ssDNA: p = self.ssDNA_weights_path
    #     elif stain_type == TechType.DAPI: p = self.DAPI_weights_path
    #     elif stain_type == TechType.HE: p = self.HE_weights_path
    #     elif stain_type == TechType.IF: p = self.IF_weights_path
    #     else: p = None
    #
    #     return p


def no_enhance(img_obj):
    return img_obj.image


def pts_on_img(img, pts, radius=1, color=(0, 255, 255), thickness=3):
    for pt in pts:
        pos = (int(pt[0]), int(pt[1]))
        cv2.circle(img, pos, 1, color, 2)
        cv2.circle(img, pos, radius, color, thickness)
    return img


def divergence(shape):
    rows, cols = shape
    diver_origin = np.zeros((rows, cols, 2))
    rows_arr = np.arange(1, rows + 1)
    cols_arr = np.arange(1, cols + 1)
    for col in range(cols):
        for row in range(rows):
            if row < rows // 2:
                r_value = -rows_arr[:rows // 2 - row].sum()
            else:
                if rows // 2 == 0:
                    r_value = rows_arr[:row - rows // 2].sum()
                else:
                    r_value = rows_arr[:row - (rows - 1) // 2].sum()
            if col < cols // 2:
                c_value = -cols_arr[:cols // 2 - col].sum()
            else:
                if cols // 2 == 0:
                    c_value = cols_arr[:col - cols // 2].sum()
                else:
                    c_value = cols_arr[:col - (cols - 1) // 2].sum()
            diver_origin[row, col] = np.array([r_value, c_value])

    diver_map = np.abs(diver_origin[:, :, 0]) + np.abs(diver_origin[:, :, 1])
    return diver_map


def save_result_on_image(enhance_func, img_obj, cp, save_dir, key):
    img: np.ndarray = enhance_func(img_obj)
    img = pts_on_img(img, cp, radius=20)
    save_path = os.path.join(save_dir, key + '_result' + '.tif')
    cbimwrite(
        output_path=save_path,
        files=img,
        compression=True
    )


class TrackPointQC(object):
    def __init__(
            self,
            cfg,
            stain_type: TechType,
    ):
        """
        This class is used to do track point detection (using deep learning object detection algo).
        Will also do track point quality control based on the distribution among fovs, counts of
        detections and the confidence score of prediction


        Track Eval:

        is used to evaluate the track points result from object detection algo.
        Considering:
            1. track pts count of each fov
            2. track pts dets confidence score of each fov
        Args:
            cfg (TrackPtParam): configuration parameter
            stain_type:

        Result:
            self.track_result: track point detection result by deep learning method
                - {'row_col': [[pt_x, pt_y, conf], angle]}
                - no dets: {}

            self.fov_mask: score for each fov
                - numpy 2d array (success)
                - np.array([]) (fail)

            self.fovs_score (dict): score for all fovs in order (fov_score > 0)
                - {'row_col': fov_score}
                - {} if no fov_score is greater than 0

            self.score: track_pts eval score, score interval is [0, 1], higher -> better
                - float
        Examples:
            >>> weight_path = "/media/Data/dzh/weights/points_detect_yolov5obb_SSDNA_20220513_pytorch.onnx"
            >>> models_config = {'ssDNA_weights_path': weight_path, 'dapi_weights_path': weight_path,  'he_weights_path': weight_path}
            >>> stain_type = TechType.ssDNA
            >>> cfg = TrackPtParam(**models_config)
            >>> tp_qc = TrackPointQC(cfg=cfg, stain_type=stain_type)

        """
        if stain_type not in cfg.SUPPORTED_STAIN_TYPE:
            clog.info(f"Track detect only support {[i.name for i in cfg.SUPPORTED_STAIN_TYPE]}, fail to initialize")
            return
        # init
        self.cfg: TrackPointsParam = cfg
        self.stain_type = stain_type
        self.ci = OBB5Detector(gpu=cfg.GPU, num_threads=cfg.num_threads)
        self.ci.load_model(
            weight_path=self.cfg.get_weights_path(self.stain_type)
        )

        #
        # self.params: Optional[TrackPtParams] = None
        # self.track_conf: float = track_conf

        # result
        self.track_result: Dict[str, List[List[List[float]], Optional[float]]] = dict()  # {"0_0": [[[x, y, confidence], ...], angle], }
        self.score: float = -1.0
        self.fov_mask: np.ndarray = np.array([])
        # self.fovs_order: list = []
        self.good_fov_count: int = -1
        self.most_freq_angle: Tuple[float, int] = (-1.0, -1)  # angle, count

        # control
        self.rgb_warning = False

    def img_read(self, val, buffer) -> CBImage:
        if isinstance(val, str):
            # small image process
            img: CBImage = cbimread(files=val)
        elif isinstance(val, list) or isinstance(val, tuple):
            # big image process
            buffer: CBImage = cbimread(files=buffer)
            img: CBImage = buffer.crop_image(border=val)
        else:
            raise NotImplementedError
        # That's all for the small pictures here
        if self.stain_type != TechType.HE and img.channel == 3:
            if self.cfg.detect_channel != -1:
                img: CBImage = img.get_channel(channel=self.cfg.detect_channel)  # 2d array
            else:
                if not self.rgb_warning:
                    clog.warning(
                        f"Your input is not single channel image and you have not provided the detect channel."
                        f"The program will convert the image to single channel image automatically."
                        f"Cause track point detect only support single channel image."
                    )
                img: CBImage = cbimread(f_rgb2gray(img.image))
                self.rgb_warning = True
        return img

    def track_detect(
            self,
            img_dict: dict,
            buffer: Optional[np.ndarray] = None,
            save_dir: Optional[str] = None
    ):
        """
        This function will do track detection using object detection (deep learning) algo.
        self.track_result will be empty if no detections

        Args:
            img_dict (dict):
                small image：{'row_col': img_path}
                big image：{'row_col': [y_begin, y_end, x_begin, x_end]}
            buffer: np.ndarray - big image
            save_dir: str save path
        """
        self.track_result = dict()
        if not hasattr(self, 'ci'):
            clog.info(f"{self.__class__.__name__} failed to initialize, can not detect")
            return
        enhance_func = pt_enhance_method.get(self.stain_type.value, self.cfg.DEFAULT_STAIN_TYPE.value)
        clog.info(f"Track points detect enhance method: {enhance_func.__name__}")
        self.ci.set_func(enhance_func)
        if self.cfg.process <= 1:
            for key, val in tqdm.tqdm(
                    img_dict.items(),
                    file=clog.tqdm_out,
                    mininterval=10,
                    desc='Track points detect'
            ):
                img_obj: CBImage = self.img_read(val, buffer)
                cp, angle = self.ci.predict(img_obj)
                if save_dir is not None:
                    # debug mode
                    save_result_on_image(enhance_func, img_obj, cp, save_dir, key)
                if angle is None or len(cp) == 0:
                    continue
                self.track_result[key] = [cp, angle]
        else:
            processes = []
            pool = mp.Pool(processes=self.cfg.process)
            clog.info(f"Track point detection using {self.cfg.process} processes")
            for key, val in tqdm.tqdm(
                    img_dict.items(),
                    file=clog.tqdm_out,
                    mininterval=10,
                    desc='Track points detect'
            ):
                img_obj = self.img_read(val, buffer)
                sub_process = pool.apply_async(self.ci.predict, args=(img_obj,))
                processes.append([key, sub_process, img_obj])
            pool.close()
            pool.join()
            for key, p, img_obj in processes:
                cp, angle = p.get()
                if save_dir is not None:
                    # debug mode
                    save_result_on_image(enhance_func, img_obj, cp, save_dir, key)
                if angle is None or len(cp) == 0:
                    continue
                self.track_result[key] = [cp, angle]

        # TODO -- HE fixed 2025.01.20
        if self.stain_type == TechType.HE:
            self.cfg.conf_filter = 0

        if self.cfg.conf_filter > 0:
            self.track_filter(track_conf=self.cfg.conf_filter)
        else:
            clog.info(f"conf_filer <= 0, skip track filter")
        self.track_eval()

    def track_eval(self, ):
        """
        This func will evaluate track cross quality for fovs

        Returns:
            self.score: fov track cross score
            self.fov_mask: 2d array, contain score for each fov
            self.fovs_order: rank all fovs based on score

        """
        self.score = -1
        self.fov_mask = np.array([])
        # self.fovs_order = []
        self.good_fov_count = -1

        if len(self.track_result) == 0:
            return

        max_row, max_col = -1, -1
        for key in self.track_result.keys():
            splits = key.split('_')
            row, col = int(splits[0]), int(splits[1])
            max_row = max(row, max_row)
            max_col = max(col, max_col)

        pt_score_1 = 1
        pt_score_2 = 2
        # pt_count_mask = np.zeros((max_row + 1, max_col + 1))  # val: count of cp * mean(conf)
        conf_mask = np.zeros((max_row + 1, max_col + 1))  # no dets: 0
        val_pt_mask = np.zeros_like(conf_mask)
        max_pt_mask = np.ones_like(conf_mask) * pt_score_2
        fovs_name = np.empty_like(conf_mask, dtype='object')
        good_fov = 0
        all_angles = []
        for key, val in self.track_result.items():
            splits = key.split('_')
            row, col = int(splits[0]), int(splits[1])
            cps, angle = val
            if len(cps) >= self.cfg.good_thresh:
                good_fov += 1
                all_angles.append(angle)
            cps_arr = np.array(cps)
            cur_counts = len(cps_arr)
            conf_mean = cps_arr.mean(axis=0)[-1]
            # pt_count_mask[row, col] = cur_counts
            if self.cfg.first_level_thr <= cur_counts < self.cfg.second_level_thr:
                val_pt_mask[row, col] = pt_score_1
            elif cur_counts >= self.cfg.second_level_thr:
                val_pt_mask[row, col] = pt_score_2
            conf_mask[row, col] = conf_mean
            fovs_name[row, col] = key
        if len(all_angles) != 0:
            occurence_count = Counter(all_angles)
            most_common_one = occurence_count.most_common(1)[0]
            self.most_freq_angle = most_common_one
        # diver_map = divergence(val_pt_mask.shape)
        # template_mask = diver_map * max_pt_mask
        val_pt_mask_norm = val_pt_mask / max_pt_mask
        result_mask = val_pt_mask_norm * conf_mask
        score = result_mask.sum()

        self.score = score
        self.fov_mask = result_mask
        self.good_fov_count = good_fov

        # fovs_score = {}
        # for row, col in np.ndindex(result_mask.shape):
        #     cur_score = result_mask[row, col]
        #     if cur_score > 0:
        #         cur_name = fovs_name[row, col]
        #         fovs_score[cur_name] = cur_score
        #
        # if len(fovs_score) != 0:
        #     self.fovs_order = [k for k, v in sorted(fovs_score.items(), key=lambda item: item[1], reverse=True)]

    def track_filter(self, angle_dif=1, track_conf=0.5):
        """
        Clear abnormal track inspection points
        Args:
            angle_dif: Maximum angle difference from mode
            track_conf: Minimum confidence level of track points
        """
        clog.info(f"Filtering track point result based on confidence > {track_conf}")
        new_result = dict()
        for k, v in self.track_result.items():
            # if np.abs(v[1] - self.most_freq_angle[0]) < angle_dif:
            new_track = list()
            for i in v[0]:
                if i[2] > track_conf:
                    new_track.append(i)

            v[0] = new_track
            if len(v[0]) > 0:
                new_result[k] = v

        self.track_result = new_result

    def pts_loc_to_global(self, img_dict, pts):
        if len(pts) == 0:
            return pts
        new_result = {}
        for key, value in pts.items():
            y_begin, y_end, x_begin, x_end = img_dict[key]
            cross_points = np.array(value[0])
            cross_points[:, :2] += np.array([x_begin, y_begin])
            cross_points = cross_points.tolist()
            new_result[key] = [cross_points, value[1]]
        return new_result


def large_split(
        large_path: str,
        save_dir: Union[None, str],
        h: int,
        w: int,
        overlap: Union[float, int]
):
    img_path = os.path.join(large_path)
    mosaic = cbimread(img_path, only_np=True)
    wsi = SplitWSI(
        img=mosaic,
        win_shape=(h, w),
        overlap=overlap,
        batch_size=1,
        need_fun_ret=False,
        need_combine_ret=False
    )
    _box_lst, _fun_ret, _dst = wsi.f_split2run()
    img_dict = {}
    for y_begin, y_end, x_begin, x_end in _box_lst:
        row = y_begin // h
        col = x_begin // w
        img_dict[f"{row}_{col}"] = [y_begin, y_end, x_begin, x_end]
        if save_dir is not None:
            fov_ = mosaic[y_begin: y_end, x_begin: x_end]
            out_p = os.path.join(save_dir, f"{row}_{col}.tif")
            cbimwrite(
                output_path=out_p,
                files=fov_,
                compression=True
            )
    return img_dict, mosaic


def image_location(src_fovs: dict, rows: int, cols: int):
    """Large image coordinate generation function"""
    location = np.zeros((rows, cols, 2), dtype=int)
    for k, v in src_fovs.items():
        row, col = [int(i) for i in k.split('_')]
        location[row, col] = [v[2], v[0]]

    return location


def run_detect(
        img_file: str,
        cfg: TrackPointsParam,
        stain_type,
        h: int,
        w: int,
        overlap: Union[float, int],
        save_dir=None,
):
    # The large image needs to be cropped
    img_dict, buffer = large_split(
        large_path=img_file,
        save_dir=save_dir,
        h=h,
        w=w,
        overlap=overlap
    )

    tp_qc = TrackPointQC(cfg, stain_type)
    tp_qc.track_detect(
        img_dict=img_dict,
        buffer=buffer,
        save_dir=save_dir
    )
    track_result = tp_qc.pts_loc_to_global(
        img_dict=img_dict,
        pts=tp_qc.track_result
    )
    if save_dir is not None:
        only_cp = []
        for i in track_result.values():
            only_cp.extend(i[0])
        save_result_on_image(
            enhance_func=tp_qc.ci.img_func,
            img_obj=cbimread(buffer),
            cp=only_cp,
            save_dir=save_dir,
            key='stitch'
        )

    # location = np.zeros((self._rows, self._cols, 2), dtype=int)
    # for k, v in self._src_fovs.items():
    #     row, col = [int(i) for i in k.split('_')]
    #     location[row, col] = [v[2], v[0]]

    track_points = {}
    for k, v in track_result.items():
        pts1 = np.array(v[0])[:, :2]
        pts = np.zeros((pts1.shape[0], 4))
        pts[:, :2] = pts1
        track_points[k] = pts

    info = {
        'track_points': track_points,
        'good_fov_count': tp_qc.good_fov_count,
        'score': tp_qc.score,
        'fov_location': image_location(src_fovs=img_dict,
                                       rows=math.ceil(buffer.shape[0] / h),
                                       cols=math.ceil(buffer.shape[1] / w))
    }

    return TrackPointsInfo(**info)


if __name__ == '__main__':
    # constant
    h, w = 2000, 2000
    overlap = 0.0

    # big img
    weight_path = r"E:\03.users\liuhuanlin\01.data\cellbin2\weights\points_detect_yolov5obb_SSDNA_20220513_pytorch.onnx"
    img_file = r"E:\03.users\liuhuanlin\01.data\cellbin2\stitch/SS200000789TL_C6_S_DAPI.tif"
    show_result = r"E:\03.users\liuhuanlin\01.data\cellbin2\output"

    stain_type = TechType.ssDNA
    models_config = {'ssDNA_weights_path': weight_path, 'DAPI_weights_path': weight_path,
                     'HE_weights_path': weight_path}
    cfg = TrackPointsParam(**models_config)

    run_detect(
        img_file=img_file,
        cfg=cfg,
        stain_type=stain_type,
        h=h,
        w=w,
        overlap=overlap,
        save_dir=show_result,  # no need
    )
