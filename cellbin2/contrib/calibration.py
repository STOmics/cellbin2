# -*- coding: utf-8 -*-
"""
🌟 Create Time  : 2024/9/26 9:39
🌟 Author  : CB🐂🐎 - lizepeng
🌟 File  : calibrate.py
🌟 Description  :
"""
import math
import imreg_dft

import cv2 as cv
import numpy as np
import tifffile as tif

from typing import Tuple, Union, List
from pydantic import BaseModel, Field

from cellbin2.contrib.param import CalibrationInfo
from cellbin2.image import cbimread


class CalibrationParam(BaseModel):
    offset_thr: int = Field(20, description="阈值，用于判断是否通过的标志位")


class Calibrate:
    """

        * 使用FFT进行图像间的匹配
        * 两图模态需要相同或接近，否则计算准确性不高
        * 可进行平移计算及旋转缩放计算

    """

    def __init__(
            self,
            src_image: Union[str, np.ndarray] = None,
            dst_image: Union[str, np.ndarray] = None,
            method: int = 0,
            down_size: int = 4000
    ):
        """
        初始化参数，变换 dst 到 src ！！！

        Args:
            src_image: image path | array  表示目标配准图像
            dst_image: image path | array  表示待配准图像
            method: 校准使用方法  0 | 1
                * 0 表示只做平移校准 求得参数仅为 offset
                * 1 表示做仿射变换校准 求得参数为 scale, rotate, offset
            down_size: 计算FFT时，图像最长边缩放至该参数

        """
        self.method = (0 if method == 0 else 1)

        self.src_image = self.parse_img(src_image)
        self.dst_image = self.parse_img(dst_image)

        self.down_size = down_size

    @staticmethod
    def parse_img(im):
        """

        Args:
            im:

        Returns:

        """
        # TODO 可接其他io方式
        if im is None: return

        if isinstance(im, str):
            _im = tif.imread(im)
        elif isinstance(im, np.ndarray):
            _im = im
        else:
            raise ValueError("Image data parsing error.")

        return _im

    @staticmethod
    def _consistent_image(im0: np.ndarray, im1: np.ndarray, method="max"):
        """

        Args:
            im0:
            im1:
            method:
                min:
                max:
                scale:

        Returns:

        """
        if im0.shape == im1.shape:
            return im0, im1

        _shape = np.array([im0.shape, im1.shape])
        if method == "min":
            new_shape = np.min(_shape, axis=0)
            _im0, _im1 = map(lambda x: x[:new_shape[0], :new_shape[1]], (im0, im1))

        elif method == "max":
            new_shape = np.max(_shape, axis=0)
            _im0 = cv.copyMakeBorder(im0, 0, int(new_shape[0] - im0.shape[0]),
                                     0, int(new_shape[1] - im0.shape[1]),
                                     cv.BORDER_CONSTANT, value=0)
            _im1 = cv.copyMakeBorder(im1, 0, int(new_shape[0] - im1.shape[0]),
                                     0, int(new_shape[1] - im1.shape[1]),
                                     cv.BORDER_CONSTANT, value=0)
        elif method == "same":
            _im0 = im0
            _im1 = np.zeros_like(im0, dtype=im0.dtype)
            cx, cy = int(_im1.shape[1] / 2), int(_im1.shape[0] / 2)

            if im1.shape[0] <= im0.shape[0]:
                _h = im1.shape[0]
            else:
                _h = im0.shape[0]

            if im1.shape[1] <= im0.shape[1]:
                _w = im1.shape[1]
            else:
                _w = im0.shape[1]

            _im1[cy - int(_h / 2): cy + _h - int(_h / 2), cx - int(_w / 2): cx + _w - int(_w / 2)] = \
                im1[int(im1.shape[0] / 2) - int(_h / 2): int(im1.shape[0] / 2) + _h - int(_h / 2),
                    int(im1.shape[1] / 2) - int(_w / 2): int(im1.shape[1] / 2) + _w - int(_w / 2)]

        return _im0, _im1

    @staticmethod
    def resize_image(image, size: Union[int, float, Tuple, List, np.ndarray]):
        """

        Args:
            image:
            size: (h, w)

        Returns:

        """
        if isinstance(size, (float, int)):
            src = cv.resize(image, [round(image.shape[1] * size), round(image.shape[0] * size)])
        else:
            src = cv.resize(image, [size[1], size[0]])
        return src

    @staticmethod
    def trans_by_mat(im, m, shape):
        """

        Args:
            im:
            m:
            shape: h, w

        Returns:

        """
        result = cv.warpPerspective(im, m, (shape[1], shape[0]))
        return result

    def set_src(self, im):
        self.src_image = self.parse_img(im)

    def set_dst(self, im):
        self.dst_image = self.parse_img(im)

    def calibration(self):
        """
        * 对图像进行缩放、尺寸统一处理
        * 并进行校准操作

        Returns:

        """
        down_scale = max(self.src_image.shape) / self.down_size

        self.src_image, self.dst_image = self._consistent_image(
            self.src_image, self.dst_image, 'same'
        )

        src_img = self.resize_image(
            self.src_image, 1 / down_scale
        )
        dst_img = self.resize_image(
            self.dst_image, 1 / down_scale
        )

        if self.method == 0:
            ret = imreg_dft.translation(src_img, dst_img)
        else:
            ret = imreg_dft.similarity(src_img, dst_img)

        # 解析结果
        offset = np.round(ret.get('tvec')[::-1] * down_scale)
        score = ret.get('success')
        scale = ret.get('scale', 1)
        rotate = ret.get('angle', 0)

        # trans_mat[:2, 2] = offset
        trans_info = {"score": score, "offset": offset, "scale": scale, "rotate": rotate}
        # new_dst = imreg_dft.transform_img(img=self.dst_image, scale=scale, angle=rotate, tvec=offset[::-1])
        new_dst = cbimread(self.dst_image)
        new_dst = new_dst.trans_image(
            scale = float(scale),
            rotate = rotate,
        )
        new_dst = new_dst.trans_image(
            offset = offset,
            dst_size = new_dst.shape
        )

        _, result = self._consistent_image(self.src_image, new_dst.image, 'same')

        return result, trans_info


def multi_channel_align(
        cfg: CalibrationParam,
        fixed_image: str,
        moving_image: str,
        method: int = 0
) -> CalibrationInfo:
    assert method in [0, 1]
    cal = Calibrate(fixed_image, moving_image, method=method)
    new_dst, trans_info = cal.calibration()
    x, y = trans_info['offset']
    d = math.sqrt(x * x + y * y)
    trans_info['pass_flag'] = d <= cfg.offset_thr and 1 or 0

    return CalibrationInfo(**trans_info)


def main(args):
    cfg = CalibrationParam()
    multi_channel_align(cfg, args.src_image, args.dst_image, method=args.method)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-src", "--src_image", action="store", dest="src_image", type=str, required=True,
                        help="Src image path.")
    parser.add_argument("-dst", "--dst_image", action="store", dest="dst_image", type=str, required=True,
                        help="Dst image path.")
    parser.add_argument("-m", "--method", action="store", dest="method", type=int, required=False, default=0,
                        help="Translation = 0 | Similarity = 1.")

    parser.set_defaults(func=main)
    (para, _) = parser.parse_known_args()
    para.func(para)
