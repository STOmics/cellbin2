# -*- coding: utf-8 -*-
"""
🌟 Create Time  : 2024/9/26 9:39
🌟 Author  : CB🐂🐎 - lizepeng
🌟 File  : calibrate.py
🌟 Description  :
"""
import os
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
    offset_thr: int = Field(20, description="Threshold, a flag used to determine whether it has passed or not")


class Calibrate:
    """

        *Using FFT for image matching
        *The two modalities need to be the same or close, otherwise the calculation accuracy is not high
        *Can perform translation calculation and rotation scaling calculation

    """

    def __init__(
            self,
            src_image: Union[str, np.ndarray] = None,
            dst_image: Union[str, np.ndarray] = None,
            same_image: Union[str, np.ndarray] = None,
            output_path: str = '',
            method: int = 0,
            down_size: int = 4000
    ):
        """
        Initialize parameters, transform dst to src!!!

        Args:
            src_image: image path | array  Representing the target registration image
            dst_image: image path | array  Representing the image to be registered
            method: Calibration usage method  0 | 1
                * 0 Indicating that only translational calibration is used to obtain parameters that are only offset
                * 1 Perform affine transformation calibration to obtain parameters such as scale, rotate, and offset
            down_size: When calculating FFT, the longest edge of the image is scaled to this parameter

        """
        self.method = (0 if method == 0 else 1)

        self.src_image = self.parse_img(src_image)
        self.dst_image = self.parse_img(dst_image)

        self.same_image = self.parse_img(same_image)
        self.output_path = output_path

        self.down_size = down_size

    @staticmethod
    def parse_img(im):
        """

        Args:
            im:

        Returns:

        """
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

    @staticmethod
    def get_mass(image):
        """

        Args:
            image:

        Returns:

        """
        M = cv.moments(image)
        cx_cv = round(M['m10'] / M['m00'])
        cy_cv = round(M['m01'] / M['m00'])

        return np.array([cx_cv, cy_cv])

    def mass_align(self):
        """

        Returns:

        """
        src_mass, dst_mass = self.get_mass(self.src_image), self.get_mass(self.dst_image)
        offset = src_mass - dst_mass

        _di = cbimread(self.dst_image)

        _di = _di.trans_image(offset=offset, dst_size = self.src_image.shape)

        return _di.image

    def calibration(self):
        """
        *Scale and size the image uniformly
        *And perform calibration operations

        Returns:

        """
        # 一致性
        if self.method:
            norm_scale = min(self.src_image.shape) / max(self.dst_image.shape)
            self.dst_image = cv.resize(
                self.dst_image,
                (int(self.dst_image.shape[1] * norm_scale), int(self.dst_image.shape[0] * norm_scale))
            )

            self.dst_image = self.mass_align()
        else:
            self.src_image, self.dst_image = self._consistent_image(
                self.src_image, self.dst_image, 'same'
            )

        # 填充
        # pad_rate = 0.1
        # padding_size = int(max(self.src_image.shape) * pad_rate)
        # self.src_image = cv.copyMakeBorder(
        #     self.src_image, padding_size, padding_size, padding_size, padding_size,
        #     borderType = cv.BORDER_CONSTANT, value=0
        # )
        # self.dst_image = cv.copyMakeBorder(
        #     self.dst_image, padding_size, padding_size, padding_size, padding_size,
        #     borderType = cv.BORDER_CONSTANT, value=0
        # )

        down_scale = max(self.src_image.shape) / self.down_size

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

        # Analysis results
        offset = np.round(ret.get('tvec')[::-1] * down_scale)
        score = ret.get('success')
        scale = ret.get('scale', 1)
        rotate = ret.get('angle', 0)

        trans_info = {"score": score, "offset": offset, "scale": scale, "rotate": rotate}

        if self.same_image is not None:
            new_dst = cbimread(self.same_image)
        else:
            new_dst = cbimread(self.dst_image)
        new_dst = new_dst.trans_image(
            scale = float(scale),
            rotate = rotate,
        )

        _offset = (np.array(self.src_image.shape[:2]) - np.array(new_dst.shape[:2])) / 2
        _offset = _offset[::-1]

        result = new_dst.trans_image(
            offset = offset + _offset,
            dst_size = self.src_image.shape[:2]
        )

        # _, result = self._consistent_image(self.src_image, new_dst.image, 'same')
        if len(self.output_path) > 0:
            result.write(self.output_path)

        return result.image, trans_info


def multi_channel_align(
        cfg: CalibrationParam,
        fixed_image: str,
        moving_image: str,
        same_image: Union[str, np.ndarray] = None,
        output_path: str = '',
        method: int = 0
) -> CalibrationInfo:
    assert method in [0, 1]
    cal = Calibrate(fixed_image, moving_image, same_image, output_path, method=method)
    new_dst, trans_info = cal.calibration()
    x, y = trans_info['offset']
    d = math.sqrt(x * x + y * y)
    trans_info['pass_flag'] = d <= cfg.offset_thr and 1 or 0

    return CalibrationInfo(**trans_info)


def main(args):
    cfg = CalibrationParam()
    multi_channel_align(cfg, args.src_image, args.dst_image, args.same_image, args.output, method=args.method)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-src", "--src_image", action="store", dest="src_image", type=str, required=True,
                        help="Src image path.")
    parser.add_argument("-dst", "--dst_image", action="store", dest="dst_image", type=str, required=True,
                        help="Dst image path.")

    parser.add_argument("-same", "--same_image", action = "store", dest = "same_image", type = str, required = False,
                        default = '', help = "Same trans image path.")

    parser.add_argument("-m", "--method", action="store", dest="method", type=int, required=False, default=0,
                        help="Translation = 0 | Similarity = 1.")

    parser.add_argument("-o", "--output", action="store", dest="output", type=str, required=False, default = "",
                        help="Result output dir.")

    parser.set_defaults(func=main)
    (para, _) = parser.parse_known_args()
    para.func(para)
