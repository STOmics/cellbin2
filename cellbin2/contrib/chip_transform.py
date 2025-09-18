# -*- coding: utf-8 -*-
"""
🌟 Create Time  : 2025/6/19 16:10
🌟 Author  : CB🐂🐎 - lizepeng
🌟 File  : chip_transform.py
🌟 Description  : 
🌟 Key Words  :
"""
import cv2 as cv
import numpy as np

from typing import Union

from cellbin2.utils import clog
from cellbin2.image import cbimread, cbimwrite
from cellbin2.contrib.calibration import Calibrate
from cellbin2.image import CBImage


def _to_color(
        image: np.ndarray,
        color_space: str = "HSV"
):
    if image.ndim == 3:
        if color_space == "HSV":
            _image = cv.cvtColor(image, cv.COLOR_RGB2HSV)[:, :, 1]
        elif color_space == "HLS":
            _image = cv.cvtColor(image, cv.COLOR_RGB2HLS)[:, :, 1]
            _image = 255 - _image
            _image = _image.astype(np.uint8)
        else:
            raise ValueError("Color space must be HSV or HLS")
    else:
        _image = image

    return cbimread(_image)


def chip_transform(
        fixed_image: Union[np.ndarray, str],
        moving_image: Union[np.ndarray, str],
        output_path: str = None,
        color_space: str = "HSV",
        scale: list = None,
        method: int = 1
) -> CBImage:
    """

    Args:
        fixed_image:
        moving_image:
        output_path:
        color_space: "HSV" or "HLS"
        scale: [fixed_image microscope magnification, moving_image microscope magnification], usually [2, 1]
        method: int
            0: 原图配准
            1:

    Returns:

    """
    if scale is None: scale = [2, 1]

    fixed_image = cbimread(fixed_image)
    moving_image = cbimread(moving_image)
    moving_image = moving_image.trans_image(flip_lr = True)

    fi, mi = map(
        lambda x: _to_color(x, color_space),
        [fixed_image.image, moving_image.image]
    )

    scale.append(scale[1])
    fi, mi, moving_image = map(
        lambda x: x[0].resize_image(1 / x[1]),
        zip([fi, mi, moving_image], scale)
    )

    clog.info("Chip transform start.")
    new_mi, trans_info = Calibrate(
        src_image = fi.image,
        dst_image = mi.image,
        same_image = moving_image.image,
        method = method
    ).calibration()

    if output_path != None:
        clog.info("Chip transform -- write image...")
        _new_mi = cbimread(new_mi)
        _new_mi = _new_mi.resize_image(2)
        cbimwrite(output_path, new_mi)
        cbimwrite(output_path.replace("_regist", "_20X_regist"), _new_mi)

    return CBImage(new_mi) 

if __name__ == "__main__":
    img = chip_transform(
        fixed_image = r"D:\02.data\temp\temp\src_image\LXX_2_HE_d2.tif",
        moving_image = r"D:\02.data\temp\temp\src_image\lxx_宫颈癌2\test_v3.tif",
        output_path = r"D:\02.data\temp\temp\src_image\lxx_宫颈癌2\regis_v3.tif",
    )
