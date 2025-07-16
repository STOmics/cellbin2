# -*- coding: utf-8 -*-
"""
🌟 Create Time  : 2025/5/28 14:39
🌟 Author  : CB🐂🐎 - lizepeng
🌟 File  : scan_method.py
🌟 Description  : 
🌟 Key Words  :
"""
import os
import re
import enum

import numpy as np
import tifffile as tif

from typing import Union


class Scanning(enum.Enum):
    """ stitching_type:
        # row and col & col and row
        RaC = 0
        CaR = 1

        # coordinate
        Coordinate = 2

        # row by row
        RbR_RD = 31
        RbR_LD = 32
        RbR_RU = 33
        RbR_LU = 34

        # col by col
        CbC_DR = 41
        CbC_DL = 42
        CbC_UR = 43
        CbC_UL = 44

        # snake by row
        SbR_RD = 51
        SbR_LD = 52
        SbR_RU = 53
        SbR_LU = 54

        # snake by col
        SbC_DR = 61
        SbC_DL = 62
        SbC_UR = 63
        SbC_UL = 64
    """
    # row and col & col and row
    RaC = 0
    CaR = 1

    # coordinate
    Coordinate = 2

    # row by row
    RbR_RD = 31
    RbR_LD = 32
    RbR_RU = 33
    RbR_LU = 34

    # col by col
    CbC_DR = 41
    CbC_DL = 42
    CbC_UR = 43
    CbC_UL = 44

    # snake by row
    SbR_RD = 51
    SbR_LD = 52
    SbR_RU = 53
    SbR_LU = 54

    # snake by col
    SbC_DR = 61
    SbC_DL = 62
    SbC_UR = 63
    SbC_UL = 64


def search_files(file_path, exts):
    files_ = list()
    for root, dirs, files in os.walk(file_path):
        if len(files) == 0:
            continue
        for f in files:
            fn, ext = os.path.splitext(f)
            if ext in exts:
                files_.append(os.path.join(root, f))

    return files_


def trans_path2class(images_dict, **kwargs):
    """
    Args:
        images_dict:
        **kwargs:
            sdt: stereo data type -- 'dolphin' | ''
    Returns:
    """
    flip = kwargs.get('flip')
    new_images_dict = dict()
    for k, v in images_dict.items():
        if flip == 1:
            new_images_dict[k] = ImageBase(v)
        elif flip == 2:
            new_images_dict[k] = ImageBase(v, flip_ud=True)
        elif flip == 3:
            new_images_dict[k] = ImageBase(v, flip_lr=True)

    return new_images_dict


class ScanMethod:
    def __init__(self, scan_method: Union[int, Scanning] = Scanning.RaC):
        if isinstance(scan_method, int):
            self.scan_method = Scanning(scan_method)
        else:
            self.scan_method = scan_method

    def to_default(
            self,
            images_path: list,
            rows: int,
            cols: int,
            name_pattern: str = '*_{xxx}_{xxx}_*.tif',
            **kwargs
    ) -> dict:
        """
        Returns: return RaC method
        """
        images_dict = self.get_images_index(images_path, name_pattern)

        if self.scan_method == Scanning.RaC:
            pass

        elif self.scan_method == Scanning.CaR:
            images_dict = self.car_trans(images_dict)

        elif self.scan_method == Scanning.Coordinate:
            pass

        elif 30 < self.scan_method.value < 40:
            images_dict = self.rbr_trans(images_dict, rows, cols)

        elif 40 < self.scan_method.value < 50:
            images_dict = self.cbc_trans(images_dict, rows, cols)

        elif 50 < self.scan_method.value < 60:
            images_dict = self.sbr_trans(images_dict, rows, cols)

        elif 60 < self.scan_method.value < 70:
            images_dict = self.sbc_trans(images_dict, rows, cols)

        images_dict = trans_path2class(images_dict, **kwargs)

        return images_dict

    @staticmethod
    def car_trans(images_dict: dict) -> dict:
        new_images_dict = dict()
        for k, v in images_dict.items():
            _k = k.split('_')[::-1]
            new_images_dict['_'.join(_k)] = v
        return new_images_dict

    def rbr_trans(self, images_dict: dict, rows: int, cols: int) -> dict:
        pass

    def cbc_trans(self, images_dict: dict, rows: int, cols: int) -> dict:
        pass

    def sbr_trans(self, images_dict: dict, rows: int, cols: int) -> dict:
        new_images_dict = dict()
        for k, v in images_dict.items():
            r = int(k) // cols
            c = int(k) % cols

            if r % 2 == 1:
                c = cols - c - 1

            if self.scan_method == Scanning.SbR_RD:
                pass
            elif self.scan_method == Scanning.SbR_LD:
                c = cols - c - 1
            elif self.scan_method == Scanning.SbR_RU:
                r = rows - r - 1
            elif self.scan_method == Scanning.SbR_LU:
                r = rows - r - 1
                c = cols - c - 1

            new_images_dict[f"{r:04}_{c:04}"] = v

        return new_images_dict

    def sbc_trans(self, images_dict: dict, rows: int, cols: int) -> dict:
        pass

    @staticmethod
    def _get_pattern(name_pattern: str):
        ind_list = list()
        ind_flag = False

        ind = 0
        for i in name_pattern:
            if i == '{':
                ind_flag = True
            if i == 'x' and ind_flag:
                ind += 1
            if i == '}':
                ind_flag = False
                ind_list.append(ind)
                ind = 0

        re_name = name_pattern
        for i in ind_list:
            _i = '{' + 'x' * i + '}'
            re_name = re_name.replace(_i, '\d{' + str(i) + '}', 1)
        re_name = re_name.split("*")[1]

        return re_name, ind_list

    def get_images_index(
            self,
            images_path: list,
            name_pattern: str
    ) -> dict:
        images_dict = dict()
        re_name, ind_list = self._get_pattern(name_pattern)

        for ip in images_path:
            if os.path.isdir(ip):
                continue
            pat = re.compile(re_name)
            res = pat.search(os.path.basename(ip))
            if res is None:
                raise AttributeError(f"{os.path.basename(ip)} is not match {re_name}")
            res = res.group(0)

            image_ind = list()
            for i in ind_list:
                _pat = re.compile('\d{' + str(i) + '}')
                _ind = _pat.search(res).group(0)
                image_ind.append(_ind)
                res = res.replace(_ind, '', 1)

            if len(image_ind) > 1:
                name = '_'.join(image_ind)
            else:
                name = image_ind[0]

            images_dict[name] = ip

        return images_dict


class ImageBase:
    """

    """
    def __init__(self, image: Union[str, np.ndarray], **kwargs):
        self.image = image

        self.flip_ud = kwargs.get('flip_ud', False)
        self.flip_lr = kwargs.get('flip_lr', False)

        self.rot = kwargs.get('rot', 0)

        self.to_gray = kwargs.get('to_gray', False)

    def get_image(self):
        if isinstance(self.image, str):
            _image = tif.imread(self.image)
        elif isinstance(self.image, np.ndarray):
            _image = self.image.copy()
        else:
            raise ValueError('Image type error.')

        if self.flip_ud:
            _image = _image[::-1, :]

        if self.flip_lr:
            _image = _image[:, ::-1]

        if self.rot != 0:
            pass

        if self.to_gray:
            pass

        return _image


if __name__ == '__main__':
    data_path = r"D:\02.data\chengjinghao\anno\B03204C211\ssDNA\B03204C211"
    imgs = search_files(data_path, exts=['.tif'])

    sm = ScanMethod(Scanning.SbR_RD)
    # imd = sm.to_default(
    #     images_path=imgs,
    #     rows=7,
    #     cols=5,
    #     name_pattern='*_{xxxx}_{xxxx}_*.tif',
    # )

    imd = sm.to_default(
        images_path=imgs,
        rows=7,
        cols=5,
        name_pattern='*s{xx}*',
    )
    print(1)
