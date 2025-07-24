# -*- coding: utf-8 -*-
"""
🌟 Create Time  : 2025/5/28 17:22
🌟 Author  : CB🐂🐎 - lizepeng
🌟 File  : mfws.py
🌟 Description  : 
🌟 Key Words  :
"""
import os
import glog
import argparse

import numpy as np
import tifffile as tif

from glob import glob
from typing import Union
from prettytable import PrettyTable


from modules.scan_method import ScanMethod, Scanning
from modules.stitching import Stitching
version = '0.1'


def _stitch_info_print(**kwargs):

    keys = list(kwargs.keys())
    nk = list()
    for k in keys:
        _k = list(
            map(
                lambda x: x.capitalize(),
                k.split("_")
            )
        )
        _k = " ".join(_k)
        nk.append(_k)

    pt = PrettyTable(nk)

    pt.add_row(list(kwargs.values()))

    glog.info(f"Basic mfws config info as, \n{pt}")


def stitching(
        image_path: str = '',
        rows: int = None,
        cols: int = None,
        start_row: int = 1,
        start_col: int = 1,
        end_row: int = -1,
        end_col: int = -1,
        name_pattern: str = '*_{xxx}_{xxx}_*',
        overlap: str = '0.1',
        fusion_flag: int = 0,
        scope_flag: int = 0,
        down_sample: int = 1,
        flip: int = 1,
        proc_count: int = 5,
        output_path: str = '',
        stereo_data: str = '',
        fft_channel: int = 0,
        file_pattern: str = '',
        stitching_type=0,
        **kwargs
) -> Union[None, np.ndarray]:
    """
    Image stitch function
    The format of the small image is as follows：
    -------------------------
       0_0, 0_1, ... , 0_n
       1_0, 1_1, ... , 1_n
       ...
       m_0, m_1, ... , m_n
    -------------------------
    Of which, m and n denote row and col

    Args:
        image_path:

        rows:

        cols:

        start_row: must >= 1, means stitch start row and end row, if image has 20 rows and 20 cols,
            start_row = 1 and end_row = 10 express only stitch row == 0 -> row == 9,
            same as numpy slice, and other area will not stitch

        start_col: Same as 'start_row'

        end_row: As shown above

        end_col: As shown above

        name_pattern:

        stitching_type:
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

        overlap: scope overlap '{overlap_x}_{overlap_y}', like '0.1_0.1'

        fusion_flag: whether or not fuse image, 0 is false

        scope_flag: scope stitch | algorithm stitch

        down_sample: down-simpling size

        flip:

        proc_count: multi-process core count

        output_path:

        stereo_data:
            - dolphin:
            - T1:
            - cellbin:

        fft_channel:

        file_pattern: re lambda, like '*.A.*.tif'

    Returns:

    Examples:
        >>>
        stitching(
            image_path = r"",
            rows = 23,
            cols = 19,
            stitching_type = 51,
            name_pattern = '*s{xx}*',
            start_row = 2,
            start_col = 2,
            end_row = 4,
            end_col = 4,
            down_sample = 2,
            stereo_data = 'cellbin',
            scope_flag = 0,
            fusion_flag = 1
        )

    """
    #  ------------------- Dedicated interface
    if len(stereo_data) > 0:
        stereo_data = stereo_data.lower()

        if stereo_data == 'cellbin':
            stitching_type = 0  # RC
            name_pattern = '*_{xxxx}_{xxxx}_*'
        else:
            name_pattern = '*C{xxx}R{xxx}*'

            if stereo_data == 't1':
                stitching_type = 1
            elif stereo_data == 'dolphin':
                stitching_type = 0
            else:
                raise ValueError('Stereo data error.')

    else:
        glog.info("Not stereo data type, using default parameter. ")

    # TODO
    #   stitching coordinate solving method
    stitch_method = 'LS-V' if stereo_data == 'dolphin' else 'cd'

    #  -------------------
    _stitch_info_print(
        stereo_data=stereo_data if len(stereo_data) > 0 else None,
        rows=rows,
        cols=cols,
        start_row=start_row,
        start_col=start_col,
        end_row=rows if end_row == -1 else end_row,
        end_col=cols if end_col == -1 else end_col,
        overlap=overlap,
        name_pattern=name_pattern,
        scope_flag=True if scope_flag else False,
        fusion_flag=True if fusion_flag else False,
        down_sample=down_sample,
        proc_count=proc_count,
        stitching_type=Scanning(stitching_type)
    )

    if len(file_pattern) > 0:
        images_path = glob(os.path.join(image_path, file_pattern))
    else:
        images_path = glob(os.path.join(image_path, '*'))

    if len(images_path) == 0:
        glog.error("No image found.")
        return

    sm = ScanMethod(stitching_type)
    imd = sm.to_default(
        images_path=images_path,
        rows=rows,
        cols=cols,
        name_pattern=name_pattern,
        sdt=stereo_data,
        flip=flip
    )

    if '_' in overlap:
        overlap_x, overlap_y = map(float, overlap.split('_'))
    else:
        overlap_x = overlap_y = float(overlap)

    sti = Stitching(
        rows=rows,
        cols=cols,
        start_row=start_row,
        start_col=start_col,
        end_row=end_row,
        end_col=end_col,
        overlap_x=overlap_x,
        overlap_y=overlap_y,
        channel=fft_channel,
        fusion=fusion_flag,
        down_sample=down_sample,
        proc_count=proc_count,
        stitch_method=stitch_method
    )

    if scope_flag:
        img = sti.stitch_by_rule(imd)
    else:
        img = sti.stitch_by_mfws(imd)

    if os.path.isdir(output_path):
        tif.imwrite(os.path.join(output_path, 'mfws.tif'), img)
    else:  # file
        tif.imwrite(os.path.join(output_path), img)


def main():
    """
    Examples:
        >>>
            python mfws.py
            -i "./"
            -r 23
            -c 19
            -sr 2
            -sc 2
            -er 4
            -ec 4
            -np *s{xx}*
            -overlap 0.1
            -s
            -f
            -d 2
            -proc 5
            -save_name my_data
            -o "./"
            -fft_channel 0
            -stereo_data cellbin
            -file_pattern "*.tif"

    Returns:

    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", action="version", version='mfws == {}'.format(version))

    parser.add_argument("-i", "--input", action="store", dest="input", type=str, required=True,
                        help="Input image dir.")
    parser.add_argument("-o", "--output", action="store", dest="output", type=str, required=True,
                        help="Result output dir.")
    parser.add_argument("-r", "--rows", action="store", dest="rows", type=int, required=True,
                        help="Image rows.")
    parser.add_argument("-c", "--cols", action="store", dest="cols", type=int, required=True,
                        help="Image cols.")

    parser.add_argument("-sr", "--start_row", action="store", dest="start_row", type=int, required=False,
                        default=1, help="Image start row.")
    parser.add_argument("-sc", "--start_col", action="store", dest="start_col", type=int, required=False,
                        default=1, help="Image start col.")
    parser.add_argument("-er", "--end_row", action="store", dest="end_row", type=int, required=False,
                        default=-1, help="Image end row.")
    parser.add_argument("-ec", "--end_col", action="store", dest="end_col", type=int, required=False,
                        default=-1, help="Image end col.")
    parser.add_argument("-proc", "--proc", action="store", dest="proc", type=int, required=False, default=5,
                        help="multi-process count.")
    parser.add_argument("-overlapx", "--overlapx", action="store", dest="overlapx", type=str, required=False,
                        default='0.1', help="Overlap - 0.1 or 0.1_0.1 .")
    parser.add_argument("-overlapy", "--overlapy", action="store", dest="overlapy", type=str, required=False,
                        default='0.1', help="Overlap - 0.1 or 0.1_0.1 .")
    parser.add_argument("-f", "--fusion", action="store_true", dest="fusion", required=False,
                        help="Fuse.")
    parser.add_argument("-scan_type", "--scan_type", action="store", dest="scan_type", type=int,
                        required=False, default=0, help="scan type")
    parser.add_argument("-np", "--name_pattern", action="store", dest="name_pattern", type=str,
                        required=False, help="Name pattern, r{rrr}_c{ccc}.tif")
    parser.add_argument("-mt", "--method", action="store_true", dest="method", required=False,
                        help="stitch method.")
    parser.add_argument("-thumbnail", "--thumbnail", action="store", dest="thumbnail", type=float, required=False,
                        default=1, help="down sampling.")
    parser.add_argument("-channel", "--channel", action="store", dest="channel", type=str,
                        required=False, default='', help="File name -- such as '*.A.*.tif'")
    parser.add_argument("-fft_channel", "--fft_channel", action="store", dest="fft_channel", type=int, required=False,
                        default=0, help="FFT channel.")

    # cellbin | dolphin | t1
    parser.add_argument("-device", "--device", action="store", dest="device", type=str,
                        required=False, default='cellbin', help="Stereo data id.")
    args = parser.parse_args()

    overlap = '{}_{}'.format(args.overlapx, args.overlapy)

    stitching(
        image_path=args.input,
        rows=args.rows,
        cols=args.cols,
        start_row=args.start_row,
        start_col=args.start_col,
        end_row=args.end_row,
        end_col=args.end_col,
        name_pattern=args.name_pattern,
        overlap=overlap,
        fusion_flag=args.fusion,
        scope_flag=args.method,
        down_sample=args.thumbnail,
        proc_count=args.proc,
        output_path=args.output,
        stereo_data=args.device,
        file_pattern=args.channel,
        fft_channel=args.fft_channel,
        stitching_type=args.scan_type
    )


if __name__ == '__main__':
    main()
