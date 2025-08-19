# -*- coding: utf-8 -*-
"""
🌟 Create Time  : 2025/5/28 17:22
🌟 Author  : CB🐂🐎 - lizepeng
🌟 File  : main.py
🌟 Description  : 
🌟 Key Words  :
"""
import os
import argparse

import glog
import numpy as np
import tifffile as tif
from glob import glob
from typing import Union
from prettytable import PrettyTable

from mfws.modules.scan_method import ScanMethod, Scanning, StitchingMethod
from mfws.modules.stitching import Stitching
from mfws import version


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
        name_index_0: bool = False,
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
        name_index_0:

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
            - CG:

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
            stereo_data = 'CG',
            scope_flag = 0,
            fusion_flag = 1
        )

    """
    #  ------------------- Dedicated interface
    stereo_data = stereo_data.lower()
    if stereo_data == 'cg':
        stitching_type = 0  # RC
        name_pattern = '*_{xxxx}_{xxxx}_*'
    elif stereo_data == 't1':
        name_pattern = '*C{xxx}R{xxx}*'
        stitching_type = 1
        name_index_0 = False
    elif stereo_data == 'dolphin':
        name_pattern = '*C{xxx}R{xxx}*'
        stitching_type = 0
        name_index_0 = False
        start_row, start_col = [start_col, start_row]
        end_row, end_col = [end_col, end_row]
        rows, cols = [cols, rows]
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
        scope_flag=StitchingMethod(scope_flag),
        fusion_flag=True if fusion_flag else False,
        down_sample=down_sample,
        proc_count=proc_count,
        stitching_type=Scanning(stitching_type),
        name_index_0=name_index_0
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
        flip=flip,
        name_index_0=name_index_0
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

    if StitchingMethod(scope_flag) == StitchingMethod.Hardware:
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
            python main.py
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
            -stereo_data CG
            -file_pattern "*.tif"

    Returns:

    """

    parser = argparse.ArgumentParser(usage='Multiple FFT Weighted Stitching')
    parser.add_argument("-v", "--version", action="version", version='mfws: v{}'.format(version))

    parser.add_argument("-i", "--input", action="store", dest="input", type=str, required=True,
                        help="File directory path")
    parser.add_argument("-o", "--output", action="store", dest="output", type=str, required=True,
                        help="Output file path or directory path")
    parser.add_argument("-r", "--rows", action="store", dest="rows", type=int, required=True,
                        help="Maximum number of scan row")
    parser.add_argument("-c", "--cols", action="store", dest="cols", type=int, required=True,
                        help="Maximum number of scan column")

    parser.add_argument("-sr", "--start_row", action="store", dest="start_row", type=int, required=False,
                        default=1, help="Start row of image to be stitched, start with 1 instead of 0")
    parser.add_argument("-sc", "--start_col", action="store", dest="start_col", type=int, required=False,
                        default=1, help="Start col of image to be stitched, start with 1 instead of 0")
    parser.add_argument("-er", "--end_row", action="store", dest="end_row", type=int, required=False,
                        default=-1, help="End row of image to be stitched")
    parser.add_argument("-ec", "--end_col", action="store", dest="end_col", type=int, required=False,
                        default=-1, help="End col of image to be stitched")
    parser.add_argument("-proc", "--proc", action="store", dest="proc", type=int, required=False, default=5,
                        help="Number of processes used, should be set reasonably based on the computing power of "
                             "the hardware platform")
    parser.add_argument("-overlapx", "--overlapx", action="store", dest="overlapx", type=float, required=False,
                        default=0.1, help="Number of overlapping pixels in the horizontal direction / width of FOV")
    parser.add_argument("-overlapy", "--overlapy", action="store", dest="overlapy", type=float, required=False,
                        default=0.1, help="Number of overlapping pixels in the vertical direction / height of FOV")
    parser.add_argument("-f", "--fusion", action="store_true", dest="fusion", required=False,
                        help="Fusion Solution: 1 - no fusion, 2 - with sin method")
    parser.add_argument("-scan_type", "--scan_type", action="store", dest="scan_type", type=int,
                        required=False, default=0, help="Scanning method")
    parser.add_argument("-np", "--name_pattern", action="store", dest="name_pattern", type=str,
                        required=False, help="Name pattern, r{rrr}_c{ccc}.tif")
    parser.add_argument("-name_index_0", "--name_index_0", action="store_false", dest="name_index_0",
                        help="Is row and column index numbers in the file name start with 0?")

    parser.add_argument("-mt", "--method", action="store", dest="method", required=False, type=int, default=1,
                        help="Stitching method: 1 - mfws, 2 - Use overlap to complete mechanical stitching")
    parser.add_argument("-thumbnail", "--thumbnail", action="store", dest="thumbnail", type=float, required=False,
                        default=1, help="Downsampling control parameter, a decimal between 0 and 1")
    parser.add_argument("-channel", "--channel", action="store", dest="channel", type=str,
                        required=False, default='',
                        help="In a multi-layer image scenario, the labels of the layers to be spliced")
    parser.add_argument("-fft_channel", "--fft_channel", action="store", dest="fft_channel", type=int, required=False,
                        default=0, help="Channel used to calculate translation")
    parser.add_argument("-flip", "--flip", action="store", dest="flip", type=int, required=False,
                        default=1, help="Flipping FOV during stitching: 1 - not, 2 - up & down, 3 - left and right")

    # CG | dolphin | t1
    parser.add_argument("-device", "--device", action="store", dest="device", type=str,
                        required=False, default='CG', help="Device Type: CG, T1, dolphin")
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
        name_index_0=args.name_index_0,
        overlap=overlap,
        fusion_flag=args.fusion,
        scope_flag=args.method,
        down_sample=args.thumbnail,
        proc_count=args.proc,
        output_path=args.output,
        stereo_data=args.device,
        file_pattern=args.channel,
        fft_channel=args.fft_channel,
        stitching_type=args.scan_type,
        flip=args.flip
    )


if __name__ == '__main__':
    main()
