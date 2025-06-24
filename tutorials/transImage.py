# -*- coding: utf-8 -*-
# @Time    : 2025/2/27 10:37
# @Author  : fengning
# @File    : transImage.py
import argparse
import os.path

import numpy as np
from typing import Union
from cellbin2.image import cbimread, cbimwrite
from cellbin2.image.augmentation import f_ij_16_to_8


def trans_image(img_file: Union[np.ndarray, str],
              output_dir: str,
              **kwargs
              ):
    """

    Args:
        img_file: np.ndarray or Path(str)
        output_dir: Path(str)
        **kwargs:
            scale:
                - list | tuple, [scale_x, scale_y]
            rotate: 小角度旋转，正数表示逆时针旋转
            rot90: 正整数表示逆时针旋转90°
            offset: [x, y] 正数表示向左向下
            dst_size: (height, width)
            flip_lr:
            flip_ud:
            trans_mat:

    Returns:file

    """
    t_img = cbimread(img_file)

    result = t_img.trans_image(**kwargs)
    image = f_ij_16_to_8(result.image)

    if os.path.isdir(output_dir):
        name = 'Transformed_' + os.path.basename(img_file).split('.')[0] + '.tif'
        output_file = os.path.join(output_dir, name)
    else:
        output_file = output_dir

    cbimwrite(output_file, image)


if __name__ == '__main__':
    # trans_image(img_file= r"D:\temp\SS200000059_NC\fov_stitched_DAPI.tif",
    #           output_dir= r'D:\temp\\SS200000059_NC\test',
    #           scale=[0.5, 0.5],
    #           rotate=3
    #           )

    usage_str = f"python {os.path.basename(__file__)} \\ \n" \
                f"-i Path/to/img.tif \\ \n" \
                f"-o Path/to/output(.tif) \\ \n" \
                f"-s 0.5 0.5 \\ \n" \
                f"-r 1.00999 \\ \n" \
                f"-R 1 \\ \n" \
                f"-x 1345 2274 \\ \n" \
                f"-d 15540 22440 \\ \n" \
                f"-fl \\ \n" \
                f"-fu \\ \n" \
                f"** if you input offset, you should input dst_size **"

    parser = argparse.ArgumentParser(usage=usage_str)
    parser.add_argument("-i", "--input",
                        required=True, help="the input file(can be np.ndarray or path)")
    parser.add_argument("-o", "--output",
                        required=True, help="the output directory or the output file dir")

    parser.add_argument("-s", "--scale", default=None, nargs='+', type=float, help="scale: x y")
    parser.add_argument("-r", "--rotate", default=None, type=float, help="rotate degree")
    parser.add_argument("-R", "--rot90", default=None, type=int, help="rot90, 1 means 90 degrees ")
    parser.add_argument("-x", "--offset", default=None, nargs='+',  type=float, help="offset: x y")
    parser.add_argument("-d", "--dst_size", default=None, nargs='+', type=int, help="dst_size")
    parser.add_argument("-fl", "--flip_lr", action='store_true', help="flip left-right")
    parser.add_argument("-fu", "--flip_ud", action='store_true', help="flip up-down")
    # parser.add_argument("-m", "--trans_mat", default=None, nargs='+', type=float, help="trans_mat")

    args = parser.parse_args()

    trans_image(img_file=args.input,
              output_dir=args.output,
              scale=args.scale,
              rotate=args.rotate,
              rot90=args.rot90,
              offset=args.offset,
              dst_size=args.dst_size,
              flip_lr=args.flip_lr,
              flip_ud=args.flip_ud)
    print('done')

