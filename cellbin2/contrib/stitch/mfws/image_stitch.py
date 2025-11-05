import os
import re
import argparse
import tifffile

import numpy as np
import cv2 as cv
from glob import glob

try:
    from .modules import stitching
except ImportError:
    from modules import stitching
from typing import Union


def search_files(file_path, exts):
    files_ = list()
    for root, dirs, files in os.walk(file_path):
        if len(files) == 0:
            continue
        for f in files:
            fn, ext = os.path.splitext(f)
            if ext in exts: files_.append(os.path.join(root, f))

    return files_


def rc_key(row: int, col: int):
    return '{}_{}'.format(str(row).zfill(4), str(col).zfill(4))


def filename2index(file_name, style='motic', row_len=None):
    file_name = os.path.basename(file_name)
    if style.lower() in ['motic', 'cghd']:
        tags = os.path.splitext(file_name)[0].split('_')
        xy = list()
        for tag in tags:
            if (len(tag) == 4) and tag.isdigit():
                if int(tag) < 999:
                    xy.append(tag)
        x_str = xy[0]
        y_str = xy[1]
        return [int(y_str), int(x_str)]
    elif style.lower() == 'zeiss':
        line = os.path.splitext(file_name)[0].split('_')
        c = int(float(line[2]))
        r = int(float(line[1]))
        return [c, r]
    elif style.lower() == "leica dm6b":
        num = file_name.split("_")[1][1:]
        x = int(int(num) / row_len)
        y = int(int(num) % row_len)
        if x % 2 == 1:
            y = row_len - y - 1
        return [y, x]
    else:
        return None


def chip_qc_filename2index(file_name):

    file_name = os.path.basename(file_name)
    pat = re.compile(r'C\d+R\d+')
    res = pat.search(file_name).group(0)

    r = int(res[1:4])
    c = int(res[5:])
    return [c, r]


def select_row_col(src_fovs, process_rule, row, col):
    st_row, end_row = map(int, row.split('_'))
    st_col, end_col = map(int, col.split('_'))

    new_rows = end_row - st_row
    new_cols = end_col - st_col

    new_src_fovs = dict()
    for k, v in src_fovs.items():
        _r, _c = map(int, k.split('_'))
        if st_row <= _r < end_row and st_col <= _c < end_col:
            new_src_fovs[f'{_r - st_row:04}_{_c - st_col:04}'] = v

    new_process_rule = dict()
    if process_rule is not None:
        for k, v in process_rule.items():
            _r, _c = map(int, k.split('_'))
            if st_row <= _r < end_row and st_col <= _c < end_col:
                new_process_rule[f'{_r - st_row:04}_{_c - st_col:04}'] = v
    else: new_process_rule = None

    return new_src_fovs, new_process_rule, new_rows, new_cols


def images_path2dict(
        images_path,
        style = 'motic',
        row_len = None,
        stereo_data = 'cellbin',
        file_type = '',
):
    """

    Args:
        images_path:
        style:
        row_len:
        stereo_data:
        file_type:

    Returns:

    """
    if len(file_type) == 0:
        image_support = ['.jpg', '.png', '.tif', '.tiff', '.TIFF']
        fov_images = search_files(images_path, exts=image_support)
    else:
        fov_images = glob(os.path.join(images_path, file_type))

    src_fovs = dict()
    rows = cols = -1
    for it in fov_images:
        if stereo_data == "cellbin":
            col, row = filename2index(it)
        else:
            col, row = chip_qc_filename2index(it)
            if stereo_data == 't1':
                col, row = row, col

        if row > rows: rows = row
        if col > cols: cols = col

        if stereo_data == "dolphin":
            if col % 2 == 0:
                fov = stitching.ImageBase(it, flip_ud = True)
            else:
                fov = stitching.ImageBase(it)
        else:
            fov = stitching.ImageBase(it)

        src_fovs[rc_key(row, col)] = fov

    return src_fovs, rows + 1, cols + 1


def create_loc(rows, cols, shape, overlap):
    height, width = shape
    overlap_x, overlap_y = overlap
    fov_loc = np.zeros((rows, cols, 2), dtype=int)
    for i in range(rows):
        for j in range(cols):
            fov_loc[i, j, 0] = j * (width - int(width * overlap_x))
            fov_loc[i, j, 1] = i * (height - int(height * overlap_y))

    return fov_loc


def stitch_image(
        image_path: str = '',
        process_rule: dict = None,
        overlap: str = '0.1',
        name: str = '',
        fuse_flag: bool = True,
        scope_flag: bool = False,
        down_size: int = 1,
        row_slice: str = '-1_-1',
        col_slice: str = '-1_-1',
        output_path: str = '',
        stereo_data: str = 'cellbin',
        file_type: str = '',
        debug: bool = False
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

        name: image name

        process_rule:

        overlap: scope overlap

        fuse_flag: whether or not fuse image

        scope_flag: scope stitch | algorithm stitch

        down_size: down-simpling size

        row_slice: means stitch start row and end row,
             if image has 20 rows and 20 cols, row = '0_10' express only stitch row == 0 -> row == 9,
             same as numpy slice, and other area will not stitch

        col_slice: same as 'row'

        output_path:

        stereo_data:
            - V3:
            - dolphin:
            - T1:
            - cellbin:

        file_type: re lambda, like '*.A.*.tif'

        debug:

    Returns:

    Examples:
        >>>

    """
    stereo_data = stereo_data.lower()
    if isinstance(image_path, str) and os.path.isdir(image_path):
        src_fovs, rows, cols = images_path2dict(
            image_path, stereo_data=stereo_data, file_type=file_type
        )# , style="leica dm6b", row_len=5)

    elif isinstance(image_path, dict):
        src_fovs = image_path
        rows, cols = np.array(
            [list(map(int, k.split('_'))) for k in image_path.keys()]
        ).max(axis = 0) + 1

    else:
        raise ImportError("Image path format error.")

    if '-1' not in row_slice and '-1' not in col_slice:
        src_fovs, process_rule, rows, cols = select_row_col(
            src_fovs, process_rule, row_slice, col_slice
        )

    stitch = stitching.Stitching()
    stitch.set_size(rows, cols)

    if '_' in overlap:
        overlap_x, overlap_y = map(float, overlap.split('_'))
    else:
        overlap_x = overlap_y = float(overlap)

    stitch.set_overlap([overlap_x, overlap_y])

    loc = None
    if scope_flag:
        _img = list(src_fovs.items())[0][1]
        shape = _img.get_image().shape
        loc = create_loc(rows, cols, shape, [overlap_x, overlap_y])

    if loc is not None:
        stitch.set_global_location(loc)

    stitch.stitch(
        src_fovs,
        process_rule=process_rule,
        fuse_flag=fuse_flag,
        down_size=down_size,
        stitch_method= 'LS-V' if stereo_data == 'dolphin' else 'cd',
        debug=debug,
        output_path = output_path
    )

    image = stitch.get_image()
    #
    if os.path.isdir(output_path):
        if len(name) == 0: name = 'image'
        tifffile.imwrite(
            os.path.join(output_path, f'{name}_fov_stitched.tif'), image
        )
    else:
        return image


def main(args, para):
    stitch_image(
        image_path = args.input,
        overlap = args.overlap,
        name = args.name,
        fuse_flag = args.fuse,
        down_size = args.down,
        scope_flag = args.scope,
        row_slice = args.row,
        col_slice = args.col,
        output_path = args.output,
        stereo_data = args.id,
        file_type = args.file_type,
        debug = args.debug
    )


def arg_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", action="store", dest="input", type=str, required=True,
                        help="Tar file / Input image dir.")

    # scope overlap
    parser.add_argument("-overlap", "--overlap", action="store", dest="overlap", type=str, required=False,
                        default='0.1', help="Overlap - 0.1 or 0.1_0.1 .")

    # scope stitch or algorithm stitch
    parser.add_argument("-s", "--scope", action = "store_true", dest = "scope",
                        required = False, help = "Scope stitch.")

    # fuse
    parser.add_argument("-f", "--fuse", action = "store_true", dest = "fuse", required = False, help = "Fuse.")

    # down-sampling
    parser.add_argument("-d", "--down", action = "store", dest = "down", type = float, required = False,
                        default = 1, help = "Down-sampling.")

    # block selection
    parser.add_argument("-row", "--row", action = "store", dest = "row", type = str, required = False,
                        default = '-1_-1', help = "Image select block - row.")
    parser.add_argument("-col", "--col", action = "store", dest = "col", type = str, required = False,
                        default = '-1_-1', help = "Image select block - col.")

    parser.add_argument("-n", "--name", action="store", dest="name", type=str, required=False,
                        default = '', help="Name.")
    parser.add_argument("-o", "--output", action="store", dest="output", type=str, required=False,
                        default = '', help="Result output dir.")

    parser.add_argument("-debug", "--debug", action = "store_true", dest = "debug", required = False, help = "debug.")

    """
    interface by stereo data --
       V3: 
       dolphin:
       T1:
       cellbin:
    any case is fine. 
    """
    parser.add_argument("-id", "--id", action = "store", dest = "id", type = str, required = False,
                        default = 'cellbin', help = "Stereo data id.")
    parser.add_argument("-file_type", "--file_type", action = "store", dest = "file_type", type = str,
                        required = False, default = '', help = "File name -- such as '*.A.*.tif'.")

    parser.set_defaults(func=main)
    (para, args) = parser.parse_known_args()
    para.func(para, args)


if __name__ == "__main__":
    arg_parser()


