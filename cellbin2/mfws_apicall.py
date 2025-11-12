import os
import re
import sys
import tarfile
import tempfile
import glob
import shutil
import argparse

CURR_PATH = os.path.dirname(os.path.realpath(__file__))
CB2_PATH = os.path.dirname(CURR_PATH)
sys.path.append(CB2_PATH)

from contrib.stitch.mfws.main import stitching

def get_max_rows_cols(folder_path):

    tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    pattern = r'_(\d{4})_(\d{4})_'
    rows = []
    cols = []
    
    for tif_file in tif_files:
        match = re.search(pattern, tif_file)
        if match:
            row = int(match.group(1))
            col = int(match.group(2))
            rows.append(row)
            cols.append(col)
    
    if not rows or not cols:
        raise ValueError("No valid FOV positions found in filenames")
        
    max_rows = max(rows) + 1
    max_cols = max(cols) + 1

    return max_rows, max_cols

def resolve_image_path(image_input, stain_type=None):
    image_input = os.path.abspath(image_input)
    lower = image_input.lower()

    # 已经是目录或单个 tif 文件
    if os.path.isdir(image_input) or lower.endswith('.tif') or lower.endswith('.tiff'):
        return image_input, None

    # 处理 tar.gz / tgz 包
    if lower.endswith('.tar.gz') or lower.endswith('.tgz'):
        tmp = tempfile.TemporaryDirectory(prefix='mfws_')
        try:
            with tarfile.open(image_input, 'r:*') as tf:
                tf.extractall(tmp.name)
        except Exception:
            tmp.cleanup()
            raise

        root = tmp.name
        stain_dir = os.path.join(root, stain_type)

        if os.path.isdir(stain_dir):
            return stain_dir, tmp

    # 其他未知后缀，直接返回原始（调用方可做进一步判断）
    return image_input, None

def file_to_stitch(image_input, sn, stain_type, base_dir):

    abs_input = os.path.abspath(image_input)
    lower = abs_input.lower()

    image_path, cleanup = resolve_image_path(image_input, stain_type=stain_type)

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    stitch_path = os.path.join(base_dir, f"{sn}_stitch.tif")

    try:
        # 若解析到的是单个 tif 文件（可能是直接传入的文件，或解压后找到的 tif）
        if os.path.isfile(image_path) and (image_path.lower().endswith('.tif') or image_path.lower().endswith('.tiff')):
            if os.path.exists(stitch_path):
                os.remove(stitch_path)
            shutil.move(image_path, stitch_path)
            return stitch_path

        # 若解析到的是目录
        if os.path.isdir(image_path):
            tifs = glob.glob(os.path.join(image_path, '*.tif')) + glob.glob(os.path.join(image_path, '*.tiff'))
            if tifs:
                # 使用第一个 tif 文件并改名到 parent(image_path)/{sn}_stitch.tif
                src = tifs[0]
                if os.path.exists(stitch_path):
                    os.remove(stitch_path)
                shutil.move(src, stitch_path)
                return stitch_path
            else:
                subfolder_path = [f for f in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, f))]
                sub_tifs_path = os.path.join(image_path, subfolder_path[0]) 
                row, col = get_max_rows_cols(sub_tifs_path)

            if os.path.exists(stitch_path):
                os.remove(stitch_path)
            stitching(
                image_path = sub_tifs_path,
                output_path = stitch_path,
                rows= row,
                cols= col,
                start_row=0,
                start_col=0,
                overlap = '0.1_0.1',
                name_pattern = '*_{xxxx}_{xxxx}_*',
                scope_flag = True,
                fuse = 1
            )
            return stitch_path

        # 其他情况（无法解析），返回原始输入的父目录下的默认 stitch_path
        return stitch_path  
    finally:
        if cleanup:
            cleanup.cleanup()

def main():
    parser = argparse.ArgumentParser(description='Preprocess and stitch MFWS image input')
    parser.add_argument('-i', '--input', required=True, help='Path to image input (tif file, directory, or tar/tar.gz)')
    parser.add_argument('-c', '--chip', required=True, help='Chip/sample name (used to name stitched output)')
    parser.add_argument('-s', '--stain', required=True, default='ssDNA', help='Stain type (e.g. DAPI) used to find subfolder inside tar')
    parser.add_argument('-p', '--base-dir', required=True, default='./temp_output', help='Base directory to write stitched output')
    args = parser.parse_args()

    stitch_path = file_to_stitch(
        image_input=args.input,
        sn=args.chip,
        stain_type=args.stain,
        base_dir=args.base_dir
    )

if __name__ == '__main__':
    main()