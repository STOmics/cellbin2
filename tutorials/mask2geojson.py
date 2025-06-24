import copy
import os
import shutil

import tifffile
import cv2
import json
# from wsi_split import SplitWSI
import numpy as np
import sys
import glob
from scipy import ndimage
import rasterio.features


def f_ij_16_to_8(img, chunk_size=1000):
    """
    16 bits img to 8 bits

    :param img: (CHANGE) np.array
    :param chunk_size: chunk size (bit)
    :return: np.array
    """

    if img.dtype == 'uint8':
        return img
    dst = np.zeros(img.shape, np.uint8)
    p_max = np.max(img)
    p_min = np.min(img)
    scale = 256.0 / (p_max - p_min + 1)
    for idx in range(img.shape[0] // chunk_size + 1):
        sl = slice(idx * chunk_size, (idx + 1) * chunk_size)
        win_img = copy.deepcopy(img[sl])
        win_img = np.int16(win_img)
        win_img = (win_img & 0xffff)
        win_img = win_img - p_min
        win_img[win_img < 0] = 0
        win_img = win_img * scale + 0.5
        win_img[win_img > 255] = 255
        dst[sl] = np.array(win_img).astype(np.uint8)
    return dst


def feature_dct(coords, id_no):
    g_dct = {
        'type': 'Polygon',
        'coordinates': coords
    }

    p_dct = {
        'objectType': 'annotation',
        'classification': {'name': 'cells', 'color': [255, 0, 0]}
    }

    f_dct = {
        'type': 'Feature',
        'id': id_no,
        'geometry': g_dct,
        'properties': p_dct
    }

    return f_dct


def write_qupath_object(out_file, mask):
    j_dct = {
        'type': 'FeatureCollection',
        'features': []
    }

    lab, n = ndimage.label(mask)
    cnts = rasterio.features.shapes(lab, mask=mask, connectivity=8)
    count = 0
    for i in cnts:
        cnt = i[0]['coordinates']

        if len(cnt[0]) < 3:
            continue
        count += 1
        f_dct = feature_dct(cnt, count)
        j_dct['features'].append(f_dct)

    with open("{}".format(out_file), 'w') as jf:
        json.dump(j_dct, jf)


def del_files(filepath):
    """
    删除某一目录下的所有文件或文件夹
    :param filepath: 路径
    :return:
    """
    if os.path.isfile(filepath):
        os.remove(filepath)
    else:
        del_list = os.listdir(filepath)
        for f in del_list:
            file_path = os.path.join(filepath, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        os.rmdir(filepath)
    return


def split_and_save(img_file, mask_file, tar_path, win_size=256):
    name = os.path.split(img_file)[-1]
    name = os.path.splitext(name)[0]

    img_dir = os.path.join(tar_path, name, "img")
    json_dir = os.path.join(tar_path, name, "json")

    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    img = tifffile.imread(img_file)
    img = np.squeeze(img)
    img = f_ij_16_to_8(img)
    mask = tifffile.imread(mask_file)
    mask = np.squeeze(mask)
    mask = np.clip(mask, 0, 1)
    mask = np.uint8(mask * 255)

    sp_run = SplitWSI(img, (win_size, win_size), 0, 0, False, False, False, np.uint8)
    box_lst, _, _ = sp_run.f_split2run()

    for i in range(len(box_lst)):
        y_begin, y_end, x_begin, x_end = box_lst[i]
        win_mask = mask[y_begin:y_end, x_begin:x_end]
        if not np.sum(win_mask) > 0:
            continue
        win_img = img[y_begin:y_end, x_begin:x_end]
        write_qupath_object(os.path.join(json_dir, f"{name}.json"), win_mask)
        tifffile.imwrite(os.path.join(img_dir, f"{name}.tif"), win_img,
                         compression='zlib')


if __name__ == '__main__':
    img_input = r"C:\Users\shican\Desktop\test\img"  # 输入图片文件夹路径
    mask_input = r"C:\Users\shican\Desktop\test\mask"  # 输入mask文件夹路径
    tar_path = r"C:\Users\shican\Desktop\test\json"  # 输出文件夹路径
    win_size = 256  # 小图大小

    for img_file in glob.glob(img_input + "/*.tif"):
        file_name = img_file.replace('\\', '/').split('/')[-1].split('.')[0]
        mask_file = os.path.join(mask_input, file_name + ".tif")
        print(mask_file)

        if os.path.isfile(mask_file):
            print("Image file:", img_file)
            print("Mask file:", mask_file)
            split_and_save(img_file, mask_file, tar_path, win_size=win_size)
        else:
            print("No mask file found for:", img_file)