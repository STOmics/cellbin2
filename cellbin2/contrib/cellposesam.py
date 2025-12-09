# RUN CELLPOSE
import os
import glob
from math import ceil
import numpy as np
import pip
import tifffile
import argparse
import tqdm
import logging
models_logger = logging.getLogger(__name__)
import cv2
from skimage.morphology import remove_small_objects


from cellbin2.image.augmentation import f_ij_16_to_8_v2 as f_ij_16_to_8
from cellbin2.image.augmentation import f_rgb2gray
from cellbin2.contrib.cellpose_segmentor import instance2semantics, f_instance2semantics_max, poolingOverlap
from cellbin2.image.mask import f_instance2semantics
from cellbin2.image import cbimread, cbimwrite
from cellbin2.contrib.cell_segmentor import CellSegParam
from cellbin2.utils import clog

def instance2semantics(ins: np.ndarray) -> np.ndarray:
    """
    :param ins: Instance mask (0-N)
    :return: Semantics mask (0-1)
    """
    ins_ = ins.copy()
    h, w = ins_.shape[:2]
    tmp0 = ins_[1:, 1:] - ins_[:h - 1, :w - 1]
    ind0 = np.where(tmp0 != 0)

    tmp1 = ins_[1:, :w - 1] - ins_[:h - 1, 1:]
    ind1 = np.where(tmp1 != 0)
    ins_[ind1] = 0
    ins_[ind0] = 0
    ins_[np.where(ins_ > 0)] = 1
    return np.array(ins_, dtype=np.uint8)

'''def cellposesam_pred_test(img_path, 
                   use_gpu, 
                   stain_type):
    try:
        import cellpose
    except ImportError:
        pip.main(['install', 'git+https://www.github.com/mouseland/cellpose.git'])
    if cellpose.version != '4.0.8':
        pip.main(['install', 'git+https://www.github.com/mouseland/cellpose.git'])
    import cellpose
    print("111111111")
    from cellpose import models, core, io, plot
    
    print("11111222")
    try:
        import patchify
    except ImportError:
        pip.main(['install', 'patchify==0.2.3'])
    import patchify

    
    img = io.imread(str(img_path))
    
    new_model_path = "/home/wangaoli/.cellpose/models/cpsam"
    img = io.imread(str(img_path))
    model = models.CellposeModel(gpu=False,
                             pretrained_model=new_model_path)
    
    masks = model.eval(img, batch_size=32)[0]
    semantics = instance2semantics(masks)
    return semantics
'''
def cellposesam_pred(img_path, 
                   cfg, 
                   use_gpu, 
                   model_dir,
                   output_path=None,
                   photo_size=2048,
                   photo_step=2000,):
    try:
        import cellpose
    except ImportError:
        pip.main(['install', 'git+https://www.github.com/mouseland/cellpose.git'])
    if cellpose.version != '4.0.8':
        pip.main(['install', 'git+https://www.github.com/mouseland/cellpose.git'])
    import cellpose

    from cellpose import models, core, io, plot
    try:
        import patchify
    except ImportError:
        pip.main(['install', 'patchify==0.2.3'])
    import patchify
    overlap = photo_size - photo_step
    if (overlap % 2) == 1:
        overlap = overlap + 1
    act_step = ceil(overlap / 2)
    logging.getLogger('cellpose.models').setLevel(logging.WARNING)
    model = models.CellposeModel(gpu = use_gpu, pretrained_model=model_dir)
    img = cbimread(img_path, only_np=True)
    img = f_ij_16_to_8(img)
    img = f_rgb2gray(img, True)

    res_image = np.pad(img, ((act_step, act_step), (act_step, act_step)), 'constant')
    res_a = res_image.shape[0]
    res_b = res_image.shape[1]
    re_length = ceil((res_a - (photo_size - photo_step)) / photo_step) * photo_step + (
            photo_size - photo_step)
    re_width = ceil((res_b - (photo_size - photo_step)) / photo_step) * photo_step + (
            photo_size - photo_step)
    regray_image = np.pad(res_image, ((0, re_length - res_a), (0, re_width - res_b)), 'constant')
    patches = patchify.patchify(regray_image, (photo_size, photo_size), step=photo_step)
    wid = patches.shape[0]
    high = patches.shape[1]
    a_patches = np.full((wid, high, (photo_size - overlap), (photo_size - overlap)), 255, dtype=np.uint8)

    for i in tqdm.tqdm(range(wid), desc='Segment cells with [Cellpose]'):
        for j in range(high):
            img_data = patches[i, j, :, :]
            masks = model.eval(img_data, diameter=None, channels=[0, 0])[0]
            masks = f_instance2semantics_max(masks)
            a_patches[i, j, :, :] = masks[act_step:(photo_size - act_step),
                                    act_step:(photo_size - act_step)]

    patch_nor = patchify.unpatchify(a_patches,
                                    ((wid) * (photo_size - overlap), (high) * (photo_size - overlap)))
    nor_imgdata = np.array(patch_nor)
    after_wid = patch_nor.shape[0]
    after_high = patch_nor.shape[1]
    cropped_1 = nor_imgdata[0:(after_wid - (re_length - res_a)), 0:(after_high - (re_width - res_b))]
    cropped_1 = np.uint8(remove_small_objects(cropped_1 > 0, min_size=2))
    if output_path is not None:
        name = os.path.splitext(os.path.basename(img_path))[0]
        c_mask_path = os.path.join(output_path, f"{name}_v3_mask.tif")
        cbimwrite(output_path=c_mask_path, files=cropped_1, compression=True)
    return cropped_1

if __name__ == '__main__':
    img_path = "/storeData/USER/data/01.CellBin/00.user/wangaoli/data/raw_data/时空多蛋白数据/Xenium_Multimodal/Mouse_Brain/channel1_crop.tif"
    model_path = "/home/wangaoli/.cellpose/models/cpsam"
    mask =  cellposesam_pred(img_path, 
                    use_gpu = True, 
                    model_dir = model_path,
                    stain_type = None)
    tifffile.imwrite("/storeData/USER/data/01.CellBin/00.user/wangaoli/data/raw_data/时空多蛋白数据/Xenium_Multimodal/Mouse_Brain/channel1_crop_mask.tif", mask)

