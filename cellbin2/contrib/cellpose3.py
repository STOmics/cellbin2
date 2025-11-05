# RUN CELLPOSE
import os
import glob
from math import ceil
import numpy as np
from cellpose import models, io,denoise
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





def cellpose3_pred(img_path, 
                   cfg, 
                   use_gpu, 
                   stain_type,
                   output_path=None,
                   photo_size=2048,
                   photo_step=2000,):
    try:
        import cellpose
    except ImportError:
        pip.main(['install', 'cellpose==3.1.1.2'])
    import cellpose
    if cellpose.version != '3.1.1.2':
        pip.main(['install', 'cellpose==3.1.1.2'])
    import cellpose
    try:
        import patchify
    except ImportError:
        pip.main(['install', 'patchify==0.2.3'])
    from cellpose import models, io,denoise
    import patchify
    overlap = photo_size - photo_step
    if (overlap % 2) == 1:
        overlap = overlap + 1
    act_step = ceil(overlap / 2)
    logging.getLogger('cellpose.models').setLevel(logging.WARNING)
    if stain_type == "DAPI" or stain_type =="ssDNA":
        model = denoise.CellposeDenoiseModel(gpu=use_gpu, model_type="nuclei")
    else:
        model = denoise.CellposeDenoiseModel(gpu=use_gpu, model_type="cyto3")

    img = tifffile.imread(img_path)
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
            masks, flows, styles, diams = model.eval(img_data, diameter=None, channels=[0, 0])
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
    # #model = denoise.CellposeDenoiseModel(gpu=True, model_type="cyto3",
    #                                    restore_type="denoise_cyto3")
    #model = models.CellposeModel(gpu=True, model_type='/storeData/USER/data/01.CellBin/00.user/fanjinghong/code/cs-benchmark/src/methods/models/cyto3_train_on_cellbinDB')

    semantics = instance2semantics(cropped_1)
    return semantics

