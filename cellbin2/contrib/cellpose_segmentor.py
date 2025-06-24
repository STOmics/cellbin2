import os
import sys
from math import ceil
import pip
import tqdm
import numpy.typing as npt
import numpy as np
from skimage.morphology import remove_small_objects

from cellbin2.image.augmentation import f_ij_16_to_8_v2 as f_ij_16_to_8
from cellbin2.image.augmentation import f_rgb2gray
from cellbin2.image.mask import f_instance2semantics
from cellbin2.image import cbimread, cbimwrite
from cellbin2.contrib.cell_segmentor import CellSegParam
from cellbin2.utils import clog


# os.environ['CELLPOSE_LOCAL_MODELS_PATH'] = "/media/Data/dzh/weights"


def asStride(arr, sub_shape, stride):
    '''Get a strided sub-matrices view of an ndarray.
    See also skimage.util.shape.view_as_windows()
    '''
    s0, s1 = arr.strides[:2]
    m1, n1 = arr.shape[:2]
    m2, n2 = sub_shape
    view_shape = (1 + (m1 - m2) // stride[0], 1 + (n1 - n2) // stride[1], m2, n2) + arr.shape[2:]
    strides = (stride[0] * s0, stride[1] * s1, s0, s1) + arr.strides[2:]
    subs = np.lib.stride_tricks.as_strided(arr, view_shape, strides=strides)
    return subs


def poolingOverlap(mat, ksize, stride=None, method='max', pad=False):
    '''Overlapping pooling on 2D or 3D data.
    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <stride>: tuple of 2 or None, stride of pooling window.
              If None, same as <ksize> (non-overlapping pooling).
    <method>: str, 'max for max-pooling,
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           (n-f)//s+1, n being <mat> size, f being kernel size, s stride.
           if pad, output has size ceil(n/s).
    Return <result>: pooled matrix.
    '''

    m, n = mat.shape[:2]
    ky, kx = ksize
    if stride is None:
        stride = (ky, kx)
    sy, sx = stride

    _ceil = lambda x, y: int(np.ceil(x / float(y)))

    mat = np.where(mat == 0, np.nan, mat)

    if pad:
        ny = _ceil(m, sy)
        nx = _ceil(n, sx)
        size = ((ny - 1) * sy + ky, (nx - 1) * sx + kx) + mat.shape[2:]
        mat_pad = np.full(size, np.nan)
        mat_pad[:m, :n, ...] = mat
    else:
        mat_pad = mat[:(m - ky) // sy * sy + ky, :(n - kx) // sx * sx + kx, ...]

    view = asStride(mat_pad, ksize, stride)
    if method == 'max':
        result = np.nanmax(view, axis=(2, 3))
    else:
        result = np.nanmean(view, axis=(2, 3))
    result = np.nan_to_num(result)
    return result


def f_instance2semantics_max(ins):
    ins_m = poolingOverlap(ins, ksize=(2, 2), stride=(1, 1), pad=True, method='mean')
    mask = np.uint8(np.subtract(np.float64(ins), ins_m))
    ins[mask != 0] = 0
    ins = f_instance2semantics(ins)
    return ins


def main(
        file_path,
        gpu: int,
        model_dir: str,
        model_name='cyto2',
        output_path=None,
        photo_size=2048,
        photo_step=2000,
) -> np.ndarray:
    """
    Args:
        file_path: 待处理图像存储为文件夹路径
        photo_size: 剪裁成小图时，小图的大小
        photo_step: 剪裁图像时的步长
        model_name: 模型名，默认cyto2
        output_path: 保存的文件夹

    Returns:
        cropped_1: 细胞分割结果（semantic）
    """
    os.environ['CELLPOSE_LOCAL_MODELS_PATH'] = model_dir  # 这样设置后，cellpose会去这里找
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}"
    try:
        import cellpose
    except ImportError:
        pip.main(['install', 'cellpose==3.0.11'])
    import cellpose
    if cellpose.version != '3.0.11':
        pip.main(['install', 'cellpose==3.0.11'])
    try:
        import patchify
    except ImportError:
        pip.main(['install', 'patchify==0.2.3'])
    from cellpose import models
    import patchify

    overlap = photo_size - photo_step
    if (overlap % 2) == 1:
        overlap = overlap + 1
    act_step = ceil(overlap / 2)

    model = models.Cellpose(gpu=True, model_type=model_name)

    img = cbimread(file_path, only_np=True)
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
    if output_path is not None:
        name = os.path.splitext(os.path.basename(file_path))[0]
        c_mask_path = os.path.join(output_path, f"{name}_v3_mask.tif")
        cbimwrite(output_path=c_mask_path, files=cropped_1, compression=True)
    return cropped_1


demo = """
python cellpose_segmentor.py \
-i
"xxx/B02512C5_after_tc_regist.tif"
-o
xxx/tmp
-m
xxx/models
-n
cyto2
-g
0
"""


def segment4cell(input_path: str, cfg: CellSegParam) -> npt.NDArray[np.uint8]:
    mask = main(
        file_path=input_path,
        gpu=cfg.GPU,
        model_dir=os.path.dirname(cfg.IF_weights_path)
    )

    return mask


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(usage=f"{demo}")
    parser.add_argument('-i', "--input", help="the input img path")
    parser.add_argument('-o', "--output", help="the output file")
    parser.add_argument("-m", "--model_dir", help="model dir")
    parser.add_argument("-n", "--model_name", help="model name", default="cyto2")
    parser.add_argument("-g", "--gpu", help="the gpu index", default="-1")

    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    model_name = args.model_name
    gpu = args.gpu
    model_dir = args.model_dir

    main(
        file_path=input_path,
        gpu=gpu,
        model_dir=model_dir,
        model_name=model_name,
        output_path=output_path
    )
    sys.exit()

    # model = r'E:\03.users\liuhuanlin\01.data\cellbin2\weights'
    # input_path = r'E:\03.users\liuhuanlin\01.data\cellbin2\output\B03624A2_DAPI_10X.tiff'
    # cfg = CellSegParam(**{'IF_weights_path': model, 'GPU': 0})
    # mask = segment4cell(input_path, cfg)
    # cbimwrite(r'E:\03.users\liuhuanlin\01.data\cellbin2\output\res_mask.tiff', mask)
