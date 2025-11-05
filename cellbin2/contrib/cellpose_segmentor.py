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
def instance2semantics(ins):
    """
    instance to semantics
    Args:
        ins(ndarray):labeled instance

    Returns(ndarray):mask
    """
    ins[np.where(ins > 0)] = 1
    return np.array(ins, dtype=np.uint8)

def asStride(arr, sub_shape, stride):
    """
    Get a strided sub-matrices view of an ndarray.

    This function is similar to `skimage.util.shape.view_as_windows()`.

    Args:
        arr (ndarray): The input ndarray.
        sub_shape (tuple): The shape of the sub-matrices.
        stride (tuple): The step size along each axis.

    Returns:
        ndarray: A view of strided sub-matrices.
    """
    s0, s1 = arr.strides[:2]
    m1, n1 = arr.shape[:2]
    m2, n2 = sub_shape
    view_shape = (1 + (m1 - m2) // stride[0], 1 + (n1 - n2) // stride[1], m2, n2) + arr.shape[2:]
    strides = (stride[0] * s0, stride[1] * s1, s0, s1) + arr.strides[2:]
    subs = np.lib.stride_tricks.as_strided(arr, view_shape, strides=strides)
    return subs


def poolingOverlap(mat, ksize, stride=None, method='max', pad=False):
    """
    Perform overlapping pooling on 2D or 3D data.

    Args:
        mat (ndarray): The input array to pool.
        ksize (tuple of 2): Kernel size in (ky, kx).
        stride (tuple of 2, optional): Stride of the pooling window. If None, it defaults to the kernel size (non - overlapping pooling).
        method (str, optional): Pooling method, 'max' for max - pooling, 'mean' for mean - pooling.
        pad (bool, optional): Whether to pad the input matrix or not. If not padded, the output size will be (n - f)//s+1, where n is the matrix size, f is the kernel size, and s is the stride. If padded, the output size will be ceil(n/s).

    Returns:
        ndarray: The pooled matrix.
    """

    m, n = mat.shape[:2]
    ky, kx = ksize
    if stride is None:
        stride = (ky, kx)
    sy, sx = stride

    _ceil = lambda x, y: int(np.ceil(x / float(y)))

    # Replace zeros with NaNs to handle them in max and mean calculations
    mat = np.where(mat == 0, np.nan, mat)

    if pad:
        # Calculate the padded size
        ny = _ceil(m, sy)
        nx = _ceil(n, sx)
        size = ((ny - 1) * sy + ky, (nx - 1) * sx + kx) + mat.shape[2:]
        mat_pad = np.full(size, np.nan)
        mat_pad[:m, :n, ...] = mat
    else:
        # Ensure the matrix is large enough for the kernel if not padding
        mat_pad = mat[:(m - ky) // sy * sy + ky, :(n - kx) // sx * sx + kx, ...]

    # Create a view of the matrix with the specified stride
    view = asStride(mat_pad, ksize, stride)
    if method == 'max':
        # Perform max-pooling and convert NaNs back to zeros
        result = np.nanmax(view, axis=(2, 3))
    else:
        # Perform mean-pooling and convert NaNs back to zeros
        result = np.nanmean(view, axis=(2, 3))
    result = np.nan_to_num(result)
    return result


def f_instance2semantics_max(ins):
    """
    Processes an instance segmentation mask to remove small objects and converts it to a semantic segmentation mask.

    Args:
        ins (numpy.ndarray): The instance segmentation mask.

    Returns:
        numpy.ndarray: The semantic segmentation mask.
    """
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
    Main function to perform cell segmentation using Cellpose model.

    Args:
        file_path (str): Path to the input image file.
        gpu (int): Index of the GPU to be used.
        model_dir (str): Directory where the Cellpose model is stored.
        model_name (str, optional): Name of the model to be used. Defaults to 'cyto2'.
        output_path (str, optional): Path to save the output file. Defaults to None.
        photo_size (int, optional): Size of the patches to be used for segmentation. Defaults to 2048.
        photo_step (int, optional): Step size for patch extraction. Defaults to 2000.

    Returns:
        np.ndarray: Segmented cell mask as a numpy array.
    """
    os.environ['CELLPOSE_LOCAL_MODELS_PATH'] = model_dir  # Set the path for Cellpose to find the model
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
    import logging
    overlap = photo_size - photo_step
    if (overlap % 2) == 1:
        overlap = overlap + 1
    act_step = ceil(overlap / 2)
    logging.getLogger('cellpose.models').setLevel(logging.WARNING)
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


def segment4cell(input_path: str, cfg: CellSegParam, gpu: int) -> npt.NDArray[np.uint8]:
    mask = main(
        file_path=input_path,
        gpu=gpu,
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
