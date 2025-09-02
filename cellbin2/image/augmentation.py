import copy
import numpy as np
from PIL import Image
import cv2
from numba import njit, prange, jit
import math
from skimage.exposure import equalize_adapthist
from cellbin2.utils.common import TechType


def find_thresh(img: np.ndarray):
    """
    :param img: gray image
    :type img: numpy nd array
    :return: hmin, hmax
    :rtype: int, int
    """
    if not isinstance(img, np.ndarray):
        raise Exception(f"Input must be numpy nd array")
    input_dims = len(img.shape)
    if input_dims != 2:
        raise Exception(f"Input should be 2 dimensional array, but input array is {input_dims} dimensions")

    # Constants
    limit = img.size / 10
    threshold = img.size / 5000
    n_bins = 256
    if img.dtype != 'uint8':
        bit_max = 65536
    else:
        bit_max = 256

    hist_min = np.min(img)
    hist_max = np.max(img)

    bin_size = (hist_max - hist_min) / n_bins
    hist, bins = np.histogram(img.flatten(), n_bins, [hist_min, hist_max])

    hmin = 0
    hmax = bit_max - 1

    for i in range(1, len(hist) - 1):
        count = hist[i]
        if count > limit:
            continue
        if count > threshold:
            hmin = i
            break
    for i in range(len(hist) - 1, 0, -1):
        count = hist[i]
        if count > limit:
            continue
        if count > threshold:
            hmax = i
            break

    hmin = hist_min + hmin * bin_size
    hmax = hist_min + hmax * bin_size

    hmin, hmax = int(hmin), int(hmax)

    hmin = max(0, hmin)
    hmax = min(bit_max - 1, hmax)

    if hmax > hmin:
        return hmin, hmax
    else:
        return 0, 0


def f_rgb2gray(img, need_not=False):
    """
    rgb2gray

    :param img: (CHANGE) np.array
    :param need_not: if need bitwise_not
    :return: np.array
    """
    if img.ndim == 3:
        if img.shape[0] == 3 and img.shape[1] > 3 and img.shape[2] > 3:
            img = img.transpose(1, 2, 0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if need_not:
            img = cv2.bitwise_not(img)
    return img


def f_gray2bgr(img):
    """
    gray2bgr

    :param img: (CHANGE) np.array
    :return: np.array
    """

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def f_clahe_rgb(img, kernel_size=128):
    image_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # split the image into L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(image_lab)

    # apply CLAHE to lightness channel
    clahe = cv2.createCLAHE(clipLimit=2.56, tileGridSize=(math.ceil(img.shape[0] / kernel_size),
                                                          math.ceil(img.shape[1] / kernel_size)))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L channel with the original A and B channel
    merged_channels = cv2.merge((cl, a_channel, b_channel))

    # convert iamge from LAB color model back to RGB color model
    final_image = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)
    return final_image


def f_rgb2hsv(img, channel=-1, need_not=False):
    """
    rgb2hsv

    :param img: (CHANGE) np.array
    :param need_not: if need bitwise_not
    :param channel:
    :return: np.array
    """
    if img.ndim == 3:
        if img.shape[0] == 3 and img.shape[1] > 3 and img.shape[2] > 3:
            img = img.transpose(1, 2, 0)

        img = f_ij_16_to_8(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        if channel > -1:
            img = img[:, :, channel]

        if need_not:
            img = cv2.bitwise_not(img)

        img = img.astype(np.float32)
        img = img * ((img / np.mean(img)) ** 2)
        img[img > 255] = 255

    return img.astype(np.uint8)


def f_padding(img, top, bot, left, right, mode='constant', value=0):
    """
    update by dengzhonghan on 2023/2/23
    1. support 3d array padding.
    2. not support 1d array padding.

    Args:
        img (): numpy ndarray (2D or 3D).
        top (): number of values padded to the top direction.
        bot (): number of values padded to the bottom direction.
        left (): number of values padded to the left direction.
        right (): number of values padded to the right direction.
        mode (): padding mode in numpy, default is constant.
        value (): constant value when using constant mode, default is 0.

    Returns:
        pad_img: padded image.

    """

    if mode == 'constant':
        if img.ndim == 2:
            pad_img = np.pad(img, ((top, bot), (left, right)), mode, constant_values=value)
        elif img.ndim == 3:
            pad_img = np.pad(img, ((top, bot), (left, right), (0, 0)), mode, constant_values=value)
    else:
        if img.ndim == 2:
            pad_img = np.pad(img, ((top, bot), (left, right)), mode)
        elif img.ndim == 3:
            pad_img = np.pad(img, ((top, bot), (left, right), (0, 0)), mode)
    return pad_img


def f_resize(img, shape=(1024, 2048), mode="NEAREST"):
    """
    resize img with pillow

    :param img: (CHANGE) np.array
    :param shape: tuple
    :param mode: An optional resampling filter. This can be one of Resampling.NEAREST,
     Resampling.BOX, Resampling.BILINEAR, Resampling.HAMMING, Resampling.BICUBIC or Resampling.LANCZOS.
     If the image has mode “1” or “P”, it is always set to Resampling.NEAREST.
     If the image mode specifies a number of bits, such as “I;16”, then the default filter is Resampling.NEAREST.
     Otherwise, the default filter is Resampling.BICUBIC
    :return:np.array
    """
    imode = Image.NEAREST
    if mode == "BILINEAR":
        imode = Image.BILINEAR
    elif mode == "BICUBIC":
        imode = Image.BICUBIC
    elif mode == "LANCZOS":
        imode = Image.LANCZOS
    elif mode == "HAMMING":
        imode = Image.HAMMING
    elif mode == "BOX":
        imode = Image.BOX
    if img.dtype != 'uint8':
        imode = Image.NEAREST
    img = Image.fromarray(img)
    img = img.resize((shape[1], shape[0]), resample=imode)
    img = np.array(img).astype(np.uint8)
    return img


def f_percentile_threshold(img, percentile=99.9):
    """
    Threshold an image to reduce bright spots

    :param img: (CHANGE) numpy array of image data
    :param percentile: cutoff used to threshold image
    :return: np.array: thresholded version of input image

    2023/09/20 @fxzhao Add overwrite_input parameter to save the temporary memory overhead of percentile
    """

    # non_zero_vals = img[np.nonzero(img)]
    non_zero_vals = img[img > 0]

    # only threshold if channel isn't blank
    if len(non_zero_vals) > 0:
        img_max = np.percentile(non_zero_vals, percentile, overwrite_input=True)

        # threshold values down to max
        threshold_mask = img > img_max
        img[threshold_mask] = img_max

    return img


def f_equalize_adapthist(img, kernel_size=None, enhance_times=None):
    """
    Pre-process images using Contrast Limited Adaptive
    Histogram Equalization (CLAHE).

    :param img: (CHANGE) (numpy.array): numpy array of phase image data.
    :param kernel_size: (integer): Size of kernel for CLAHE,
            defaults to 1/8 of image size.
    :param enhance_times: (integer): Number of times to apply CLAHE, defaults to 1.
    :return: numpy.array:Pre-processed image

    2023/09/20 @fxzhao replace scikit-image methods with OpenCV methods, improve computational speed and reduce memory usage
    """
    # return equalize_adapthist(img, kernel_size=kernel_size)
    if kernel_size is None:
        kernel_size = 128
    clahe = cv2.createCLAHE(clipLimit=2.56, tileGridSize=(math.ceil(img.shape[0] / kernel_size),
                                                          math.ceil(img.shape[1] / kernel_size)))
    for _ in range(enhance_times or 1):
        img = clahe.apply(img)
    return img

def f_equalize_adapthist_V2(img, kernel_size=None):    #this function is originally f_equalize_adapthist() from cellbin1.2.0.16, since there is already a function with the same name, it has been renamed to V2.
    """
    Pre-process images using Contrast Limited Adaptive
    Histogram Equalization (CLAHE).

    :param img: (CHANGE) (numpy.array): numpy array of phase image data.
    :param kernel_size: (integer): Size of kernel for CLAHE,
            defaults to 1/8 of image size.
    :return: numpy.array:Pre-processed image
    """
    return equalize_adapthist(img, kernel_size=kernel_size)

@njit(parallel=True)
def rescale_intensity_v2(img, out_range):
    imin = np.min(img)
    imax = np.max(img)
    _, omax = out_range

    for i in prange(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = ((img[i][j] - imin) / (imax - imin)) * omax
    return img


def f_histogram_normalization(img):
    """
    If one of the inputs is a constant-value array, it will
    be normalized as an array of all zeros of the same shape.

    :param img: (CHANGE) (numpy.array): numpy array of phase image data.
    :return: numpy.array:image data with dtype float32.

    2023/09/20 @fxzhao Using numba to speed up rescale_intensity method
    """

    img = img.astype('float32')
    sample_value = img[(0,) * img.ndim]
    if (img == sample_value).all():
        return np.zeros_like(img)
    # img = rescale_intensity(img, out_range=(0.0, 1.0))
    img = rescale_intensity_v2(img, out_range=(0.0, 1.0))
    return img


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


@njit
def sliding_window_sum(new_w, new_h, down_size, vision_img, sum_image):
    for _h in range(new_h):
        for _w in range(new_w):
            value = np.sum(vision_img[_h * down_size: (_h + 1) * down_size,
                           _w * down_size: (_w + 1) * down_size])
            sum_image[_h, _w] = value
    return sum_image


def enhance_vision_image(vision_img, down_size=10):
    new_h = vision_img.shape[0] // down_size
    new_w = vision_img.shape[1] // down_size
    vision_img = vision_img.astype(np.float32)
    sum_image = np.zeros([new_h, new_w], dtype=np.float32)
    sum_image = sliding_window_sum(
        new_w=new_w,
        new_h=new_h,
        down_size=down_size,
        vision_img=vision_img,
        sum_image=sum_image
    )
    sum_image = np.array(sum_image, dtype=np.uint8)
    sum_image = cv2.resize(sum_image, (vision_img.shape[1], vision_img.shape[0]))
    return sum_image


@njit(parallel=True)
def f_ij_16_to_8_v2(img):
    """
    2023/09/20 @fxzhao upgrade vesion for f_ij_16_to_8,accelarate with numba
    2023/10/16 @fxzhao support 3-channel image input 
    """

    dst = np.zeros(img.shape, np.uint8)
    p_max = np.max(img)
    p_min = np.min(img)
    scale = 256.0 / (p_max - p_min + 1)

    def process(v):
        v = np.int16(v)
        v = (v & 0xffff)
        v = v - p_min
        v = max(v, 0)
        v = v * scale + 0.5
        v = np.uint8(min(v, 255))
        return v

    if img.ndim == 3:
        for i in prange(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(img.shape[2]):
                    v = img[i][j][k]
                    dst[i][j][k] = process(v)
    else:
        for i in prange(img.shape[0]):
            for j in range(img.shape[1]):
                v = img[i][j]
                dst[i][j] = process(v)
    return dst


def enhance(arr, mode, thresh):
    """
    Only support 2D array

    Args:
        arr (): 2D numpy array
        mode (): enhance mode
        thresh (): threshold

    Returns:

    """
    data = arr.ravel()
    min_v = np.min(data)
    data_ = data[np.where(data <= thresh)]
    if len(data_) == 0:
        return 0, 0
    if mode == 'median':
        var_ = np.std(data_)
        thr = np.median(data_)
        max_v = thr + var_
    elif mode == 'hist':
        freq_count, bins = np.histogram(data_, range(min_v, int(thresh + 1)))
        count = np.sum(freq_count)
        freq = freq_count / count
        thr = bins[np.argmax(freq)]
        max_v = thr + (thr - min_v)
    else:
        raise Exception('Only support median and histogram')

    return min_v, max_v


def encode(arr, min_v, max_v):
    """
    Encode image with min and max pixel value

    Args:
        arr (): 2D numpy array
        min_v (): min value obtained from enhance method
        max_v (): max value

    Returns:
        mat: encoded mat

    """
    if min_v >= max_v:
        arr = arr.astype(np.uint8)
        return arr
    mat = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.uint8)
    v_w = max_v - min_v
    mat[arr < min_v] = 0
    mat[arr > max_v] = 255
    pos = (arr >= min_v) & (arr <= max_v)
    mat[pos] = (arr[pos] - min_v) * (255 / v_w)
    return mat


def f_ij_auto_contrast_v2(img):
    limit = img.size / 10
    threshold = img.size / 5000
    if img.dtype != 'uint8':
        bit_max = 65536
    else:
        bit_max = 256
    hist_min = img.min()
    hist_max = img.max()
    n_bins = 256
    bin_size = (hist_max - hist_min) / n_bins
    hist, _ = np.histogram(img.flatten(), n_bins, [hist_min, hist_max])
    hmin = 0
    hmax = bit_max - 1
    for i in range(1, len(hist) - 1):
        count = hist[i]
        if count > limit:
            continue
        if count > threshold:
            hmin = i
            break
    for i in range(len(hist) - 1, 0, -1):
        count = hist[i]
        if count > limit:
            continue
        if count > threshold:
            hmax = i
            break
    if hmax > hmin:
        hmin = hist_min + hmin * bin_size
        hmax = hist_min + hmax * bin_size
        # hmax = int(hmax * bit_max / 256)
        # hmin = int(hmin * bit_max / 256)
        img[img < hmin] = hmin
        img[img > hmax] = hmax
        cv2.normalize(img, img, 0, bit_max - 1, cv2.NORM_MINMAX)
    return img


def f_ij_auto_contrast_v3(img):
    limit = img.size / 10
    threshold = img.size / 5000

    if img.dtype != 'uint8':
        bit_max = 65536
    else:
        bit_max = 256
    hist, _ = np.histogram(img.flatten(), bit_max, [0, bit_max])
    hmin = 0
    hmax = bit_max - 1
    for i in range(1, bit_max - 1):
        # for i in range(np.min(img) + 1, np.max(img)):
        count = hist[i]
        if count > limit:
            continue
        if count > threshold:
            hmin = i
            break
    for i in range(bit_max - 2, 0, -1):
        # for j in range(np.max(img) - 1, 0, -1):
        count = hist[i]
        if count > limit:
            continue
        if count > threshold:
            hmax = i
            break
    dst = copy.deepcopy(img)
    if hmax > hmin:
        # hmin = max(0, hmin - 30)
        # hmax = int(hmax * bit_max / 256)
        # hmin = int(hmin * bit_max / 256)
        dst[dst < hmin] = hmin
        dst[dst > hmax] = hmax
        # cv2.normalize(dst, dst, 0, bit_max - 1, cv2.NORM_MINMAX)
        if bit_max == 256:
            dst = np.uint8((dst - hmin) / (hmax - hmin) * (bit_max - 1))
        elif bit_max == 65536:
            # dst = np.uint16((dst - hmin) / (hmax - hmin) * (bit_max - 1))
            dst = np.uint8((dst - hmin) / (hmax - hmin) * 255)
    return dst


def f_ij_auto_contrast(img):
    """
        auto contrast from imagej
        Args:
            img(ndarray): img array

        Returns(ndarray):img array

        """
    limit = img.size / 10
    threshold = img.size / 5000
    if img.dtype != 'uint8':
        bit_max = 65536
    else:
        bit_max = 256
    hist, _ = np.histogram(img.flatten(), 256, [0, bit_max])
    hmin = 0
    hmax = bit_max - 1
    for i in range(1, len(hist) - 1):
        count = hist[i]
        if count > limit:
            continue
        if count > threshold:
            hmin = i
            break
    for i in range(len(hist) - 2, 0, -1):
        count = hist[i]
        if count > limit:
            continue
        if count > threshold:
            hmax = i
            break
    if hmax > hmin:
        hmax = int(hmax * bit_max / 256)
        hmin = int(hmin * bit_max / 256)
        img[img < hmin] = hmin
        img[img > hmax] = hmax
        cv2.normalize(img, img, 0, bit_max - 1, cv2.NORM_MINMAX)
    return img


def dapi_enhance(img_obj):
    """
    if you implement a new enhance method, the returned arr must be in bgr format

    Args:
        img_obj ():

    Returns:
        bgr_arr: numpy array in bgr format

    """
    depth = img_obj.depth
    th = int((1 << depth) * (1 - 0.618))
    arr = img_obj.image  # 2d array
    try:
        min_v, max_v = enhance(arr, mode='hist', thresh=th)
    except:
        min_v, max_v = enhance(arr, mode='median', thresh=th)
    enhance_arr = encode(arr, min_v, max_v)  # 2d array
    bgr_arr = f_gray2bgr(enhance_arr)  # 3d array in bgr format
    return bgr_arr


def he_enhance(img_obj):
    """
    add by @limin on 2023/05/15

    """
    arr = img_obj.image
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)  # todo: rgb or bgr
    if arr.dtype == 'uint16':
        arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    MAX_RANGE = np.power(2, 8)
    arr_invert = (MAX_RANGE - arr).astype(np.uint8)
    arr_invert = cv2.equalizeHist(arr_invert)
    bgr_arr = f_gray2bgr(arr_invert)  # 3d array in bgr format
    return bgr_arr

#
# pt_enhance_method = {
#     StainType.ssDNA.value: dapi_enhance,
#     StainType.DAPI.value: dapi_enhance,
#     StainType.HE.value: he_enhance
# }
#
line_enhance_method = {
    TechType.HE: he_enhance
}
#
# clarity_enhance_method = {
#     StainType.HE.value: f_rgb2gray
# }
