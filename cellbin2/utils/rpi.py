import os
import math

import h5py
import tqdm
from pathlib import Path
from typing import Union, Dict, Tuple, Type, Any
import numpy as np
import numpy.typing as npt
from objtyping import objtyping
import cv2

from cellbin2.image.augmentation import find_thresh, f_rgb2gray
from cellbin2.utils import HDF5, clog
from cellbin2.image import cbimread, CBImage


def get_tissue_mask(mask: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    mask[mask > 0] = 255
    return mask


def get_cell_outline(mask: npt.NDArray[np.uint8], line_width=1) -> npt.NDArray[np.uint8]:
    # TODO: official product -> line_width = 2
    image = np.where(mask != 0, 1, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))  # elliptical structure
    image = cv2.morphologyEx(image, cv2.MORPH_ERODE, kernel, iterations=None)
    edge = np.zeros((image.shape), dtype=np.uint8)
    contours, hierachy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    r = cv2.drawContours(edge, contours, -1, (255, 255, 255), line_width)
    return r


class ChannelImage(object):
    def __init__(self):
        # self.CellMaskRaw = Image(TrackLayer=False, MinGrayLevel=0, MaxGrayLevel=255)
        # self.Image = Image(TrackLayer=True)
        # self.TissueMaskRaw = Image(TrackLayer=False, MinGrayLevel=0, MaxGrayLevel=255)
        pass


class Bin(object):
    def __init__(self):
        self.XimageNumber: int = 0
        self.YimageNumber: int = 0
        self.sizex: int = 0
        self.sizey: int = 0

    def split(self, mat: np.ndarray, img_siz=256):
        self.sizey, self.sizex = mat.shape[:2]
        self.XimageNumber, self.YimageNumber = [math.ceil(self.sizex / img_siz), math.ceil(self.sizey / img_siz)]

        for x in range(self.XimageNumber):
            for y in range(self.YimageNumber):
                # deal with last row/column images
                x_end = min(((x + 1) * img_siz), self.sizex)
                y_end = min(((y + 1) * img_siz), self.sizey)
                if mat.ndim == 3:
                    small_im = mat[y * img_siz:y_end, x * img_siz:x_end, :]
                else:
                    small_im = mat[y * img_siz:y_end, x * img_siz:x_end]
                setattr(self, f'{x}/{y}', small_im)


class Image(object):
    def __init__(self, color=(255, 255, 255), GrayLevelElbow=0, MaxGrayLevel=255, MinGrayLevel=0, TrackLayer=False):
        self.Color = color
        self.GrayLevelElbow = GrayLevelElbow
        self.MaxGrayLevel = MaxGrayLevel
        self.MinGrayLevel = MinGrayLevel
        self.TrackLayer = TrackLayer

    def pyramid(self, mat: CBImage, mag=(1, 10, 50, 100, 150)):
        # TODO: official product -> mag=(2, 10, 50, 100, 150)
        for bin_siz in tqdm.tqdm(mag, desc='pyramid', file=clog.tqdm_out, mininterval=3):
            down_image = mat.image[::bin_siz, ::bin_siz]
            b = Bin()
            b.split(down_image)
            setattr(self, 'bin_{}'.format(bin_siz), b)


class MetaInfo(object):
    def __init__(self):
        self.imgSize: int = 256
        self.sizex: int = 0
        self.sizey: int = 0
        self.version: str = '0.0.2'
        self.x_start: int = 0
        self.y_start: int = 0


class RecordPyramidImage(HDF5):
    def __init__(self):
        super(RecordPyramidImage, self).__init__()
        self.metaInfo = MetaInfo()

    def create(self, data: Dict[str, Dict[str, str]]):
        for channel_name, channel_path in data.items():
            channel_data = ChannelImage()
            width, height = 0, 0
            for im_name, im_path in channel_path.items():
                if not os.path.exists(im_path):
                    continue
                mat = cbimread(im_path)
                width, height = mat.width, mat.height
                clog.info(f'Create pyramid for {im_path}')
                if 'tissue' in im_name.lower():
                    setattr(channel_data, im_name, Image(TrackLayer=False, MinGrayLevel=0, MaxGrayLevel=255))
                    im = cbimread(get_tissue_mask(mat.image))
                elif "cell" in im_name.lower():
                    setattr(channel_data, im_name, Image(TrackLayer=False, MinGrayLevel=0, MaxGrayLevel=255))
                    im = cbimread(get_cell_outline(mat.image))
                else:
                    setattr(channel_data, im_name, Image(TrackLayer=True, MinGrayLevel=0, MaxGrayLevel=255))
                    if mat.channel == 3:
                        channel_data.Image.MinGrayLevel, channel_data.Image.MaxGrayLevel = 0, 0
                    else:
                        channel_data.Image.MinGrayLevel, channel_data.Image.MaxGrayLevel = find_thresh(mat.image)
                    im = mat

                getattr(channel_data, im_name).pyramid(im)
            setattr(self, channel_name, channel_data)

            if width > 0 and height > 0:
                self.metaInfo.sizex = width
                self.metaInfo.sizey = height

def read(h5_path: Union[str, Path]) -> Tuple[Type[RecordPyramidImage], Dict[Any, Type[ChannelImage]]]:
    """
    :param h5_path: local rpi file path
    :return:
    """

    from cellbin2.utils import h52dict

    dct = {}
    with h5py.File(h5_path, 'r') as fd:
        h52dict(fd, dct)
    rpi_dct = {}
    image_dct = {}

    for k, v in dct.items():
        if k in ['metaInfo']:
            rpi_dct[k] = v
        else:
            image_dct[k] = objtyping.from_primitive(v, ChannelImage)
    ipr = objtyping.from_primitive(rpi_dct, RecordPyramidImage)

    return ipr, image_dct


def write(h5_path: Union[str, Path], extra_images: Dict[str, Dict[str, str]]):
    """
    :param h5_path: local path for saving rpi file 
    :param extra_images: image set path
    :return:
    """
    if not isinstance(h5_path, Path):
        h5_path = Path(h5_path)
    assert h5_path.name.endswith('.rpi'), '{}, expected file suffix is .rpi'.format(os.path.basename(h5_path))

    rpi = RecordPyramidImage()
    rpi.create(extra_images)
    rpi.write(h5_path, extra_images)
    return 0


def readrpi(h5, bin_size, staintype='ssDNA', tType="Image"):
    """ Merge image patches back to large image. """
    # h5 = h5py.File(h5_path, 'r')
    # get attributes
    imgSize = h5['metaInfo'].attrs['imgSize']
    group = h5[staintype][tType][r"bin_%d" % bin_size]
    width = group.attrs['sizex']
    height = group.attrs['sizey']
    # initialize image
    arr = group[f'{0}/{0}'][()]
    if len(arr.shape) != 3:
        im = np.zeros((height, width), dtype=group['0/0'][()].dtype)
    else:
        im = np.zeros((height, width, 3), dtype=group['0/0'][()].dtype)
    # recontruct image
    for i in range(group.attrs['XimageNumber']):
        for j in range(group.attrs['YimageNumber']):
            small_im = group[f'{i}/{j}'][()]
            x_end = min(((i + 1) * imgSize), width)
            y_end = min(((j + 1) * imgSize), height)
            im[j * imgSize:y_end, i * imgSize:x_end] = small_im
    return im


def main():
    import argparse
    import json
    from os.path import basename
    demo_json = """
    {
        "DAPI": { 
            "CellMask": "A02677B5/A02677B5_DAPI_mask.tif",
            "TissueMask": "A02677B5/A02677B5_DAPI_tissue_cut.tif"
        }
    }
    """
    usage = f"python {basename(__file__)} -i JSON_FILE -o OUTPUT_PATH"
    description = usage + "\n" + f"A demo -i input should be like: {demo_json}"
    parser = argparse.ArgumentParser(
        usage=description
    )
    parser.add_argument("-i", dest="data", help="The path of json file.", metavar="JSON FILE")
    parser.add_argument("-o", action="store", type=str, required=True, metavar="OUTPUT_PATH",
                        help="The results output path.", dest="output_path")
    args = parser.parse_args()
    data = args.data
    with open(data, 'r', encoding='utf-8') as f:
        data = json.load(f)
    write(h5_path=args.output_path, extra_images=data)


if __name__ == '__main__':
    main()
