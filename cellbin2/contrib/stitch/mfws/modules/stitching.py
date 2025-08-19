# -*- coding: utf-8 -*-
import time
import glog

import numpy as np
import tifffile as tif

from typing import Union

from .wsi_stitch import StitchingWSI
from .fov_aligner import FOVAligner
from .global_location import GlobalLocation
from .scan_method import ImageBase


class Stitching(object):
    def __init__(
            self,
            rows,
            cols,
            start_row: int = 1,
            start_col: int = 1,
            end_row: int = -1,
            end_col: int = -1,
            overlap_x: float = 0.1,
            overlap_y: float = 0.1,
            channel: int = 0,
            proc_count: int = 1,
            fusion: int = 0,
            down_sample: int = 1,
            **kwargs
    ):
        """
        Args:
            start_row:
            start_col:
            end_row:
            end_col:
            channel:
            overlap_x:
            overlap_y:
            proc_count:
            fusion:
        """
        # input parameters
        self.start_ind = [start_row, start_col]
        self.end_ind = [end_row, end_col]

        self.overlap = [overlap_x, overlap_y]

        self.channel = channel
        self.proc_count = proc_count

        self.fusion = fusion
        self.down_sample = down_sample

        # internal parameters
        self._fov_location: Union[None, np.ndarray] = None
        self._fov_x_jitter: Union[None, np.ndarray] = None
        self._fov_y_jitter: Union[None, np.ndarray] = None

        self._rows = rows
        self._cols = cols

        self._slice_rows, self._slice_cols = self._rows, self._cols

        self._fov_height = self._fov_width = self._fov_channel = None
        self._fov_dtype = None

        self._mosaic_height = self._mosaic_width = None

        # control parameters

    def set_location(self, loc: np.ndarray):
        """
        Args:
            loc:

        Returns:
        """
        self._fov_location = loc

    def set_jitter(self, x_jitter: np.ndarray, y_jitter: np.ndarray):
        """
        Args:
            x_jitter:
            y_jitter:

        Returns:
        """
        self._fov_x_jitter = x_jitter
        self._fov_y_jitter = y_jitter

    def _init_param(self, image_dict: dict):
        """
        Args:
            image_dict:

        Returns:
        """
        _image_path = list(image_dict.values())[0]

        # TODO 可调用其他解析方式
        if isinstance(_image_path, str):
            img = tif.imread(_image_path)
        elif isinstance(_image_path, ImageBase):
            img = _image_path.get_image()

        self._fov_dtype = img.dtype
        self._fov_height, self._fov_width = img.shape[:2]
        self._fov_channel = img.shape[2] if len(img.shape) > 2 else 1

    def _get_loc_by_mfws(self, image_dict: dict, method: str = 'cd'):
        """
        Returns:
        """
        if self._fov_x_jitter is None and self._fov_y_jitter is None:
            glog.info('No jitter information, calculate using algorithms.')
            self._get_jitter(
                image_dict,
                fft_channel=int(self.channel),
                multi=self.proc_count
            )

        start_time = time.time()
        glog.info('Start location mode')

        lm = GlobalLocation()
        lm.set_size(self._slice_rows, self._slice_cols)
        lm.set_image_shape(self._fov_height, self._fov_width)
        lm.set_jitter(self._fov_x_jitter, self._fov_y_jitter)
        lm.set_overlap(self.overlap[0], self.overlap[1])

        lm.create_location(method)

        self._fov_location = lm.fov_loc_array

        glog.info("location calculation time, {}s".format(round(time.time() - start_time, 2)))

    def _get_jitter(self, image_dict: dict,  fft_channel: int = 0, multi=5):
        """
        Args:
            image_dict:
            fft_channel:
            multi:

        Returns:
        """
        jm = FOVAligner(
            image_dict,
            self._slice_rows, self._slice_cols,
            multi=True if multi > 1 else False, channel=fft_channel,
            overlap=self.overlap,
            i_shape=[self._fov_height, self._fov_width]
        )
        # TODO
        jm.set_process(multi)
        start_time = time.time()
        glog.info("Start jitter mode.")
        jm.create_jitter()
        self._fov_x_jitter = jm.horizontal_jitter
        self._fov_y_jitter = jm.vertical_jitter

        if np.max(self._fov_x_jitter) == np.min(self._fov_x_jitter):
            self._fov_x_jitter = np.zeros_like(self._fov_x_jitter) - 999
            self._fov_x_jitter[:, 1:, 0] = - int(self._fov_width * self.overlap[0])
            self._fov_x_jitter[:, 1:, 1] = 0
        if np.max(self._fov_y_jitter) == np.min(self._fov_y_jitter):
            self._fov_y_jitter = np.zeros_like(self._fov_y_jitter) - 999
            self._fov_y_jitter[1:, :, 0] = 0
            self._fov_y_jitter[1:, :, 1] = - int(self._fov_height * self.overlap[1])

        glog.info("Jitter calculation time, {}s".format(round(time.time() - start_time), 2))

    def _slice_images(self, image_dict: dict):
        """
        Args:
            image_dict:

        Returns:
        """
        if self.end_ind[0] == -1:
            self.end_ind[0] = self._rows
        if self.end_ind[1] == -1:
            self.end_ind[1] = self._cols

        assert self.start_ind[0] <= self.end_ind[0] and self.start_ind[1] <= self.end_ind[1], \
            "Invalid slice index."

        new_image_dict = dict()
        for k, v in image_dict.items():
            r, c = map(int, k.split('_'))
            if self.start_ind[0] - 1 <= r <= self.end_ind[0] - 1 \
                    and self.start_ind[1] - 1 <= c <= self.end_ind[1] - 1:
                _r = r - self.start_ind[0] + 1
                _c = c - self.start_ind[1] + 1
                new_image_dict[f"{_r:04}_{_c:04}"] = v

        self._slice_rows = self.end_ind[0] - self.start_ind[0] + 1
        self._slice_cols = self.end_ind[1] - self.start_ind[1] + 1

        return new_image_dict

    def stitch_by_rule(self, image_dict: dict):
        """ stitching based on hardware information """
        self._init_param(image_dict)
        image_dict = self._slice_images(image_dict)

        loc = self.create_loc(
            self._slice_rows,
            self._slice_cols,
            (self._fov_height, self._fov_width),
            self.overlap
        )

        self._fov_location = loc

        img = self.stitch_by_location(image_dict, loc)
        return img

    def stitch_by_mfws(self, image_dict: dict):
        """ stitching by algorithm """
        self._init_param(image_dict)
        image_dict = self._slice_images(image_dict)

        self._get_loc_by_mfws(image_dict)
        img = self.stitch_by_location(image_dict, self._fov_location)

        return img

    def stitch_by_location(self, image_dict: dict, loc: np.ndarray):
        """ stitching based on coordinates, channel multiplexing in multi-channel scenarios """
        self._init_param(image_dict)

        wsi = StitchingWSI()
        wsi.set_overlap(self.overlap)
        wsi.mosaic(
            image_dict,
            loc,
            multi=False,
            fuse_flag=True if self.fusion else False,
            down_sample=self.down_sample
        )

        return wsi.buffer

    def export_mosaic(self, output: str):
        """ Save mosaic/thumbnail (adaptive magnification) """
        pass

    @staticmethod
    def create_loc(rows, cols, shape, overlap):
        height, width = shape
        overlap_x, overlap_y = overlap
        fov_loc = np.zeros((rows, cols, 2), dtype=int)
        for i in range(rows):
            for j in range(cols):
                fov_loc[i, j, 0] = j * (width - int(width * overlap_x))
                fov_loc[i, j, 1] = i * (height - int(height * overlap_y))

        return fov_loc

    @property
    def fov_location(self,):
        """ Global stitching coordinates """
        return self._fov_location

    @property
    def x_jitter(self,):
        """ Jitter results in the horizontal scanning direction """
        return self._fov_x_jitter

    @property
    def y_jitter(self,):
        """ Jitter results in the vertical scanning direction """
        return self._fov_y_jitter

    @property
    def mosaic_size(self,):
        """ Original resolution mosaic size """
        return self._mosaic_height, self._mosaic_width
