import numpy as np
import copy
import math
import tqdm
import cv2 as cv
import glog

from .image import Image


def rc_key(row: int, col: int):
    return '{}_{}'.format(str(row).zfill(4), str(col).zfill(4))


class StitchingWSI(object):
    def __init__(self, ):
        self.fov_rows = None
        self.fov_cols = None
        self.fov_height = self.fov_width = 0
        self.fov_channel = self.fov_dtype = 0
        self.fov_location = None
        self._overlap_x = self._overlap_y = 0.1
        self.buffer = None
        self.mosaic_width = self.mosaic_height = None
        self._fuse_size = 50

    def set_overlap(self, overlap):
        self._overlap_x, self._overlap_y = overlap

    def set_fuse_size(self, fuse_size):
        self._fuse_size = fuse_size

    def _init_parm(self, src_image: dict):
        test_image_path = list(src_image.values())[0]
        test_image_path = test_image_path.image
        img = Image()
        img.read(test_image_path)

        self.fov_height = img.height
        self.fov_width = img.width
        self.fov_channel = img.channel
        self.fov_dtype = img.dtype

    def _set_location(self, loc):
        if loc is not None:
            h, w = loc.shape[:2]
            assert (h == self.fov_rows and w == self.fov_cols)
            self.fov_location = loc
        else:
            self.fov_location = np.zeros((self.fov_rows, self.fov_cols, 2), dtype = int)
            for i in range(self.fov_rows):
                for j in range(self.fov_cols):
                    self.fov_location[i, j] = [
                        int(j * self.fov_width * (1 - self._overlap_x)),
                        int(i * self.fov_height * (1 - self._overlap_y))
                    ]

        x0 = np.min(self.fov_location[:, :, 0])
        y0 = np.min(self.fov_location[:, :, 1])
        self.fov_location[:, :, 0] -= x0
        self.fov_location[:, :, 1] -= y0
        x1 = np.max(self.fov_location[:, :, 0])
        y1 = np.max(self.fov_location[:, :, 1])
        self.mosaic_width, self.mosaic_height = [x1 + self.fov_width, y1 + self.fov_height]

    def mosaic(self, src_image: dict, loc = None, down_sample = 1, multi = False, fuse_flag = True):

        k = [i * (90 / self._fuse_size) for i in range(0, self._fuse_size)][::-1]  # fusion ratio

        self.fov_rows, self.fov_cols = loc.shape[:2]
        self._init_parm(src_image)

        self._set_location(loc)
        h, w = (int(self.mosaic_height / down_sample), int(self.mosaic_width / down_sample))
        if self.fov_channel == 1:
            self.buffer = np.zeros((h + 1, w + 1), dtype = self.fov_dtype)
        else:
            self.buffer = np.zeros((h + 1, w + 1, self.fov_channel), dtype = self.fov_dtype)

        if multi:
            pass
        else:
            for i in tqdm.tqdm(range(self.fov_rows), desc = 'FOVs Stitching', mininterval = 5):
                for j in range(self.fov_cols):

                    blend_flag_h = False
                    blend_flag_v = False

                    if rc_key(i, j) in src_image.keys():
                        img = src_image[rc_key(i, j)]
                        arr = img.get_image()

                        x, y = self.fov_location[i, j]
                        x_, y_ = (int(x / down_sample), int(y / down_sample))

                        # ------------- 融合
                        if fuse_flag:
                            if j > 0:
                                if rc_key(i, j - 1) in src_image.keys():
                                    blend_flag_h = True
                                    _, dif_h = self.fov_location[i, j, ...] - self.fov_location[i, j - 1, ...]
                                    if dif_h >= 0:
                                        source_h = copy.deepcopy(
                                            self.buffer[y:y + self.fov_height - dif_h, x:x + self._fuse_size])
                                    else:
                                        source_h = copy.deepcopy(
                                            self.buffer[y - dif_h:y + self.fov_height, x:x + self._fuse_size])

                            if i > 0:
                                if rc_key(i - 1, j) in src_image.keys():
                                    blend_flag_v = True
                                    dif_v, _ = self.fov_location[i, j, :] - self.fov_location[i - 1, j, :]
                                    if dif_v >= 0:
                                        source_v = copy.deepcopy(
                                            self.buffer[y:y + self._fuse_size, x:x + self.fov_width - dif_v])
                                    else:
                                        source_v = copy.deepcopy(
                                            self.buffer[y:y + self._fuse_size, x - dif_v:x + self.fov_width])
                        ###########
                        _h, _w = arr.shape[:2]
                        b_h, b_w = int(np.ceil(_h / down_sample)), int(np.ceil(_w / down_sample))
                        _arr = cv.resize(arr, (b_w, b_h))
                        if self.fov_channel == 1:
                            self.buffer[y_: y_ + b_h, x_: x_ + b_w] = _arr
                        else:
                            self.buffer[y_: y_ + b_h, x_: x_ + b_w, :] = _arr

                        ###########
                        if fuse_flag:
                            try:
                                if blend_flag_h:
                                    result_h, _y = self.blend_image_h(arr, source_h, x, y, dif_h, k, self._fuse_size)
                                    _h, _w = result_h.shape[:2]
                                    self.buffer[_y:_y + _h, x:x + _w, ...] = result_h

                                    if dif_h >= 0:
                                        arr[:_h, :_w] = result_h
                                    else:
                                        arr[-dif_h:, :_w] = result_h

                                if blend_flag_v:
                                    result_v, _x = self.blend_image_v(arr, source_v, x, y, dif_v, k, self._fuse_size)
                                    _h, _w = result_v.shape[:2]
                                    self.buffer[y:y + _h, _x:_x + _w] = result_v
                            except Exception as e:
                                glog.warning(f"{e}")
                                pass
                        ###########

    def save(self, output_path, compression = False):
        img = Image()
        img.image = self.buffer
        img.write(output_path, compression = compression)

    def _multi_set_index(self, k = 2):
        pass

    def _multi_set_image(self, src_image, index):
        """
        index: [start_row, start_col, end_row, end_col]
        """
        s_row, s_col, e_row, e_col = index
        for row in range(s_row, e_row):
            for col in range(s_col, e_col):
                pass

    def blend_image_h(self, mat, source, x, y, dif, k, size):

        if dif >= 0:
            temp_1 = mat[:self.fov_height - dif, :size]
            _y = y
        else:
            temp_1 = mat[-dif:, :size]
            _y = y - dif

        result = np.zeros_like(source)
        for i in range(size):
            result[:, i] = source[:, i] * math.sin(math.radians(k[i])) + temp_1[:, i] * (
                    1 - math.sin(math.radians(k[i])))
        return result, _y

    def blend_image_v(self, mat, source, x, y, dif, k, size):

        if dif >= 0:
            temp_1 = mat[:size, :self.fov_width - dif]
            _x = x
        else:
            temp_1 = mat[:size, -dif:]
            _x = x - dif

        result = np.zeros_like(source)
        for i in range(size):
            result[i, :] = source[i, :] * math.sin(math.radians(k[i])) + temp_1[i, :] * (
                    1 - math.sin(math.radians(k[i])))
        return result, _x

    def save(self, output_path, compression = False):
        img = Image()
        img.image = self.buffer
        img.write(output_path, compression = compression)


def main():
    src_image = {}
    wsi = StitchingWSI()
    wsi.set_overlap(0.1)
    wsi.mosaic(src_image)


if __name__ == '__main__': main()
