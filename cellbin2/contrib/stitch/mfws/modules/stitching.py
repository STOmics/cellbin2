import numpy as np
import os
import time
import tifffile
import cv2
import glog

from .wsi_stitch import StitchingWSI
from .fov_aligner import FOVAligner
from .global_location import GlobalLocation
from .image import Image


class Stitching:
    """
    The splicing module class includes splicing coordinate calculation and template derivation
    """
    def __init__(self, is_stitched=False):
        """
        Args:
            is_stitched: True | False Used to determine whether it is a spliced large image

        Returns:

        """
        self.fov_location = None
        self.fov_x_jitter = None
        self.fov_y_jitter = None
        self.rows = self.cols = None
        self._fov_height = self._fov_width = self._fov_channel = None
        self._fov_dtype = None
        self.height = self.width = None
        self._overlap_x = self._overlap_y = 0.1

        self.stitch_method = 'cd'
        self.debug = False
        self.output_path = ''

        self.__jitter_diff = None
        self.__jitter_x_diff = None
        self.__jitter_y_diff = None
        self.__offset_diff = None
        self.__template_max_value = -1
        self.__template_mean_value = -1
        self.__template_std_value = -1
        self.__template_qc_conf = -1
        self.__template_re_conf = -1
        self.__template_global_diff = -1  # @lizepeng update on 2023/05/17
        self.__image = None
        self._set_location_flag = False
        self._is_stitched = is_stitched
        self._template = None

    def _init_parm(self, src_image: dict):
        test_image_path = list(src_image.values())[0]
        test_image_path = test_image_path.image

        if isinstance(test_image_path, str) or \
                (isinstance(test_image_path, np.ndarray) and len(test_image_path) != 4):
            img = Image()
            img.read(test_image_path)

            self._fov_height = img.height
            self._fov_width = img.width
            self._fov_channel = img.channel
            self._fov_dtype = img.dtype
        else:
            self._fov_height = test_image_path[1]
            self._fov_width = test_image_path[3]

    def set_overlap(self, overlap):
        if isinstance(overlap, float):
            self._overlap_x = self._overlap_y = overlap
        elif isinstance(overlap, (list, tuple)):
            self._overlap_x, self._overlap_y = overlap
        else:
            raise ValueError("Overlap setting error.")

    def set_size(self, rows, cols):
        """

        Args:
            rows:
            cols:

        Returns:

        """
        try:
            rows = int(rows)
        except ValueError:
            glog.error("Rows type error.")

        try:
            cols = int(cols)
        except ValueError:
            glog.error("Cols type error.")

        self.rows = rows
        self.cols = cols

    def set_global_location(self, loc):
        """

        Args:
            loc:Splicing coordinate information

        Returns:

        """
        assert type(loc) == np.ndarray, "Location type error."
        self.fov_location = loc
        self._set_location_flag = True

    def set_jitter(self, h_j, v_j):
        """

        Args:
            h_j: Horizontal offset matrix
            v_j: Vertical offset matrix

        Returns:

        """
        if h_j is not None and v_j is not None:
            assert h_j.shape == v_j.shape, "Jitter ndim is diffient"
            self.fov_x_jitter = h_j
            self.fov_y_jitter = v_j

    def _get_jitter(self, src_fovs, fft_channel=0, process=5):
        """
        Calculate FFT feature offset
        Args:
            src_fovs: {'row_col':'image_path'}
            fft_channel: Calculate FFT feature offset
        """
        jitter_model = FOVAligner(src_fovs,
                                  self.rows, self.cols,
                                  multi=True, channel=fft_channel,
                                  overlap=[self._overlap_x, self._overlap_y],
                                  i_shape=[self._fov_height, self._fov_width]
                                  )
        jitter_model.set_process(process)
        if self.fov_x_jitter is None or self.fov_y_jitter is None:
            start_time = time.time()
            glog.info('Start jitter mode.')
            jitter_model.create_jitter()
            self.fov_x_jitter = jitter_model.horizontal_jitter
            self.fov_y_jitter = jitter_model.vertical_jitter

            if np.max(self.fov_x_jitter) == np.min(self.fov_x_jitter):
                self.fov_x_jitter = np.zeros_like(self.fov_x_jitter) - 999
                self.fov_x_jitter[:, 1:, 0] = - int(self._fov_width * self._overlap_x)
                self.fov_x_jitter[:, 1:, 1] = 0
            if np.max(self.fov_y_jitter) == np.min(self.fov_y_jitter):
                self.fov_y_jitter = np.zeros_like(self.fov_y_jitter) - 999
                self.fov_y_jitter[1:, :, 0] = 0
                self.fov_y_jitter[1:, :, 1] = - int(self._fov_height * self._overlap_y)

            self.__jitter_diff, self.__jitter_x_diff, self.__jitter_y_diff = \
                jitter_model.offset_eval(self._fov_height, self._fov_width, [self._overlap_x, self._overlap_y])
            end_time = time.time()
            glog.info("Caculate jitter time -- {}s".format(end_time - start_time))
        else:
            jitter_model.horizontal_jitter = self.fov_x_jitter
            jitter_model.vertical_jitter = self.fov_y_jitter
            self.__jitter_diff, self.__jitter_x_diff, self.__jitter_y_diff = \
                jitter_model.offset_eval(self._fov_height, self._fov_width, [self._overlap_x, self._overlap_y])
            glog.info("Have jitter matrixs, skip this mode.")

        if self.debug and os.path.isdir(self.output_path):
            np.save(os.path.join(self.output_path, 'fov_x_jitter.npy'), self.fov_x_jitter)
            np.save(os.path.join(self.output_path, 'fov_y_jitter.npy'), self.fov_y_jitter)

    def _get_location(self, ):
        """
        Returns:
        """
        if self.fov_location is None:
            start_time = time.time()
            glog.info('Start location mode.')
            location_model = GlobalLocation()
            location_model.set_size(self.rows, self.cols)
            location_model.set_image_shape(self._fov_height, self._fov_width)
            location_model.set_jitter(self.fov_x_jitter, self.fov_y_jitter)
            # 'cd' 'LS-H' 'LS-V'
            location_model.set_overlap(self._overlap_x, self._overlap_y)
            location_model.create_location(self.stitch_method)
            self.fov_location = location_model.fov_loc_array
            self.__offset_diff = location_model.offset_diff
            end_time = time.time()
            glog.info("Caculate location time -- {}s".format(end_time - start_time))
        else:
            glog.info("Have location coord, skip this mode.")

        if self.debug and os.path.isdir(self.output_path):
            np.save(os.path.join(self.output_path, 'fov_location.npy'), self.fov_location)

    @staticmethod
    def _get_fovs_by_rule(
            src_fovs: dict,
            stitch_rule: dict = None,
    ) -> dict:
        """

        Args:
            src_fovs:
            stitch_rule:
                -- {'0000_0001': {'flip': 0, 'rot': 0}, ...}

        Returns:

        """
        if isinstance(list(src_fovs.items())[0][1], ImageBase):
            return src_fovs

        new_src_fovs = dict()
        if stitch_rule is None:
            for k, v in src_fovs.items():
                img = ImageBase(v)
                new_src_fovs[k] = img
        else:
            assert isinstance(stitch_rule, dict), "'stitch_rule' must be a dict."
            for k, v in stitch_rule.items():
                flip = v.get('flip', 0)
                rot = v.get('rot', 0)
                fov = src_fovs[k]
                if flip == 1: img = ImageBase(fov, flip_lr = True, rot=rot)
                elif flip == 2: img = ImageBase(fov, flip_ud = True, rot=rot)
                else: img = ImageBase(fov, rot=rot)
                new_src_fovs[k] = img

        return new_src_fovs

    def stitch(
            self,
            src_fovs: dict,
            stitch=True,
            process_rule: dict = None,
            fft_channel=0,
            fuse_flag=True,
            down_size=1.0,
            stitch_method='cd',
            debug: bool = False,
            output_path: str = ''
    ):
        """
        Args:
            src_fovs: {'row_col':'image_path'}
            stitch: Do you want to stitch the images together
            process_rule: Splicing rules
            fft_channel: Select FFT feature channel
            fuse_flag:
            down_size:
            stitch_method: 'cd' | 'LS-V' | 'LS-H'
            debug:
            output_path:
        """
        self.stitch_method = stitch_method
        self.debug = debug
        self.output_path = output_path

        src_fovs = self._get_fovs_by_rule(src_fovs, process_rule)

        self._init_parm(src_fovs)

        if not self._is_stitched:
            if not self._set_location_flag:

                self._get_jitter(src_fovs=src_fovs, fft_channel=fft_channel)

                self._get_location()

            # stitch
            if stitch:
                start_time = time.time()
                glog.info('Start stitch mode.')
                wsi = StitchingWSI()
                wsi.set_overlap([self._overlap_x, self._overlap_y])
                wsi.mosaic(
                    src_fovs,
                    self.fov_location,
                    multi=False,
                    fuse_flag=fuse_flag,
                    down_sample=down_size
                )
                end_time = time.time()
                glog.info("Stitch image time -- {}s".format(end_time - start_time))

                self.__image = wsi.buffer
        else:
            glog.info("Image is stitched, skip all stitch operations.")

    def get_image(self):
        return self.__image

    def get_all_eval(self):
        """

        Returns: Dict Various evaluation indicators

        """
        eval = dict()

        if not self._is_stitched:
            eval['stitch_diff'] = self.__offset_diff
            eval['jitter_diff'] = self.__jitter_diff

            eval['stitch_diff_max'] = np.max(self.__offset_diff)
            eval['jitter_diff_max'] = np.max(self.__jitter_diff)

        eval['template_max'] = self.__template_max_value
        eval['template_mean'] = self.__template_mean_value
        eval['template_std'] = self.__template_std_value
        eval['template_qc_conf'] = self.__template_qc_conf
        eval['template_re_conf'] = self.__template_re_conf

        return eval

    def get_jitter(self):
        """

        Returns:

        """
        return self.fov_x_jitter, self.fov_y_jitter

    def get_template_global_eval(self):
        """

        Returns: Array Matrix for template evaluation

        """
        return self.__template_global_diff

    def get_jitter_eval(self):
        """
        Returns: Splicing offset heatmap display, microscope splicing
        """
        if not self._is_stitched:
            return self.__jitter_x_diff, self.__jitter_y_diff
        return -1, -1

    def get_stitch_eval(self, ):
        """
        Returns: Splicing offset heatmap display, self-developed splicing
        """
        jitter_x_diff = np.zeros([self.rows, self.cols]) - 1
        jitter_y_diff = np.zeros([self.rows, self.cols]) - 1

        for row in range(self.rows):
            for col in range(self.cols):
                if row > 0 and self.fov_y_jitter[row, col, 0] != 999:
                    _jit = self.fov_location[row, col] - self.fov_location[row - 1, col] - [0, self._fov_height]
                    _dif = _jit - self.fov_y_jitter[row, col]
                    jitter_y_diff[row, col] = (_dif[0] ** 2 + _dif[1] ** 2) ** 0.5

                if col > 0 and self.fov_x_jitter[row, col, 0] != 999:
                    _jit = self.fov_location[row, col] - self.fov_location[row, col - 1] - [self._fov_width, 0]
                    _dif = _jit - self.fov_x_jitter[row, col]
                    jitter_x_diff[row, col] = (_dif[0] ** 2 + _dif[1] ** 2) ** 0.5

        return jitter_x_diff, jitter_y_diff

    def mosaic(self, ):
        pass


class ImageBase:
    """

    """
    def __init__(self, image, **kwargs):
        self.image = image

        self.flip_ud = kwargs.get('flip_ud', False)
        self.flip_lr = kwargs.get('flip_lr', False)
        self.rot = kwargs.get('rot', 0)

        self.to_gray = kwargs.get('to_gray', False)

    def get_image(self):
        if isinstance(self.image, str):
            _image = cv2.imread(self.image, -1)
        else: _image = self.image.copy()

        if self.flip_ud:
            _image = _image[::-1, :]

        if self.flip_lr:
            _image = _image[:, ::-1]

        if self.rot != 0:
            pass

        if self.to_gray:
            pass

        return _image
