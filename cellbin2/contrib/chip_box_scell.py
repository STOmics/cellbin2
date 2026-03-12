# -*- coding: utf-8 -*-
import math
import os
import sys

import tifffile
import cv2 as cv
import numpy as np
from typing import Union

from cellbin2.utils.common import TechType
from cellbin2.contrib.alignment import ChipFeature
from cellbin2.contrib.alignment.basic import Alignment, AlignMode, ChipBoxInfo
from cellbin2.utils import clog
from cellbin2.image import CBImage
from cellbin2.matrix import box_detect as bd
from cellbin2.image import cbimread, cbimwrite
#from cellbin2.contrib.param import TemplateInfo
from cellbin2.contrib.chip_detector import ChipParam, detect_chip


class MatrixBoxDetector(object):
    def __init__(self, binsize=21, down_size=4, morph_size=9, gene_base_size=19992/2):
        self.gene_base_size = gene_base_size
        self.binsize = binsize
        self.down_size = down_size
        self.morph_size = morph_size
        self.image: np.ndarray = np.array([])

    @staticmethod
    def _walks_image(dst, box_size, begin_xy=None, end_xy=None):
        """

        Args:
            dst:
            box_size:

        Returns:

        """

        if begin_xy is not None:
            bx, by = begin_xy
        else:
            bx = by = 0

        if end_xy is not None:
            ex, ey = end_xy
        else:
            ey, ex = np.array(dst.shape) - box_size

        max_value = 0
        max_x = max_y = 0

        for i in range(by, ey):
            for j in range(bx, ex):
                _dst = dst[i: i + box_size[0], j: j + box_size[1]]
                if np.sum(_dst) > max_value:
                    max_value = np.sum(_dst)
                    max_x = j
                    max_y = i

        return max_x, max_y

    def get_box4maximize_area(self, dst, box_size,
                              min_size=256, step=5):
        """

        Args:
            dst:
            box_size:
            min_size:
            step:

        Returns:

        """
        if min(dst.shape) < min_size:

            x, y = self._walks_image(dst, box_size)

            return x, y

        else:

            _dst = cv.pyrDown(dst)

            x, y = self.get_box4maximize_area(_dst, box_size=box_size // 2)

            x, y = map(lambda k: k * 2, (x, y))

            x, y = self._walks_image(dst, box_size=box_size,
                                     begin_xy=[x - step, y - step],
                                     end_xy=[x + step, y + step])

            return x, y

    def detect(self, matrix: np.ndarray, chip_size: Union[int, float, list, tuple] = 1.0):
        """

        Args:
            matrix:
            chip_size:

        Returns:

        """
        if isinstance(chip_size, (int, float)):
            chip_size = [chip_size, chip_size]

        gene_size = np.array([int(self.gene_base_size * chip_size[0]),
                              int(self.gene_base_size * chip_size[1])])

        self.image = matrix
        gene_image = cv.filter2D(self.image, -1, np.ones((self.binsize, self.binsize), np.float32))
        _, gene_image = cv.threshold(gene_image, 0, 255, cv.THRESH_BINARY)

        gene_image_s = cv.resize(gene_image,
                                 (gene_image.shape[1] // self.down_size, gene_image.shape[0] // self.down_size))

        x, y = self.get_box4maximize_area(gene_image_s, box_size = gene_size // self.down_size)

        x = x * self.down_size + gene_size[1] / 2
        y = y * self.down_size + gene_size[0] / 2

        # x, y = map(lambda k: k * self.down_size + gene_size / 2, (x, y))
        center = [x, y]

        new_box = [[center[0] + i * gene_size[1] / 2, center[1] + j * gene_size[0] / 2]
                   for i in [-1, 1] for j in [-1, 1]]
        new_box = np.array(new_box)[(0, 1, 3, 2), :]

        return new_box


def detect_chip_box(
        matrix: np.ndarray,
        chip_size: Union[int, float, list, tuple] = 1.0
) -> ChipBoxInfo:
    """

    Args:
        matrix:
        chip_size: "A1C3" == 3 * 2

    Returns:

    """
    mbd = MatrixBoxDetector()

    # need chip size, "A1C3" == 2 * 3
    if isinstance(chip_size, (list, tuple)):
        chip_size = chip_size[::-1]

    box = mbd.detect(matrix, chip_size)
    cbi = ChipBoxInfo(LeftTop=box[0], LeftBottom=box[1],
                      RightBottom=box[2], RightTop=box[3])

    return cbi


class ChipAlignment(Alignment):
    """
    Satisfy the requirements of TissueBin: Utilize the bimodal chip angles as feature points,
    calculate transformation parameters, and achieve registration. The error is about 100pix
    """

    def __init__(
            self,
            flip_flag: bool = True,
            rot90_flag: bool = True
    ):
        super(ChipAlignment, self).__init__()

        self.register_matrix: np.matrix = None
        self.transforms_matrix: np.matrix = None

        self.rot90_flag = rot90_flag
        self._hflip = flip_flag
        self.no_trans_flag = False

        self.max_length = 9996  # Maximum size of image downsampling

    def registration_image(self,
                           file: Union[str, np.ndarray, CBImage]):
        """ To treat the transformed image, call the image processing library to return
        the transformed image according to the alignment parameters """

        if not isinstance(file, CBImage):
            image = cbimread(file)
        else:
            image = file

        if self.no_trans_flag:
            # TODO
            result = None
        else:
            result = image.trans_image(
                scale=[1 / self._scale_x, 1 / self._scale_y],
                rotate=self._rotation,
                rot90=self.rot90,
                offset=self.offset,
                dst_size=self._fixed_image.mat.shape,
                flip_lr=self.hflip
            )

        return result

    def align_stitched(
            self,
            fixed_image: ChipFeature,
            moving_image: ChipFeature
    ):
        """

        Args:
            fixed_image:
            moving_image:

        Returns:

        """
        self._rotation = -moving_image.chip_box.Rotation
        self._scale_x = moving_image.chip_box.ScaleX
        self._scale_y = moving_image.chip_box.ScaleY
        self._fixed_image = fixed_image

        if self.no_trans_flag:
            self.align_transformed(fixed_image, moving_image)
        else:
            transformed_image = self.transform_image(file=moving_image.mat)

            transformed_feature = ChipFeature()
            transformed_feature.set_mat(transformed_image)

            trans_mat = self.get_coordinate_transformation_matrix(
                moving_image.mat.shape,
                [1 / self._scale_x, 1 / self._scale_y],
                self._rotation
            )

            trans_points = self.get_points_by_matrix(moving_image.chip_box.chip_box, trans_mat)
            transformed_feature.chip_box.set_chip_box(trans_points)

            self.align_transformed(fixed_image, transformed_feature)

    def align_transformed(
            self,
            fixed_image: ChipFeature,
            moving_image: ChipFeature
    ):
        """

        Args:
            fixed_image:
            moving_image:

        Returns:

        """
        self._fixed_image = fixed_image
        moving_image.set_mat(
            self._fill_image(moving_image.mat.image, moving_image.chip_box.chip_box)
        )

        coord_index = [0, 1, 2, 3]
        if self.rot90_flag:
            range_num = 4
        else:
            range_num = 1

        if self.hflip:
            new_box = self.transform_points(points=moving_image.chip_box.chip_box,
                                            shape=moving_image.mat.shape, flip=0)
            new_mi = np.fliplr(moving_image.mat.image)
        else:
            new_box = moving_image.chip_box.chip_box
            new_mi = moving_image.mat.image

        if new_mi.ndim == 3: new_mi = new_mi[:, :, 0]

        down_size = max(fixed_image.mat.shape) // self.max_length

        register_info = dict()
        for index in range(range_num):
            register_image, M = self.get_matrix_by_points(
                new_box[coord_index, :] / down_size, fixed_image.chip_box.chip_box / down_size,
                True, new_mi[::down_size, ::down_size], np.array(fixed_image.mat.shape) // down_size
            )

            lu_x, lu_y = map(int, fixed_image.chip_box.chip_box[0] / down_size)
            rd_x, rd_y = map(int, fixed_image.chip_box.chip_box[2] / down_size)

            _wsi_image = register_image[lu_y: rd_y, lu_x:rd_x]
            _gene_image = fixed_image.mat.image[::down_size, ::down_size][lu_y: rd_y, lu_x:rd_x]

            ms = self.multiply_sum(_wsi_image, _gene_image)
            # _, res = self.dft_align(_gene_image, _wsi_image, method = "sim")

            clog.info(f"Rot{index * 90}, Score: {ms}")
            register_info[index] = {"score": ms, "mat": M}  # , "res": res}

            coord_index.append(coord_index.pop(0))

        best_info = sorted(register_info.items(), key=lambda x: x[1]["score"], reverse=True)[0]
        if self.rot90_flag: self._rot90 = range_num - best_info[0]
        _mat = self.get_coordinate_transformation_matrix(
            moving_image.mat.shape, [1, 1], 90 * best_info[0]
        )
        _box = self.get_points_by_matrix(new_box, _mat)
        _box = self.check_border(_box)
        # self._offset = (fixed_image.chip_box.chip_box[0, 0] - (_box[0, 0] + _box[1, 0]) / 2,
        #                 fixed_image.chip_box.chip_box[0, 1] - (_box[0, 1] + _box[3, 1]) / 2)

        self._offset = (fixed_image.chip_box.chip_box[0, 0] - _box[0, 0],
                        fixed_image.chip_box.chip_box[0, 1] - _box[0, 1])


def chip_align(
        moving_image: ChipFeature,
        fixed_image: ChipFeature,
        from_stitched: bool = True,
        flip_flag: bool = True,
        rot90_flag: bool = True
):
    """
    :param moving_image: The image to be registered is usually a stained image (such as ssDNA, HE)
    :param fixed_image: Fixed image
    :param from_stitched: Starting from the stitching diagram
    :param flip_flag:
    :param rot90_flag:

    Returns:

    """
    ca = ChipAlignment(flip_flag=flip_flag, rot90_flag=rot90_flag)
    if moving_image.tech_type is TechType.HE:
        from cellbin2.image.augmentation import f_rgb2hsv
        moving_image.set_mat(mat=f_rgb2hsv(moving_image.mat.image, channel=1, need_not=False))
    if from_stitched: ca.align_stitched(fixed_image=fixed_image, moving_image=moving_image)
    else: ca.align_transformed(fixed_image=fixed_image, moving_image=moving_image)

    info = {
            'offset': tuple(list(ca.offset)),
            'counter_rot90': ca.rot90,
            'flip': ca.hflip,
            'register_score': ca.score,
            'register_mat': ca.registration_image(moving_image.mat),
            'dst_shape': (fixed_image.mat.shape[0], fixed_image.mat.shape[1]),
            'method': AlignMode.ChipBox
        }

    return info


def main():
    from pathlib import Path
    _repo_root = str(Path(__file__).resolve().parents[3])
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from cellbin2.matrix.box_detect import detect_chip_box
    # move image loading
    moving_image = ChipFeature()
    moving_image.tech_type = TechType.DAPI
    moving_mat = cbimread(r"/")
    moving_image.set_mat(moving_mat)
    h_flipped_img = moving_mat.trans_image(flip_lr=True)
    h, w = h_flipped_img.shape[:2]
    #
    cfg = ChipParam(
        **{"stage1_weights_path":
               r"/",
           "stage2_weights_path":
               r"/"})


    #file_path = r"D:\cellbin_data\芯片单细胞\manlin_data\染色图\H1-3-ssdna\H1-3-ssdna.tif"
    m_info = detect_chip(h_flipped_img.image, cfg=cfg, stain_type=TechType.DAPI, 
                     actual_size=(9996, 9996), is_debug=True)[0]
    canvas_size = (23520, 23520)
    moving_image.set_chip_box(m_info)
    
    chip_corners = [
        m_info.LeftTop,   # LeftTop
        m_info.LeftBottom,    # LeftBottom
        m_info.RightTop,    # RightTop
        m_info.RightBottom    # RightBottom
    ]

    
    rotation_angle = m_info.Rotation  # 旋转角度
    
    #rotated_img = img.trans_image(rotate=rotation_angle, scale=[m_info.ScaleX, m_info.ScaleY], dst_size=canvas_size)
    rotated_img = h_flipped_img.trans_image(rotate=rotation_angle)
    cbimwrite(r"/", rotated_img)

    #image_center = (rotated_img.shape[1]/2, rotated_img.shape[0]/2)  # 图像中心
    
    new_h, new_w = rotated_img.shape[:2]

    theta = math.radians(rotation_angle)

    # OpenCV rotation matrix
    rotated_img = np.array(rotated_img._image)
    M = cv.getRotationMatrix2D((w/2, h/2), rotation_angle, 1)
    pts = np.array(chip_corners)
    pts = np.hstack([pts, np.ones((4,1))])
    rot = (M @ pts.T).T
    # canvas padding compensate
    rot[:,0] += (new_w - w)/2
    rot[:,1] += (new_h - h)/2
    moving_max_y = int(max(rot[1,1],rot[3,1]))
    moving_min_y = int(min(rot[0,1],rot[2,1]))
    moving_max_x = int(max(rot[2,0],rot[3,0]))
    moving_min_x = int(min(rot[0,0],rot[1,0]))

    moving_height, moving_width = moving_max_y - moving_min_y, moving_max_x - moving_min_x
    chip_region = rotated_img[moving_min_y:moving_max_y, moving_min_x:moving_max_x]
    target_size = 9996
    chip_resized = cv.resize(chip_region, (target_size, target_size))


    # 90 rot
    rotated_versions = []
    current_img = chip_resized.copy()

    for i in range(4):
        if i > 0:  
            current_img = cv.rotate(current_img, cv.ROTATE_90_CLOCKWISE)
        rotated_versions.append(current_img.copy())

    # Put in Matrix form
    matrix_path = r"/"
    matrix = tifffile.imread(matrix_path)
    print(f"detect_chip_box source: {bd.__file__}")
    box = bd.detect_chip_box(matrix, chip_size = [1, 1])
    print(box)
    quit()
    regist_centroid_x, regist_centroid_y = calculate_centroid(matrix)
    regist_top_left_x = int(box.LeftTop[0])
    regist_top_left_y = int(box.LeftTop[1])
    print(f"Box LeftTop: ({regist_top_left_x}, {regist_top_left_y})")

    base_canvas = np.zeros(canvas_size, dtype=rotated_img.dtype)

    for i, rotated_chip in enumerate(rotated_versions):
        canvas = base_canvas.copy()
        
        # put chip in the canvas
        canvas[regist_top_left_y:regist_top_left_y+target_size, regist_top_left_x:regist_top_left_x+target_size] = rotated_chip
        
        # savw
        output_path = rf"/"
        cbimwrite(output_path, canvas)


    
if __name__ == '__main__':
    main()
    # Ensure imports resolve to this repo (not an installed `cellbin2` elsewhere).
    



    


