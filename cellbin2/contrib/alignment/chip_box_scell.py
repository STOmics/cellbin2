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
    def __init__(self, binsize=21, down_size=4, morph_size=9, gene_base_size=10000):
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
            flip_flag: bool = False,
            rot90_flag: bool = True
    ):
        super(ChipAlignment, self).__init__()

        self.register_matrix: np.matrix = None
        self.transforms_matrix: np.matrix = None

        self.rot90_flag = rot90_flag
        self._hflip = flip_flag
        self.no_trans_flag = False

        self.max_length = 10000  # Maximum size of image downsampling

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
            transformed_image = self.transform_image(file=moving_image.mat) #transform moving image

            transformed_feature = ChipFeature()
            transformed_feature.set_mat(transformed_image)

            trans_mat = self.get_coordinate_transformation_matrix(
                moving_image.mat.shape,
                [1 / self._scale_x, 1 / self._scale_y],
                self._rotation
            ) # fetch transformation matrix M

            trans_points = self.get_points_by_matrix(moving_image.chip_box.chip_box, trans_mat)
            transformed_feature.chip_box.set_chip_box(trans_points) # set new corner point loations

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
        register_info = dict()
        
        down_size = max(fixed_image.mat.shape) // self.max_length
        
        for flip_state in [0, 1]:
            if flip_state == 1:
                new_box = self.transform_points(points=moving_image.chip_box.chip_box,
                                            shape=moving_image.mat.shape, flip=0)
                new_mi = np.fliplr(moving_image.mat.image)
            else:
                new_box = moving_image.chip_box.chip_box
                new_mi = moving_image.mat.image
            
            if new_mi.ndim == 3:
                new_mi = new_mi[:, :, 0]
            
            for rot_index in range(4):
                register_image, M = self.get_matrix_by_points(
                    new_box[coord_index, :] / down_size, 
                    fixed_image.chip_box.chip_box / down_size,
                    True, 
                    new_mi[::down_size, ::down_size], 
                    np.array(fixed_image.mat.shape) // down_size
                )
                
                lu_x, lu_y = map(int, fixed_image.chip_box.chip_box[0] / down_size)
                rd_x, rd_y = map(int, fixed_image.chip_box.chip_box[2] / down_size)
                
                _wsi_image = register_image[lu_y: rd_y, lu_x:rd_x]
                _gene_image = fixed_image.mat.image[::down_size, ::down_size][lu_y: rd_y, lu_x:rd_x]
                
                ms = self.multiply_sum(_wsi_image, _gene_image)
                
                transform_key = f"flip_{flip_state}_rot_{rot_index}"
                clog.info(f"Flip: {flip_state}, Rot: {rot_index * 90}°, Score: {ms}")
                
                register_info[transform_key] = {
                    "score": ms, 
                    "mat": M,
                    "flip_state": flip_state,
                    "rot_index": rot_index
                }
                
                coord_index.append(coord_index.pop(0))
        
        best_info = sorted(register_info.items(), key=lambda x: x[1]["score"], reverse=True)[0]
        best_transform = best_info[1]
        
        self._flip_state = best_transform["flip_state"]
        self._rot90 = best_transform["rot_index"]
        
        clog.info(f"Best transform - Flip: {self._flip_state}, Rot: {self._rot90 * 90}°, Score: {best_transform['score']}")
        
        if self._flip_state == 1:
            new_box = self.transform_points(points=moving_image.chip_box.chip_box,
                                        shape=moving_image.mat.shape, flip=0)
        else:
            new_box = moving_image.chip_box.chip_box
        
        _mat = self.get_coordinate_transformation_matrix(
            moving_image.mat.shape, [1, 1], 90 * self._rot90
        )
        _box = self.get_points_by_matrix(new_box, _mat)
        _box = self.check_border(_box)
        
        self._offset = (fixed_image.chip_box.chip_box[0, 0] - _box[0, 0],
                        fixed_image.chip_box.chip_box[0, 1] - _box[0, 1])
        
        return register_info


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
    #from cellbin2.matrix.box_detect import detect_chip_box
    # move image loading
    moving_image = ChipFeature()
    moving_mat = cbimread(r"/")
    moving_image.set_mat(moving_mat)
    #h_flipped_img = moving_mat.trans_image(flip_lr=True)
    h, w = moving_mat.shape[:2]
    #moving_image.set_mat(h_flipped_img.image)
    #
    cfg = ChipParam(
        **{"stage1_weights_path":
               r"/",
           "stage2_weights_path":
               r"/"})

    m_info = detect_chip(moving_mat.image, cfg=cfg, 
                     actual_size=(10000, 10000), is_debug=False, stain_type=TechType.DAPI)[0]
    #canvas_size = (23520, 23520)
    moving_image.set_chip_box(m_info)


    # Put in Matrix form
    fixed_image = ChipFeature()
    fixed_image.tech_type = TechType.Transcriptomics
    fixed_image.set_mat(r"/")
    f_info = detect_chip_box(fixed_image.mat.image, chip_size = (1, 1))
    fixed_image.set_chip_box(f_info)

    result = chip_align(moving_image, fixed_image)
    print(result)
    cbimwrite(r"/",  result["register_mat"]._image)


    matrix_path = r"/"
    matrix = tifffile.imread(matrix_path)
    print(f"detect_chip_box source: {bd.__file__}")
    #box = bd.detect_chip_box(matrix, chip_size = [1, 1])
    box = f_info
    regist_top_left_x = int(box.LeftTop[0])
    regist_top_left_y = int(box.LeftTop[1])
    print(f"Box LeftTop: ({regist_top_left_x}, {regist_top_left_y})")


    
if __name__ == '__main__':
    main()
    # Ensure imports resolve to this repo (not an installed `cellbin2` elsewhere).
    



    


