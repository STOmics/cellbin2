import copy

import cv2 as cv
import numpy as np
from enum import Enum

from numba import njit, prange
from typing import Union, List, Tuple, Any
from pydantic import BaseModel, Field

from cellbin2.image import CBImage
from cellbin2.image import cbimread
from cellbin2.utils.common import iPlaceHolder, TechType, fPlaceHolder, bPlaceHolder
from cellbin2.contrib.template.inference import TemplateInfo


class AlignMode(Enum):
    TemplateCentroid = 1
    Template00Pt = 2
    ChipBox = 3


class ChipBoxInfo(BaseModel):
    LeftTop: List[float] = Field([fPlaceHolder, fPlaceHolder], description='Left-Up XY')
    LeftBottom: List[float] = Field([fPlaceHolder, fPlaceHolder], description='Left-Down XY')
    RightTop: List[float] = Field([fPlaceHolder, fPlaceHolder], description='Right-Up XY')
    RightBottom: List[float] = Field([fPlaceHolder, fPlaceHolder], description='Right-Down XY')
    ScaleX: float = Field(fPlaceHolder, description='The x scale of fixed image')
    ScaleY: float = Field(fPlaceHolder, description='The y scale of fixed image')
    ChipSize: Tuple[float, float] = Field((fPlaceHolder, fPlaceHolder), description='Chip shape')
    Rotation: float = Field(fPlaceHolder, description='Chip rotate')
    IsAvailable: bool = Field(bPlaceHolder, description='Do you recommend downstream processes to use this parameter')

    @property
    def chip_box(self) -> np.ndarray:
        """Arrange in the top left, bottom left, bottom right, bottom right, and top right directions

        Returns:

        """
        points = np.array([self.LeftTop,
                           self.LeftBottom,
                           self.RightBottom,
                           self.RightTop])
        return points

    def set_chip_box(
            self, cb: np.ndarray
    ) -> None:
        self.LeftTop, self.LeftBottom, self.RightBottom, self.RightTop = \
            [list(i) for i in cb]


class ChipFeature(BaseModel):
    tech_type: TechType = TechType.ssDNA
    chip_box: ChipBoxInfo = ChipBoxInfo()
    template: TemplateInfo = TemplateInfo()
    point00: Tuple[int, int] = (0, 0)
    mat: Union[str, CBImage] = ''
    anchor_point: Tuple[int, int] = (0, 0)

    class Config:
        arbitrary_types_allowed = True
    # @property
    # def chip_box(self, ):
    #     return self._chip_box

    def set_chip_box(self, chip_box):
        self.chip_box = chip_box

    def set_point00(self, points):
        if isinstance(points, tuple) and len(points) == 2:
            self.point00 = points

    def set_anchor_point(self, points):
        if isinstance(points, tuple) and len(points) == 2:
            self.anchor_point = points

    # @property
    # def point00(self, ):
    #     return self._point00

    # @property
    # def mat(self, ):
    #     return self._mat

    def set_mat(self, mat):
        if not isinstance(mat, CBImage):
            self.mat = cbimread(mat)
        else:
            self.mat = mat

    # @property
    # def template(self, ):
    #     return self._template

    def set_template(self, template):
        if not isinstance(template, TemplateInfo):
            self.template.template_points = template
        else:
            self.template = template


class Alignment(object):
    """ Registration base class

    """

    def __init__(self, ):
        # input
        self._scale_x: float = 1.
        self._scale_y: float = 1.
        self._rotation: float = 0.

        # output
        self._offset: Tuple[float, float] = (0., 0.)
        self._rot90: int = 0
        self._hflip: bool = True
        self._score: int = iPlaceHolder

        # self.registration_image: CBImage
        self._fixed_image: ChipFeature
        self._moving_image: ChipFeature

    @property
    def offset(self, ) -> Tuple[float, float]:
        return self._offset

    @property
    def rot90(self, ) -> int:
        return self._rot90

    @property
    def hflip(self, ) -> bool:
        return self._hflip

    @property
    def score(self, ) -> int:
        return self._score

    def transform_image(
            self,
            file: Union[str, np.ndarray, CBImage]
    ):
        """ To treat the transformed image, call the image processing library to return
        the normalized image according to the normalization parameters """

        if not isinstance(file, CBImage):
            image = cbimread(file)
        else:
            image = file

        result = image.trans_image(
            scale=[1 / self._scale_x, 1 / self._scale_y],
            rotate=-self._rotation,
        )

        return result

    def registration_image(self,
                           file: Union[str, np.ndarray, CBImage]):
        """ To treat the transformed image, call the image processing library to return
        the transformed image according to the alignment parameters """

        if not isinstance(file, CBImage):
            image = cbimread(file)
        else:
            image = file

        result = image.trans_image(
            scale=[1 / self._scale_x, 1 / self._scale_y],
            rotate=-self._rotation,
            rot90=self.rot90,
            offset=self.offset,
            dst_size=self._fixed_image.mat.shape,
            flip_lr=self.hflip
        )

        return result

    def get_coordinate_transformation_matrix(self, shape, scale, rotate):
        """
        The true transformation matrix of the positions of points before and after image transformation,
        Unlike cv2. getRotationMatrix2D and cv2. warp Perspective,
        which are coordinate systems based on the origin (0,0)

        Args:
            shape: h, w
            scale:
            rotate:

        Returns:

        """
        # central rotate transform matrix 
        mat_scale_rotate = self.scale_rotate2mat(scale, rotate)
        mat_center_f = self._matrix_eye_offset(-shape[1] / 2, -shape[0] / 2)

        # get size and offset of transformed image 
        x, y, _, _ = self.get_scale_rotate_shape(shape, scale, rotate)
        mat_offset = self._matrix_eye_offset(x / 2, y / 2)

        # final transform matrix 
        mat_result = mat_offset * mat_scale_rotate * mat_center_f
        return mat_result

    def get_scale_rotate_shape(self, shape, scale, rotate):
        """
        Obtain the scale size of the rotated and scaled image
        Args:
            shape: h, w
            scale:
            rotate:

        Returns:
            x, y
        """
        mat = self.scale_rotate2mat(scale, rotate)
        points = np.array([[0, 0],
                           [0, shape[0]],
                           [shape[1], 0],
                           [shape[1], shape[0]]])

        result = mat[:2, :] @ np.concatenate([points, np.ones((points.shape[0], 1))], axis=1).transpose(1, 0)
        x = result[0, :].max() - result[0, :].min()
        y = result[1, :].max() - result[1, :].min()

        x_d, y_d = np.min(np.array(result), axis=1)

        return x, y, x_d, y_d

    @staticmethod
    def scale_rotate2mat(
            scale: Union[int, float, List, Tuple] = 1,
            rotate: Union[int, float] = 0,
            offset: Union[List, Tuple] = None,
            rotate_first_flag: bool = True
    ) -> np.matrix:
        """
        Matrix transformation of scaling, rotating, and then translating

        Args:
            scale:
            rotate:
            offset: [x, y] Default Last Action
            rotate_first_flag:

        Returns:

        """
        if isinstance(scale, (int, float)):
            scale_x = scale_y = scale
        else:
            scale_x, scale_y = scale

        mat_scale = np.mat([[scale_x, 0, 0],
                            [0, scale_y, 0],
                            [0, 0, 1]])

        mat_rotate = np.mat([[np.cos(np.radians(rotate)), -np.sin(np.radians(rotate)), 0],
                             [np.sin(np.radians(rotate)), np.cos(np.radians(rotate)), 0],
                             [0, 0, 1]])

        if offset is not None:
            mat_offset = np.mat([[1, 0, offset[0]],
                                 [0, 1, offset[1]],
                                 [0, 0, 1]])

            if rotate_first_flag:
                mat = mat_offset * mat_scale * mat_rotate
            else:
                mat = mat_offset * mat_rotate * mat_scale
        else:
            if rotate_first_flag:
                mat = mat_scale * mat_rotate
            else:
                mat = mat_rotate * mat_scale

        return mat

    @staticmethod
    def get_points_by_matrix(points, mat):
        """
        New coordinates of image points after transformation matrix
        Args:
            points:
            mat:

        Returns:

        """
        if points.ndim == 1:
            _points = np.array([points])
        else:
            _points = points

        _points = _points[:, :2]

        new_points = mat[:2, :] @ np.concatenate([
            _points, np.ones((_points.shape[0], 1))],
            axis=1
        ).transpose(1, 0)

        return np.array(new_points).squeeze().transpose()

    @staticmethod
    def get_matrix_by_points(points_src, points_dst,
                             need_image=False,
                             src: Union[CBImage, np.ndarray] = None,
                             dst_shape: tuple = None
                             ):
        """
        Obtain matrix through point transformation
        Args:
            src:
            points_src:
            points_dst:
            dst_shape:
            need_image:

        Returns:

        """
        result = M = None

        if src is None: return result, M

        M = cv.getPerspectiveTransform(np.array(points_src, dtype=np.float32),
                                       np.array(points_dst, dtype=np.float32))

        if need_image:
            src = src if isinstance(src, np.ndarray) else src.image
            result = cv.warpPerspective(src, M, (dst_shape[1], dst_shape[0]))

        return result, M

    @staticmethod
    def transform_points(**kwargs):
        """
        Corner Flip
        Args:
            **kwargs:
                points:
                shape:
                flip: 0 | 1, Y axis flip and X axis flip

        Returns:

        """

        points = kwargs.get("points", None)
        shape = kwargs.get("shape", None)

        if points is None or shape is None:
            return

        flip = kwargs.get("flip", None)
        if flip == 0:
            points[:, 0] = shape[1] - points[:, 0]
            points = points[[3, 2, 1, 0], :]
        elif flip == 1:
            points[:, 1] = shape[0] - points[:, 1]
            points = points[[1, 0, 3, 2], :]
        else:
            raise ValueError

        return points

    @staticmethod
    def _matrix_eye_offset(x, y, n=3):
        """

        Args:
            x:
            y:

        Returns:

        """
        mat = np.eye(n)
        mat[:2, 2] = [x, y]
        return mat

    @staticmethod
    def get_mass(image):
        """

        Args:
            image:

        Returns:

        """
        M = cv.moments(image)
        cx_cv = round(M['m10'] / M['m00'])
        cy_cv = round(M['m01'] / M['m00'])

        return cx_cv, cy_cv

    @staticmethod
    def check_border(file: np.ndarray):
        """ Check array, default (left-up, left_down, right_down, right_up)

        Args:
            file:

        Returns:

        """
        if not isinstance(file, np.ndarray): return None
        assert file.shape == (4, 2), "Array shape error."

        file = file[np.argsort(np.mean(file, axis=1)), :]
        if file[1, 0] > file[2, 0]:
            file = file[(0, 2, 1, 3), :]

        file = file[(0, 1, 3, 2), :]

        return file

    @staticmethod
    def _fill_image(
            image: np.ndarray,
            chip_box: np.ndarray
    ):
        """

        Args:
            image:
            chip_box:

        Returns:

        """
        contours = list()
        _temp = np.zeros_like(image, dtype = image.dtype)

        _chip_box = np.int_(chip_box)
        contours.append(_chip_box.reshape(_chip_box.shape[0], 1, -1))
        contours = tuple(contours)

        _temp = cv.drawContours(_temp, contours, -1, 1, cv.FILLED)
        image = image * _temp

        return image

    @staticmethod
    @njit(parallel=True)
    def multiply_sum(a, b):
        """
        Calculate the cumulative sum of matrix multiplication
        """
        res = 0
        (h, w) = a.shape
        for i in prange(h):
            for j in range(w):
                res += a[i][j] * b[i][j]
        return res


def transform_points(
        points: np.ndarray,
        src_shape: Tuple[int, int],
        scale: Union[int, float, list, tuple] = 1,
        rotation: Union[float, int] = 0,
        offset: tuple = (0, 0),
        flip: int = -1
) -> [np.ndarray, np.matrix]:
    """

    Args:
        points: n * 2/4 array size -- (x, y)
        src_shape: original image size -- (h, w)
        scale:
        rotation:
        offset: This value is defined as the value obtained after completing all transformation operations -- (x, y)
        flip: -1 means not flipping 0 to x coordinate and flipping 1 to y coordinate

    Returns:
        new_points:
        mat: Only the transformation matrix for scale and rotate

    """
    align = Alignment()

    src_shape = list(map(lambda x: x - 1, src_shape))

    mat = align.get_coordinate_transformation_matrix(shape=src_shape, scale=scale, rotate=rotation)
    if flip == 0:
        points[:, 0] = src_shape[1] - points[:, 0]
    elif flip == 1:
        points[:, 1] = src_shape[0] - points[:, 1]
    else:
        pass

    new_points = align.get_points_by_matrix(points, mat)

    new_points = new_points + offset

    p = copy.copy(points)
    p[:, :2] = new_points

    return p, mat


if __name__ == "__main__":
    src_points = np.array([[0, 0],
                           [0, 99],
                           [99, 99],
                           [99, 0]])
    dst_points = transform_points(src_points,
                                  src_shape=(100, 100),
                                  scale=1,
                                  rotation=180,
                                  offset=(0, 0),
                                  flip=0)
    print(dst_points)

    _points, _ = transform_points(
        np.loadtxt(r"D:\02.data\temp\A03599D1\00pt\temp_rot0.txt"),
        (22346, 24406),
        rotation = 90
    )
    np.savetxt(r"D:\02.data\temp\A03599D1\00pt\temp_rot0__1.txt", _points)
