import math
import cv2 as cv
import numpy as np

from scipy.spatial.distance import cdist
from numba import njit, prange

from cellbin2.utils import clog
from cellbin2.image.augmentation import f_padding


def rotate(ptx, pty, angle, original_shape, new_shape):
    px, py = ptx, pty
    ori_h, ori_w = original_shape
    new_h, new_w = new_shape
    cx = ori_w / 2
    cy = ori_h / 2
    rad = math.radians(angle)
    new_px = cx + (px - cx) * math.cos(rad) + (py - cy) * math.sin(rad)
    new_py = cy + -((px - cx) * math.sin(rad)) + (py - cy) * math.cos(rad)
    x_offset, y_offset = (new_w - ori_w) / 2, (new_h - ori_h) / 2
    new_px += x_offset
    new_py += y_offset
    return new_px, new_py


@njit(parallel=True)
def multiply_sum(a, b):
    """
    2023/09/20 @fxzhao calculate the cumulative sum after matrix multiplication
    """
    res = 0
    (h, w) = a.shape
    for i in prange(h):
        for j in range(w):
            res += a[i][j] * b[i][j]
    return res


def sub_sum(a, b):
    """
    2024/01/03 @lzp
    """
    b = cv.dilate(b, np.ones((9, 9)))
    sub = a.astype(np.float32) - b.astype(np.float32)
    sub = np.abs(sub)
    return np.sum(sub)


class AlignByTrack:
    def __init__(self):
        """
        Vision image: generated based on gene expression matrix
        Transformed image obtained based on
            - stitched image
            - scale and rotation

        Transformed image should be in the same scale compared to vision image

        Transformed image and vision image only contain:
            - n times 90 degree rotation (e.g. 90, 180, 270, etc.)
            - x, y direction offsets

        Args:
            self.x_template (list): chip template on x direction
            self.y_template (list): chip template on y direction
            self.fov_size (float): length of a period on chip template
            self.dist_thresh (float): maximum distance threshold
            self.transformed_shape (tuple): shape of tranformed image
            self.transformed_mass (nd array): mass center of transformed image
            self.vision_shape (tuple): shape of vision image
            self.vision_mass (nd array): mass center of vision image
            self.vision_img (nd array): vision image
            self.transformed_image (nd array): transformed image
            self.transformed_cfov_pts (nd array): selected cross points on transform image
            self.vision_cfov_pts (nd array): selected cross points on vision image
        """
        self.search_angle_set = (0, 90, 180, 270)
        self.search_range_x = [-2, -1, 0, 1, 2]
        self.search_range_y = [-2, -1, 0, 1, 2]

        self.x_template = None
        self.y_template = None
        self.fov_size = None
        self.dist_thresh = None
        self.transformed_shape = None
        self.transformed_mass = None
        self.vision_shape = None
        self.vision_mass = None
        self.vision_img = None
        self.transformed_image = None
        self.transformed_cfov_pts = None
        self.vision_cfov_pts = None

        # registration method compatible with HE type data
        self.new_method: bool = False

    def set_chip_template(self, chip_template):
        self.x_template = chip_template[0]
        self.y_template = chip_template[1]
        self.fov_size = np.sum(self.x_template)
        self.dist_thresh = np.sum(self.x_template) * 2 / 3

    @staticmethod
    def get_mass(image):
        """

        Args:
            image:

        Returns:

        """

        if image.ndim == 3:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        M = cv.moments(image)
        cx_cv = round(M['m10'] / M['m00'])
        cy_cv = round(M['m01'] / M['m00'])

        return cx_cv, cy_cv

    @staticmethod
    def adjust_cross(stitch_template, scale_x, scale_y, fov_stitched_shape,
                     new_shape, chip_template, rotation, flip=True):
        scale_shape = np.array([fov_stitched_shape[0] * scale_y, fov_stitched_shape[1] * scale_x])  # I wrote it incorrectly before!!!

        stitch_template[:, 0] = stitch_template[:, 0] * scale_x
        stitch_template[:, 1] = stitch_template[:, 1] * scale_y

        pts = stitch_template[:, :2]
        ids = stitch_template[:, 2:]

        new_px, new_py = rotate(
            pts[:, 0:1],
            pts[:, 1:2],
            rotation,
            original_shape=scale_shape,
            new_shape=new_shape,
        )

        if flip:
            new_px = new_shape[1] - 1 - new_px  # flip, note that the shape of image is used here, so -1 is required for index calculation
            chip_xlen, chip_ylen = [len(chip_template[0]), len(chip_template[1])]
            ids[:, 0] = chip_xlen - 1 - ids[:, 0]
        pts_ids = np.hstack((new_px, new_py, ids))
        return pts_ids

    @staticmethod
    def flip_points(points, shape, chip_template, axis=0):
        """

        Args:
            points: [x, y]
            shape: [h, w] == [y, x]
            axis:
                - 0: x
                - 1: y

        Returns:

        """
        if axis == 0:
            points[:, 0] = shape[1] - 1 - points[:, 0]
            points[:, 2] = len(chip_template[0]) - 1 - points[:, 2]
        else:
            points[:, 1] = shape[0] - 1 - points[:, 1]
            points[:, 3] = len(chip_template[1]) - 1 - points[:, 3]

        return points

    @staticmethod
    def cal_score(transformed_image, vision_image, offset, method=False):
        """
        2023/09/20 @fxzhao use slicing instead of padding to speed up and reduce memory usage
        """
        if method:
            left_x = int(round(abs(offset[0])))
            if offset[0] > 0:
                transformed_image = f_padding(transformed_image, 0, 0, left_x, 0)
            else:
                vision_image = f_padding(vision_image, 0, 0, left_x, 0)

            up_y = int(round(abs(offset[1])))
            if offset[1] > 0:
                transformed_image = f_padding(transformed_image, up_y, 0, 0, 0)
            else:
                vision_image = f_padding(vision_image, up_y, 0, 0, 0)

            shape_vision = np.shape(vision_image)
            shape_transform = np.shape(transformed_image)

            if shape_vision[0] > shape_transform[0]:
                transformed_image = f_padding(transformed_image, 0, shape_vision[0] - shape_transform[0], 0, 0)
            else:
                vision_image = f_padding(vision_image, 0, shape_transform[0] - shape_vision[0], 0, 0)

            if shape_vision[1] > shape_transform[1]:
                transformed_image = f_padding(transformed_image, 0, 0, 0, shape_vision[1] - shape_transform[1])
            else:
                vision_image = f_padding(vision_image, 0, 0, 0, shape_transform[1] - shape_vision[1])
            score = sub_sum(vision_image, transformed_image)
            return score
        else:
            x, y = 0, 0
            x0, y0 = 0, 0
            if offset[0] < 0:
                x = int(round(abs(offset[0])))
                x0 = 0
            else:
                x = 0
                x0 = int(round(abs(offset[0])))
            if offset[1] < 0:
                y = int(round(abs(offset[1])))
                y0 = 0
            else:
                y = 0
                y0 = int(round(abs(offset[1])))
            shape_vision = np.shape(vision_image)
            shape_white = np.shape(transformed_image)
            h, w = min(shape_vision[0]-y0, shape_white[0]-y), min(shape_vision[1]-x0, shape_white[1]-x)
            score = multiply_sum(vision_image[y0:y0+h, x0:x0+w], transformed_image[y:y+h, x:x+w])
            return score

    @staticmethod
    def get_pts_based_on_ids(pts_ids, keep_ids=(4, 4)):
        keep = (pts_ids[:, 2] == keep_ids[0]) & (pts_ids[:, 3] == keep_ids[1])
        selected_pts = pts_ids[keep][:, :2]
        return selected_pts

    @staticmethod
    def down_sample_normalize(img):
        img = img[::5, ::5]
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 100
        return img

    @staticmethod
    def get_new_shape(old_shape, angle):
        angle = math.radians(angle)
        angle_sin = math.sin(angle)
        angle_cos = math.cos(angle)
        h, w = old_shape
        new_w = round(h * math.fabs(angle_sin) + w * math.fabs(angle_cos))
        new_h = round(w * math.fabs(angle_sin) + h * math.fabs(angle_cos))
        new_shape = (new_h, new_w)
        return new_shape

    @staticmethod
    def get_rough_offset(offset_guess, rot_guess, old_shape, new_shape, transformed_pts, vision_pts, dist_thresh):
        rot_x, rot_y, = rotate(
            transformed_pts[:, 0:1],
            transformed_pts[:, 1:2],
            rot_guess,
            old_shape,
            new_shape
        )
        transformed_pts_temp = np.hstack((rot_x, rot_y))

        # get qualified pts
        dist = cdist(transformed_pts_temp + offset_guess, vision_pts)
        qualified = np.min(dist, axis=1) <= dist_thresh
        transformed_pts_qualified = transformed_pts_temp[qualified]
        dist_qualified = dist[qualified]
        vision_pt_qualified = vision_pts[np.argmin(dist_qualified, axis=1)]

        if len(transformed_pts_qualified) > 0:
            x_offset = -np.median(np.array(transformed_pts_qualified.T[0] - vision_pt_qualified.T[0]))
            y_offset = -np.median(np.array(transformed_pts_qualified.T[1] - vision_pt_qualified.T[1]))
        else:
            x_offset, y_offset = 0, 0
        return x_offset, y_offset

    def search_fov(self, offset_ori, angle):
        white_image = self.transformed_image
        white_image = np.rot90(white_image, angle // 90)
        vision_image = self.vision_img
        score_max = np.Inf if self.new_method else 0
        offset_last = []

        # Iterate through the matching scores of 9 surrounding FOVs 
        # traverse the matching degree of 9 FOVs up, down, left and right in turn
        for row in self.search_range_x:
            for col in self.search_range_y:
                offset_temp = [offset_ori[0] + col * self.fov_size, offset_ori[1] + row * self.fov_size]
                score_temp = self.cal_score(
                    white_image, vision_image, np.array(offset_temp) / 5,
                    method = self.new_method
                )

                if self.new_method:
                    if score_temp < score_max:
                        score_max = score_temp
                        offset_last = offset_temp
                else:
                    if score_temp > score_max:
                        score_max = score_temp
                        offset_last = offset_temp

        return np.array(offset_last), score_max

    def get_best_in_all_angles_offsets(self):
        score_record = []
        offset_record = []
        for num, angle in enumerate(self.search_angle_set):
            old_shape = self.transformed_shape
            new_shape = self.get_new_shape(old_shape, angle)
            rot_mass_x, rot_mass_y, = rotate(
                self.transformed_mass[0],
                self.transformed_mass[1],
                angle,
                self.transformed_shape,
                new_shape,
            )

            white_mass_temp = np.array([rot_mass_x, rot_mass_y])
            offset_temp = self.vision_mass - white_mass_temp
            rough_x_offset, rough_y_offset = self.get_rough_offset(
                offset_guess=offset_temp,
                rot_guess=angle,
                old_shape=old_shape,
                new_shape=new_shape,
                transformed_pts=self.transformed_cfov_pts,
                vision_pts=self.vision_cfov_pts,
                dist_thresh=self.dist_thresh
            )

            if rough_x_offset != 0 and rough_y_offset != 0:
                offset, score = self.search_fov(np.array([rough_x_offset, rough_y_offset]), angle)
            else:
                if self.new_method:
                    offset, score = [0, 0], np.Inf
                else:
                    offset, score = [0, 0], 0

            offset_record.append(offset)
            clog.info(f"Angle: {angle}, Score: {score}")
            score_record.append(score)

        md = min if self.new_method else max
        rot_type = score_record.index(md(score_record))
        offset = offset_record[rot_type]

        return offset, rot_type, md(score_record)

    @staticmethod
    def _enhance_vision_image(vision_img, down_size = 20, e = 3):
        new_h = vision_img.shape[0] // down_size
        new_w = vision_img.shape[1] // down_size
        sum_image = np.zeros([new_h, new_w], dtype=np.float32)

        for _h in range(new_h):
            for _w in range(new_w):
                value = np.sum(vision_img[_h * down_size: (_h + 1) * down_size,
                               _w * down_size: (_w + 1) * down_size])
                sum_image[_h, _w] = value

        sum_image = sum_image * ((sum_image / np.mean(sum_image) / e) ** 2)
        sum_image[sum_image > 255] = 255
        sum_image = np.array(sum_image, dtype=np.uint8)
        sum_image = cv.resize(sum_image, (vision_img.shape[1] // 5, vision_img.shape[0] // 5))

        return sum_image

    def run(self,
            transformed_image,
            vision_img,
            vision_cp,
            stitch_tc,
            flip,
            rot90_flag,
            new_method: bool = False
            ):
        """

        Args:
            transformed_image:
            vision_img:
            vision_cp:
            stitch_tc:
            flip:
            rot90_flag:
            new_method:

        Returns:

        """
        if not rot90_flag:
            self.search_angle_set = tuple([0])

        if transformed_image.ndim == 3:
            transformed_image = cv.cvtColor(transformed_image, cv.COLOR_BGR2GRAY)

        self.new_method = new_method
        self.transformed_shape = transformed_image.shape
        self.vision_shape = vision_img.shape

        if flip:
            self.transformed_image = np.fliplr(transformed_image)
            stitch_tc = self.flip_points(
                stitch_tc, self.transformed_shape,
                [self.x_template, self.y_template]
            )
        else:
            self.transformed_image = transformed_image

        self.transformed_mass = self.get_mass(self.transformed_image)
        self.vision_mass = self.get_mass(vision_img)

        self.transformed_image = self.down_sample_normalize(self.transformed_image)

        if self.new_method:
            self.vision_img = self._enhance_vision_image(vision_img)
        else:
            self.vision_img = self.down_sample_normalize(vision_img)

        self.transformed_cfov_pts = self.get_pts_based_on_ids(stitch_tc)
        self.vision_cfov_pts = self.get_pts_based_on_ids(vision_cp)

        offset, rot_type, score = self.get_best_in_all_angles_offsets()
        return offset, rot_type, score


if __name__ == '__main__':
    pass
