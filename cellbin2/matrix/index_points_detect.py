import math
import numpy as np
from typing import List
from cellbin2.contrib.alignment.basic import TemplateInfo


class IndexPointsDetector(object):
    def __init__(self, ref: list):
        self.template = ref
        self.image: np.ndarray = np.array([])

    def get_mass(self, image=None):
        """
        2023/09/20 @fxzhao remove type convert, no impact for the results
        """
        if image is None:
            image = self.image
        image = image.astype(float)  # IMPORTANT: NO EDIT
        image_x = np.sum(image, 0)
        xx = np.array(range(len(image_x)))
        xx_cal = xx * image_x
        x_mass = np.sum(xx_cal) / np.sum(image_x)

        image_y = np.sum(image, 1)
        yy = np.array(range(len(image_y)))
        yy_cal = yy * image_y
        y_mass = np.sum(yy_cal) / np.sum(image_y)
        return np.array([x_mass, y_mass])

    @staticmethod
    def find_cross_ind(position, summation):
        result = []
        for _ in range(len(summation) - np.max(position) - 1):
            position += 1
            result.append(sum(summation[position]))
        first_ind = result.index(min(result)) + 2
        return first_ind

    def find_first_tp(self, mass_center, find_range=3000):
        min_x = int(round(mass_center[0]) - find_range)
        max_x = int(round(mass_center[0]) + find_range)
        min_y = int(round(mass_center[1]) - find_range)
        max_y = int(round(mass_center[1]) + find_range)
        find_image = self.image[min_y: max_y, min_x: max_x].astype(float)
        x_sum = np.sum(find_image, 0)  # vertical
        y_sum = np.sum(find_image, 1)  # horizontal
        position_x, position_y = np.cumsum(np.insert(np.array(self.template), 0, 0, axis=1)[:, :-1], axis=1)
        x_first = self.find_cross_ind(position_x, x_sum)
        y_first = self.find_cross_ind(position_y, y_sum)
        x_first += min_x
        y_first += min_y
        return x_first, y_first

    @staticmethod
    def one_to_all(template, mid_pos, length):
        step_size = sum(template)
        upper = math.ceil((length - mid_pos) / step_size)
        lower = math.ceil(mid_pos / step_size)
        interval = np.concatenate(
            (
                np.cumsum(np.tile(template, upper)),
                np.array([0]),
                np.cumsum(np.tile(-np.array(template[::-1]), lower))
            )
        )
        index = np.concatenate(
            (
                np.tile(np.arange(len(template)), lower),
                np.tile(np.arange(len(template)), upper),
                np.array([0])
            )
        )
        interval.sort()
        interval = mid_pos + interval
        interval = interval.reshape(-1, 1)
        index = index.reshape(-1, 1)
        combined = np.concatenate(
            (interval, index),
            axis=1
        )
        return combined

    def find_cross_state(self, chip_start, chip_size):
        """矩阵模板推导前的质控, 返回二态:
            0 — 正常: 矩阵表达重心在理论芯片区域内，通过；
            1 — 异常: 矩阵表达重心在理论芯片区域外，不通过；
        """
        mass_center = self.get_mass()
        x_min, y_min = chip_start[0], chip_start[1]
        x_max, y_max = chip_start[0] + chip_size[0], chip_start[1] + chip_size[1]
        in_region = (x_min <= mass_center[0] <= x_max) and (y_min <= mass_center[1] <= y_max)
        if not in_region:
            return 1
        else:
            return 0

    def find_cross(self, gene_exp, chip_start=None, chip_size=None):
        self.image = gene_exp
        if chip_start is not None and chip_size is not None:
            h, w = self.image.shape
            x_min = max(int(round(chip_start[0])), 0)
            y_min = max(int(round(chip_start[1])), 0)
            x_max = min(int(round(chip_start[0] + chip_size[0])), w)
            y_max = min(int(round(chip_start[1] + chip_size[1])), h)
            roi = self.image[y_min: y_max, x_min: x_max]
            mass_center = self.get_mass(roi) + np.array([x_min, y_min])
        else:
            mass_center = self.get_mass()
        x_mid, y_mid = self.find_first_tp(mass_center)
        h, w = self.image.shape
        x_all = self.one_to_all(self.template[0], x_mid, w)
        y_all = self.one_to_all(self.template[1], y_mid, h)

        all_comb = np.concatenate(
            (
                np.tile(
                    np.expand_dims(x_all, 1),
                    (1, y_all.shape[0], 1)
                ),
                np.tile(
                    np.expand_dims(y_all, 0),
                    (x_all.shape[0], 1, 1)
                )
            ),
            axis=2
        ).reshape(-1, 4)

        all_comb[:, [1, 2]] = all_comb[:, [2, 1]]
        final_result = all_comb[
            np.logical_and.reduce(
                (
                    all_comb[:, 0] <= w,
                    all_comb[:, 1] <= h,
                    all_comb[:, 0] >= 0,
                    all_comb[:, 1] >= 0
                )
            )
        ]
        return final_result

    @staticmethod
    def matrix_can_track(matrix, chip_name):
        """根据芯片名推导理论芯片区域，并对矩阵表达重心做质控。

        Returns:
            track_state: 0 — 正常(重心在芯片区域内); 1 — 异常(重心在区域外)
            chip_start: 理论芯片区域左上角坐标 [x, y]
            chip_norm_size: 理论芯片区域尺寸 [w, h]
        """
        from cellbin2.utils.stereo_chip import StereoChip
        sc = StereoChip()
        sc.parse_info(chip_no=chip_name, print_flag=False)
        chip_start = np.array(sc.zero_zero_point) - np.array(sc.zero_zero_chip_point)
        chip_norm_size = sc.norm_chip_size
        ild = IndexPointsDetector(ref=[])
        ild.image = matrix
        track_state = ild.find_cross_state(chip_start, chip_norm_size)
        return track_state, chip_start, chip_norm_size


def detect_cross_points(ref: List[List[int]], matrix: np.ndarray,
                        chip_start=None, chip_size=None) -> TemplateInfo:
    ild = IndexPointsDetector(ref=ref)
    pts = ild.find_cross(matrix, chip_start=chip_start, chip_size=chip_size)

    ti = TemplateInfo(template_recall=1.,
                      template_valid_area=1.,
                      trackcross_qc_pass_flag=1,
                      trackline_channel=0,
                      rotation=0.,
                      scale_x=1., scale_y=1.,
                      template_points=pts)

    return ti


def main():
    import tifffile

    # SN = 'B04660F514'
    # vision_image = tifffile.imread(r"F:\FBS\TS_0618\B04660F514.tif")
    SN = 'Y10043F514'
    vision_image = tifffile.imread(r"F:\FBS\TS_0618\Y10043F514.tif")
    # SN = 'Y40422LA'
    # vision_image = tifffile.imread(r"F:\FBS\TS_0618\Y40422LA.tif")
    points_path = r"F:\FBS\TS_0618\%s.txt"%SN
    matrix = vision_image
    ref = [[240, 300, 330, 390, 390, 330, 300, 240, 420],
           [240, 300, 330, 390, 390, 330, 300, 240, 420]]

    ild = IndexPointsDetector(ref=ref)
    track_state, chip_start, chip_norm_size = ild.matrix_can_track(matrix, SN)
    if track_state == 0:
        pts = detect_cross_points(ref, matrix, chip_start=chip_start, chip_size=chip_norm_size)
        np.savetxt(points_path, pts.template_points[:, :2])
        print("%s asd"%SN)
    else:
        print("Matrix data invalid for track point inference.")


if __name__ == '__main__':
    main()
