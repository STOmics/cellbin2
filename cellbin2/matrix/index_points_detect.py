import math
import numpy as np
from typing import List
from cellbin2.contrib.alignment.basic import TemplateInfo


class IndexPointsDetector(object):
    def __init__(self, ref: list):
        self.template = ref
        self.image: np.ndarray = np.array([])

    def get_mass(self, ):
        """
        2023/09/20 @fxzhao 移除类型转换,不影响结果
        """
        image = self.image.astype(float)  # 别动这里！！！
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

    def find_cross(self, gene_exp):
        self.image = gene_exp
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


def detect_cross_points(ref: List[List[int]], matrix: np.ndarray) -> TemplateInfo:
    ild = IndexPointsDetector(ref=ref)
    pts = ild.find_cross(matrix)

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

    matrix_path = r'E:/03.users/liuhuanlin/01.data/cellbin2/D03951C1.tif'
    points_path = r'E:\03.users\liuhuanlin\01.data\cellbin2\output\points.txt'
    matrix = tifffile.imread(matrix_path)
    ref = [[240, 300, 330, 390, 390, 330, 300, 240, 420],
           [240, 300, 330, 390, 390, 330, 300, 240, 420]]

    pts = detect_cross_points(ref, matrix)
    np.savetxt(points_path, pts)


if __name__ == '__main__':
    main()
