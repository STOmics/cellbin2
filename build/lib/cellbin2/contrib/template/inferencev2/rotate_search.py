"""By lizepeng1 - 2024/01/12"""

import os
import numpy as np
from sklearn.linear_model import LinearRegression


class RotateSearch:
    """
    Track inspection points for angle determination
    """
    def __init__(self,
                 search_range=90,
                 search_point_num=5,
                 dis_range=5000,
                 point_range=20):
        """
        Args:
            search_range:
            search_point_num:
            dis_range:
            point_range:
        """
        self.search_range = search_range
        self.dis_range = dis_range
        self.point_range = point_range
        self.search_point_num = search_point_num

    @staticmethod
    def _center_point_search(points, n=1):
        """
        Center point matching
        Args:
            points:
        """
        points = np.array(points)
        x_mean = np.mean(points[:, 0])
        y_mean = np.mean(points[:, 1])
        return sorted(points, key=lambda x: ((x[0] - x_mean) ** 2) + (x[1] - y_mean) ** 2)[:n]

    @staticmethod
    def _get_distance_from_point_to_line(point, line_point1, line_point2):
        """
        Calculate the distance from a point to a straight line
        Args:
            point:
            line_point1:
            line_point2:
        """
        # When the coordinates of two points are the same, return the distance between the points
        if line_point1 == line_point2:
            point_array = np.array(point)
            point1_array = np.array(line_point1)
            return np.linalg.norm(point_array - point1_array)
        # Calculate the three parameters of a straight line
        A = line_point2[1] - line_point1[1]
        B = line_point1[0] - line_point2[0]
        C = (line_point1[1] - line_point2[1]) * line_point1[0] + \
            (line_point2[0] - line_point1[0]) * line_point1[1]
        # Calculate the distance based on the formula for the distance from a point to a straight line
        distance = np.abs(A * point[0] + B * point[1] + C) / (np.sqrt(A**2 + B**2))
        return distance

    @staticmethod
    def _fitting_line2rotate(points):
        """
        Fit line segments with points and obtain angles
        """
        line_model = LinearRegression()
        _x = points[:, 0].reshape(-1, 1)
        _y = points[:, 1].reshape(-1, 1)

        line_model.fit(_x, _y)
        _k = line_model.coef_[0][0]
        _rotate = np.degrees(np.arctan(_k))

        return _rotate

    def distance_by_line(self, line_points, points):
        """
        Value based on distance
        Args:
            line_points: points -- [[x1, y1], [x2, y2]]
            points:
        """
        dis_list = list()
        points_list = list()
        for point in points:
            dis = self._get_distance_from_point_to_line(point, line_points[0], line_points[1])
            if dis <= self.point_range:
                dis_list.append(dis)
                points_list.append(point)

        return dis_list, points_list

    def get_rotate(self, points, fit=False):
        """
        Obtain angle value
        """
        if len(points) < 3: return None
        center_points = self._center_point_search(points, self.search_point_num)

        points_rotate_list = list()
        points_all_list = list()
        for _points in center_points:
            rotate_min = None
            rotate_min_dis = None
            points_min_list = None
            for _rotate in range(-self.search_range, self.search_range):
                line_point_left = [_points[0] + self.dis_range * np.cos(np.radians(_rotate)),
                                   _points[1] + self.dis_range * np.sin(np.radians(_rotate))]
                line_point_right = [_points[0] - self.dis_range * np.cos(np.radians(_rotate)),
                                    _points[1] - self.dis_range * np.sin(np.radians(_rotate))]
                dis_list, points_list = self.distance_by_line([line_point_left, line_point_right], points)

                if rotate_min_dis is None:
                    rotate_min = _rotate
                    rotate_min_dis = dis_list
                    points_min_list = points_list
                else:
                    d_m = np.mean(dis_list)
                    p_m = np.mean(rotate_min_dis)

                    if len(rotate_min_dis) < len(dis_list):
                        rotate_min = _rotate
                        rotate_min_dis = dis_list
                        points_min_list = points_list
                    elif len(rotate_min_dis) == len(dis_list):
                        if d_m < p_m:
                            rotate_min = _rotate
                            rotate_min_dis = dis_list
                            points_min_list = points_list
                    else: pass

            points_rotate_list.append(rotate_min)
            points_all_list.append(points_min_list)

        best_rotate = max(points_rotate_list, key=points_rotate_list.count)

        if fit:
            fitting_points = np.array(points_all_list[points_rotate_list.index(best_rotate)])
            best_rotate = self._fitting_line2rotate(fitting_points)

        if best_rotate < -45: best_rotate += 90
        elif best_rotate > 45: best_rotate -= 90

        return best_rotate


if __name__ == "__main__":

    import h5py
    import json

    ipr_path = r"D:\02.data\fengning\SS200000135TL_D1\result_h\SS200000135TL_D1_20231220_104256_0.1.ipr"
    pts = {}
    with h5py.File(ipr_path) as conf:
        qc_pts = conf['QCInfo/CrossPoints/']
        for i in qc_pts.keys():
            pts[i] = conf['QCInfo/CrossPoints/' + i][:]
    pts = sorted(pts.items(), key=lambda x: x[1].shape[0], reverse=True)
    rs = RotateSearch()
    for k, v in pts:
        rotate = rs.get_rotate(v)
        print(k, rotate)
    #################################################
    # target_points = np.loadtxt(r"D:\02.data\hanqinju\wuhan_2_(3_3).txt")
    # rs = RotateSearch()
    # rotate = rs.get_rotate(target_points)
    # print(rotate)
