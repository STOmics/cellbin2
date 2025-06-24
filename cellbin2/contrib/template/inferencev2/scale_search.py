"""By lizepeng1 - 2024/01/15"""

import os
import numpy as np


class ScaleSearch:
    """
    track检点进行尺度判定, 需要给明角度
    """
    def __init__(self,
                 search_range=[0.3, 1.7],
                 chip_template=None):
        self.search_range = search_range
        self.chip_template = chip_template

        self.template = list()
        self.rotation = None
        self.scale_x = self.scale_y = None

    @staticmethod
    def _center_point_search(points, n=1):
        """
        中心点匹配
        Args:
            points:
        """
        points = np.array(points)
        x_mean = np.mean(points[:, 0])
        y_mean = np.mean(points[:, 1])
        return sorted(points, key=lambda x: ((x[0] - x_mean) ** 2) + (x[1] - y_mean) ** 2)[:n]

    @staticmethod
    def pair_to_template(temp_qc, temp_re, threshold=10, dis=False):
        """
        one point of temp0 map to only one point of temp1
        Args:
            dis: 距离测量
        """
        import scipy.spatial as spt

        temp_src = np.array(temp_re)[:, :2]
        temp_dst = np.array(temp_qc)[:, :2]
        tree = spt.cKDTree(data=temp_src)
        distance, index = tree.query(temp_dst, k=1)
        if dis:
            new_dis = [i for i in distance if i < threshold]
            return new_dis

        if isinstance(threshold, (int, float)):
            thr_index = index[distance < threshold]
            points_qc = temp_dst[distance < threshold]
        elif isinstance(threshold, list):
            threshold1, threshold2 = threshold
            thr_index = index[(threshold1 < distance) & (distance < threshold2)]
            points_qc = temp_dst[(threshold1 < distance) & (distance < threshold2)]

        points_re = np.array(temp_re)[thr_index]
        return [points_re, points_qc]

    def _point_inference(self, src_pt: tuple, region: tuple):
        '''
        search stand template from bin file by key(chip_no).
        src_pt :(x, y, ind_x, ind_y)
        region: (height, width)
        '''
        if len(self.template) > 0:
            self.template = list()

        x0, y0, ind_x, ind_y = src_pt

        k0 = np.tan(np.radians(self.rotation))
        if k0 == 0: k0 = 0.00000001
        k1 = -1 / k0

        y_intercept0 = y0 - k0 * x0
        x_intercept0 = (y0 - k1 * x0) * k0

        dy = abs(k0 * region[1])
        y_region = (-dy, region[0] + dy)
        dx = abs(k0 * region[0])
        x_region = (-dx, region[1] + dx)

        self.y_intercept = self._get_intercept(self.scale_y, y_intercept0, y_region, ind_y, self.chip_template[1])
        self.x_intercept = self._get_intercept(self.scale_x, x_intercept0, x_region, ind_x, self.chip_template[0])
        self._create_cross_points(k0)

    def _get_intercept(self, scale, intercept0, region, ind, templ):
        idx = intercept0
        intercept = [[idx, ind]]
        s, e = region
        item_count = len(templ)
        # face to large
        while idx < e:
            ind = int(ind % item_count)
            item_len = (templ[ind] * scale) / np.cos(np.radians(self.rotation))
            idx += item_len
            intercept.append([idx, (ind + 1) % item_count])
            ind += 1
        # face to small
        idx, ind = intercept[0]
        while idx > s:
            ind -= 1
            ind = int(ind % item_count)
            item_len = (templ[ind] * scale) / np.cos(np.radians(self.rotation))
            idx -= item_len
            intercept.append([idx, ind])
        return sorted(intercept, key=(lambda x: x[0]))

    def _create_cross_points(self, k):
        for x_ in self.x_intercept:
            for y_ in self.y_intercept:
                x, ind_x = x_
                y, ind_y = y_
                x0 = (x - k * y) / (pow(k, 2) + 1)
                y0 = k * x0 + y
                self.template.append([x0, y0, ind_x, ind_y])

    def _valid_scale_judge(self, target_points, rate=1.2, search_thresh=None):
        """
        搜索尺度时，解决对于小scale尺度匹配距离过小问题
        """
        point_re, point_qc = self.pair_to_template(target_points, self.template, search_thresh)

        x_min = min(np.min(point_re[:, 0]), np.min(point_qc[:, 0]))
        x_max = max(np.max(point_re[:, 0]), np.max(point_qc[:, 0]))
        y_min = min(np.min(point_re[:, 1]), np.min(point_qc[:, 1]))
        y_max = max(np.max(point_re[:, 1]), np.max(point_qc[:, 1]))

        points_count = 0

        valid_temp = list()
        for point in self.template:
            if x_min <= point[0] <= x_max and \
                y_min <= point[1] <= y_max:
                points_count += 1
                valid_temp.append(point)

        if points_count / len(target_points) <= rate or \
                points_count - len(target_points) <= 10:  # 匹配点圈内点数量不多于1.2倍或不多于10个检点数
            return True
        elif points_count / len(target_points) <= 2 * rate or \
                points_count - len(target_points) <= 20:

            min_dif = np.inf
            for idx in range(len(self.chip_template[0])):
                re_count = len([pt for pt in point_re if pt[2] == idx])
                tp_count = len([pt for pt in valid_temp if pt[2] == idx])
                if tp_count > 0:
                    dif_rate = abs(re_count - tp_count) / tp_count
                    if dif_rate < min_dif:
                        min_dif = dif_rate

                re_count = len([pt for pt in point_re if pt[3] == idx])
                tp_count = len([pt for pt in valid_temp if pt[3] == idx])
                if tp_count > 0:
                    dif_rate = abs(re_count - tp_count) / tp_count
                    if dif_rate < min_dif:
                        min_dif = dif_rate

            if min_dif < 0.2:
                return True

        return False

    def _index_search(self, target_points, center_point=None):
        """
        Args:
            target_points: 单个FOV的点集
        Return:
            best_center_point: FOV中心点的坐标及索引
        """
        best_center_point = list()
        if center_point is None:
            center_point = self._center_point_search(target_points)[0]
        chip_len = len(self.chip_template[0])
        range_thresh = np.max(target_points[:, :2], axis=0)[::-1]
        min_distance = np.Inf
        # search_thresh = self.scale_x * np.max(self.chip_template) * 0.5
        for index_x in range(chip_len):
            for index_y in range(chip_len):
                _center_point = np.concatenate((center_point[:2], [index_x, index_y]))
                self._point_inference(_center_point, range_thresh)
                distance = self.pair_to_template(target_points, self.template, max(range_thresh), dis=True)
                _judge = self._valid_scale_judge(target_points, search_thresh=max(range_thresh))
                if _judge and (np.sum(distance) < min_distance):
                    min_distance = np.sum(distance)
                    best_center_point = _center_point

        return min_distance, best_center_point

    def get_scale(self, points, rotate, n=5):
        """
        尺度搜索
        Args:
            points:
            rotate:
            n: 点数
        Return:
            scale:
            best_point:
        """
        _min_points = points.min(axis =0)
        points = points - _min_points
        self.rotation = rotate
        if len(points) < 5: return None, None
        scale_list = [i / 10 for i in range(int(self.search_range[0] * 10),
                                            int(self.search_range[1] * 10 + 1))]
        cp_info = list()
        center_points = self._center_point_search(points, n=n)
        for cp in center_points:
            dis_info = list()
            for _scale in scale_list:
                self.scale_x = self.scale_y = _scale
                _distance, _best_center_point = self._index_search(points, center_point=cp)
                dis_info.append([_scale, _distance, _best_center_point])

            dis_info = sorted(dis_info, key=lambda x: x[1], reverse=False)
            cp_info.append(dis_info[0])

        scale_cp = [i[0] for i in cp_info]
        scale = max(scale_cp, key=scale_cp.count)
        _bp = [i[2] for i in cp_info if i[0] == scale][0]
        if len(_bp) == 0: best_point = None
        else: best_point = _bp + _min_points

        return scale, best_point


if __name__ == "__main__":
    import h5py
    import json

    chip_template = [[240, 300, 330, 390, 390, 330, 300, 240, 420],
                     [240, 300, 330, 390, 390, 330, 300, 240, 420]]
    ipr_path = r"D:\02.data\fengning\FP200009110_C2-B\FP200009110_C2-B_20240101_095929_0.1.ipr"
    pts = {}
    with h5py.File(ipr_path) as conf:
        qc_pts = conf['QCInfo/CrossPoints/']
        for i in qc_pts.keys():
            pts[i] = conf['QCInfo/CrossPoints/' + i][:]
    pts = sorted(pts.items(), key=lambda x: x[1].shape[0], reverse=True)
    for k, v in pts[1:]:
        rs = ScaleSearch(chip_template=chip_template)
        scale, best_point = rs.get_scale(v, 0)
        print(k, len(v), scale, best_point)
