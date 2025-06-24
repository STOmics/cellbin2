###################################################
"""reference template v2 for image, must need QC data.
create by lizepeng, 2023/4/13 14:09
"""
####################################################

import numpy as np

from typing import List
from cellbin2.contrib.template.inferencev1.inference_v1 import TemplateReferenceV1, TemplateReferenceV1Param
from cellbin2.contrib.template.inferencev2.scale_search import ScaleSearch
from cellbin2.contrib.template.inferencev2.rotate_search import RotateSearch
from cellbin2.utils import clog
from pydantic import BaseModel, Field


class TemplateReferenceV2Param(BaseModel):
    v2_ssDNA_pass_thr: float = Field(0.1, description="")
    v2_HE_pass_thr: float = Field(0.01, description="")  # TODO: lzp add HE qc pass
    v2_scale_range_thr: List[float] = Field([0.3, 1.7], description="尺度适配范围")
    v2_rotate_range_thr: int = Field(35, description="角度适配范围")  # TODO：未用到，删除？@lizep
    v2_search_range_thr: int = Field(500, description="搜索阈值")  # TODO：未用到，删除？@lizep
    v2_rotate_fov_min_thr: int = Field(7, description="")
    v2_scale_limits: float = Field(0.5, description="Upper and lower limits")


class TemplateReferenceV2(TemplateReferenceV1):
    """
    模板推导算法V2
    分别通过点来拟合大致角度和尺度
    再通过初步校准得到局部区域的精确角度尺度值
    最终全局推导模板
    """

    MINIMIZE_METHOD = ['nelder-mead', 'slsqp', 'bfgs']

    def __init__(self, cfg: TemplateReferenceV1Param, cfg2: TemplateReferenceV2Param):
        super(TemplateReferenceV2, self).__init__(cfg)

        self.scale_range = cfg2.v2_scale_range_thr
        self.rotate_range = cfg2.v2_rotate_range_thr
        self.search_thresh = cfg2.v2_search_range_thr
        self.rotate_fov_min = cfg2.v2_rotate_fov_min_thr
        self.scale_limits = cfg2.v2_scale_limits

        self.set_scale_flag = False
        self.set_rotate_flag = False

        # 2023/08/28新增 用于开发对接track线模板FOV信息
        self.fov_index = None
        self.fov_best_point = None

    def get_fov_info(self, ):
        return self.fov_index, self.fov_best_point

    def set_threshold_v2(self,
                         scale_range=None,
                         rotate_range=None,
                         rotate_fov_min=None):
        """
        模板推导V2阈值
        """
        if scale_range is not None:
            if isinstance(scale_range, (int, float)):
                down_limit = scale_range - self.scale_limits
                self.scale_range = [
                    0.3 if down_limit < 0.3 else down_limit,
                    scale_range + self.scale_limits
                ]
            else:
                self.scale_range = scale_range
        if rotate_range is not None:
            self.rotate_range = rotate_range
        if rotate_fov_min is not None:
            self.rotate_fov_min = rotate_fov_min

    def set_scale(self, scale_x: float, scale_y: float):
        self.scale_x = self._to_digit(scale_x)
        self.scale_y = self._to_digit(scale_y)
        assert self.scale_x is not None and self.scale_y is not None, "Input is not a number."
        self.set_scale_flag = True

    def set_rotate(self, r: float):
        self.rotation = self._to_digit(r)
        assert self.rotation is not None, "Input is not a number."
        self.set_rotate_flag = True

    def set_qc_points(self, pts):
        """
        pts: {index: [x, y, ind_x, ind_y], ...}
        """
        if self.fov_loc_array is None:
            print("Please init global location.")
            return

        assert isinstance(pts, dict), "QC Points is error."
        for ind in pts.keys():
            points = np.array(pts[ind])
            points[:, :2] = np.round(points[:, :2], 2)
            self.qc_pts[ind] = points

    def _template_correct(self, qc_pts, n=5):
        """
        Args:
            qc_pts: 已按照点数排序的点集 max->min
            n: track角度遍历搜索所需FOV数量
        """
        rotate_list = list()
        point_list = list()
        best_point = None
        index = 0

        #####角度搜索
        if not self.set_rotate_flag:
            rs = RotateSearch()
            for pts in qc_pts[:n]:
                target_points = pts[1]
                rotate = rs.get_rotate(target_points)
                rotate_list.append(rotate)

            if len(rotate_list) == 0:
                return None

            if len(set(rotate_list)) == 1 and list(set(rotate_list))[0] is None:
                return None

            tmp_dict = dict()
            for i in set(rotate_list):
                tmp_dict[i] = rotate_list.count(i)
            tmp_list = sorted(tmp_dict, key=lambda x: tmp_dict[x], reverse=True)
            if tmp_list[0] is None:
                tmp_rot = tmp_list[1]
            else:
                tmp_rot = tmp_list[0]
            tmp_ind = rotate_list.index(tmp_rot)

            if rotate_list.count(tmp_rot) == 1:
                for rot in rotate_list:
                    if rot is None: continue
                    rp_count = rotate_list.count(rot + 1)
                    rm_count = rotate_list.count(rot - 1)
                    if rp_count * rm_count > 0:
                        continue
                    elif rp_count > 0:
                        tmp_rot = min([rot, rot + 1], key=rotate_list.index)
                        tmp_ind = rotate_list.index(tmp_rot)
                        break
                    elif rm_count > 0:
                        tmp_rot = min([rot, rot - 1], key=rotate_list.index)
                        tmp_ind = rotate_list.index(tmp_rot)
                        break
            self.rotation = tmp_rot
            # index = tmp_ind
            index_list = [i for i, r in enumerate(rotate_list) if r == self.rotation][:n]

        #####尺度搜索
        info_list = list()
        if not self.set_scale_flag:
            rs = ScaleSearch(chip_template=self.chip_no, search_range=self.scale_range)
            for id in index_list:
                scale, best_point = rs.get_scale(qc_pts[id][1], self.rotation)
                info_list.append([id, scale, best_point, self.rotation])

        return info_list

    def reference_template_v2(self, method_threshold=0.1):
        """模板推导算法V2"""
        self._check_parm()
        # self._qc_points_to_gloabal(all_points=True)

        if len(self.qc_pts) == 0:
            self.flag_skip_reference = True
            clog.info("QC track points is None, quit template reference.")
            return

        qc_pts = sorted(self.qc_pts.items(), key=lambda x: x[1].shape[0], reverse=True)
        # index, best_point = self._template_correct(qc_pts, self.rotate_fov_min)
        info_list = self._template_correct(qc_pts, self.rotate_fov_min)

        if info_list is None:
            self.flag_skip_reference = True
            clog.info("Rotate or Scale is None, quit template reference.")
            return

        best_fov_re_conf = 0
        for index, scale, best_point, rotation in info_list:
            if scale is None or best_point is None or rotation is None:
                continue
            try:
                clog.info(f"Reference template by {qc_pts[index][0]}")
                fov_flag = False
                self.fov_index = qc_pts[index][0]

                if len(best_point) == 0:
                    clog.info("Template reference failed.")
                    return

                self.scale_x = scale
                self.scale_y = scale
                self.rotation = rotation

                best_method_re_conf = 0
                for method in self.MINIMIZE_METHOD:
                    self.flag_skip_reference = False
                    self.set_minimize_method(method=method)
                    self.first_template_correct(target_points=qc_pts[index][1],
                                                index=qc_pts[index][0],
                                                center_points=best_point)

                    self.fov_best_point = self.template

                    clog.info(f"Use method {method}")
                    self.reference_template('multi')

                    valid_area, _, _, _, re_conf = self.get_template_eval()

                    if re_conf > best_method_re_conf:
                        best_method_result = [self.template.copy(),
                                              self.scale_x, self.scale_y, self.rotation,
                                              self.fov_index, self.fov_best_point.copy()]
                        best_method_re_conf = re_conf

                    if valid_area > method_threshold or re_conf > method_threshold:
                        fov_flag = True
                        break
                    else:
                        self.scale_x = scale
                        self.scale_y = scale
                        self.rotation = rotation
                        clog.info("Change reference template method then try again.")

                if best_method_re_conf > best_fov_re_conf:
                    best_fov_result = best_method_result
                    best_fov_re_conf = best_method_re_conf

                if fov_flag: break
            except Exception as e:
                clog.info(f"Error with {str(e)}")
                continue

        try:
            self.template, self.scale_x, self.scale_y, self.rotation, \
            self.fov_index, self.fov_best_point = best_fov_result
        except NameError:
            self.flag_skip_reference = True
            clog.info("Template is None, quit template reference.")
            return


if __name__ == '__main__':
    import h5py

    ipr_path = r"E:\hanqingju\BigChip_QC\SS200001018TR_C1D3\SS200001018TR_C1D3_20230524_175345_0.1.ipr"
    pts = {}
    with h5py.File(ipr_path) as conf:
        qc_pts = conf['QCInfo/CrossPoints/']
        for i in qc_pts.keys():
            pts[i] = conf['QCInfo/CrossPoints/' + i][:]
        loc = conf['Research/Stitch/StitchFovLocation'][...]

    chipno = [[240, 300, 330, 390, 390, 330, 300, 240, 420],
              [240, 300, 330, 390, 390, 330, 300, 240, 420]]

    # chipno = [[112, 144, 208, 224, 224, 208, 144, 112, 160],
    #           [112, 144, 208, 224, 224, 208, 144, 112, 160]]

    tr = TemplateReferenceV2()

    tr.set_chipno(chipno)
    tr.set_fov_location(loc)
    tr.set_qc_points(pts)

    tr.reference_template_v2()

    dct = tr.get_template_eval()
    mat = tr.get_global_eval()
    print(1)
