import os.path
import numpy as np
from pydantic import BaseModel, Field

from cellbin2.contrib.template.point_detector import TrackPointsParam
from cellbin2.contrib.template.inferencev1.track_line import TrackLinesParam
from cellbin2.contrib.template.inferencev1.inference_v1 import TemplateReferenceV1Param
from cellbin2.contrib.template.inferencev2.inference_v2 import TemplateReferenceV2Param
from cellbin2.utils.common import iPlaceHolder, fPlaceHolder
from cellbin2.image.augmentation import line_enhance_method
from cellbin2.utils.common import TechType
from cellbin2.utils import clog
from typing import List


class TemplateInfo(BaseModel):
    template_points: np.ndarray = Field(np.array([]), description="All exported template points")
    template_recall: float = Field(
        fPlaceHolder, description="The proportion of identified track points that can be matched back to the template")
    template_valid_area: float = Field(fPlaceHolder, description="Proportion of track point distribution area")
    trackcross_qc_pass_flag: int = Field(
        iPlaceHolder, description="Whether the number and distribution of track points meet the requirements")
    trackline_channel: int = Field(iPlaceHolder, description="Detecting the channel of the track line")
    rotation: float = Field(fPlaceHolder, description="Horizontal angle of track line")
    scale_x: float = Field(fPlaceHolder, description="Horizontal scale")
    scale_y: float = Field(fPlaceHolder, description="Vertical scale")

    class Config:
        arbitrary_types_allowed = True


class TemplateReference(object):
    def __init__(self, ref: list,
                 stain_type: TechType,
                 track_points_config: TrackPointsParam,
                 track_lines_config: TrackLinesParam,
                 template_v1_config: TemplateReferenceV1Param,
                 template_v2_config: TemplateReferenceV2Param):
        # input: algorithm
        self.ref = ref
        self.stain_type: TechType = stain_type
        self.config_points: TrackPointsParam = track_points_config
        self.config_lines: TrackLinesParam = track_lines_config
        self.config_v1: TemplateReferenceV1Param = template_v1_config
        self.config_v2: TemplateReferenceV2Param = template_v2_config

        # input: data
        self.file_path: str = ''
        self.fov_w: int = 0
        self.fov_h: int = 0
        self.overlap: float = 0
        self.est_scale: float = 1.0

        # output
        # self.points_info: TrackPointsInfo
        # self.template_info: TemplateInfo = TemplateInfo()

    def _detect_track_points(self, ):
        from cellbin2.contrib.template import point_detector

        self.points_info = point_detector.run_detect(
            img_file=self.file_path,
            cfg=self.config_points,
            stain_type=self.stain_type,
            h=self.fov_h,
            w=self.fov_w,
            overlap=self.overlap,
            save_dir=None,  # 非必须
        )

    def _detect_track_lines(self, ):
        from cellbin2.contrib.template.inferencev1.track_line import TrackLineQC

        track_ln_pc = TrackLineQC(
            magnification=10,
            scale_range=0.8,
            channel=self.config_lines.channel,
            chip_template=self.ref,
        )

        track_info = sorted(self.points_info.track_points.items(),
                            key=lambda x: x[1].shape[0],
                            reverse=True)[:10]

        track_ln_pc.set_preprocess_func(
            line_enhance_method.get(self.stain_type, None)
        )

        line_fovs = {}
        for ti in track_info:
            r, c = map(int, ti[0].split("_"))
            track_loc = list(self.points_info.fov_location[r, c])
            track_loc = track_loc + list(map(lambda x: x + self.fov_w, track_loc))
            track_loc = [track_loc[i] for i in [1, 3, 0, 2]]
            line_fovs[ti[0]] = [track_loc, None]

        track_ln_pc.line_detect(line_fovs, self.file_path)
        track_ln_pc.track_match()

        line_result = track_ln_pc.matching_result
        line_score = track_ln_pc.score
        best_match = track_ln_pc.get_best_match()

        return line_score, best_match

    @staticmethod
    def _get_template_info(tr) -> TemplateInfo:
        max_value, mean_value, std_value, qc_conf, re_conf = tr.get_template_eval()
        global_diff = tr.get_global_eval()

        clog.info("Reference template: max=={:.3f}  mean=={:.3f}  "
                  "std=={:.3f}  qc_conf=={:.3f}  re_conf=={:.3f}".format(
            max_value, mean_value, std_value, qc_conf, re_conf)
        )
        tr.template = np.array(tr.template)  # need numpy array which can dump into h5 file automatically
        template_info = TemplateInfo(**{
            "scale_x": tr.scale_x,
            "scale_y": tr.scale_y,
            "rotation": tr.rotation,
            "template_points": tr.template,
            "template_recall": re_conf,
            "template_valid_area": max_value
        })
        return template_info

    def _inference_v1(self, match) -> TemplateInfo:
        from cellbin2.contrib.template.inferencev1 import TemplateReferenceV1

        scale_x, scale_y = match[2: 4]
        rotate = match[4]
        index, correct_points = match[0: 2]
        r, c = map(int, index.split("_"))
        correct_points = np.array(correct_points) + (
                self.points_info.fov_location[r, c].tolist() + [0, 0]
        )

        self.points_info.track_points[index] = correct_points

        tr = TemplateReferenceV1(cfg=self.config_v1)
        tr.set_scale(scale_x, scale_y)
        tr.set_rotate(rotate)
        tr.set_chipno(self.ref)
        tr.set_fov_location(self.points_info.fov_location)
        tr.set_qc_points(index, self.points_info.track_points)

        tr.first_template_correct(target_points=correct_points, index=index)
        tr.reference_template(mode='multi')

        return self._get_template_info(tr)

    def _inference_v2(
            self,
            rotate=None, scale_x=None, scale_y=None, method_threshold=0.1
    ) -> TemplateInfo:
        from cellbin2.contrib.template.inferencev2.inference_v2 import TemplateReferenceV2

        tr = TemplateReferenceV2(cfg=self.config_v1, cfg2=self.config_v2)

        if scale_x is not None and scale_y is not None:
            tr.set_scale(scale_x, scale_y)
        if rotate is not None:
            tr.set_rotate(rotate)

        tr.set_threshold_v2(scale_range = self.est_scale)
        tr.set_chipno(self.ref)
        tr.set_fov_location(self.points_info.fov_location)
        tr.set_qc_points(self.points_info.track_points)

        clog.info(f"Global template using method V2")
        tr.reference_template_v2(method_threshold=method_threshold)

        return self._get_template_info(tr)

    def inference(self, file_path: str, fov_wh: list, est_scale: float, overlap: float):
        self.file_path = file_path
        self.fov_w, self.fov_h = fov_wh
        self.overlap = overlap
        self.est_scale = est_scale

        # Step 1: Point detection
        self._detect_track_points()

        # Step 2: Based on point matching template
        template_info_v2 = self._inference_v2()

        # Step 3: If 2 fails, use the line matching template
        template_info_v1 = None
        # TODO: @dzh update threshold based on stain_type, lzp check this
        if self.stain_type == TechType.HE:
            thresh = self.config_v2.v2_HE_pass_thr
        else:
            thresh = self.config_v2.v2_ssDNA_pass_thr
        if template_info_v2.template_recall < thresh and template_info_v2.template_valid_area < thresh:
            score, match = self._detect_track_lines()
            if score == 1:
                template_info_v1 = self._inference_v1(match)

        if template_info_v1 is not None:
            template_info = template_info_v2 if \
                max(template_info_v2.template_recall,
                    template_info_v2.template_valid_area) > \
                max(template_info_v1.template_recall,
                    template_info_v1.template_valid_area) else template_info_v1
        else:
            template_info = template_info_v2

        if template_info.template_recall > thresh or template_info.template_valid_area > thresh:
            template_info.trackcross_qc_pass_flag = 1
        else:
            template_info.trackcross_qc_pass_flag = 0

        return self.points_info, template_info


def template_inference(file_path: str,
                       stain_type: TechType,
                       ref: List[List],
                       track_points_config: TrackPointsParam,
                       track_lines_config: TrackLinesParam,
                       template_v1_config: TemplateReferenceV1Param,
                       template_v2_config: TemplateReferenceV2Param,
                       overlap: float = 0.0,
                       fov_wh=[2000, 2000],
                       est_scale=1.0
                       ):
    """
    Template reference starting entry
    """
    clog.info('Next execute Template, it include module [Points detect, line detect, inference]')
    tr = TemplateReference(stain_type=stain_type,
                           ref=ref,
                           track_points_config=track_points_config,
                           track_lines_config=track_lines_config,
                           template_v1_config=template_v1_config,
                           template_v2_config=template_v2_config)

    points_info, template_info = \
        tr.inference(file_path=file_path, fov_wh=fov_wh, est_scale=est_scale, overlap=overlap)

    return points_info, template_info


def main():
    weight_root = r'E:\03.users\liuhuanlin\01.data\cellbin2\weights'
    file_path = r"E:\03.users\liuhuanlin\01.data\cellbin2\stitch\A03599D1_DAPI.tif"
    stain_type = TechType.ssDNA
    fov_wh = [2000, 2000]
    track_points_config = TrackPointsParam(**{
        'ssDNA_weights_path': os.path.join(weight_root, 'points_detect_yolov5obb_SSDNA_20220513_pytorch.onnx'),
        'DAPI_weights_path': os.path.join(weight_root, 'points_detect_yolov5obb_SSDNA_20220513_pytorch.onnx'),
        'HE_weights_path': os.path.join(weight_root, 'points_detect_yolov5obb_SSDNA_20220513_pytorch.onnx')
    })
    track_lines_config = TrackLinesParam(**{})

    points_info, template_info = template_inference(ref=[[240, 300, 330, 390, 390, 330, 300, 240, 420],
                                                         [240, 300, 330, 390, 390, 330, 300, 240, 420]],
                                                    track_points_config=track_points_config,
                                                    track_lines_config=track_lines_config,
                                                    template_v1_config=TemplateReferenceV1Param(**{}),
                                                    template_v2_config=TemplateReferenceV2Param(**{}),
                                                    file_path=file_path,
                                                    stain_type=stain_type,
                                                    fov_wh=fov_wh,
                                                    overlap=0)
    np.savetxt(r'E:\03.users\liuhuanlin\01.data\cellbin2\stitch\A03599D1_DAPI_template.txt',
               template_info.template_points)


if __name__ == '__main__':
    main()
