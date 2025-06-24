from typing import Tuple, Union, Dict
import os
import cv2
import numpy as np
from cellbin2.image import CBImage, cbimread, cbimwrite
from cellbin2.utils.common import TechType
from cellbin2.utils import ipr
from cellbin2.utils import clog
from cellbin2.modules.metadata import ProcFile
from cellbin2.utils.stereo_chip import StereoChip
from cellbin2.contrib.alignment import ChipBoxInfo
from cellbin2.utils.config import Config
from cellbin2.contrib import chip_detector
from cellbin2.contrib import clarity
from cellbin2.contrib import inference
from cellbin2.contrib.template.point_detector import TrackPointsInfo
from cellbin2.contrib.alignment import ChipFeature, RegistrationInput, get_alignment_00
from cellbin2.contrib.alignment.basic import AlignMode
from cellbin2.utils.plot_funcs import get_view_image


def scale_estimate(image_file, param_chip):
    image = cbimread(image_file.file_path)
    mx = max(image.width, image.height) / max(param_chip.width, param_chip.height)
    my = min(image.width, image.height) / min(param_chip.width, param_chip.height)
    return (mx + my) / 2


def estimate_fov_size(
        image_file: ProcFile,
        param_chip: StereoChip,
        fov_wh
) -> tuple:
    scale = scale_estimate(image_file, param_chip)  # 尺度估计
    clog.info('Using the image and chip prior size, calculate scale == {}'.format(scale))
    wh = (int(fov_wh[0] * scale), int(fov_wh[1] * scale))
    clog.info('Estimate1 FOV-WH from {} to {}'.format(fov_wh, wh))
    return wh, scale


def detect_chip(
        image_file: ProcFile,
        param_chip: StereoChip,
        config: Config,
        debug: bool,
        output_path: str,
) -> ChipBoxInfo:
    actual_size = param_chip.norm_chip_size
    # if debug is False, the returned dic debug_image_dic is empty
    info, debug_image_dic = chip_detector.detect_chip(file_path=image_file.file_path,
                                                      cfg=config.chip_detector,
                                                      stain_type=image_file.tech,
                                                      actual_size=actual_size,
                                                      is_debug=debug)
    if debug and len(debug_image_dic) != 0:
        enhance_img = debug_image_dic['enhance']
        left_up_img = debug_image_dic['left_up']
        left_down_img = debug_image_dic['left_down']
        right_down_img = debug_image_dic['right_down']
        right_up_img = debug_image_dic['right_up']

        tmp_img1 = cv2.vconcat([left_up_img, left_down_img])
        tmp_img2 = cv2.vconcat([right_up_img, right_down_img])

        result_img = cv2.hconcat([tmp_img1, tmp_img2])
        result_img = cv2.hconcat([enhance_img, result_img])

        cbimwrite(os.path.join(output_path, 'detect_chip_debug.tif'), result_img)

    return info


def run_clarity(
        image_file: ProcFile,
        config: Config
):
    c: clarity.ClarityOutput = clarity.run_detect(
        img_file=image_file.file_path,
        cfg=config.clarity,
        stain_type=image_file.tech
    )
    return c


def inference_template(
        cut_siz: Tuple[int, int],
        est_scale: float,
        image_file: ProcFile,
        param_chip: StereoChip,
        config: Config,
        overlap=0.0
) -> Tuple[TrackPointsInfo, inference.TemplateInfo]:
    points_info, template_info = inference.template_inference(
        ref=param_chip.fov_template,
        track_points_config=config.track_points,
        track_lines_config=config.track_lines,
        template_v1_config=config.template_ref_v1,
        template_v2_config=config.template_ref_v2,
        file_path=image_file.file_path,
        stain_type=image_file.tech,
        fov_wh=cut_siz,
        est_scale = est_scale,
        overlap=overlap)
    return points_info, template_info


def pre_registration(
        image_file: ProcFile,
        param_chip: StereoChip,
        channel_image: Union[ipr.ImageChannel, ipr.IFChannel],
        config: Config,
        output_path: str
):
    moving_image = ChipFeature(
        tech_type=image_file.tech,
        chip_box=channel_image.box_info,
        template=channel_image.stitched_template_info,
        point00=param_chip.zero_zero_point,
        anchor_point = param_chip.zero_zero_chip_point,
        mat=cbimread(image_file.file_path)
    )
    re_input = RegistrationInput(
        moving_image=moving_image,
        ref=param_chip.fov_template,
        dst_shape=(param_chip.height, param_chip.width),
        from_stitched=True,
        rot90_flag=config.registration.rot90,
        flip_flag=config.registration.flip
    )
    re_out = get_alignment_00(re_input=re_input)

    return re_out


def run_qc(
        image_file: ProcFile,
        param_chip: StereoChip,
        config: Config,
        output_path,
        debug: bool,
        fov_wh=(2000, 2000),
) -> Union[ipr.ImageChannel, ipr.IFChannel]:
    if image_file.tech is TechType.IF:
        channel_image = ipr.IFChannel()
    else:
        channel_image = ipr.ImageChannel()

    # 估计 & 第一次更新裁图尺寸
    cut_siz, est_scale = estimate_fov_size(
        image_file=image_file,
        param_chip=param_chip,
        fov_wh=fov_wh
    )

    if image_file.chip_detect:
        chip_info = detect_chip(
            image_file=image_file,
            param_chip=param_chip,
            config=config,
            debug=debug,
            output_path=output_path
        )
        channel_image.QCInfo.ChipBBox.update(box=chip_info)
        channel_image.QCInfo.ChipDetectQCPassFlag = 1 if chip_info.IsAvailable else 0
        if chip_info.IsAvailable:
            # 第二次更新裁图尺寸
            scale = (chip_info.ScaleY + chip_info.ScaleX) / 2
            clog.info('Using the image chip box, calculate scale == {}'.format(scale))
            cut_siz = (int(fov_wh[0] * scale), int(fov_wh[1] * scale))
            clog.info('Estimate2 FOV-WH from {} to {}'.format(fov_wh, cut_siz))
    channel_image.ImageInfo.FOVHeight = cut_siz[1]
    channel_image.ImageInfo.FOVWidth = cut_siz[0]
    if image_file.quality_control:
        c = run_clarity(
            image_file=image_file,
            config=config
        )
        channel_image.QCInfo.update_clarity(c)

    if image_file.registration.trackline:
        points_info, template_info = inference_template(
            cut_siz=cut_siz,
            est_scale = est_scale,
            image_file=image_file,
            param_chip=param_chip,
            config=config,
        )
        channel_image.update_template_points(points_info=points_info, template_info=template_info)
        if template_info.trackcross_qc_pass_flag:
            channel_image.QCInfo.TrackCrossQCPassFlag = 1
            clog.info('Template Scale is {}, rotation is {}'.format(
                (template_info.scale_x, template_info.scale_y), template_info.rotation))

    if image_file.chip_detect and param_chip.is_after_230508():  # 满足配准前置的条件
        if chip_info.IsAvailable and template_info.trackcross_qc_pass_flag:
            clog.info('The chip-data meets the pre-registration conditions')
            pre_out = pre_registration(
                image_file=image_file,
                param_chip=param_chip,
                channel_image=channel_image,
                output_path=output_path,
                config=config
            )
            channel_image.Register.Register00.update(pre_out)
            channel_image.Register.Method = AlignMode.Template00Pt.name
    # TODO 芯片框QC恒为0 25/03/04
    cpf = 0
    # cpf = 1 if channel_image.QCInfo.ChipDetectQCPassFlag == 1 else 0
    tcf = 1 if channel_image.QCInfo.TrackCrossQCPassFlag == 1 else 0
    channel_image.QCInfo.QCPassFlag = (cpf or tcf)

    clog.info('ImageQC result is {}'.format(channel_image.QCInfo.QCPassFlag))
    return channel_image
