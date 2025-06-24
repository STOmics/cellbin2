import numpy as np
from pydantic import BaseModel, Field
from typing import Dict, Tuple, Union, Optional

from cellbin2.image import cbimread, CBImage
from cellbin2.utils.ipr import ImageChannel, IFChannel
from cellbin2.contrib.alignment.basic import transform_points
from cellbin2.contrib.alignment.basic import ChipBoxInfo
from cellbin2.utils.common import TechType
from cellbin2.utils.stereo_chip import StereoChip
from cellbin2.modules.metadata import ProcParam, ProcFile
from cellbin2.modules import naming
from cellbin2.utils import clog


def read_transform(
        image_file: ProcFile,
        param_chip: StereoChip,
        channel_images: Dict[str, Union[IFChannel, ImageChannel]],
        files: Dict[int, ProcFile],
        research_mode: bool
):
    c_name = image_file.get_group_name(sn=param_chip.chip_name)
    offset = (0, 0)
    if research_mode:
        clog.info(f"Research mode, calibration operation is considered")
        if image_file.channel_align != -1:  # 如果该图是校准图，则先切换到其对齐图上
            if channel_images[c_name].Calibration.CalibrationQCPassFlag:  # 校准通过
                clog.info(f"Research mode, calibration qc flag passed, will perform calibration")
                # 先平移
                offset = (channel_images[c_name].Calibration.Scope.OffsetX,
                          channel_images[c_name].Calibration.Scope.OffsetY)
            else:  # 校准不通过，不贸然操作
                offset = (0, 0)
        else:
            offset = (0, 0)
    else:
        clog.info(f"Product mode, no calibration operation is performed ")
    s = (1., 1.)  # default
    r = 0.  # default
    if image_file.tech != TechType.IF:
        # 获取配准参数
        if channel_images[c_name].QCInfo.TrackCrossQCPassFlag:
            s = (1 / channel_images[c_name].Register.ScaleX,
                 1 / channel_images[c_name].Register.ScaleY)
            r = channel_images[c_name].Register.Rotation
        else:
            s = (1 / channel_images[c_name].QCInfo.ChipBBox.ScaleX,
                 1 / channel_images[c_name].QCInfo.ChipBBox.ScaleY)
            r = channel_images[c_name].QCInfo.ChipBBox.Rotation
    else:
        reuse_channel = image_file.registration.reuse
        reuse_g_name = files[reuse_channel].get_group_name(sn=param_chip.chip_name)
        if reuse_channel != -1:
            s = (
                1 / channel_images[reuse_g_name].Register.ScaleX,
                1 / channel_images[reuse_g_name].Register.ScaleY
            )
            r = channel_images[reuse_g_name].Register.Rotation

    return s, r, offset


def run_transform(
        file: ProcFile,
        channel_images: Dict[str, Union[IFChannel, ImageChannel]],
        param_chip: StereoChip,
        files: Dict[int, ProcFile],
        cur_f_name: naming.DumpImageFileNaming,
        if_track: bool,
        research_mode: bool
):
    clog.info(f"Running transform module")
    scale, rotation, offset = read_transform(
        image_file=file,
        param_chip=param_chip,
        channel_images=channel_images,
        files=files,
        research_mode=research_mode,
    )
    transform_image = cbimread(file.file_path).trans_image(
        scale=scale, rotate=rotation, offset=offset
    )
    trans_im_shape = transform_image.shape
    g_name = file.get_group_name(sn=param_chip.chip_name)
    image_info = channel_images[g_name]
    if if_track:
        info = image_info.Stitch.ScopeStitch
        stitch_template = image_info.Stitch.TemplatePoint
        qc_template = image_info.QCInfo.CrossPoints.stack_points
        stitch_trans_template, _ = transform_points(
            src_shape=(info.GlobalHeight, info.GlobalWidth),
            scale=scale, rotation=-rotation,
            points=stitch_template,
            offset=offset
        )

        qc_trans_template, _ = transform_points(
            src_shape=(info.GlobalHeight, info.GlobalWidth),
            scale=scale, rotation=-rotation,
            points=qc_template,
            offset=offset
        )
        stitch_chip_box = image_info.QCInfo.ChipBBox.get().chip_box
        trans_chip_box, _ = transform_points(
            src_shape=(info.GlobalHeight, info.GlobalWidth),
            scale=(1 / image_info.QCInfo.ChipBBox.ScaleX,
                   1 / image_info.QCInfo.ChipBBox.ScaleY),
            rotation=-image_info.QCInfo.ChipBBox.Rotation,
            points=stitch_chip_box,
            offset=offset
        )
        trans_chip_box_info = ChipBoxInfo()
        trans_chip_box_info.set_chip_box(trans_chip_box)
        trans_chip_box_info.IsAvailable = image_info.QCInfo.ChipBBox.IsAvailable
        trans_chip_box_info.ScaleX, trans_chip_box_info.ScaleY = 1.0, 1.0
        trans_chip_box_info.Rotation = 0.0

        image_info.Stitch.TrackPoint = qc_template
        image_info.Stitch.TransformTemplate = stitch_trans_template
        image_info.Stitch.TransformTrackPoint = qc_trans_template
        # 输出：参数写入ipr、txt、tif
        np.savetxt(cur_f_name.transformed_template, image_info.Stitch.TransformTemplate)
        np.savetxt(cur_f_name.transformed_track_template, image_info.Stitch.TransformTrackPoint)
        image_info.Stitch.TransformChipBBox.update(trans_chip_box_info)
    image_info.Stitch.TransformShape = trans_im_shape
    transform_image.write(file_path=cur_f_name.transformed_image)
