import numpy as np
from pydantic import BaseModel, Field
import os
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
from cellbin2.contrib.chip_transform import chip_transform


def read_transform(
        image_file: ProcFile,
        param_chip: StereoChip,
        channel_images: Dict[str, Union[IFChannel, ImageChannel]],
        files: Dict[int, ProcFile],
        research_mode: bool
):
    """
    Reads and applies transformations to an image based on various parameters.

    Args:
        image_file (ProcFile): The image file object.
        param_chip (StereoChip): The stereo chip parameters.
        channel_images (Dict[str, Union[IFChannel, ImageChannel]]): Dictionary of channel images.
        files (Dict[int, ProcFile]): Dictionary of files.
        research_mode (bool): Flag indicating if research mode is enabled.

    Returns:
        Tuple: A tuple containing scale factors (s), rotation angle (r), and offset.
    """
    c_name = image_file.get_group_name(sn=param_chip.chip_name)
    offset = (0, 0)
    if research_mode:
        clog.info(f"Research mode, calibration operation is considered")
        if image_file.channel_align != -1:  # If the image is a calibration image, switch to its alignment image first
            if channel_images[c_name].Calibration.CalibrationQCPassFlag:  # Calibration passed
                clog.info(f"Research mode, calibration qc flag passed, will perform calibration")
                # Perform translation first
                offset = (channel_images[c_name].Calibration.Scope.OffsetX,
                          channel_images[c_name].Calibration.Scope.OffsetY)
            else:  # Calibration failed, do not perform any operation rashly
                offset = (0, 0)
        else:
            offset = (0, 0)
    else:
        clog.info(f"Product mode, no calibration operation is performed ")
    s = (1., 1.)  # Default scale factors
    r = 0.  # Default rotation angle
    if image_file.registration.reuse == -1:
        # Get registration parameters
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
    """
    Apply transformations to an image based on parameters such as scale, rotation, and offset.

    Args:
        file: The processing file containing the image to be transformed.
        channel_images (dict): Dictionary mapping channel names to image channel objects.
        param_chip: Parameters for the stereo chip.
        files (dict): Dictionary mapping file IDs to processing files.
        cur_f_name: Naming convention object for file output.
        if_track (bool): Boolean flag indicating if tracking is enabled.
        research_mode (bool): Boolean flag indicating if research mode is enabled.
    """
    clog.info(f"Running transform module")
    # Read transformation parameters
    scale, rotation, offset = read_transform(
        image_file=file,
        param_chip=param_chip,
        channel_images=channel_images,
        files=files,
        research_mode=research_mode,
    )
    # Apply transformations to the image
    if file.chip_matching != -1:
        fixed_path = files[file.chip_matching].file_path
        clog.info('Transform [moving, fixed] == ({}, {})'.format(
            os.path.basename(file.file_path), os.path.basename(fixed_path)))
        chip_scale = [1, float(file.magnification) / files[file.chip_matching].magnification]

        registed_image_tmp = chip_transform(
            fixed_image=fixed_path,
            moving_image=file.file_path,
            scale=chip_scale,
        )
        transform_image = registed_image_tmp.trans_image(
            scale=scale, rotate=rotation, offset=offset
        )
    else:
        transform_image = cbimread(file.file_path).trans_image(
            scale=scale, rotate=rotation, offset=offset
        )
    trans_im_shape = transform_image.shape
    g_name = file.get_group_name(sn=param_chip.chip_name)
    image_info = channel_images[g_name]
    if if_track:
        # Transform stitch and QC template points
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
        # Transform stitch chip box
        stitch_chip_box = image_info.QCInfo.ChipBBox.get().chip_box
        trans_chip_box, _ = transform_points(
            src_shape=(info.GlobalHeight, info.GlobalWidth),
            scale=(1 / image_info.QCInfo.ChipBBox.ScaleX,
                   1 / image_info.QCInfo.ChipBBox.ScaleY),
            rotation=-image_info.QCInfo.ChipBBox.Rotation,
            points=stitch_chip_box,
            offset=offset
        )
        # Update transformed chip box info
        trans_chip_box_info = ChipBoxInfo()
        trans_chip_box_info.set_chip_box(trans_chip_box)
        trans_chip_box_info.IsAvailable = image_info.QCInfo.ChipBBox.IsAvailable
        trans_chip_box_info.ScaleX, trans_chip_box_info.ScaleY = 1.0, 1.0
        trans_chip_box_info.Rotation = 0.0

        # Update image info with transformed points and chip box
        image_info.Stitch.TrackPoint = qc_template
        image_info.Stitch.TransformTemplate = stitch_trans_template
        image_info.Stitch.TransformTrackPoint = qc_trans_template
        # Output transformed points to files
        np.savetxt(cur_f_name.transformed_template, image_info.Stitch.TransformTemplate)
        np.savetxt(cur_f_name.transformed_track_template, image_info.Stitch.TransformTrackPoint)
        image_info.Stitch.TransformChipBBox.update(trans_chip_box_info)
    # Update transformed image shape in image info
    image_info.Stitch.TransformShape = trans_im_shape
    # Write the transformed image to file
    transform_image.write(file_path=cur_f_name.transformed_image)
