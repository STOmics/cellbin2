from typing import Dict, Union, Optional
import numpy as np
import os
from pydantic import BaseModel

from cellbin2.modules.metadata import ProcFile
from cellbin2.modules import naming
from cellbin2.contrib.alignment import registration, RegistrationOutput, template_00pt_check
from cellbin2.utils.ipr import IFChannel, ImageChannel
from cellbin2.utils import clog
from cellbin2.contrib.alignment.basic import ChipFeature, AlignMode
from cellbin2.modules.extract.matrix_extract import extract4stitched
from cellbin2.utils.stereo_chip import StereoChip
from cellbin2.utils.config import Config
from cellbin2.contrib.alignment.basic import transform_points
from cellbin2.image import cbimread, cbimwrite
from cellbin2.utils.rle import RLEncode


class RegistrationParam(BaseModel):
    HE_channel: int
    rot90: bool
    flip: bool
    flag_pre_registration: bool
    flag_chip_registration: bool


def transform_to_register(
        cur_f_name: naming.DumpImageFileNaming,
        info: Optional[RegistrationOutput] = None,
        cur_c_image: Optional[Union[IFChannel, ImageChannel]] = None
):
    """
    Transforms and registers images based on provided parameters.

    Args:
        cur_f_name (naming.DumpImageFileNaming): The current file naming object.
        info (Optional[RegistrationOutput], optional): The registration output information. Defaults to None.
        cur_c_image (Optional[Union[IFChannel, ImageChannel]], optional): The current channel image. Defaults to None.
    """
    dct = {
        cur_f_name.transformed_image: cur_f_name.registration_image,
        cur_f_name.transform_cell_mask: cur_f_name.cell_mask,
        cur_f_name.transform_cell_mask_raw: cur_f_name.cell_mask_raw,
        cur_f_name.transform_tissue_mask: cur_f_name.tissue_mask,
        cur_f_name.transform_tissue_mask_raw: cur_f_name.tissue_mask_raw,
        # self._naming.transform_cell_correct_mask: self._naming.cell_correct_mask,
        cur_f_name.transformed_template: cur_f_name.register_template,
        cur_f_name.transformed_track_template: cur_f_name.register_track_template
    }
    if info is None and cur_c_image is None:
        for src, dst in dct.items():
            if not os.path.exists(src):
                continue
            os.rename(src, dst)
    else:
        for src, dst in dct.items():
            src_path = src
            dst_path = dst
            # if os.path.exists(dst_path):
            #     continue
            if os.path.exists(src_path):
                if os.path.splitext(src_path)[1] == ".txt":  # Or other judgment
                    points, _ = transform_points(
                        src_shape=cur_c_image.Stitch.TransformShape,
                        points=np.loadtxt(src_path),
                        rotation=(4 - info.counter_rot90) * 90,
                        flip=0 if info.flip else -1,
                        offset=info.offset
                    )
                    np.savetxt(dst_path, points)
                    if dst == cur_f_name.register_template:
                        cur_c_image.Register.RegisterTemplate = points
                    if dst == cur_f_name.register_track_template:
                        cur_c_image.Register.RegisterTrackTemplate = points
                else:
                    dst_image = cbimread(src_path).trans_image(
                        flip_lr=info.flip, rot90=info.counter_rot90, offset=info.offset,
                        dst_size=info.dst_shape)
                    cbimwrite(dst_path, dst_image)
        if os.path.exists(cur_f_name.tissue_mask):
            tissue_mask = cbimread(cur_f_name.tissue_mask, only_np=True)
            cur_c_image.TissueSeg.TissueSegShape = list(tissue_mask.shape)
            bmr = RLEncode()
            t_mask_encode = bmr.encode(tissue_mask)
            cur_c_image.TissueSeg.TissueMask = t_mask_encode
        if os.path.exists(cur_f_name.cell_mask):
            cell_mask = cbimread(cur_f_name.cell_mask, only_np=True)
            cur_c_image.CellSeg.CellSegShape = list(cell_mask.shape)
            bmr = RLEncode()
            c_mask_encode = bmr.encode(cell_mask)
            cur_c_image.CellSeg.CellMask = c_mask_encode


def run_register(
        image_file: ProcFile,
        cur_f_name: naming.DumpImageFileNaming,
        files: Dict[int, ProcFile],
        channel_images: Dict[str, Union[IFChannel, ImageChannel]],
        output_path: str,
        param_chip: StereoChip,
        config: Config,
        debug: bool
):
    """
    This module integrates the overall logic for image registration and
    returns registration parameters for downstream use.

    There are several scenarios:
    1. IF image: Returns the registration parameters of the reused image.
    2. Image + Matrix: Performs pre-registration, centroid method, and chip box registration.
    3. Image + Image: Not supported yet.

    Returns (RegisterOutput): Registration parameters
    """
    # TODO: The flip and rot90 switches have been passed in the config.
    #  Enable these switches internally in the registration process.
    clog.info(f"Running register module")
    sn = param_chip.chip_name

    g_name = image_file.get_group_name(sn=sn)
    param1 = channel_images[g_name]
    if image_file.registration.reuse != -1:
        f_name = files[image_file.registration.reuse].get_group_name(sn=sn)
        info = channel_images[f_name].get_registration()
        clog.info('Get registration param from ipr')
    else:

        """ Constructing parameters for the moving image """
        moving_image = ChipFeature(
            tech_type=image_file.tech,
        )
        moving_image.tech_type = image_file.tech
        moving_image.set_mat(cur_f_name.transformed_image)
        # It is recommended not to read from ipr here
        if param1.QCInfo.TrackCrossQCPassFlag:
            moving_image.set_template(param1.Stitch.TransformTemplate)  # param1.transform_template_info
        if param1.QCInfo.ChipDetectQCPassFlag:
            moving_image.set_chip_box(param1.Stitch.TransformChipBBox.get())

        """ Constructing parameters for the fixed image """
        fixed = files[image_file.registration.fixed_image]
        if fixed.is_matrix:
            # Scenario 1: The fixed image is a matrix
            cm = extract4stitched(
                image_file=fixed,
                param_chip=param_chip,
                m_naming=naming.DumpMatrixFileNaming(
                    sn=sn,
                    m_type=fixed.tech.name,
                    save_dir=output_path
                ),
                detect_feature=True,
                config=config
            )
            ##! 补充bin_size
            bin_size = 100
            if bin_size != 1:
                fixed_image = ChipFeature(
                    tech_type=fixed.tech,
                    template=cm.template,
                )
            else:
                fixed_image = ChipFeature(
                    tech_type=fixed.tech,
                    template=cm.template,
                    chip_box=cm.chip_box,
                )
            fixed_image.set_mat(cm.heatmap)
            param1.Register.MatrixTemplate = cm.template.template_points
            ##! 补充bin_size
            bin_size = 100
            if bin_size == 1:
                param1.Register.GeneChipBBox.update(fixed_image.chip_box)
        else:
            raise Exception("Not supported yet")

        """ Starting the registration process """
        if param1.Register.Method == AlignMode.Template00Pt.name:  # Pre-registration has been done previously
            # Get registration parameters from ipr
            pre_info = param1.Register.Register00.get().to_dict()

            ##! 补充bin_size
            bin_size = 100
            if bin_size != 1:
                from cellbin2.contrib.alignment.template_centroid import centroid
                _info = centroid(
                    moving_image=moving_image,
                    fixed_image=fixed_image,
                    ref=param_chip.fov_template,
                    from_stitched=False,
                    flip_flag=config.registration.flip,
                    rot90_flag=config.registration.rot90
                )
            else:    
                _info = template_00pt_check(
                    moving_image=moving_image,
                    fixed_image=fixed_image,
                    offset_info=pre_info,
                    fixed_offset=(cm.x_start, cm.y_start),
                    flip_flag=config.registration.flip,
                    rot90_flag=config.registration.rot90
                )
            info = RegistrationOutput(**_info)

        else:
            # TODO
            """
            Currently, both the centroid method and the chip box registration are performed because 
            the QC now considers the QC successful if either the template derivation or the chip box detection passes.
            
            Therefore, the following selection first performs the centroid method registration, 
            followed by the chip box registration.
            
            This change is being made to prepare for the mutual correction of 
            the two registration algorithms in the future.
            """
            chip_re = 1 if config.registration.flag_chip_registration and param1.QCInfo.ChipDetectQCPassFlag else 0

            info, temp_info = registration(
                moving_image=moving_image,
                fixed_image=fixed_image,
                ref=param_chip.fov_template,
                from_stitched=False,
                qc_info=(param1.QCInfo.TrackCrossQCPassFlag, chip_re),
                flip_flag=config.registration.flip,
                rot90_flag=config.registration.rot90
            )

            clog.info(f"Track cross registration: {info}")
            clog.info(f"Chip box registration: {temp_info}")

            if temp_info is not None and debug:
                temp_info.register_mat.write(
                    os.path.join(output_path, f"{sn}_chip_box_register.tif")
                )
                np.savetxt(
                    os.path.join(output_path, f"{sn}_chip_box_register.txt"),
                    temp_info.offset
                )
                param1.Register.RegisterChip.update(temp_info)

            # TODO Need to add ipr writing for chip box registration
            info = info if info is not None else temp_info

    param1.update_registration(info)
    transform_to_register(
        info=info,
        cur_f_name=cur_f_name,
        cur_c_image=param1
    )
