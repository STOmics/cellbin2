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


def transform_to_register(
        cur_f_name: naming.DumpImageFileNaming,
        info: Optional[RegistrationOutput] = None,
        cur_c_image: Optional[Union[IFChannel, ImageChannel]] = None
):
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
                if os.path.splitext(src_path)[1] == ".txt":  # 或其他判断
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
    这个模块的任务就是对配准整体逻辑的整合，返回一个下游要用的配准参数
    这里有以下几种情况：
    1. if图，返回reuse图的配准参数
    2. 影像图+矩阵：前置配准、重心法、芯片框配准
    3. 影像图+影像图：暂不支持

    返回（RegisterOutput）：配准参数
    """
    # TODO: config种已传入flip和rot90的开关，在registration，lzp在配准内部启用开关
    clog.info(f"Running register module")
    sn = param_chip.chip_name

    g_name = image_file.get_group_name(sn=sn)
    param1 = channel_images[g_name]
    if image_file.registration.reuse != -1:
        f_name = files[image_file.registration.reuse].get_group_name(sn=sn)
        info = channel_images[f_name].get_registration()
        clog.info('Get registration param from ipr')
    else:

        """ 动图参数构建 """
        moving_image = ChipFeature(
            tech_type=image_file.tech,
        )
        moving_image.tech_type = image_file.tech
        moving_image.set_mat(cur_f_name.transformed_image)
        # 这里建议不要去ipr读，而是
        if param1.QCInfo.TrackCrossQCPassFlag:
            moving_image.set_template(param1.Stitch.TransformTemplate)  # param1.transform_template_info
        if param1.QCInfo.ChipDetectQCPassFlag:
            moving_image.set_chip_box(param1.Stitch.TransformChipBBox.get())

        """ 静图参数构建 """
        fixed = files[image_file.registration.fixed_image]
        if fixed.is_matrix:
            # 场景1：静图是矩阵
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
            fixed_image = ChipFeature(
                tech_type=fixed.tech,
                template=cm.template,
                chip_box=cm.chip_box,
            )
            fixed_image.set_mat(cm.heatmap)
            param1.Register.MatrixTemplate = cm.template.template_points
            param1.Register.GeneChipBBox.update(fixed_image.chip_box)
        else:
            raise Exception("Not supported yet")

        """ 配准开始 """
        if param1.Register.Method == AlignMode.Template00Pt.name:  # 先前做了前置配准
            # 从ipr获取配准参数
            pre_info = param1.Register.Register00.get().to_dict()
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
                目前以下两个配准都做，因QC现在是模板推导和芯片框检测任意一个过了，都会判定QC成功
                所以下面的选择首先是重心法的配准，其次是芯片框配准
                这种改动是为了两种配准算法后续相互纠错做准备
            """
            info, temp_info = registration(
                moving_image=moving_image,
                fixed_image=fixed_image,
                ref=param_chip.fov_template,
                from_stitched=False,
                qc_info=(param1.QCInfo.TrackCrossQCPassFlag, param1.QCInfo.ChipDetectQCPassFlag),
                flip_flag=config.registration.flip,
                rot90_flag=config.registration.rot90
            )
            clog.info(f"{info}")
            clog.info(f"{temp_info}")
            if temp_info is not None and debug:
                temp_info.register_mat.write(
                    os.path.join(output_path, f"{sn}_chip_box_register.tif")
                )
                np.savetxt(
                    os.path.join(output_path, f"{sn}_chip_box_register.txt"),
                    temp_info.offset
                )
                param1.Register.RegisterChip.update(temp_info)

            # TODO 需要增加芯片框配准的ipr写入
            info = info if info is not None else temp_info

    param1.update_registration(info)
    transform_to_register(
        info=info,
        cur_f_name=cur_f_name,
        cur_c_image=param1
    )
