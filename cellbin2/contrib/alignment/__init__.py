import numpy as np
from pydantic import BaseModel, Field
from typing import Tuple, Union, List, Any, Optional, Dict

from cellbin2.contrib.alignment.basic import AlignMode, ChipFeature, ChipBoxInfo
from cellbin2.contrib.alignment.chip_box import chip_align
from cellbin2.contrib.alignment.template_centroid import centroid
from cellbin2.contrib.alignment.template_00pt import template_00pt_align, template_00pt_check
from cellbin2.utils import clog


class RegistrationInput(BaseModel):
    moving_image: ChipFeature
    fixed_image: Optional[ChipFeature] = None
    ref: Tuple[List, List] = Field(..., description="模板周期，仅在模板相关配准方法下用到")
    dst_shape: Optional[Tuple[int, int]] = Field(None, description="the shape of fixed image ")
    from_stitched: bool
    rot90_flag: bool
    flip_flag: bool


class RegistrationOutput(BaseModel):
    counter_rot90: int = Field(0, description='')
    flip: bool = Field(True, description='')
    register_score: int = Field(-999, description='')
    # offset 多样原因为前置为暂定四个方向
    offset: Union[Dict, Tuple[float, float]] = Field((0., 0.), description='')
    register_mat: Any = Field(None, description='')
    method: AlignMode = Field(AlignMode.TemplateCentroid, description='')
    dst_shape: Tuple[int, int] = Field((0, 0), description='')


class Registration00Offset(BaseModel):
    offset: list
    dist: float

    class Config:
        arbitrary_types_allowed = True


class Registration00Output(BaseModel):
    rot0: Registration00Offset
    rot90: Registration00Offset
    rot180: Registration00Offset
    rot270: Registration00Offset

    def to_dict(self):
        info_dict = {
            0: {
                'offset': self.rot0.offset,
                'dist': self.rot0.dist
            },
            1: {
                'offset': self.rot90.offset,
                'dist': self.rot90.dist
            },
            2: {
                'offset': self.rot180.offset,
                'dist': self.rot180.dist
            },
            3: {
                'offset': self.rot270.offset,
                'dist': self.rot270.dist
            }
        }
        return info_dict


def registration(moving_image: ChipFeature,
                 fixed_image: ChipFeature,
                 ref: Tuple[List, List],
                 from_stitched: bool = True,
                 qc_info: tuple = (0, 0),
                 flip_flag: bool = True,
                 rot90_flag: bool = True,
                 ) -> (RegistrationOutput, RegistrationOutput):
    """
    :param moving_image: 待配准图，通常是染色图（如ssDNA、HE）
    :param fixed_image: 固定图，通常是矩阵，支持TIF/GEM/GEF及数组
    :param ref: 模板周期，仅在模板相关配准方法下用到
    :param from_stitched: 从拼接图配准
    :param qc_info: QC flag 信息
    :param flip_flag:
    :param rot90_flag:
    :return: RegistrationInfo
    """
    # TODO 临时兼容性改动
    #  11/22 by lizepeng
    if qc_info[0]:
        res_template = centroid(
            moving_image=moving_image,
            fixed_image=fixed_image,
            ref=ref,
            from_stitched=from_stitched,
            flip_flag = flip_flag,
            rot90_flag = rot90_flag
        )
    else:
        res_template = None

    if qc_info[1]:
        res_chip_box = chip_align(
            moving_image=moving_image,
            fixed_image=fixed_image,
            from_stitched=from_stitched,
            flip_flag = flip_flag,
            rot90_flag = rot90_flag
        )
    else:
        res_chip_box = None

    if res_template is not None:
        cent_info = RegistrationOutput(**res_template)
    else:
        cent_info = None

    if res_chip_box is not None:
        chip_info = RegistrationOutput(**res_chip_box)
    else:
        chip_info = None

    return cent_info, chip_info


def get_alignment_00(re_input: RegistrationInput) -> Registration00Output:
    res = template_00pt.template_00pt_align(
        moving_image = re_input.moving_image,
        ref = re_input.ref,
        dst_shape = re_input.dst_shape,
        from_stitched = re_input.from_stitched,
        flip_flag = re_input.flip_flag,
        rot90_flag = re_input.rot90_flag
    )
    new_res = {
        "rot0": res['offset'][0],
        "rot90": res['offset'][1],
        "rot180": res['offset'][2],
        "rot270": res['offset'][3],
        "offset": res['offset'],
        "method": res["method"],
        "dst_shape": res['dst_shape']
    }
    reg_o = Registration00Output(**new_res)
    return reg_o


if __name__ == '__main__':
    from cellbin2.image import cbimread
    from cellbin2.contrib.param import TemplateInfo, ChipBoxInfo
    from cellbin2.utils.common import TechType

    template_ref = ([240, 300, 330, 390, 390, 330, 300, 240, 420],
                    [240, 300, 330, 390, 390, 330, 300, 240, 420])

    # 移动图像信息
    moving_image = ChipFeature()
    moving_image.tech_type = TechType.DAPI
    moving_mat = cbimread(r'E:\03.users\liuhuanlin\01.data\cellbin2\stitch\A03599D1_DAPI.tif')
    moving_image.set_mat(moving_mat)
    img_tpl = TemplateInfo(template_recall=1.0, template_valid_area=1.0,
                           trackcross_qc_pass_flag=1, trackline_channel=0,
                           rotation=-0.53893, scale_x=1.0000665084, scale_y=1.00253792,
                           template_points=np.loadtxt(
                               r"E:/03.users/liuhuanlin/01.data/cellbin2/stitch/DAPI_matrix_template.txt"))
    moving_image.set_template(np.array(img_tpl))
    img_box = ChipBoxInfo(left_top=[162.28519045689168, 499.231306034147],
                          left_bottom=[377.99806165682605, 20502.069199051202],
                          right_top=[20210.76317636481, 314.47198219153387],
                          right_bottom=[20393.560877706364, 20277.53345880944],
                          scale_x=1.0006002898773978, scale_y=1.0028676122685343,
                          chip_size=(20004.000995228937, 20049.329304472536),
                          rotation=-0.5280016679897553,
                          is_available=True)
    moving_image.set_chip_box(img_box)

    # 固定对象信息
    fixed_image = ChipFeature()
    fixed_image.tech_type = TechType.Transcriptomics
    fixed_image.set_mat(r'E:\03.users\liuhuanlin\01.data\cellbin2\stitch\A03599D1_gene.tif')
    matrix_tpl = TemplateInfo(template_recall=1.0, template_valid_area=1.0,
                              trackcross_qc_pass_flag=1, trackline_channel=0,
                              rotation=0., scale_x=1., scale_y=1.,
                              template_points=np.loadtxt(
                                  r"E:/03.users/liuhuanlin/01.data/cellbin2/stitch/A03599D1_gene.txt"))
    fixed_image.set_template(np.array(matrix_tpl))
    matrix_box = ChipBoxInfo(left_top=[124., 1604.], left_bottom=[124., 21596.],
                             right_bottom=[20116., 21596.], right_top=[20116., 1604.])

    # 多种方案的测试
    methods = [AlignMode.TemplateCentroid, AlignMode.Template00Pt, AlignMode.ChipBox, AlignMode.Voting]
    for m in methods[:1]:
        info = registration(moving_image=moving_image, fixed_image=fixed_image,
                            ref=template_ref, mode=AlignMode.TemplateCentroid)
        print(m, info)
