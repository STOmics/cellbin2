import numpy as np
from cellbin2.contrib.alignment.basic import Alignment, AlignMode, ChipFeature
from cellbin2.utils.common import TechType
from cellbin2.contrib.track_align import AlignByTrack
from typing import List, Tuple


class TemplateCentroid(Alignment):
    """
    满足CellBin需求，利用模板周期性及组织形态，通过遍历实现变换参数的获取。实现配准，误差约10pix
    """
    def __init__(
            self,
            ref: Tuple[List, List] = ([], []),
            flip_flag: bool = True,
            rot90_flag: bool = True
    ):
        super(TemplateCentroid, self).__init__()
        self._reference = ref

        self._hflip = flip_flag
        self._rot90_flag = rot90_flag

    def _mask_process(
            self,
            cf: ChipFeature,
            mask_flag: bool
    ) -> np.ndarray:
        """

        Args:
            cf: ChipFeature

        Returns:

        """
        if not mask_flag: return cf.mat.image

        if cf.chip_box.IsAvailable:
            image = self._fill_image(cf.mat.image, cf.chip_box.chip_box)
        else:
            image = cf.mat.image

        return image

    def align_stitched(self, fixed_image: ChipFeature, moving_image: ChipFeature):
        """

        Args:
            fixed_image:
            moving_image:

        Returns:

        """
        self._scale_x, self._scale_y = moving_image.template.scale_x, moving_image.template.scale_y
        self._rotation = -moving_image.template.rotation
        self._fixed_image = fixed_image

        transformed_image = self.transform_image(file=moving_image.mat)

        transformed_feature = ChipFeature()
        transformed_feature.set_mat(transformed_image)

        trans_mat = self.get_coordinate_transformation_matrix(
            moving_image.mat.shape,
            [1 / self._scale_x, 1 / self._scale_y],
             -self._rotation
        )

        trans_points = self.get_points_by_matrix(
            np.array(moving_image.template.template_points),
            trans_mat
        )

        transformed_feature.set_template(
            np.concatenate(
                [trans_points, np.array(moving_image.template.template_points)[:, 2:]],
                axis=1
            )
        )

        self.align_transformed(fixed_image, transformed_feature)

    def align_transformed(
            self,
            fixed_image: ChipFeature,
            moving_image: ChipFeature,
            mask_flag = True
    ):
        """

        Args:
            fixed_image:
            moving_image:
            mask_flag: whether or not mask coverage through chip box

        Returns:

        """

        abt = AlignByTrack()
        abt.set_chip_template(chip_template=self._reference)

        mi = self._mask_process(moving_image, mask_flag)

        self._offset, self._rot90, score = abt.run(
            mi, fixed_image.mat.image,
            np.array(fixed_image.template.template_points),
            np.array(moving_image.template.template_points),
            self.hflip,
            self._rot90_flag,
            new_method=True if moving_image.tech_type == TechType.HE else False
        )


def centroid(moving_image: ChipFeature,
             fixed_image: ChipFeature,
             ref: Tuple[List, List],
             from_stitched: bool = True,
             flip_flag: bool = True,
             rot90_flag: bool = True
             ):
    """
    :param moving_image: 待配准图，通常是染色图（如ssDNA、HE）
    :param fixed_image: 固定图，通常是矩阵，支持TIF/GEM/GEF及数组
    :param ref: 模板周期，仅在模板相关配准方法下用到
    :param from_stitched: 从拼接图开始
    :param flip_flag:
    :param rot90_flag:
    :return: RegistrationInfo
    """
    ta = TemplateCentroid(ref=ref, flip_flag=flip_flag, rot90_flag=rot90_flag)
    if moving_image.tech_type is TechType.HE:
        from cellbin2.image.augmentation import f_rgb2hsv
        moving_image.set_mat(mat=f_rgb2hsv(moving_image.mat.image, channel=1, need_not=False))
    if from_stitched:
        ta.align_stitched(fixed_image=fixed_image, moving_image=moving_image)
    else:
        ta.align_transformed(fixed_image=fixed_image, moving_image=moving_image)

    info = {
            'offset': tuple(list(ta.offset)),
            'counter_rot90': ta.rot90,
            'flip': ta.hflip,
            'register_score': ta.score,
            'dst_shape': (fixed_image.mat.shape[0], fixed_image.mat.shape[1]),
            'method': AlignMode.TemplateCentroid
        }

    return info


if __name__ == '__main__':
    from cellbin2.image import cbimread, cbimwrite
    from cellbin2.contrib.param import TemplateInfo

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
                           template_points=
                           np.loadtxt(r"E:/03.users/liuhuanlin/01.data/cellbin2/stitch/DAPI_matrix_template.txt"))
    moving_image.set_template(img_tpl)

    # 固定对象信息
    fixed_image = ChipFeature()
    fixed_image.tech_type = TechType.Transcriptomics
    fixed_image.set_mat(r'E:\03.users\liuhuanlin\01.data\cellbin2\stitch\A03599D1_gene.tif')
    matrix_tpl = TemplateInfo(template_recall=1.0, template_valid_area=1.0,
                              trackcross_qc_pass_flag=1, trackline_channel=0,
                              rotation=0., scale_x=1., scale_y=1.,
                              template_points=
                              np.loadtxt(r"E:/03.users/liuhuanlin/01.data/cellbin2/stitch/A03599D1_gene.txt"))
    fixed_image.set_template(matrix_tpl)

    info = centroid(moving_image=moving_image, fixed_image=fixed_image, ref=template_ref)
    print(info)
    cbimwrite(r'E:\03.users\liuhuanlin\01.data\cellbin2\stitch\A03599D1_DAPI_regist.tif', info.register_mat)
