import numpy as np

from typing import List, Tuple, Union, Dict
from scipy.spatial.distance import cdist

from cellbin2.utils.common import TechType
from cellbin2.contrib.track_align import AlignByTrack
from cellbin2.contrib.alignment.basic import Alignment, ChipFeature, transform_points
from cellbin2.contrib.alignment import AlignMode
from cellbin2.image import CBImage, cbimread
from cellbin2.utils import clog


class Template00PtAlignment(Alignment):
    """
    Satisfy CellBin's requirements and utilize chip cutting characteristics to achieve alignment
    of the starting point of the first cycle in the upper left corner. Implement registration with an error of 10pix
    """

    def __init__(self,
                 ref: Tuple[List, List] = ([], []),
                 shape: Tuple[int, int] = (0, 0),
                 flip_flag: bool = True,
                 rot90_flag: bool = True
                 ):
        super(Template00PtAlignment, self).__init__()
        self._reference = ref
        self._register_shape = shape
        self._hflip = flip_flag
        self._rot90_search_flag = rot90_flag

        self.fixed_template: np.ndarray = np.array([])
        self.fixed_box: List[float] = [0, 0, 0, 0]
        self.offset_info: Dict[int, dict] = {}

    @staticmethod
    def _rot90_points(
            points: np.ndarray,
            shape: Union[List, Tuple],
            ind: int,
            reference: Tuple[List, List]
    ) -> np.ndarray:
        """

        Args:
            points: array -- n * 4
            shape: h, w
            ind: 0 1 2 3 -- rot90
            reference:

        Returns:

        """
        ind = ind % 4

        if ind == 0: return points.copy()

        _points, _ = transform_points(
            points[:, :2], shape, rotation = 90 * ind
        )

        k = len(reference[0]) - 1
        if ind == 1: _xy_ind = np.abs(points[:, 2:][:, ::-1] - [k, 0])
        elif ind == 2: _xy_ind = np.abs(points[:, 2:][:, ::-1] - [k, k])
        else: _xy_ind = np.abs(points[:, 2:][:, ::-1] - [0, k])

        new_points = np.concatenate([np.array(_points), _xy_ind], axis = 1)
        return new_points

    def registration_image(
            self,
            file: Union[str, np.ndarray, CBImage]
    ) -> CBImage:
        """
        Starting from the stitching diagram
        Args:
            file:

        Returns:

        """
        if not isinstance(file, CBImage):
            image = cbimread(file)
        else:
            image = file

        offset_info = sorted(self.offset_info.items(), key = lambda x: x[1]['dist'])[0]
        rot90 = offset_info[0]
        offset = offset_info[1]['offset']

        result = image.trans_image(
            scale=[1 / self._scale_x, 1 / self._scale_y],
            rotate=self._rotation,
            rot90=4 - rot90,
            offset=offset,
            dst_size=self._register_shape,
            flip_ud=True  # default pre-registration: vertical flip for alignment 
        )

        return result

    def align_stitched(self, moving_image: ChipFeature, ):
        """

        Args:
            moving_image:

        Returns:

        """
        self._scale_x, self._scale_y = moving_image.template.scale_x, moving_image.template.scale_y
        self._rotation = -moving_image.template.rotation

        transformed_image = self.transform_image(file=moving_image.mat)

        transformed_feature = ChipFeature()
        transformed_feature.set_mat(transformed_image)

        trans_mat = self.get_coordinate_transformation_matrix(
            moving_image.mat.shape,
            [1 / self._scale_x, 1 / self._scale_y],
            self._rotation
        )

        trans_points = self.get_points_by_matrix(
            np.array(moving_image.template.template_points),
            trans_mat
        )

        chip_points = self.get_points_by_matrix(
            moving_image.chip_box.chip_box,
            trans_mat
        )

        transformed_feature.set_point00(moving_image.point00)
        transformed_feature.set_anchor_point(moving_image.anchor_point)
        transformed_feature.chip_box.set_chip_box(chip_points)
        transformed_feature.set_template(
            np.concatenate(
                [trans_points, np.array(moving_image.template.template_points)[:, 2:]],
                axis=1
            )
        )

        self.align_transformed(transformed_feature)

    def align_transformed(self, moving_image: ChipFeature):
        """

        Args:
            moving_image:

        Returns:

        """
        if self.hflip:
            points = AlignByTrack.flip_points(
                moving_image.template.template_points,
                moving_image.mat.shape, self._reference,
                axis=1
            )
            chip_box_points = self.transform_points(
                points=moving_image.chip_box.chip_box,
                shape=moving_image.mat.shape,
                flip=1)
        else:
            points = moving_image.template.template_points
            chip_box_points = moving_image.chip_box.chip_box

        if self._rot90_search_flag: rot90_ind = [0, 1, 2, 3]
        else: rot90_ind = [0]

        for ri in rot90_ind:

            _points = self._rot90_points(
                points,
                moving_image.mat.shape,
                ind=ri,
                reference = self._reference
            )

            new_chip_points, _ = transform_points(
                chip_box_points,
                moving_image.mat.shape,
                rotation = 90 * ri
            )

            new_chip_points = self.check_border(new_chip_points)
            # _points = points.copy()
            _points[:, :2] = _points[:, :2] - new_chip_points[0]

            _points = self.get_lt_zero_point(_points)
            _points = _points[(_points[:, 0] > 0) & (_points[:, 1] > 0)]

            # px, py = sorted(
            #     _points.tolist(),
            #     key=lambda x: np.abs(x[0] - moving_image.anchor_point[0]) + np.abs(x[1] - moving_image.anchor_point[1])
            # )[0] + chip_box_points[0]

            _dist = cdist(_points, np.array([moving_image.anchor_point]))
            index = np.argmin(_dist)
            min_dist = _dist[index]
            px, py = _points[index] + new_chip_points[0]
            offset = [moving_image.point00[0] - px, moving_image.point00[1] - py]
            self.offset_info[ri] = {'offset': offset, 'dist': min_dist}

    @staticmethod
    def get_lt_zero_point(template_points, x_index=0, y_index=0):
        """
        Args:
            template_points: np.array, template points -- shape == (*, 4)
            x_index:
            y_index:
        Returns:
            zero_template_points: np.array
        """
        zero_template_points = template_points[(template_points[:, 3] == y_index) &
                                               (template_points[:, 2] == x_index)][:, :2]
        return zero_template_points


def template_00pt_check(
        moving_image: ChipFeature,
        fixed_image: ChipFeature,
        offset_info: dict,
        fixed_offset: tuple = (0, 0),
        flip_flag: bool = True,
        rot90_flag: bool = True,
        max_length: int = 9996
) -> dict:
    """

    Args:
        moving_image:
        fixed_image:
        offset_info:
        fixed_offset: Starting xy information of matrix image
        flip_flag:
        rot90_flag:
        max_length:

    Returns:

    """
    if moving_image.tech_type is TechType.HE:
        from cellbin2.image.augmentation import f_rgb2hsv
        moving_image.set_mat(
            mat=f_rgb2hsv(moving_image.mat.image, channel=1, need_not=False)
        )

    # TODO: now ,binX use the Template00Pt register offset directly without calibration.  Further calibration is required.
    if not rot90_flag or fixed_image.binX != 1:
        return {
            'offset': (offset_info[0]["offset"] - np.array(fixed_offset)).tolist(),
            'flip': flip_flag,
            'register_score': -1,
            'counter_rot90': 2,
            'method': AlignMode.Template00Pt,
            'dst_shape': (fixed_image.mat.shape[0], fixed_image.mat.shape[1])
        }

    offset_info = sorted(offset_info.items(), key = lambda x: x[1]['dist'])

    down_size = max(fixed_image.mat.shape) // max_length

    lu_x, lu_y = map(int, fixed_image.chip_box.chip_box[0] / down_size)
    rd_x, rd_y = map(int, fixed_image.chip_box.chip_box[2] / down_size)
    _gene_image = fixed_image.mat.image[::down_size, ::down_size][lu_y: rd_y, lu_x:rd_x]

    temp_cbi = CBImage(Alignment._fill_image(moving_image.mat.image, moving_image.chip_box.chip_box))
    moving_image.mat = temp_cbi

    if flip_flag: mm = moving_image.mat.trans_image(flip_ud = True)
    else: mm = moving_image.mat

    register_info = dict()
    for rot_ind, _info in offset_info:
        offset = _info["offset"]
        _offset = (np.array(offset) - np.array(fixed_offset)).tolist()
        register_image = mm.trans_image(
            rot90 = (4 - rot_ind) % 4 ,
            offset = _offset,
            dst_size = fixed_image.mat.shape
        )

        _register_image = register_image.resize_image(1 / down_size).image
        _wsi_image = _register_image[lu_y: rd_y, lu_x:rd_x]

        ms = Alignment.multiply_sum(_wsi_image, _gene_image)

        register_info[rot_ind] = {"score": ms, "offset": _offset}
        clog.info(f"Rot{rot_ind * 90}, Score: {ms}")

    best_info = sorted(register_info.items(), key = lambda x: x[1]["score"], reverse = True)[0]

    rot90_count = best_info[0]
    rot90_count = rot90_count if rot90_count % 2 == 1 else np.abs(rot90_count - 2)

    check_info = {
        'offset': best_info[1]["offset"],
        'flip': flip_flag,
        'register_score': best_info[1]["score"],
        'counter_rot90': rot90_count,
        # 'register_mat': tpa.registration_image(moving_image.mat),
        'method': AlignMode.Template00Pt,
        'dst_shape': (fixed_image.mat.shape[0], fixed_image.mat.shape[1])
    }
    return check_info


def template_00pt_align(
        moving_image: ChipFeature,
        ref: Tuple[List, List],
        dst_shape: Tuple[int, int],
        from_stitched: bool = True,
        flip_flag: bool = True,
        rot90_flag: bool = True
):
    """
    :param moving_image: The image to be registered is usually a stained image (such as ssDNA, HE)
    :param ref: Template cycle, only used in template related registration methods
    :param dst_shape: Theoretical size of registration map
    :param from_stitched
    :param flip_flag:
    :param rot90_flag:
    :return: dict
    """
    tpa = Template00PtAlignment(ref=ref, shape=dst_shape, flip_flag=flip_flag, rot90_flag=rot90_flag)
    if from_stitched:
        tpa.align_stitched(moving_image=moving_image)
    else:
        tpa.align_transformed(moving_image=moving_image)

    info = {
        # 'offset': tuple(list(tpa.offset)),
        'offset': tpa.offset_info,
        'flip': tpa.hflip,
        'register_score': tpa.score,
        'register_mat': tpa.registration_image(moving_image.mat),
        'method': AlignMode.Template00Pt,
        'dst_shape': dst_shape
    }

    return info


if __name__ == '__main__':
    import os
    from cellbin2.image import cbimread, cbimwrite
    from cellbin2.contrib.template.inference import TemplateInfo
    from cellbin2.utils.stereo_chip import StereoChip
    from cellbin2.contrib.chip_detector import ChipParam, detect_chip

    template_ref = ([240, 300, 330, 390, 390, 330, 300, 240, 420],
                    [240, 300, 330, 390, 390, 330, 300, 240, 420])

    import pickle
    with open(r"D:\02.data\temp\A03599D1\00pt\moving.pickle", "rb") as file:
        moving = pickle.load(file)
    info = template_00pt_align(moving_image = moving, ref = template_ref, dst_shape = (23520, 20580))
    # tpa = Template00PtAlignment()
    # aaa = tpa._rot90_points(
    #     points=np.loadtxt(r"D:\02.data\temp\A03599D1\cellbin2\A03599D1_Transcriptomics_matrix_template.txt"),
    #     shape=[23494, 20580],
    #     ind=3,
    #     reference = template_ref
    # )

    # move image 
    moving_image = ChipFeature()
    moving_image.tech_type = TechType.DAPI
    moving_mat = cbimread(r"E:\03.users\liuhuanlin\01.data\cellbin2\stitch\A03599D1_gene.tif")
    moving_image.set_mat(moving_mat)
    img_tpl = TemplateInfo(template_recall=1.0, template_valid_area=1.0,
                           trackcross_qc_pass_flag=1, trackline_channel=0,
                           rotation=0.11493999999999997, scale_x=0.9797232231120153, scale_y=0.978182155478651,
                           template_points=np.loadtxt(
                               r"E:\03.users\liuhuanlin\01.data\cellbin2\stitch\DAPI_matrix_template.txt"))
    moving_image.set_template(img_tpl)

    cfg = ChipParam(
        **{"DAPI_stage1_weights_path":
               r"E:/03.users/liuhuanlin/01.data/cellbin2/weights\chip_detect_obb8n_640_SD_202409_pytorch.onnx",
           "DAPI_stage2_weights_path":
               r"E:/03.users/liuhuanlin/01.data/cellbin2/weights\chip_detect_yolo8m_1024_SD_202409_pytorch.onnx"})

    # file_path = r"E:\03.users\liuhuanlin\01.data\cellbin2\stitch\A03599D1_DAPI.tif"
    m_info = detect_chip(moving_mat.image, cfg=cfg, stain_type=TechType.DAPI, actual_size=(19992, 19992))
    moving_image.set_chip_box(m_info)

    # theoretical origin position of the matrix 
    chip_mask_file = os.path.join(r'E:\03.users\liuhuanlin\02.code\cellbin2\cellbin\config\chip_mask.json')
    sc = StereoChip(chip_mask_file)
    sc.parse_info('A03599D1')
    moving_image.set_point00(sc.zero_zero_point)

    info = template_00pt_align(moving_image=moving_image, ref=template_ref, dst_shape=(sc.height, sc.width))
    print(info)
    cbimwrite(r'E:\03.users\liuhuanlin\01.data\cellbin2\stitch\A03599D1_DAPI_regist00.tif', info.register_mat)
