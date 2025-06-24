import os.path

import cv2
import h5py
import cv2 as cv
import numpy as np
from typing import Union

import tifffile
from scipy.spatial.distance import cdist

from cellbin2.image import CBImage, cbimread
from cellbin2.contrib.alignment.basic import transform_points
from cellbin2.image.augmentation import f_ij_16_to_8, dapi_enhance, he_enhance
from cellbin2.contrib.alignment.basic import TemplateInfo, ChipBoxInfo
from cellbin2.contrib.template.inferencev1 import TemplateReferenceV1
from typing import Dict

pt_enhance_method = {
    'ssDNA': dapi_enhance,
    'DAPI': dapi_enhance,
    'HE': he_enhance
}


def get_tissue_corner_points(
        tissue_data: np.ndarray,
        k: int = 9
):
    _tissue = tissue_data.copy()

    _tissue = cv.dilate(
        _tissue,
        kernel=np.ones([k, k], dtype=np.uint8)
    )

    contours, _ = cv.findContours(
        _tissue,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_NONE
    )

    max_contours = sorted(contours, key=cv.contourArea, reverse=True)[0]

    x, y, w, h = cv.boundingRect(max_contours)

    corner_points = np.array(
        [[x, y], [x, y + h], [x + w, y], [x + w, y + h]]
    )

    dis = cdist(
        max_contours.squeeze(),
        corner_points
    )

    result_points = max_contours.squeeze()[np.argmin(dis, axis=0)]

    return result_points


def crop_image(corner_temp_points, points, image,
               image_size, image_type,
               draw_radius, template_color, draw_thickness
               ):
    cp_image_list = list()
    coord_list = list()
    for cp in corner_temp_points:
        x, y = map(int, cp[:2])
        if x <= image_size // 2:
            x_left = 0
            x_right = image_size
        elif x + image_size // 2 > image.width:
            x_left = image.width - image_size
            x_right = image.width
        else:
            x_left = x - image_size // 2
            x_right = x + image_size // 2

        if y <= image_size // 2:
            y_up = 0
            y_down = image_size
        elif y + image_size // 2 > image.height:
            y_up = image.height - image_size
            y_down = image.height
        else:
            y_up = y - image_size // 2
            y_down = y + image_size // 2

        _ci = image.crop_image([y_up, y_down, x_left, x_right])
        coord_list.append([y_up, y_down, x_left, x_right])
        enhance_func = pt_enhance_method.get(image_type, "DAPI")
        _ctp = [i for i in points if (i[0] > x_left) and
                (i[1] > y_up) and
                (i[0] < x_right) and
                (i[1] < y_down)]
        _ctp = np.array(_ctp)[:, :2] - [x_left, y_up]
        _ci = enhance_func(_ci)

        for i in _ctp:
            cv.circle(_ci, list(map(int, i))[:2],
                      draw_radius * 2, template_color, draw_thickness)
        cp_image_list.append(_ci)

    return cp_image_list, coord_list


def template_painting(
        image_data: Union[str, np.ndarray, CBImage],
        tissue_seg_data: Union[str, np.ndarray, CBImage],
        image_type: str,
        qc_points: np.ndarray = None,
        template_points: np.ndarray = None,
        image_size: int = 2048,
        track_color: tuple = (0, 0, 255),
        template_color: tuple = (0, 255, 0),
        chip_rect_color: tuple = (0, 255, 255),
        tissue_rect_color: tuple = (255, 255, 0),
        draw_radius: int = 5,
        draw_thickness: int = 2
) -> Union[np.ndarray, list, list]:
    """

    Args:
        image_data:
        tissue_seg_data:
        image_type: str -- ssDNA, DAPI, HE
        qc_points:
        template_points:
        image_size: image height size
        track_color:
        template_color:
        chip_rect_color:
        tissue_rect_color:
        draw_radius:
        draw_thickness:

    Returns:

    """
    image = cbimread(image_data)
    if image_type in ['DAPI', 'ssDNA']:
        image = image.to_gray()

    tissue_image = cbimread(tissue_seg_data)

    _temp, _track = TemplateReferenceV1.pair_to_template(
        qc_points, template_points
    )

    corner_points = np.array([[0, 0], [0, image.height],
                              [image.width, image.height], [image.width, 0]])

    points_dis = cdist(_temp[:, :2], corner_points)
    corner_temp_points = _temp[np.argmin(points_dis, axis=0)]

    tissue_corner_points = get_tissue_corner_points(tissue_image.image)
    tissue_points_dis = cdist(_temp[:, :2], tissue_corner_points)
    corner_tissue_temp_points = _temp[np.argmin(tissue_points_dis, axis=0)]

    ########################
    # 芯片角最近track点
    cp_image_list, cp_coord_list = crop_image(
        corner_temp_points, _temp, image,
        image_size, image_type,
        draw_radius, template_color, draw_thickness
    )
    # 组织边缘track点
    tissue_image_list, tissue_coord_list = crop_image(
        corner_tissue_temp_points, _temp, image,
        image_size, image_type,
        draw_radius, template_color, draw_thickness
    )
    ########################

    track_list = _track.tolist()
    _unpair = [i for i in qc_points[:, :2].tolist() if i not in track_list]

    rate = image_size / image.image.shape[0]
    _image = image.resize_image(rate)

    if len(_image.image.shape) == 2:
        _image = cv.cvtColor(f_ij_16_to_8(_image.image), cv.COLOR_GRAY2BGR)
    else:
        _image = f_ij_16_to_8(_image.image)
    for i in np.array(_unpair):
        cv.circle(_image, list(map(int, i * rate)),
                  draw_radius, track_color, draw_thickness)

    for i in np.array(_temp):
        cv.circle(_image, list(map(int, i[:2] * rate)),
                  draw_radius, template_color, draw_thickness)

    for cc in cp_coord_list:
        y0, y1, x0, x1 = cc
        _p = np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0]], dtype=np.int32)

        cv.polylines(_image, [(_p * rate).astype(np.int32)],
                     True, chip_rect_color, draw_thickness)

    for tc in tissue_coord_list:
        y0, y1, x0, x1 = tc
        _p = np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0]], dtype=np.int32)

        cv.polylines(_image, [(_p * rate).astype(np.int32)],
                     True, tissue_rect_color, draw_thickness)

    return _image, cp_image_list, tissue_image_list


def chip_box_painting(
        image_data: Union[str, np.ndarray, CBImage],
        chip_info: Union[np.ndarray, ChipBoxInfo] = None,
        ipr_path: str = None,
        layer: str = None,
        image_size: int = 2048,
        chip_color: tuple = (0, 255, 0),
        draw_thickness: int = 5
) -> np.ndarray:
    """

    Args:
        image_data:
        chip_info:
        ipr_path:
        layer:
        image_size:
        chip_color:
        draw_thickness:

    Returns:

    """
    if ipr_path is not None:
        with h5py.File(ipr_path) as conf:
            p_lt = conf[f'{layer}/QCInfo/ChipBBox'].attrs['LeftTop']
            p_lb = conf[f'{layer}/QCInfo/ChipBBox'].attrs['LeftBottom']
            p_rb = conf[f'{layer}/QCInfo/ChipBBox'].attrs['RightBottom']
            p_rt = conf[f'{layer}/QCInfo/ChipBBox'].attrs['RightTop']
        points = np.array([p_lt, p_lb, p_rb, p_rt])

    elif chip_info is not None:
        points = chip_info.chip_box

    else:
        raise ValueError("Chip info not found.")

    image = cbimread(image_data)
    rate = image_size / image.image.shape[0]
    # _image = image.resize_image(rate)
    # _image = cv.cvtColor(f_ij_16_to_8(_image.image), cv.COLOR_GRAY2BGR)

    image = image.image
    image = f_ij_16_to_8(image)
    if len(image.shape) == 2:
        image = cv2.equalizeHist(image)
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    else:
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

    points = points * rate
    points = points.reshape(-1, 1, 2)
    cv.polylines(image, [points.astype(np.int32)],
                 True, chip_color, draw_thickness)

    return image


def get_view_image(
        image: Union[np.ndarray, str],
        points: np.ndarray,
        is_matrix: bool = False,
        downsample_size: int = 2000,
        crop_size: int = 500,
        color: tuple = (0, 0, 255),
        radius: int = 10,
        thickness: int = 1,
        scale_line_pixels: int = 5,
        scale_line_length: int = 3,
        output_path: str = "",
        ) -> Dict[str, np.ndarray]:
    """

    Args:
        image: input image or image path
        points: detected chip box points
        is_matrix:
        downsample_size: enhanced image downsample size
        crop_size: crop image size
        color: chip box color,(B, G, R)
        radius: circle radius, it must be a multiple of scale_line_pixels pixels
        thickness:
        scale_line_pixels:  units of scale line
        scale_line_length: # scale line length
        output_path:

    Returns:

    """
    if isinstance(image, str):
        image = tifffile.imread(image)
    image_list = list()
    output_dic = {}
    image = f_ij_16_to_8(image)
    if is_matrix:
        image_enhance = cv.filter2D(image, -1, np.ones((21, 21), np.float32))
        image_enhance = (image_enhance > 0).astype(np.uint8) * 255
        crop_size *= 2
        radius *= 2
    elif image.ndim == 3:  # HE image
        image_enhance = cv.cvtColor(cv.bitwise_not(cv.cvtColor(image, cv.COLOR_BGR2GRAY)), cv.COLOR_GRAY2BGR)
    else:
        image_enhance = cv2.equalizeHist(image)
        image_enhance = cv2.cvtColor(image_enhance, cv2.COLOR_GRAY2BGR)

    image_enhance = cv2.resize(image_enhance, (downsample_size, downsample_size), interpolation=cv2.INTER_NEAREST)
    image_list.append(image_enhance)  # save enhance and resize image

    for fp in points:
        x, y = map(lambda k: int(k), fp)
        _x = _y = crop_size

        if x < crop_size:
            _x = x
            x = crop_size

        if y < crop_size:
            _y = y
            y = crop_size

        if x > image.shape[1] - crop_size:
            _x = 2 * crop_size + x - image.shape[1]
            x = image.shape[1] - crop_size

        if y > image.shape[0] - crop_size:
            _y = 2 * crop_size + y - image.shape[0]
            y = image.shape[0] - crop_size

        _image = image[y - crop_size: y + crop_size, x - crop_size: x + crop_size]
        if not is_matrix:
            if _image.ndim == 3:
                _image = cv.cvtColor(cv.bitwise_not(cv.cvtColor(_image, cv.COLOR_BGR2GRAY)), cv.COLOR_GRAY2BGR)
            else:
                _image = cv.cvtColor(cv.equalizeHist(_image), cv.COLOR_GRAY2BGR)
        else:
            _image = cv.filter2D(_image, -1, np.ones((21, 21), np.float32))
            _image = (_image > 0).astype(np.uint8) * 255
            _image = cv.cvtColor(_image, cv.COLOR_GRAY2BGR)

        scale_line_list = []
        line1 = np.array([[_x, _y - 2*radius], [_x, _y + 2*radius]], np.int32).reshape((-1, 1, 2))
        for tmp_y in range(_y-radius, _y+radius, scale_line_pixels):
            scale_line_list.append(np.array([[_x, tmp_y], [_x+scale_line_length, tmp_y]], np.int32).reshape((-1, 1, 2)))
        line2 = np.array([[_x - 2*radius, _y], [_x + 2*radius, _y]], np.int32).reshape((-1, 1, 2))
        for tmp_x in range(_x-radius, _x+radius, scale_line_pixels):
            scale_line_list.append(np.array([[tmp_x, _y], [tmp_x, _y-scale_line_length]], np.int32).reshape((-1, 1, 2)))

        cv.circle(_image, [_x, _y], radius, color[::-1], thickness)
        cv2.putText(_image, f'r={radius}', (_x+5, _y-5), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)

        cv.polylines(_image, pts=[line1, line2], isClosed=False,
                     color=color, thickness=thickness, lineType=cv.LINE_8)
        cv.polylines(_image, pts=scale_line_list, isClosed=False,
                     color=color, thickness=1, lineType=cv.LINE_8)

        image_list.append(np.array(_image))

    name_list = ["enhance", "left_up", "left_down", "right_down", "right_up"]
    for name, im in zip(name_list, image_list):
        if os.path.isdir(output_path):
            cv.imwrite(os.path.join(output_path, f"{name}.tif"), im)
        output_dic[name] = im
    return output_dic


if __name__ == '__main__':
    import h5py
    from cellbin2.image import cbimwrite

    image_dic = get_view_image(image = r"D:\hedongdong1\Workspace\01.chip_box_detect\show_interface\test_data\C04144G513_ssDNA_stitch.tif",
                   points = np.loadtxt(r"D:\hedongdong1\Workspace\01.chip_box_detect\show_interface\test_data\C04144G513_ssDNA_stitch.txt"),
                   output_path = r"D:\hedongdong1\Workspace\01.chip_box_detect\show_interface\test_result")
    print(len(image_dic))
    enhance_img = image_dic['enhance']
    left_up_img = image_dic['left_up']
    left_down_img = image_dic['left_down']
    right_down_img = image_dic['right_down']
    right_up_img = image_dic['right_up']

    tmp_img1 = cv2.vconcat([left_up_img, left_down_img])
    tmp_img2 = cv2.vconcat([right_up_img, right_down_img])

    result_img = cv2.hconcat([tmp_img1, tmp_img2])
    result_img = cv2.hconcat([enhance_img, result_img])

    cbimwrite(os.path.join(r'D:\hedongdong1\Workspace\01.chip_box_detect\show_interface\test_result', 'detect_chip_debug.tif'), result_img)

    # register_img = "/media/Data/dzh/data/cellbin2/test/SS200000135TL_D1_demo/SS200000135TL_D1_ssDNA_regist.tif"
    # tissue_cut = "/media/Data/dzh/data/cellbin2/test/SS200000135TL_D1_demo/SS200000135TL_D1_ssDNA_tissue_cut.tif"
    # with h5py.File("/media/Data/dzh/data/cellbin2/test/SS200000135TL_D1_demo/SS200000135TL_D1.ipr", "r") as f:
    #     template_points = f["ssDNA"]["Register"]["RegisterTemplate"][...]
    #     track_points = f["ssDNA"]["Register"]["RegisterTrackTemplate"][...]
    # _image, cp_image_list, tissue_image_list = template_painting(
    #     image_data=register_img,
    #     tissue_seg_data=tissue_cut,
    #     image_type="ssDNA",
    #     qc_points=track_points,
    #     template_points=template_points,
    # )
    # cbimwrite("/media/Data/dzh/data/cellbin2/test/SS200000135TL_D1_demo/assets/image/ssDNA_trackpoint.png", _image)
