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


def pad_to_target_size(image, target_h, target_w, padding_color=(0, 0, 0)):
    """
    Fill the image to the target size (centered filling)

     Args:
        image (numpy.ndarray): image
        target_h (int):
        target_w (int):
        padding_color (tuple):

     Returns:
        padded_image (numpy.ndarray): padded image
    """
    h, w = image.shape[:2]

    pad_top = (target_h - h) // 2
    pad_bottom = target_h - h - pad_top
    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left

    padded_image = cv.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=padding_color
    )

    return padded_image

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


def crop_image(corner_qc_points, temp_points, qc_points,
               image, image_size, image_type,
               draw_radius, template_color, qc_color, draw_thickness
               ):
    """
        Crop regions of interest from an image around specified corner points, and annotate them with template and QC points.

        Args:
            corner_qc_points (list): List of corner points around which to crop images. Each point should contain at least x,y coordinates.
            temp_points (list): List of template points.
            qc_points (list): List of qc points.
            image: Input image CBImage(Class).
            image_size (int): Desired size (width and height) of the output cropped images.
            image_type (str): Type(stain) of the image, used to determine the enhancement method.
            draw_radius (int): Radius for drawing circles around points.
            template_color: Color to use for drawing template points.
            qc_color: Color to use for drawing qc points.
            draw_thickness (int): Thickness of the circles drawn around points.

        Returns:
            tuple: A tuple containing:
                - cp_image_list (list): List of cropped and enhanced image regions with points marked.
                - coord_list (list): List of coordinates used for each crop [y_up, y_down, x_left, x_right].

        The function crops square regions of specified size around each corner point, adjusts for image boundaries,
        enhances each cropped region based on image type, and marks template and qc points within each region.
        """
    cp_image_list = list()
    coord_list = list()
    for cp in corner_qc_points:
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
        temp_ctp = [i for i in temp_points if (i[0] > x_left) and
                (i[1] > y_up) and
                (i[0] < x_right) and
                (i[1] < y_down)]
        qc_ctp = [i for i in qc_points if (i[0] > x_left) and
                    (i[1] > y_up) and
                    (i[0] < x_right) and
                    (i[1] < y_down)]
        temp_ctp = np.array(temp_ctp)[:, :2] - [x_left, y_up]
        qc_ctp = np.array(qc_ctp)[:, :2] - [x_left, y_up]
        _ci = enhance_func(_ci)

        for i in temp_ctp:
            cv.circle(_ci, list(map(int, i))[:2],
                      draw_radius , template_color, draw_thickness)
        for i in qc_ctp:
            cv.circle(_ci, list(map(int, i))[:2],
                      draw_radius , qc_color, draw_thickness)

        cp_image_list.append(_ci)

    return cp_image_list, coord_list


def template_painting(
        image_data: Union[str, np.ndarray, CBImage],
        tissue_seg_data: Union[str, np.ndarray, CBImage],
        image_type: str,

        qc_points: np.ndarray = None,
        template_points: np.ndarray = None,
        image_size: int = 1024,
        qc_color: tuple = (0, 255, 0),
        template_color: tuple = (0, 0, 255),
        true_color: tuple = (0, 255, 0),
        miss_color: tuple = (0, 0, 255),
        chip_rect_color: tuple = (255, 0, 255),
        tissue_rect_color: tuple = (255, 255, 0),
        idx_color: tuple = (255, 255, 255),
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
        true_color: color of points where the error between track_point and template_point is less than 10 pixels
        miss_color: over than 10 pixels color
        chip_rect_color:
        tissue_rect_color:
        idx_color: chip rect and tissue rect index color
        draw_radius:
        draw_thickness:


    Returns:

    """
    image = cbimread(image_data)
    if image_type in ['DAPI', 'ssDNA']:
        image = image.to_gray()

    tissue_image = cbimread(tissue_seg_data)

    _temp, _qc = TemplateReferenceV1.pair_to_template(
        qc_points, template_points, threshold=5
    )

    # qc_track point closest to chip corner
    chip_corner_points = np.array([[0, 0], [0, image.height],
                              [image.width, image.height], [image.width, 0]])
    chip_points_dis = cdist(_qc[:, :2], chip_corner_points)
    corner_chip_qc_points = _qc[np.argmin(chip_points_dis, axis=0)]

    # qc_track point closest to tissue corner
    tissue_corner_points = get_tissue_corner_points(tissue_image.image)
    tissue_points_dis = cdist(_qc[:, :2], tissue_corner_points)
    corner_tissue_qc_points = _qc[np.argmin(tissue_points_dis, axis=0)]

    ########################
    # qc template track point for chip edge
    cp_image_list, cp_coord_list = crop_image(
        corner_chip_qc_points, template_points, qc_points,
        image, image_size, image_type,
        draw_radius, template_color, qc_color, draw_thickness
    )
    # qc template track point for tissue edge
    tissue_image_list, tissue_coord_list = crop_image(
        corner_tissue_qc_points, template_points, qc_points,
        image, image_size, image_type,
        draw_radius, template_color, qc_color, draw_thickness
    )
    ########################

    qc_list = _qc.tolist()
    _unpair = [i for i in qc_points[:, :2].tolist() if i not in qc_list]

    rate = image_size*2 / max(image.width, image.height)
    _image = image.resize_image(rate)

    if len(_image.image.shape) == 2:
        _image = cv.cvtColor(f_ij_16_to_8(_image.image), cv.COLOR_GRAY2BGR)
    else:
        _image = f_ij_16_to_8(_image.image)
    for i in np.array(_unpair):
        cv.circle(_image, list(map(int, i * rate)),
                  draw_radius, miss_color, draw_thickness)

    for i in np.array(_qc):
        cv.circle(_image, list(map(int, i[:2] * rate)),
                  draw_radius, true_color, draw_thickness)

    small_idx = 0
    for cc in cp_coord_list:
        small_idx += 1
        y0, y1, x0, x1 = cc
        _p = np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0]], dtype=np.int32)

        cv.polylines(_image, [(_p * rate).astype(np.int32)],
                     True, chip_rect_color, draw_thickness)

        # draw index on left_top
        text_pos = (int(x0 * rate) + 10, int(y0 * rate) + 40)
        cv.putText(_image, str(small_idx), text_pos, cv.FONT_HERSHEY_SIMPLEX, 1.2, idx_color, 2)

    for tc in tissue_coord_list:
        small_idx += 1
        y0, y1, x0, x1 = tc
        _p = np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0]], dtype=np.int32)

        cv.polylines(_image, [(_p * rate).astype(np.int32)],
                     True, tissue_rect_color, draw_thickness)

        # draw index in left_top
        text_pos = (int(x0 * rate) + 10, int(y0 * rate) + 40)
        cv.putText(_image, str(small_idx), text_pos, cv.FONT_HERSHEY_SIMPLEX, 1.2, idx_color, 2)

    _image = pad_to_target_size(_image, image_size*2, image_size*2, (0, 0, 0))

    return _image, cp_image_list, tissue_image_list


def chip_box_painting(
        image_data: Union[str, np.ndarray, CBImage],
        chip_info: Union[np.ndarray, ChipBoxInfo] = None,
        ipr_path: str = None,
        layer: str = None,
        image_size: int = 2048,
        chip_color: tuple = (0, 255, 0),
        draw_thickness: int = 3
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
        Chip frame detection results and four part image
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

    ###
    rate = image_size / max(image.width, image.height)
    _image = image.resize_image(rate).image

    if len(_image.shape) == 2:
        # Ensure image is 8-bit single channel before histogram equalization
        _image_8bit = f_ij_16_to_8(_image)
        _image = cv2.equalizeHist(_image_8bit)
        _image = cv.cvtColor(_image, cv.COLOR_GRAY2BGR)
    else:
        _image = f_ij_16_to_8(_image)

    _points = points * rate
    _points = _points.reshape(-1, 1, 2)
    cv.polylines(_image, [_points.astype(np.int32)],
                 True, chip_color, draw_thickness)
    ### part image
    image = f_ij_16_to_8(image.image)
    if len(image.shape) == 2:
        image = cv2.equalizeHist(image)
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    cv.polylines(image, [points.astype(np.int32)],
                 True, chip_color, draw_thickness)

    half_image_size = image_size//2
    padded_image = cv.copyMakeBorder(
        image,
        half_image_size,
        half_image_size,
        half_image_size,
        half_image_size,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    chipbox_part_image_lists = []
    for point in points:
        x, y = int(point[0]), int(point[1])
        # calculate cutting area (center point (x,y), range is image_size x image_size)
        x_start = x  # since original image already padded with 'half_image_size',no offset required for the coordinate
        y_start = y
        x_end = x_start + image_size
        y_end = y_start + image_size

        # automatic restrict boundary (prevent crossing line) 
        chip = padded_image[
               max(0, y_start): min(padded_image.shape[0], y_end),
               max(0, x_start): min(padded_image.shape[1], x_end)
               ]
        if chip.shape[0] < image_size or chip.shape[1] < image_size:
            pad_bottom = image_size - chip.shape[0]
            pad_right = image_size - chip.shape[1]
            chip = cv2.copyMakeBorder(
                chip,
                top=0,
                bottom=pad_bottom,
                left=0,
                right=pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0))

        chipbox_part_image_lists.append(chip)

    return _image, chipbox_part_image_lists


def cell_seg_painting(
        image_data: Union[str, np.ndarray, CBImage],
        cell_mask_data: Union[str, np.ndarray, CBImage],
        tissue_seg_data: Union[str, np.ndarray, CBImage],
        image_type: str,
        corrected_mask_data: Union[str, np.ndarray, CBImage] = None,
        image_size: int = 1024,
        cell_color: tuple = (0, 255, 0),
        corrected_color: tuple = (255, 255, 0),
        tissue_color: tuple = (0, 0, 255),
        edge_rect_color: tuple = (255, 0, 255),
        density_rect_color: tuple = (255, 255, 0),
        idx_color: tuple = (255, 255, 255),
        draw_thickness: int = 2,
        contour_thickness: int = 1
) -> Union[np.ndarray, list, list]:
    """
    Create enhanced cell segmentation visualization similar to template_painting.
    
    Args:
        image_data: Input image data (path or array)
        cell_mask_data: Cell segmentation mask
        tissue_seg_data: Tissue segmentation mask  
        image_type: Image type (ssDNA, DAPI, HE)
        corrected_mask_data: Optional corrected cell mask
        image_size: Size for cropped regions
        cell_color: Color for cell contours
        corrected_color: Color for corrected cell contours
        tissue_color: Color for tissue boundary
        edge_rect_color: Color for edge region rectangles
        density_rect_color: Color for density region rectangles
        idx_color: Color for region index text
        draw_thickness: Thickness for rectangles
        contour_thickness: Thickness for cell contours
        
    Returns:
        Tuple of (main_image, edge_image_list, density_image_list)
    """
    # Read input data
    image = cbimread(image_data)
    if image_type in ['DAPI', 'ssDNA']:
        image = image.to_gray()
    
    cell_mask = cbimread(cell_mask_data, only_np=True)
    tissue_mask = cbimread(tissue_seg_data, only_np=True)
    
    if corrected_mask_data is not None:
        corrected_mask = cbimread(corrected_mask_data, only_np=True)
    else:
        corrected_mask = None
    
    # Get 5 high cell density regions (original cellseg regions)
    density_points = get_high_density_regions(cell_mask, tissue_mask, num_regions=5)
    
    ########################
    # Crop 5 high density regions
    density_image_list, density_coord_list = crop_cell_seg_regions(
        density_points, cell_mask, corrected_mask,
        image, image_size, image_type,
        cell_color, corrected_color, contour_thickness
    )
    # No edge regions needed - only the 5 original cellseg regions
    edge_image_list, edge_coord_list = [], []
    ########################
    
    # Create main overview image
    rate = image_size*2 / max(image.width, image.height)
    _image = image.resize_image(rate)
    
    if len(_image.image.shape) == 2:
        _image = cv.cvtColor(f_ij_16_to_8(_image.image), cv.COLOR_GRAY2BGR)
    else:
        _image = f_ij_16_to_8(_image.image)
    
    # Draw tissue boundary
    tissue_resized = cv.resize(tissue_mask.astype(np.uint8), 
                              (int(image.width * rate), int(image.height * rate)))
    contours, _ = cv.findContours(tissue_resized, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(_image, contours, -1, tissue_color, contour_thickness)
    
    # Draw rectangles for the 5 cellseg regions (numbered 1-5)
    small_idx = 0
    for dc in density_coord_list:
        small_idx += 1
        y0, y1, x0, x1 = dc
        _p = np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0]], dtype=np.int32)
        
        cv.polylines(_image, [(_p * rate).astype(np.int32)],
                     True, density_rect_color, draw_thickness)
        
        # Draw index as 1-5 (matching the small images)
        text_pos = (int(x0 * rate) + 10, int(y0 * rate) + 40)
        cv.putText(_image, f"{small_idx}", text_pos, cv.FONT_HERSHEY_SIMPLEX, 1.2, idx_color, 2)
    
    _image = pad_to_target_size(_image, image_size*2, image_size*2, (0, 0, 0))
    
    # Return overview image and 5 cellseg images (edge_image_list is empty now)
    return _image, edge_image_list, density_image_list


def get_high_density_regions(cell_mask, tissue_mask, num_regions=5, region_size=1024):
    """
    Find regions with high cell density for detailed visualization.
    
    Args:
        cell_mask: Cell segmentation mask
        tissue_mask: Tissue segmentation mask
        num_regions: Number of high density regions to find
        region_size: Size of each region
        
    Returns:
        List of center points for high density regions
    """
    from cellbin2.image.wsi_split import SplitWSI
    
    # Combine cell and tissue masks
    combined_mask = cell_mask * tissue_mask
    
    # Split into regions
    sp = SplitWSI(combined_mask, win_shape=(region_size, region_size))
    sp._f_split()
    boxes = sp.box_lst
    
    # Count cells in each region
    density_scores = []
    for box in boxes:
        y_begin, y_end, x_begin, x_end = box
        sub_mask = combined_mask[y_begin:y_end, x_begin:x_end]
        
        # Count connected components (cells)
        contours, _ = cv.findContours(sub_mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        cell_count = len(contours)
        
        # Calculate center point
        center_x = (x_begin + x_end) // 2
        center_y = (y_begin + y_end) // 2
        
        density_scores.append({
            'center': [center_x, center_y],
            'cell_count': cell_count,
            'box': box
        })
    
    # Sort by cell density and get top regions
    density_scores.sort(key=lambda x: x['cell_count'], reverse=True)
    
    # Return center points of top density regions
    top_regions = density_scores[:num_regions]
    return [region['center'] for region in top_regions if region['cell_count'] > 0]


def crop_cell_seg_regions(center_points, cell_mask, corrected_mask,
                         image, image_size, image_type,
                         cell_color, corrected_color, contour_thickness):
    """
    Crop regions around specified center points and add cell segmentation overlays.
    
    Args:
        center_points: List of center points for cropping
        cell_mask: Cell segmentation mask
        corrected_mask: Optional corrected cell mask
        image: Input image
        image_size: Size of cropped regions
        image_type: Image type for enhancement
        cell_color: Color for cell contours
        corrected_color: Color for corrected cell contours
        contour_thickness: Thickness of contours
        
    Returns:
        Tuple of (image_list, coord_list)
    """
    image_list = []
    coord_list = []
    
    for center_point in center_points:
        x, y = map(int, center_point[:2])
        
        # Calculate crop boundaries
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
        
        # Crop image
        _ci = image.crop_image([y_up, y_down, x_left, x_right])
        coord_list.append([y_up, y_down, x_left, x_right])
        
        # Enhance image
        enhance_func = pt_enhance_method.get(image_type, dapi_enhance)
        _ci = enhance_func(_ci)
        
        # Ensure image is BGR
        if len(_ci.shape) == 2:
            _ci = cv.cvtColor(_ci, cv.COLOR_GRAY2BGR)
        
        # Crop masks
        cell_crop = cell_mask[y_up:y_down, x_left:x_right]
        
        # Draw cell contours
        contours, _ = cv.findContours(cell_crop.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(_ci, contours, -1, cell_color, contour_thickness)
        
        # Draw corrected cell contours if available
        if corrected_mask is not None:
            corrected_crop = corrected_mask[y_up:y_down, x_left:x_right]
            corrected_contours, _ = cv.findContours(corrected_crop.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(_ci, corrected_contours, -1, corrected_color, contour_thickness)
        
        # Add cell count text
        cell_count = len(contours)
        cv.putText(_ci, f"Cells: {cell_count}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        image_list.append(_ci)
    
    return image_list, coord_list


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
