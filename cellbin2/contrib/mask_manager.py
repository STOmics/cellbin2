import os.path
from typing import Optional, Any, Tuple
from pydantic import BaseModel, Field
import cv2
import numpy as np
import tifffile
from skimage.measure import label
from skimage.morphology import remove_small_objects

from cellbin2.image import CBImage
from cellbin2.contrib.alignment.basic import ChipBoxInfo
from cellbin2.modules.metadata import TechType
from cellbin2.contrib.tissue_segmentor import TissueSegParam
from cellbin2.utils import clog
from cellbin2.utils.pro_monitor import process_decorator


class MaskManagerInfo(BaseModel):
    tissue_mask: Any = Field(None, description='tissue segmentation mask')
    cell_mask: Any = Field(None, description='cell segmentation mask')
    chip_box: Optional[ChipBoxInfo] = Field(None, description='chip box')
    stain_type: TechType = Field(None, description='staining type')
    method: int = Field(None, description='usage, 0 for stable version, 1 for development version')


class BestTissueCellMaskInfo(BaseModel):
    best_tissue_mask: Any = Field(None, description='fusioned tissue segmentation mask')
    best_cell_mask: Any = Field(None, description='fusioned cell segmentation mask')


class BestTissueCellMask:
    init_flag = True

    @staticmethod
    def init(input_data: MaskManagerInfo) -> bool:

        if input_data.cell_mask is None or input_data.tissue_mask is None:
            clog.error(f"init failed-->cell mask or tissue mask is None")
            return False

        if input_data.cell_mask.shape != input_data.tissue_mask.shape:
            clog.error(f"init failed-->the shape of the cell mask and tissue mask do not match")
            return False

        if input_data.stain_type is None:
            clog.error(f"init failed-->stain type is None")
            return False

        if input_data.method is None:
            clog.error(f"init failed-->method is None")
            return False

        clog.info('init success')
        return True

    @staticmethod
    def crop_chip_mask(chip_box: ChipBoxInfo, tissue_mask: np.ndarray,
                       cell_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x1, y1 = chip_box.LeftTop
        x2, y2 = chip_box.RightTop
        x3, y3 = chip_box.RightBottom
        x4, y4 = chip_box.LeftBottom
        points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        points = np.array(points).astype(np.int32)
        full_zeros_mask = np.zeros_like(tissue_mask)
        cv2.fillConvexPoly(full_zeros_mask, points, 255)

        cell_mask[full_zeros_mask == 0] = 0
        tissue_mask[full_zeros_mask == 0] = 0

        clog.info('tissue mask and cell mask update with chip box')
        return tissue_mask, cell_mask

    @staticmethod
    def best_cell_mask(tissue_mask: np.ndarray, cell_mask: np.ndarray) -> np.ndarray:
        """
        Filter cells by contours, removing cells not within the tissue region
        
        Parameters:
        tissue_mask (numpy.ndarray): Tissue mask image
        cell_mask (numpy.ndarray): Cell mask image
        
        Returns:
        numpy.ndarray: Filtered cell mask image
        """
        clog.info(f"calling function: best_cell_mask() ")
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            tissue_roi = tissue_mask[y:y+h, x:x+w]
            contour_roi = contour - np.array([x, y])

            roi_mask = np.zeros(shape=(h, w), dtype=np.uint8)
            cv2.fillPoly(roi_mask, pts=[contour_roi], color=(1,))

            cell_and_tissue = cv2.bitwise_and(roi_mask, tissue_roi)

            if not np.array_equal(cell_and_tissue, roi_mask):
                cv2.fillPoly(cell_mask, pts=[contour], color=(0,))

        return cell_mask

    @staticmethod
    def best_tissue_mask(tissue_mask: np.ndarray, cell_mask: np.ndarray, kernel_size: int) -> np.ndarray:
        clog.info(f"calling function: best_tissue_mask() ")
        clog.info(f"cell mask dilate kernel size:{kernel_size}")
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        if cell_mask.shape != tissue_mask.shape:
            clog.error('the shape of the cell mask and tissue mask do not match')
            return tissue_mask

        cell_mask_filter = cv2.bitwise_and(cell_mask, tissue_mask)

        dilated_cell_mask = cv2.dilate(cell_mask_filter, kernel, iterations=1)

        dilated_cell_mask = dilated_cell_mask > 0
        min_size = int(np.sum(kernel) * 2.25)
        filter_mask = remove_small_objects(dilated_cell_mask, min_size=min_size)
        filter_mask = np.uint8(filter_mask)
        filter_mask[filter_mask > 0] = 1

        tmp_mask = np.zeros_like(filter_mask, dtype=np.uint8)
        contours, _ = cv2.findContours(filter_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.fillPoly(img=tmp_mask, pts=[cnt], color=1)

        result_tissue_mask = cv2.bitwise_and(tissue_mask, tmp_mask)
        return result_tissue_mask

    @classmethod
    def get_best_tissue_cell_mask(cls, input_data: MaskManagerInfo) -> BestTissueCellMaskInfo:
        """
        Perform operations on masks based on input data.

        Args:
            input_data (MaskManagerInfo):
                The input data containing the following:
                - tissue mask: The tissue mask data.
                - cell mask: The cell mask data.
                - method (int): Currently can be 0 or 1. 0 represents the default operation, using tissue segmentation to filter cell segmentation; 1 represents the R & D version, using all information to output filtered cell and tissue segmentation.
                - stain type (TechType): The staining type from the TechType enumeration.
                - chip box (ChipBoxInfo, optional): Information about the chip box.

        Returns:
            BestTissueCellMaskInfo:
                The output containing the optimized cell mask and the optimized tissue mask.
        """
        cls.init_flag = False

        tissue_mask = input_data.tissue_mask
        cell_mask = input_data.cell_mask
        chip_box = input_data.chip_box
        stain_type = input_data.stain_type
        method = input_data.method

        kernel_size = 200
        crop_tissue_mask = None
        crop_cell_mask = None

        output_data = BestTissueCellMaskInfo()

        clog.info(f'processing stain type:{stain_type}')
        clog.info(f"received parameter method: {method}")
        cls.init_flag = cls.init(input_data)
        if not cls.init_flag:
            clog.info(f"return input tissue mask and cell mask")
            output_data.best_tissue_mask = tissue_mask
            output_data.best_cell_mask = cell_mask
            return output_data

        if method == 1 and stain_type == TechType.DAPI:
            clog.error(f"stain type: {stain_type} do not support method: {method}")
            clog.info(f"execute method 0 and return input tissue mask and best cell mask")
            output_data.best_tissue_mask = tissue_mask
            # output_data.best_cell_mask = cell_mask
            output_data.best_cell_mask = cls.best_cell_mask(tissue_mask=tissue_mask, cell_mask=cell_mask)
            return output_data


        if stain_type == TechType.HE:
            kernel_size = 250

        tissue_mask[tissue_mask > 0] = 1
        cell_mask[cell_mask > 0] = 1

        crop_tissue_mask, crop_cell_mask = tissue_mask, cell_mask  # TODO: hdd check this
        if chip_box:
            if 0:  # TODO chip detect is not stable, turn it off for now by dzh 2025/03/25
                crop_tissue_mask, crop_cell_mask = cls.crop_chip_mask(chip_box, tissue_mask, cell_mask)
            else:
                clog.warning('chip box is not available')
        else:
            clog.warning('chip box is None')

        if method == 0:
            output_data.best_cell_mask = cls.best_cell_mask(tissue_mask=tissue_mask, cell_mask=cell_mask)
            output_data.best_tissue_mask = tissue_mask
        elif method == 1:
            output_data.best_cell_mask = cls.best_cell_mask(tissue_mask=crop_tissue_mask, cell_mask=crop_cell_mask)
            output_data.best_tissue_mask = cls.best_tissue_mask(tissue_mask=crop_tissue_mask, cell_mask=crop_cell_mask,
                                                                kernel_size=kernel_size)
        else:
            clog.error(f'method only support 0 or 1, method:{method}')
            clog.info(f"return input tissue mask and cell mask")
            output_data.best_tissue_mask = tissue_mask
            output_data.best_cell_mask = cell_mask
            return output_data

        return output_data


def instance2semantics(ins: np.ndarray) -> np.ndarray:
    """
    :param ins: Instance mask (0-N)
    :return: Semantics mask (0-1)
    """
    ins_ = ins.copy()
    h, w = ins_.shape[:2]
    tmp0 = ins_[1:, 1:] - ins_[:h - 1, :w - 1]
    ind0 = np.where(tmp0 != 0)

    tmp1 = ins_[1:, :w - 1] - ins_[:h - 1, 1:]
    ind1 = np.where(tmp1 != 0)
    ins_[ind1] = 0
    ins_[ind0] = 0
    ins_[np.where(ins_ > 0)] = 1
    return np.array(ins_, dtype=np.uint8)


@process_decorator('GiB')
def merge_cell_mask(
        nuclear_mask: np.ndarray,
        membrane_mask: np.ndarray,
        conflict_cover: str = "nuclear"
) -> CBImage:
    """
    :param nuclear_mask: original nuclear_mask
        -- instance format: (0 for background, each cell is assigned a unique ID), maximum pixel value is total number of cells
        -- semantic format (binary mask)

    :param membrane_mask: original membrane_mask
        -- instance format: (0 for background, each cell is assigned a unique ID), maximum pixel value is total number of cells
        -- semantic format (binary mask)

    :param conflict_cover: conflict subjects nuclear | membrane

    :return: merged mask
    """
    if len(np.unique(nuclear_mask)) != 2:
        nuclear_mask_sem = instance2semantics(nuclear_mask)
        if conflict_cover == "membrane":
            nuclear_mask_ins = nuclear_mask.copy()
    else:
        nuclear_mask_sem = nuclear_mask.copy()
        if conflict_cover == "membrane":
            nuclear_mask_ins = label(nuclear_mask, connectivity=1)
    del nuclear_mask

    if len(np.unique(membrane_mask)) != 2:
        membrane_mask_sem = instance2semantics(membrane_mask)
        if conflict_cover == "nuclear":
            membrane_mask_ins = membrane_mask.copy()
    else:
        membrane_mask_sem = membrane_mask.copy()
        if conflict_cover == "nuclear":
            membrane_mask_ins = label(membrane_mask, connectivity=1)
    del membrane_mask

    mask_ = np.uint8(nuclear_mask_sem & membrane_mask_sem)

    if conflict_cover == "nuclear":
        remove_cell_id = np.unique(mask_ * membrane_mask_ins)
        del mask_
        membrane_mask_ins = np.uint32(~np.isin(membrane_mask_ins, remove_cell_id)) * membrane_mask_ins
        membrane_mask_sem = instance2semantics(membrane_mask_ins)
        del membrane_mask_ins
    else:
        remove_cell_id = np.unique(mask_ * nuclear_mask_ins)
        del mask_
        nuclear_mask_ins = np.uint32(~np.isin(nuclear_mask_ins, remove_cell_id)) * nuclear_mask_ins
        nuclear_mask_sem = instance2semantics(nuclear_mask_ins)
        del nuclear_mask_ins

    _ = membrane_mask_sem * 2 + nuclear_mask_sem * 1 #0 background，1 nuclei，2 cell
    return CBImage(membrane_mask_sem)


if __name__ == '__main__':
    from cellbin2.image import cbimread

    chip_box = ChipBoxInfo()

    # HE test data, path for cell segementation mask and tissue segmentation mask
    cell_mask_path = r"F:\01.users\hedongdong\cellbin2_test\cell_mask\C04042E3_HE_regist_v3_mask.tif"
    tissue_mask_path = r"F:\01.users\hedongdong\cellbin2_test\result_mask\C04042E3_HE_regist.tif"

    chip_point = {
        'left_top': [2183, 2190],
        'right_top': [22183, 2199],
        'right_bottom': [22189, 22195],
        'left_bottom': [2181, 22214]
    }
    stain_type = TechType.DAPI
    method = 1

    # # ssDNA test data, path for cell segementation mask and tissue segmentation mask 
    # cell_mask_path = r"F:\01.users\hedongdong\cellbin2_test\cell_mask\A04535A4C6_fov_stitched_v3_mask.tif"
    # tissue_mask_path = r"F:\01.users\hedongdong\cellbin2_test\result_mask\A04535A4C6_fov_stitched.tif"
    #
    # chip_point = {
    #     'left_top': [1136, 1256],
    #     'right_top': [40680, 1396],
    #     'right_bottom': [40434, 60732],
    #     'left_bottom': [910, 60576]
    # }
    # stain_type = TechType.ssDNA

    chip_box.IsAvailable = False
    chip_box.LeftTop = chip_point['left_top']
    chip_box.RightTop = chip_point['right_top']
    chip_box.RightBottom = chip_point['right_bottom']
    chip_box.LeftBottom = chip_point['left_bottom']

    cell_mask = cbimread(cell_mask_path, only_np=True)
    tissue_mask = cbimread(tissue_mask_path, only_np=True)

    input_data = MaskManagerInfo()
    input_data.tissue_mask = tissue_mask
    input_data.cell_mask = cell_mask
    input_data.chip_box = chip_box
    input_data.method = method
    input_data.stain_type = stain_type

    best_tissue_cell_mask = BestTissueCellMask.get_best_tissue_cell_mask(input_data=input_data)
    # print(best_tissue_cell_mask.chip_box)

    output_cell_mask = best_tissue_cell_mask.best_cell_mask
    output_tissue_mask = best_tissue_cell_mask.best_tissue_mask

    output_cell_mask[output_cell_mask > 0] = 255
    output_tissue_mask[output_tissue_mask > 0] = 255

    tifffile.imwrite(os.path.join(r"F:\01.users\hedongdong\cellbin2_test\best_merge_result", 'cell_mask.tif'),
                     output_cell_mask, compression='zlib')
    tifffile.imwrite(os.path.join(r"F:\01.users\hedongdong\cellbin2_test\best_merge_result", 'tissue_mask.tif'),
                     output_tissue_mask, compression='zlib')
