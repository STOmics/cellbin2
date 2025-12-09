from typing import Optional, Union

from cellbin2.image import cbimwrite
from cellbin2.utils.config import Config
from cellbin2.modules.metadata import ProcFile
from cellbin2.utils.common import TechType
from cellbin2.contrib import cell_segmentor
from pathlib import Path
from cellbin2.utils import ipr
from cellbin2.utils.rle import RLEncode
from cellbin2.contrib import cellpose_segmentor
import os
import numpy as np


def run_cell_seg(
        image_file: ProcFile,
        image_path: Path,
        save_path: Path,
        config: Config,
        channel_image: Optional[Union[ipr.ImageChannel, ipr.IFChannel]] = None
):
    """
    Run cell segmentation based on the type of technology used in the image file.

    Args:
        image_file (ProcFile): The image file containing the cell data.
        image_path (Path): The path to the input image file.
        save_path (Path): The path to save the segmented cell mask.
        config (Config): The configuration settings for cell segmentation.
        channel_image (Optional[Union[ipr.ImageChannel, ipr.IFChannel]]): The channel image data, if available.

    Returns:
        cell_mask: The segmented cell mask.
    """
    # check input data stain type and its referred model
    stain_type = str(image_file.tech.name)
    print(stain_type)
    cellseg_model_path = getattr(config.cell_segmentation, f"{stain_type}_weights_path")
    cellseg_model = os.path.basename(cellseg_model_path)

    if cellseg_model == 'cpsam':
        from cellbin2.contrib import cellposesam
        cell_mask = cellposesam.cellposesam_pred(
            img_path=str(image_path),
            cfg=config.cell_segmentation,
            use_gpu=True
        )
    elif cellseg_model == 'cyto2torch_0' or cellseg_model == 'cyto3' or cellseg_model == 'cellpose3':
        print("Using cellpose_segmentor for cell segmentation")
        cell_mask = cellpose_segmentor.segment4cell(
            input_path=str(image_path),
            cfg=config.cell_segmentation,
            use_gpu=True,
            stain_type = stain_type
        )
    else:
        cell_mask, fast_mask = cell_segmentor.segment4cell(
            input_path=str(image_path),
            cfg=config.cell_segmentation,
            s_type=image_file.tech,
            fast=False,
            gpu=0
        )
    
    # transform cell_mask from instance to 0 1 semantics mask
    cell_mask[np.where(cell_mask > 0)] = 1
    semantics = np.array(cell_mask, dtype=np.uint8)
    cbimwrite(str(save_path), semantics)
    # Here we do not save, the mask based on the registration image will be saved later
    # if channel_image is not None:
    #     channel_image.CellSeg.CellSegShape = list(cell_mask.shape)
    #     # channel_image.CellSeg.CellSegTrace =
    #     bmr = RLEncode()
    #     c_mask_encode = bmr.encode(cell_mask)
    #     channel_image.CellSeg.CellMask = c_mask_encode
    return cell_mask
