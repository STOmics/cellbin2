from typing import Optional, Union

from cellbin2.image import cbimwrite
from cellbin2.utils.config import Config
from cellbin2.modules.metadata import ProcFile
from cellbin2.utils.common import TechType
from cellbin2.contrib import cell_segmentor
from pathlib import Path
from cellbin2.utils import ipr
from cellbin2.utils.rle import RLEncode
import os
from cellpose import io

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

    cellpose_model_path = config.cell_segmentation.IF_weights_path
    cellpose_model = os.path.basename(cellpose_model_path)
    print("Using %s as the IF segmentation model"%(cellpose_model))
    print("model path: %s"(cellpose_model_path))

    if image_file.tech == TechType.IF:
        if cellpose_model == 'cyto2torch_0' or 'cyto2':
            print("Using cellpose_segmentor_2 for cell segmentation")
            from cellbin2.contrib import cellpose_segmentor_2
            cell_mask = cellpose_segmentor_2.segment4cell(
                input_path=str(image_path),
                cfg=config.cell_segmentation,
            )
        elif cellpose_model == 'cpsam':
            print("Using cpsam_segmentor for cell segmentation")
            from cellbin2.contrib import cpsam_segmentor
            image = io.imread(str(image_path))
            cell_mask = cpsam_segmentor.predict_cpsam(
                model_path=cellpose_model_path,
                image=image,
                batch_size=32,
                use_gpu=True
            )
    else:
        print("Using cell_segmentor for cell segmentation")
        cell_mask, fast_mask = cell_segmentor.segment4cell(
            input_path=str(image_path),
            cfg=config.cell_segmentation,
            s_type=image_file.tech,
            fast=False,
            gpu=0
        )
    cbimwrite(str(save_path), cell_mask)
    # Here we do not save, the mask based on the registration image will be saved later
    # if channel_image is not None:
    #     channel_image.CellSeg.CellSegShape = list(cell_mask.shape)
    #     # channel_image.CellSeg.CellSegTrace =
    #     bmr = RLEncode()
    #     c_mask_encode = bmr.encode(cell_mask)
    #     channel_image.CellSeg.CellMask = c_mask_encode
    return cell_mask
