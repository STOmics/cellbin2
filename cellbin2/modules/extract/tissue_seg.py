from typing import Union, Optional
from pathlib import Path

from cellbin2.image import cbimwrite
from cellbin2.contrib.tissue_segmentor import TissueSegInputInfo
from cellbin2.utils.config import Config
from cellbin2.modules.metadata import ProcFile

from cellbin2.utils import ipr
from cellbin2.utils.stereo_chip import StereoChip
from cellbin2.utils.rle import RLEncode


def run_tissue_seg(
        image_file: ProcFile,
        image_path: Path,
        save_path: Path,
        config: Config,
        chip_info: StereoChip,
        binx: int = 1,
        channel_image: Optional[Union[ipr.ImageChannel, ipr.IFChannel]] = None
):
    """
    Run tissue segmentation on the given image.

    Args:
        image_file (ProcFile): The image file object.
        image_path (Path): The path to the input image.
        save_path (Path): The path to save the resulting tissue mask.
        config (Config): Configuration object containing tissue segmentation settings.
        chip_info (StereoChip): Information about the stereo chip used.
        channel_image (Optional[Union[ipr.ImageChannel, ipr.IFChannel]]): Optional channel image object to update with tissue segmentation data.

    Returns:
        np.ndarray: The resulting tissue mask.
    """
    from cellbin2.contrib.tissue_segmentor import segment4tissue
    tissue_input = TissueSegInputInfo(
        weight_path_cfg=config.tissue_segmentation,
        input_path=image_path,
        stain_type=image_file.tech,
        chip_size=chip_info.chip_specif,
        binx=binx
    )
    tissue_mask_output = segment4tissue(tissue_input)
    tissue_mask = tissue_mask_output.tissue_mask
    cbimwrite(str(save_path), tissue_mask)

    # if channel_image is not None:
    #     # Update the channel image with tissue segmentation shape
    #     channel_image.TissueSeg.TissueSegShape = list(tissue_mask.shape)
    #
    #     # Encode the tissue mask using Run-Length Encoding
    #     bmr = RLEncode()
    #     t_mask_encode = bmr.encode(tissue_mask)
    #
    #     # Update the channel image with the encoded tissue mask
    #     channel_image.TissueSeg.TissueMask = t_mask_encode

    return tissue_mask
