from typing import Union, Tuple, List, Any, Optional
import cv2
import tifffile
from pydantic import BaseModel, Field
import numpy.typing as npt
import numpy as np
from pathlib import Path
import os
from cellbin2.utils.common import TechType
from cellbin2.dnn.tissue_segmentor.detector import TissueSegmentationBcdu
from cellbin2.utils import clog
from cellbin2.dnn.tissue_segmentor.utils import SupportModel
from cellbin2.contrib.param import TissueSegOutputInfo
from cellbin2.contrib.base_module import BaseModule
from cellbin2.utils.weights_manager import download_by_names


class TissueSegParam(BaseModel, BaseModule):
    ssDNA_weights_path: str = Field(r"tissueseg_bcdu_S_240618_tf.onnx",
                                    description="name of the ssdna model")
    DAPI_weights_path: str = Field(r"tissueseg_bcdu_SDI_230523_tf.onnx",
                                   description="name of the dapi model")
    HE_weights_path: str = Field(r"tissueseg_bcdu_H_20241018_tf.onnx",
                                 description="name of the he model")
    Transcriptomics_weights_path: str = Field(r"tissueseg_bcdu_rna_220909_tf.onnx",
                                              description="name of the transcriptomics model")
    Protein_weights_path: str = Field(r"tissueseg_bcdu_rna_220909_tf.onnx",
                                      description="name of the Protein model")
    IF_weights_path: Optional[str] = Field('-', description='IF using tradition algorithm, no model applied')
    GPU: int = Field(-1, description='GPU id, default -1 (CPU)')
    num_threads: int = Field(1, description="name of the model")
    best_tissue_mask_method: int = Field(0, description="0: use the best tissue mask, 1: use the tissue mask with the highest score")

    # def get_weights_path(self, stain_type):
    #     p = ''
    #     if stain_type == TechType.ssDNA:
    #         p = self.ssdna_weights_path
    #     elif stain_type == TechType.DAPI:
    #         p = self.dapi_weights_path
    #     elif stain_type == TechType.HE:
    #         p = self.he_weights_path
    #     elif stain_type == TechType.Transcriptomics:
    #         p = self.transcriptomics_weights_path
    #     elif stain_type == TechType.Protein:
    #         p = self.protein_weights_path
    #     elif stain_type == TechType.IF:
    #         p = self.if_weights_path
    #     return p


class TissueSegInputInfo(BaseModel):
    weight_path_cfg: TissueSegParam = Field('', description='config file for different staining tissue segmentation, absolute path required for the checkpoint')
    input_path: Union[str, Path] = Field('', description='input image ')
    stain_type: TechType = Field('', description='staining type of input image ')
    chip_size: Tuple[Union[float, int], Union[float, int]] = Field(None, description='height and width for the chip')  # S0.5 -> float; S1 -> int
    threshold_list: Tuple[int, int] = Field(None, description='input lower and upper bound of threshold (applies to IF images only)')
    binx: int = Field(1, description='binX for image')


class TissueSegmentation:
    def __init__(
            self,
            support_model: SupportModel,
            cfg: TissueSegParam,
            stain_type: TechType,
            gpu: int = -1,
            num_threads: int = 0,
            threshold_list: Tuple[int, int] = None,
            chip_size: List = None,
            is_big_chip: bool = False
    ):
        """
        Initialize the TissueSegmentation class with the given parameters.
        
        Args:
            support_model (SupportModel): The model support information.
            cfg (TissueSegParam): The configuration parameters for the tissue segmentation.
            stain_type (TechType): The type of stain used in the input image.
            gpu (int): The GPU index to be used for computation. Default is -1 (CPU).
            num_threads (int): The number of threads to be used when computing on the CPU. Default is 0.
            threshold_list (Tuple[int, int]): The threshold values for low and high intensity. Only used for IF images.
            chip_size (List): The size of the chip to be segmented.
            is_big_chip (bool): Flag indicating if the chip is a big chip that requires special preprocessing and postprocessing.
        """
        super(TissueSegmentation, self).__init__()
        self.cfg = cfg
        self.stain_type = stain_type
        self.INPUT_SIZE = (512, 512, 1)
        self.threshold_list = threshold_list
        self.chip_size = chip_size
        self.is_big_chip = is_big_chip
        self.gpu = gpu
        self.num_threads = num_threads

        # Get the model path based on the stain type
        self.model_path = self.cfg.get_weights_path(self.stain_type)
        self.model_name, self.mode = os.path.splitext(os.path.basename(self.model_path))

        # Download the model if it does not exist and the stain type is not IF
        if (not os.path.exists(self.model_path)) and self.stain_type != TechType.IF:
            clog.info(f"{self.model_path} does not exist, will download automatically.")
            download_by_names(
                save_dir=os.path.dirname(self.model_path),
                weight_names=[os.path.basename(self.model_path)]
            )

        # Check if the stain type is supported by the model
        if self.stain_type not in support_model.SUPPORTED_STAIN_TYPE_BY_MODEL[self.model_name]:
            clog.warning(
                f"{self.stain_type.name} not in supported list of model: {self.model_name} \n"
                f"{self.model_name} supported stain type list:\n"
                f"{[i.name for i in support_model.SUPPORTED_STAIN_TYPE_BY_MODEL[self.model_name]]}"
            )
            return

        # Initialize preprocessing and postprocessing modules based on whether it is a big chip
        if self.is_big_chip:
            from cellbin2.dnn.tissue_segmentor.big_chip_preprocess import BigChipTissueSegPreprocess
            from cellbin2.dnn.tissue_segmentor.big_chip_postprocess import BigChipTissueSegPostprocess

            self.pre_process = BigChipTissueSegPreprocess(self.model_name, support_model, self.chip_size)
            self.post_process = BigChipTissueSegPostprocess(self.model_name, support_model)
        else:
            from cellbin2.dnn.tissue_segmentor.preprocess import TissueSegPreprocess
            from cellbin2.dnn.tissue_segmentor.postprocess import TissueSegPostprocess
            self.pre_process = TissueSegPreprocess(self.model_name, support_model)
            self.post_process = TissueSegPostprocess(self.model_name, support_model)

        # Initialize the tissue segmentation model with the specified parameters
        self.tissue_seg = TissueSegmentationBcdu(input_size=self.INPUT_SIZE,
                                                 gpu=self.gpu,
                                                 mode=self.mode,
                                                 num_threads=self.num_threads,
                                                 stain_type=self.stain_type,
                                                 threshold_list=self.threshold_list,
                                                 preprocess=self.pre_process,
                                                 postprocess=self.post_process,

                                                 )

        # Load the model weights if the stain type is not IF
        if self.stain_type != TechType.IF:
            clog.info("Start loading model weight")
            self.tissue_seg.f_init_model(self.model_path)
            clog.info("End loading model weight")
        else:
            clog.info(f"Stain type: {self.stain_type} does not need model")

    def run(self, img: Union[str, npt.NDArray], binx: int = 1) -> TissueSegOutputInfo:
        """
        Perform tissue segmentation on the input image.

        Args:
            img (Union[str, npt.NDArray]): The input image, which can be provided as a file path or a NumPy array.

        Returns:
            TissueSegOutputInfo: The segmentation mask.
        """

        clog.info("start tissue seg")
        if self.is_big_chip:
            mask = self.tissue_seg.f_predict_big_chip(img=img, chip_size=self.chip_size, binx=binx)
        else:
            mask = self.tissue_seg.f_predict(img=img, binx=binx)
        clog.info("end tissue seg")

        return mask


def compute_chip_size(input_img: np.ndarray) -> list:
    """
    Calculate the chip size of the input image.

    The function divides the height and width of the input image by 20000 to compute
    the chip size. If the input image has a shape of 1, 2, or 3 along the first axis,
    it is transposed before computing the dimensions.

    Args:
        input_img (np.ndarray): The input image as a numpy array.

    Returns:
        list: A list containing the computed chip height and width.
    """
    input_img = input_img.squeeze()
    if input_img.shape[0] in [1, 2, 3]:
        input_img = np.transpose(input_img, [1, 2, 0])

    hei, wid = input_img.shape[0], input_img.shape[1]
    chip_hei, chip_wid = int(hei/20000), int(wid/20000)
    return [chip_hei, chip_wid]


def segment4tissue(input_data: TissueSegInputInfo) -> TissueSegOutputInfo:
    """
    Perform tissue segmentation on the input image.

    Args:
        input_data (TissueSegInputInfo):
            An instance containing the following fields:
                - weight_path_cfg (TissueSegParam): Configuration for the tissue segmentation model weights.
                - input_path (str): Absolute path to the input image.
                - stain_type (TechType): The staining type of the input image.
                - gpu (int): GPU index to use for computation. Default is -1, which means using CPU.
                - chip_size (Tuple[int, int]): The height and width of the chip. If not provided, it will be computed based on the image size.
                - threshold_list (Tuple[int, int], optional):
                    The lower and upper thresholds for segmentation. Only applicable for IF images.
                    If provided, these thresholds will be used for segmentation. If not provided, the OTSU algorithm will be used.

    Returns:
        TissueSegOutputInfo:
            An instance containing the following fields:
                - tissue_mask (np.ndarray): The output tissue segmentation mask.
                - threshold_list (Tuple[int, int]):
                    The thresholds used for segmentation. Only applicable for IF images.
                    If input_data.threshold_list is None, the returned thresholds are those calculated by the OTSU algorithm and the theoretical maximum grayscale value
                    (uint8: 255, uint16: 65535). If input_data.threshold_list is not None, the returned thresholds are the same as the input thresholds.
    """

    from cellbin2.image import cbimread

    is_big_chip = False
    input_path = input_data.input_path
    cfg = input_data.weight_path_cfg
    s_type = input_data.stain_type
    gpu = input_data.weight_path_cfg.GPU
    chip_size = input_data.chip_size
    threshold_list = input_data.threshold_list
    binx = input_data.binx

    clog.info(f"input stain type:{s_type}")
    clog.info(f"input bin_size:{binx}")

    support_model = SupportModel()

    # read user input image
    img = cbimread(input_path, only_np=True)
    if chip_size is None:
        clog.warning(f'input chip size is None, compute chip size with image size')
        chip_size = compute_chip_size(img)  # return chip height and chip width
    if chip_size[0] > 1 or chip_size[1] > 1:
        is_big_chip = True
    clog.info(f'the chip size of the image:{os.path.basename(input_path)}, height:{chip_size[0]}, width:{chip_size[1]}, is_big_chip:{is_big_chip}')
    if len(img.shape) == 3 and s_type != TechType.HE:
        clog.warning(
            'the input image is an RGB image, bug the stain type is not HE,convert the RGB image to GRAY image')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # initialize tissue segmentation model
    tissue_seg = TissueSegmentation(
        support_model=support_model,
        cfg=cfg,
        stain_type=s_type,
        gpu=gpu,
        num_threads=0,
        threshold_list=threshold_list,
        chip_size=chip_size,
        is_big_chip=is_big_chip
    )
    seg_mask = tissue_seg.run(img=img, binx=binx)

    return seg_mask


def main():
    """
    Main function to perform tissue segmentation on an input image.

    This function parses command-line arguments for the input image path, output file path,
    model path, stain type, chip size, mode (onnx or tf), and GPU index. It initializes the
    tissue segmentation model with the given parameters, processes the input image, and saves
    the resulting segmented image.

    Args:
        input (str): Path to the input image file.
        output (str): Path to save the output segmented image file.
        model (str): Path to the model file.
        stain (str): Type of stain used in the input image.
        chip_size (list): Height and width of the chip.
        mode (str): Mode of the model ('onnx' or 'tf').
        gpu (int): Index of the GPU to use.
    """
    import argparse
    parser = argparse.ArgumentParser(description="you should add those parameter")
    parser.add_argument('-i', "--input",
                        default=r".\test_image\A03599D1_DAPI_fov_stitched.tif",
                        required=True, help="the input img path")
    parser.add_argument('-o', "--output",
                        default=r".\result_mask\cellbin2\A03599D1_DAPI_fov_stitched.tif",
                        required=True, help="the output file")
    parser.add_argument("-p", "--model",
                        default=r".\model\tissueseg_bcdu_SDI_230523_tf.onnx",
                        required=True, help="model path")
    parser.add_argument("-s", "--stain", default='dapi', required=True,
                        choices=['he', 'ssdna', 'dapi', 'transcriptomics', 'protein', 'if'], help="stain type")
    parser.add_argument("-c", "--chip_size", default=None, nargs='+', type=int, help="the height and width of the chip")
    parser.add_argument("-m", "--mode", default='onnx', choices=['onnx', 'tf'], help="onnx or tf")
    parser.add_argument("-g", "--gpu", default=0, type=int, help="the gpu index")
    args = parser.parse_args()

    usr_stype_to_inner = {
        'ssdna': TechType.ssDNA,
        'dapi': TechType.DAPI,
        "he": TechType.HE,
        "transcriptomics": TechType.Transcriptomics,
        'protein': TechType.Protein,
        'if': TechType.IF
    }

    input_path = args.input
    output_path = args.output
    model_path = args.model  # model path, end with onnx 
    user_s_type = args.stain
    gpu = args.gpu
    chip_size = args.chip_size
    # stain type from user input to inner type
    s_type = usr_stype_to_inner.get(user_s_type)

    cfg = TissueSegParam()
    if s_type != TechType.IF:
        setattr(cfg, f"{s_type.name}_weights_path", model_path)
    input_data = TissueSegInputInfo()

    input_data.input_path = input_path
    input_data.weight_path_cfg = cfg
    input_data.stain_type = s_type
    input_data.weight_path_cfg.GPU = gpu
    input_data.chip_size = chip_size
    # input_data.threshold_list = 34, 60

    clog.info(f"image path:{input_path}")
    seg_result = segment4tissue(input_data=input_data)

    seg_mask = seg_result.tissue_mask
    print(seg_mask.shape)
    if seg_result.threshold_list:
        print(*seg_result.threshold_list)
    seg_mask[seg_mask > 0] = 255
    tifffile.imwrite(output_path, seg_mask, compression='zlib')


if __name__ == '__main__':
    main()
