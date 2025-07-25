import os
from typing import Union, Tuple, List
import matplotlib.pyplot as plt
import numpy.typing as npt
import numpy as np
from pydantic import BaseModel, Field

from cellbin2.dnn.segmentor.detector import Segmentation
from cellbin2.dnn.segmentor.cell_trace import get_trace as get_t
from cellbin2.dnn.segmentor.cell_trace import get_trace_v2 as get_t_v2
from cellbin2.dnn.segmentor.cell_trace import cal_area, cal_int, get_partial_res, cell_int_hist
from cellbin2.utils import clog
from cellbin2.utils.common import TechType
from cellbin2.contrib.fast_correct import run_fast_correct
from cellbin2.dnn.segmentor.postprocess import CellSegPostprocess
from cellbin2.dnn.segmentor.preprocess import CellSegPreprocess
from cellbin2.dnn.segmentor.utils import SUPPORTED_STAIN_TYPE_BY_MODEL, TechToWeightName
from cellbin2.utils.pro_monitor import process_decorator
from cellbin2.contrib.base_module import BaseModule
from cellbin2.image import cbimread
from cellbin2.utils.common import fPlaceHolder, iPlaceHolder
from cellbin2.utils.weights_manager import download_by_names


class CellSegParam(BaseModel, BaseModule):
    ssDNA_weights_path: str = Field("cellseg_bcdu_SHDI_221008_tf.onnx", description="name of the model")
    DAPI_weights_path: str = Field("cellseg_bcdu_SHDI_221008_tf.onnx", description="name of the model")
    HE_weights_path: str = Field("cellseg_bcdu_H_240823_tf.onnx", description="name of the model")
    IF_weights_path: str = Field("cyto2torch_0", description="name of the model")
    Transcriptomics_weights_path: str = Field("cellseg_unet_RNA_20230606.onnx", description="name of the model")
    Protein_weights_path: str = Field("cellseg_unet_RNA_20230606.onnx", description="name of the model")
    num_threads: int = Field(1, description="name of the model")
    GPU: int = Field(0, description="name of the model")

    # def get_weights_path(self, stain_type):
    #     if stain_type == TechType.ssDNA or stain_type == TechType.DAPI:
    #         p = self.ssDNA_weights_path
    #     elif stain_type == TechType.IF:
    #         p = self.IF_weights_path
    #     elif stain_type == TechType.HE:
    #         p = self.HE_weights_path
    #     elif stain_type == TechType.Transcriptomics or TechType.Protein:
    #         p = self.Transcriptomics_weights_path
    #     else: p = None
    #
    #     return p


class CellSegmentation:
    def __init__(
            self,
            cfg: CellSegParam,
            stain_type: TechType,
            gpu: int = -1,
            num_threads: int = 0,
    ):
        """
        Initialize the CellSegmentation class with the given configuration and stain type.

        Args:
            cfg (CellSegParam): Configuration parameters for the cell segmentation model.
            stain_type (TechType): The type of stain used in the input images.
            gpu (int, optional): The index of the GPU to be used for computations. 
                                  Use -1 to indicate CPU usage. Defaults to -1.
            num_threads (int, optional): The number of threads to be used when running on the CPU. 
                                         Defaults to 0.
        """
        super(CellSegmentation, self).__init__()
        self.cfg = cfg
        self.stain_type = stain_type
        self._model_path = self.cfg.get_weights_path(self.stain_type)
        if not os.path.exists(self._model_path):
            clog.info(f"{self._model_path} does not exist, will download automatically.")
            download_by_names(
                save_dir=os.path.dirname(self._model_path),
                weight_names=[os.path.basename(self._model_path)]
            )
        self.model_name, self.mode = os.path.splitext(os.path.basename(self._model_path))
        if self.stain_type not in SUPPORTED_STAIN_TYPE_BY_MODEL[self.model_name]:
            clog.warning(
                f"{self.stain_type.name} not in supported list "
                f"{[i.name for i in SUPPORTED_STAIN_TYPE_BY_MODEL[self.model_name]]}"
            )
            return
        if self.stain_type == TechType.Transcriptomics:
            self._WIN_SIZE = (512, 512)
            self._OVERLAP = 0.1
        else:
            self._WIN_SIZE = (256, 256)
            self._OVERLAP = 16
        self.pre_process = CellSegPreprocess(
            model_name=self.model_name
        )
        self.post_process = CellSegPostprocess(
            model_name=self.model_name
        )
        self._gpu = gpu

        self._num_threads = num_threads

        self._cell_seg = Segmentation(
            mode=self.mode[1:],
            gpu=self._gpu,
            num_threads=self._num_threads,
            win_size=self._WIN_SIZE,
            overlap=self._OVERLAP,
            stain_type=self.stain_type,
            preprocess=self.pre_process,
            postprocess=self.post_process
        )
        clog.info("Start loading model weight")
        self._cell_seg.f_init_model(model_path=self._model_path)
        clog.info("End loading model weight")

    @process_decorator('GiB')
    def run(self, img: Union[str, npt.NDArray]) -> npt.NDArray[np.uint8]:
        """
        Run cell prediction on the given image.

        Args:
            img (Union[str, npt.NDArray]): The input image. This can be either a file path (str) or a numpy array (npt.NDArray).

        Returns:
            npt.NDArray[np.uint8]: The predicted cell segmentation mask as a numpy array of uint8 type.

        Raises:
            AttributeError: If the `_cell_seg` attribute is not initialized.
        """
        # Check if the _cell_seg attribute is initialized
        if not hasattr(self, '_cell_seg'):
            clog.info(f"{self.__class__.__name__} failed to initialize, can not predict")
            # Return a zeroed mask of the same shape as the input image
            mask = np.zeros_like(img, dtype='uint8')
        else:
            clog.info("start cell segmentation")
            # Predict the cell segmentation mask using the _cell_seg object
            mask = self._cell_seg.f_predict(img)
            clog.info("end cell segmentation")
        return mask

    @classmethod
    def run_fast(cls, mask: npt.NDArray, distance: int, process: int) -> npt.NDArray[np.uint8]:
        """
        Applies a fast correction to the mask if the distance is greater than 0.

        Args:
            mask (npt.NDArray): The mask to be corrected.
            distance (int): The distance parameter for the fast correction.
            process (int): The number of processes to be used for the correction.

        Returns:
            npt.NDArray[np.uint8]: The corrected mask if distance > 0, else the original mask.
        """
        if distance > 0:
            fast_mask = run_fast_correct(
                mask_path=mask,
                distance=distance,
                n_jobs=process
            )
            return fast_mask
        else:
            clog.info(f"distance is: {distance} which is less than 0, return mask as it is")
            return mask

    @staticmethod
    def get_trace(mask):
        """
        Process a mask for cell tracing. Depending on the size of the mask, either a standard or a faster
        version of the tracing algorithm is used to reduce memory usage for large images.
        
        Args:
            mask (npt.NDArray): The mask to be processed.
        
        Returns:
            npt.NDArray: The processed mask.
        
        Note:
            For masks with a height greater than 40,000 pixels, the accelerated version `get_t_v2` is used.
            Otherwise, the standard version `get_t` is used.
        """
        if mask.shape[0] > 40000:
            return get_t_v2(mask)
        else:
            return get_t(mask)

    @classmethod
    def get_stats(
            cls,
            c_mask_p,
            cor_mask_p,
            t_mask_p,
            register_img_p,
            keep=5,
            size=1024,
            save_dir=None,
    ) -> Tuple[float, float, float, List[Tuple[npt.NDArray, List[Tuple[int, int, int, int]]]], plt.figure]:
        """
        Calculate various statistics and visualizations based on input masks and an image.

        Args:
            c_mask_p (str): Path to the cell segmentation mask (single channel).
            cor_mask_p (str): Path to the corrected mask (single channel).
            t_mask_p (str): Path to the tissue segmentation mask (single channel).
            register_img_p (str): Path to the registered image. The image will be converted to grayscale and inverted if it's an H&E stained RGB image.
            keep (int, optional): Number of images to keep. Defaults to 5.
            size (int, optional): Size of the cropped images. Defaults to 1024.
            save_dir (str, optional): Directory to save the output images. Defaults to None.

        Returns:
            Tuple[float, float, float, List[Tuple[npt.NDArray, List[Tuple[int, int, int, int]]]], plt.figure]:
                - First element: Ratio of cell segmentation mask area to tissue mask area.
                - Second element: Ratio of corrected mask area to tissue mask area.
                - Third element: Intensity ratio between cell segmentation mask and tissue mask.
                - Fourth element: List of images (of size `size`) with their corresponding coordinates (y_begin, y_end, x_begin, x_end).
                - Fifth element: Intensity histogram figure of the cell segmentation mask.

        Examples:
            >>> c_mask_p = "/path/to/cell/mask.tif"
            >>> t_mask_p = "/path/to/tissue/mask.tif"
            >>> cor_mask_p = "/path/to/corrected/mask.tif"
            >>> register_img_p = "/path/to/registered/image.tif"
            >>> save_dir = "/path/to/save/directory"
            >>> area_ratio, area_ratio_cor, int_ratio, cell_with_outline, fig = CellSegmentation.get_stats(
            ...     c_mask_p=c_mask_p,
            ...     cor_mask_p=cor_mask_p,
            ...     t_mask_p=t_mask_p,
            ...     register_img_p=register_img_p,
            ...     save_dir=save_dir)
            >>> assert area_ratio == expected_value
            >>> assert area_ratio_cor == expected_value
            >>> assert int_ratio == expected_value
            >>> fig.savefig(os.path.join(save_dir, f"test.png"))
        """

        @process_decorator('GiB')
        def get_cell_stats():
            from cellbin2.image.augmentation import f_ij_16_to_8_v2
            from cellbin2.image import cbimread, cbimwrite

            # Read the input images and masks
            register_img = cbimread(register_img_p, only_np=True)
            c_mask = cbimread(c_mask_p, only_np=True)
            t_mask = cbimread(t_mask_p, only_np=True)
            if cor_mask_p != "":
                cor_mask = cbimread(cor_mask_p, only_np=True)
            else:
                cor_mask = None

            # Convert 16-bit images to 8-bit
            register_img = f_ij_16_to_8_v2(register_img)

            # Calculate the area ratio
            area_ratio = cal_area(cell_mask=c_mask, tissue_mask=t_mask)
            clog.info(f"cell mask area / tissue mask area = {area_ratio}")
            if cor_mask is not None:
                area_ratio_cor = cal_area(cell_mask=cor_mask, tissue_mask=t_mask)
            else:
                area_ratio_cor = fPlaceHolder
            clog.info(f"correct mask area / tissue mask area = {area_ratio_cor}")

            # Calculate the intensity ratio
            int_ratio = cal_int(
                c_mask=c_mask,
                t_mask=t_mask,
                register_img=register_img
            )
            clog.info(f"cell mask intensity / tissue mask intensity = {int_ratio}")

            # Get the intensity histogram figure of the cell mask
            fig = cell_int_hist(c_mask=c_mask, register_img=register_img)
            clog.info(f"cell mask intensity calculation finished")

            # Get partial visualization images
            cell_with_outline = get_partial_res(
                c_mask=c_mask,
                t_mask=t_mask,
                register_img=register_img,
                keep=keep,
                k=size
            )
            if save_dir is not None:
                for i, v in enumerate(cell_with_outline):
                    im, box = v
                    box = [str(i) for i in box]
                    cord_str = "_".join(box)
                    save_path = os.path.join(save_dir, f"{cord_str}.tif")
                    cbimwrite(save_path, im)
            return area_ratio, area_ratio_cor, int_ratio, cell_with_outline, fig

        return get_cell_stats()


def s_main():
    """
    Main function to execute cell segmentation.
    
    This function parses command-line arguments and orchestrates the cell segmentation process.
    It supports different stain types and can optionally perform tissue segmentation and fast correction.
    """
    import argparse
    from cellbin2.image import cbimwrite

    # Setting up argument parser
    parser = argparse.ArgumentParser(description="Cell segmentation script. You should add parameters.")
    parser.add_argument('-i', "--input", required=True, help="The input image path.")
    parser.add_argument('-o', "--output", required=True, help="The output directory.")
    parser.add_argument("-p", "--model", required=True, help="Model directory.")
    parser.add_argument("-s", "--stain", required=True, choices=['he', 'ssdna', 'dapi', 'scell'], help="Stain type.")
    parser.add_argument("-f", "--fast", action='store_true', help="If run fast correction.")
    parser.add_argument("-t", "--tissue", action='store_true', help="If run tissue segmentation.")
    parser.add_argument("-m", "--mode", choices=['onnx', 'tf'], help="Model mode: ONNX or TensorFlow.", default="onnx")
    parser.add_argument("-g", "--gpu", type=int, help="The GPU index.", default=0)
    args = parser.parse_args()

    # Mapping user stain type to internal TechType
    usr_stype_to_inner = {
        'ssdna': TechType.ssDNA,
        'dapi': TechType.DAPI,
        "he": TechType.HE,
        "scell": TechType.Transcriptomics
    }

    # Extracting and validating arguments
    input_path = args.input
    output_path = args.output
    model_dir = args.model
    mode = args.mode
    gpu = args.gpu
    user_s_type = args.stain
    fast = args.fast
    tc = args.tissue

    # Convert user stain type to internal type
    s_type = usr_stype_to_inner.get(user_s_type)

    # Generate output file names
    name = os.path.splitext(os.path.basename(input_path))[0]
    os.makedirs(output_path, exist_ok=True)
    c_mask_path = os.path.join(output_path, f"{name}_v3_mask.tif")
    t_mask_path = os.path.join(output_path, f"{name}_tissue_mask.tif")
    f_mask_path = os.path.join(output_path, f"{name}_v3_corr_mask.tif")

    # Perform tissue segmentation if requested
    tm = None
    if tc:
        from cellbin2.utils.config import Config
        from cellbin2.contrib.tissue_segmentor import segment4tissue
        from cellbin2.contrib.tissue_segmentor import TissueSegInputInfo
        from cellbin2 import CB2_DIR
        c_file = os.path.join(CB2_DIR, 'cellbin2/config/cellbin.yaml')
        conf = Config(c_file, weights_root=model_dir)
        ti = TissueSegInputInfo(
            weight_path_cfg=conf.tissue_segmentation,
            input_path=input_path,
            stain_type=s_type,
        )
        to = segment4tissue(ti)
        tm = to.tissue_mask
        cbimwrite(t_mask_path, tm * 255)

    # Configure model paths based on mode
    cfg = CellSegParam()
    for p_name in cfg.model_fields:
        default_name = getattr(cfg, p_name)
        if not p_name.endswith('_weights_path'):
            continue
        if mode == 'tf':
            default_name = default_name.replace(".onnx", ".hdf5")
        setattr(cfg, p_name, os.path.join(model_dir, default_name))

    # Execute cell segmentation
    c_mask, f_mask = segment4cell(
        input_path=input_path,
        cfg=cfg,
        s_type=s_type,
        gpu=gpu,
        fast=fast,
    )
    if tm is not None:
        c_mask = tm * c_mask

    # Save segmentation masks
    cbimwrite(c_mask_path, c_mask * 255)
    if f_mask is not None:
        if tm is not None:
            f_mask = tm * f_mask
        cbimwrite(f_mask_path, f_mask * 255)


def segment4cell(
        input_path: str,
        cfg: CellSegParam,
        s_type: TechType,
        gpu: int,
        fast: bool
) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """
    Perform cell segmentation on the given input image.

    Parameters:
    - input_path (str): The path to the input image file.
    - cfg (CellSegParam): Configuration parameters for cell segmentation.
    - s_type (TechType): The type of staining technology used.
    - gpu (int): The GPU device index to use for computation. Use -1 for CPU.
    - fast (bool): Whether to apply fast correction to the segmentation mask.

    Returns:
    - Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]: A tuple containing the original segmentation mask and the fast corrected mask (if fast correction is applied).
    """
    # read user input image
    img = cbimread(input_path, only_np=True)

    # initialize cell segmentation model
    cell_seg = CellSegmentation(
        cfg=cfg,
        stain_type=s_type,
        gpu=gpu,
        num_threads=0,
    )

    # run cell segmentation
    mask = cell_seg.run(img=img)

    # fast correct
    fast_mask = None
    if fast:
        fast_mask = CellSegmentation.run_fast(mask=mask, distance=10, process=5)
        return mask, fast_mask

    return mask, fast_mask


if __name__ == '__main__':
    s_main()
