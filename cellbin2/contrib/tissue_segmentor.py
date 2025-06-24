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
    IF_weights_path: Optional[str] = Field('-', description='IF 使用传统算法，没有相应的模型')
    GPU: int = Field(-1, description='gpu编号，默认为-1，使用cpu')
    num_threads: int = Field(1, description="name of the model")

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
    weight_path_cfg: TissueSegParam = Field('', description='组织分割不同染色权重配置文件，需要权重的绝对路径')
    input_path: Union[str, Path] = Field('', description='输入图像')
    stain_type: TechType = Field('', description='输入图像的染色类型')
    chip_size: Tuple[Union[float, int], Union[float, int]] = Field(None, description='芯片的高和宽')  # S0.5 -> float; S1 -> int
    threshold_list: Tuple[int, int] = Field(None, description='输入阈值的下限和上限，仅针对IF图像')


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
        Args:
            cfg (TissueSegParam): network param
            stain_type (TechType): image stain type
            gpu (int): gpu index
            num_threads (int): default is 0,When you use the CPU
            threshold_list (list[float, float]): threshold_l, threshold_h, only for IF images
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
        self.model_path = self.cfg.get_weights_path(self.stain_type)
        self.model_name, self.mode = os.path.splitext(os.path.basename(self.model_path))
        if (not os.path.exists(self.model_path)) and self.stain_type != TechType.IF:
            clog.info(f"{self.model_path} does not exists, will download automatically. ")
            download_by_names(
                save_dir=os.path.dirname(self.model_path),
                weight_names=[os.path.basename(self.model_path)]
            )
        if self.stain_type not in support_model.SUPPORTED_STAIN_TYPE_BY_MODEL[self.model_name]:
            clog.warning(
                f"{self.stain_type.name} not in supported list of model: {self.model_name} \n"
                f"{self.model_name} supported stain type list:\n"
                f"{[i.name for i in support_model.SUPPORTED_STAIN_TYPE_BY_MODEL[self.model_name]]}"
            )
            return

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

        self.tissue_seg = TissueSegmentationBcdu(input_size=self.INPUT_SIZE,
                                                 gpu=self.gpu,
                                                 mode=self.mode,
                                                 num_threads=self.num_threads,
                                                 stain_type=self.stain_type,
                                                 threshold_list=self.threshold_list,
                                                 preprocess=self.pre_process,
                                                 postprocess=self.post_process,
                                                 )

        if self.stain_type != TechType.IF:
            clog.info("start loading model weight")
            self.tissue_seg.f_init_model(self.model_path)
            clog.info("end loading model weight")
        else:
            clog.info(f"stain type: {self.stain_type} do not need model")

    def run(self, img: Union[str, npt.NDArray]) -> TissueSegOutputInfo:

        clog.info("start tissue seg")
        if self.is_big_chip:
            mask = self.tissue_seg.f_predict_big_chip(img=img, chip_size=self.chip_size)
        else:
            mask = self.tissue_seg.f_predict(img=img)
        clog.info("end tissue seg")

        return mask


def compute_chip_size(input_img: np.ndarray) -> list:
    input_img = input_img.squeeze()
    if input_img.shape[0] in [1, 2, 3]:
        input_img = np.transpose(input_img, [1, 2, 0])

    hei, wid = input_img.shape[0], input_img.shape[1]
    chip_hei, chip_wid = int(hei/20000), int(wid/20000)
    return [chip_hei, chip_wid]


def segment4tissue(input_data: TissueSegInputInfo) -> TissueSegOutputInfo:
    """
    Args:
        input_data: TissueSegInputInfo类型:
            weight_path_cfg: 组织分割不同染色权重配置文件，需要权重的绝对路径, TissueSegParam类型
            input_path: 输入图像绝对路径
            stain_type: 输入图像的染色类型
            gpu: gpu编号，默认为-1，使用cpu
            chip_size: 芯片尺寸，高和宽
            threshold_list: 输入阈值的下限和上限，仅针对IF图像, 可选输入，默认为None；
                            该参数不为None则使用输入的阈值下限和上限对图像进行分割，该参数为None则代表IF图像使用OTSU算法
    Returns:
        TissueSegOutputInfo:
            tissue_mask: 输出的组织分割mask
            threshold_list: 返回的阈值列表，仅针对IF图像。如果input_data中的threshold_list为空，则这里返回OTSU计算出的阈值和图像的理
                            论最大灰度值(uint8:255, uint16:65535); 如果input_data中的threshold_list不为空，则返回threshold_list
    """

    from cellbin2.image import cbimread

    is_big_chip = False
    input_path = input_data.input_path
    cfg = input_data.weight_path_cfg
    s_type = input_data.stain_type
    gpu = input_data.weight_path_cfg.GPU
    chip_size = input_data.chip_size
    threshold_list = input_data.threshold_list

    clog.info(f"input stain type:{s_type}")

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
    seg_mask = tissue_seg.run(img=img)

    return seg_mask


def main():
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
    model_path = args.model  # 模型路径，以onnx结尾
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
