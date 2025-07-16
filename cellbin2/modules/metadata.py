import os.path

from prettytable import PrettyTable
from cellbin2.utils import clog
import json
import copy
import pprint

from typing import List, Dict, Type, Any, Tuple, Optional
from objtyping import objtyping
import pathlib
from pydantic import BaseModel, model_validator

from cellbin2.contrib import chip_detector
from cellbin2.contrib import clarity
from cellbin2.utils.common import TechType
from cellbin2.utils.config import Config
from cellbin2.utils import obj2dict, dict2json
from cellbin2.utils.ipr import bPlaceHolder, fPlaceHolder, iPlaceHolder, sPlaceHolder


def map_kit2type(kit_name: str):
    """
    Maps the kit name to its corresponding type.

    Args:
        kit_name (str): The name of the kit.

    Returns:
        str: The type of the kit, which can be 'Transcriptomics', 'Protein', or 'UNKNOWN'.
    """
    if 'Stereo-seq' in kit_name:
        return TechType.Transcriptomics.name
    if 'Stereo-CITE' in kit_name:
        return TechType.Protein.name
    return TechType.UNKNOWN


class ProcRegistration(BaseModel):
    """
    A class to represent a process registration.

    Attributes:
        fixed_image (int): The fixed image identifier.
        trackline (bool): A flag indicating whether to track the line.
        reuse (int): The reuse identifier.
    """
    fixed_image: int
    trackline: bool
    reuse: int


class ProcFile(BaseModel):
    """
    ProcFile class to handle file operations and validations for processing.
    """
    file_path: str
    tech_type: str
    chip_detect: bool
    quality_control: bool
    tissue_segmentation: bool
    cell_segmentation: bool
    channel_align: int
    registration: ProcRegistration
    _supported_matrix = ['.gef', '.gz', '.gem']
    _supported_image = ['.tif', '.tiff', '.TIF', '.TIFF']

    @property
    def tech(self,) -> TechType:
        """
        Returns the TechType enum value corresponding to the tech_type string.
        """
        return TechType[self.tech_type]

    @property
    def is_image(self,) -> bool:
        """
        Checks if the file is an image based on its suffix.
        """
        if pathlib.Path(self.file_path).suffix in self._supported_image:
            return True
        else:
            return False

    @property
    def is_matrix(self,) -> bool:
        """
        Checks if the file is a matrix based on its suffix.
        """
        if pathlib.Path(self.file_path).suffix in self._supported_matrix:
            return True
        else:
            return False

    @property
    def tag(self,):
        """
        Returns the tag extracted from the file name.
        """
        return os.path.basename(self.file_path).split('.')[0]

    @property
    def is_exists(self,):
        """
        Checks if the file exists at the given file_path.
        """
        return pathlib.Path.exists(pathlib.Path(self.file_path))

    def get_group_name(self, sn, pattern='IF'):
        """
        Generates a group name based on the serial number (sn) and pattern.
        """
        if pattern in self.tag:
            if sn in self.tag:
                _n = self.tag.split(sn)
                g_name = '_'.join(_n).strip("_")
            else:
                _n = self.tag.split(pattern)
                g_name = '_'.join(_n).strip("_") + f'_{pattern}'
        elif self.tech == TechType.IF:
            g_name = self.tag + f'_{pattern}'
        else:
            g_name = self.tech.name
        return g_name

    def valid_check(self, cfg: Config):
        """
        Validates the file processing parameters against the provided configuration.
        """
        # chip_detect valid check
        if self.chip_detect and self.tech not in chip_detector.SUPPORTED_STAIN_TYPE:
            clog.warning(
                f"{self.tech.name} not in chip_detect supporting list "
                f"{[st.name for st in chip_detector.SUPPORTED_STAIN_TYPE]}. "
                f"Change {self.file_path} chip_detect to false"
            )
            self.chip_detect = False
        # quality_control(clarity) valid check
        if self.quality_control and self.tech not in clarity.SUPPORTED_STAIN_TYPE:
            clog.warning(
                f"{self.tech.name} not in quality_control supporting list "
                f"{[st.name for st in clarity.SUPPORTED_STAIN_TYPE]}. "
                f"Change quality_control to false"
            )
            self.quality_control = False
        # tissue_segmentation valid check
        if self.tissue_segmentation and self.tech not in cfg.tissue_segmentation.supported_model:
            clog.warning(
                f"{self.tech.name} not in tissue_segmentation supporting list "
                f"{[st.name for st in cfg.tissue_segmentation.supported_model]}. "
                f"Change {self.file_path} tissue_segmentation to false"
            )
            self.tissue_segmentation = False
        # cell_segmentation valid check
        if self.cell_segmentation and self.tech not in cfg.cell_segmentation.supported_model:
            clog.warning(
                f"{self.tech.name} not in cell_segmentation supporting list "
                f"{[st.name for st in cfg.cell_segmentation.supported_model]}. "
                f"Change {self.file_path} cell_segmentation to false"
            )
            self.cell_segmentation = False


class ProcMolecularFile(BaseModel):
    """
    A class to represent a processed molecular file.

    Attributes:
        exp_matrix (int): The experimental matrix.
        cell_mask (List[int]): The cell mask.
        extra_method (str): An additional molecular classification method, currently not used.
    """
    exp_matrix: int
    cell_mask: List[int]
    extra_method: str = ''  # additional molecular classification methods, currently not available
    correct_r: int


class Run(BaseModel):
    """
    A class representing a Run object with various boolean attributes.
    
    Attributes:
        qc (bool): Whether or not quality control is performed.
        alignment (bool): Whether or not alignment is performed.
        matrix_extract (bool): Whether or not matrix extraction is performed.
        report (bool): Whether or not a report is generated.
        annotation (bool): Whether or not annotation is performed.
    """
    qc: bool
    alignment: bool
    matrix_extract: bool
    report: bool
    annotation: bool


class ProcParam(BaseModel):
    """
    ProcParam class is used to handle and validate the parameters for image and molecular file processing.
    It contains methods for printing file information, checking inputs, and retrieving image files.
    """
    image_process: Dict[str, ProcFile] = {}
    molecular_classify: Dict[str, ProcMolecularFile] = {}
    run: Run = None

    @staticmethod
    def print_files_info(files: dict, mode: str = 'imageQC'):
        """
        Prints the information of files to be processed in a tabular format.
        
        Args:
            files (dict): Dictionary of files to be processed.
            mode (str): Mode of processing, either 'imageQC' or 'Scheduler'.
        """
        a_table = PrettyTable()
        if mode == 'imageQC':
            a_table.field_names = ['ID', 'file_name', 'stain_type', 'template', 'detect_chip', 'clarity', 'align']
            for idx, pif in files.items():
                a_table.add_row([idx, os.path.basename(pif.file_path), pif.tech_type, pif.registration.trackline,
                                 pif.chip_detect, pif.quality_control, pif.channel_align])
        elif mode == 'Scheduler':
            a_table.field_names = ['ID', 'file_name', 'stain_type', 'registration', 'tissue_seg', 'cell_seg', 'align']
            for idx, pif in files.items():
                a_table.add_row([idx, os.path.basename(pif.file_path), pif.tech_type, pif.registration.fixed_image,
                                 pif.tissue_segmentation, pif.cell_segmentation, pif.channel_align])

        clog.info('{} files to be processed, the information as follows,\n{}'.format(len(files), a_table))

    def check_inputs(self, cfg: Config):
        """
        Checks the validity of the input configuration for each image file.
        
        Args:
            cfg (Config): Configuration object for validation.
        """
        for i, v in self.image_process.items():
            v.valid_check(cfg=cfg)

    def get_image_files(self, do_image_qc: bool = True,
                        do_scheduler: bool = True,
                        cheek_exists: bool = False) -> Dict[int, ProcFile]:
        """
        Retrieves and filters image files based on the specified criteria.
        
        Args:
            do_image_qc (bool): Flag to include image QC files.
            do_scheduler (bool): Flag to include scheduler files.
            cheek_exists (bool): Flag to check if files exist.
        
        Returns:
            Dict[int, ProcFile]: Dictionary of filtered image files.
        """
        images = copy.copy(self.image_process)
        images = {int(idx): image for idx, image in images.items()}
        if do_image_qc:
            images = {
                idx: image for idx, image in images.items()
                if image.is_image and
                   (image.chip_detect or
                    image.quality_control or
                    image.channel_align != -1 or
                    image.registration.trackline)
            }

        if do_scheduler:
            images = {
                idx: image for idx, image in images.items()
                if image.tissue_segmentation or
                   image.cell_segmentation or
                   image.registration.trackline or
                   image.channel_align != -1
            }
            # add matrix
            for idx, matrix in self.molecular_classify.items():
                images[matrix.exp_matrix] = self.image_process[str(matrix.exp_matrix)]

        if cheek_exists:
            images = {int(idx): image for idx, image in images.items() if image.is_exists}

        return images

    def get_molecular_classify(self) -> Dict[int, ProcMolecularFile]:
        """
        Retrieves molecular classification files.
        
        Returns:
            Dict[int, ProcMolecularFile]: Dictionary of molecular classification files.
        """
        files = {int(idx): f for idx, f in self.molecular_classify.items()}
        return files


def print_main_modules(pp: ProcParam, sn: str):
    """
    Print the main modules of the processing pipeline.

    Args:
        pp (ProcParam): The processing parameters object.
        sn (str): The sample name.

    Returns:
        None
    """
    show_table = PrettyTable()
    show_table.field_names = ["SN", "QC", "Alignment", "Matrix_extract", "Report", "Annotation"]
    show_table.add_row([sn, pp.run.qc, pp.run.alignment, pp.run.matrix_extract, pp.run.report, pp.run.annotation])
    clog.info(f"\n {show_table}")


def read_param_file(file_path: str, cfg: Config, out_path: Optional[str] = None) -> ProcParam:
    """
    Reads a parameter file and returns a ProcParam object.

    Args:
        file_path (str): Path to the parameter file.
        cfg (object): Configuration object.
        out_path (str, optional): Optional path to save the processed parameters as a JSON file.

    Returns:
        ProcParam: A ProcParam object containing the parameters read from the file.
    """
    with open(file_path, 'r') as fd:
        dct = json.load(fd)
    pp = ProcParam(**dct)
    pp.check_inputs(cfg=cfg)
    if out_path is not None:
        dict2json(pp.model_dump(), json_path=out_path)
    return pp


def main():
    param_file = "/media/Data1/user/dengzhonghan/code/cellbin2dev/cellbin2/demo_data/qc_result/A02677B5/A02677B5_params.json"
    cfg_file = "/media/Data1/user/dengzhonghan/code/cellbin2dev/cellbin2/cellbin2/config/cellbin.yaml"
    sn = "A02677B5"
    cfg = Config(cfg_file)
    # out = "/media/Data/dzh/data/cellbin2/tmp"
    # im_path = "/media/Data/dzh/data/cellbin2/demo_data/C04042E3/C04042E3_fov_stitched.tif"
    # track_s_type = "HE"
    # matrix_path = "/media/Data/dzh/data/cellbin2/demo_data/C04042E3/C04042E3.raw.gef"
    # if_path = "/media/Data/dzh/data/cellbin2/demo_data/SS200000045_M5/SS200000045_M5_ATP_IF_fov_stitched.tif," \
    #           "/media/Data/dzh/data/cellbin2/demo_data/SS200000045_M5/SS200000045_M5_CD31_IF_fov_stitched.tif," \
    #           "/media/Data/dzh/data/cellbin2/demo_data/SS200000045_M5/SS200000045_M5_NeuN_IF_fov_stitched.tif"
    with open(param_file, 'r') as fd:
        dct = json.load(fd)
    pp = read_param_file(file_path=param_file, cfg=cfg)
    print_main_modules(pp, sn)
    # template = getattr(pp.image_process, track_s_type)
    # template.file_path = im_path
    #
    # trans_tp = pp.image_process['Transcriptomics']
    # trans_tp.file_path = matrix_path
    # print()
    # print(pp.analysis.report)


if __name__ == '__main__':
    main()
