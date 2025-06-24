import os
import sys
from copy import deepcopy
from typing import Dict, Optional

CURR_PATH = os.path.dirname(os.path.realpath(__file__))
CB2_PATH = os.path.dirname(CURR_PATH)
sys.path.append(CB2_PATH)
from cellbin2.utils.common import TechType
from cellbin2.modules import naming
from cellbin2.utils import clog
import cellbin2
from cellbin2.utils import ipr
from cellbin2.modules.metadata import read_param_file, ProcParam, ProcFile, print_main_modules
from cellbin2.utils.config import Config
from cellbin2.modules.metrics import ImageSource
from cellbin2.utils import dict2json
from cellbin2.utils.common import KIT_VERSIONS, KIT_VERSIONS_R, sPlaceHolder, bPlaceHolder, ErrorCode
from cellbin2.utils.pro_monitor import process_decorator
from cellbin2.utils.weights_manager import DEFAULT_WEIGHTS_DIR

CONFIG_PATH = os.path.join(CURR_PATH, 'config')
# DEFAULT_WEIGHTS_DIR = os.path.join(CURR_PATH, "weights")

CONFIG_FILE = os.path.join(CONFIG_PATH, 'cellbin.yaml')
CHIP_MASK_FILE = os.path.join(CONFIG_PATH, 'chip_mask.json')
DEFAULT_PARAM_FILE = os.path.join(CONFIG_PATH, 'default_param.json')
SUPPORTED_TRACK_STAINED_TYPES = (TechType.ssDNA.name, TechType.DAPI.name, TechType.HE.name)
SUPPORTED_STAINED_Types = []


class CellBinPipeline(object):

    def __init__(self, config_file: str, chip_mask_file: str, weights_root: str) -> None:
        # alg
        self._chip_mask_file = chip_mask_file
        self._config_file = config_file
        self._weights_root = weights_root

        # data
        self._chip_no: str = ''
        self._input_image: str = ''
        self._stain_type: str = TechType.ssDNA.name
        self._param_file: str = ''
        self._output_path: str = ''
        self._matrix_path: str = ''
        self._kit: str = ''

        # naming
        self._naming: Optional[naming.DumpPipelineFileNaming] = None

        # 内部需要的
        self.pp: ProcParam
        self.config: Config

        #
        self._if_report = False
        self.more_images: Optional[Dict] = None
        self._protein_matrix_path: str = ''
        self.research: bool = False

    def image_quality_control(self, ):
        """ 完成图像 QC 流程 """
        if self.pp.run.qc:
            from cellbin2.modules import image_qc
            # if self._naming.ipr.exists():
            #     clog.info('Image QC has been done')
            #     return 0
            s_code = image_qc.image_quality_control(
                weights_root=self._weights_root,
                chip_no=self._chip_no,
                input_image=self._input_image,
                stain_type=self._stain_type,
                param_file=self._param_file,
                output_path=self._output_path,
                debug=self._debug,
                research_mode=self.research
            )
            if s_code != 0:
                sys.exit(1)

    def image_analysis(self, ):
        """ 完成图像、配准、校准、分割、矩阵提取等分析流程 """
        if self.pp.run.alignment:
            from cellbin2.modules import scheduler
            # if self._naming.rpi.exists():
            #     clog.info('scheduler has been done')
            #     return 0
            scheduler.scheduler_pipeline(weights_root=self._weights_root, chip_no=self._chip_no,
                                         input_image=self._input_image, stain_type=self._stain_type,
                                         param_file=self._param_file, output_path=self._output_path,
                                         matrix_path=self._matrix_path,
                                         ipr_path=self._naming.ipr, kit=self._kit, debug=self._debug,
                                         research_mode=self.research)

    def m_extract(self):
        if self.pp.run.matrix_extract:
            from cellbin2.modules.extract.matrix_extract import extract4matrix
            mcf = self.pp.get_molecular_classify()
            files = self.pp.get_image_files(do_image_qc=False, do_scheduler=True, cheek_exists=True)
            p_naming = naming.DumpPipelineFileNaming(chip_no=self._chip_no, save_dir=self._output_path)
            for idx, m in mcf.items():
                if m.exp_matrix == -1:
                    continue
                matrix = files[m.exp_matrix]
                extract4matrix(
                    p_naming=p_naming,
                    image_file=matrix,
                    m_naming=naming.DumpMatrixFileNaming(
                        sn=self._chip_no,
                        m_type=matrix.tech.name,
                        save_dir=self._output_path
                    ),
                )

    def metrics(self, ):
        """ 计算指标 """
        def _get_stitched(files: Dict[int, ProcFile], g_name: str, sn: str):
            for i, v in files.items():
                if v.get_group_name(sn) == g_name:
                    return v.file_path

        if self.pp.run.report:
            from cellbin2.modules import metrics
            if self._naming.metrics.exists():
                clog.info('Metrics step has been done')
                return
            config = Config(self._config_file, self._weights_root)
            pp = read_param_file(file_path=self._param_file, cfg=config)
            files = pp.get_image_files(do_image_qc=False, do_scheduler=True, cheek_exists=True)
            ipr_file = str(self._naming.ipr)
            rpi_file = str(self._naming.rpi)
            # 图片类的输入
            ipr_r, channel_images = ipr.read(ipr_file)
            src_img_dict = {}
            for c_name, c_info in channel_images.items():
                c_pipeline_name = naming.DumpImageFileNaming(
                    sn=self._chip_no, stain_type=c_name,
                    save_dir=self._output_path
                )
                src_img_dict[c_name] = {}
                for filed_name, filed in ImageSource.model_fields.items():
                    if filed_name == "cell_correct_mask":
                        fp = getattr(self._naming, f"final_cell_mask")
                    elif filed_name == 'stitch_image':
                        fp = _get_stitched(files, g_name=c_name, sn=self._chip_no)
                    else:
                        fp = getattr(c_pipeline_name, filed_name)
                    if not os.path.exists(str(fp)):
                        fp = ""
                    src_img_dict[c_name][filed_name] = str(fp)
            gene_matrix = pp.get_molecular_classify().items()
            matrix_dict = {}
            for idx, m in gene_matrix:
                matrix = files[m.exp_matrix]
                cur_m_type = matrix.tech.name
                if cur_m_type not in matrix_dict:
                    cur_m_name = naming.DumpMatrixFileNaming(sn=self._chip_no, m_type=cur_m_type,
                                                             save_dir=self._output_path)
                    cur_m_src_files = metrics.MatrixArray(
                        tissue_bin_matrix=str(cur_m_name.tissue_bin_matrix),
                        cell_bin_matrix=str(cur_m_name.cell_bin_matrix),
                        cell_bin_adjusted_matrix=str(cur_m_name.cell_correct_bin_matrix),
                        bin1_matrix=str(matrix.file_path),
                        matrix_type=matrix.tech
                    )
                    matrix_dict[cur_m_type] = cur_m_src_files
            matrix_lists = list(matrix_dict.values())
            fs = metrics.FileSource(
                ipr_file=ipr_file, rpi_file=rpi_file, matrix_list=[matrix_lists[0]], sn=self._chip_no,
                image_dict=src_img_dict)  # TODO 蛋白矩阵没放进去
            metrics.calculate(param=fs, output_path=self._output_path)
            clog.info("Metrics generated")

    def export_report(self, ):
        if self.pp.run.report:
            """ 生成及导出报告 """
            from cellbin2.modules import report_m

            src_file_path = self._naming.metrics
            report_m.creat_report(matric_json=src_file_path, save_path=self._output_path)

    def usr_inp_to_param(self):
        if self._kit.endswith("R"):
            self.research = True
        self.config = Config(self._config_file, self._weights_root)
        if self._param_file is None:
            if self._input_image is None:
                raise Exception(f"the input image can not be empty if param file is not provided")
            tech, version = self._kit.split("V")
            if self._kit.endswith("R"):
                param_file = os.path.join(CONFIG_PATH, tech.strip(" ") + " R" + ".json")
            else:
                param_file = os.path.join(CONFIG_PATH, tech.strip(" ") + ".json")
            pp = read_param_file(file_path=param_file, cfg=self.config)
            new_pp = ProcParam(run=pp.run)
            # track image (ssDNA, HE, DAPI)
            im_count = 0
            trans_exp_idx = -1
            protein_exp_idx = -1
            nuclear_cell_idx = -1

            template = pp.image_process[self._stain_type]
            template.file_path = self._input_image
            new_pp.image_process[str(im_count)] = template
            nuclear_cell_idx = im_count
            im_count += 1

            # 转录组矩阵
            if self._matrix_path is not None:
                trans_tp = pp.image_process[TechType.Transcriptomics.name]
                trans_tp.file_path = self._matrix_path
                new_pp.image_process[str(im_count)] = trans_tp
                trans_exp_idx = im_count
                new_pp.image_process[str(nuclear_cell_idx)].registration.fixed_image = trans_exp_idx
                im_count += 1

            # more images, IF, H&E
            if self.more_images is not None:
                for stain_type, file_path in self.more_images.items():
                    if TechType.IF.name in stain_type:
                        # IF images
                        inner_stain_type = getattr(TechType, stain_type, TechType.IF)
                        im_process_cp = deepcopy(pp.image_process[inner_stain_type.name])
                        im_process_cp.file_path = file_path
                        # im_process_cp.tech_type = stain_type
                        new_pp.image_process[str(im_count)] = im_process_cp
                        new_pp.image_process[str(im_count)].registration.reuse = nuclear_cell_idx
                        im_count += 1
                    else:
                        # H&E
                        raise Exception("Not supported")

            if self._protein_matrix_path is not None:
                protein_tp = pp.image_process[TechType.Protein.name]
                protein_tp.file_path = self._protein_matrix_path
                new_pp.image_process[str(im_count)] = protein_tp
                protein_exp_idx = im_count
                if new_pp.image_process[str(nuclear_cell_idx)].registration.fixed_image == -1:
                    new_pp.image_process[str(nuclear_cell_idx)].registration.fixed_image = protein_exp_idx

            # end of image part info parsing

            # 矩阵提取
            if trans_exp_idx != -1:
                trans_m_tp = pp.molecular_classify[TechType.Transcriptomics.name]
                trans_m_tp.exp_matrix = trans_exp_idx
                trans_m_tp.cell_mask = [nuclear_cell_idx]
                new_pp.molecular_classify['0'] = trans_m_tp

            if protein_exp_idx != -1:
                protein_m_tp = pp.molecular_classify[TechType.Protein.name]
                protein_m_tp.exp_matrix = protein_exp_idx
                protein_m_tp.cell_mask = [nuclear_cell_idx]
                new_pp.molecular_classify['1'] = protein_m_tp
            new_pp.run.report = True if self._if_report else False
            param_f_p = self._naming.input_json
            dict2json(new_pp.model_dump(), json_path=param_f_p)
            self._param_file = param_f_p
            self.pp = new_pp
        else:
            self.pp = read_param_file(
                file_path=self._param_file,
                cfg=self.config
            )
        print_main_modules(self.pp, self._chip_no)

    def run(self, chip_no: str, input_image: str, more_images: str,
            stain_type: str, param_file: str,
            output_path: str, matrix_path: str, protein_matrix_path: str, kit: str, if_report: bool, debug: bool):
        """ 全分析流程 """
        self._chip_no = chip_no
        self._input_image = input_image
        self.more_images = more_images
        self._stain_type = stain_type
        self._param_file = param_file
        self._output_path = output_path
        self._matrix_path = matrix_path
        self._protein_matrix_path = protein_matrix_path
        self._kit = kit
        self._if_report = if_report
        self._debug = debug
        self._naming = naming.DumpPipelineFileNaming(chip_no=chip_no, save_dir=self._output_path)
        # self.pipe_run_state = PipelineRunState(self._chip_no, self._output_path)

        self.usr_inp_to_param()
        self.image_quality_control()  # 图像质控
        self.image_analysis()  # 图像分析
        self.m_extract()  # 矩阵提取
        self.metrics()  # 指标计算
        self.export_report()  # 生成报告


@process_decorator('GiB')
def pipeline(
        chip_no,
        input_image,
        more_images,
        stain_type,
        param_file,
        output_path,
        matrix_path,
        protein_matrix_path,
        kit,
        if_report,
        weights_root,
        debug=False
):
    """
        :param weights_root: CNN权重文件本地存储目录路径
        :param chip_no: 样本芯片号
        :param input_image: 染色图本地路径
        :param stain_type: 染色图对应的染色类型
        :param param_file: 入参文件本地路径
        :param kit: 测序技术
        :param output_path: 输出文件本地存储目录路径
        :param matrix_path: 表达矩阵本地存储路径
        :return: int(状态码)
    """
    os.makedirs(output_path, exist_ok=True)
    clog.log2file(output_path)
    clog.info(f"CellBin Version: {cellbin2.__version__}")
    try:
        if weights_root is None:
            # if user does not provide weight path, use default
            weights_root = DEFAULT_WEIGHTS_DIR
        else:
            if not os.path.isdir(weights_root):
                weights_root = os.path.join(CURR_PATH, 'weights')
            else:
                weights_root = weights_root

        cbp = CellBinPipeline(config_file=CONFIG_FILE, chip_mask_file=CHIP_MASK_FILE, weights_root=weights_root)
        cbp.run(
            chip_no=chip_no,
            input_image=input_image,
            more_images=more_images,
            stain_type=stain_type,
            param_file=param_file,
            output_path=output_path,
            matrix_path=matrix_path,
            protein_matrix_path=protein_matrix_path,
            kit=kit,
            if_report=if_report,
            debug=debug
        )
    except Exception as e:
        clog.exception('Unexpected Error: ')
        sys.exit(ErrorCode.unexpectedError.value)


def main(args, para):
    chip_no = args.chip_no
    input_image = args.input_image
    more_images = args.more_images
    stain_type = args.stain_type
    param_file = args.param_file
    output_path = args.output_path
    matrix_path = args.matrix_file
    protein_matrix_path = args.protein_matrix_file
    kit = args.kit
    weights_root = args.weights_root
    if_report = args.report
    debug = args.debug

    pipeline(
        chip_no,
        input_image,
        more_images,
        stain_type,
        param_file,
        output_path,
        matrix_path,
        protein_matrix_path,
        kit,
        if_report,
        weights_root,
        debug=debug
    )


if __name__ == '__main__':  # main()
    import argparse

    _VERSION_ = '0.1'
    usage_str = f"\npython {os.path.basename(__file__)} \\ \n" \
                f"-c  A03599D1 \\ \n" \
                f"-i  A03599D1_DAPI_fov_stitched.tif \\ \n" \
                f"-mi IF=A02677B5_IF.tif \\ \n" \
                f"-s  DAPI \\ \n" \
                f"-m  A03599D1.raw.gef \\ \n" \
                f"-pr A03599D1.protein.raw.gef \\ \n" \
                f"-w  /cellbin2/weights \\ \n" \
                f"-o  /cellbin2/test/A03599D1_demo1_1 \\ \n" \
                f"-k  \"Stereo-CITE T FF V1.0 R\" \\ \n" \
                f"-r"

    # 负责接收字典类的参数
    class MoreimsKwargs(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            mif_im_count = 0
            setattr(namespace, self.dest, dict())
            for value in values:
                try:
                    # split it into key and value
                    key, value = value.split('=')
                    # assign into dictionary
                    if key not in getattr(namespace, self.dest).keys():
                        getattr(namespace, self.dest)[key] = value
                    else:
                        if key == TechType.IF.name:
                            key = f"unknown{mif_im_count}_{key}"
                            getattr(namespace, self.dest)[key] = value
                    if TechType.IF.name in key:
                        mif_im_count += 1
                except ValueError:
                    print(f"Input error: {value}, input like 'key=value'.")


    parser = argparse.ArgumentParser(
        usage=usage_str
    )
    parser.add_argument("-v", "--version", action="version", version=_VERSION_)
    parser.add_argument("-c", action="store", type=str, required=True, metavar="CHIP_NUMBER", dest="chip_no",
                        help="The SN of chip.")
    parser.add_argument("-i", action="store", type=str, metavar="TRACK_IMAGE_FILE_PATH", dest="input_image",
                        help=f"The path of track image file, choices are: {{{','.join(SUPPORTED_TRACK_STAINED_TYPES)}}}.")
    parser.add_argument("-s", action="store", type=str, metavar="TRACK_IMAGE_STAIN_TYPE", dest="stain_type",
                        choices=SUPPORTED_TRACK_STAINED_TYPES,
                        help=f"The stain type of input image, choices are {{{','.join(SUPPORTED_TRACK_STAINED_TYPES)}}}.")
    parser.add_argument("-mi", action=MoreimsKwargs, nargs="+", dest="more_images",
                        help="The path of other image input file.", metavar="{STAIN_TYPE}={FILE_PATH}")
    parser.add_argument("-m", action="store", type=str, metavar="TRANSCRIPTOMICS_MATRIX_FILE", dest="matrix_file",
                        help="The path of transcriptomics matrix file.")
    parser.add_argument("-pr", action="store", type=str, metavar="PROTEIN_MATRIX_FILE", dest="protein_matrix_file",
                        help="The path of protein matrix file.")
    parser.add_argument("-k", action="store", type=str, default="Stereo-CITE T FF V1.0 R", metavar="KIT_VERSION",
                        dest="kit", choices=KIT_VERSIONS + KIT_VERSIONS_R, help="Kit version")
    parser.add_argument("-p", action="store", type=str, help="The path of input param file.",
                        metavar="PARAM_FILE", dest="param_file")
    parser.add_argument("-w", action="store", type=str, metavar="WEIGHTS_DIR",
                        help="The weights root folder.", dest="weights_root")
    parser.add_argument("-o", action="store", type=str, required=True, metavar="OUTPUT_PATH",
                        help="The results output folder.", dest="output_path")
    parser.add_argument("-r", action="store_true", help="If run report.", dest="report")
    parser.add_argument("-d", action="store_true", default=bPlaceHolder, help="Debug mode", dest="debug")

    parser.set_defaults(func=main)
    (para, args) = parser.parse_known_args()
    para.func(para, args)
