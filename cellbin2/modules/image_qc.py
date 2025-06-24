# 图像质控
import os
import sys
from typing import List, Dict, Any, Tuple, Union
from pathlib import Path

import numpy as np

from cellbin2.utils.config import Config
from cellbin2.utils.stereo_chip import StereoChip
from cellbin2.utils import clog
from cellbin2.image import cbimread, CBImage
from cellbin2.utils.weights_manager import WeightDownloader
from cellbin2.modules.metadata import read_param_file, ProcFile, ProcParam
from cellbin2.utils import ipr
from cellbin2.modules import naming
from cellbin2.modules.extract.qc import run_qc
from cellbin2.utils.common import ErrorCode, TechType
from cellbin2.utils.tar import save_tar


class ImageQC(object):
    """
        入参合法性、模板推导（点/线识别）、芯片检测、清晰度、校准
    """

    def __init__(self, config_file: str, chip_mask_file: str, weights_root: str):
        self.weights_root = weights_root
        self.param_chip = StereoChip(chip_mask_file)
        self.config = Config(config_file, weights_root)
        self._fov_wh = (2000, 2000)  # 初始裁图尺寸
        self._files: Dict[int, ProcFile] = {}
        self._ipr = ipr.ImageProcessRecord()
        self._channel_images: Dict[str, Union[ipr.ImageChannel, ipr.IFChannel]] = {}
        self.p_naming: naming.DumpPipelineFileNaming = None

    def _align_channels(self, image_file: ProcFile):
        from cellbin2.contrib import calibration

        fixed = self._files[image_file.channel_align]
        if fixed is None:
            return 1
        clog.info('Calibration[moving, fixed] == ({}, {})'.format(
            os.path.basename(image_file.file_path), os.path.basename(fixed.file_path)))

        r = calibration.multi_channel_align(cfg=self.config.calibration,
                                            moving_image=image_file.file_path,
                                            fixed_image=fixed.file_path)
        self._channel_images[image_file.get_group_name(sn=self.param_chip.chip_name)].Calibration.update(r)
        clog.info('[Offset-XY, score, Flag] == ({}, {}, {})'.format(r.offset, r.score, r.pass_flag))

        return 0

    def _dump_ipr(self, output_path: Union[str, Path]):
        ipr.write(file_path=output_path, ipr=self._ipr, extra_images=self._channel_images)
        clog.info('Dump ipr to {}'.format(output_path))

    def _weights_check(self, ):
        weights = []
        for idx, f in self._files.items():
            stain_type = f.tech
            if f.registration.trackline:
                wp = self.config.track_points.get_weights_path(stain_type)
                if wp is None:
                    clog.warning('Points detect get weights path failed')
                    sys.exit(ErrorCode.weightDownloadFail.value)
                weights.append(os.path.basename(wp))

            if f.chip_detect:
                wp1 = self.config.chip_detector.get_stage1_weights_path()
                wp2 = self.config.chip_detector.get_stage2_weights_path()
                for wp in [wp1, wp2]:
                    if wp is None:
                        clog.warning('Chip detect get weights path failed')
                        sys.exit(ErrorCode.weightDownloadFail.value)
                    weights.append(os.path.basename(wp))

            if f.quality_control:
                wp = self.config.clarity.get_weights_path(stain_type)
                if wp is None:
                    clog.warning('Clarity get weights path failed')
                    sys.exit(ErrorCode.weightDownloadFail.value)
                weights.append(os.path.basename(wp))

        weights = list(set(weights))

        wd = WeightDownloader(save_dir=self.weights_root)
        flag = wd.download_weight_by_names(weight_names=weights)
        if flag != 0:
            sys.exit(ErrorCode.weightDownloadFail.value)

        return flag

    def _data_check(self, ):
        if len(self._files) < 1:
            clog.warning('No data was found that needed to be analyzed')
            return 0
        else:
            clog.info('Start verifying data format')
            wh = {}
            for idx, f in self._files.items():
                if not os.path.exists(f.file_path):
                    clog.error('Missing file, {}'.format(f.file_path))
                    sys.exit(ErrorCode.missFile.value)  # 缺失文件，非正常退出
                image = cbimread(f.file_path)
                wh[f.tag] = [image.width, image.height]
            s = np.unique(list(wh.values()), axis=0)
            if s.shape[0] != 1:
                clog.error(f'The sizes of the images are inconsistent: {wh}')
                sys.exit(ErrorCode.sizeInconsistent.value)
            clog.info('Images info as (size, channel, depth) == ({}, {}, {})'.format(
                s[0], image.channel, image.depth))
        return 0

    def run(
            self,
            chip_no: str,
            input_image: str,
            stain_type: str,
            param_file: str,
            output_path: str,
            debug: bool,
            research_mode: bool
    ):
        """ Phase1: 输入准备工作 """
        # 芯片信息加载
        self.param_chip.parse_info(chip_no)
        self.p_naming = naming.DumpPipelineFileNaming(chip_no, save_dir=output_path)

        # 数据加载
        pp = read_param_file(
            file_path=param_file,
            cfg=self.config,
            out_path=self.p_naming.input_json
        )

        # 只加载与ImageQC相关的文件，同时检查该文件是否存在
        self._files = pp.get_image_files(do_image_qc=True, do_scheduler=False, cheek_exists=False)
        pp.print_files_info(self._files, mode='imageQC')

        # 数据校验失败则退出（尺寸、通道及位深等信息）
        flag = self._data_check()
        if flag != 0:
            return 1
        clog.info('Check data finished, as state: PASS')

        # 模型加载
        flag = self._weights_check()
        if flag != 0:
            clog.warning('Weight file preparation failed, program will exit soon')
            return 1
        clog.info('Prepare DNN weights files finished, as state: PASS')

        """ Phase2: 计算 """

        # 遍历QC过程
        if len(self._files) == 0:
            clog.info('Finished with no data do imageQC')
            return 0
        files = []  # contain files which will be compressed into tar.gz
        for idx, f in self._files.items():
            clog.info('======>  Image[{}] QC, {}'.format(idx, f.file_path))
            if f.registration.trackline:
                channel_image = run_qc(
                    image_file=f,
                    param_chip=self.param_chip,
                    config=self.config,
                    output_path=output_path,
                    debug=debug
                )
                self._channel_images[f.get_group_name(sn=self.param_chip.chip_name)] = channel_image
                files.append((f.file_path, f"{f.tech.name}/{f.tag}.tif"))
            elif f.channel_align != -1:
                channel_image = ipr.IFChannel()
                self._channel_images[f.get_group_name(sn=self.param_chip.chip_name)] = channel_image
                self._align_channels(f)
                files.append((f.file_path, f"{f.tech.name}/{f.get_group_name(sn=self.param_chip.chip_name)}/{f.tag}.tif"))
            else:
                channel_image = ipr.ImageChannel()
            image = cbimread(f.file_path)
            if f.tech == TechType.IF:  # 产品是这样处理的，SN_IF.tif -> ipr stain_type = SN_IF
                s_type = f.get_group_name(sn=self.param_chip.chip_name)
            else:
                s_type = f.tech_type
            channel_image.update_basic_info(
                chip_name=chip_no,
                channel=image.channel,
                width=image.width,
                height=image.height,
                stain_type=s_type,
                depth=image.depth
            )

        """ Phase3: 输出 """
        self._dump_ipr(self.p_naming.ipr)
        if research_mode:
            files.append((self.p_naming.ipr, os.path.basename(self.p_naming.ipr)))
            clog.info(f"Research mode, start to compress results into {self.p_naming.tar_gz}")
            save_tar(
                save_path=self.p_naming.tar_gz,
                files=files
            )
        clog.info(f"ImageQC finished")
        return 0


def image_quality_control(weights_root: str, chip_no: str, input_image: str,
                          stain_type: str, param_file: str, output_path: str, debug: bool = False, research_mode=False):
    """
    :param weights_root: CNN权重文件本地存储目录路径
    :param chip_no: 样本芯片号
    :param input_image: 染色图本地路径
    :param stain_type: 染色图对应的染色类型
    :param param_file: 入参文件本地路径
    :param output_path: 输出文件本地存储目录路径
    :return: int(状态码)
    """
    curr_path = os.path.dirname(os.path.realpath(__file__))
    config_file = os.path.join(curr_path, r'../config/cellbin.yaml')
    chip_mask_file = os.path.join(curr_path, r'../config/chip_mask.json')

    iqc = ImageQC(config_file=config_file, chip_mask_file=chip_mask_file, weights_root=weights_root)
    return iqc.run(chip_no=chip_no, input_image=input_image, stain_type=stain_type, param_file=param_file,
                   output_path=output_path, debug=debug, research_mode=research_mode)


def main(args, para):
    image_quality_control(weights_root=args.weights_root,
                          chip_no=args.chip_no,
                          input_image=args.image_path,
                          stain_type=args.stain_type,
                          param_file=args.param_file,
                          output_path=args.output_path)


if __name__ == '__main__':
    import argparse

    _VERSION_ = '2.0'

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", action="version", version=_VERSION_)
    parser.add_argument("-c", "--chip_no", action="store", dest="chip_no", type=str, required=True,
                        help="The SN of chip.")
    parser.add_argument("-i", "--image_path", action="store", dest="image_path", type=str, required=False,
                        help="The path of input file.")
    parser.add_argument("-s", "--stain_type", action="store", dest="stain_type", type=str, required=False,
                        help="The stain type of input image.")
    parser.add_argument("-p", "--param_file", action="store", dest="param_file", type=str, required=False,
                        default='', help="The path of input param file.")
    parser.add_argument("-o", "--output_path", action="store", dest="output_path", type=str, required=True,
                        help="The results output folder.")
    parser.add_argument("-w", "--weights_root", action="store", dest="weights_root", type=str, required=True,
                        help="The weights folder.")
