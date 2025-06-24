import sys
import os
import shutil
from typing import List, Dict, Any, Tuple, Union, Optional
from pathlib import Path

import numpy as np

from cellbin2.utils.config import Config
from cellbin2.utils.common import TechType, FILES_TO_KEEP, ErrorCode, FILES_TO_KEEP_RESEARCH
from cellbin2.utils.stereo import generate_stereo_file
from cellbin2.utils.tar import update_ipr_in_tar
from cellbin2.utils import clog
from cellbin2.image import cbimread, CBImage
from cellbin2.utils.stereo_chip import StereoChip
from cellbin2.utils import ipr, rpi
from cellbin2.utils.weights_manager import WeightDownloader
from cellbin2.image import cbimwrite
from cellbin2.modules import naming, run_state
from cellbin2.modules.metadata import ProcParam, ProcFile, read_param_file
from cellbin2.contrib.fast_correct import run_fast_correct
from cellbin2.utils.pro_monitor import process_decorator
from cellbin2.modules.extract.register import run_register, transform_to_register
from cellbin2.modules.extract.transform import run_transform
from cellbin2.modules.extract.tissue_seg import run_tissue_seg
from cellbin2.modules.extract.cell_seg import run_cell_seg
from cellbin2.contrib.mask_manager import BestTissueCellMask, MaskManagerInfo
from cellbin2.modules.extract.matrix_extract import extract4stitched


class Scheduler(object):
    """
        配准、分割及校准、矩阵提取
    """

    def __init__(self, config_file: str, chip_mask_file: str, weights_root: str):
        self.weights_root = weights_root
        self.param_chip = StereoChip(chip_mask_file)
        self.config = Config(config_file, weights_root)
        self._files: Dict[int, ProcFile] = {}
        self._ipr = ipr.ImageProcessRecord()
        self._channel_images: Dict[str, Union[ipr.IFChannel, ipr.ImageChannel]] = {}  # ipr.ImageChannel
        self._output_path: str = ''

        self.p_naming: naming.DumpPipelineFileNaming = None
        self.matrix_file = None
        # self._image_naming: naming.DumpImageFileNaming
        # self._matrix_naming: naming.DumpMatrixFileNaming

    @process_decorator('GiB')
    def _dump_ipr(self, output_path: str):
        ipr.write(file_path=output_path, ipr=self._ipr, extra_images=self._channel_images)
        clog.info('Dump ipr to {}'.format(output_path))

    @process_decorator('GiB')
    def _dump_rpi(self, rpi_path: str):
        data = {}
        for idx, f in self._files.items():
            if f.is_image:
                g_name = f.get_group_name(sn=self.param_chip.chip_name)
                n = naming.DumpImageFileNaming(
                    sn=self.param_chip.chip_name, stain_type=g_name, save_dir=self._output_path)

                data[g_name] = {}
                if os.path.exists(n.cell_mask):
                    data[g_name]['CellMask'] = n.cell_mask
                elif os.path.exists(n.transform_cell_mask):
                    data[g_name]['CellMaskTransform'] = n.transform_cell_mask
                if self.debug:
                    if os.path.exists(n.cell_mask_raw):
                        data[g_name]['CellMaskRaw'] = n.cell_mask_raw
                    elif os.path.exists(n.transform_cell_mask_raw):
                        data[g_name]['CellMaskRawTransform'] = n.transform_cell_mask_raw

                if os.path.exists(n.registration_image):
                    data[g_name]['Image'] = n.registration_image
                elif os.path.exists(n.transformed_image):
                    data[g_name]['Image'] = n.transformed_image

                if os.path.exists(n.tissue_mask):
                    data[g_name]['TissueMask'] = n.tissue_mask
                elif os.path.exists(n.transform_tissue_mask):
                    data[g_name]['TissueMaskTransform'] = n.transform_tissue_mask

                if self.debug:
                    if os.path.exists(n.tissue_mask_raw):
                        data[g_name]['TissueMaskRaw'] = n.tissue_mask_raw
                    elif os.path.exists(n.transform_tissue_mask_raw):
                        data[g_name]['TissueMaskRawTransform'] = n.transform_tissue_mask_raw
        data['final'] = {}
        data['final']['CellMask'] = self.p_naming.final_nuclear_mask
        data['final']['TissueMask'] = self.p_naming.final_tissue_mask
        data['final']['CellMaskCorrect'] = self.p_naming.final_cell_mask
        rpi.write(h5_path=rpi_path, extra_images=data)
        clog.info('Dump rpi to {}'.format(rpi_path))

    def _weights_check(self, ):
        weights = []
        for idx, f in self._files.items():
            if f.cell_segmentation:
                wp = self.config.cell_segmentation.get_weights_path(f.tech)
                weights.append(os.path.basename(wp))

            if f.tissue_segmentation:
                wp = self.config.tissue_segmentation.get_weights_path(f.tech)
                weights.append(os.path.basename(wp))

        weights = list(set(weights))
        wd = WeightDownloader(save_dir=self.weights_root)
        flag = wd.download_weight_by_names(weight_names=weights)
        if flag != 0: clog.warning('Failed to retrieve the weights file from local or server')

        return flag

    def _data_check(self, ):
        if len(self._files) < 1:
            clog.warning('No data was found that needed to be analyzed')
            return 3
        else:
            wh = []
            for idx, f in self._files.items():
                if not f.is_image:
                    continue
                if not os.path.exists(f.file_path):
                    clog.warning('Missing file, {}'.format(f.file_path))
                    sys.exit(ErrorCode.missFile.value)  # 缺失文件，非正常退出
                image = cbimread(f.file_path)
                wh.append([image.width, image.height])

            s = np.unique(wh, axis=0)
            if s.shape[0] > 1:
                clog.warning('The sizes of the images are inconsistent')
                return 1
            elif s.shape[0] == 1:
                clog.info(
                    'Images info as (size, channel, depth) == ({}, {}, {})'.format(s[0], image.channel, image.depth))
            else:
                clog.info('No image data need deal')
                return 2
        return 0

    def run_segmentation(
            self,
            f,
            im_path,
            ts_raw_save_path,
            cs_raw_save_path,
            ts_save_path,
            cs_save_path,
            cur_c_image: Optional[Union[ipr.ImageChannel, ipr.IFChannel]] = None
    ):
        final_tissue_mask = None
        final_cell_mask = None
        if f.tissue_segmentation:
            tissue_mask = run_tissue_seg(
                image_file=f,
                image_path=im_path,
                save_path=ts_raw_save_path,
                config=self.config,
                chip_info=self.param_chip,
                channel_image=cur_c_image
            )
            final_tissue_mask = tissue_mask
        if f.cell_segmentation:
            cell_mask = run_cell_seg(
                image_file=f,
                image_path=im_path,
                save_path=cs_raw_save_path,
                config=self.config,
                channel_image=cur_c_image
            )
            final_cell_mask = cell_mask
        if f.tissue_segmentation and f.cell_segmentation:
            tissue_mask = cbimread(ts_raw_save_path, only_np=True)
            cell_mask = cbimread(cs_raw_save_path, only_np=True)
            c_box = None
            if cur_c_image is not None:
                c_box = cur_c_image.Stitch.TransformChipBBox.get()
            input_data = MaskManagerInfo(
                tissue_mask=tissue_mask,
                cell_mask=cell_mask,
                chip_box=c_box,
                method=1,
                stain_type=f.tech
            )
            btcm = BestTissueCellMask.get_best_tissue_cell_mask(input_data=input_data)
            final_tissue_mask = btcm.best_tissue_mask
            final_cell_mask = btcm.best_cell_mask
        if final_cell_mask is not None:
            cbimwrite(
                output_path=cs_save_path,
                files=final_cell_mask
            )
        if final_tissue_mask is not None:
            cbimwrite(
                output_path=ts_save_path,
                files=final_tissue_mask
            )

    def run_single_image(self):
        # 遍历单张图像，执行单张图的模块
        for idx, f in self._files.items():
            clog.info('======>  File[{}] CellBin, {}'.format(idx, f.file_path))
            if f.is_image:
                g_name = f.get_group_name(sn=self.param_chip.chip_name)
                cur_f_name = naming.DumpImageFileNaming(
                    sn=self.param_chip.chip_name,
                    stain_type=g_name,
                    save_dir=self._output_path
                )
                cur_c_image = None
                # stitch
                shutil.copy2(f.file_path, cur_f_name.stitch_image)
                if self._channel_images is not None and self._ipr is not None:
                    # 传ipr进来了
                    if not f.tech == TechType.IF:
                        # TODO: deal with two versions of ipr
                        qc_ = self._channel_images[g_name].QCInfo
                        if hasattr(qc_, 'QcPassFlag'):
                            qc_flag = getattr(qc_, 'QcPassFlag')
                        else:
                            qc_flag = getattr(qc_, 'QCPassFlag')
                        if qc_flag != 1:  # 现有条件下无法配准
                            clog.warning('Image QC not pass, cannot deal this pipeline')
                            sys.exit(ErrorCode.qcFail.value)
                    # Transform 操作：Transform > segmentation > mask merge & expand
                    cur_c_image = self._channel_images[g_name]
                    # transform in & out
                    run_transform(
                        file=f,
                        channel_images=self._channel_images,
                        param_chip=self.param_chip,
                        files=self._files,
                        cur_f_name=cur_f_name,
                        if_track=f.registration.trackline,
                        research_mode=self.research_mode,
                    )
                    self.run_segmentation(
                        f=f,
                        im_path=cur_f_name.transformed_image,
                        ts_raw_save_path=cur_f_name.transform_tissue_mask_raw,
                        cs_raw_save_path=cur_f_name.transform_cell_mask_raw,
                        ts_save_path=cur_f_name.transform_tissue_mask,
                        cs_save_path=cur_f_name.transform_cell_mask,
                        cur_c_image=cur_c_image
                    )
                else:
                    # 没传ipr进来，直接基于输入的图进行分割
                    shutil.copy2(cur_f_name.stitch_image, cur_f_name.transformed_image)
                    self.run_segmentation(
                        f=f,
                        im_path=cur_f_name.stitch_image,
                        ts_raw_save_path=cur_f_name.transform_tissue_mask_raw,
                        cs_raw_save_path=cur_f_name.transform_cell_mask_raw,
                        ts_save_path=cur_f_name.transform_tissue_mask,
                        cs_save_path=cur_f_name.transform_cell_mask,
                        cur_c_image=cur_c_image
                    )
            else:
                if f.tissue_segmentation or f.cell_segmentation:
                    cur_m_naming = naming.DumpMatrixFileNaming(
                        sn=self.param_chip.chip_name,
                        m_type=f.tech.name,
                        save_dir=self._output_path,
                    )
                    cm = extract4stitched(
                        image_file=f,
                        param_chip=self.param_chip,
                        m_naming=cur_m_naming,
                        config=self.config,
                        detect_feature=False
                    )
                    if f.tissue_segmentation:
                        run_tissue_seg(
                            image_file=f,
                            image_path=cur_m_naming.heatmap,
                            save_path=cur_m_naming.tissue_mask,
                            chip_info=self.param_chip,
                            config=self.config,
                        )
                    if f.cell_segmentation:
                        run_cell_seg(
                            image_file=f,
                            image_path=cur_m_naming.heatmap,
                            save_path=cur_m_naming.cell_mask,
                            config=self.config,
                        )

    def run_mul_image(self):
        # 这里涉及多张图的配合，因为是配准。所以默认但张图的处理都结束了
        for idx, f in self._files.items():
            if f.is_image:
                clog.info('======>  File[{}] CellBin, {}'.format(idx, f.file_path))
                g_name = f.get_group_name(sn=self.param_chip.chip_name)
                cur_f_name = naming.DumpImageFileNaming(
                    sn=self.param_chip.chip_name,
                    stain_type=g_name,
                    save_dir=self._output_path
                )
                if self._channel_images is not None and self._ipr is not None:
                    if f.registration.fixed_image == -1 and f.registration.reuse == -1:
                        continue
                    if f.registration.fixed_image == -1 and self._files[
                        f.registration.reuse].registration.fixed_image == -1:
                        continue
                    run_register(
                        image_file=f,
                        cur_f_name=cur_f_name,
                        files=self._files,
                        channel_images=self._channel_images,
                        output_path=self._output_path,
                        param_chip=self.param_chip,
                        config=self.config,
                        debug=self.debug
                    )
                    if f.registration.fixed_image != -1:
                        fixed = self._files[f.registration.fixed_image]
                        if fixed.is_matrix:
                            self.matrix_file = self._files[f.registration.fixed_image]
                else:
                    transform_to_register(
                        cur_f_name=cur_f_name
                    )

    def run_merge_masks(self):
        for idx, m in self.molecular_classify_files.items():
            clog.info('======>  Extract[{}], {}'.format(idx, m))
            picked_mask = m.cell_mask
            final_nuclear_path = self.p_naming.final_nuclear_mask
            final_t_mask_path = self.p_naming.final_tissue_mask
            final_cell_mask_path = self.p_naming.final_cell_mask
            if len(picked_mask) == 1:  # 就一个mask
                im_naming = naming.DumpImageFileNaming(
                    sn=self.param_chip.chip_name,
                    stain_type=self._files[picked_mask[0]].get_group_name(sn=self.param_chip.chip_name),
                    save_dir=self._output_path
                )
                to_fast = final_nuclear_path
            else:  # 两个mask，现在默认第一个是核mask，第二个是膜mask
                im_naming1 = naming.DumpImageFileNaming(
                    sn=self.param_chip.chip_name,
                    stain_type=self._files[picked_mask[0]].get_group_name(sn=self.param_chip.chip_name),
                    save_dir=self._output_path
                )
                im_naming2 = naming.DumpImageFileNaming(
                    sn=self.param_chip.chip_name,
                    stain_type=self._files[picked_mask[1]].get_group_name(sn=self.param_chip.chip_name),
                    save_dir=self._output_path
                )
                if im_naming1.stain_type == TechType.IF.name:
                    im_naming = im_naming2
                else:
                    im_naming = im_naming1
                merged_cell_mask_path = im_naming.cell_mask_merged
                if not os.path.exists(merged_cell_mask_path):
                    from cellbin2.contrib.mask_manager import merge_cell_mask
                    # TODO: 这里暂时就默认im_naming1是核分割结果，im_naming2是膜分割结果
                    if not im_naming1.cell_mask.exists() or im_naming2.cell_mask.exists():
                        continue
                    mask1 = cbimread(im_naming1.cell_mask, only_np=True)
                    mask2 = cbimread(im_naming2.cell_mask, only_np=True)
                    merged_mask = merge_cell_mask(mask1, mask2)
                    cbimwrite(merged_cell_mask_path, merged_mask)
                to_fast = merged_cell_mask_path
            if im_naming.cell_mask.exists():
                shutil.copy2(im_naming.cell_mask, final_nuclear_path)
            if im_naming.tissue_mask.exists():
                shutil.copy2(im_naming.tissue_mask, final_t_mask_path)
            # here, we got final cell mask and final tissue mask
            if not os.path.exists(final_cell_mask_path) and to_fast.exists():
                fast_mask = run_fast_correct(
                    mask_path=to_fast,
                    distance=self.config.cell_correct.expand_r,
                    n_jobs=self.config.cell_correct.process
                )
                cbimwrite(final_cell_mask_path, fast_mask)

    def run(self, chip_no: str, input_image: str,
            stain_type: str, param_file: str,
            output_path: str, ipr_path: str,
            matrix_path: str, kit: str, debug: bool, research_mode: bool):

        self._output_path = output_path
        self.debug = debug
        self.research_mode = research_mode
        # 芯片信息加载
        self.param_chip.parse_info(chip_no)
        self.p_naming = naming.DumpPipelineFileNaming(chip_no=chip_no, save_dir=self._output_path)

        # 数据加载
        pp = read_param_file(
            file_path=param_file,
            cfg=self.config,
            out_path=self.p_naming.input_json
        )

        self._files = pp.get_image_files(do_image_qc=False, do_scheduler=True, cheek_exists=False)
        self.molecular_classify_files = pp.get_molecular_classify()
        pp.print_files_info(self._files, mode='Scheduler')

        # 数据校验失败则退出
        flag1 = self._data_check()
        if flag1 not in [0, 2]:
            return 1
        if flag1 == 0:
            if os.path.exists(ipr_path):
                self._ipr, self._channel_images = ipr.read(ipr_path)
            else:
                clog.info(f"No existing ipr founded, assumes qc has not been done before")
                self._ipr, self._channel_images = None, None

        # 模型加载
        flag2 = self._weights_check()
        if flag2 != 0:
            sys.exit(1)

        self.run_single_image()  # transform->tissue seg->cellseg
        self.run_mul_image()  # registration between images

        if flag1 == 0 and self._channel_images is not None and self._ipr is not None:
            self._dump_ipr(self.p_naming.ipr)

        self.run_merge_masks()  # merge multi masks if needed

        if flag1 in [0, 2]:
            self._dump_rpi(self.p_naming.rpi)
        if research_mode:
            if self.p_naming.tar_gz.exists() and self.p_naming.ipr.exists():
                update_ipr_in_tar(
                    tar_path=self.p_naming.tar_gz,
                    ipr_path=self.p_naming.ipr,
                )

            if self.matrix_file is not None:
                matrix_naming = naming.DumpMatrixFileNaming(
                    sn=chip_no,
                    m_type=self.matrix_file.tech.name,
                    save_dir=output_path
                )
                matrix_template = matrix_naming.matrix_template
            else:
                matrix_template = Path("")

            generate_stereo_file(
                registered_image=self.p_naming.rpi,
                compressed_image=self.p_naming.tar_gz,
                matrix_template=matrix_template,
                save_path=self.p_naming.stereo,
                sn=chip_no
            )
        if not self.debug:
            f_to_keep = FILES_TO_KEEP_RESEARCH if research_mode else FILES_TO_KEEP
            self.del_files(f_to_keep)

    def del_files(self, f_to_keep):
        all_ = []
        k_ = []
        remove_ = []
        for idx, f in self._files.items():
            g_name = f.get_group_name(sn=self.param_chip.chip_name)
            if f.is_matrix:
                f_name = naming.DumpMatrixFileNaming(
                    sn=self.param_chip.chip_name,
                    m_type=f.tech.name,
                    save_dir=self._output_path,
                )
            else:
                f_name = naming.DumpImageFileNaming(
                    sn=self.param_chip.chip_name,
                    stain_type=g_name,
                    save_dir=self._output_path
                )
            for p in dir(f_name):
                att = getattr(f_name, p)
                pt = f_name.__class__.__dict__.get(p)
                if isinstance(pt, property) and att.exists():
                    all_.append(att)
                    if pt not in f_to_keep:
                        remove_.append(att)
                    else:
                        k_.append(att)
        for p_p in dir(self.p_naming):
            p_att = getattr(self.p_naming, p_p)
            p_pt = self.p_naming.__class__.__dict__.get(p_p)
            if isinstance(p_pt, property) and p_att.exists():
                all_.append(p_att)
                if p_pt not in f_to_keep:
                    remove_.append(p_att)
                else:
                    k_.append(p_att)
        for f in os.listdir(self._output_path):
            path = os.path.join(self._output_path, f)
            if Path(path) in remove_:
                os.remove(path)


def scheduler_pipeline(weights_root: str, chip_no: str, input_image: str, stain_type: str,
                       param_file: str, output_path: str, matrix_path: str, ipr_path: str,
                       kit: str, debug: bool = False, research_mode=False):
    """
    :param weights_root: CNN权重文件本地存储目录路径
    :param chip_no: 样本芯片号
    :param input_image: 染色图本地路径
    :param stain_type: 染色图对应的染色类型
    :param param_file: 入参文件本地路径
    :param output_path: 输出文件本地存储目录路径
    :param matrix_path: 表达矩阵本地存储路径
    :param ipr_path: 图像记录文件本地存储路径
    :param kit:
    :return: int(状态码)
    """
    curr_path = os.path.dirname(os.path.realpath(__file__))
    config_file = os.path.join(curr_path, r'../config/cellbin.yaml')
    chip_mask_file = os.path.join(curr_path, r'../config/chip_mask.json')

    iqc = Scheduler(config_file=config_file, chip_mask_file=chip_mask_file, weights_root=weights_root)
    iqc.run(chip_no=chip_no,
            input_image=input_image,
            stain_type=stain_type,
            param_file=param_file,
            output_path=output_path,
            ipr_path=ipr_path,
            matrix_path=matrix_path, kit=kit, debug=debug, research_mode=research_mode)


def main(args, para):
    scheduler_pipeline(weights_root=args.weights_root, chip_no=args.chip_no,
                       input_image=args.image_path, stain_type=args.stain_type,
                       param_file=args.param_file, output_path=args.output_path,
                       ipr_path=args.ipr_path, matrix_path=args.matrix_path, kit=args.kit)


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
    parser.add_argument("-k", "--kit", action="store", dest="kit", type=str, required=False,
                        help="Kit Type.(Transcriptomics, Protein)")
    parser.add_argument("-r", "--ipr_path", action="store", dest="ipr_path", type=str, required=True,
                        help="Path of image process record file.")
    parser.add_argument("-m", "--matrix_path", action="store", dest="matrix_path", type=str, required=True,
                        help="Path of matrix file.")
