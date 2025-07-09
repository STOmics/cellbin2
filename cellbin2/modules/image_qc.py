# Image Quality Control
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
    def __init__(self, config_file: str, chip_mask_file: str, weights_root: str):
        """
        Initialize the ImageQC class.

        This class is in charge of image quality control operations.

        Args:
            config_file (str): The path to the configuration file.
            chip_mask_file (str): The path to the chip mask file.
            weights_root (str): The path to the weights folder.
        """
        self.weights_root = weights_root
        self.param_chip = StereoChip(chip_mask_file)
        self.config = Config(config_file, weights_root)
        self._fov_wh = (2000, 2000)  # Initial cropping size
        self._files: Dict[int, ProcFile] = {}
        self._ipr = ipr.ImageProcessRecord()
        self._channel_images: Dict[str, Union[ipr.ImageChannel, ipr.IFChannel]] = {}
        self.p_naming: naming.DumpPipelineFileNaming = None

    def _align_channels(self, image_file: ProcFile):
        """
        Aligns the channels of an image using calibration data.

        Args:
            image_file (ProcFile): The image file to be aligned.

        Returns:
            int: Returns 1 if calibration data is missing; otherwise, returns 0.
        """
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

    def _match_channels(self, image_file: ProcFile, output_path: str):
        """
        Matches the channels of an image using calibration data.

        Args:
            image_file (ProcFile): The image file to be matched.

        Returns:
            int: Returns 1 if calibration data is missing; otherwise, returns 0.
        """
        from cellbin2.contrib.chip_transform import chip_transform

        fixed = self._files[image_file.chip_matching]
        if fixed is None:
            return 1
        clog.info('Transform [moving, fixed] == ({}, {})'.format(
            os.path.basename(image_file.file_path), os.path.basename(fixed.file_path)))

        scale = [1, float(image_file.magnification) / float(fixed.magnification)]

        # TODO save path name
        chip_transform(
            fixed_image = fixed.file_path,
            moving_image=image_file.file_path,
            scale = scale,
            output_path = os.path.join(output_path, 'chip_matching.tif')
        )

        # TODO info record
        # self._channel_images[image_file.get_group_name(sn=self.param_chip.chip_name)].Calibration.update(r)

    def _dump_ipr(self, output_path: Union[str, Path]):
        """
        Write the ipr to the specified output path.

        Args:
            output_path (Union[str, Path]): The path where the ipr will be written.

        Returns:
            None
        """
        # Write the ipr to the specified output path
        ipr.write(file_path=output_path, ipr=self._ipr, extra_images=self._channel_images)
        # Log the output path
        clog.info('Dump ipr to {}'.format(output_path))

    def _weights_check(self, ):
        """
        Check and download weights for the files in the current instance.

        This method iterates through the files in the current instance, checks
        if weights are required for each file, and downloads them if necessary.
        It also ensures that the weights are unique and then attempts to download
        them using the WeightDownloader class.

        Returns:
            int: The status code of the weight download process.
        """
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
        """
        Check the data files and their formats.

        This method verifies that the data files exist and have consistent sizes.
        If any file is missing or if the sizes of the images are inconsistent,
        an error message is logged and the program exits with an appropriate error code.

        Returns:
            int: Always returns 0.
        """
        if len(self._files) < 1:
            clog.warning('No data was found that needed to be analyzed')
            return 0
        else:
            clog.info('Start verifying data format')
            wh = {}
            re = {}
            for idx, f in self._files.items():
                if not os.path.exists(f.file_path):
                    clog.error('Missing file, {}'.format(f.file_path))
                    sys.exit(ErrorCode.missFile.value)  # missing file, abnormal exit
                image = cbimread(f.file_path)
                wh[idx] = [image.width, image.height]
                re[idx] = {'channel_align': f.channel_align, 'chip_matching': f.chip_matching}

                clog.info('Images ({}) info as (size, channel, depth) == ({}, {}, {})'.format(
                    f.tag, image.shape, image.channel, image.depth))

            for idx, k in re.items():
                if k['chip_matching'] == -1 and k['channel_align'] == -1:
                    continue
                else:
                    if k['channel_align'] != -1:
                        if wh[idx] != wh[k['channel_align']]:
                            clog.error(f'The sizes of the images are inconsistent: {wh}')
                            sys.exit(ErrorCode.sizeInconsistent.value)

            # s = np.unique(list(wh.values()), axis=0)
            # if s.shape[0] != 1:
            #     clog.error(f'The sizes of the images are inconsistent: {wh}')
            #     sys.exit(ErrorCode.sizeInconsistent.value)
            # clog.info('Images info as (size, channel, depth) == ({}, {}, {})'.format(
            #     s[0], image.channel, image.depth))
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
        """
        Executes the Image Quality Control (ImageQC) process.

        Args:
            chip_no (str): The chip number used to load chip information.
            input_image (str): The path to the input image.
            stain_type (str): The staining type used for image processing.
            param_file (str): The path to the parameter file containing configuration details for image processing.
            output_path (str): The path where output files will be saved.
            debug (bool): Whether to enable debug mode, which outputs additional debug information.
            research_mode (bool): Whether to enable research mode, which compresses the result files.

        Returns:
            int: A status code. 0 indicates success, and 1 indicates failure.
        """

        """ Phase1: Input Preparation """
        # Load chip information and initialize file naming rules
        self.param_chip.parse_info(chip_no)
        self.p_naming = naming.DumpPipelineFileNaming(chip_no, save_dir=output_path)

        # Load data from the parameter file
        pp = read_param_file(
            file_path=param_file,
            cfg=self.config,
            out_path=self.p_naming.input_json
        )

        # Load only files related to ImageQC and check if they exist
        self._files = pp.get_image_files(do_image_qc=True, do_scheduler=False, cheek_exists=False)
        pp.print_files_info(self._files, mode='imageQC')

        # Exit if data validation fails (e.g., size, channel, or bit depth issues)
        flag = self._data_check()
        if flag != 0:
            return 1
        clog.info('Check data finished, as state: PASS')

        # Load the model weights
        flag = self._weights_check()
        if flag != 0:
            clog.warning('Weight file preparation failed, program will exit soon')
            return 1
        clog.info('Prepare DNN weights files finished, as state: PASS')

        """ Phase2: Computation """

        # Iterate through the QC process
        if len(self._files) == 0:
            clog.info('Finished with no data to perform ImageQC')
            return 0
        files = []  # Contains files that will be compressed into a tar.gz archive
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
                # do calibration
                channel_image = ipr.IFChannel()
                self._channel_images[f.get_group_name(sn=self.param_chip.chip_name)] = channel_image
                self._align_channels(f)
                files.append(
                    (f.file_path, f"{f.tech.name}/{f.get_group_name(sn=self.param_chip.chip_name)}/{f.tag}.tif")
                )
            elif f.chip_matching != -1:
                channel_image = ipr.IFChannel()
                self._channel_images[f.get_group_name(sn=self.param_chip.chip_name)] = channel_image
                self._match_channels(f, output_path)
                files.append(
                    (f.file_path, f"{f.tech.name}/{f.get_group_name(sn=self.param_chip.chip_name)}/{f.tag}.tif")
                )
            else:
                channel_image = ipr.ImageChannel()
                self._channel_images[f.get_group_name(sn=self.param_chip.chip_name)] = channel_image
            image = cbimread(f.file_path)
            if f.tech == TechType.IF:  # Product-specific handling: SN_IF.tif -> ipr stain_type = SN_IF
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

        """ Phase3: Output """
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
                          stain_type: str, param_file: str, output_path: str,
                          debug: bool = False, research_mode=False):
    """
    Perform image quality control tasks.

    This function initializes an ImageQC object and runs the quality control process.

    Args:
        weights_root (str): Local directory path where the CNN weight files are stored.
        chip_no (str): Serial number of the sample chip.
        input_image (str): Local path to the stained image.
        stain_type (str): Stain type corresponding to the input image.
        param_file (str): Local path to the input parameter file.
        output_path (str): Local directory path where the output files will be stored.
        debug (bool, optional): Boolean flag to enable debug mode. Defaults to False.
        research_mode (bool, optional): Boolean flag to enable research mode. Defaults to False.

    Returns:
        int: Status code
    """
    curr_path = os.path.dirname(os.path.realpath(__file__))
    config_file = os.path.join(curr_path, r'../config/cellbin.yaml')
    chip_mask_file = os.path.join(curr_path, r'../config/chip_mask.json')

    iqc = ImageQC(config_file=config_file, chip_mask_file=chip_mask_file, weights_root=weights_root)
    return iqc.run(chip_no=chip_no, input_image=input_image, stain_type=stain_type, param_file=param_file,
                   output_path=output_path, debug=debug, research_mode=research_mode)


def main(args, para):
    """
    Main function to execute image quality control.

    This function parses the command - line arguments and parameters,
    and then invokes the `image_quality_control` function to carry out
    the quality control process on the input image.

    Args:
        args (argparse.Namespace): Parsed command - line arguments that contain input parameters.
        para (dict): Additional parameters for the quality control process.

    Returns:
        None
    """
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
