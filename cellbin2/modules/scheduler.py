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
from cellbin2.modules.cellmask_fixer import CellMaskFixer


class Scheduler(object):
    """
    A scheduler class responsible for managing the registration, segmentation, calibration, and matrix extraction processes.

    Attributes:
        weights_root (str): The root directory for storing CNN weight files.
        param_chip (StereoChip): An instance of StereoChip for handling chip mask information.
        config (Config): An instance of Config for managing configuration settings.
        _files (Dict[int, ProcFile]): A dictionary to store processed files.
        _ipr (ImageProcessRecord): An instance of ImageProcessRecord for managing image processing records.
        _channel_images (Dict[str, Union[IFChannel, ImageChannel]]): A dictionary to store channel images.
        _output_path (str): The output directory path.
        p_naming (DumpPipelineFileNaming): An instance of DumpPipelineFileNaming for managing file naming conventions.
        matrix_file: The matrix file to be processed.
    """

    def __init__(self, config_file: str, chip_mask_file: str, weights_root: str):
        """
        Initialize the StereoPipeline object with the given configuration file, chip mask file, and weights root.

        Args:
            config_file (str): Path to the configuration file.
            chip_mask_file (str): Path to the chip mask file.
            weights_root (str): Root directory for weights.
        """
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
        """
        Dumps the internal representation (ipr) to a specified output path.

        Args:
            output_path (str): The path where the ipr will be written.

        Returns:
            None
        """
        ipr.write(file_path=output_path, ipr=self._ipr, extra_images=self._channel_images)
        clog.info('Dump ipr to {}'.format(output_path))

    @process_decorator('GiB')
    def _dump_rpi(self, rpi_path: str):
        """
        Dumps the registration pipeline (rpi) data to a specified path.

        Args:
            rpi_path (str): The path where the rpi data will be saved.

        This method iterates through the files in the pipeline, checks for the existence of various image and mask files,
        and collects the paths of these files into a dictionary. It then writes this dictionary to an HDF5 file using the
        `rpi.write` method. Additionally, it logs a message indicating the path where the rpi data was saved.
        """
        data = {}
        for idx, f in self._files.items():
            g_name = f.get_group_name(sn=self.param_chip.chip_name)
            n = naming.DumpImageFileNaming(
                sn=self.param_chip.chip_name, stain_type=g_name, save_dir=self._output_path)
                
            if f.is_image:
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
            else:
                if g_name == 'Transcriptomics' and not f.is_image and f.cell_segmentation:
                    data[g_name] = {} 
                    if os.path.exists(n.cell_mask):
                        data[g_name]['CellMask'] = n.cell_mask
        data['final'] = {}
        data['final']['CellMask'] = self.p_naming.final_nuclear_mask
        data['final']['TissueMask'] = self.p_naming.final_tissue_mask
        data['final']['CellMaskCorrect'] = self.p_naming.final_cell_mask
        rpi.write(h5_path=rpi_path, extra_images=data)
        clog.info('Dump rpi to {}'.format(rpi_path))

    def _weights_check(self, ):
        """
        Check and download the weights for cell and tissue segmentation.

        This method iterates through the files in the `_files` dictionary and checks if each file has cell or tissue
        segmentation enabled. If so, it retrieves the corresponding weights path from the configuration and appends the
        base name of the weights path to the `weights` list. After collecting all unique weights, it attempts to download
        them using the `WeightDownloader` class. If the download fails, a warning is logged.

        Returns:
            int: The status code of the weight download operation. 0 indicates success, non-zero indicates failure.
        """
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
        """
        Check the data files for consistency and availability.

        This method checks if there are any data files to be analyzed. If no data
        is found, it logs a warning and returns an error code. If data is found,
        it checks if each file exists and if it is an image. If any file is missing
        or not an image, it logs a warning and exits the program. If all files are
        valid images, it checks if all images have the same size. If they do, it
        logs the image information and returns 0. If they don't, it logs a warning
        and returns an error code. If no image data is found, it logs a message
        and returns an error code.

        Returns:
            int: An error code indicating the result of the data check.
        """
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
                    sys.exit(ErrorCode.missFile.value)  # missing file, abnormal exit
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
        """
        Run tissue and cell segmentation on the given image file.

        Args:
            f (ProcFile): The file object containing segmentation settings.
            im_path (Path): The path to the image file.
            ts_raw_save_path (Path): The path to save the raw tissue segmentation mask.
            cs_raw_save_path (Path): The path to save the raw cell segmentation mask.
            ts_save_path (str): The path to save the final tissue segmentation mask.
            cs_save_path (str): The path to save the final cell segmentation mask.
            cur_c_image (object, optional): The current channel image or IF channel image, if available.

        Returns:
            None
        """
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
                method=self.config.tissue_segmentation.best_tissue_mask_method,
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
        """
        Process each single image in the pipeline.

        This method iterates over each image file, performs necessary transformations,
        and applies segmentation. It handles both cases where IPR data is available and
        where it is not.
        """
        # Iterate over each file in the dataset
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
                # Copy the stitched image
                shutil.copy2(f.file_path, cur_f_name.stitch_image)
                if self._channel_images is not None and self._ipr is not None:
                    # IPR data is provided
                    if f.registration.reuse == -1:
                        # TODO: handle two versions of IPR
                        qc_ = self._channel_images[g_name].QCInfo
                        if hasattr(qc_, 'QcPassFlag'):
                            qc_flag = getattr(qc_, 'QcPassFlag')
                        else:
                            qc_flag = getattr(qc_, 'QCPassFlag')
                        if qc_flag != 1:  # Cannot register under current conditions
                            clog.warning('Image QC not pass, cannot deal this pipeline')
                            sys.exit(ErrorCode.qcFail.value)
                    # Perform transform operations: Transform > Segmentation > Mask merge & expand
                    cur_c_image = self._channel_images[g_name]
                    # Transform input and output
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
                    # IPR data is not provided, perform segmentation directly on the input image
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
                print('Processing matrix file: {}'.format(f.file_path))
                print('Matrix file tech type: {}'.format(f.tech.name))
                print('Tissue segmentation enabled: {}'.format(f.tissue_segmentation))
                print('Cell segmentation enabled: {}'.format(f.cell_segmentation))
                
                if f.tissue_segmentation or f.cell_segmentation:
                    cur_m_naming = naming.DumpMatrixFileNaming(
                        sn=self.param_chip.chip_name,
                        m_type=f.tech.name,
                        save_dir=self._output_path,
                    )
                    print('Matrix file naming configuration:')
                    print('- Heatmap path: {}'.format(cur_m_naming.heatmap))
                    print('- Tissue mask path: {}'.format(cur_m_naming.tissue_mask))
                    print('- Cell mask path: {}'.format(cur_m_naming.cell_mask))
                    
                    print('Starting matrix extraction with extract4stitched...')
                    cm = extract4stitched(
                        image_file=f,
                        param_chip=self.param_chip,
                        m_naming=cur_m_naming,
                        config=self.config,
                        detect_feature=False
                    )
                    print('Matrix extraction completed')

                    if f.tissue_segmentation:
                        print('Starting tissue segmentation...')
                        run_tissue_seg(
                            image_file=f,
                            image_path=cur_m_naming.heatmap,
                            save_path=cur_m_naming.tissue_mask,
                            chip_info=self.param_chip,
                            config=self.config,
                        )
                        print('Tissue segmentation completed')

                    if f.cell_segmentation:
                        print('Starting cell segmentation...')
                        run_cell_seg(
                            image_file=f,
                            image_path=cur_m_naming.heatmap,
                            save_path=cur_m_naming.cell_mask,
                            config=self.config,
                        )
                        print('Cell segmentation completed')

    def run_mul_image(self):
        """
        Process multiple images for registration.

        This method handles the registration of multiple images. It iterates over
        the files in self._files and processes only the images. If the image is
        registered, it performs the registration process. If not, it transforms
        the image to a registered format.

        :return: None
        """
        # involve coorperation of multiple images
        # since this is registration, single-image processing is considered complete by default
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
        """
        This method processes and merges cell masks for each molecular classification file.
        It extracts the cell mask from the file, determines the naming convention, and merges
        the masks if necessary. Finally, it corrects the cell mask using the fast correction
        algorithm and saves the results.

        :return: None
        """
        for idx, m in self.molecular_classify_files.items():
            core_mask = [] #list for nuclei masks
            interior_mask = [] #list for interior masks
            cell_mask = [] #list for boundary masks
            matrix_mask = [] #list for matrix masks
            clog.info('======>  Extract[{}], {}'.format(idx, m))
            distance =  m.correct_r
            final_nuclear_path = self.p_naming.final_nuclear_mask 
            final_t_mask_path = self.p_naming.final_tissue_mask
            final_cell_mask_path = self.p_naming.final_cell_mask 
            print(final_cell_mask_path)
            core_mask = m.cell_mask["nuclei"]
            interior_mask = m.cell_mask["interior"]
            cell_mask = m.cell_mask["boundary"]
            matrix_mask = m.cell_mask["matrix"]


            # integrate nuclei, interior, cell seperatly 

            if len(cell_mask) != 0: #cell mask exist
                if len(cell_mask) == 1:
                    im_naming = naming.DumpImageFileNaming(
                    sn=self.param_chip.chip_name,
                    stain_type=self._files[cell_mask[0]].get_group_name(sn=self.param_chip.chip_name),
                    save_dir=self._output_path
                )
                    print(im_naming.cell_mask)
                    merged_cell_mask = cbimread(im_naming.cell_mask, only_np=True)
                else:
                    print("multiple cell masks exist")
                    #TODO: merge multiple cell masks, return final_cell_mask = merged cell masks
            else: #no cell mask
                merged_cell_mask = []
            
            if len(interior_mask) != 0: #interior mask exist
                if len(interior_mask) == 1:
                    im_naming = naming.DumpImageFileNaming(
                    sn=self.param_chip.chip_name,
                    stain_type=self._files[interior_mask[0]].get_group_name(sn=self.param_chip.chip_name),
                    save_dir=self._output_path
                )
                    print(im_naming.cell_mask)
                    merged_interior_mask = cbimread(im_naming.cell_mask, only_np=True)
                else:
                    print("multiple interior masks exist")
                    #TODO: merge multiple cell masks, return final_cell_mask = merged cell masks
            else: #no interior mask
                merged_interior_mask = []
            
            if len(core_mask) != 0: #core mask exist
                if len(core_mask) == 1:
                    im_naming = naming.DumpImageFileNaming(
                    sn=self.param_chip.chip_name,
                    stain_type=self._files[core_mask[0]].get_group_name(sn=self.param_chip.chip_name),
                    save_dir=self._output_path
                )
                    final_nuclear_path = im_naming.cell_mask
                    print(im_naming.cell_mask)
                    merged_core_mask = cbimread(im_naming.cell_mask, only_np=True)
                else:
                    print("multiple core masks exist")
                    #TODO: merge multiple cell masks, return final_cell_mask = merged cell masks
            else: #no core mask
                merged_core_mask = []


            #  --------------------nuclei expand--------------------
            if len(interior_mask) == 0 and len(cell_mask) == 0:  
                #merged_cell_mask_path = self._output_path + "core_extend_mask.tif"
                to_fast = final_nuclear_path
                if not os.path.exists(final_cell_mask_path) and os.path.exists(to_fast):
                    fast_mask = run_fast_correct(
                        mask_path=to_fast,
                        distance = distance,
                        n_jobs=self.config.cell_correct.process
                    )
                    cbimwrite(final_cell_mask_path, fast_mask)
            # --------------------nuclei cell merge----------------------
            elif len(interior_mask) == 0 and len(cell_mask) != 0:
                from cellbin2.contrib.mask_manager import merge_cell_mask
                from cellbin2.contrib.multimodal_cell_merge import interior_cell_merge
                save_path = os.path.join(self._output_path, "multimodal_mid_file")
                os.makedirs(save_path, exist_ok=True)
                #merged_mask = merge_cell_mask(merged_cell_mask, merged_core_mask)
                output_nuclei = merge_cell_mask(merged_cell_mask, merged_core_mask)
                output_nuclei_path = os.path.join(save_path, f"output_nuclei_mask.tif")
                cbimwrite(output_nuclei_path, output_nuclei)
                #merged_cell_mask_path = self._output_path + "/core_cell_merged_mask.tif"
                # expand nuclei
                fast_mask = run_fast_correct(
                    mask_path=output_nuclei_path,
                    distance = distance,
                    n_jobs=self.config.cell_correct.process
                )
                expand_nuclei_path = os.path.join(save_path, "expand_nuclei.tif")
                cbimwrite(expand_nuclei_path, fast_mask)
                # merge expanded nuclei with cell
               
                expand_nuclei = cbimread(expand_nuclei_path, only_np=True)
                final_mask = interior_cell_merge(merged_cell_mask, expand_nuclei, overlap_threshold=0.9, save_path="")
                cbimwrite(final_cell_mask_path, final_mask)
            # --------------------multimodal merge--------------------
            elif len(interior_mask) != 0 and len(cell_mask) != 0:
                from cellbin2.contrib.multimodal_cell_merge import multimodal_merge
                from cellbin2.contrib.multimodal_cell_merge import interior_cell_merge
                save_path = os.path.join(self._output_path, "multimodal_mid_file")
                os.makedirs(save_path, exist_ok=True)
                merged_mask = multimodal_merge(merged_core_mask, merged_cell_mask, merged_interior_mask, overlap_threshold=0.5, save_path = save_path)
                # expand nuclei
                output_nuclei = os.path.join(save_path, "output_nuclei_mask.tif")
                fast_mask = run_fast_correct(
                    mask_path=output_nuclei,
                    distance = distance,
                    n_jobs=self.config.cell_correct.process
                )
                expand_nuclei_path = os.path.join(save_path, "expand_nuclei.tif")
                cbimwrite(expand_nuclei_path, fast_mask)
                
                # merge expanded nuclei with cell
                expand_nuclei_path = os.path.join(save_path, "expand_nuclei.tif")
                cell_mask_add_interior_path = os.path.join(save_path, "cell_mask_add_interior.tif")
                cell_mask_add_interior = cbimread(cell_mask_add_interior_path, only_np=True)
                
                expand_nuclei = cbimread(expand_nuclei_path, only_np=True)
                final_mask = interior_cell_merge(cell_mask_add_interior, expand_nuclei, overlap_threshold=0.9, save_path="")
                #final_mask = cbimread(os.path.join(save_path2, "cell_mask_add_interior.tif"), only_np=True)
                cbimwrite(final_cell_mask_path, final_mask)
            # --------------------matrix fix--------------------
            if len(matrix_mask) != 0: #interior mask exist
                if len(matrix_mask) == 1:
                    im_naming = naming.DumpImageFileNaming(
                    sn=self.param_chip.chip_name,
                    stain_type=self._files[matrix_mask[0]].get_group_name(sn=self.param_chip.chip_name),
                    save_dir=self._output_path
                )
                    matrix_mask_path = im_naming.cell_mask
                else:
                    print("multiple interior masks exist")
                    #TODO: merge multiple cell masks, return final_cell_mask = merged cell masks
            else: #no interior mask
                merged_interior_mask = []
            cmf=CellMaskFixer(source_imge=str(matrix_mask_path),refer_image=str(final_nuclear_path),sn=self.param_chip.chip_name)
            cmf.fix_notsinglecell2mask(out_path=self._output_path, save=True)




    def run(self, chip_no: str, input_image: str,
            stain_type: str, param_file: str,
            output_path: str, ipr_path: str,
            matrix_path: str, kit: str, debug: bool, research_mode: bool):
        """
        Run the main pipeline for processing the images.

        Args:
            chip_no (str): The serial number of the chip.
            input_image (str): The path to the input image file.
            stain_type (str): The type of stain used in the image.
            param_file (str): The path to the parameter file.
            output_path (str): The directory where the output files will be saved.
            ipr_path (str): The path to the image process record file.
            matrix_path (str): The path to the matrix file.
            kit (str): The type of kit used (e.g., Transcriptomics, Protein).
            debug (bool): Flag to enable debug mode.
            research_mode (bool): Flag to enable research mode.

        Returns:
            None
        """

        self._output_path = output_path
        self.debug = debug
        self.research_mode = research_mode
        # Load chip information
        self.param_chip.parse_info(chip_no)
        self.p_naming = naming.DumpPipelineFileNaming(chip_no=chip_no, save_dir=self._output_path)

        # Load data
        pp = read_param_file(
            file_path=param_file,
            cfg=self.config,
            out_path=self.p_naming.input_json
        )

        self._files = pp.get_image_files(do_image_qc=False, do_scheduler=True, cheek_exists=False)
        self.molecular_classify_files = pp.get_molecular_classify()
        pp.print_files_info(self._files, mode='Scheduler')

        # Exit if data validation fails
        flag1 = self._data_check()
        if flag1 not in [0, 2]:
            return 1
        if flag1 == 0:
            if os.path.exists(ipr_path):
                self._ipr, self._channel_images = ipr.read(ipr_path)
            else:
                clog.info(f"No existing ipr founded, assumes qc has not been done before")
                self._ipr, self._channel_images = None, None

        # Load model
        flag2 = self._weights_check()
        if flag2 != 0:
            sys.exit(1)

        self.run_single_image()  # Process a single image (transform -> tissue seg -> cellseg)
        self.run_mul_image()  # Register images

        if flag1 == 0 and self._channel_images is not None and self._ipr is not None:
            self._dump_ipr(self.p_naming.ipr)

        self.run_merge_masks()  # Merge multiple masks if needed

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
        """
        Deletes files from the output directory, excluding those specified to be kept.

        Args:
            f_to_keep (list): List of file properties that should not be deleted.

        Returns:
            None
        """
        # List to store all file paths
        all_ = []
        # List to store file paths that should be kept
        k_ = []
        # List to store file paths that should be removed
        remove_ = []

        # Iterate over files in self._files
        for idx, f in self._files.items():
            # Get group name for the file
            g_name = f.get_group_name(sn=self.param_chip.chip_name)

            # Check if the file is a matrix
            if f.is_matrix:
                # Create DumpMatrixFileNaming object for matrix files
                f_name = naming.DumpMatrixFileNaming(
                    sn=self.param_chip.chip_name,
                    m_type=f.tech.name,
                    save_dir=self._output_path,
                )
            else:
                # Create DumpImageFileNaming object for image files
                f_name = naming.DumpImageFileNaming(
                    sn=self.param_chip.chip_name,
                    stain_type=g_name,
                    save_dir=self._output_path
                )

            # Iterate over attributes of the file naming object
            for p in dir(f_name):
                att = getattr(f_name, p)
                pt = f_name.__class__.__dict__.get(p)

                # Check if the attribute is a property and the file exists
                if isinstance(pt, property) and att.exists():
                    all_.append(att)

                    # Check if the file should be removed or kept
                    if pt not in f_to_keep:
                        remove_.append(att)
                    else:
                        k_.append(att)

        # Iterate over attributes of self.p_naming
        for p_p in dir(self.p_naming):
            p_att = getattr(self.p_naming, p_p)
            p_pt = self.p_naming.__class__.__dict__.get(p_p)

            # Check if the attribute is a property and the file exists
            if isinstance(p_pt, property) and p_att.exists():
                all_.append(p_att)

                # Check if the file should be removed or kept
                if p_pt not in f_to_keep:
                    remove_.append(p_att)
                else:
                    k_.append(p_att)

        # Iterate over files in the output directory
        for f in os.listdir(self._output_path):
            path = os.path.join(self._output_path, f)

            # Remove the file if it is in the remove_ list
            if Path(path) in remove_:
                os.remove(path)


def scheduler_pipeline(weights_root: str, chip_no: str, input_image: str, stain_type: str,
                       param_file: str, output_path: str, matrix_path: str, ipr_path: str,
                       kit: str, debug: bool = False, research_mode=False):
    """
    This function serves as the main pipeline for scheduling the image processing tasks.

    Args:
        weights_root (str): Local directory path for storing CNN weight files.
        chip_no (str): Sample chip number.
        input_image (str): Local path of the stained image.
        stain_type (str): Staining type corresponding to the stained image.
        param_file (str): Local path of the parameter file.
        output_path (str): Local directory path for storing output files.
        matrix_path (str): Local directory path for storing the expression matrix.
        ipr_path (str): Local directory path for storing the image processing record file.
        kit (str): Kit type (e.g., Transcriptomics, Protein).
        debug (bool, optional): Debug mode flag. Defaults to False.
        research_mode (bool, optional): Research mode flag. Defaults to False.

    Returns:
        int: Status code.
    """
    curr_path = os.path.dirname(os.path.realpath(__file__))
    config_file = os.path.join(curr_path, r'../config/cellbin.yaml')
    chip_mask_file = os.path.join(curr_path, r'../config/chip_mask.json')

    # Initialize the Scheduler with configuration and weights
    iqc = Scheduler(config_file=config_file, chip_mask_file=chip_mask_file, weights_root=weights_root)
    # Run the main pipeline with the provided parameters
    iqc.run(chip_no=chip_no,
            input_image=input_image,
            stain_type=stain_type,
            param_file=param_file,
            output_path=output_path,
            ipr_path=ipr_path,
            matrix_path=matrix_path, kit=kit, debug=debug, research_mode=research_mode)


def main(args, para):
    """
    The main function that initializes and runs the scheduler pipeline.

    Args:
        args (argparse.Namespace): Parsed command - line arguments containing the configuration and input data.
        para (dict): Additional parameters required for the pipeline.

    Returns:
        None
    """
    # Call the scheduler_pipeline function with the parsed arguments
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
