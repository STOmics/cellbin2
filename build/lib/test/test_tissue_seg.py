#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author  : hedongdong1
# @Time    : 2024/11/28 15:34
# @File    : test_tissue_seg.py
# @annotation    :
import os.path

import pytest
import tifffile

from cellbin2.contrib.tissue_segmentor import segment4tissue, TissueSegInputInfo, TissueSegOutputInfo, TissueSegParam
from cellbin2.utils.common import TechType


TEST_DATA = [
    (
        # ssDNA
        r"/path/to/image",  # input image folder path
        r"/path/to/mask",  # if the input is a folder, the output must also be a folder
        r"/path/to/tissueseg_bcdu_SDI_230523_tf.onnx",  # onnx file path
        "ssdna",  # stain type
        [3, 3],  # chip size,height and width, if the image path is a folder, this parameter will be used for all the images
        "onnx",  # onnx mode or tf mode, currently only the onnx mode is supported
        "0"  # GPU num, -1 represents the CPU
    ),
    (
        # DAPI
        r"/path/to/image",  # input image folder path
        r"/path/to/mask",  # if the input is a folder, the output must also be a folder
        r"/path/to/tissueseg_bcdu_SDI_230523_tf.onnx",  # onnx file path
        "dapi",  # stain type
        [],  # set empty list,if do not know chip size
        "onnx",  # onnx mode or tf mode, only the onnx mode is supported currently
        "0"  # GPU num, -1 represents the CPU
    ),
    (
        # HE
        r"/path/to/image/he.tif",  # input image path
        r"/path/to/mask/he.tif",  # output mask  path
        r"/path/to/tissueseg_bcdu_H_20241018_tf.onnx",  # onnx file path
        "he",  # stain type
        [3, 3],  # chip size,height and width, if the image path is a folder, this parameter will be used for all the images
        "onnx",  # onnx mode or tf mode, currently only the onnx mode is supported
        "0"  # GPU num, -1 represents the CPU
    ),
    (
        # IF do not need model
        r"/path/to/image/if.tif",  # input image path
        r"/path/to/image/if.tif",  # output mask path
        r"",
        "if",  # stain type
        [2, 2],  # chip size,height and width, if the image path is a folder, this parameter will be used for all the images
        "onnx",  # onnx mode or tf mode, currently only the onnx mode is supported
        "0"  # GPU num, -1 represents the CPU
    ),
    (
        r"/path/to/image/rna.tif",  # input image path
        r"/path/to/mask/rna.tif",  # output mask path
        r"/path/to/tissueseg_bcdu_rna_220909_tf.onnx",
        "transcriptomics",  # stain type
        [3, 3],  # chip size,height and width, if the image path is a folder, this parameter will be used for all the images
        "onnx",  # onnx mode or tf mode, currently only the onnx mode is supported
        "0"  # GPU num, -1 represents the CPU
    )
]

USR_STYPE_TO_INNER = {
        'ssdna': TechType.ssDNA,
        'dapi': TechType.DAPI,
        "he": TechType.HE,
        "transcriptomics": TechType.Transcriptomics,
        'protein': TechType.Protein,
        'if': TechType.IF
    }

class TestTissueSeg:
    @pytest.mark.parametrize("input_dir, output_dir, model_dir, stain_type, chip_size, model_mode, gpu_num", TEST_DATA)
    def test_tissue_seg(self,
                        input_dir: str,
                        output_dir: str,
                        model_dir: str,
                        stain_type: str,
                        chip_size: list,
                        model_mode: str,
                        gpu_num: str
                        ):
        cfg = TissueSegParam()
        stain_type = USR_STYPE_TO_INNER[stain_type]
        if stain_type != TechType.IF:
            setattr(cfg, f"{stain_type.name}_weights_path", model_dir)
            setattr(cfg, "GPU", gpu_num)
        print(f"info===> stain type: {stain_type}, set {stain_type} model path:{model_dir}")
        if os.path.isdir(input_dir):
            assert os.path.isdir(output_dir), 'the input path is a folder, so the output path should also be a folder'
            for tmp in os.listdir(input_dir):
                input_path = os.path.join(input_dir, tmp)
                output_path = os.path.join(output_dir, tmp)
                if isinstance(chip_size, list) and len(chip_size) == 2:
                    input_data = TissueSegInputInfo(
                        weight_path_cfg=cfg,
                        input_path=input_path,
                        stain_type=stain_type,
                        chip_size=chip_size
                    )
                else:
                    input_data = TissueSegInputInfo(
                        weight_path_cfg=cfg,
                        input_path=input_path,
                        stain_type=stain_type
                    )

                seg_result = segment4tissue(input_data=input_data)
                seg_mask = seg_result.tissue_mask
                print(seg_mask.shape)
                if seg_result.threshold_list:
                    print(*seg_result.threshold_list)
                seg_mask[seg_mask > 0] = 255
                tifffile.imwrite(output_path, seg_mask, compression='zlib')
        else:
            if isinstance(chip_size, list) and len(chip_size) == 2:
                input_data = TissueSegInputInfo(
                    weight_path_cfg=cfg,
                    input_path=input_dir,
                    stain_type=stain_type,
                    chip_size=chip_size
                )
            else:
                input_data = TissueSegInputInfo(
                    weight_path_cfg=cfg,
                    input_path=input_dir,
                    stain_type=stain_type
                )

            seg_result = segment4tissue(input_data=input_data)
            seg_mask = seg_result.tissue_mask
            print(seg_mask.shape)
            if seg_result.threshold_list:
                print(*seg_result.threshold_list)
            seg_mask[seg_mask > 0] = 255
            tifffile.imwrite(output_dir, seg_mask, compression='zlib')
