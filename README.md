<div align="center">
  <img src="docs/images/cellbin.png"><br/>
  <h1 align="center">
    cellbin2: A framework for generating single-cell gene expression data
  </h1>
</div>

## Introduction
CellBin is an image processing pipeline designed to delineate cell boundaries for spatial analysis. It consists of several image analysis steps. Given the image and gene expression data as input, CellBin performs image registration, tissue segmentation, nuclei segmentation, and molecular labeling (i.e., cell border expanding), ultimately defining the molecular boundaries of individual cells. It incorporates a suite of self-developed algorithms, including deep-learning models, for each of the analysis task. The processed data is then mapped onto the chip to extract molecular information, resulting in an accurate single-cell expression matrix. (Cover image) For more information on CellBin, please refer to the following link.

**Cellbin2** is an upgraded version of the original [CellBin](https://github.com/STOmics/CellBin) platform with two key enhancements:
1. **Expanded Algorithm Library**: Incorporates additional image processing algorithms to serve broader application scenarios like single-cell RNA-seq, Plant cellbin.
2. **Configurable Architecture**: Refactored codebase allows users to customize analysis pipelines through [JSON](cellbin2/config/demos/sample.json) and [YAML](cellbin2/config/cellbin.yaml) configuration files.

[CellBin introduction](docs/md/CellBin_1.0/CellBin解决方案技术说明.md) (Chinese) 

***Tweets*** <br>
[Stereo-seq CellBin introduction](https://mp.weixin.qq.com/s/2-lE5OjPpjitLK_4Z0QI3Q) (Chinese)  <br>
[Stereo-seq CellBin application intro](https://mp.weixin.qq.com/s/PT3kPvsmrB3oQleEIMPkjQ)  (Chinese)  <br>
[Stereo-seq CellBin cell segmentation database introduction](https://mp.weixin.qq.com/s/OYJhAH6Bq1X1CQIYwugxkw) (Chinese)  <br>
[CellBin: The Core Image Processing Pipeline in SAW for Generating Single-cell Gene Expression Data for Stereo-seq](https://en.stomics.tech/news/stomics-blog/1017.html) (English)  <br>
[A Practical Guide to SAW Output Files for Stereo-seq](https://en.stomics.tech/news/stomics-blog/1108.html) (English)  <br>

***Paper related*** <br>
[CellBin: a highly accurate single-cell gene expression processing pipeline for high-resolution spatial transcriptomics](https://www.biorxiv.org/content/10.1101/2023.02.28.530414v5) [(GitHub Link)](https://github.com/STOmics) <br>
[Generating single-cell gene expression profiles for high-resolution spatial transcriptomics based on cell boundary images](https://gigabytejournal.com/articles/110) [(GitHub Link)](https://github.com/STOmics/STCellbin) <br>
[CellBinDB: A Large-Scale Multimodal Annotated Dataset for Cell Segmentation with Benchmarking of Universal Models](https://www.biorxiv.org/content/10.1101/2024.11.20.619750v2) [(GitHub Link)](https://github.com/STOmics/cs-benchmark) <br>

***Video tutorial*** <br>
[Cell segmentation tool selection and application](https://www.bilibili.com/video/BV1Ct421H7ST/?spm_id_from=333.337.search-card.all.click) (Chinese) <br>
[One-stop solution for spatial single-cell data acquisition](https://www.bilibili.com/video/BV1Me4y1T77T/?spm_id_from=333.337.search-card.all.click) (Chinese) <br>
[Single-cell processing framework for high resolution spatial omics](https://www.bilibili.com/video/BV1M14y1q7YR/?spm_id_from=333.788.recommend_more_video.12) (Chinese) 

## Installation and Quick Start
Linux
```shell
# Clone the main repository
git clone https://github.com/STOmics/cellbin2
# git clone -b dev https://github.com/STOmics/cellbin2

# Create and activate a Conda environment
conda create --name cellbin2 python=3.8
conda activate cellbin2

# Install package dependencies
cd cellbin2
pip install .[cp,rs]

# For development mode (optional):
# pip install -e .[cp,rs]      # Editable install with basic extras
# pip install -e .[cp,rs,rp]   # Editable install including report module

# Execute the demo (takes ~30-40 minutes on GPU hardware)
python demo.py
```

**GPU Troubleshooting:**  
If the demo runs on CPU instead of GPU, see our [troubleshooting guide]() to verify your GPU setup.

**Output Verification:**  
After completion, validate the output integrity by comparing your results with the [Outputs](#outputs). 


## Tutorials
### Core Workflow
The `cellbin_pipeline.py` script serves as the main entry point for CellBin2 analysis. It supports using JSON file for customized configuration

📘 **Configuration Guide**:<br>
See [JSON Configuration Documentation](docs/v2/JsonConfigurationDocumention.md) for full parameter specifications.

### Command Template

```shell
# configuration (requires complete parameters in JSON)
CUDA_VISIBLE_DEVICES=0 python cellbin2/cellbin_pipeline.py -c <SN> -p <config.json> -o <output_dir> 
```

### Parameters

| Parameter    | Description                                                                                                   | Examples                                                  |
| :----------- |:--------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------|
| `-c`         | Serial number of chip                                                                                         | `A12345C6`                                                |
| `-p`         | Path to custom configuration file<br/> [JSON Configuration Documentation](docs/v2/JsonConfigurationDocumention.md) | [`config/custom.json`](cellbin2/config/demos/sample.json) |
| `-o`         | Output directory                                                                                              | `results/A12345C6`                                        |


### Common Use Cases

#### Case 1: Stereo-seq T FF <br>
ssDNA
```shell
CUDA_VISIBLE_DEVICES=0 python cellbin2/cellbin_pipeline.py \
-c SN \
-p tif_gef.json\
-o output/SN \
```

#### Case 2:Stereo-CITE <br>
DAPI + IF + trans gef
```shell
CUDA_VISIBLE_DEVICES=0 python cellbin2/cellbin_pipeline.py \
-c SN \
-i SN.tif \
-s DAPI \
-mi IF=SN_IF.tif \
-m SN.raw.gef \
-o test/SN \
-k "Stereo-CITE T FF V1.1 R"
```

#### Case 3:Stereo-CITE
DAPI + protein gef
```shell
CUDA_VISIBLE_DEVICES=0 python cellbin2/cellbin_pipeline.py \
-c SN \
-i SN_fov_stitched.tif \
-s DAPI \
-pr IF=SN.protein.tissue.gef \
-o /test/SN \
-k "Stereo-CITE T FF V1.1 R"
```

#### Case 4:Stereo-CITE
DAPI + IF + trans gef + protein gef
```shell
CUDA_VISIBLE_DEVICES=0 python cellbin2/cellbin_pipeline.py \
-c SN \ # chip number
-i SN_DAPI_fov_stitched.tif \  # ssDNA, DAPI, HE data path
-mi IF=SN_IF.tif \
-s DAPI \  # stain type (ssDNA, DAPI, HE)
-m SN.raw.gef \  # Transcriptomics gef path
-pr SN.protein.raw.gef \  # protein gef path
-o test/SN \ # output dir
-k "Stereo-CITE T FF V1.1 R"
```

#### Case 5:Single RNA <br>
trans gef
```shell
CUDA_VISIBLE_DEVICES=0 python cellbin2/cellbin_pipeline.py \
-c SN \ # chip number
-p only_matrix.json \ # Personalized Json File
-o test/SN \ # output dir
```
please modify [only_matrix.json](cellbin2/config/demos/only_matrix.json)<br>


#### Case 6: Plant cellbin<br>
ssDNA + FB + trans gef
```shell
CUDA_VISIBLE_DEVICES=0 python cellbin2/cellbin_pipeline.py \
-c SN \ # chip number
-p Plant.json \ # Personalized Json File
-o test/SN \ # output dir
```
please modify [Plant.json](cellbin2/config/demos/Plant.json)<br>


#### Case 7: Multi-stain cellbin <br>
ssDNA + HE + trans gef
 ```shell
 CUDA_VISIBLE_DEVICES=0 python cellbin2/cellbin_pipeline.py \
 -c SN \ # chip number
 -i SN_ssDNA_fov_stitched.tif \  # ssDNA,DAPI data path
 -mi HE=SN_HE_fov_stitched.tif \ # HE data path. This image has been registered with ssDNA(DAPI) image
 -s ssDNA \  # stain type (ssDNA, DAPI)
 -m SN.raw.gef \  # Transcriptomics gef path
 -o test/SN \ # output dir
 -k "Stereo-CITE T FF V1.1 R"
 ```

> more examples, please visit [example.md](docs/v2/example.md)

## ErrorCode
refer to [error.md](docs/v2/error.md)

<a id="outputs"></a>
## Outputs
| File Name | Description |
| ---- | ---- |
| SN_cell_mask.tif | Final cell mask |
| SN_mask.tif | Final nuclear mask |
| SN_tissue_mask.tif | Final tissue mask |
| SN_params.json | CellBin 2.0 input params |
| SN.ipr | Image processing record |
| metrics.json | CellBin 2.0 Metrics |
| CellBin_0.0.1_report.html | CellBin 2.0 report |
| SN.rpi | Recorded image processing (for visualization) |
| SN.stereo | A JSON-formatted manifest file that records the visualization files in the result |
| SN.tar.gz | tar.gz file |
| SN_DAPI_mask.tif | Cell mask on registered image |
| SN_DAPI_regist.tif | Registered image |
| SN_DAPI_tissue_cut.tif | Tissue mask on registered image |
| SN_IF_mask.tif | Cell mask on registered image |
| SN_IF_regist.tif | Registered image |
| SN_IF_tissue_cut.tif | Tissue mask on registered image |
| SN_Transcriptomics_matrix_template.txt | Track template on gene matrix |

- **Image files (`*.tif`):** Inspect using [ImageJ](https://imagej.net/ij/)
- **Gene expression file** (generated only when matrix_extract module is enabled): 
  Visualize with [StereoMap v4](https://www.stomics.tech/service/stereoMap_4_1/docs/kuai-su-kai-shi.html#ke-shi-hua-shu-ru-wen-jian).   


## Other content


## Reference
https://github.com/STOmics/CellBin <br>
https://github.com/MouseLand/cellpose <br>
https://github.com/matejak/imreg_dft <br>
https://github.com/rezazad68/BCDU-Net <br>
https://github.com/libvips/pyvips <br>
https://github.com/vanvalenlab/deepcell-tf <br>
https://github.com/ultralytics/ultralytics <br>
