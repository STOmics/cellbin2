#stereomap4.0 version manual scheme

- [stereomap4.0Version manual plan](#stereomap40 version manual solution)
    - [Pipelineintroduce](#pipeline introduction)
    - [sawSoftware Manual（Download link included）](#saw software manual comes with download link)
- [StereoMap imageQC](#stereomap-image qc)
- [SAW count](#saw-count)
- [StereoMap Visual viewing manual image processing](#stereomap-visual viewing manual image processing)
  - [Step1：upload image Import pictures](#step1upload-image-image-import picture)
  - [Step2：image registration Image registration](#step2image-registration-image registration)
  - [Step3: tissue segmentation Organizational segmentation](#step3-tissue-segmentation-organization segmentation)
  - [Step4: cell segmentation Cell segmentation](#step4-cell-segmentation-cell segmentation)
  - [Step5: export Export](#step5-export-export)
- [SAW realign](#saw-realign)
      - [Willtar.gzRetrieveSAWprocess，SAW convert Export images](#Take targz back to the saw process saw-convert-export image)
    - [Task delivery](#Task Delivery)
- [StereoMap Visual viewing](#stereomap-visual viewing)

Purpose: When the CellBin automation results do not meet the requirements, manual solutions can be used to meet the requirements.

Scope of application: The data must be run through the saw v8 process, and the matrix file format is .stereo

**Tool Introduction**

**StereoMap (manual processing)**

StereoMap and ImageStudio are merged into one software StereoMap4.0, which is divided into three modules: visualization, image processing and widgets. Image processing is compatible with some functions of ImageStudio, and is modified to Step by step to reduce user learning costs and improve user satisfaction.

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/ZWGl0wN0NgwMO34Y/img/bcfdd062-2094-45dc-a6ec-2a52c591f92d.png)

| Tool Name | Version of Use | Tool Description | Download Address (Huada Network Disk) | Notes |
| --- | --- | --- | --- | --- |
| StereoMap | StereoMap-4.0.1 | Visualization, manually process images for registration, tissue segmentation, cell segmentation, image QC | [https://www.stomics.tech/products/BioinfoTools/OfflineSoftware](https://www.stomics.tech/products/BioinfoTools/OfflineSoftware) | The input must be the result of running saw v8 |

**If there is a version update, it is strongly recommended to manually uninstall the historical version and then reinstall it. It is not recommended to overwrite the installation**
    

**SAW**

The Stereo-seq Analysis Workflow (SAW) software suite is a bundled set of pipelines for locating sequencing reads to their spatial location on tissue sections, quantifying spatial feature expressions, and visually presenting spatial expression distributions. SAW combines microscope image processing to generate spatial feature expression matrix for Stereo-seq sequencing platform data. The analyst can then perform downstream analysis with the output file as a starting point

### Pipeline Introduction

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/ZWGl0wN0NgwMO34Y/img/763ed9f0-ec86-4fd2-8941-bc3c5434e6d7.png)

### saw software manual (with download link)

saw instructions manual link: [https://stereotoolss-organization.gitbook.io/saw-user-manual/](https://stereotoolss-organization.gitbook.io/saw-user-manual/)

**Operation Process**

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/2522290e-2f5f-42d7-9499-e613f2647bee.png)

# StereoMap Image QC

1. Open StereoMap and click Tools
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/29f08ebf-b44e-4c3b-b2ea-248a3813cc05.png)

2. Click start to enter the QC interface
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/26d37411-884a-4cd4-a0e9-08705dac0c27.png)

3. Drag the chip folder to be QC into the interface, click RUN after filling in it, start QC
    

\*File path must be named in English, the file name is chip number (for example A03990A1, there cannot be \_ or numbers) otherwise it may affect QC results.

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/d56a61c1-17f5-40b9-b800-d047385ac888.png)

4. The image output after QC is completed becomes a TAR file (storing the original image of the IPR and microscope), and the storage path of the output file can be changed in the settings.
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/dc5039f4-ab12-498c-904a-9f876fbbd66c.png)

# SAW count

**Before running SAW count**

[To the right] The use of parameters is slightly different depending on the STOmics product line. Before performing the analysis, please pay attention to the information about the test kit version and select the appropriate SAW version

To generate spatial feature counts of individual libraries from fresh frozen (FF) samples using automatic registration and tissue detection on microscope-stained images, run using the following parameters

    cd /saw/runs
    
    saw count \
        --id=Demo_Mouse_Brain \  ##task id
        --sn=SS200000135TL_D1 \  ##SN information of Stereo-seq chip 
        --omics=transcriptomics \  ##omics information
        --kit-version="Stereo-seq T FF V1.2" \  ##kit version
        --sequencing-type="PE100_50+100" \  ##sequencing type
        --chip-mask=/path/to/chip/mask \  ##path to the chip mask
        --organism=<organism> \  ##usually refer to species
        --tissue=<tissue> \  ##sample tissue
        --fastqs=/path/to/fastq/folders \  ##path to FASTQs
        --reference=/path/to/reference/folder \  ##path to reference index
        --image-tar=/path/to/image/tar  ##path to the compressed image
        --output=/path/to/output

After the first run, you can get the files from the demo and manual registration in StereoMap. After a series of manual processes, the image will be returned to obtain new results

    cd /saw/runs
    
    saw realign \
        --id=Adjuated_Demo_Mouse_Brain \  ##task id
        --sn=SS200000135TL_D1 \  ##SN information of Stereo-seq Chip 
        --count-data=/path/to/previous/SAW/count/task/folder \  ##output folder of previous SAW count
        #--adjusted-distance=10 \  ##default to 10 pixel
        --realigned-image-tar=/path/to/realigned/image/tar   ##realigned image .tar.gz from StereoMap

\*For more information, please see [Manual Processing Tutorial](https://stereotoolss-organization.gitbook.io/saw-user-manual-v8.0/tutorials/with-manually-processed-files)


[To the right] Take this chip as an example

```
**SN**：C02533C1

**Speices**: mouse

**Tissue**: kidney

**Chip size**: 1\*1

**Stain type**: H&E
```


**FF Analysis**

    ## The position of the double# annotation mark needs to be modified
    
    SN=C02533C1  ##Modified to the chip SN analyzed this time
    saw=/PATH/to/saw  ## If the saw software used is updated and modified in time, you only need to change the number after a in saw-v8.0.0a7 to indicate the version number of the internal test software used.
    data=/jdfssz3/ST_STOMICS/P20Z10200N0039_autoanalysis/stomics/tmpForDemo/${SN}
    image=/ldfssz1/ST_BIOINTEL/P20Z10200N0039/guolongyu/8.0.0/Image_v4/${SN}
    tar=$(find ${image} -maxdepth 1 -name \*.tar.gz | head -1)
    
    source ${saw}/bin/env
    export LD_LIBRARY_PATH=${saw}/lib/geftools/lib:$LD_LIBRARY_PATH
    
    ${saw}/bin/saw count \
        #--id=Let_me_see_see_mouse_kidney \ ##You can modify the ID by yourself, be careful not to repeat the tasks (title)
        --sn=${SN} \
        --omics=transcriptomics \
        --kit-version='Stereo-seq T FF V1.2' \（No changes to parameters are required）
        --sequencing-type='PE100_50+100' \（The current version does not require，But for the sake of compatibility with subsequent）
        --organism=mouse \  ##Fill in the species (cannot fill in)
        --tissue=kidney \  ##Fill in tissue or disease (can not fill in)
        --chip-mask=${data}/mask/${SN}.barcodeToPos.h5 \
        --fastqs=${data}/reads \ 
        --reference=/PATH/reference/mouse \  ##mouse/human/rat for selection, just modify the last folder name
        --image-tar=${tar} \
        --local-cores=48

**FFPE analysis**

    ## The position of the double# annotation mark needs to be modified
    
    SN=C02533C1  ##Modified to the chip SN analyzed this time
    saw=/PATH/to/saw  ## If the saw software used is updated and modified in time, you only need to change the number after a in saw-v8.0.0a7 to indicate the version number of the internal test software used.
    mask=/path/to/chip/mask/file/SN.mask.h5  ##Modify the mask path to FFPE
    fastq=/path/to/fastq/folder  ##Modify the FASTQs path to FFPE. If there are multiple directories, pass multiple folders.
    image=/path/to/QC/image/tar ##Modify to QC image TAR packet path
    
    source ${saw}/bin/env
    export LD_LIBRARY_PATH=${saw}/lib/geftools/lib:$LD_LIBRARY_PATH
    
    ${saw}/bin/saw count \
        --id=What_about_FFPE_mouse_kidney \  ##You can modify the ID by yourself, be careful not to repeat the tasks
        --sn=${SN} \
        --omics=transcriptomics \
        --kit-version='Stereo-seq N FFPE V1.0' \
        --sequencing-type='PE75_25+59' \
        --organism=mouse \  ##Fill in the species
        --tissue=kidney \  ##Fill in tissue or disease
        --chip-mask=${mask} \
        --fastqs=${fastq} \ 
        --reference=/PATH/reference/human \  
        --image-tar=${tar} \
        --microbiome-detect \ ##Select by yourself and can be closed
        --local-cores=48
       #--image=/path/to/TIFF
    

**Output result description**

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/ZWGl0wN0NgwMO34Y/img/1c98331f-b830-4dc6-acf3-53d0c9b67eb1.png)

![1713279982523.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/ZWGl0wN0NgwMO34Y/img/d9fef8da-657e-4152-b977-84e096e68b27.png)


Download `visualization.tar.gz` locally decompressed, use StereoMap software for visual viewing and manual image processing.


# StereoMap Visual View Manual Image Processing

Open StereoMap and select image processing

## Step1: upload image Import image

1. Select the staining type (ssDNA, DAPI, DAPI&mIF, H&E)![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/3c022a77-dd41-48a8-857c-5e2d0353f2bc.png)
    
2. Import the picture after selection. The supported data formats of the picture are: `.tar.gz``.stereo``.tif``.tiff` (TAR.GZ after QC of 4.0 gadget, visual .stereo file (must have an image)). Drag the picture directly into the corresponding area, and the information on the right will be automatically filled in.
    

[To the right] If TIFF large image data will be re-evaluated

[Right] If it is a TAR.GZ or .stereo file, it will read the result of the image QC and display it. After waiting for the reading, click NEXT

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/20830336-4c09-412e-b4ec-e6db7262a4cf.png)

## Step2: image registration

1. Add a matrix at the right morphology, the matrix must be a `.stereo` file
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/b775b5d0-40a7-4b78-9f8c-3d84ddf75831.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/708c0e17-b0cb-4bdd-b521-887d0ae1b291.png)

2. Image registration
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/24e38782-2f23-4ff4-b20e-4f655b777d7a.png)

[To the right] The registration process consists of two stages, roughly matching the direction of the image according to the morphology, and fine-tuning the position and proportions so that it completely overlaps with the spatial feature expression map. You can also select Chip track lines to display trace lines derived from the spatial feature expression matrix to aid in fine alignment

[To the right] To roughly align the image, you need to match the microscope image and feature density map in the orientation

Use the **"Flip" tool**![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/a6bdd0d8-1508-4a67-b9b6-819e6050d3fc.png) Mirror image

Use the ** control knob**![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/84a56c87-cbb2-4899-941c-ff9da08681fb.png) Rotate the image in the same direction

[To the right] Once the direction of the image and the spatial feature expression matrix match, the fine alignment step can be continued. During the fine alignment phase, you need to move the image to a position where the tissue can overlap

Use the **"Move" panel by setting the four directions: step size and panning![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/e458a984-63a2-4021-b07d-954b999c674e.png)

The size of the microscope image may be different from the spatial feature expression graph. You can use the ** Scale tool **![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/ca901cc8-d9f6-4cf5-b6bd-e685a8cfe579.png) to adjust the scale

[To the right] You can align the image by selecting "Chip Track Line" to display the reference track line template and allow the track lines to overlap directly. At this time, the chip trace line template is a representation of the matrix. If the track line becomes dark, you can manually adjust the contrast of the image![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/ef50bc5c-0e26-429e-b9e5-465c7b9f8207.png) , brightness![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/12ccdc77-2eb3-4ac2-a6ef-06dc7a863639.png) and opacity![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/e1a11f41-b2fd-4b18-ab62-91d52ab782f1.png)

3. Click NEXT after registration is completed to confirm
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/47619929-7ac2-4f05-a406-0c0d42294843.png)

## Step3: tissue segmentation

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/87d49f25-9fc3-4a2b-8bab-af9f48efb090.png)

1. Manually modify the incomplete segmentation areas
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/83628193-da40-479e-9f09-06021389118d.png) ![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/03672f8b-2485-4f3b-9c1d-3fb46b9259c1.png)

2. To select or edit an organization area, use **lasso**![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/883aea29-c74a-4629-a91d-d022832b491b.png) , **paintbrush**![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/dcd560e9-bb47-4346-8612-401c6b3916e2.png) and ** Eraser**![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/a292b2e4-f1e8-422b-b45c-09c6a30f0a46.png) tool. **Lasso** is often used to select or deselect large areas, while **Brush** and **Eraser** tools are more suitable for smaller areas, such as areas around tissue or small holes in tissue.
    

[Right]**Brush** ![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/dcd560e9-bb47-4346-8612-401c6b3916e2.png) Fill the area

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/dde3898d-2da3-45f0-82a4-949d53916401.png)

[Right]** Eraser**![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/a292b2e4-f1e8-422b-b45c-09c6a30f0a46.png) Eliminate area

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/f9bf7b38-7ce0-47e2-91ad-d1c0966478c9.png)

3. Modification is completed
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/bcdf84b7-157b-4945-9a1d-02d3802d20f7.png)

4. Click NEXT to enter step4
    

## Step4: cell segmentation

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/d9d9ad2d-3341-4987-a73e-18b1ecec247e.png)

1. Modify
    

[To the right] Use **lasso**![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/5d63bae8-4d5e-4e01-98fa-a56e58814e0c.png) , **paintbrush**![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/11db2b46-1453-47fb-9eff-83ac77784350.png) and ** Eraser**![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/f932b18e-ef35-450d-b239-37aba923890e.png) tool to select or edit the cell. **Lasso** is best for deselecting large areas (such as backgrounds), while the **Brush** and **Eraser** tools are more suitable for smaller areas, such as marking cells or detaching cell clusters.

[To the right] It is recommended to import a cell mask file in `.tif` format. You can access your file system by clicking the **Segmentation mask** drop-down list and clicking it![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/9a9363fc-8c51-4614-ad8d-32be2c66824a.png) . If the import result is not satisfactory, you can replace it by clicking ![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/d584b570-922d-480a-a0c8-8126d45aa26a.png)

2. After the modification is completed, click NEXT to enter step5
    

## Step5: export

1. The final step is to export the results of image registration, tissue segmentation and cell segmentation. Click "Export Image Processing Record" to generate the `.tar.gz` file.
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/4b10a2af-f350-4061-8689-11c7140b4ebb.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/06a4478d-e208-406b-a74d-7b30a73acb31.png)

2. Click finish and close to complete the manual image processing
    

# SAW realign

Retrieval process does not distinguish between staining and experimental types when manually processing image data.

Before running realign analysis, you need to use the visual file output from SAW count

##### Connect tar.gz back to the SAW process, SAW convert exports the image

    saw=/PATH/SAW/saw
    ${saw}/bin/saw convert tar2img \
    --image-tar=/PATH/SN.tar.gz \
    --image=/PATH/tar_to_img

Manually process the image using ImageProcessing in StereoMap to obtain a realigned image TAR package, which is used as input access process analysis

Script cluster path:

    ## The position of the double# annotation mark needs to be modified
    
    SN=C02533C1  ##Modified to the chip SN analyzed this time
    saw=saw=/PATH/SAW/saw  ##Saw software used,
    countData=/path/to/count/data/Let_me_see_see_mouse_kidney  ##count output directory of automatic process
    tar=/path/to/realigned/image/tar  ##After manual operation, be careful not to be together
    
    source ${saw}/bin/env
    export LD_LIBRARY_PATH=${saw}/lib/geftools/lib:$LD_LIBRARY_PATH
    
    ${saw}/bin/saw realign \
        --id=${SN}_realigned \  ##You can modify the ID by yourself, be careful not to repeat
        --sn=${SN} \
        --count-data=${countData} \
       #--adjusted-distance=20 \ ##Cell correction distance can be modified. If the user is very satisfied with the manual circle selection result or the third-party result, it can be set to 0 to close the cell correction step.
        --realigned-image-tar=${tar}
       #--no-matrix ##Matrix can be output without outputting the matrix and subsequent analysis
       #--no-report ##You can not output reports
    
    

### Task delivery

    nohup sh <your_shell_script.sh> &

# StereoMap Visual View

[Delivery results and visualization can be viewed in the cloud platform](https://cloud.stomics.tech/#/dashboard)
