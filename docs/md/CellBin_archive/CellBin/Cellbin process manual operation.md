# Manual operation of Cellbin process
- [CellbinManual operation of the process](#cellbin process manual operation)
- [Related materials](#Related Materials)
- [Standardized process version](#Standardized Process Version)
  - [StereoMap imageQC](#stereomap-image qc)
  - [Scene1：QCsuccess](#Scenario 1qc successful)
    - [SAW countThere is a picture process](#saw-count has a diagram flow)
      - [Output result description](#Output result description)
      - [StereoMap Visual viewing manual image processing](#stereomap-visual viewing manual image processing)
    - [Problem arises：](#Problem arises)
    - [There is a problem with registration](#There is a problem with registration)
    - [There is a problem with organizational division](#There is a problem with organizational division)
    - [There is a problem with cell segmentation](#There is a problem with cell segmentation)
    - [SAW realign](#saw-realign)
  - [Scene2：QCfail](#Scene 2qc failed)
    - [SAW countNo picture process](#saw-count no-picture process)
    - [stereomapRegistration](#stereomap registration)
    - [SAW realign](#saw-realign-1)
    - [stereomapVisual viewing](#stereomap visual viewing)

# Related Materials

SAW Software Manual: [Overview | SAW User Manual V8.0 (gitbook.io)](https://stereotoolss-organization.gitbook.io/saw-user-manual-v8.0)

Cloud Platform：[Login (stomics.tech)](https://cloud.stomics.tech/#/login)

# Standardized process version

**Input file (as long as you have the following three files, you can perform standardized processes)**
```

Stereomap4.0 tar.gz output from QC

fastqs: fq.gz

mask：SN.barcodeToPos.h5
```

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/63bc610d-8f89-4bfd-b54e-e2f512c96c7d.png)


**Experimental Chip**

```
SN：A02497C1

species：mouse

tissue：kidney

chip size：1*1

stain sype：HE
```


## StereoMap Image QC

1. Open StereoMap and click Tools
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/29f08ebf-b44e-4c3b-b2ea-248a3813cc05.png)

2. Click start to enter the QC interface
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/26d37411-884a-4cd4-a0e9-08705dac0c27.png)

3. Drag the chip folder to be QC into the interface, click RUN after filling in it, start QC
    

* The file path must be named in English, the file name is chip number (for example, A03990A1, there cannot be \_ or numbers) otherwise it may affect the QC results.

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/d56a61c1-17f5-40b9-b800-d047385ac888.png)

4. The image output after QC is completed becomes a TAR file (storing the original image of the IPR and microscope), and the storage path of the output file can be changed in the settings.
    

The output path is modified, that is, viewing method:

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/973eab17-c641-45df-aa3c-dfd742201f6a.png)

Find the output tar.gz in local

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/dc5039f4-ab12-498c-904a-9f876fbbd66c.png)

## Scenario 1: QC success

After QC is successful, you need to connect the .tar.gz file to the SAW count process to obtain the process results, including the image results. The operation is as follows:

### SAW count has a graph process

**Before running SAW count**

Depending on the STOmics product line, the use of parameters is also slightly different. Before performing the analysis, please pay attention to the information about the test kit version and select the appropriate SAW version


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

* For more information, please see [Manual Processing Tutorial](https://stereotoolss-organization.gitbook.io/saw-user-manual-v8.0/tutorials/with-manually-processed-files)


Take this chip as an example

```
**SN**：C0XXXXC1

**Speices**: mouse

**Tissue**: kidney

**Chip size**: 1\*1

**Stain type**: H&E
```


    ## The position of the double# annotation mark needs to be modified
    
    SN=C0XXXXC1  ##Modified to the chip SN analyzed this time
    saw=/PATH/saw
    data=/PATH/tmpForDemo/${SN}
    image=/PATH/${SN}
    tar=$(find ${image} -maxdepth 1 -name \*.tar.gz | head -1)
    
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

#### Output result description

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn5gQE9bxyOo83/img/67019e41-a911-4918-85ac-23b5ddc05488.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn5gQE9bxyOo83/img/c6c3074c-df95-4beb-bc99-dd28928b0e64.png)


Download `visualization.tar.gz` locally decompressed, use StereoMap software for visual viewing and manual image processing.


#### StereoMap Visual View Manual Image Processing

Open StereoMap, select image processing and drag tar.gz into the decompressed visualization to visualize it; if you are not satisfied with the visualization results, you can directly modify it manually on the stereomap.

### Problem arises:

If a problem occurs when checking, it needs to be modified in Stereomap's ImageProcessing. The following is a solution based on different problems.

### There is a problem with registration

* #### Step1: upload image Import image

Select the staining type (ssDNA, DAPI, DAPI&mIF, H&E) in step1. After the selection is completed, you can directly drag it into tar.gz in visualization; (The data format supported by the imported image is: `.tar.gz``.stereo``.tif``.tiff`)
![1718242973936_70A17E01-8C35-495b-BCDF-46FBC6D0F898.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/55270990-3e09-4471-937d-5b23e362489f.png)

* #### Step2: image registration

1. Add a matrix at the morphology on the right and select the `.stereo` file
    

![1718243014318_7BB9B794-A75E-4c77-9E0D-B915BF1F6963.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/60d3cd4a-6f60-4c52-8944-7953de3145cf.png)

2. After opening the matrix, you find that the registration is incorrect. Borrow the toolbar on the right for manual registration.
    

![45351fee0f0b192ceaa3f62cd603f2f2.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/2c38c644-627a-4f85-b83d-665cb188f042.png)

3. Modification is completed
    

![1718243128168_DA79F6F2-7D01-4b8d-952F-237B072576D8.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/69494d5b-8bc9-4208-b366-fa2507a1c1f0.png)

* #### Step3: tissue segmentation Organize segmentation (can be skipped)

[If modification is required，Please check the organizational segmentation section described below。](#step3-tissue-segmentation-organization segmentation)

* #### Step4: cell segmentation (can be skipped)

[If modification is required，Please check the cell segmentation section described below。](#step4-cell-segmentation-cell segmentation)

* #### Step5: export

1. The final step is to export the results of image registration, tissue segmentation and cell segmentation. Click "Export Image Processing Record"** to generate the `.tar.gz` file
    

![1718243168590_43CB426D-233B-473c-865E-FABC9DB2A7BA.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/a248823e-df8b-440b-b024-6a584a48ac08.png)

2. The manual processing of `.tar.gz` and new registration diagram`.tif` can be found in the export path.
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/f3aed7cf-7e46-45ad-8f09-2b8f0822e6a2.png)

### There is something wrong with the organization segmentation

* #### Step1: upload image Import image

Select the staining type (ssDNA, DAPI, DAPI&mIF, H&E) in step1. After the selection is completed, you can directly drag it into tar.gz in visualization; (The data format supported by the imported image is: `.tar.gz``.stereo``.tif``.tiff`)
![1718242973936_70A17E01-8C35-495b-BCDF-46FBC6D0F898.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/55270990-3e09-4471-937d-5b23e362489f.png)

* #### Step2: image registration

1. You can skip it without modification, just click "Next"
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn5gQE9bxyOo83/img/035fe6a7-ab9d-4c12-854d-b109e6be6251.png)

* #### Step3: tissue segmentation

1. After entering step3, use the toolbar on the right to manually modify the unsatisfied organization segmentation part.
    
2. As shown in the figure, some tissues are divided and not covered. Choose a brush to fill and apply the uncovered tissue.
    
3. Click next after the modification is completed
    

![d3b305b9e7fe6d876e7f54ea711a9b5b.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/7a5da132-24bf-4a6f-9b93-8e2db3975698.png)

* #### Step4: cell segmentation (can be skipped)

If modification is required，Please check out the following[Cell segmentation part](#step4-cell-segmentation-cell segmentation)

* #### Step5: export

1. The final step is to export the results of image registration, tissue segmentation and cell segmentation. Click "Export Image Processing Record"** to generate the `.tar.gz` file
    

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/08671588-f36f-4ee4-a39d-629920e87d5a.png)

2. The manual processing of `.tar.gz` and new registration diagram`.tif` can be found in the export path.
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/f3aed7cf-7e46-45ad-8f09-2b8f0822e6a2.png)

### There is a problem with cell segmentation

* #### Step1: upload image Import image

Select the staining type (ssDNA, DAPI, DAPI&mIF, H&E) in step1. After the selection is completed, you can directly drag it into tar.gz in visualization; (The data format supported by the imported image is: `.tar.gz``.stereo``.tif``.tiff`)
![1718242973936_70A17E01-8C35-495b-BCDF-46FBC6D0F898.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/55270990-3e09-4471-937d-5b23e362489f.png)

* #### Step2: image registration

2. You can skip it without modification, just click "Next"
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn5gQE9bxyOo83/img/035fe6a7-ab9d-4c12-854d-b109e6be6251.png)

* #### Step3: tissue segmentation Organize segmentation (can be skipped)

If modification is required，Please check the above[Organizational segmentation](#step3-tissue-segmentation-organization segmentation).


* #### Step4: cell segmentation

1. After entering step4, you can manually modify the unsatisfied cell segmentation part with the tool in the toolbar on the right.
    
2.  As shown，The red circled part is"Multiple points"，Therefore, use an eraser to wipe off the multiple parts
    

![feeef791f5b7a7c574c9167879dda770.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/c40b7d71-e66a-4cf3-aaa3-3f56acab071d.png)

3. If the current tool is not convenient to modify, you can use external tools to generate a new cell segmentation result image (image in .tif format) and then import it. The import method is as follows:
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn5gQE9bxyOo83/img/1068822f-595f-41de-b791-09ba26b93011.png)


Here, an external tool Qupath generates cell segmentation results images, which can be used with reference to the following documents:

[Cell segmentation solution (ssDNA, DAPI, H&E) - Qupath operating instructions](../QuPath_to_Cellbin/2.Qupath partial operation SOP.md)


4. Click next after you are satisfied with the modification
    

* #### Step5: export

1. The final step is to export the results of image registration, tissue segmentation and cell segmentation. Click "Export Image Processing Record"** to generate the `.tar.gz` file
    

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/47320feb-64ce-46ff-8b6b-4173c28bec33.png)

2. The manual processing of `.tar.gz` and new registration diagram`.tif` can be found in the export path.
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/f3aed7cf-7e46-45ad-8f09-2b8f0822e6a2.png)

### SAW realign

Retrieval process does not distinguish between staining and experimental types when manually processing image data.

After visually checking the manually modified image, connect the results output by SAW count and the tar.gz obtained after manual modification to SAW realign, and run the SAW process again. The output file composition type is the same as the file output by SAW count at the beginning.

    ## The position of the double# annotation mark needs to be modified
    
    SN=A02497C1  ##Modified to the chip SN analyzed this time
    saw=/PATH/saw  ## If the saw software used is updated and modified in time, you only need to change the number after a in saw-v8.0.0a7 to indicate the version number of the internal test software used.
    countData=/PATH/countData ##count output directory of automatic process
    tar=/PATH/to/A02497C1_XXX.tar.gz  ##After manual operation, be careful not to be together
    
    export LD_LIBRARY_PATH=${saw}/lib/geftools/lib:$LD_LIBRARY_PATH
    
    ${saw}/bin/saw realign \
        --id=${SN}_realigned \  ##You can modify the ID by yourself, be careful not to repeat
        --sn=${SN} \
        --count-data=${countData} \
       #--adjusted-distance=20 \ ##Cell correction distance can be modified. If the user is very satisfied with the manual circle selection result or the third-party result, it can be set to 0 to close the cell correction step.
        --realigned-image-tar=${tar}
       #--no-matrix ##Matrix can be output without outputting the matrix and subsequent analysis
       #--no-report ##You can not output reports
    
    

Output:

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/3a0cd14c-af53-42aa-8a7d-b936609aa26b.png)

Finally, open outs>visualization in stereomap to view the final process results visually

## Scenario 2: QC failure

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/a/R9KY50Al2S7OGJeA/c28bb23b3083421f8a1d0b7cbbcc32310976.png)

### SAW count without graph process

** At this time, QC fails and the picture does not meet the standards. SAW count needs to select the process without pictures, that is, do not enter image**

**Input file**
```

fastqs: fq.gz

mask：SN.barcodeToPos.h5
```

**Experimental Chip**
```

SN：B0XXXXXB5

species：mouse

tissue：brain

chip size：1*1

stain sype：HE
```


    SN=B0XXXXXB5   ##Modified to the chip SN analyzed this time
    saw=/PATH/SAW/saw-v8.0.1  ##Saw software used
    data=/path/temp_demo/${SN} #Data path
    
    export LD_LIBRARY_PATH=${saw}/lib/geftools/lib:$LD_LIBRARY_PATH
    
    ${saw}/bin/saw count \
    --id=zc7 \  ##You can modify the ID by yourself, be careful not to repeat the tasks (title)
    --sn=${SN} \
    --omics=transcriptomics \
    --kit-version='Stereo-seq T FF V1.2' \（No changes to parameters are required）
    --sequencing-type='PE100_50+100' \（The current version does not require，But for the sake of compatibility with subsequent）
    --organism=mouse \  ##Fill in the species (cannot fill in)
    --tissue=kidney \  ##Fill in tissue or disease (can not fill in)
    --chip-mask=/PATH/to/B01020B5.barcodeToPos.h5    \
    --fastqs=/PATH/reads/       \ 
    --reference=/PATH/reference/mouse    \  ##mouse/human/rat for selection, just modify the last folder name
    --local-cores=48

**SAW count without graph flow output file is: **

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/5246ab4f-f602-4fd5-aebd-45e634e1c738.png)

### Stereomap registration

1. Use stereomap to open the tar.gz after QC, enter step2, select the .stereo file for manual registration
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/dea252db-e42a-4c0b-b441-f36177d361bc.png)

2. Before and after registration (manual registration will have inevitable errors)
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/078b59b9-d08a-4367-8021-30b3a3fb5a07.png) ![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/7d6e5885-797f-459b-a5ef-f2ead1b1071b.png)

3. Skip step3 and step4, save and export registered tar.gz in step5
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/6456016a-4f08-44f4-b1d7-9080ab988b7a.png)

4. Generate `.tar.gz` file and new registration diagram
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/d93b197f-5283-4fe5-8983-f87d407619e6.png)

### SAW realign

Connect the results of the previous SAW count output and the tar.gz obtained after manual modification to SAW realign, and run the SAW process again. The output file composition type is the same as the file output from SAW count at the beginning.

    ## The position of the double# annotation mark needs to be modified
    
    SN=B01020B5  ##Modified to the chip SN analyzed this time
    saw=/PATH/to/saw  ## If the saw software used is updated and modified in time, you only need to change the number after a in saw-v8.0.0a7 to indicate the version number of the internal test software used.
    countData=/PATH/to/count ##count output directory of automatic process
    tar=/PATH/TO/tar.tar  ##After manual operation, be careful not to be together
    
    export LD_LIBRARY_PATH=${saw}/lib/geftools/lib:$LD_LIBRARY_PATH
    
    ${saw}/bin/saw realign \
        --id=${SN}_realigned \  ##You can modify the ID by yourself, be careful not to repeat
        --sn=${SN} \
        --count-data=${countData} \
       #--adjusted-distance=20 \ ##Cell correction distance can be modified. If the user is very satisfied with the manual circle selection result or the third-party result, it can be set to 0 to close the cell correction step.
        --realigned-image-tar=${tar}
       #--no-matrix ##Matrix can be output without outputting the matrix and subsequent analysis
       #--no-report ##You can not output reports
    
    

### stereomap visual viewing

Finally, open outs>visualization in stereomap to view the final process results. If you find that the results of tissue segmentation and cell segmentation do not meet the requirements during the viewing, you will modify them according to the instructions of "There is a problem with tissue segmentation" and "There is a problem with cell segmentation" in the above scenario 1.
