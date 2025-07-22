# stereomap4.0版本手动方案

- [stereomap4.0版本手动方案](#stereomap40版本手动方案)
    - [Pipeline介绍](#pipeline介绍)
    - [saw软件手册（内附下载链接）](#saw软件手册内附下载链接)
- [StereoMap 图像QC](#stereomap-图像qc)
- [SAW count](#saw-count)
- [StereoMap 可视化查看手动图像处理](#stereomap-可视化查看手动图像处理)
  - [Step1：upload image 导入图片](#step1upload-image-导入图片)
  - [Step2：image registration 图像配准](#step2image-registration-图像配准)
  - [Step3: tissue segmentation 组织分割](#step3-tissue-segmentation-组织分割)
  - [Step4: cell segmentation 细胞分割](#step4-cell-segmentation-细胞分割)
  - [Step5: export 导出](#step5-export-导出)
- [SAW realign](#saw-realign)
      - [将tar.gz接回SAW流程，SAW convert 导出图像](#将targz接回saw流程saw-convert-导出图像)
    - [任务投递](#任务投递)
- [StereoMap 可视化查看](#stereomap-可视化查看)

目的：当CellBin自动化结果不满足需求时，可使用手动方案来处理，以达到需求

适用范围：数据必须是跑过saw v8流程，矩阵文件格式是.stereo

**工具介绍**

**StereoMap（手动处理）**

StereoMap 与 ImageStudio 合并为一个软件 StereoMap4.0，分为可视化、图像处理及小工具三个模块，图像处理兼容 ImageStudio 的部分功能，修改为 Step by step 的方式，降低用户的学习成本，提升用户满意度

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/ZWGl0wN0NgwMO34Y/img/bcfdd062-2094-45dc-a6ec-2a52c591f92d.png)

|  工具名称  |  使用版本  |  工具说明  |  下载地址（华大网盘）  |  备注  |
| --- | --- | --- | --- | --- |
|  StereoMap  |  StereoMap-4.0.1  |  可视化，手动处理图像进行配准，组织分割，细胞分割，图像QC  |  [https://www.stomics.tech/products/BioinfoTools/OfflineSoftware](https://www.stomics.tech/products/BioinfoTools/OfflineSoftware)  |  输入必须是跑过saw v8的结果  |

*   **如果有版本更新，强烈建议手动卸载历史版本后，再重新安装，不建议覆盖安装**
    

**SAW**

Stereo-seq Analysis Workflow (SAW) 软件套件是一套捆绑式pipelines，用于将测序读数定位到其在组织切片上的空间位置，量化空间特征表达，并直观地呈现空间表达分布。SAW 结合显微镜图像处理来Stereo-seq sequencing platform的数据，生成空间特征表达矩阵。然后，分析人员可以以输出文件为起点进行下游分析

### Pipeline介绍

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/ZWGl0wN0NgwMO34Y/img/763ed9f0-ec86-4fd2-8941-bc3c5434e6d7.png)

### saw软件手册（内附下载链接）

saw说明手册链接：[https://stereotoolss-organization.gitbook.io/saw-user-manual/](https://stereotoolss-organization.gitbook.io/saw-user-manual/)

**操作流程**

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/2522290e-2f5f-42d7-9499-e613f2647bee.png)

# StereoMap 图像QC

1.  打开StereoMap 点击Tools
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/29f08ebf-b44e-4c3b-b2ea-248a3813cc05.png)

2.  点击start 进入QC界面
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/26d37411-884a-4cd4-a0e9-08705dac0c27.png)

3.  将待QC的芯片文件夹拖入界面，填写完毕后点击RUN开始QC
    

\*文件路径需用英文命名，文件名为芯片号（例A03990A1，不能有\_或者数字）否则有可能影响QC结果

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/d56a61c1-17f5-40b9-b800-d047385ac888.png)

4.  QC 完成后的图像输出变为一个 TAR 文件（存储 IPR 及显微镜的原图），输出文件的储存路径可在设置中更改
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/dc5039f4-ab12-498c-904a-9f876fbbd66c.png)

# SAW count

**运行SAW count之前**

[向右]根据 STOmics 产品线的不同，参数的使用也略有不同。在进行分析之前，请注意有关检测试剂盒版本的信息，选择合适的SAW版本

要使用显微镜染色图像上的自动配准和组织检测从新鲜冷冻 （FF） 样品生成单个文库的空间特征计数，请使用以下参数运行

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

首次运行后，您可以从 StereoMap 中的演示和手动配准中获取文件。经过一系列手动过程后，图像将被传回以获得新结果

    cd /saw/runs
    
    saw realign \
        --id=Adjuated_Demo_Mouse_Brain \  ##task id
        --sn=SS200000135TL_D1 \  ##SN information of Stereo-seq Chip 
        --count-data=/path/to/previous/SAW/count/task/folder \  ##output folder of previous SAW count
        #--adjusted-distance=10 \  ##default to 10 pixel
        --realigned-image-tar=/path/to/realigned/image/tar   ##realigned image .tar.gz from StereoMap

\*更多信息请查看[手动处理教程](https://stereotoolss-organization.gitbook.io/saw-user-manual-v8.0/tutorials/with-manually-processed-files)


[向右]以此芯片为例

```
**SN**：C02533C1

**Speices**: mouse

**Tissue**: kidney

**Chip size**: 1\*1

**Stain type**: H&E
```


**FF分析**

    ## 双#号注释标记的位置需要修改
    
    SN=C02533C1  ##修改为此次分析的芯片SN
    saw=/PATH/to/saw  ##使用的saw软件，如有更新及时修改，只用改saw-v8.0.0a7中a后面的数字，表示使用的内测软件版本号
    data=/jdfssz3/ST_STOMICS/P20Z10200N0039_autoanalysis/stomics/tmpForDemo/${SN}
    image=/ldfssz1/ST_BIOINTEL/P20Z10200N0039/guolongyu/8.0.0/Image_v4/${SN}
    tar=$(find ${image} -maxdepth 1 -name \*.tar.gz | head -1)
    
    source ${saw}/bin/env
    export LD_LIBRARY_PATH=${saw}/lib/geftools/lib:$LD_LIBRARY_PATH
    
    ${saw}/bin/saw count \
        #--id=Let_me_see_see_mouse_kidney \  ##可自行修改ID，注意任务不要重复（title）
        --sn=${SN} \
        --omics=transcriptomics \
        --kit-version='Stereo-seq T FF V1.2' \（目前不需要更改参数）
        --sequencing-type='PE100_50+100' \（目前版本不需要，但为了兼容后续的）
        --organism=mouse \  ##填写物种（可不填）
        --tissue=kidney \  ##填写组织or病症（可不填）
        --chip-mask=${data}/mask/${SN}.barcodeToPos.h5 \
        --fastqs=${data}/reads \ 
        --reference=/PATH/reference/mouse \  ##mouse/human/rat供选择，修改最后的文件夹名称即可
        --image-tar=${tar} \
        --local-cores=48

**FFPE分析**

    ## 双#号注释标记的位置需要修改
    
    SN=C02533C1  ##修改为此次分析的芯片SN
    saw=/PATH/to/saw  ##使用的saw软件，如有更新及时修改，只用改saw-v8.0.0a7中a后面的数字，表示使用的内测软件版本号
    mask=/path/to/chip/mask/file/SN.mask.h5  ##修改为FFPE的mask路径
    fastq=/path/to/fastq/folder  ##修改为FFPE的FASTQs路径，如有多个目录则传多个文件夹
    image=/path/to/QC/image/tar ##修改为QC后的图像TAR包路径
    
    source ${saw}/bin/env
    export LD_LIBRARY_PATH=${saw}/lib/geftools/lib:$LD_LIBRARY_PATH
    
    ${saw}/bin/saw count \
        --id=What_about_FFPE_mouse_kidney \  ##可自行修改ID，注意任务不要重复
        --sn=${SN} \
        --omics=transcriptomics \
        --kit-version='Stereo-seq N FFPE V1.0' \
        --sequencing-type='PE75_25+59' \
        --organism=mouse \  ##填写物种
        --tissue=kidney \  ##填写组织or病症
        --chip-mask=${mask} \
        --fastqs=${fastq} \ 
        --reference=/PATH/reference/human \  
        --image-tar=${tar} \
        --microbiome-detect \ ##自行选择 可以关闭
        --local-cores=48
       #--image=/path/to/TIFF
    

**输出结果说明**

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/ZWGl0wN0NgwMO34Y/img/1c98331f-b830-4dc6-acf3-53d0c9b67eb1.png)

![1713279982523.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/ZWGl0wN0NgwMO34Y/img/d9fef8da-657e-4152-b977-84e096e68b27.png)


下载`visualization.tar.gz`在本地解压，使用StereoMap软件进行可视化查看和手动图像处理。


# StereoMap 可视化查看手动图像处理

打开StereoMap，选择image processing

## Step1：upload image 导入图片

1.  选择染色类型（ssDNA, DAPI, DAPI&mIF, H&E）![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/3c022a77-dd41-48a8-857c-5e2d0353f2bc.png)
    
2.  选择完毕后导入图片 图片支持的数据格式为：`.tar.gz``.stereo``.tif``.tiff`（TIFF 大图、4.0 小工具 QC 后的 TAR.GZ 及 SAW 8.0 输出的可视化.stereo 文件（必须有图像）），将图片直接拖入对应区域，右侧信息会自动填入
    

[向右]若TIFF 大图数据会重新做 QC 评估

[向右]若是TAR.GZ 或者.stereo 文件会读取图像 QC 的结果并做展示，等待读取完毕后，点击NEXT

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/20830336-4c09-412e-b4ec-e6db7262a4cf.png)

## Step2：image registration 图像配准

1.  在右侧morphology处添加矩阵，矩阵必须是`.stereo`文件
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/b775b5d0-40a7-4b78-9f8c-3d84ddf75831.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/708c0e17-b0cb-4bdd-b521-887d0ae1b291.png)

2.  图像配准
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/24e38782-2f23-4ff4-b20e-4f655b777d7a.png)

[向右]配准过程包括两个阶段，根据形态大致匹配图像的方向，并微调位置和比例，使其与空间特征表达图完全重叠。您还可以选择“芯片轨迹线 chip track lines”来显示从空间要素表达式矩阵派生的轨迹线，以帮助精细对齐

[向右]要大致对齐图像，您需要在方向上匹配显微镜图像和特征密度图

使用**“翻转”工具**![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/a6bdd0d8-1508-4a67-b9b6-819e6050d3fc.png)镜像图像

使用**控制旋钮**![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/84a56c87-cbb2-4899-941c-ff9da08681fb.png)沿同一方向旋转图像

[向右]一旦图像的方向和空间要素表达式矩阵匹配，就可以继续进行精细对齐步骤。在精细对齐阶段，您需要将图像移动到组织可以重叠的位置

通过设置步长和平移四个方向来使用**“移动”面板**![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/e458a984-63a2-4021-b07d-954b999c674e.png)

显微镜图像的尺寸可能与空间特征表达式图不同，您可以使用**比例工具**![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/ca901cc8-d9f6-4cf5-b6bd-e685a8cfe579.png) 调整比例

[向右]您可以通过选中“芯片轨迹线”来对齐图像，以显示参考轨迹线模板，并让轨迹线直接重叠。此时，芯片轨迹线模板是矩阵的表示。如果轨道线变暗，则可以手动调整图像 的对比度![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/ef50bc5c-0e26-429e-b9e5-465c7b9f8207.png) 、亮度![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/12ccdc77-2eb3-4ac2-a6ef-06dc7a863639.png) 和不透明度![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/e1a11f41-b2fd-4b18-ab62-91d52ab782f1.png)

3.  配准完毕后点击NEXT，确认
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/47619929-7ac2-4f05-a406-0c0d42294843.png)

## Step3: tissue segmentation 组织分割

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/87d49f25-9fc3-4a2b-8bab-af9f48efb090.png)

1.  手动对分割不完整的地方进行修改
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/83628193-da40-479e-9f09-06021389118d.png) ![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/03672f8b-2485-4f3b-9c1d-3fb46b9259c1.png)

2.  要选择或编辑组织区域，请结合使用**套索**![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/883aea29-c74a-4629-a91d-d022832b491b.png) 、**画笔** ![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/dcd560e9-bb47-4346-8612-401c6b3916e2.png) 和**橡皮擦**![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/a292b2e4-f1e8-422b-b45c-09c6a30f0a46.png) 工具。**套索**通常用于选择或取消选择大区域，而**“画笔**”和**“橡皮擦**”工具更适合较小的区域，例如组织周围的区域或组织中的小孔。
    

[向右]**画笔** ![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/dcd560e9-bb47-4346-8612-401c6b3916e2.png) 填充区域

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/dde3898d-2da3-45f0-82a4-949d53916401.png)

[向右]**橡皮擦**![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/a292b2e4-f1e8-422b-b45c-09c6a30f0a46.png) 消除区域

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/f9bf7b38-7ce0-47e2-91ad-d1c0966478c9.png)

3.  修改完毕
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/bcdf84b7-157b-4945-9a1d-02d3802d20f7.png)

4.  点击NEXT，进入step4
    

## Step4: cell segmentation 细胞分割

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/d9d9ad2d-3341-4987-a73e-18b1ecec247e.png)

1.  修改
    

[向右]使用**套索**![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/5d63bae8-4d5e-4e01-98fa-a56e58814e0c.png) 、**画笔** ![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/11db2b46-1453-47fb-9eff-83ac77784350.png) 和**橡皮擦**![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/f932b18e-ef35-450d-b239-37aba923890e.png) 工具选择或编辑单元格。**套索**最适合用于取消选择大区域（如背景），而**“画笔**”和**“橡皮擦**”工具更适合较小的区域，例如标记单元格或分离单元格簇。

[向右]建议导入一个`.tif`格式的cell mask文件。您可以通过单击 **Segmentation mask** 下拉列表，然后单击来访问您的文件系统![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/9a9363fc-8c51-4614-ad8d-32be2c66824a.png) 来导入它。如果导入的结果不令人满意，可以通过单击![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/d584b570-922d-480a-a0c8-8126d45aa26a.png) 导入新cell mask来替换它

2.  修改完毕后点击NEXT，进入step5
    

## Step5: export 导出

1.  最后一步是导出图像配准、组织分割和细胞分割的结果。单击**“导出图像处理记录”**，生成`.tar.gz`文件。
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/4b10a2af-f350-4061-8689-11c7140b4ebb.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/06a4478d-e208-406b-a74d-7b30a73acb31.png)

2.  点击finish and close 完成手动图像处理
    

# SAW realign

realign流程接回手动处理图像数据时不区分染色类型和实验类型

在运行realign分析之前，需要使用SAW count输出的可视化文件

#### 将tar.gz接回SAW流程，SAW convert 导出图像

    saw=/PATH/SAW/saw
    ${saw}/bin/saw convert tar2img \
    --image-tar=/PATH/SN.tar.gz \
    --image=/PATH/tar_to_img

在StereoMap中使用ImageProcessing对图像进行手动处理，得到一个realigned 图像TAR包，作为输入接入流程分析

脚本集群路径：

    ## 双#号注释标记的位置需要修改
    
    SN=C02533C1  ##修改为此次分析的芯片SN
    saw=saw=/PATH/SAW/saw  ##使用的saw软件，
    countData=/path/to/count/data/Let_me_see_see_mouse_kidney  ##count自动流程的输出目录
    tar=/path/to/realigned/image/tar  ##手动操作后tar存放的路径，注意不要多个在一起
    
    source ${saw}/bin/env
    export LD_LIBRARY_PATH=${saw}/lib/geftools/lib:$LD_LIBRARY_PATH
    
    ${saw}/bin/saw realign \
        --id=${SN}_realigned \  ##可自行修改ID，注意不要重复
        --sn=${SN} \
        --count-data=${countData} \
       #--adjusted-distance=20 \  ##可以修改细胞修正距离，如果用户对于手动圈选结果or第三方结果非常满意时可以设置为0来关闭细胞修正步骤
        --realigned-image-tar=${tar}
       #--no-matrix  ##可以不输出矩阵及后面的分析
       #--no-report  ##可以不输出报告
    
    

### 任务投递

    nohup sh <your_shell_script.sh> &

# StereoMap 可视化查看

[可在云平台中查看投递结果和可视化](https://cloud.stomics.tech/#/dashboard)