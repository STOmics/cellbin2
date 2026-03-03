# Cellbin流程手动操作
- [Cellbin流程手动操作](#cellbin流程手动操作)
- [相关材料](#相关材料)
- [标准化流程版本](#标准化流程版本)
  - [StereoMap 图像QC](#stereomap-图像qc)
  - [场景1：QC成功](#场景1qc成功)
    - [SAW count有图流程](#saw-count有图流程)
      - [输出结果说明](#输出结果说明)
      - [StereoMap 可视化查看手动图像处理](#stereomap-可视化查看手动图像处理)
    - [问题出现：](#问题出现)
    - [配准有问题](#配准有问题)
    - [组织分割有问题](#组织分割有问题)
    - [细胞分割有问题](#细胞分割有问题)
    - [SAW realign](#saw-realign)
  - [场景2：QC失败](#场景2qc失败)
    - [SAW count无图流程](#saw-count无图流程)
    - [stereomap配准](#stereomap配准)
    - [SAW realign](#saw-realign-1)
    - [stereomap可视化查看](#stereomap可视化查看)

# 相关材料

SAW软件手册：[Overview | SAW User Manual V8.0 (gitbook.io)](https://stereotoolss-organization.gitbook.io/saw-user-manual-v8.0)

云平台：[Login (stomics.tech)](https://cloud.stomics.tech/#/login)

# 标准化流程版本

**输入文件（只要具备以下三个文件就可以进行标准化流程）**
```

stereomap4.0 QC输出的tar.gz

fastqs: fq.gz

mask：SN.barcodeToPos.h5
```

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/63bc610d-8f89-4bfd-b54e-e2f512c96c7d.png)


**实验芯片**

```
SN：A02497C1

species：mouse

tissue：kidney

chip size：1*1

stain sype：HE
```


## StereoMap 图像QC

1.  打开StereoMap 点击Tools
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/29f08ebf-b44e-4c3b-b2ea-248a3813cc05.png)

2.  点击start 进入QC界面
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/26d37411-884a-4cd4-a0e9-08705dac0c27.png)

3.  将待QC的芯片文件夹拖入界面，填写完毕后点击RUN开始QC
    

* 文件路径需用英文命名，文件名为芯片号（例A03990A1，不能有\_或者数字）否则有可能影响QC结果

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/d56a61c1-17f5-40b9-b800-d047385ac888.png)

4.  QC 完成后的图像输出变为一个 TAR 文件（存储 IPR 及显微镜的原图），输出文件的储存路径可在设置中更改
    

输出路径修改即查看方法：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/973eab17-c641-45df-aa3c-dfd742201f6a.png)

在本地中找到输出的tar.gz

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/oJGq768aKdx2nAKe/img/dc5039f4-ab12-498c-904a-9f876fbbd66c.png)

## 场景1：QC成功

QC成功之后需要将.tar.gz文件接入SAW的count流程，才可以获得流程结果，其中包含图像结果。操作如下：

### SAW count有图流程

**运行SAW count之前**

根据 STOmics 产品线的不同，参数的使用也略有不同。在进行分析之前，请注意有关检测试剂盒版本的信息，选择合适的SAW版本


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

* 更多信息请查看[手动处理教程](https://stereotoolss-organization.gitbook.io/saw-user-manual-v8.0/tutorials/with-manually-processed-files)


以此芯片为例

```
**SN**：C0XXXXC1

**Speices**: mouse

**Tissue**: kidney

**Chip size**: 1\*1

**Stain type**: H&E
```


    ## 双#号注释标记的位置需要修改
    
    SN=C0XXXXC1  ##修改为此次分析的芯片SN
    saw=/PATH/saw
    data=/PATH/tmpForDemo/${SN}
    image=/PATH/${SN}
    tar=$(find ${image} -maxdepth 1 -name \*.tar.gz | head -1)
    
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

#### 输出结果说明

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn5gQE9bxyOo83/img/67019e41-a911-4918-85ac-23b5ddc05488.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn5gQE9bxyOo83/img/c6c3074c-df95-4beb-bc99-dd28928b0e64.png)


下载`visualization.tar.gz`在本地解压，使用StereoMap软件进行可视化查看和手动图像处理。


#### StereoMap 可视化查看手动图像处理

打开StereoMap，选择image processing后拖入解压好的visualization中的tar.gz，进行可视化查看；若对可视化结果不满意，可直接在stereomap上进行手动修改

### 问题出现：

若在检查出问题，需要在Stereomap的ImageProcessing中进行修改，以下根据不同的问题情况给出解决的方案。

### 配准有问题

* #### Step1：upload image 导入图片

在step1中选择染色类型（ssDNA, DAPI, DAPI&mIF, H&E），选择完毕后可直接拖入visualization中的tar.gz; （导入图片支持的数据格式为：`.tar.gz``.stereo``.tif``.tiff`）
![1718242973936_70A17E01-8C35-495b-BCDF-46FBC6D0F898.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/55270990-3e09-4471-937d-5b23e362489f.png)

* #### Step2：image registration 图像配准

1.  在右侧morphology处添加矩阵，选择`.stereo`文件
    

![1718243014318_7BB9B794-A75E-4c77-9E0D-B915BF1F6963.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/60d3cd4a-6f60-4c52-8944-7953de3145cf.png)

2.  打开矩阵后发现配准有误，借用右方的工具栏进行手动配准
    

![45351fee0f0b192ceaa3f62cd603f2f2.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/2c38c644-627a-4f85-b83d-665cb188f042.png)

3.  修改完毕
    

![1718243128168_DA79F6F2-7D01-4b8d-952F-237B072576D8.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/69494d5b-8bc9-4208-b366-fa2507a1c1f0.png)

* #### Step3: tissue segmentation 组织分割（可跳过）

[若需要修改，请查看下面介绍的组织分割部分。](#step3-tissue-segmentation-组织分割)

* #### Step4: cell segmentation 细胞分割（可跳过）

[若需要修改，请查看下面介绍的细胞分割部分。](#step4-cell-segmentation-细胞分割)

* #### Step5: export 导出

1.  最后一步是导出图像配准、组织分割和细胞分割的结果。单击**“导出图像处理记录”**，生成`.tar.gz`文件
    

![1718243168590_43CB426D-233B-473c-865E-FABC9DB2A7BA.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/a248823e-df8b-440b-b024-6a584a48ac08.png)

2.  可在导出路径下找到手动处理后的`.tar.gz`和新的配准图`.tif`
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/f3aed7cf-7e46-45ad-8f09-2b8f0822e6a2.png)

### 组织分割有问题

* #### Step1：upload image 导入图片

在step1中选择染色类型（ssDNA, DAPI, DAPI&mIF, H&E），选择完毕后可直接拖入visualization中的tar.gz; （导入图片支持的数据格式为：`.tar.gz``.stereo``.tif``.tiff`）
![1718242973936_70A17E01-8C35-495b-BCDF-46FBC6D0F898.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/55270990-3e09-4471-937d-5b23e362489f.png)

* #### Step2：image registration 图像配准

1.  不需要修改可跳过，直接点“Next”
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn5gQE9bxyOo83/img/035fe6a7-ab9d-4c12-854d-b109e6be6251.png)

* #### Step3: tissue segmentation 组织分割

1.  进入step3后借助右侧工具栏手动修改不满意的组织分割部分
    
2.  如图，有部分组织分割未被覆盖，选择画笔对未被覆盖的组织进行填补涂抹
    
3.  修改完毕后点击next
    

![d3b305b9e7fe6d876e7f54ea711a9b5b.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/7a5da132-24bf-4a6f-9b93-8e2db3975698.png)

* #### Step4: cell segmentation 细胞分割（可跳过）

若需要修改，请查看下面介绍的[细胞分割部分](#step4-cell-segmentation-细胞分割)

* #### Step5: export 导出

1.  最后一步是导出图像配准、组织分割和细胞分割的结果。单击**“导出图像处理记录”**，生成`.tar.gz`文件
    

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/08671588-f36f-4ee4-a39d-629920e87d5a.png)

2.  可在导出路径下找到手动处理后的`.tar.gz`和新的配准图`.tif`
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/f3aed7cf-7e46-45ad-8f09-2b8f0822e6a2.png)

### 细胞分割有问题

* #### Step1：upload image 导入图片

在step1中选择染色类型（ssDNA, DAPI, DAPI&mIF, H&E），选择完毕后可直接拖入visualization中的tar.gz; （导入图片支持的数据格式为：`.tar.gz``.stereo``.tif``.tiff`）
![1718242973936_70A17E01-8C35-495b-BCDF-46FBC6D0F898.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/55270990-3e09-4471-937d-5b23e362489f.png)

* #### Step2：image registration 图像配准

2.  不需要修改可跳过，直接点“Next”
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn5gQE9bxyOo83/img/035fe6a7-ab9d-4c12-854d-b109e6be6251.png)

* #### Step3: tissue segmentation 组织分割（可跳过）

若需要修改，请查看上面介绍的[组织分割部分](#step3-tissue-segmentation-组织分割)。


* #### Step4: cell segmentation 细胞分割

1.  进入step4后可借助右侧工具栏里的工具对不满意的细胞分割部分进行手动修改
    
2.  如图，红色圈出的部分为"多分"，故用橡皮擦把多分的部分擦去
    

![feeef791f5b7a7c574c9167879dda770.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/c40b7d71-e66a-4cf3-aaa3-3f56acab071d.png)

3.  若当前的工具不便于修改，可以用外部工具生成一张新的细胞分割结果图像（.tif格式的图像），再导入。导入方式如下：
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn5gQE9bxyOo83/img/1068822f-595f-41de-b791-09ba26b93011.png)


在此提供一个外部工具Qupath生成细胞分割结果图像，可参考以下文档进行操作：

[细胞分割解决方案（ssDNA,DAPI,H&E）——Qupath操作说明书](../QuPath_to_Cellbin/2.Qupath部分操作SOP.md)


4.  修改满意后点击next
    

* #### Step5: export 导出

1.  最后一步是导出图像配准、组织分割和细胞分割的结果。单击**“导出图像处理记录”**，生成`.tar.gz`文件
    

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/47320feb-64ce-46ff-8b6b-4173c28bec33.png)

2.  可在导出路径下找到手动处理后的`.tar.gz`和新的配准图`.tif`
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/f3aed7cf-7e46-45ad-8f09-2b8f0822e6a2.png)

### SAW realign

realign流程接回手动处理图像数据时不区分染色类型和实验类型

可视化检查手动修改的图像后，将SAW count输出的结果和手动修改后得到的tar.gz接入SAW realign，重新跑一次SAW流程，输出的文件组成类型与一开始SAW count输出的文件一样

    ## 双#号注释标记的位置需要修改
    
    SN=A02497C1  ##修改为此次分析的芯片SN
    saw=/PATH/saw  ##使用的saw软件，如有更新及时修改，只用改saw-v8.0.0a7中a后面的数字，表示使用的内测软件版本号
    countData=/PATH/countData ##count自动流程的输出目录
    tar=/PATH/to/A02497C1_XXX.tar.gz  ##手动操作后tar存放的路径，注意不要多个在一起
    
    export LD_LIBRARY_PATH=${saw}/lib/geftools/lib:$LD_LIBRARY_PATH
    
    ${saw}/bin/saw realign \
        --id=${SN}_realigned \  ##可自行修改ID，注意不要重复
        --sn=${SN} \
        --count-data=${countData} \
       #--adjusted-distance=20 \  ##可以修改细胞修正距离，如果用户对于手动圈选结果or第三方结果非常满意时可以设置为0来关闭细胞修正步骤
        --realigned-image-tar=${tar}
       #--no-matrix  ##可以不输出矩阵及后面的分析
       #--no-report  ##可以不输出报告
    
    

输出：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/3a0cd14c-af53-42aa-8a7d-b936609aa26b.png)

最后在stereomap打开outs>visualization中可视化查看最终流程结果

## 场景2：QC失败

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/a/R9KY50Al2S7OGJeA/c28bb23b3083421f8a1d0b7cbbcc32310976.png)

### SAW count无图流程

**此时QC失败，图片不达标，SAW count要选择无图流程，即不输入image**

**输入文件**
```

fastqs: fq.gz

mask：SN.barcodeToPos.h5
```

**实验芯片**
```

SN：B0XXXXXB5

species：mouse

tissue：brain

chip size：1*1

stain sype：HE
```


    SN=B0XXXXXB5   ##修改为此次分析的芯片SN
    saw=/PATH/SAW/saw-v8.0.1  ##使用的saw软件
    data=/path/temp_demo/${SN} #数据路径
    
    export LD_LIBRARY_PATH=${saw}/lib/geftools/lib:$LD_LIBRARY_PATH
    
    ${saw}/bin/saw count \
    --id=zc7 \  ##可自行修改ID，注意任务不要重复（title）
    --sn=${SN} \
    --omics=transcriptomics \
    --kit-version='Stereo-seq T FF V1.2' \（目前不需要更改参数）
    --sequencing-type='PE100_50+100' \（目前版本不需要，但为了兼容后续的）
    --organism=mouse \  ##填写物种（可不填）
    --tissue=kidney \  ##填写组织or病症（可不填）
    --chip-mask=/PATH/to/B01020B5.barcodeToPos.h5    \
    --fastqs=/PATH/reads/       \ 
    --reference=/PATH/reference/mouse    \  ##mouse/human/rat供选择，修改最后的文件夹名称即可
    --local-cores=48

**SAW count无图流程输出文件为：**

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/5246ab4f-f602-4fd5-aebd-45e634e1c738.png)

### stereomap配准

1.  用stereomap打开QC后的tar.gz，进入step2后，选入.stereo文件进行手动配准
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/dea252db-e42a-4c0b-b441-f36177d361bc.png)

2.  配准前和配准后（手动配准会存在不可避免地误差）
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/078b59b9-d08a-4367-8021-30b3a3fb5a07.png) ![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/7d6e5885-797f-459b-a5ef-f2ead1b1071b.png)

3.  step3和step4跳过，在step5保存导出配准好的tar.gz
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/6456016a-4f08-44f4-b1d7-9080ab988b7a.png)

4.  生成`.tar.gz`文件和新的配准图
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8K4nyR2g0z3jqLbj/img/d93b197f-5283-4fe5-8983-f87d407619e6.png)

### SAW realign

将之前SAW count输出的结果和手动修改后得到的tar.gz接入SAW realign，重新跑一次SAW流程，输出的文件组成类型与一开始SAW count输出的文件一样

    ## 双#号注释标记的位置需要修改
    
    SN=B01020B5  ##修改为此次分析的芯片SN
    saw=/PATH/to/saw  ##使用的saw软件，如有更新及时修改，只用改saw-v8.0.0a7中a后面的数字，表示使用的内测软件版本号
    countData=/PATH/to/count ##count自动流程的输出目录
    tar=/PATH/TO/tar.tar  ##手动操作后tar存放的路径，注意不要多个在一起
    
    export LD_LIBRARY_PATH=${saw}/lib/geftools/lib:$LD_LIBRARY_PATH
    
    ${saw}/bin/saw realign \
        --id=${SN}_realigned \  ##可自行修改ID，注意不要重复
        --sn=${SN} \
        --count-data=${countData} \
       #--adjusted-distance=20 \  ##可以修改细胞修正距离，如果用户对于手动圈选结果or第三方结果非常满意时可以设置为0来关闭细胞修正步骤
        --realigned-image-tar=${tar}
       #--no-matrix  ##可以不输出矩阵及后面的分析
       #--no-report  ##可以不输出报告
    
    

### stereomap可视化查看

最后在stereomap打开outs>visualization中可视化查看最终流程结果，若查看中发现组织分割和细胞分割的结果不符合要求，则按照上述场景1中的“组织分割有问题”和“细胞分割有问题”的说明进行修改。

