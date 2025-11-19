# CellBin解决方案技术说明


## **引言**

本文档分为两大部分：“Cellbin的介绍说明”，“Cellbin的使用方法”，旨在帮助用户更好的理解“Cellbin”，并提供获得细胞精度级别的基因矩阵的解决方案。使用此文档的用户需要提前获得测序数据以及生物影像图。

## **CellBin**


<details open>
<summary style="font-size: 15pt;">CellBin介绍</summary>


空间组学（spatial omics）可以在连续的空间维度对组织和细胞探测生命体多组学的表达和调控特征，近些年来发展迅速，且应用广泛。其中，华大发布的stereo-seq具有高分辨率、超大视场的优势，可以助力构建脑科学的空间图谱、挖掘生物体发育和再生的时空景观、揭示发病机制的分子分析等，截止9月发表研究性论文82篇，综述23篇，预印61篇。

stereo-seq为了深入解析组织结构和功能关系提供了新的视角，但解读数据十分依赖针对性的方法。从信息维度上讲，stereo-seq可以延续了单细胞的分析逻辑的同时，加入空间坐标维度，可以解读出整体和局部空间区域内多种细胞个体及个体之间的关联信息。然而，stereo-seq的分子检测不以细胞为基本单元，空间转录组数据必须借助算法来生成细胞意义上的转录组信息。华大发布的基于Stereo-seq系列的CellBin一站式细胞分割产品方案，借助同芯片的图像（ssDNA、DAPI、H&E）信息，可以直接拿到准确的单细胞表达谱信息。

</details>


<details>
<summary style="font-size: 15pt;">CellBin框架说明</summary>


CellBin是一套将时空测序数据与生物影像图相结合，最后获得细胞级别精度的基因矩阵数据的流程，它嵌入在StereoMap和SAW流程当中，关系如下：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1X3lE63B7m04nJbv/img/335efa9d-3b54-4d7f-8948-c026929451e3.png)

CellBin流程包含QC，拼接，配准，组织分割，细胞分割，细胞修正，生成cellbin矩阵这几个模块。其中QC模块与图像手动处理部分（详见《CellBin解决方案技术说明》中的QC介绍）放在StereoMap v4软件中执行，剩余模块在Stereo-seq Analysis Workflow (SAW) 流程中执行。

**StereoMap v4**:  StereoMap 是一款无需编程的桌面端软件，包括 Stereo-seq 数据交互式的可视化、图像的手动处理及小工具三个模块，支持用户在本地电脑桌面端进行几十块数据可视化和简单的探索分析。交互式可视化模块支持多组学、多模态时空数据的展示和基础数据挖掘；图像手动处理模块迁移ImageStudio部分图像处理功能，修改为 Step by step 的方式，降低用户的学习成本，以便为流程化的方式提供图像处理功能。小工具中的 QC 评估图像是否可以调用 SAW 自动的图像分析算法。

**Stereo-seq Analysis Workflow (SAW) ：** 时空标准分析流程软件（SAW）是一套捆绑式pipelines，时空标准分析流程软件，核心是将Stereo-seq测序FASTQs的reads定位到其在组织切片上的空间位置，量化基因表达，直观呈现其在空间上的分布。同时提供辅助小工具，支持用户更方便进入下游分析。

**QC**：对生物影像图中拍摄到的芯片 track 线进行质检，判断是否达到后续自动化配准模块要求

**拼接（可选）**：如果提供的影像图为拼接好的大图，则跳过这一步。若提供的影像图为显微镜小图，则会将小图拼接成一张大图。

**配准**：基于track线的高精度自动化配准分为两部分：第一部分：有 track 且 QC 通过的情况下，通过track线将计算出图像相对表达矩阵的缩放尺度，以及旋转角度；第二部分：拼接好的影像大图与测序下机后的基因生成的基因可视化图（gene）的做形态学上配准，得到90度倍数的旋转或是翻转，以及超过位移参数。

**组织分割**：对影像图中的生物切片的组织轮廓做分割。

**细胞分割** ：对影像图中的生物切片的细胞核/膜轮廓做分割。

**细胞修正** ：将分割后的细胞轮廓往外扩大10pixels（默认），作为新的细胞分割区域结果。

以上算法模块的详细介绍可以看：[https://www.biorxiv.org/content/10.1101/2023.02.28.530414v5](https://www.biorxiv.org/content/10.1101/2023.02.28.530414v5)

</details>


<details>
<summary style="font-size: 15pt;">CellBin自动化使用</summary>



先使用StereoMap进行QC，后使用SAW，执行SAW count可得到CellBin结果

以**ssDNA Demo：SS200000135TL\_D1** 为例：

**数据链接：**

[http://116.6.21.110:8090/share/21bb9df9-e6c5-47c5-9aa8-29f2d23a6df4](http://116.6.21.110:8090/share/21bb9df9-e6c5-47c5-9aa8-29f2d23a6df4)

下载 **SS200000135TL_D1_v5_mouse_brain_Stereo-seq_T_FF** 整个文件夹；image是QC输入，mask、reads和reference是SAW的输入。

*   **如何执行QC：**

如何使用Stereomap 中的QC功能，可阅读操作文档：[https://stereotoolss-organization.gitbook.io/stereomap-user-manual/tutorials/navigation-for-tools/stereo-seq-image-qc](https://stereotoolss-organization.gitbook.io/stereomap-user-manual/tutorials/navigation-for-tools/stereo-seq-image-qc)

如何使用Stereomap v4中的其他功能，可阅读操作文档：[https://stereotoolss-organization.gitbook.io/stereomap-user-manual/tutorials/navigation](https://stereotoolss-organization.gitbook.io/stereomap-user-manual/tutorials/navigation)
*    **如何判断QC结果：**

**QC失败：** 在QC时如果软件界面出现以下情况 “The results of this image analysis evaluation are: FAIL”，则表示QC失败。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1X3lE63B7m04nJbv/img/30b658a3-fca5-488e-b164-a692402f1eed.png)

若图像QC失败，则只能执行SAW count无图流程，即不使用_**"--image-tar "**_ 参数。SAW流程结束后，若需要图像结果，可用Stereomap V4进行手动处理。详情请查看SOP：[《Cellbin流程手动操作》](../CellBin_archive/CellBin/Cellbin流程手动操作.md)-场景2:QC失败

**QC成功：** 在QC时如果软件界面出现以下情况 “The results of this image analysis evaluation are: **PASS**”，则表示QC成功。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1X3lE63B7m04nJbv/img/51d6c550-ef6f-4f54-85b9-e23fc3bdb1ba.png)

若图像QC成功，需要跑SAW count有图流程，即增加_**"--image-tar "** 参数。

*   **如何执行SAW count：**
    

如何使用SAW count，可阅读操作文档：[https://stereotoolss-organization.gitbook.io/saw-user-manual-v8.0/analysis/pipelines/saw-commands](https://stereotoolss-organization.gitbook.io/saw-user-manual-v8.0/analysis/pipelines/saw-commands)

如何使用SAW中的其他功能，可阅读详细操作文档：[https://stereotoolss-organization.gitbook.io/saw-user-manual-v8.0](https://stereotoolss-organization.gitbook.io/saw-user-manual-v8.0)


</details>


<details>
<summary style="font-size: 15pt;">如何查看CellBin结果</summary>




跑完SAW count后，流程结果如下：

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1X3lE63B7m04nJbv/img/2581b561-2392-4518-8667-5de32716e13e.png)


下载outs文件夹中的“SN.report.tar.gz”到本地解压打开。 ![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1X3lE63B7m04nJbv/img/5f6c957e-e5ac-439b-8669-909e379e73de.png)

文件夹中的“report.html”文件为本次流程结果的统计报告，打开报告后可查看CellBin结果。 详情请查看此文档[《如何查看Cellbin指标》](../CellBin_archive/CellBin/如何查看Cellbin指标.md)

</details>


<details>
<summary style="font-size: 15pt;">手动处理方案</summary>



了解CellBin指标后，如果在上述查看CellBin结果中遇到QC失败/配准异常/组织分割异常/细胞分割异常等图像问题，可使用StereoMap中的手动处理工具进行修改，再次接回SAW流程，得到符合预期的结果，具体操作请看SOP：[《Cellbin流程手动操作》](3.Cellbin流程手动操作.md)

细胞分割模块也可使用第三方工具进行修改，再次接回SAW流程，得到符合预期的结果，具体操作请查看SOP：[《QuPath处理细胞分割操作说明》](../CellBin_archive/QuPath_to_Cellbin/QuPath处理细胞分割操作说明.md)


</details>
