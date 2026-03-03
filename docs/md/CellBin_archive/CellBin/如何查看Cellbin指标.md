# 如何查看Cellbin指标

- [如何查看Cellbin指标](#如何查看cellbin指标)
  - [**图像指标检查：**](#图像指标检查)
  - [**基因指标检查：**](#基因指标检查)
  - [**图像细节查看**](#图像细节查看)

数据在跑完SAW流程之后，会生成一份统计报告，放在结果目录的outs中的SN.report.tar.gz，将其解压输出report.html，为统计报告。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/20e59f76-9deb-474c-a49e-92e958a0b1f6.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/659859e1-2017-493f-ac02-0ab3cfade4c1.png)

## **图像指标检查：**

**配准：**

配准情况可在报告中的“Summary”分页中查看，如下所示：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/aed165d7-5e54-4fe8-93a0-7e71adee9b18.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/0b3c608c-be1c-48dc-968e-3e8ddae9e8a8.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/baf123d9-cc06-48f1-9fbf-8853667b155b.png)

可通过拉动下面透明度调节来判断影像图与基因可视化图的是否存在吻合，如右上所示。若吻合，则说明图像空间位置上大致匹配。

**组织分割：**

组织分割情况可在“Summary”分页中的“Tissue Segmentation”查看，如下所示：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/abe4a815-8cb5-40d5-93b9-3e24cd21f614.png)

该模块分为两部分，左边图像中灰色为影像图，紫色区域为组织分割的结果。可用鼠标在图像上滑动放大，观测组织分割是否分割正确。若紫色分割的区域符合自己的分析要求，则可以使用该结果。若不符合，则需要用Stereomap v4工具重新进行手动组织分割。

右边部分为该组织分割结果下的基因指标，一般需要关注“Fraction MID in Spots Under Tissue”指标，若Fraction MID in Spots Under Tissue <50%，则需再仔细检查数据是否有异常。

## **基因指标检查：**

在确认配准和组织分割无异常后，进行Tissue bin相关统计指标查看。若有问题，则需后续手动处理。

1.  **在报告分页“Square Bin”中查看Tissue bin 基因指标**：
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/8166da25-9734-47b1-b306-12f2fc524f6a.png)

一般建议:

*   bin200 Median MID > 5000
    
*   bin20 Median gene type > 200
    

1.  **在"Cell Bin"分页查看 Cellbin 指标**：
    

Cellbin指标可在统计报告中的"Cell Bin"分页查看，如下所示：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/7ba1d1e4-e3c6-4851-b966-ca2c079cc3ba.png)

该页面的指标展示的是**修正后的细胞分割**的基因统计结果。根据以下指标初步判断细胞分割是否符合预期：

*   **Cell Count**
    
*   **Cell Area指标**
    
*   **Cell Area distribution**图
    

cell area distribution可在"Cell Bin"分页查看，如下所示：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/e21fe9b6-d550-4bc4-be44-844b1d427a3d.png) ![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/13bf57c5-8ce1-4ff5-84ae-635151ac672d.png)

用户基于组织的情况了解，对细胞数目，细胞的面积等初步判断是否符合预期。若有异常或不确认，则需进一步可视化查看。若无大异常，则可继续查看。

*   **CellBin Median Gene Type 是否高于 Bin20 Median Gene Type**
    

一般情况，CellBin的基因数会高于bin20，但也有部分情况不满足。比如Cell area小于bin20，这种可计算单位面积下的基因数。另外，扩散，细胞分割异常，也会导致该情况不符合预期。

*   **Cell total MID/tissue total MID>50%(组织覆盖区域内的cellbin的MID总和占总组织覆盖区域MID比例>50%)**，计算方式如下：
    

Cell total MID/tissue total MID = Cell Mean MID × Cell Count / Number of MID Under Tissue Coverage

*   **Cellbin的指标优于bin20的指标(如基因数>200、MID数)，如聚类注释优于bin20**；指标情况可通过报告上的指标进行对比，如下所示：
    

1.  指标对比：
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/07e60501-d656-4243-a199-fd0ad747b689.png) ![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/6dd29ea5-544c-4413-8beb-4c2d549229d5.png)

2.  聚类对比：
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/cd2dab67-559d-419f-95eb-5af8382da324.png)

报告中没有没有提供bin20的聚类展示，只提供了bin200的。该部分若用户通过上述的观察觉得有比较的意义，可以用stereopy工具生成bin20的聚类，与报告中的cellbin聚类结果进行比较。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/c002cdeb-aef8-40f3-aaff-597906670853.png)

对整体结果有个初步的评估后，若还想要进一步确认，则可使用StereoMap软件进行可视化。

## **图像细节查看**

图像细节可以在StereoMap的可视化功能中查看，操作如下：

在流程结果的outs文件夹中解压visualization.tar.gz文件，解压后得到结果如下：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/7a1a25a0-4c29-4136-9c99-561baf1830d4.png)

打开StereoMap4.0.0中的Visual Explore可视化查看可视化结果。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/b3c65286-9f00-4956-9cca-5920a0b5bc6b.png)

建议观察以下几个部分：

**1、配准是否达到细胞级别精度：**

打开Image，和模板点，关闭“Gene Heatmap”，如下所示：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/344ef670-f64b-42e7-9bbc-22ee69317759.png)

调节图像亮度，使得芯片背景清晰，如下所示：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/6ddccbca-7059-4060-806a-a91cfaf61374.png)

如果黄色的点都落在了芯片的track点上，或者最远距离小于10pixels（可用模板旁边的尺子![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/f6e5896a-8fb1-467c-a353-30d4f999a161.png) 测量。对于5um，以鼠脑细胞为例，半个细胞），则配准达到细胞级别精度，否则需要手动重新配准。

**2、组织分割是否符合要求：**

打开Image，和TissueMask，如下所示。观察组织分割是否符合自己的预期。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/b9f99b7d-c9c9-4ea7-9365-7a7c5d3cb280.png)

**3、细胞分割是否符合要求：**

关闭Gene Heatmap，打开Image和CellMask\_adjusted，并对CellMask\_adjusted更改成除白色之外的其他颜色，如下所示。可以查看细胞分割的效果。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/c4169922-e469-46f5-b93f-7590385edc82.png)

检查感兴趣的区域分割是否满足分析的要求，是否存在遗漏，多分，错分的情况，若不符合要求，则可以在Stereomap中的Image Processing中修正细胞分割结果，再重跑SAW流程，获得新的流程结果。
