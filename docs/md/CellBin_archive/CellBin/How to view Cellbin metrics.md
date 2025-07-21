# How to view Cellbin metrics

- [How to viewCellbinindex](#How to view cellbin metrics)
  - [**Image indicator check：**](#Image Indicator Check)
  - [**Genetic indicator examination：**](#Gene indicator check)
  - [**View image details**](#View image details)

After the data is completed, a statistical report will be generated, and SN.report.tar.gz is placed in outs of the result directory, decompress and output report.html, which is a statistical report.

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/20e59f76-9deb-474c-a49e-92e958a0b1f6.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/659859e1-2017-493f-ac02-0ab3cfade4c1.png)

## **Image metrics check: **

**Registration: **

Registration status can be viewed in the "Summary" page in the report, as shown below:

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/aed165d7-5e54-4fe8-93a0-7e71adee9b18.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/0b3c608c-be1c-48dc-968e-3e8ddae9e8a8.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/baf123d9-cc06-48f1-9fbf-8853667b155b.png)

You can determine whether the image map matches the gene visualization map by pulling the transparency adjustment below, as shown in the upper right. If it matches, it means that the location of the image space is roughly matched.

**Organization segmentation: **

Organizational segmentation can be viewed in the "Tissue Segmentation" in the "Summary" page, as shown below:

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/abe4a815-8cb5-40d5-93b9-3e24cd21f614.png)

The module is divided into two parts. The gray color in the image on the left is the image and the purple area is the result of tissue segmentation. You can use the mouse to slide and enlarge the image to observe whether the tissue segmentation is correct. If the purple-divided area meets your own analysis requirements, you can use this result. If it does not match, you need to use the Stereomap v4 tool to reorganize and split manually.

The right part is the genetic indicators under the tissue segmentation results. Generally, you need to pay attention to the "Fraction MID in Spots Under Tissue" indicator. If Fraction MID in Spots Under Tissue <50%, you need to carefully check whether there are any abnormalities in the data.

## **Genetic indicator check: **

After confirming that there are no abnormalities in registration and organizational segmentation, check the relevant statistical indicators of Tissue bin. If there is any problem, it needs to be handled manually later.

1. **View Tissue bin gene metrics in the report page "Square Bin"**:
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/8166da25-9734-47b1-b306-12f2fc524f6a.png)

General suggestions:

*   bin200 Median MID > 5000
    
*   bin20 Median gene type > 200
    

1.  **exist"Cell Bin"View page Cellbin index**：
    

CellbinThe indicators can be found in the statistical report"Cell Bin"View page，As shown below：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/7ba1d1e4-e3c6-4851-b966-ca2c079cc3ba.png)

The indicators on this page show the gene statistics of ** modified cell segmentation**. Based on the following indicators, it is preliminary to determine whether cell segmentation meets expectations:

*   **Cell Count**
    
* **Cell Area Indicators**
    
* **Cell Area distribution**
    

cell area distributionAvailable in"Cell Bin"View page，As shown below：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/e21fe9b6-d550-4bc4-be44-844b1d427a3d.png) ![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/13bf57c5-8ce1-4ff5-84ae-635151ac672d.png)

Based on the tissue situation, users will make preliminary judgments on whether the number of cells, area of cells, etc. are in line with expectations. If there is any abnormality or no confirmation, further visual viewing is required. If there are no major exceptions, you can continue to view.

* **CellBin Median Gene Type Is it higher than Bin20 Median Gene Type**
    

Generally speaking, the number of CellBin genes will be higher than bin20, but some cases are not satisfied. For example, if the Cell area is less than bin20, this can calculate the number of genes per unit area. In addition, diffusion and abnormal cell segmentation can also cause this situation to not meet expectations.

* **Cell total MID/tissue total MID>50% (the total MID of cellbins in the organizational coverage area accounts for >50% of the MID of the total MID of the organizational coverage area)**, the calculation method is as follows:
    

Cell total MID/tissue total MID = Cell Mean MID × Cell Count / Number of MID Under Tissue Coverage

* **Cellbin's indicators are better than bin20's indicators (such as gene number > 200, MID number), such as cluster annotation is better than bin20**; the indicator situation can be compared with the indicators on the report, as shown below:
    

1. Indicator comparison:
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/07e60501-d656-4243-a199-fd0ad747b689.png) ![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/6dd29ea5-544c-4413-8beb-4c2d549229d5.png)

2. Cluster comparison:
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/cd2dab67-559d-419f-95eb-5af8382da324.png)

There is no cluster display of bin20 in the report, only bin200 is provided. If the user feels that this part is of comparison through the above observations, he can use the stereopy tool to generate bin20 clustering and compare it with the cellbin clustering results in the report.

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/c002cdeb-aef8-40f3-aaff-597906670853.png)

After a preliminary evaluation of the overall results, if you want to further confirm, you can use StereoMap software to visualize it.

## **View image details**

Image details can be viewed in the visualization function of StereoMap, as follows:

Unzip the visualization.tar.gz file in the outs folder of the process results. After decompression, the result is as follows:

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/7a1a25a0-4c29-4136-9c99-561baf1830d4.png)

Open Visual Explore Visualization in StereoMap4.0.0 to view visual results.

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/b3c65286-9f00-4956-9cca-5920a0b5bc6b.png)

It is recommended to observe the following parts:

**1. Whether the registration reaches cell-level accuracy: **

Open Image, and template points, and close "Gene Heatmap", as shown below:

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/344ef670-f64b-42e7-9bbc-22ee69317759.png)

Adjust the image brightness to make the chip background clear, as shown below:

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/6ddccbca-7059-4060-806a-a91cfaf61374.png)

If the yellow dots fall on the track point of the chip, or the farthest distance is less than 10 pixels (can be measured by the ruler next to the template! [image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/f6e5896a-8fb1-467c-a353-30d4f999a161.png). For 5um, take mouse brain cells as an example, half a cell), the registration reaches cell-level accuracy, otherwise manual reregistration is required.

**2. Whether the organizational division meets the requirements: **

Open Image, and TissueMask as shown below. Observe whether the organizational segmentation meets your expectations.

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/b9f99b7d-c9c9-4ea7-9365-7a7c5d3cb280.png)

**3. Whether cell segmentation meets the requirements: **

Close Gene Heatmap, open Image and CellMask\_adjusted, and change CellMask\_adjusted to other colors other than white, as shown below. You can view the effect of cell segmentation.

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonaA4QW1PjnXxj/img/c4169922-e469-46f5-b93f-7590385edc82.png)

Check whether the segmentation of the region of interest meets the analysis requirements, whether there are omissions, multiple points, and missed points. If the requirements do not meet the requirements, you can correct the cell segmentation results in Image Processing in Stereomap, and then run the SAW process again to obtain new process results.
