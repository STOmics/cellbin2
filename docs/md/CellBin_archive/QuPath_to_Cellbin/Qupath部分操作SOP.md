# Qupath部分操作SOP

- [Qupath部分操作SOP](#qupath部分操作sop)
    - [内存设置](#内存设置)
    - [亮度设置与调节](#亮度设置与调节)
    - [橡皮擦功能](#橡皮擦功能)
    - [组织分割](#组织分割)
      - [用ImageJ导出图像。](#用imagej导出图像)
    - [细胞分割](#细胞分割)

本次SOP所使用的qupath为QuPath-0.3.2。其他版本的操作若与下文展示不一致，请自行google搜索教程。

### 内存设置

若遇到大的图像无法打开时，可以通过更改以下内存设置进行调整。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/91e8db48-282e-4e2b-bb9a-78284d307d2b.png)

根据自己的电脑条件，更改箭头所指位置的大小。例如我的电脑是32G，我将qupath最大内存设置为30G。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/b57cfba5-5e3a-4eb2-bf46-f9be09315872.png)

### 亮度设置与调节

当图像拖入qupath时，再没有设置的情况下，qupath会对图像做默认亮度增强。如果将图像保持默认亮度的情况下，可以做如下设置：

点击以下亮度设置：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/2e62f4ad-a05a-4c2c-9a1e-9b4146650ec3.png)

点击keep settings，并把Min display设置为0，Max display 设置为255（若是16bit，则设置为65535）

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/1711d035-f60b-47cf-bdaa-1c53696947cc.png)

### 橡皮擦功能

橡皮擦功能只能在使用Brush与Wand这两种画笔时使用。![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/7245e738-7313-4615-8087-6874cb0f2557.png)

使用方法为：

点击其中一种工具，在按住Alt键，即可对选中的label进行涂抹。不同工具会有不同的涂抹效果。如下所示。用户可以自己选择喜欢的效果。

### 组织分割

该种手动组织分割方法无法保留镂空，需注意！

*   对要分割的组织进行框选，如下所示：
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/2ef86e8b-8fa9-42dc-8967-c7abdbd90bc8.png)

*   点击Classify > Pixel classification > Create thresholder。
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/100aed6e-74ac-4084-8ff0-78cdbb041fba.png)

*   在弹出的窗口中可以选择适合的参数。
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/2349e3d2-ceec-4bc6-9cc1-6462894330ea.png)

当Threshold有阈值时，框选出的区域会出现蓝色的掩膜，为组织分割出来的效果。可以根据效果调整参数。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/13f37cdf-dde8-4412-9eda-97ff8a4f4f5f.png)

最后3个参数设置如下：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/6bd3c3dd-e7ee-4aca-a68d-0decc8538ef2.png)

选择Any annotations的时候，只会对框选的区域进行组织分割。

展示如下：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/0500a752-9d29-42e4-901b-f0a70f9a8c23.png)

*   点击最后三个点中的Enable buttons for unsaved classifiers.
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/317c65b4-d3da-4c69-95d4-d5ceba9bada0.png)

点击Create objsects，选择ALL annotations：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/03e41b17-55fd-4f53-813c-86f616ef7df6.png)

*   勾选Split objects和Create objects for ignored classes.
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/4a368b98-6b13-4416-80e0-72c78a44579b.png)

左侧Annotations这一列会出现组织分割的结果，Region\*为标注区域，Ignore\*为标注区域内部镂空的位置。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/2186ad59-f45b-40ed-8f1e-2d5eb7ad96ed.png)

若想手动修改已经完成的 标注，可以点击此标注，然后点击unlock。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/44ede20c-b23b-4a9e-808f-b2483a284d0a.png)

然后选择画笔在对此标注进行修改，如下所示：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/6a9a4f9c-9d5b-44fa-bbf1-2594150728c6.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/e6ba1473-3069-4820-88b8-34ae9f44bf72.png)

#### 用ImageJ导出图像。

*   修改好之后，取消任何对标注的选择。点击以下按钮。
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/ad9cf4c8-4569-44c6-b16c-d6754d36b197.png)

*   点击确认，在点击是，会出现ImageJ。
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/323eff43-55c8-46f3-86cb-4ad9c88d5a04.png)

会弹出标注好的图像与ImageJ软件。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/03749834-4833-4906-9be8-2e8545edf6c8.png)

*   出现上图时，先用方框工具点击一下任意背景区域，确定整张图没有被框选到。
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/d4d45325-1395-4a27-8d5b-fa10a5b711be.png)

*   在ImageJ选择此按钮，
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/3cb61596-dd52-4e74-acfc-0d785dd42531.png)

*   勾选Black background。
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/8495c563-be2b-4bc9-ab9b-82565ae9e2bc.png)

*   再选择此按钮中的Create Mask
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/5a0afdc8-a672-410b-bac8-4dc52c2d0fdd.png)

*   组织分割的图像最终生成：
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/b46cdd5d-aff7-4947-a38a-f7bdba160fa8.png)

### 细胞分割

*   先选择一小块区域调试组织分割参数
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/6f3ff101-f492-4159-9341-00dae1a68a49.png)

*   点击
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/09ca63ce-5271-4a6a-8916-6f68c41c7be7.png)

*   调整适合的参数，使得细胞分割满足要求，其中以下几项操作比较重要。
    

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/yBRq17B5bJzQldv1/img/42448dcf-13c7-4224-9c05-4ee3b05e940c.png)

（1）Threshold，判断为细胞核的阈值，数值越小，分割出越多的细胞核。

（2）cell expansion,细胞扩充，一般建议设置在2左右。若针对细小密集的细胞可先把该值调低，后续再进行修正

（3）sigma,针对细胞核的调整数值越高，细胞核被分的越碎。

（4）Background radius，可设置背景的范围，该值越大，细胞之间的空隙越大。也可将多个细胞合并。

（5）Minimun，Maximum area 细胞核面积上下限。

以上几项参数，建议结合调试。

对于密集模糊的细胞分割的参数建议：

先将cell expansion设置为0。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/a10ff7f8-79a0-4899-b527-038538bb15e7.png)

该参数为细胞密度，可以设置为0

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/a21cd230-a604-4af2-9ed1-3a08c0532053.png)

效果如下：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/b8279dc8-99be-4387-9286-66f4a28ecedb.png)

说明图像过于密集模糊，导致算法无法自动区分出细胞核。如果将此区域填充满的需求，此时可以再将cell expansion设置大一些，如下所示：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/43ebbe16-469b-49e0-abfe-efbc9e0fa57b.png)

以上建议只是个人经验，还是需要大家根据图像调试出合适的参数。

*   调试好参数后，删除标注，重新选择细胞分割的区域，再进行细胞分割。
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/f2e63312-55ed-4dc9-b1a0-168df0d76f79.png)

其中紫色的区域为算法视为阴性的细胞，若觉得不合理想要去除，可以用以下操作：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/751eb1d9-52cd-4af2-9ec9-57058261d0c4.png)

会全选所有的Negative细胞，然后按“delete”，即可删除。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/5886f11f-72a0-4fd3-93fe-fa2027d5b9c3.png)

选择最开始框选的label（若无，则忽略这个步骤。）

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/2e723b8b-2388-4fcc-8cc4-ee32ce8308e1.png)

然后按delete删除。会问你是否保留细胞分割结果，请点击“是”。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/d51cdee4-21f0-4f34-8f39-ca14b37c9e79.png)

最后用ImageJ导出，操作同组织分割中的**用ImageJ导出图像**。最终出现以下结果。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/2f16f4f5-63f5-4c18-b7b9-f7d19de29190.png)

若细胞之间挨得紧密，会出现以下结果，细胞之间没有完全分开：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/f00e2a2b-1f93-4085-9a49-5e049fa8152a.png)

点击ImageJ中的分水岭操作，如下：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/78a3b229-2bee-4680-bad3-eb23e7f6c58b.png)

细胞之间就会分开

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/0ef78b95-4a5e-4bf6-b338-d4e6dc21686f.png)