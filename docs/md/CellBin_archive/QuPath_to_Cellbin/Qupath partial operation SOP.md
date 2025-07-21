# Qupath partial operation SOP

- [QupathPartial operationsSOP](#qupath partial operation sop)
    - [Memory settings](#Memory settings)
    - [Brightness setting and adjustment](#Brightness setting and adjustment)
    - [Eraser function](#Eraser function)
    - [Organizational segmentation](#Organization segmentation)
      - [useImageJExport images。](#Export images using imagej)
    - [Cell segmentation](#Cell segmentation)

The qupath used in this SOP is QuPath-0.3.2. If the operations of other versions are inconsistent with the following display, please search the tutorial yourself.

### Memory settings

If you encounter a large image that cannot be opened, you can adjust it by changing the following memory settings.

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/91e8db48-282e-4e2b-bb9a-78284d307d2b.png)

According to your computer conditions, change the size of the position pointed by the arrow. For example, my computer is 32G, and I set the maximum qupath memory to 30G.

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/b57cfba5-5e3a-4eb2-bf46-f9be09315872.png)

### Brightness setting and adjustment

When the image is dragged into qupath, if there is no setting, qupath will make the default brightness enhancement to the image. If you keep the image brightness under default, you can make the following settings:

Click the following brightness settings:

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/2e62f4ad-a05a-4c2c-9a1e-9b4146650ec3.png)

Click keep settings, and set Min display to 0 and Max display to 255 (if it is 16 bit, it is set to 65535)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/1711d035-f60b-47cf-bdaa-1c53696947cc.png)

### Eraser function

The eraser function can only be used when using two brushes: Brush and Wand. ![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/7245e738-7313-4615-8087-6874cb0f2557.png)

How to use it is:

Click one of the tools and hold down the Alt key to apply the selected label. Different tools will have different smear effects. As shown below. Users can choose their favorite effect.

### Organizational segmentation

This manual tissue segmentation method cannot retain hollowing, so you need to pay attention!

*Block the organization to be divided as follows:
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/2ef86e8b-8fa9-42dc-8967-c7abdbd90bc8.png)

* Click Classify > Pixel classification > Create thresholder.
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/100aed6e-74ac-4084-8ff0-78cdbb041fba.png)

* In the pop-up window, you can select the appropriate parameters.
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/2349e3d2-ceec-4bc6-9cc1-6462894330ea.png)

When Threshold has a threshold, a blue mask will appear in the area selected in the box, which will be the effect of segmenting the tissue. The parameters can be adjusted according to the effect.

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/13f37cdf-dde8-4412-9eda-97ff8a4f4f5f.png)

The last 3 parameters are set as follows:

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/6bd3c3dd-e7ee-4aca-a68d-0decc8538ef2.png)

When selecting Any annotations, only the selected areas will be organized and divided.

Displayed as follows:

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/0500a752-9d29-42e4-901b-f0a70f9a8c23.png)

* Click on Enable buttons for unsaved classes in the last three points.
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/317c65b4-d3da-4c69-95d4-d5ceba9bada0.png)

Click Create objectives and select ALL annotations:

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/03e41b17-55fd-4f53-813c-86f616ef7df6.png)

* Check Split objects and Create objects for ignored classes.
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/4a368b98-6b13-4416-80e0-72c78a44579b.png)

The Annotations column on the left will have the result of organizational division. Region\* is the label area, and Ignore\* is the hollow position inside the label area.

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/2186ad59-f45b-40ed-8f1e-2d5eb7ad96ed.png)

If you want to manually modify the completed label, you can click this label and then click unlock.

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/44ede20c-b23b-4a9e-808f-b2483a284d0a.png)

Then select the brush to modify this annotation, as shown below:

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/6a9a4f9c-9d5b-44fa-bbf1-2594150728c6.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/e6ba1473-3069-4820-88b8-34ae9f44bf72.png)

#### Export images using ImageJ.

* After modification, cancel any selection of markup. Click the following button.
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/ad9cf4c8-4569-44c6-b16c-d6754d36b197.png)

* Click to confirm, and when clicking Yes, ImageJ will appear.
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/323eff43-55c8-46f3-86cb-4ad9c88d5a04.png)

The marked image and ImageJ software will pop up.

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/03749834-4833-4906-9be8-2e8545edf6c8.png)

* When the above picture appears, first use the box tool to click on any background area to confirm that the entire picture is not selected by the box.
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/d4d45325-1395-4a27-8d5b-fa10a5b711be.png)

* Select this button in ImageJ,
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/3cb61596-dd52-4e74-acfc-0d785dd42531.png)

* Check Black background.
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/8495c563-be2b-4bc9-ab9b-82565ae9e2bc.png)

* Select Create Mask in this button again
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/5a0afdc8-a672-410b-bac8-4dc52c2d0fdd.png)

* The final generation of the tissue segmented image:
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/b46cdd5d-aff7-4947-a38a-f7bdba160fa8.png)

### Cell segmentation

* First select a small area to debug the organization segmentation parameters
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/6f3ff101-f492-4159-9341-00dae1a68a49.png)

* Click
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/09ca63ce-5271-4a6a-8916-6f68c41c7be7.png)

* Adjust appropriate parameters so that cell segmentation meets the requirements. The following operations are more important.
    

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/yBRq17B5bJzQldv1/img/42448dcf-13c7-4224-9c05-4ee3b05e940c.png)

(1) Threshold, determines the threshold value of the cell nucleus. The smaller the value, the more nuclei are divided.

(2) cell expansion, cell expansion, generally recommended to set it at around 2. If you are targeting small and dense cells, you can first lower the value and then correct it later.

(3) sigma, the higher the adjustment value for the cell nucleus, the more fragmented the nucleus is divided.

(4) Background radius, you can set the range of the background. The larger the value, the larger the gap between cells. Multiple cells can also be combined.

(5) Minimun, Maximum area The upper and lower limits of the nuclear area of the cell.

The above parameters are recommended in combination with debugging.

Parameters for densely fuzzy cell segmentation are recommended:

First set cell expansion to 0.

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/a10ff7f8-79a0-4899-b527-038538bb15e7.png)

This parameter is cell density, which can be set to 0

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/a21cd230-a604-4af2-9ed1-3a08c0532053.png)

The effects are as follows:

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/b8279dc8-99be-4387-9286-66f4a28ecedb.png)

This shows that the image is too dense and blurry, which causes the algorithm to be unable to automatically distinguish the nucleus. If you fill in the needs of this area, you can set the cell expansion larger at this time, as shown below:

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/43ebbe16-469b-49e0-abfe-efbc9e0fa57b.png)

The above suggestions are just personal experience, and you still need to debug appropriate parameters based on the image.

* After debugging the parameters, delete the annotation, reselect the area for cell segmentation, and then perform cell segmentation.
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/f2e63312-55ed-4dc9-b1a0-168df0d76f79.png)

The purple area is a cell that the algorithm considers negative. If you think it is unreasonable and want to remove it, you can use the following operations:

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/751eb1d9-52cd-4af2-9ec9-57058261d0c4.png)

All Negative cells will be selected, and then press "delete" to delete.

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/5886f11f-72a0-4fd3-93fe-fa2027d5b9c3.png)

Select the label selected in the initial box (if none, ignore this step.)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/2e723b8b-2388-4fcc-8cc4-ee32ce8308e1.png)

Then press delete to delete. If you will ask whether you retain the cell segmentation results, please click "Yes".

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/d51cdee4-21f0-4f34-8f39-ca14b37c9e79.png)

Finally, use ImageJ to export the image. The following results finally appear.

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/2f16f4f5-63f5-4c18-b7b9-f7d19de29190.png)

If the cells are close together, the following results will occur, and the cells are not completely separated:

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/f00e2a2b-1f93-4085-9a49-5e049fa8152a.png)

Click the watershed operation in ImageJ, as follows:

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/78a3b229-2bee-4680-bad3-eb23e7f6c58b.png)

The cells will separate

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJdkeM6Wgn3p8/img/0ef78b95-4a5e-4bf6-b338-d4e6dc21686f.png)