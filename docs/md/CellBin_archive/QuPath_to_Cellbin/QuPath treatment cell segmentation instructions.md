# QuPath treatment of cell segmentation instructions

- [QuPathDo cell segmentation operation documents](#documentation of cell segmentation operation in qupath)
- [Background introduction](#Background introduction)
- [Purpose](#Purpose)
- [Range of use](#Scope of use)
- [Introduction to tools and scripts](#Introduction to tools and scripts)
- [Operation steps](#Operation steps)
  - [Automatic segmentation](#Automatic segmentation)
    - [1. Fluorescence staining：byssDNAAs an example](#1-Fluorescence staining takes SSDNA as an example)
    - [2. H\&Edyeing](#2-he staining)
  - [Batch deletion of unwanted cell segmentationmask](#Batch delete unwanted cell segmentation mask)
    - [1. Fluorescence staining：byssDNAModify as an examplemask](#1-Fluorescence staining uses SSDNA as an example to modify mask)
    - [2. H\&EDyeing modificationmask](#2-he staining modified mask)
  - [Manual segmentation](#Manual segmentation)
    - [1. Fluorescence staining：byssDNAAs an example](#1-Fluorescence staining takes SSDNA as an example-1)
    - [2. H\&Edyeing](#2-he staining-1)

# Background introduction

Cell segmentation refers to the process of accurately segmenting the cell boundaries in microscope images. This is a very important task in the fields of biomedical image processing and computer vision, because cell segmentation can be used to study cell morphology, quantity and distribution, as well as to have wide applications in disease diagnosis, drug screening, and basic research.
In microscopic images, there are significant differences in the shape, size and color of cells, which can pose challenges to cell segmentation. To achieve accurate cell segmentation, image processing techniques such as threshold segmentation, edge detection, and region growth can be used. These methods use preprocessing and feature extraction to segment cells using specific algorithms.
If the current algorithm is not competent for complex cell segmentation tasks, this manual provides the use of the cell segmentation tool Qupath to solve cell segmentation manually.

# Purpose

Use the QuPath tool to completely and accurately segment the cell shape, correct or generate cell segmentation results.

# Scope of use

Currently, it is suitable for cell segmentation of ssDNA and HE staining. This manual will use ssDNA and HE staining diagrams respectively as examples

# Tools and script introduction

| Tool Name | Version of Use | Tool Description | Download Address |
| :-----------------: |:-----:| :----------------------------------------------------------: | :----------------------------------------------------------: |
| QuPath | 0.5.1 | Qupath Software | [Release v0.5.1 · qupath/qupath · GitHub](https://github.com/qupath/qupath/releases/tag/v0.5.1) |
| mask2geojson.py script | \ | Convert the original cell segmentation binarized image into a script tool for geojson adapted to Qupath software | [mask2geojson.py](https://github.com/STOmics/CellBin-R/blob/main/tutorials/mask2geojson.py) |
| ImageJ | \ | Picture viewing and image editing software | https://imagej.net/ij/download.html |


# Operation steps

## Automatic segmentation

### 1. Fluorescence staining: Take ssDNA as an example

* #### <li id='1'>**step1**</li>
    **Drag the image into QuPath, select Fluorescence, and click Apply**

<img src="../../images/QuPath_cellbin_SOP/img1.png" alt="img1" style="zoom:80%;" />

* #### **step2**
    **Use matrix frames to frame all the pictures (you can also select the required parts in the local frame)**

![img1](../../images/QuPath_cellbin_SOP/img2.png)

<img src="../../images/QuPath_cellbin_SOP/boxing out pictures.png" alt="Frame the picture" style="zoom:80%;" />

* #### **step3**
    **Click Analyze>Cell detection>cell detection to adjust parameters**

  ![img3](../../images/QuPath_cellbin_SOP/img3.png)

* #### **step4**
    **Adjust appropriate parameters so that cell segmentation meets the requirements**

![img4](../../images/QuPath_cellbin_SOP/img4.png)

(1) Requested pixel size: the requested pixel size. Turn it up, the actual physical area represented by each pixel becomes larger, and the details of the image look blurry, which is equivalent to viewing the image at a lower resolution. By reducing the actual physical area represented by each pixel becomes smaller, and the details of the image will be clearer, which is equivalent to viewing the image at a higher resolution.

(2) Background radius: background radius. When the cells are large or densely packed, too large will lead to the incorrect identification of background noise as part of the cell. When the cells are small or the background noise is high, the background changes around the cells cannot be accurately captured.

(3) Median filter radius: median filter radius, adjusting the filtering radius will increase the filtering radius and smooth the image more effectively. If the setting is too large, it may cause the image to be lost and the cell edges may become blurred. Downsizing reduces the filtered neighborhood range and retains more details of the image. If the radius is too small, it may not be enough to remove noise in the image.

(4) Sigma: Adjustment for cell nucleus, the lower the value, the more fragmented the cells are.

(5) Minimum area: the size of the smallest cell or tissue area that will be retained during cell detection. If the detected areas are less than this set value, these areas will be ignored or discarded. Increased values will allow larger cells or tissue regions to be detected and analyzed.

(6) Maximum area: the area of the largest "hole" or gap that will be automatically filled during cell detection. If the void in an area is greater than this set value, the void will not be automatically filled, but will be retained as a separate area. The value reduction can allow smaller holes to be filled.

(7) Threshold: The threshold value of the cell nucleus is determined. The smaller the value, the more cells are divided.

(8) Cell expansion: Cell expansion, the smaller the value, the more the cells are.

The effects are as follows:

<img src="../../images/QuPath_cellbin_SOP/ssDNA parameter regulation example.png" alt="Example of ssDNA parameter regulation" style="zoom:80%;" />

* #### **step5**
    **Delete matrix box**

  <img src="../../images/QuPath_cellbin_SOP/Delete matrix box.png" alt="Delete the matrix box" style="zoom:80%;" />

* #### <li id='6'>**step6**</li>
    **Send the image to ImageJ**

![img5](../../images/QuPath_cellbin_SOP/img5.png)



* #### **step7**
    **Cancel the first item and click OK**

![cancel_first](../../images/QuPath_cellbin_SOP/cancel_first.png)

* #### **step8**
    **Click Edit>Selection>Create Mask to generate mask map**

  <img src="../../images/QuPath_cellbin_SOP/create_mask.png" alt="create_mask" style="zoom:80%;" />

* #### **step9（optional）**
    **Set the watershed**

If the cells are close together, there will be no complete separation between the cells.

<img src="../../images/QuPath_cellbin_SOP/Set the watershed.png" alt="Set up a watershed" style="zoom: 50%;" />

Click Process>Binary>Watershed

<img src="../../images/QuPath_cellbin_SOP/Watershed Steps.png" alt="Watershed Steps" style="zoom:80%;" />

The cells will separate

<img src="../../images/QuPath_cellbin_SOP/Watershed Legend.png" alt="Watershed legend" style="zoom:50%;" />

* #### **step10**
    **Click File>Save As>Tiff to save mask in tiff format**

![save_image](../../images/QuPath_cellbin_SOP/save_image.png)

### 2. H&E staining

2.1 [Same as above。](#1)Select H&E staining when importing the picture in the first step, click Apply

<img src="../../images/QuPath_cellbin_SOP/HE import.png" alt="HE import" style="zoom:50%;" />

2.2 Example of H&E automatic segmentation parameters:

<img src="../../images/QuPath_cellbin_SOP/HE parameter example.png" alt="HE parameter example" style="zoom: 80%;" />

## Batch delete unwanted cell segmentation mask

### 1. Fluorescence staining: Modify mask using ssDNA as an example

This method is used to modify only locally unsatisfactory cell segmentation results, and ImageJ is required. The operation is as follows:

* #### **<li id='2'>step1</li>**
  **Drag the mask image and the original image into ImageJ, select an image, click Image>Type to ensure that the position depths of the two images are the same**

<img src="../../images/QuPath_cellbin_SOP/Set bit depth.png" alt="Set the bit depth" style="zoom:80%;" />

* #### **step2**
  **Click Color>Merge Channels to combine the mask image with the original image**

<img src="../../images/QuPath_cellbin_SOP/merge_mask.png" alt="merge_mask" style="zoom: 80%;" />		

* #### **step3**
  **Select the appropriate color between the mask image and the original image. It is recommended to choose red for the mask image and gray for the original image. Select Keep source images to keep the source image, click OK to observe the area that needs to be deleted**

<img src="../../images/QuPath_cellbin_SOP/Set merge color.png" alt="Set merge color" style="zoom:80%;" />

* #### **step4**
  ** Use the appropriate tool to frame the parts that need to be deleted, and press the keyboard Delete to delete them**

  ![delete_useless](../../images/QuPath_cellbin_SOP/delete_useless.png)

  <img src="../../images/QuPath_cellbin_SOP/delete_before.png" alt="delete_before" style="zoom: 40%;" />

  <img src="../../images/QuPath_cellbin_SOP/delete_after.png" alt="delete_after" style="zoom:40%;" />

* #### **step5**
  **After the operation is completed, click Image>Color>Split Channels to separate the merge image**
  
  <img src="../../images/QuPath_cellbin_SOP/split_channels.png" alt="split_channels" style="zoom:67%;" />

* #### **step6**
  **Return the image back to the 8bit image and save it**
  
  <img src="../../images/QuPath_cellbin_SOP/ to 8-bit deep.png" alt="Transform to 8-bit deep" style="zoom:40%;" />

* #### **step7(optional)**
  **Modify the saved mask diagram further**

  This method is used to convert the modified cell segmentation results into geojson and fill in the QuPath with new cell segmentation. Use mask2geojson.py to modify the path in the script using the above **"Introduction to Tools and Scripts".
generategeojsonback，OpenQupath，Sequentially transfer the images、geojsonDrag into the frame，Follow the above[Automatic segmentation](#Automatic segmentation) method, use Qupath to fill in the newly added cell segmentation finally generates a cell segmentation mask image that meets the expectations.

### 2. H&E staining modification mask
[Same operationssDNA](#2)


## Manual segmentation

### 1. Fluorescence staining: Take ssDNA as an example

* #### **<li id='3'>step1</li>**
  **Import the image into QuPath, select Fluorescence, and click Apply**

* #### **step2**
  **Split with the right tools**

(1) Use the brush tool

[Video link](../../images/QuPath_cellbin_SOP/brushes.mp4)

Press and hold the left mouse button to expand the mask range

Press and hold alt and press the left mouse button to narrow the scope from outside the mask

(2) Use the polygon tool

[Video link](../../images/QuPath_cellbin_SOP/polygon.mp4)

Click the generated node with the left mouse button, connect to the last node and double-click to complete the box selection

(3) For multi-points, press the right mouse button to select Delete after selecting

![img7](../../images/QuPath_cellbin_SOP/img7.png)

(4) Modify mask on mobile node

![img8](../../images/QuPath_cellbin_SOP/img8.png)

Double-click to select

<img src="../../images/QuPath_cellbin_SOP/img6.png" alt="img6" style="zoom:50%;" />

Drag the node to modify the mask

[Video link](../../images/QuPath_cellbin_SOP/grag_node.mp4)

* #### After the segmentation is completed, please return [step6 in automatic segmentation](#6)

### 2. H&E staining

[Operation andssDNAsame](#3)
