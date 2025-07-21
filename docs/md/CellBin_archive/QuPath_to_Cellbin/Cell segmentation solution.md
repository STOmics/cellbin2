# Cell segmentation solutions (ssDNA, DAPI, H&E)—Qupath operating instructions

- [Cell segmentation solutions（ssDNA,DAPI,H\&E）——QupathOperating Instructions](#Cell segmentation solution ssdnadapihequpath operating instructions)
  - [SOPUses](#sop's purpose)
    - [Background introduction：](#Background introduction)
    - [Purpose：](#Purpose)
    - [Range of use：](#Scope of use)
  - [Introduction to tools and scripts](#Introduction to tools and scripts)
  - [Operation steps](#Operation steps)
    - [QupathBasic Operation](#qupath basic operation)
    - [Different dyeing treatment methods：](#Different dyeing treatment methods)
      - [-- ssDNA/DAPI](#---ssdnadapi)
      - [-- H\&E](#---he)
    - [Batch deletion of unwanted cell segmentationmask：](#Batch delete unwanted cell segmentation mask)
    - [Put the cellsmaskConvert to put it inQupathofgeojson：](#Convert cell mask into geojson that can be placed in qupath)

## Uses of SOP
### Background introduction:
Cell segmentation refers to the process of accurately segmenting the cell boundaries in microscope images. This is a very important task in the fields of biomedical image processing and computer vision, because cell segmentation can be used to study cell morphology, quantity and distribution, as well as to have wide applications in disease diagnosis, drug screening, and basic research. <br>
In microscopic images, there are significant differences in the shape, size and color of cells, which can pose challenges to cell segmentation. To achieve accurate cell segmentation, image processing techniques such as threshold segmentation, edge detection, and region growth can be used. These methods use preprocessing and feature extraction to segment cells using specific algorithms. <br>
If the current algorithm is not competent for complex cell segmentation tasks, this manual provides the use of the cell segmentation tool Qupath to solve cell segmentation manually. <br>
### Purpose:
Use the QuPath tool to completely and accurately segment the cell shape, correct or generate cell segmentation results. <br>

### Scope of use:
Currently suitable for cell segmentation of ssDNA, DAPI, and HE staining. This manual will use ssDNA and HE staining diagrams respectively as examples <br>

## Tools and scripts introduction
| Tool Name | Version of Use | Tool Description | Download Address |
|-----------------|-------|-----------------------------------------|----------|
| QuPath | 0.4.3 | Qupath Software | https://github.com/qupath/qupath/releases/tag/v0.4.3 |
| mask2geojson.py script | \ | Convert the original cell segmentation binary image into a script tool for geojson adapted to Qupath software | [mask2geojson.py](/tutorials/mask2geojson.py)|
| ImageJ | Viewing and editing images can be used to gray and invert the three-channel image to segment the results of some cells. |https://imagej.net/ij/download.html |

## Operation steps
### Qupath basic operation
This SOP can be read:<br>
[Qupath partial operation SOP](2.Qupath partial operation SOP.md)

### Different dyeing methods:

#### -- ssDNA/DAPI
It can be directly placed in qupath. <br>
For denser cells, you can reduce the minimum area, sigma, and Background radius. <br>
For darker cells, adjust the Threshold appropriately. Or give background increments. <br>
For sparse cells, the cell expansion can be adjusted appropriately. <br>
(You can try to refer to any dyeing in the above three ways)<br>


#### -- H&E
**If it is an H&E image, the image needs to be converted into an 8-bit grayscale image, and inverted, and then placed in the Qupath. <br>**

**-- If the image is less than 2G, you can directly put ImageJ in for inverting the color operation, as follows: <br>**
a. Drag the image into ImageJ<br>

b.image>Type>8-bit<br>
<img src="../../../images/cell segmentation solution/image2Type28-bit.png"  style="zoom: 33%;" />

c. Invert the picture<br>
<img src="../../../images/cell segmentation solution/invert the image.png"  style="zoom: 33%;" />

d.image>Adusj>Threshold, the larger the value, the more obvious the separation is<br>
<img src="../../../images/cell segmentation solution/imageAdusjtThreshold.png"  style="zoom: 33%;" />

Click set after adjusting the value, and then file -save as -tiff<br>
<img src="../../../images/cell segmentation solution/click set.png"  style="zoom: 33%;" />
<img src="../../../images/cell segmentation solution/click set2.png"  style="zoom: 33%;" />


**--If the image is larger than 2G, you cannot directly put ImageJ in reverse operation. You need to use a script to convert the HE image into an 8bit grayscale image and invert the color. The following method is provided: <br>**
```shell
pip install cell-bin
```

```python
import tifffile
from cellbin.image.augmentation import f_rgb2gray
from cellbin.image.augmentation import f_ij_16_to_8

img = "Path/to/img.tif"
img = tifffile.imread(input)
img = f_rgb2gray(img)
img = f_ij_16_to_8(img)
img = 255 - img
tifffile.imwrite("Path/to/save/new.tif", img)
```
If there is a requirement, you can put the saved image into ImageJ to execute the "d.image>Adusj>Threshold" step. <br>



### Batch delete unwanted cell segmentation mask:
This method is used to modify only locally unsatisfactory cell segmentation results, and ImageJ is required. Operation as follows:<br>

1. First, merge the mask image with the original image. Observe the areas that need to be deleted. <br>
2. Click Backspace to delete the part in the circle<br>
<center class="half">
<img src="../../../images/cell segmentation solution/box selection area 1.png"  style="zoom: 33%;" />
<img src="../../../images/cell segmentation solution/box selection area 2.png"  style="zoom: 33%;" />
</center>

3. After the operation is completed, separate the merge diagram<br>
<center class="half">
<img src="../../../images/cell segmentation solution/merge isolation.png"  style="zoom: 33%;" />
<img src="../../../images/cell segmentation solution/isolation look.png"  style="zoom: 33%;" />
</center>

4. Turn the image back to the 8-bit image.

### Convert the cell mask into a geojson that can be placed in Qupath:
This method is used to convert the repaired cell segmentation results into geojson. Use mask2geojson.py to modify the path in the script using the above **"Introduction to Tools and Scripts". <br>
After generating geojson, open Qupath, drag the image and geojson into the picture frame in turn, and follow the above method of generating cell segmentation results of the SOP of "Qupath Basic Operations"**, use Qupath to supplement the newly added cell segmentation and finally generate a cell segmentation mask image that meets the expectations.
