## Multi-Channel End to End Cell Segmentation Documentation  
Cellbin2 now supports cell segmentation for multi-channel images using end-to-end deep learning models.

## Tutorial

### **Model Download:** Model Download Link 
Before using the cell segmentation tools, please download the pre-trained model. Alternatively, you can use your own fine-tuned model (note: only models fine-tuned from CellposeSAM are supported).

| Medel Name | Description | Download Link|
| ---- | ---- |----|
| CellposeSam | Base model for general cell segmentation | [Downlod](https://bgipan.genomics.cn/v3.php/download/ln-file?FileId=13737945&ShareKey=D4BFhYPwZZOcc16MpZ4G&VersionId=7121704&UserId=38259&Password=Y5Sk&Policy=eyJBSyI6IjBlNjE5YzFjYzZlMGRhZTZlZmMzMjMxMDVjN2I4YWM1IiwiQWF0IjoxLCJBaWQiOiJKc1FDc2pGM3lyN0tBQ3lUIiwiQ2lkIjoiYTFkMTM0MjAtNTI5ZS00YzFhLWEzMGMtMGI0YzA0NzM5ZTBlIiwiRXAiOjkwMCwiRGF0ZSI6IlR1ZSwgMDkgRGVjIDIwMjUgMDc6MzY6MzYgR01UIn0%3D&Signature=9980c83d371885a65c272eea42e31e11b036a210)   |
### **Input Preparation:** Image Structure Requirements

Currently, only RGB images are supported, where each channel represents a distinct cellular component. Please ensure your image follows the channel structure below:

| Channel Position | Color | Cellular Component|
| ---- | ---- |----|
| Channel 1  | Red | Membrane |
| Channel 2 | Green | Interior |
| Channel 3 | Blue | Nuclei | 

Image format: Tiff


### **Usage**
To perform cell segmentation, run the script with the required parameters:
```shell
# Minimal configuration (requires complete parameters in JSON)
python cellbin2/contrib/cellposesam.py -i <inputfile/input.tif> -o <outputfile/> -m <modelpath/model> -g <0_or_1>

```

|Parameter|	Short Form|	Required|	Description	|Example|
| ---- | ------ |---- |---- |---- |
|--input|	-i|	Yes	|Path to input image file	|/data/images/sample.tif|
|--output|	-o|	Yes|	Output directory path	|/results/segmentation/|
|--model_path|	-m	|Yes	|Path to trained model |/models/cellposesam_base|
|--gpu|	-g|	No|	Use GPU (1) or CPU (0)|	1|
