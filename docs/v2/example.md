case1: only qc <br>
Find cellbin2/Stereo-CITE T FF.json<br>
change run like follows
   
```shell
"run": {
    "qc": true,
    "alignment": false,
    "matrix_extract": false,
    "report": false,
    "annotation": false
  }
```
run command like below
```shell
python cellbin2/cellbin_pipeline.py \
-c A02677B5 \
-i A02677B5.tif \
-imf A02677B5_IF.tif \
-o result/A02677B5 \
-s DAPI 
```
---
case2: only alignment with registration <br>
This mode assumes you have already done qc <br>
Find corresponding json, change run like follows <br>
   
```shell
"run": {
    "qc": false,
    "alignment": true,
    "matrix_extract": false,
    "report": false,
    "annotation": false
  }
```
run command like below <br>
-o path should be the same as qc
```shell
python cellbin2/cellbin_pipeline.py \
-c A02677B5 \
-i A02677B5.tif \
-s DAPI \
-imf A02677B5_IF.tif \
-m A02677B5.raw.gef \
-k "Stereo-CITE T FF V1.0" \
-o result/A02677B5 
```
---
case3: only alignment without registration & matrix extract <br>
This mode assumes the input image is aligned, so only run tissue and cell seg <br>
If kit type is Stereo-CITE T FF V1.0, you find Stereo-CITE T FF.json under cellbin2/config <br>
change run like follows
```shell
"run": {
    "qc": false,
    "alignment": true,
    "matrix_extract": true,
    "report": false,
    "annotation": false
  }
```
---
case 4: No matrix <br>
Stereo-seq T FF
ssDNA
```shell
-c SS200000135TL_D1 
-i /media/Data/dzh/data/cellbin2/demo_data/product_demo/SS200000135TL_D1/SS200000135TL_D1_fov_stitched_ssDNA.tif 
-s ssDNA 
-o /media/Data/dzh/data/cellbin2/test/SS200000135TL_D1 
-k "Stereo-seq T FF V1.2"
```
---
case 5: Create bin1 matrix image
```shell
cd cellbin2/matrix
python matrix.py \
-i {SN.gem.gz/SN.gef file path}
-o {output_folder}
```