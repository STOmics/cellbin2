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
-c SN \
-i SN.tif \
-imf SN_IF.tif \
-o result/SN \
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
-c SN \
-i SN.tif \
-s DAPI \
-imf SN_IF.tif \
-m SN.raw.gef \
-k "Stereo-CITE T FF V1.0" \
-o result/SN 
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
-c SN 
-i /Path/TO/SN_fov_stitched_ssDNA.tif 
-s ssDNA 
-o /Path/TO//test/SN 
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