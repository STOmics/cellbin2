## 1. Xenium data <br>
step1: convert data run xenium.py<br>
step2: 
```shell
-c
B03205D314
-p
/media/Data1/user/dengzhonghan/code/cellbin2dev/cellbin2/paper/xenium.json
-o
/media/Data1/user/dengzhonghan/data/tmp/test4paper/B03205D314
```

## 2. 1*2  Rat brain
```shell
-c
Z97822X8W8
-p
/media/Data1/user/dengzhonghan/code/cellbin2dev/cellbin2/paper/Z97822X8W8.json
-o
/media/Data1/user/dengzhonghan/data/tmp/test4paper/Z97822X8W8
```

## 3. Mouse kidney
```shell
-c
Z97502X8
-p
/media/Data1/user/dengzhonghan/code/cellbin2dev/cellbin2/paper/Z97502X8.json
-o
/media/Data1/user/dengzhonghan/data/tmp/test4paper/Z97502X8
```

## 4. 2*3 Rat brain
```shell
-c
Y98792V6T5
-p
/media/Data1/user/dengzhonghan/code/cellbin2dev/cellbin2/paper/Y98792V6T5.json
-o
/media/Data1/user/dengzhonghan/data/tmp/test4paper/Y98792V6T5
```

## 5. Mouse brain
```shell
-c
X95957V7
-p
/media/Data1/user/dengzhonghan/code/cellbin2dev/cellbin2/paper/X95957V7.json
-o
/media/Data1/user/dengzhonghan/data/tmp/test4paper/X95957V7
```

## 6. Mouse liver (stereo-CITE)
```shell
-c
Y968792V3
-p
/media/Data1/user/dengzhonghan/code/cellbin2dev/cellbin2/paper/Y968792V3.json
-o
/media/Data1/user/dengzhonghan/data/tmp/test4paper/Y968792V3
```
## 7. Arabidopsis thaliana
```shell
-c
UK799999550GO_X6
-p
/media/Data1/user/dengzhonghan/code/cellbin2dev/cellbin2/paper/UK799999550GO_X6.json
-o
/media/Data1/user/dengzhonghan/data/tmp/test4paper/UK799999550GO_X6
```

## 8. Mouse liver (IF stained image)
```shell
-c
C01344C4
-p
/media/Data1/user/dengzhonghan/code/cellbin2dev/cellbin2/paper/C01344C4.json
-o
/media/Data1/user/dengzhonghan/data/tmp/test4paper/C01344C4
```

## 9. PBMC (Stereo-cell)
```shell
-c
Z96914Z8W6
-p
/media/Data1/user/dengzhonghan/code/cellbin2dev/cellbin2/paper/Z96914Z8W6.json
-o
/media/Data1/user/dengzhonghan/data/tmp/test4paper/Z96914Z8W6
```


## VisiumHD  <br>
step1: convert data run VisiumHD.py, then rename the output result and give a fake chip number<br>
```shell
python VisiumHD.py -i /storeData/USER/data/01.CellBin/00.user/fanjinghong/data/10xHD/Visium_HD_Mouse_Lung_Fresh_Frozen_tissue/Visium_HD_Mouse_Lung_Fresh_Frozen_tissue_image.tif -h5 /storeData/USER/data/01.CellBin/00.user/fanjinghong/data/10xHD/Mouse_Lung/binned_outputs/square_002um/filtered_feature_bc_matrix.h5 -pa /storeData/USER/data/01.CellBin/00.user/fanjinghong/data/10xHD/Mouse_Lung/binned_outputs/square_002um/spatial/tissue_positions.parquet -o /storeData/USER/data/01.CellBin/00.user/fanjinghong/data/10xHD/v2paper
```
step2: run cellbin2_pipeline.py
```shell
python cellbin2/cellbin_pipeline.py -c A04547A1C3 -i /storeData/USER/data/01.CellBin/00.user/fanjinghong/data/10xHD/v2paper/A04547A1C3.tif -s HE -m /storeData/USER/data/01.CellBin/00.user/fanjinghong/data/10xHD/v2paper/A04547A1C3.gem.gz  -o /storeData/USER/data/01.CellBin/00.user/fanjinghong/data/10xHD/v2paper/output -p /storeData/USER/data/01.CellBin/00.user/fanjinghong/cellbin2/paper/VisiumHD.json
```

## Void segmentation
```shell
# code location: cellbin2/paper/Void_segmentation.py
python Void_segmentation.py \
    --cell input_cell_mask.tif \
    --raw_void_out raw_void.tif \
    --void_out final_void.tif \
    --boundary_out boundary_cells.tif \
    --in_void_out cells_in_void.tif \
    --other_out cells_other.tif \
    --dilate 2 \
    --close 2 \
    --min_area 3000 \
    --threshold 5 \
    --smooth 3
```
