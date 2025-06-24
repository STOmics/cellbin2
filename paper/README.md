Xenium data <br>
step1: convert data run xenium.py<br>
step2: 
```shell
-c
SS200000045_M6
-p
/media/Data1/user/dengzhonghan/code/cellbin2dev/cellbin2/paper/xenium.json
-o
/media/Data1/user/dengzhonghan/data/tmp/test4paper/SS200000045_M6
```


```shell
-c
B01715B4
-p
/media/Data1/user/dengzhonghan/code/cellbin2dev/cellbin2/paper/B01715B4.json
-o
/media/Data1/user/dengzhonghan/data/tmp/test4paper/B01715B4
```

```shell
-c
B03120F1
-p
/media/Data1/user/dengzhonghan/code/cellbin2dev/cellbin2/paper/B03120F1.json
-o
/media/Data1/user/dengzhonghan/data/tmp/test4paper/B03120F1
```

```shell
-c
C04042E2
-p
/media/Data1/user/dengzhonghan/code/cellbin2dev/cellbin2/paper/C04042E2.json
-o
/media/Data1/user/dengzhonghan/data/tmp/test4paper/C04042E2
```
VisiumHD  <br>
step1: convert data run VisiumHD.py, then rename the output result and give a fake chip number<br>
```shell
python VisiumHD.py -i /storeData/USER/data/01.CellBin/00.user/fanjinghong/data/10xHD/Visium_HD_Mouse_Lung_Fresh_Frozen_tissue/Visium_HD_Mouse_Lung_Fresh_Frozen_tissue_image.tif -h5 /storeData/USER/data/01.CellBin/00.user/fanjinghong/data/10xHD/Mouse_Lung/binned_outputs/square_002um/filtered_feature_bc_matrix.h5 -pa /storeData/USER/data/01.CellBin/00.user/fanjinghong/data/10xHD/Mouse_Lung/binned_outputs/square_002um/spatial/tissue_positions.parquet -o /storeData/USER/data/01.CellBin/00.user/fanjinghong/data/10xHD/v2paper
```
step2: run cellbin2_pipeline.py
```shell
python cellbin2/cellbin_pipeline.py -c A04547A1C3 -i /storeData/USER/data/01.CellBin/00.user/fanjinghong/data/10xHD/v2paper/A04547A1C3.tif -s HE -m /storeData/USER/data/01.CellBin/00.user/fanjinghong/data/10xHD/v2paper/A04547A1C3.gem.gz  -o /storeData/USER/data/01.CellBin/00.user/fanjinghong/data/10xHD/v2paper/output -p /storeData/USER/data/01.CellBin/00.user/fanjinghong/cellbin2/paper/VisiumHD.json
```
