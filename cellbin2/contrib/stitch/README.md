<div align="center">
  <img src="../../../docs/images/mfws.png" width="30%" height="auto">
  <h1 align="center">
    MFWS: Multiple FFT Weighted Stitching
  </h1>
</div>

## Introduction
We developed the image stitching algorithm named Multiple FFT Weighted Stitching (MFWS).
The algorithm procedures are as follows: 
* Calculate the overlap matrices of adjacent tiles in horizontal and vertical direction
* Obtain the global coordinates of each tile
* Complete the image mosaic and fusion generation

## Installation
```shell
pip install mfws
```

## Tutorials
We will use 3 scenario data to illustrate 2 usage methods. The parameters listed in the table below are helpful for using the tool.


| Name         | type    | Importance                          | Default value | memo                                                                                                     |
|--------------|---------|-------------------------------------|---------------|----------------------------------------------------------------------------------------------------------|
| input        | string  | <font color=#B23AEE>Required</font> | &#9744;       | File directory path                                                                                      |
| output       | string  | <font color=#B23AEE>Required</font>                            | &#9744;       | Output file path or directory path                                                                       |
| rows         | integer | <font color=#B23AEE>Required</font>                            | &#9744;       | Maximum number of scan row                                                                               |
| cols         | integer | <font color=#B23AEE>Required</font>                            | &#9744;       | Maximum number of scan column                                                                            |
| start_row    | integer | Optional                            | 1             | Start row of image to be stitched, start with 1 instead of 0                                             |
| start_col    | integer | Optional                            | 1             | Start col of image to be stitched, start with 1 instead of 0                                             |
| end_row      | integer | Optional                            | rows          | End row of image to be stitched                                                                          |
| end_col      | integer | Optional                            | cols          | End col of image to be stitched                                                                          |
| proc         | integer | Optional                            | 1             | Number of processes used, should be set reasonably based on the computing power of the hardware platform |
| overlapx     | float   | Optional                            | 0.1           | Number of overlapping pixels in the horizontal direction / width of FOV                                  |
| overlapy     | float   | Optional                            | 0.1           | Number of overlapping pixels in the vertical direction / height of FOV                                   |
| fusion       | integer | Optional                            | 1             | Fusion Solution: 1 - no fusion, 2 - with sin method                                                      |
| scan_type    | integer | Optional                            | 1             | Scanning method:                                                                                         |
| device       | string  | Optional                            | CG            | Device Type: CG, T1, dolphin                                                                             |
| thumbnail    | float   | Optional                            | 1             | Downsampling control parameter, a decimal between 0 and 1                                                |
| method       | integer | Optional                            | 1             | Stitching method: 1 - mfws, 2 - Use overlap to complete mechanical stitching                             |
| channel      | string  | Optional                            | ''            | In a multi-layer image scenario, the labels of the layers to be spliced                                  |
| fft_channel  | integer | Optional                            | 0             | Channel used to calculate translation                                                                    |
| name_pattern | string  | Optional                            | ''            | Regular expression matching characters, used to parse the file name to get the row and column index      |
| name_pattern | string  | Optional                            | ''            | Regular expression matching characters, used to parse the file name to get the row and column index      |
| name_index_0 | bool    | Optional                            | true          | Is row and column index numbers in the file name start with 0?                                           |
| flip         | int     | Optional                            | 1             | Flipping FOV during stitching: 1 - not, 2 - up & down, 3 - left and right                                |


### case 1: command line

You can use ```mfws -h``` to view the usage, use ```mfws -v``` to view software version information. 

- cellbin

    <details close>
    <summary>Motic</summary>

    > Note: File naming like `controlD1_0000_0001_2021-11-25_12-26-25-544.tif`, `0000_0001` represents the FOV of row 0 and column 1

    ```shell
    mfws 
    -i /data/image_path
    -o /data/output_path/mfws.tif 
    --rows 13 
    --cols 9 
    --overlapx 0.1 
    --overlapy 0.1 
    --method 2  
    ```
    </details>

    <details close>
    <summary>CG</summary>
  
    > Note: File naming like `A00000XX_0000_0001_20221205123304.tiff`, `0000_0001` represents the FOV of row 0 and column 1

    ```shell
    mfws
    -i /data/image_path
    -o /data/output_path/mfws.tif 
    --rows 15
    --cols 12
    --overlapx 0.1 
    --overlapy 0.1 
    --method 1
    ```

    </details>

    <details close>
    <summary>Leica</summary>
  
    > Note: File naming like `TileScan 3_s06_RAW_ch00.tif`, `s01` represents the 6 FOV in snake by row scan mode

    ```shell
    mfws 
    -i /data/image_path
    -o /data/output_path/mfws.tif 
    --rows 7 --cols 5 --overlapx 0.1 
    --overlapy 0.1 
    --method 2 
    --scan_type 51 
    --name_pattern *s{xx}* 
    --device leica
    ```

    </details>

- insitu

    <details close>
    <summary>Dolphin - Odd numbered layers</summary>
  
    > Note: File naming like `DL2_DL1000011_B_20250627142513.L001.S001.C002R015.P008.C.2025-06-27_14-29-10-022.tif`, `C002R015` represents the FOV of row 2 and column 15

    ```shell
    mfws 
    -i /data/image_path
    -o /data/output_path
    --rows
    5
    --cols
    5
    --start_row
    4
    --start_col
    4
    --overlapx
    0.1
    --overlapy
    0.1
    --method
    2
    --device
    dolphin
    --name_index_0
    --flip 2
    ```

    </details>

    <details close>
    <summary>Dolphin - Even numbered layers</summary>
  
    > Note: File naming like `DL2_DL1000011_B_20250627142513.L001.S001.C002R015.P008.C.2025-06-27_14-29-10-022.tif`, `C002R015` represents the FOV of row 2 and column 15

    ```shell
    mfws 
    -i /data/image_path 
    -o /data/output_path
    --rows
    5
    --cols
    5
    --start_row
    4
    --start_col
    4
    --overlapx
    0.1
    --overlapy
    0
    --method
    2
    --device
    dolphin
    --name_index_0
    ```

    </details>

    <details close>
    <summary>T1</summary>
  
    > Note: File naming like `DP81FocusSweep20250318org-hm.L001.S001.C003R008.P003.C.2025-03-18_11-15-37-276.tif`, `C003R008` represents the FOV of row 8 and column 3

    ```shell
    mfws 
    -i /data/image_path
    -o /data/output_path
    --rows
    5
    --cols
    5
    --start_row
    4
    --start_col
    4
    --overlapx
    0.1
    --overlapy
    0
    --method
    2
    --device
    T1
    --name_index_0
    ```
    </details>

### case 2: API

```python
from mfws.mfws import stitching


image_path = '/path/mfws/images'
stitching(
    image_path = image_path,
    overlap = '0.1_0.1',
    scope_flag = True,
    rows = 20,
    cols = 10,
    start_row = 1,
    start_col = 1,
    end_row = 20,
    end_col = 10,
    fuse = 1,
    output_path = r"/path/mfws/images/output/mfws_test.tif"
)
```
