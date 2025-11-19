# smoke test
## product mode

### case1
one step run <br>
use: Stereo-CITE_T_FF.json, Stereo-seq_N_FFPE.json, Stereo-seq_T_FF.json <br>
data cover (image+matrix):
- FF: ssDNA, H&E, DAPI+mIF
- FFPE: ssDNA, H&E, DAPI
- cite: DAPI+mIF

### case2
two steps run 
- run qc first (image)
- then run alignment (image+matrix)

## research mode
### case1
image+matrix <br>
use: xxx R.json <br>
qc + alignment + matrix extract <br>

### case2
with report <br>
image+matrix <br>
xxx R.json + -r <br>
qc + alignment + matrix extract + report <br>

### case3
image <br>
xxx R.json <br>

### case4
alignment + matrix extract <br>
image+matrix <br>
this mode assumes the input image is aligned <br>