### Configuration File Documentation  
We use a JSON file to define a multimodal bioimage analysis pipeline. Below is a structured breakdown of the configuration:

---

#### **1. Image Processing Module (`image_process`)**  
Configures parallel processing tasks for different stained image or gene. Keys `0-N` represent task slots.

| Slot             | Field                 | Example Value              | Description                                                                                                              |
|------------------|-----------------------|----------------------------|--------------------------------------------------------------------------------------------------------------------------|
| 0 - N            |                       | slot ID : (e.g., 0)        | the Numberical order of channel                                                                                          |
|                  | `file_path`           | `/Path/to/SN.tif`          | Input file path (supports TIF/GEF formats)                                                                               |
|                  | `tech_type`           | `DAPI` / `Transcriptomics` | Input technology type (options: ssDNA, HE, DAPI, IF, Transcriptomics, Protein)                                           |
|                  | `chip_detect`         | `true`/`false`             | Enable chip detection (typically for DAPI, ssDNA, HE ; execute in qc)                                                    |
|                  | `quality_control`     | `true`/`false`             | Enable image clarity control (execute in qc)                                                                             |
|                  | `tissue_segmentation` | `true`/`false`             | Perform tissue segmentation (execute in alignment)                                                                       |
|                  | `cell_segmentation`   | `true`/`false`             | Perform cell segmentation (execute in alignment)                                                                         |
|                  | `channel_align`       | `-1`/slot ID               | Channel calibration reference (`-1`=disable, `0`=use "0" channel;execute in qc)                                          |
| **Registration** |                       |                            | Configuration for image alignment and transformation                                                                     |
|                  | `fixed_image`         | `-1`                       | `-1`= fixed image; Slot number (e.g., 0)=registration with slot "0";execute in alignment)                                |
|                  | `trackline`           | `true`/`false`             | Enable trackline detection (execute in qc)                                                                               |
|                  | `reuse`               | `-1`/slot ID               | `-1`= Do not reuse parameters; Slot number (e.g., 0)=reuse slot "0" QC and registration parameters;execute in alignment) |

---

#### **2. Molecular Classification Module (`molecular_classify`)**  
Integrates multimodal data to generate molecular expression matrices.

| Slot | Field           | Example                     | Description                                                              |
|-----|-----------------|-----------------------------|--------------------------------------------------------------------------|
| 0-N |                 | gene matrix task : (e.g., 0) | the Numberical order of task                                             |
|     | `exp_matrix`    | `1`                         | Use Slot 1 (Transcriptomics or Protein) data as expression matrix source |
|     | `cell_mask`     | `[0]`                       | Use cell mask from Slot 0                                                |
|     | `correct_r`     | `10`                         | Cell mask correction radius (pixels). `0` = disable correction (execute in alignment)   |
|     | `extra_method`  | `""`                        | Reserved for custom analysis methods (empty by default,other method is To Be Extended) |


---

#### **3. Pipeline Control Module (`run`)**  
Global switches for analysis steps.

| Field             | Value   | Description                                               |
|-------------------|---------|-----------------------------------------------------------|
| `qc`              | `true`  | Enable image quality control  |
| `alignment`       | `true`  | Enable spatial alignment of multimodal data               |
| `matrix_extract`  | `true`  | Export expression matrix files                            |
| `report`          | `false` | Disable automated report generation                       |
| `annotation`      | `false` | Skip cell-type annotation                                 |

---
**Example Configuration** :

```json
{
  "image_process": {
    "0": {
      "file_path": "/Path/to/SN.tif",
      "tech_type": "DAPI",
      "chip_detect": true,
      "quality_control": true,
      "registration": {
        "fixed_image": -1,
        "trackline": true,
        "reuse": -1
      },
      "tissue_segmentation": true,
      "cell_segmentation": true,
      "correct_r": 10,
      "channel_align": -1
    },
    "1": {
      "file_path": "/Path/to/SN.raw.gef",
      "tech_type": "Transcriptomics",
      "chip_detect": false,
      "quality_control": false,
      "registration": {
        "fixed_image": -1,
        "trackline": true,
        "reuse": -1
      },
      "tissue_segmentation": true,
      "cell_segmentation": false,
      "correct_r": 0,
      "channel_align": -1
    },
    "2": {
      "file_path": "/Path/to/SN_ATP_IF.tif",
      "tech_type": "IF",
      "chip_detect": false,
      "quality_control": false,
      "registration": {
        "fixed_image": -1,
        "trackline": false,
        "reuse": -1
      },
      "tissue_segmentation": true,
      "cell_segmentation": true,
      "correct_r": 0,
      "channel_align": 0
    },
    "3": {
      "file_path": "/Path/to/SN.protein.raw.gef",
      "tech_type": "Protein",
      "chip_detect": false,
      "quality_control": false,
      "registration": {
        "fixed_image": -1,
        "trackline": true,
        "reuse": -1
      },
      "tissue_segmentation": true,
      "cell_segmentation": false,
      "correct_r": 0,
      "channel_align": -1
    }
  },
  "molecular_classify": {
      "0": {
        "exp_matrix": 1,
        "cell_mask": [
          1
        ],
        "extra_method": ""
      },
      "1": {
        "exp_matrix": 3,
        "cell_mask": [
          0,
          2
        ],
        "extra_method": ""
      }
    },
  "run": {
      "qc": false,
      "alignment": true,
      "matrix_extract": true,
      "report": false,
      "annotation": false
  }
}
```
