## Multimodal Cell Segmentaion Documentation  
The CellBin pipeline requires JSON configuration to perform segmentation post-processing operations including multimodal cell segmentation and nuclei expansion, which are implemented through specific parameter settings in the configuration file.

## Tutorial
### **Configuration File:** JSON File Structure Customization

The "molecular_classify" module utilizes a "cell_mask" dictionary within each slot to map the masks to their corresponding processed images. The keys of the dictionary, "core", "interior" and "cell", identify region classifications, while the values specify image indices from the "image_process" module. Additionally, the "correct_r" parameter defines the pixel extension radius for each cell.

```shell
"cell_mask": {
        "core":[0],
        "interior":[1],
        "cell":[2]
      }
"correct_r":10
```

In the given example above, "core":[0] specifies that the slot 0 in "image_process" module corresponds to a nuclei stain image. Each list may contain multiple images, and their masks will be merged in descending priority order. After merging all masks into a single mask, each cell is expanded by 10 pixels, as defined by the "correct_r".

Besides the JSON configuration described above, further parameter details can be found in the [JSON Configuration Documentation](../../docs/v2/JsonConfigurationDocumention.md). 

### Usage

```shell
# Minimal configuration (requires complete parameters in JSON)
CUDA_VISIBLE_DEVICES=0 python cellbin2/cellbin_pipeline.py -c <SN> -p <config.json> -o <output_dir> 
```
### Use Cases
#### 1. Nuclei Expansion Method
Full demo JSON file visit: [Nuclei Expansion JSON](../../cellbin2/config/demos/sample_core.json)  
When processing input data containing only nuclei stain images, the JSON configuration should be structured as follows:

```shell
"cell_mask": {
        "core":[0],
        "interior":[],
        "cell":[]
      }
"correct_r":10      
```

In which, "interior" and "cell" list are set as empty lists and the "correct_r" specifies the number of pixels to be expanded for each nuclei. In this case, the pipeline will automatically execute the nuclei expansion method and output the mask file:
| File Name | Description |
| ---- | ---- |
| SN_cell_mask.tif | Expanded core mask |

#### 2. Nuclei Cell Integration Method
Full demo JSON file visit: [Nuclei Cell JSON](../../cellbin2/config/demos/sample_cell_core.json) 
When the input data contains both nuclei stain and membrane stain images, the JSON configuration should be structured as follow:

```shell
"cell_mask": {
        "core":[0],
        "interior":[],
        "cell":[2]
      }
"correct_r":10      
```
This method processes each nuclei individually by first resolving overlaps between nuclei masks and cell masks. When a nuclei mask overlaps with a cell mask, the overlapping nuclei region is removed. The remaining non-overlapping nuclei regions are then incorporated into the cell mask. Finally, each cell mask is expanded by "correct_r" pixels. The output list of this method is as follow.

| File Name | Description |
| ---- | ---- |
| core_cell_merged_mask.tif | Nuclei cell integration mask |
| SN_cell_mask.tif | Expanded cell mask |

#### 3. Nuclei Interior Cell Integration Method
Full demo JSON file visit: [Nuclei Interior Cell JSON](../../cellbin2/config/demos/sample_multimodal.json)
When the input data contains all three type of stain images, the JSON file should be set as:

```shell
"cell_mask": {
        "core":[0],
        "interior":[3],
        "cell":[2]
      }
"correct_r":10      
```

This method is divided into two steps:

In the first step, membrane mask and interior mask are merged, where membrane mask has the priority. For each individual cell, if the two masks overlap, the interior mask is either removed or cut into small piece.

After membrane and interior mask are merged, nuclei mask are added. Similar to nuclei cell integration method, when nuclei mask is overlaped with the merged mask from the first step, it will be deleted. And the rest nuclei regions are added to the merged mask as final mask.

| File Name | Description |
| ---- | ---- |
| core_interior_cell_merged_mask.tif | Nuclei interior cell integration mask |
| SN_cell_mask.tif | Expanded cell mask |
| multimodal_mid_file | Intermediate mask files generated during processing |
| multimodal_mid_file/cell_mask_add_interior_add_nuclei.tif | Nuclei interior cell integration mask |
| multimodal_mid_file/cell_mask_add_interior.tif | Mask of merged cell and interior |
| multimodal_mid_file/interior_mask_final.tif | Final inteiror mask (remaining nuclei regions) |
| multimodal_mid_file/output_nuclei_mask.tif | Final nuclei mask (remaining nuclei regions) |
