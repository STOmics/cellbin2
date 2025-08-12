import pathlib
from os.path import exists

from cellbin2 import __version__

def generate_stereo_file(
        save_path: pathlib.Path,
        registered_image: pathlib.Path = pathlib.Path(""),
        compressed_image: pathlib.Path = pathlib.Path(""),
        matrix_template: pathlib.Path = pathlib.Path(""),
        gef: pathlib.Path = pathlib.Path(""),
        cellbin_gef: pathlib.Path = pathlib.Path(""),
        sn="",
):
    import json
    if not exists(save_path):
        stereo_data = {
            "file_version": "1.0.0",
            "minor_StereoMap_version": "4.0.0",
            "SAW_version": f"{__version__}",
            "pipeline": "cellbin2",
            "task_id": "",
            "run_start_time": "",
            "run_end_time": "",
            "analysis_uuid": "",
            "product_kit": "",
            "sequencing_type": "",
            "sn": f"{sn}",
            "omics": ["transcriptomics"],
            "chip_resolution": "",
            "organism": "",
            "tissue": "",

            "images": {
                "registered_image": str(registered_image.name),
                "compressed_image": str(compressed_image.name),
                "matrix_template": str(matrix_template.name),
            },

            "statistics": {
                "transcriptomics": {
                    "bin_list": [
                        "1",
                        "10",
                        "100",
                        "150",
                        "20",
                        "200",
                        "5",
                        "50",
                        "cellbin"
                    ],
                    "cell_count": "",
                    "feature_count": "",
                },
            },
            "StereoMap_explorer_files": {
                "transcriptomics": {
                    "gef": [str(gef.name)],
                    "cellbin_gef": [str(cellbin_gef.name)],
                    "h5ad": [],
                    "cellbin_h5ad": [],
                    "diffexp_csv": [],
                    "cellbin_diffexp_csv": [],
                },
            },
        }
    else:
        with save_path.open("r") as f:
            stereo_data = json.load(f)
        if registered_image.name != "":
            stereo_data["images"]["registered_image"] = str(registered_image.name)
        if compressed_image.name != "":
            stereo_data["images"]["compressed_image"] = str(compressed_image.name)
        if matrix_template.name != "":
            stereo_data["images"]["matrix_template"] = str(matrix_template.name)
        if gef.name != "":
            stereo_data["StereoMap_explorer_files"]["transcriptomics"]["gef"] = [str(gef.name)]
        if cellbin_gef.name != "":
            stereo_data["StereoMap_explorer_files"]["transcriptomics"]["cellbin_gef"] = [str(cellbin_gef.name)]
    with open(str(save_path), 'w') as fh:
        fh.write(json.dumps(stereo_data, indent=4))
