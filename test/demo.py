import subprocess
import os
import multiprocessing as mp
import tqdm

CURR_PATH = os.path.dirname(os.path.realpath(__file__))
PRJ_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CB2_PATH = os.path.join(PRJ_PATH, "cellbin2")
DEMO_DIR = os.path.join(PRJ_PATH, "demo_data")
DATA_DIR = os.path.join(DEMO_DIR, "data")
RESULT_DIR = os.path.join(DEMO_DIR, "result")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

from cellbin2.utils.weights_manager import download

NUMBER_OF_TASKS = 2
MAIN_CMD = [
    # "CUDA_VISIBLE_DEVICES=0",
    "python", os.path.join(CB2_PATH, "cellbin_pipeline.py"),
]

progress_bar = tqdm.tqdm(total=NUMBER_OF_TASKS)


def update_progress_bar(_):
    progress_bar.update()


DEMO_DATA = {
    "SS200000135TL_D1": {
        "data": {
            "SS200000135TL_D1.raw.gef": "/storeData/USER/data/00.stoDatabase/00.rawdata/00.Kit_dataset/01.Stereo_seq_transcriptomics_V1.2_Kit/SS200000135TL_D1/gem/SS200000135TL_D1.raw.gef",
            "SS200000135TL_D1_fov_stitched_ssDNA.tif": "/storeData/USER/data/01.CellBin/01.data/04.dataset/05.other_database/03.external_data/09.StitchImage/SS200000135TL_D1/SS200000135TL_D1_fov_stitched_ssDNA.tif"
        },
        "stain_type": "ssDNA",
        "kit_type": "Stereo-seq T FF V1.2"
    },
    "A02677B5": {
        "data": {
            "A02677B5_IF.tif": "/storeData/USER/data/01.CellBin/01.data/04.dataset/05.other_database/03.external_data/09.StitchImage/A02677B5/A02677B5_IF.tif",
            "A02677B5.tif": "/storeData/USER/data/01.CellBin/01.data/04.dataset/05.other_database/03.external_data/09.StitchImage/A02677B5/A02677B5.tif",
            "A02677B5.raw.gef": "/storeData/USER/data/01.CellBin/01.data/04.dataset/05.other_database/03.external_data/10.TestGem/A02677B5.raw.gef"
        },
        "stain_type": "DAPI",
        "kit_type": "Stereo-CITE T FF V1.1 R"
    },
    "B02210A5": {
        "data": {
            "B02210A5_fov_stitched.tif": "/storeData/USER/data/00.stoDatabase/00.rawdata/00.Kit_dataset/04.Stereo_CITE_Kit/B02210A5/raw_images/stitch/B02210A5_fov_stitched.tif",
            "B02210A5.protein.raw.gef": "/storeData/USER/data/00.stoDatabase/00.rawdata/00.Kit_dataset/04.Stereo_CITE_Kit/B02210A5/gem/B02210A5.protein.raw.gef"
        },
        "stain_type": "DAPI",
        "kit_type": "Stereo-CITE T FF V1.1 R"
    },
    "A02677B4": {
        "data": {
            "A02677B4.tif": "/storeData/USER/data/01.CellBin/01.data/04.dataset/05.other_database/03.external_data/09.StitchImage/A02677B4/A02677B4.tif",
            "A02677B4_IF.tif": "/storeData/USER/data/01.CellBin/01.data/04.dataset/05.other_database/03.external_data/09.StitchImage/A02677B4/A02677B4_IF.tif",
            "A02677B4.raw.gef": "/storeData/USER/data/00.stoDatabase/00.rawdata/00.Kit_dataset/04.Stereo_CITE_Kit/A02677B4/gem/A02677B4.raw.gef",
            "A02677B4.protein.raw.gef": "/storeData/USER/data/00.stoDatabase/00.rawdata/00.Kit_dataset/04.Stereo_CITE_Kit/A02677B4/gem/A02677B4.protein.raw.gef"
        },
        "stain_type": "DAPI",
        "kit_type": "Stereo-CITE T FF V1.1 R"
    },
    "B01715B4": {
        "json_path": os.path.join(CURR_PATH, "config", "case5.json")
    },
    "FP200000449TL_C3": {
        "json_path": os.path.join(CURR_PATH, "config", "case6.json")
    }
}


def auto_download_data():
    for i, v in DEMO_DATA.items():
        if 'json_path' in v:
            pass

        if 'data' in v:
            pass



def run_demo(i, v):
    o = os.path.join(RESULT_DIR, i)
    cmd = MAIN_CMD.copy()

    if "json_path" in v:
        json_file = v["json_path"]
        cmd.extend(['-c',i, '-p', json_file, '-o', o])
        print("Running:", cmd)
        subprocess.run(cmd)
        return 1

    if "data" in v:
        cmd.extend(['-c', i, '-s', v['stain_type'], '-k', v['kit_type'], '-o', o])
        data = v['data']
        for d_i, d_p in data.items():
            if "gef" in d_i:
                cmd.extend(['-m', d_p])
            elif 'IF' not in d_i:
                cmd.extend(['-i', d_p])
            else:
                cmd.extend(['-imf', d_p])
        print("Running:", cmd)
        subprocess.run(cmd)
        return 1


def main():
    auto_download_data()
    pool = mp.Pool(NUMBER_OF_TASKS)
    for i, v in DEMO_DATA.items():
        pool.apply_async(run_demo, (i, v), callback=update_progress_bar)

    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
