import subprocess
import os
import multiprocessing as mp
import tqdm

CURR_PATH = os.path.dirname(os.path.realpath(__file__))
CB2_PATH = os.path.join(CURR_PATH, "cellbin2")
DEMO_DIR = os.path.join(CURR_PATH, "demo_data")
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
            "SS200000135TL_D1_fov_stitched_ssDNA.tif": "https://bgipan.genomics.cn/v3.php/download/ln-file?FileId=6488198&ShareKey=Iqhde9tKfvWHX5q22T3n&VersionId=4370462&UserId=-1&Password=8Jhw&Policy=eyJBSyI6IkFOT05ZTU9VUyIsIkFhdCI6MSwiQWlkIjoiSnNRQ3NqRjN5cjdLQUN5VCIsIkNpZCI6ImY3NGM2NzlkLTY2ZWUtNDc1OS04ODlmLWQyMzczYTljODY5MiIsIkVwIjo5MDAsIkRhdGUiOiJGcmksIDEzIERlYyAyMDI0IDAyOjQ2OjQ1IEdNVCJ9&Signature=4c16f7d5ed7a5e83ea2d0c91176e8bd7878b6c13",
            "SS200000135TL_D1.raw.gef": "https://bgipan.genomics.cn/v3.php/download/ln-file?FileId=6488191&ShareKey=Iqhde9tKfvWHX5q22T3n&VersionId=4370458&UserId=-1&Password=8Jhw&Policy=eyJBSyI6IkFOT05ZTU9VUyIsIkFhdCI6MSwiQWlkIjoiSnNRQ3NqRjN5cjdLQUN5VCIsIkNpZCI6ImY3NGM2NzlkLTY2ZWUtNDc1OS04ODlmLWQyMzczYTljODY5MiIsIkVwIjo5MDAsIkRhdGUiOiJGcmksIDEzIERlYyAyMDI0IDAyOjQ2OjEzIEdNVCJ9&Signature=ba71c89e825435ca47975175b187d0a151b553a2"
        },
        "stain_type": "ssDNA",
        "kit_type": "Stereo-seq T FF V1.2 R"
    },
    # "C04042E3": {
    #     "data": {
    #         "C04042E3.raw.gef": "https://bgipan.genomics.cn/v3.php/download/ln-file?FileId=6488938&ShareKey=Iqhde9tKfvWHX5q22T3n&VersionId=4370866&UserId=-1&Password=8Jhw&Policy=eyJBSyI6IkFOT05ZTU9VUyIsIkFhdCI6MSwiQWlkIjoiSnNRQ3NqRjN5cjdLQUN5VCIsIkNpZCI6ImY3NGM2NzlkLTY2ZWUtNDc1OS04ODlmLWQyMzczYTljODY5MiIsIkVwIjo5MDAsIkRhdGUiOiJGcmksIDEzIERlYyAyMDI0IDAyOjQ3OjAwIEdNVCJ9&Signature=043a6214f4a9c56d6c2ca7dddb0304c2273943ab",
    #         "C04042E3_fov_stitched.tif": "https://bgipan.genomics.cn/v3.php/download/ln-file?FileId=6488941&ShareKey=Iqhde9tKfvWHX5q22T3n&VersionId=4370869&UserId=-1&Password=8Jhw&Policy=eyJBSyI6IkFOT05ZTU9VUyIsIkFhdCI6MSwiQWlkIjoiSnNRQ3NqRjN5cjdLQUN5VCIsIkNpZCI6ImY3NGM2NzlkLTY2ZWUtNDc1OS04ODlmLWQyMzczYTljODY5MiIsIkVwIjo5MDAsIkRhdGUiOiJGcmksIDEzIERlYyAyMDI0IDAyOjQ3OjEyIEdNVCJ9&Signature=83937f630d30db8d83efb99dcc716c7c8bb606fb"
    #     },
    #     "stain_type": "HE",
    #     "kit_type": "Stereo-seq T FF V1.3 R"
    # },
    # "A02677B5": {
    #     "data": {
    #         "A02677B5.tif": "https://bgipan.genomics.cn/v3.php/download/ln-file?FileId=6489271&ShareKey=Iqhde9tKfvWHX5q22T3n&VersionId=4371161&UserId=-1&Password=8Jhw&Policy=eyJBSyI6IkFOT05ZTU9VUyIsIkFhdCI6MSwiQWlkIjoiSnNRQ3NqRjN5cjdLQUN5VCIsIkNpZCI6ImY3NGM2NzlkLTY2ZWUtNDc1OS04ODlmLWQyMzczYTljODY5MiIsIkVwIjo5MDAsIkRhdGUiOiJGcmksIDEzIERlYyAyMDI0IDAyOjQ5OjA4IEdNVCJ9&Signature=de87a98bb443ccecf3c04fb3436d44994d78e218",
    #         "A02677B5.raw.gef": "https://bgipan.genomics.cn/v3.php/download/ln-file?FileId=6489270&ShareKey=Iqhde9tKfvWHX5q22T3n&VersionId=4371160&UserId=-1&Password=8Jhw&Policy=eyJBSyI6IkFOT05ZTU9VUyIsIkFhdCI6MSwiQWlkIjoiSnNRQ3NqRjN5cjdLQUN5VCIsIkNpZCI6ImY3NGM2NzlkLTY2ZWUtNDc1OS04ODlmLWQyMzczYTljODY5MiIsIkVwIjo5MDAsIkRhdGUiOiJGcmksIDEzIERlYyAyMDI0IDAyOjQ5OjI5IEdNVCJ9&Signature=2280b069d853520c6b204cda86dbfb2dc69848ff",
    #         "A02677B5_IF.tif": "https://bgipan.genomics.cn/v3.php/download/ln-file?FileId=6489272&ShareKey=Iqhde9tKfvWHX5q22T3n&VersionId=4371162&UserId=-1&Password=8Jhw&Policy=eyJBSyI6IkFOT05ZTU9VUyIsIkFhdCI6MSwiQWlkIjoiSnNRQ3NqRjN5cjdLQUN5VCIsIkNpZCI6ImY3NGM2NzlkLTY2ZWUtNDc1OS04ODlmLWQyMzczYTljODY5MiIsIkVwIjo5MDAsIkRhdGUiOiJGcmksIDEzIERlYyAyMDI0IDAyOjQ5OjQ5IEdNVCJ9&Signature=e824807daa72bcb9b7ee4adacbe32613fce3b285"
    #     },
    #     "stain_type": "DAPI",
    #     "kit_type": "Stereo-CITE T FF V1.0 R"
    # }
}


def auto_download_data():
    for i, v in DEMO_DATA.items():
        data = v['data']
        for d_i, d_l in data.items():
            local_file_path = os.path.join(DATA_DIR, d_i)
            download(
                local_file=local_file_path,
                file_url=d_l
            )
            data[d_i] = local_file_path


def run_demo(i, v):
    o = os.path.join(RESULT_DIR, i)
    cmd = [
        '-c', i,
        '-s', v['stain_type'],
        '-k', v['kit_type'],
        '-o', o
    ]
    data = v['data']
    for d_i, d_p in data.items():
        if "gef" in d_i:
            cmd.extend(
                ['-m', d_p]
            )
        elif 'IF' not in d_i:
            cmd.extend(
                ['-i', d_p]
            )
        else:
            cmd.extend(
                ['-imf', d_p]
            )
    cmd = MAIN_CMD + cmd
    print(cmd)
    subprocess.run(cmd)


def main():
    auto_download_data()
    pool = mp.Pool(NUMBER_OF_TASKS)
    for i, v in DEMO_DATA.items():
        pool.apply_async(run_demo, (i, v), callback=update_progress_bar)

    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
