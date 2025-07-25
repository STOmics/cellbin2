from os.path import join, dirname, exists
import os
import glob
import tarfile
import shutil


def save_tar(save_path, files):
    with tarfile.open(save_path, 'w:gz', compresslevel=5) as tf:
        for file, arcName in files:
            tf.add(file, arcName)


def update_ipr_in_tar(tar_path, ipr_path):

    tar_tmp_path = join(dirname(tar_path), "tar_tmp")
    if exists(tar_tmp_path):
        shutil.rmtree(tar_tmp_path)
    os.makedirs(tar_tmp_path)
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=tar_tmp_path)
        raw_ipr = glob.glob(os.path.join(tar_tmp_path, "*.ipr"))[0]
        os.remove(raw_ipr)
    shutil.copy(ipr_path, tar_tmp_path)
    if exists(tar_path):
        os.remove(tar_path)
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(tar_tmp_path, arcname="")
    if os.path.exists(tar_tmp_path):
        shutil.rmtree(tar_tmp_path)