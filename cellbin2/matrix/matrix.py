from typing import Optional
import os.path

import pandas as pd
import h5py
import numpy as np
import warnings
import gzip
from pathlib import Path
import argparse

from numba import njit, prange
from cellbin2.utils import clog
from cellbin2.contrib.alignment.basic import TemplateInfo, ChipBoxInfo
from cellbin2.image import cbimread, cbimwrite
from cellbin2.modules import naming
from pydantic import BaseModel, Field

def parse_gef_line(data, img, _dtype):
    """
    Speedup parse lines with numba
    """
    dtype_dict = {
        np.uint8: 255,
        np.uint16: 65535,
        np.uint32: 4294967295,
    }

    max_val = dtype_dict[_dtype]

    for i in prange(len(data)):
        y, x, c = data[i]["y"], data[i]["x"], data[i]["count"]
        img[y, x] = min(max_val, c + img[y, x])

    # if _dtype == np.uint8:
    #     _uint8_parse_gef_line(data, img)
    # elif _dtype == np.uint16:
    #     _uint16_parse_gef_line(data, img)
    # elif _dtype == np.uint32:
    #     _uint32_parse_gef_line(data, img)


class GeneticStandards(BaseModel):
    bin20_thr: int = Field(-1, description="")
    bin50_thr: int = Field(-1, description="")
    bin200_thr: int = Field(-1, description="")


class cMatrix(object):
    """ single matrix management """

    def __init__(self) -> None:
        self._gene_mat = np.array([])
        self.binx = 1
        self.dtype = np.uint8
        self.x_start = 65535
        self.y_start = 65535
        self.h_x_start = 0
        self.h_y_start = 0

        self._template: TemplateInfo = None
        self._chip_box: ChipBoxInfo = ChipBoxInfo()
        self.file_path: str = ''

    def read(self, file_path: Path, chunk_size=1024 * 1024 * 10):
        """
        this function copy from,
            https://dcscode.genomics.cn/stomics/saw/register/-/blob/main/register/utils/matrixloader.py?ref_type=heads
        :param file_path: matrix file path
        :param chunk_size:
        :return:
        """
        suffix = file_path.suffix
        assert suffix in ['.gz', '.gef', '.gem']
        if suffix == ".gef":
            self.binx, self.x_start, self.y_start, self._gene_mat = self._load_gef(self, file_path)
            return

        if suffix == ".gz":
            fh = gzip.open(file_path, "rb")
        else:
            fh = open(str(file_path), "rb")  # pylint: disable=consider-using-with
        title = ""
        # Move pointer to the header of line
        eoh = 0
        header = ""
        for line in fh:
            line = line.decode("utf-8")
            if not line.startswith("#"):
                title = line
                break
            header += line
            eoh = fh.tell()
        fh.seek(eoh)
        # Initlise
        title = title.strip("\n").split("\t")
        umi_count_name = [i for i in title if "ount" in i][0]
        title = ["x", "y", umi_count_name]

        max_val = 0
        for chunk in pd.read_csv(fh, sep='\t', usecols=[umi_count_name], dtype=np.uint32, chunksize=chunk_size):
            tmp_max = chunk[umi_count_name].max()
            if tmp_max > max_val:
                max_val = tmp_max

        if max_val <= 255:
            _dtype = np.uint8
            _max_val = 255
        elif max_val <= 65535:
            _dtype = np.uint16
            _max_val = 65535
        else:
            _dtype = np.uint32
            _max_val = 4294967295

        fh.seek(eoh)
        img = np.zeros((1, 1), dtype=_dtype)

        # todo There is a problem reading gem.gz and barcode_gene_exp.txt
        df = pd.read_csv(
            fh,
            sep="\t",
            header=0,
            usecols=title,
            dtype=dict(zip(title, [np.uint32] * 3)),
            chunksize=chunk_size,
        )

        _list = header.split("\n#")[-2:]
        self.h_x_start = int(_list[0].split("=")[1])
        self.h_y_start = int(_list[1].split("=")[1])
        import re
        m_bin = re.search(r'(?im)^#.*\b(?:bin(?:size|_size|x)?)[\s:=]*([0-9]+)', header)
        if m_bin:
            self.binx = int(m_bin.group(1))
        else:
            self.binx = 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for chunk in df:
                # convert data to image
                tmp_h = chunk["y"].max() + 1
                tmp_w = chunk["x"].max() + 1
                tmp_min_y = chunk["y"].min()
                tmp_min_x = chunk["x"].min()
                if tmp_min_x < self.x_start:
                    self.x_start = tmp_min_x
                if tmp_min_y < self.y_start:
                    self.y_start = tmp_min_y

                h, w = img.shape[:2]

                chunk = (
                    chunk.groupby(["x", "y"])
                    .agg(UMI_sum=(umi_count_name, "sum"))
                    .reset_index()
                )
                # chunk["UMI_sum"] = chunk["UMI_sum"].mask(chunk["UMI_sum"] > 255, 255)
                tmp_img = np.zeros(shape=(tmp_h, tmp_w), dtype=_dtype)
                tmp_img[chunk["y"], chunk["x"]] = chunk["UMI_sum"]

                # resize matrix
                ext_w = tmp_w - w
                ext_h = tmp_h - h
                if ext_h > 0:
                    img = np.pad(img, ((0, abs(ext_h)), (0, 0)), "constant")
                elif ext_h < 0:
                    tmp_img = np.pad(tmp_img, ((0, abs(ext_h)), (0, 0)), "constant")
                if ext_w > 0:
                    img = np.pad(img, ((0, 0), (0, abs(ext_w))), "constant")
                elif ext_w < 0:
                    tmp_img = np.pad(tmp_img, ((0, 0), (0, abs(ext_w))), "constant")

                # in case overflow
                tmp_img = (
                        _max_val - tmp_img
                )  # old b is gone shortly after new array is created
                np.putmask(
                    img, tmp_img < img, tmp_img
                )  # a temp bool array here, then it's gone
                img += _max_val - tmp_img  # a temp array here, then it's gone
        df.close()
        self._gene_mat = img[self.y_start:, self.x_start:]

    @staticmethod
    def _load_gef(self, file):
        """
        Sepeedup version that for gef file format of all the bin_size
        """
        chunk_size = 512 * 1024
        with h5py.File(file, "r") as fh:
            dataset = fh["/geneExp/bin1/expression"]

            if not dataset[...].size:
                clog.error("The sequencing data is empty, please confirm the {} file.".format(file))
                raise Exception("The sequencing data is empty, please confirm the {} file.".format(file))

            try:
                binx = int(dataset.attrs["resolution"][0]/dataset.attrs["dnbPitch"][0]) 
            except KeyError:
                binx = int(dataset.attrs["resolution"][0]/500)

            if binx != 1: 
                self.dtype = np.uint16
            min_x, max_x = dataset.attrs["minX"][0], dataset.attrs["maxX"][0]
            min_y, max_y = dataset.attrs["minY"][0], dataset.attrs["maxY"][0]
            width = max_x - min_x + 1
            height = max_y - min_y + 1
            img = np.zeros((height, width), self.dtype)
            img.fill(0)
            for step in range(dataset.size // chunk_size + 1):
                data = dataset[step * chunk_size: (step + 1) * chunk_size]
                parse_gef_line(data, img, self.dtype)

        return (
            binx,
            min_x,
            min_y,
            img,
        )

    @staticmethod
    def gef_gef_shape(file):
        with h5py.File(file, "r") as fh:
            dataset = fh[f"/geneExp/bin1/expression"]

            if not dataset[...].size:
                clog.error("The sequencing data is empty, please confirm the {} file.".format(file))
                raise Exception("The sequencing data is empty, please confirm the {} file.".format(file))
    
            try:
                binx = int(dataset.attrs["resolution"][0]/dataset.attrs["dnbPitch"][0]) 
            except KeyError:
                binx = int(dataset.attrs["resolution"][0]/500)
                
            min_x, max_x = dataset.attrs["minX"][0], dataset.attrs["maxX"][0]
            min_y, max_y = dataset.attrs["minY"][0], dataset.attrs["maxY"][0]
            width = max_x - min_x + 1
            height = max_y - min_y + 1
            return binx, width, height

    def detect_feature(self, ref: list, chip_size: float):
        """ track lines detection, matrix data: chip area recognition for registration """
        from cellbin2.matrix.box_detect import detect_chip_box
        from cellbin2.matrix.index_points_detect import detect_cross_points

        self._template = detect_cross_points(ref, self._gene_mat)
        self._chip_box = detect_chip_box(self._gene_mat, chip_size)

    def check_standards(self, gs: GeneticStandards):
        # TODO
        #  gs
        pass

    @property
    def template(self, ):
        return self._template

    @property
    def chip_box(self, ):
        return self._chip_box

    @property
    def heatmap(self, ):
        """ gray scale heatmap: for registration """
        return self._gene_mat


def adjust_mask_shape(gef_path, mask_path):
    binx, m_width, m_height = cMatrix.gef_gef_shape(gef_path)
    if binx==1:
        mask = cbimread(mask_path)
        if mask.width == m_width and mask.height == m_height:
            return mask_path
        mask_adjust = mask.trans_image(offset=[0, 0], dst_size=(m_height, m_width))
        path_no_ext, ext = os.path.splitext(mask_path)
        new_path = path_no_ext + "_adjust" + ".tif"
        cbimwrite(new_path, mask_adjust)
    else:
        mask = cbimread(mask_path)
        mask_binx = mask.trans_image(scale=1/binx)
        path_no_ext, ext = os.path.splitext(mask_path)
        
        if mask_binx.width == m_width and mask_binx.height == m_height:
            new_path = path_no_ext + f"_bin{binx}" + ".tif"
            cbimwrite(new_path, mask_binx)
            return new_path
        mask_adjust = mask.trans_image(offset=[0, 0], dst_size=(m_height, m_width))
        new_path = path_no_ext + "_adjust" + ".tif"
        cbimwrite(new_path, mask_adjust)
        
    return new_path


def gem_to_gef(gem_path, gef_path):
    from gefpy.bgef_writer_cy import generate_bgef
    generate_bgef(input_file=gem_path,
                  bgef_file=gef_path,
                  stromics="Transcriptomics",
                  n_thread=8,
                  bin_sizes=[1],
                  )


def save_cell_bin_data(src_path: str, dst_path: str, cell_mask: str):
    from cellbin2.utils.cell_shape import f_main
    """ fetch: single cell data (mask can get from registered image or matrix) """
    src_path = str(src_path)
    dst_path = str(dst_path)
    cell_mask = str(cell_mask)
    from gefpy import cgef_writer_cy
    if src_path.endswith(".gem.gz"):
        gef_path = os.path.join(os.path.dirname(dst_path), os.path.basename(src_path).replace(".gem.gz", ".raw.gef"))
        if os.path.exists(gef_path):
            src_path = gef_path
        else:
            gem_to_gef(src_path, gef_path)
            src_path = gef_path
    if src_path.endswith(".gef"):
        cell_mask = adjust_mask_shape(gef_path=src_path, mask_path=cell_mask)
    cgef_writer_cy.generate_cgef(dst_path, src_path, cell_mask, [256, 256])
    f_main(dst_path)
    return 0


def generate_vis_gef(src_path: str, dst_path):
    bin_sizes = [1, 10, 20, 50, 100, 200, 500]
    from gefpy.bgef_writer_cy import generate_bgef
    generate_bgef(
        input_file=src_path,
        bgef_file=dst_path,
        stromics='geneExp',
        n_thread=8,
        bin_sizes=bin_sizes,
    )


def save_tissue_bin_data(src_path: str, dst_path: str, tissue_mask: str, bin_siz: int = 1):
    """ save: BinN data within the tissue area """
    src_path = str(src_path)
    dst_path = str(dst_path)
    tissue_mask = str(tissue_mask)
    from gefpy.bgef_creater_cy import BgefCreater
    if src_path.endswith(".gef"):
        tissue_mask = adjust_mask_shape(gef_path=src_path, mask_path=tissue_mask)
    bc = BgefCreater()
    path_no_ext, ext = os.path.splitext(dst_path)
    tmp_path = f"{path_no_ext}_tmp{ext}"
    bc.create_bgef(src_path, bin_siz, tissue_mask, tmp_path)
    generate_vis_gef(tmp_path, dst_path)
    os.remove(tmp_path)
    return


def get_tissue_bin_data(file_path: str, tissue_mask: np.ndarray, bin_siz: int = 1):
    """ fetch: BinN data within the tissue area """
    from gefpy.bgef_creater_cy import BgefCreater

    bc = BgefCreater()
    return bc.get_stereo_data(file_path, bin_siz, tissue_mask)


def get_bin_n_data(file_path: str, bin_siz: int = 1):
    """ fetch: BinN data with chip area """
    from stereo.io import read_gef, read_gem

    if file_path.endswith(".gem") or file_path.endswith(".gem.gz"):
        data = read_gem(file_path, bin_type="bins", bin_size=bin_siz)
    elif file_path.endswith(".gef"):
        data = read_gef(file_path, bin_type="bins", bin_size=bin_siz)
    else:
        data = None

    return data


def main():
    from cellbin2.utils.common import TechType
    # import tifffile
    # import cv2 as cv
    #
    # file_path = Path(r'E:\03.users\liuhuanlin\01.data\cellbin2\input\A03599D1\A03599D1.protein.raw.gef')
    #
    # cm = cMatrix()
    # cm.read(file_path=file_path)
    # m = cm.heatmap
    # m = cv.filter2D(m, -1, np.ones((21, 21), np.float32))
    # # cm.get_track_lines(ref=[[240, 300, 330, 390, 390, 330, 300, 240, 420],
    # #                         [240, 300, 330, 390, 390, 330, 300, 240, 420]])
    # # np.savetxt(r'E:\03.users\liuhuanlin\01.data\cellbin2\stitch\A03599D1_gene.txt', cm.track_points)
    # tifffile.imwrite(r'E:\03.users\liuhuanlin\01.data\cellbin2\input\A03599D1\A03599D1.raw.tif', m)
    # # save_cell_bin_data(src_path=r'E:\03.users\liuhuanlin\01.data\cellbin2\input\A03599D1\A03599D1.raw.gef',
    # #                    dst_path=os.path.join(file_path, 'A03599D1.cellbin2.cgef'),
    # #                    cell_mask=os.path.join(file_path, 'mask.tif'))
    # src_file = "/media/Data/dzh/data/cellbin2/test/A03599D1_demo_1/A03599D1_Transcriptomics.tissue.gef"
    # dst_file = "/media/Data/dzh/data/cellbin2/test/A03599D1_demo_1/A03599D1_Transcriptomics.tissue.gef"
    # generate_vis_gef(src_path=src_file, dst_path=dst_file)
    _output_path = "/media/Data/dzh/data/cellbin2/demo_data/C03928D1"
    sn = 'C03928D1'
    p_naming = naming.DumpPipelineFileNaming(chip_no=sn, save_dir=_output_path)
    m_naming = naming.DumpMatrixFileNaming(sn=sn, m_type=TechType.Transcriptomics, save_dir=_output_path)
    src_path: str = "/media/Data/dzh/data/cellbin2/demo_data/C03928D1/C03928D1.gem.gz"
    # ts_cut: str = "/media/Data/dzh/data/cellbin2/test/C04144D5_demo/C04144D5_ssDNA_tissue_cut.tif"
    # dst_path: str = "/media/Data/dzh/data/cellbin2/demo_data/C04144D5/C04144D522_Transcriptomics.cellbin.gef"
    # cell_mask: str = "/media/Data/dzh/data/cellbin2/test/C04144D5_demo/C04144D5_ssDNA_mask.tif"
    # save_cell_bin_data(src_path, dst_path, cell_mask)
    # save_tissue_bin_data(
    #     src_path,
    #     str(m_naming.tissue_bin_matrix),
    #     str(p_naming.final_tissue_mask)
    # )

    save_cell_bin_data(
        src_path,
        str(m_naming.cell_bin_matrix),
        str(p_naming.final_nuclear_mask)
    )
    # generate_vis_gef(src_path,
    #                  "/media/Data/dzh/data/cellbin2/demo_data/C04144D5/C04144D5.gef")


def create_martix_image(args):
    cm = cMatrix()
    cm.read(file_path=Path(args.input))
    matrix_name = os.path.basename(args.input).split('.')[0]
    output = os.path.join(args.output, f"{matrix_name}_Transcriptomics.tif")
    cbimwrite(output, cm._gene_mat)


if __name__ == '__main__':
    # main()
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", action="store", dest="input", type=str, required=True,
                        help="Image file.")
    parser.add_argument("-o", "--output", action="store", dest="output", type=str, required=True,
                        help="Result output dir.")

    parser.set_defaults(func=create_martix_image)
    (para, args) = parser.parse_known_args()
    para.func(para)
