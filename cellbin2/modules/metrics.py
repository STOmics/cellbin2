import os
from typing import List, Dict, Union
import json
import io
import base64

import cv2
from pathlib import Path
import h5py
from pydantic import BaseModel, Field

from cellbin2.utils.matrix import cbMatrix, BinMatrix, MultiMatrix
from cellbin2.contrib.cell_segmentor import CellSegmentation
from cellbin2.contrib.clarity import ClarityQC
from cellbin2.utils.rpi import readrpi

from cellbin2.utils import ipr
from cellbin2.utils.plot_funcs import template_painting, chip_box_painting
from cellbin2.utils import dict2json
from cellbin2.utils.common import TechType
from cellbin2.modules.naming import DumpPipelineFileNaming
from cellbin2.contrib.alignment.basic import ChipBoxInfo


# define all the input parameters required by metrics, use this method to add other parameters in the future
class MatrixArray(BaseModel):
    bin1_matrix: str = Field(..., description="path for original Bin1 matrix")
    tissue_bin_matrix: str = Field(..., description="path for TissueBin matrix")
    cell_bin_matrix: str = Field(..., description="absolute path for raw CellBin matrix")
    cell_bin_adjusted_matrix: str = Field(..., description="absolute path for adjusted CellBin matrix")
    matrix_type: TechType = Field(..., description="sequencing matrix technology types, including Transcriptomics and Protein")


class ImageSource(BaseModel):
    stitch_image: str = Field(..., description="path for stitch image")
    cell_mask: str = Field(default=..., description="cell mask")
    registration_image: str = Field(default=..., description="registration image")
    tissue_mask: str = Field(default=..., description="tissue mask")
    cell_correct_mask: str = Field(default=..., description="cell correct mask")


class FileSource(BaseModel):
    ipr_file: str = Field(..., description="absolute path of image analysis record file")
    rpi_file: str = Field(..., description="absolute path of the image pyramid file")
    matrix_list: List[MatrixArray] = Field(..., description="list of new matrices generated from multiple extraction methods applied to each Bin1 matrix")
    sn: str = Field(..., description="chip id for the gef files")
    image_dict: Dict[str, ImageSource] = Field(..., description="cellbin2 file must be named the same as ipr")


BIN_OUTPUT_VIEW = 10


class Metrics(object):
    """ Metrics: aggregate and compute module-specific metrics from pipeline results for report generation """

    def __init__(self, filesource: FileSource, output_path: str):
        self.filesource = filesource
        self._RNAmultiMatrix = None
        self._ProteinmultiMatrix = None
        self._output_path = output_path
        self._dics = []  ## cellbin2. tissue class to operation
        self._rpi_bin = BIN_OUTPUT_VIEW
        self.pipe_naming = DumpPipelineFileNaming(self.filesource.sn, save_dir=self.output_path)
        self.output_data = {"matrix": {"RNA": {"statistics": {"CellBin": {}, "Adjusted": {}},
                                               "distribution": {"CellBin": {}, "Adjusted": {}},
                                               "cluster": {"spatial": [], "umap": [], "category": []},
                                               "heatmap": {"rawbin": {}, "tissuebin": {}}
                                               },
                                       "Protein": {"statistics": {"CellBin": {}, "Adjusted": {}},
                                                   "distribution": {"CellBin": {}, "Adjusted": {}},
                                                   "cluster": {"spatial": [], "umap": [], "category": []},
                                                   "heatmap": {"rawbin": {}, "tissuebin": {}}
                                                   }
                                       },
                            "image": {"register_img": {}, "param": {"x_star": {}, "y_star": {},
                                                                    "sizex": {}, "sizey": {}}},
                            "image_ipr": {},
                            "tissue_img": {},
                            "infor": {"sampleId": self.filesource.sn, "pipelineVersion": "CellBin V2"}}
        ### for stereo-seq transcriptomics and protein
        if os.path.exists(self._output_path):
            print("-" * 50)
            self.output_figure_path = os.path.join(self._output_path, "assets")
            print(f"Creat report required path in {self.output_figure_path}")
            ### RNA menu
            self.output_figure_path_rna_cellbin = os.path.join(self.output_figure_path, "rna", "cellbin")
            os.makedirs(self.output_figure_path_rna_cellbin, exist_ok=True)
            self.output_figure_path_rna_adjusted = os.path.join(self.output_figure_path, "rna", "adjusted")
            os.makedirs(self.output_figure_path_rna_adjusted, exist_ok=True)
            ### protein menu
            self.output_figure_path_protein_cellbin = os.path.join(self.output_figure_path, "protein", "cellbin")
            self.output_figure_path_protein_adjusted = os.path.join(self.output_figure_path, "protein", "adjusted")
            os.makedirs(self.output_figure_path_protein_cellbin, exist_ok=True)
            os.makedirs(self.output_figure_path_protein_adjusted, exist_ok=True)
            print("-" * 50)
            ### image menu
            self.output_figure_path_image = os.path.join(self.output_figure_path, "image")
            os.makedirs(self.output_figure_path_image, exist_ok=True)
            # temperary menu
            self.output_tmp_dir = os.path.join(self.output_figure_path_image, "tmp")
            os.makedirs(self.output_tmp_dir, exist_ok=True)
        else:
            raise Exception("Output path is not exist")
        self.load()

    def load(self):
        if len(self.filesource.matrix_list) > 0:
            for i in self.filesource.matrix_list:

                if i.matrix_type == TechType.Transcriptomics:
                    self._RNAmultiMatrix = MultiMatrix(cellbin_path=i.cell_bin_matrix,
                                                       tissuegef_path=i.tissue_bin_matrix, raw_path=i.bin1_matrix,
                                                       adjusted_path=i.cell_bin_adjusted_matrix,
                                                       matrix_type=i.matrix_type)
                elif i.matrix_type == TechType.Protein:
                    self._ProteinmultiMatrix = MultiMatrix(cellbin_path=i.cell_bin_matrix,
                                                           tissuegef_path=i.tissue_bin_matrix, raw_path=i.bin1_matrix,
                                                           adjusted_path=i.cell_bin_adjusted_matrix,
                                                           matrix_type=i.matrix_type)
                else:
                    raise Exception("worng .gef file type")

    def set_report_para(self):
        self.set_statistic_data()  ### matrix statistic data to self._json
        self.set_distribution_fig()  ### the distribution of MID,cell area, gene_type to self.output_data
        self.set_cellbin_scatterplot()  #### plot cellbin2 MID and cell-density scatterplot
        self.set_image_list()  ### heatmap need original images for every strain.
        self.create_heatmap_list()  #### raw bins and tissue-cut bins to create heatmap figures.
        self.set_cluster_data()
        ###### image report json data
        self.set_image_infor()  ##### set ipr statistic data
        self.set_cellseg_data()
        self.set_trackpoint_chipbox()
        pass

    def set_image_list(self):
        if self.filesource.rpi_file == "":
            return
        elif not os.path.exists(self.filesource.rpi_file):
            return
        else:
            h5 = h5py.File(self.filesource.rpi_file, 'r')
            self._set_image_param(h5)
            key_list = list(h5.keys())
            key_list.remove("metaInfo")
            for staintype in key_list:
                if staintype == 'final':
                    continue
                ## regist_img
                regist_img = readrpi(h5, bin_size=self._rpi_bin, staintype=staintype, tType="Image")
                self.output_data["image"]["register_img"][staintype] = self.image_array_to_base64(regist_img)
                ## tissue_img
                if "TissueMask" in h5[staintype]:
                    tissue_img = readrpi(h5, bin_size=self._rpi_bin, staintype=staintype, tType="TissueMask")
            self.output_data["tissue_img"] = self.image_array_to_base64(tissue_img)
            h5.close()

    def _set_image_param(self, h5):
        data_dic = {}
        for key, value in h5["metaInfo"].attrs.items():
            data_dic[key] = value
        self.output_data["image"]["param"]["x_star"] = str(data_dic["x_start"])
        self.output_data["image"]["param"]["y_star"] = str(data_dic["y_start"])
        self.output_data["image"]["param"]["sizex"] = str(data_dic["sizex"])
        self.output_data["image"]["param"]["sizey"] = str(data_dic["sizey"])

    def create_heatmap_list(self):
        if not self._RNAmultiMatrix == None:
            ### RNA cellbin2 data
            if self._RNAmultiMatrix.rawbin is not None:
                img_base64, colorbar_base64 = self._RNAmultiMatrix.rawbin.create_heatmap(
                    self._RNAmultiMatrix.rawbin.MID_counts)
                self.output_data["matrix"]["RNA"]["heatmap"]["rawbin"]["img"] = img_base64
                self.output_data["matrix"]["RNA"]["heatmap"]["rawbin"]["colorbar"] = colorbar_base64
            if self._RNAmultiMatrix.tissuebin is not None:
                # self._RNAmultiMatrix.tissuebin.reset(10)
                img_base64, colorbar_base64 = self._RNAmultiMatrix.tissuebin.create_heatmap(
                    self._RNAmultiMatrix.tissuebin.MID_counts)
                self.output_data["matrix"]["RNA"]["heatmap"]["tissuebin"]["img"] = img_base64
                self.output_data["matrix"]["RNA"]["heatmap"]["tissuebin"]["colorbar"] = colorbar_base64
        if not self._ProteinmultiMatrix == None:
            ### protein cellbin2 data
            if self._ProteinmultiMatrix.rawbin is not None:
                img_base64, colorbar_base64 = self._ProteinmultiMatrix.rawbin.create_heatmap(
                    self._ProteinmultiMatrix.rawbin.MID_counts)
                self.output_data["matrix"]["Protein"]["heatmap"]["rawbin"]["img"] = img_base64
                self.output_data["matrix"]["Protein"]["heatmap"]["rawbin"]["colorbar"] = colorbar_base64
            if self._ProteinmultiMatrix.tissuebin is not None:
                # self._RNAmultiMatrix.tissuebin.reset(10)
                img_base64, colorbar_base64 = self._ProteinmultiMatrix.tissuebin.create_heatmap(
                    self._ProteinmultiMatrix.tissuebin.MID_counts)
                self.output_data["matrix"]["Protein"]["heatmap"]["tissuebin"]["img"] = img_base64
                self.output_data["matrix"]["Protein"]["heatmap"]["tissuebin"]["colorbar"] = colorbar_base64

    def save_json_file(self, save_file):
        # json_data = json.dumps(self.output_data)
        # with open(save_file, "w") as file:
        #     file.write(json_data)
        dict2json(data=self.output_data, json_path=save_file)

    def set_statistic_data(self):
        if not self._RNAmultiMatrix == None:
            ### RNA cellbin2 data
            if not self._RNAmultiMatrix.cellbin == None:
                if not self._RNAmultiMatrix.tissuebin == None:
                    statistic_data = self._get_c_statistics(self._RNAmultiMatrix.cellbin,
                                                            self._RNAmultiMatrix.tissuebin)
                else:
                    statistic_data = self._get_c_statistics(self._RNAmultiMatrix.cellbin,
                                                            self._RNAmultiMatrix.rawbin)
                self.output_data["matrix"]["RNA"]["statistics"]["CellBin"] = statistic_data
            ##### RNA adjusted data
            if not self._RNAmultiMatrix.adjustedbin == None:
                if not self._RNAmultiMatrix.tissuebin == None:
                    statistic_data = self._get_c_statistics(self._RNAmultiMatrix.adjustedbin,
                                                            self._RNAmultiMatrix.tissuebin)
                else:
                    statistic_data = self._get_c_statistics(self._RNAmultiMatrix.adjustedbin,
                                                            self._RNAmultiMatrix.rawbin)
                self.output_data["matrix"]["RNA"]["statistics"]["Adjusted"] = statistic_data
        if not self._ProteinmultiMatrix == None:
            ### Protein cellbin2 data
            if not self._ProteinmultiMatrix.cellbin == None:
                if not self._ProteinmultiMatrix.tissuebin == None:
                    statistic_data = self._get_c_statistics(self._ProteinmultiMatrix.cellbin,
                                                            self._ProteinmultiMatrix.tissuebin)
                else:
                    statistic_data = self._get_c_statistics(self._ProteinmultiMatrix.cellbin,
                                                            self._ProteinmultiMatrix.rawbin)
                self.output_data["matrix"]["Protein"]["statistics"]["CellBin"] = statistic_data
            ### Protein adjusted data
            if not self._ProteinmultiMatrix.adjustedbin == None:
                if not self._ProteinmultiMatrix.tissuebin == None:
                    statistic_data = self._get_c_statistics(self._ProteinmultiMatrix.adjustedbin,
                                                            self._ProteinmultiMatrix.tissuebin)
                else:
                    statistic_data = self._get_c_statistics(self._ProteinmultiMatrix.adjustedbin,
                                                            self._ProteinmultiMatrix.rawbin)
                self.output_data["matrix"]["Protein"]["statistics"]["Adjusted"] = statistic_data
        return 0

    def _get_c_statistics(self, cellbin: cbMatrix, tissbin: BinMatrix):
        statistics_data = {}
        statistics_data["cellCount"] = str(cellbin.get_cellcount())
        statistics_data["meanCellArea"], statistics_data["medianCellArea"] = cellbin.get_cellarea()
        statistics_data["meanGeneType"], statistics_data["medianGeneType"] = cellbin.get_genetype()
        statistics_data["meanMID"], statistics_data["medianMID"] = cellbin.get_MID()
        statistics_data["Fraction_cells_gene"] = r"{:.1f}%".format(cellbin.get_faction_cell_gene(threshod=200) * 100)
        cell_to_tissue_MID = cellbin.get_total_MID() / tissbin.get_total_MID()
        statistics_data["cell_to_tissue_MID"] = r"{:.1f}%".format(cell_to_tissue_MID * 100)
        return statistics_data

    def set_distribution_fig(self):
        if self._RNAmultiMatrix is not None:
            if self._RNAmultiMatrix.cellbin is not None:
                datadict = self._RNAmultiMatrix.cellbin.plot_statistic_vio(self.output_figure_path_rna_cellbin)
                self.output_data["matrix"]["RNA"]["distribution"]["CellBin"]["MID_data"] = datadict["MID"]
                self.output_data["matrix"]["RNA"]["distribution"]["CellBin"]["celldiameter_data"] = datadict[
                    "celldiameter"]
                self.output_data["matrix"]["RNA"]["distribution"]["CellBin"]["genetype_data"] = datadict["genetype"]
            if self._RNAmultiMatrix.adjustedbin is not None:
                datadict = self._RNAmultiMatrix.adjustedbin.plot_statistic_vio(
                    self.output_figure_path_rna_adjusted)
                self.output_data["matrix"]["RNA"]["distribution"]["Adjusted"]["MID_data"] = datadict["MID"]
                self.output_data["matrix"]["RNA"]["distribution"]["Adjusted"]["celldiameter_data"] = datadict[
                    "celldiameter"]
                self.output_data["matrix"]["RNA"]["distribution"]["Adjusted"]["genetype_data"] = datadict["genetype"]
            if self._ProteinmultiMatrix is not None and self._ProteinmultiMatrix.cellbin is not None:
                datadict = self._ProteinmultiMatrix.cellbin.plot_statistic_vio(self.output_figure_path_protein_cellbin)
                self.output_data["matrix"]["Protein"]["distribution"]["CellBin"]["MID_data"] = datadict["MID"]
                self.output_data["matrix"]["Protein"]["distribution"]["CellBin"]["celldiameter_data"] = datadict["MID"]
                self.output_data["matrix"]["Protein"]["distribution"]["CellBin"]["genetype_data"] = datadict["MID"]
            if self._ProteinmultiMatrix is not None and self._ProteinmultiMatrix.adjustedbin is not None:
                datadict = self._ProteinmultiMatrix.adjustedbin.plot_statistic_vio(
                    self.output_figure_path_protein_adjusted)
                self.output_data["matrix"]["Protein"]["distribution"]["Adjusted"]["MID_data"] = datadict["MID"]
                self.output_data["matrix"]["Protein"]["distribution"]["Adjusted"]["celldiameter_data"] = datadict["MID"]
                self.output_data["matrix"]["Protein"]["distribution"]["Adjusted"]["genetype_data"] = datadict["MID"]

        pass

    def image_array_to_base64(self, img_array):
        png_path = os.path.join(self.output_path,'temp.png')
        if len(img_array.shape) == 3 and img_array.shape[-1] == 3:
            img_array = img_array[:, :, (2, 1, 0)]
        cv2.imwrite(png_path, img_array)
        with open(png_path, "rb") as f:
            img_b = f.read()
            b = io.BytesIO(img_b)

        os.remove(png_path)
        return 'data:image/png;base64,{}'.format(base64.b64encode(b.getvalue()).decode())

    def set_cellbin_scatterplot(self):
        """
        plot cellbin2 MID and cell-density scatterplot
        :return:  0
        """
        if self._RNAmultiMatrix is not None:
            if self._RNAmultiMatrix.cellbin is not None:
                self._RNAmultiMatrix.cellbin.plot_spatail_figure(key="celldensity",
                                                                 save_path=self.output_figure_path_rna_cellbin)
                cmin, cmax = self._RNAmultiMatrix.cellbin.plot_spatail_figure(key="CellBin_MID_counts",
                                                                              save_path=self.output_figure_path_rna_cellbin)
            if self._RNAmultiMatrix.adjustedbin is not None:
                self._RNAmultiMatrix.adjustedbin.plot_spatail_figure(key="Adjusted_Gene_type", colormin=cmin,
                                                                     colormax=cmax,
                                                                     save_path=self.output_figure_path_rna_adjusted)
        if self._ProteinmultiMatrix is not None:
            if self._ProteinmultiMatrix.cellbin is not None:
                self._ProteinmultiMatrix.cellbin.plot_spatail_figure(key="celldensity",
                                                                     save_path=self.output_figure_path_protein_cellbin)
                cmin, cmax = self._ProteinmultiMatrix.cellbin.plot_spatail_figure(key="CellBin_MID_counts",
                                                                                  save_path=self.output_figure_path_protein_cellbin)
            if self._ProteinmultiMatrix.adjustedbin is not None:
                self._ProteinmultiMatrix.adjustedbin.plot_spatail_figure(key="Adjusted_Gene_type", colormin=cmin,
                                                                         colormax=cmax,
                                                                         save_path=self.output_figure_path_protein_adjusted)

    def set_cluster_data(self):
        def _set_df_tojson(df, matrix_type="RNA"):
            self.output_data["matrix"][matrix_type]["cluster"]["umap"] = [list(df.umap_0.round(2).astype(float).values),
                                                                          list(df.umap_1.round(2).astype(float).values)]
            self.output_data["matrix"][matrix_type]["cluster"]["spatial"] = [
                (df.x.to_numpy() // self._rpi_bin).astype(float).tolist(),
                (df.y.to_numpy() // self._rpi_bin).astype(float).tolist()]
            self.output_data["matrix"][matrix_type]["cluster"]["category"] = df.leiden.tolist()

        if self._RNAmultiMatrix is not None:
            if self._RNAmultiMatrix.adjustedbin is not None:
                df = self._RNAmultiMatrix.adjustedbin.get_cluster_data(reset=True)
                _set_df_tojson(df, matrix_type="RNA")
            elif self._RNAmultiMatrix.cellbin:
                df = self._RNAmultiMatrix.cellbin.get_cluster_data(reset=True)
                _set_df_tojson(df, matrix_type="RNA")
        if self._ProteinmultiMatrix is not None:
            if self._ProteinmultiMatrix.adjustedbin is not None:
                df = self._ProteinmultiMatrix.adjustedbin.get_cluster_data(reset=True)
                _set_df_tojson(df, matrix_type="Protein")
            elif self._ProteinmultiMatrix.cellbin:
                df = self._ProteinmultiMatrix.cellbin.get_cluster_data(reset=True)
                _set_df_tojson(df, matrix_type="Protein")

    def set_image_infor(self):
        if self.filesource.ipr_file == "":
            return
        else:
            _ipr, channel_images = ipr.read(self.filesource.ipr_file)
            # if len(self.filesource.image_dict) != len(_ipr.layers):  TODO: turn off, test single staining image first
            #     raise Exception(
            #         f"the stain file is {len(self.filesource.image_dict)}, which is not match .ipr file number")
            # (self.output_data["image_ipr"]["ManualState"],
            #  self.output_data["image_ipr"]["StereoResepSwitch"]) = _ipr.ManualState_StereoResepSwitch
            self.output_data["image_ipr"]["ManualState"] = _ipr.ManualState.__dict__
            self.output_data["image_ipr"]["StereoResepSwitch"] = _ipr.StereoResepSwitch.__dict__
            for c_name, c_info in channel_images.items():
                layer = c_name
                self.output_data["image_ipr"][layer] = {}
                self.output_data["image_ipr"][layer]["image_info"] = {key.lower(): value for key, value in
                                                                      c_info.ImageInfo.get_attrs().items()}
                self.output_data["image_ipr"][layer]["image_info"]["image_size"] = \
                    os.path.getsize(self.filesource.image_dict[layer].stitch_image) * 8 / 1000000
                self.output_data["image_ipr"][layer]["QC_info"] = {key.lower(): value for key, value in
                                                                   c_info.QCInfo.get_attrs().items()}
                self.output_data["image_ipr"][layer]["register_info"] = {key.lower(): value for key, value in
                                                                         c_info.Register.get_attrs().items()}
                #### Image clarity fig
                clarity_matrix = c_info.QCInfo.ClarityPreds
                if len(clarity_matrix) != 0:
                    clarity_arr = ClarityQC.post_process(clarity_matrix)
                    from PIL import Image
                    img = Image.fromarray(clarity_arr)
                    clarity_fig_path = os.path.join(self.output_figure_path_image, f"{layer}_clarity.png")
                    img.save(os.path.join(clarity_fig_path))
                    self.output_data["image_ipr"][layer]["clarity"] = os.path.relpath(clarity_fig_path,
                                                                                      self._output_path)

    def set_cellseg_data(self):
        def _set_imagedict(src, outline=[]):
            _dict = {}
            _dict["src"] = 'data:image/png;base64,{}'.format(base64.b64encode(src.getvalue()).decode())
            _dict["outline"] = outline
            return _dict

        for layer in self.filesource.image_dict.keys():
            if not os.path.exists(self.filesource.image_dict[layer].cell_mask):
                continue
            area_ratio, area_ratio_cor, int_ratio, cell_with_outline, fig \
                = CellSegmentation.get_stats(
                c_mask_p=self.filesource.image_dict[layer].cell_mask,
                cor_mask_p=self.filesource.image_dict[layer].cell_correct_mask,
                t_mask_p=self.filesource.image_dict[layer].tissue_mask,
                register_img_p=self.filesource.image_dict[layer].registration_image,
            )
            self.output_data["image_ipr"][layer]["cellseg"] = {}
            self.output_data["image_ipr"][layer]["cellseg"]["area_ratio"] = str(round(area_ratio, 3) * 100) + "%"
            self.output_data["image_ipr"][layer]["cellseg"]["area_ratio_cor"] = str(
                round(area_ratio_cor, 3) * 100) + "%"
            self.output_data["image_ipr"][layer]["cellseg"]["int_ratio"] = str(round(int_ratio, 3) * 100) + "%"
            self.output_data["image_ipr"][layer]["cellseg"]["images"] = []
            for i in cell_with_outline:
                tmp_save_p = os.path.join(self.output_tmp_dir, 'temp.png')
                cv2.imwrite(tmp_save_p, i[0])
                with open(tmp_save_p, "rb") as f:
                    img_c = f.read()
                    c = io.BytesIO(img_c)  # color bar
                os.remove(tmp_save_p)
                outline = [j for j in i[1]]
                self.output_data["image_ipr"][layer]["cellseg"]["images"].append(_set_imagedict(c, outline))
            cell_intensity_name = os.path.join(self.output_figure_path_image, f"{layer}_cell_intensity.png")
            fig.savefig(cell_intensity_name)
            fig.savefig(tmp_save_p)
            # with open('./temp.png', "rb") as f:
            #     img_c = f.read()
            #     c = io.BytesIO(img_c)  # color bar
            os.remove(tmp_save_p)
            # d='data:image/png;base64,{}'.format(base64.b64encode(c.getvalue()).decode())
            self.output_data["image_ipr"][layer]["cellseg"]["cell_intensity"] = os.path.relpath(cell_intensity_name,
                                                                                                self._output_path)

    def set_trackpoint_chipbox(self):
        for layer in self.filesource.image_dict.keys():
            if layer not in ['HE', 'DAPI', 'ssDNA']:
                continue
            #### plot trackpoint figure
            with h5py.File(self.filesource.ipr_file, "r") as f:
                template_points = f[layer]["Register"]["RegisterTemplate"][...]
                track_points = f[layer]["Register"]["RegisterTrackTemplate"][...]
                if template_points.size == 0:
                    continue
                img, cp_image_list, tissue_image_list = template_painting(
                    image_data=self.filesource.image_dict[layer].registration_image,
                    tissue_seg_data=self.filesource.image_dict[layer].tissue_mask,
                    image_type=layer,
                    qc_points=track_points,
                    template_points=template_points, )
                trackpoint_name = os.path.join(self.output_figure_path_image, f"{layer}_trackpoint.png")
                cv2.imwrite(trackpoint_name, img)
                for i in range(len(cp_image_list)):
                    tmp_cp_image = cp_image_list[i]
                    tmp_cp_image_name = os.path.join(self.output_figure_path_image,
                                                     f"{layer}_trackpoint_cp_image_{i + 1}.png")
                    cv2.imwrite(tmp_cp_image_name, tmp_cp_image)
                    tmp_tissue_image = tissue_image_list[i]
                    tmp_tissue_image_name = os.path.join(self.output_figure_path_image,
                                                         f"{layer}_trackpoint_tissue_image_{i + 1}.png")
                    cv2.imwrite(tmp_tissue_image_name, tmp_tissue_image)

                    self.output_data["image_ipr"][layer][f"trackpoint_cp_image_{i + 1}"] = os.path.relpath(
                        tmp_cp_image_name, self._output_path)
                    self.output_data["image_ipr"][layer][f"trackpoint_tissue_image_{i + 1}"] = os.path.relpath(
                        tmp_tissue_image_name, self._output_path)

                # img.save(trackpoint_name)
                self.output_data["image_ipr"][layer]["trackpoint"] = os.path.relpath(trackpoint_name, self._output_path)
                #### plot chipbox figure
                # img=chip_box_painting(self.filesource.image_dict[layer]["Stitch"],ipr_path=self.filesource.ipr_file,layer=layer)
                # chipbox_painting=os.path.join(self.output_figure_path_image, f"{layer}_chipbox.png")
                # img.save(chipbox_painting)
                # self.output_data["image_ipr"][layer]["chipbox"] = chipbox_painting

                # trackpoint_name = os.path.join(self.output_figure_path_image, f"{layer}_trackpoint.png")
                # self.output_data["image_ipr"][layer]["trackpoint"] = os.path.relpath(trackpoint_name,
                #                                                                      self._output_path)
                chipbox_painting_path = os.path.join(self.output_figure_path_image, f"{layer}_chipbox.png")
                tmp_chipbox_info = ChipBoxInfo()
                tmp_chipbox_info.LeftTop = f[layer]["QCInfo"]["ChipBBox"]['LeftTop'][...]
                tmp_chipbox_info.LeftBottom = f[layer]["QCInfo"]["ChipBBox"]['LeftBottom'][...]
                tmp_chipbox_info.RightBottom = f[layer]["QCInfo"]["ChipBBox"]['RightBottom'][...]
                tmp_chipbox_info.RightTop = f[layer]["QCInfo"]["ChipBBox"]['RightTop'][...]

                _image, chipbox_part_image_lists = chip_box_painting(image_data=self.filesource.image_dict[layer].stitch_image,
                                        chip_info=tmp_chipbox_info,
                                        layer=layer,
                                        draw_thickness=3)
                cv2.imwrite(chipbox_painting_path, _image)
                self.output_data["image_ipr"][layer]["chipbox"] = os.path.relpath(chipbox_painting_path, self._output_path)
                for i in range(len(chipbox_part_image_lists)):
                    tmp_chipbox_part_image = chipbox_part_image_lists[i]
                    tmp_chipbox_part_image_path = os.path.join(self.output_figure_path_image,f"{layer}_chipbox_part_image_{i + 1}.png")
                    cv2.imwrite(tmp_chipbox_part_image_path, tmp_chipbox_part_image)
                    self.output_data["image_ipr"][layer][f"chipbox_part_image_{i + 1}"] = os.path.relpath(tmp_chipbox_part_image_path, self._output_path)

    @property
    def output_path(self):
        return self._output_path


def calculate(param: FileSource, output_path: str):
    """
    :param param: CellBin result files (multiple)
    :param output_path: temporary and static files generated after metrics calculation, for report generation 
    :return: None
    """
    pass
    # TODO: zhangying
    mcs = Metrics(param, output_path=output_path)
    mcs.set_report_para()
    mcs.save_json_file(os.path.join(mcs.pipe_naming.metrics))  # unified naming management


def main():
    from glob import glob
    main_s_type = "ssDNA"
    # path = r"/media/Data/wqs/hedongdong/tissue_segmentation/cellbin2_test/report_test_data/SS200000135TL_D1_demo"
    path = r"F:\01.users\hedongdong\cellbin2_test\report_result\pipline\SS200000135TL_D1"
    sn = "SS200000135TL_D1"  ###chip id
    ipr_file = glob(os.path.join(path, f"**.ipr"))[0]
    rpi_file = glob(os.path.join(path, f"**.rpi"))[0]
    tissue_gef = glob(os.path.join(path, f"**.tissue.gef"))[0]
    cell_gef = glob(os.path.join(path, "**.cellbin.gef"))[0]
    cell_adjust_gef = glob(os.path.join(path, "**.adjusted.cellbin.gef"))[0]
    raw_gef = r"F:\01.users\hedongdong\cellbin2_test\report_test_data\SS200000135TL_D1.raw.gef"
    stitch = r"F:\01.users\hedongdong\cellbin2_test\report_test_data\SS200000135TL_D1_fov_stitched_ssDNA.tif"
    cell_mask = glob(os.path.join(path, f"**{main_s_type}_mask.tif"))[0]
    regist = glob(os.path.join(path, "**ssDNA_regist.tif"))[0]
    tissue_mask = glob(os.path.join(path, f"**{main_s_type}_tissue_cut.tif"))[0]
    cell_adjust_mask = glob(os.path.join(path, "**_mask_edm_dis_10.tif"))[0]
    m1 = MatrixArray(
        tissue_bin_matrix=tissue_gef,
        cell_bin_matrix=cell_gef,
        cell_bin_adjusted_matrix=cell_adjust_gef,
        bin1_matrix=raw_gef,
        matrix_type=TechType.Transcriptomics
    )
    # m2 = MatrixArray(
    #     tissue_bin_matrix=tissue_gef,
    #     cell_bin_matrix=cell_gef,
    #     cell_bin_adjusted_matrix=cell_adjust_gef,
    #     bin1_matrix=raw_gef,
    #     matrix_type=TechType.Protein)

    image_dict = {
        main_s_type:
            {
                "Stitch": stitch,
                'CellMask': cell_mask,
                'Image': regist,
                'TissueMask': tissue_mask,
                "AdjustedMask": cell_adjust_mask
            }
    }

    imagesource = ImageSource(
        stitch_image=stitch,
        cell_mask=cell_mask,
        registration_image=regist,
        tissue_mask=tissue_mask,
        cell_correct_mask=cell_adjust_mask
    )

    image_dict = {
        main_s_type: imagesource
    }
    # fs = FileSource(ipr_file=ipr_file, rpi_file=rpi_file, matrix_list=[m1])
    fs = FileSource(ipr_file=ipr_file, rpi_file=rpi_file, matrix_list=[m1], sn=sn, image_dict=image_dict)

    output_path = r"F:\01.users\hedongdong\cellbin2_test\report_result\pipline\report"
    calculate(param=fs, output_path=output_path)


if __name__ == '__main__':
    main()
