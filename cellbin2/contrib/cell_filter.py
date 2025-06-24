from cellbin2.utils.matrix import cbMatrix
import numpy as np
import pandas as pd
import cv2
import tifffile


class FilterCells(object):
    def __init__(self, geffile: [str, cbMatrix], mask: [str, np.array]):
        if isinstance(geffile, str):
            self.cbm = cbMatrix(geffile)
        elif isinstance(geffile, cbMatrix):
            self.cbm = geffile
        if isinstance(mask, str):
            self._read(mask)
        else:
            self.cellmask = mask
        self._cell_pd = pd.DataFrame()
        self._split_number = None

    @property
    def split_num(self):
        """

        :return: split chip with cell numbers
        """
        if self._split_number == None:
            self._split_number = self.cbm.shape[0] // 5000 + 1
        return self._split_number

    def _read(self, mask):
        self.cellmask = tifffile.imread(mask)

    def _cal_filtercell(self, df, key="area", para=3, save_key="not_singlecell_area"):
        count_data = df[key].copy()
        count_data = count_data.sort_values(ascending=False)
        #### cut off calculate
        MADs = (count_data - count_data.median()).abs().median()
        madian = count_data.median()
        cutoff_filter_max = madian + para * MADs
        # cutoff_filter_min=madian-para*MADs
        df[save_key] = 0
        df[save_key] = (df[key] > cutoff_filter_max)

        df[save_key] = df[save_key].astype(np.int64)
        return df

    def filter_doublecells(self):

        celllist = self.cbm.raw_data.cells.obs.index.totolist()
        celllist.remove("0.0")
        self.cbm.raw_data.tl.filter_cells(cell_list=celllist, inplace=True)
        self._cell_pd = self.cbm.raw_data.cells.obs.copy(deep=True)
        self._cell_pd["x"] = self.cbm.raw_data.position[:, 0]
        self._cell_pd["y"] = self.cbm.raw_data.position[:, 1]
        self._cell_pd["n_counts"] = self.cbm.cell_MID_counts
        if "area" not in self._cell_pd.columns:
            self._cell_pd["area"] = self.get_cellarea(self.cellmask)
        ###split chip with cell numbers
        self._cell_pd = self._split_chipRegion()

        df_list = []
        for i in range(self.split_num):
            for j in range(self.split_num):
                df_list.append(self._cal_filtercell(
                    self._cell_pd[(self._cell_pd["x_area"] == i) & (self._cell_pd["y_area"] == j)], key="area",
                    para=3, save_key="not_singlecell_area"))
        _cell_pd = pd.concat(df_list, axis=0)
        ### cal the doublefinder cells with n_counts
        df_list = []
        for i in range(self.split_num):
            for j in range(self.split_num):
                df_list.append(
                    self._cal_filtercell(_cell_pd[(_cell_pd["x_area"] == i) & (_cell_pd["y_area"] == j)],
                                         key="n_counts", para=3.5, save_key="not_singlecell_n_counts"))
        self._cell_pd = pd.concat(df_list, axis=0)

        ###combine the two factors
        self._cell_pd["not_singlecell_two_factor"] = self._cell_pd["not_singlecell_n_counts"] | self._cell_pd[
            "not_singlecell_area"]
        ## 1 means the cell is not single cell(is double cells)
        return self._cell_pd["not_singlecell_two_factor"].to_frame()

    def _split_chipRegion(self):
        splitx = np.linspace(0, self._cell_pd.x.max(), self.split_num + 1)
        splity = np.linspace(0, self._cell_pd.y.max(), self.split_num + 1)
        # num = 0
        x_dic = {}
        y_dic = {}
        for i in np.arange(self.split_num):
            x_dic[i] = [splitx[i], splitx[i + 1]]
            y_dic[i] = [splity[i], splity[i + 1]]
        self._cell_pd["x_area"] = -1
        self._cell_pd["y_area"] = -1

        for key in x_dic.keys():
            self._cell_pd.loc[
                (self._cell_pd["x"] > x_dic[key][0]) & (self._cell_pd["x"] < x_dic[key][1]), "x_area"] = key

        for key in y_dic.keys():
            self._cell_pd.loc[
                (self._cell_pd["y"] > y_dic[key][0]) & (self._cell_pd["y"] < y_dic[key][1]), "y_area"] = key
        return self._cell_pd

    def result_to_txt(self, save_path=""):
        self._cell_pd["not_singlecell_two_factor"].to_frame().to_csv(save_path, sep="\t", index=True)

    def get_cellarea(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
        cellarea_list = [cv2.contourArea(contours[i]) for i in range(len(contours))]
        return cellarea_list

    @classmethod
    def filter_to_gef(cls, filter_df: pd.DataFrame, cbm: cbMatrix, save_path="", key="not_singlecell_two_factor"):
        """
                用过滤的 结果的文件 过滤原始的.gef or .gem
                :param cbm: 过滤前的
                :param save_path: 过滤后的cellbin结果
                :return:  过滤后的cellbin （cbmatrix）格式
                """
        contain_cell = filter_df[filter_df[key] == 0]
        cbm.raw_data.tl.filter_cells(cell_list=contain_cell, inplace=True)
        from stereo.io import write_mid_gef
        write_mid_gef(cbm.raw_data, save_path)
        return cbm


def filter_pipline(geffile, cellmask, output_filter_file="", output_filter_gef=""):
    """
        过滤算法流程
        :param geffile:  需要过滤的cellbin 单细胞数据 ，可以接受gef，gem
        :param cellmask:  单细胞产生的mask
        :param output_filter_file: 是否过滤的输出文件：输出为 txt， 两列，第一列是细胞名称，第二列是是否删除，1 代表删除，0代表保留
        :param output_filter_gef:  根据过滤文件，产生过滤后的gef
        :return: 过滤后的gef
        """
    fc = FilterCells(geffile, mask=cellmask)
    fc.filter_doublecells()
    fc.result_to_txt(output_filter_file)  ###### 输出为 txt， 两列，第一列是细胞名称，第二列是是否删除，1 代表删除，0代表保留
    df = pd.read_csv(output_filter_file, header=0, sep="\t")
    cbm = cbMatrix(geffile)
    FilterCells.filter_to_gef(df, cbm, output_filter_gef)


def main():
    import os
    path = "Z:\MySyncFiles"
    geffile = os.path.join(path, 'D04167E2.cellbin.txt')  ## example
    cellmask = os.path.join(path, 'D04167E2_mask.tif')
    filter_pipline(geffile, cellmask, output_filter_file=os.path.join(path, "D04167E2.filter.txt"),
                   output_filter_gef=os.path.join(path, "D04167E2.filter.gef"))
    # fc=FilterCells(geffile,mask=cellmask)
    # fc.filter_doublecells()
    # fc
