from cellbin2.utils.matrix import cbMatrix
from cellbin2.image import cbimwrite
import numpy as np
import pandas as pd
import cv2
import tifffile
import os
from pathlib import Path


class FilterCells(object):
    def __init__(self, geffile: [str, cbMatrix], mask: [str, np.ndarray]):
        if isinstance(geffile, str):
            self.cbm = cbMatrix(geffile)
        elif isinstance(geffile, cbMatrix):
            self.cbm = geffile
        else:
            raise TypeError("geffile must be str or cbMatrix")

        if isinstance(mask, str):
            self._read(mask)
        else:
            self.cellmask = mask

        self._cell_pd = pd.DataFrame()
        self._split_number = None

    @property
    def split_num(self):
        """
        split chip with cell numbers
        """
        if self._split_number is None:
            self._split_number = self.cbm.shape[0] // 5000 + 1
        return self._split_number

    def _read(self, mask):
        self.cellmask = tifffile.imread(mask)

    def _cal_filtercell(self, df, key="area", para=3, save_key="not_singlecell_area"):
        if len(df) == 0:
            return df

        count_data = df[key].copy().sort_values(ascending=False)
        mads = (count_data - count_data.median()).abs().median()
        median = count_data.median()
        cutoff_filter_max = median + para * mads

        df = df.copy()
        df[save_key] = (df[key] > cutoff_filter_max).astype(np.int64)
        return df

    def filter_doublecells(self):
        celllist = self.cbm.raw_data.cells.obs.index.tolist()
        self.cbm.raw_data.tl.filter_cells(cell_list=celllist, inplace=True)

        self._cell_pd = self.cbm.raw_data.cells.obs.copy(deep=True)

        # obs.index may be duplicated or all 0; use row order as internal cell id.
        cell_ids = pd.Index(np.arange(len(self._cell_pd), dtype=np.int64))
        self._cell_pd["cell_id_int"] = cell_ids.to_numpy()

        # Coordinates from cbMatrix/stereo object (normal gef/gem matrix;
        # encrypted-matrix cellid.bin path removed).
        obs = self._cell_pd

        if {"x", "y"}.issubset(set(obs.columns)):
            self._cell_pd["x"] = obs["x"].astype(np.int64).to_numpy()
            self._cell_pd["y"] = obs["y"].astype(np.int64).to_numpy()

        elif {"cell_x", "cell_y"}.issubset(set(obs.columns)):
            self._cell_pd["x"] = obs["cell_x"].astype(np.int64).to_numpy()
            self._cell_pd["y"] = obs["cell_y"].astype(np.int64).to_numpy()

        elif hasattr(self.cbm.raw_data, "position"):
            pos = self.cbm.raw_data.position

            if isinstance(pos, pd.DataFrame):
                if {"x", "y"}.issubset(set(pos.columns)):
                    self._cell_pd["x"] = pos["x"].astype(np.int64).to_numpy()
                    self._cell_pd["y"] = pos["y"].astype(np.int64).to_numpy()
                elif pos.shape[1] >= 2:
                    self._cell_pd["x"] = pos.iloc[:, 0].astype(np.int64).to_numpy()
                    self._cell_pd["y"] = pos.iloc[:, 1].astype(np.int64).to_numpy()
                else:
                    raise ValueError(
                        f"Unsupported raw_data.position DataFrame columns: "
                        f"columns={list(pos.columns)}"
                    )

            elif isinstance(pos, dict):
                if "x" not in pos or "y" not in pos:
                    raise ValueError(
                        f"Unsupported raw_data.position dict keys: "
                        f"keys={list(pos.keys())}"
                    )
                self._cell_pd["x"] = np.asarray(pos["x"]).astype(np.int64)
                self._cell_pd["y"] = np.asarray(pos["y"]).astype(np.int64)

            elif isinstance(pos, np.ndarray) and pos.dtype.names is not None:
                if "x" in pos.dtype.names and "y" in pos.dtype.names:
                    self._cell_pd["x"] = pos["x"].astype(np.int64)
                    self._cell_pd["y"] = pos["y"].astype(np.int64)
                else:
                    raise ValueError(
                        f"Unsupported raw_data.position dtype names: "
                        f"dtype.names={pos.dtype.names}"
                    )

            elif isinstance(pos, np.ndarray) and pos.ndim == 2 and pos.shape[1] >= 2:
                self._cell_pd["x"] = pos[:, 0].astype(np.int64)
                self._cell_pd["y"] = pos[:, 1].astype(np.int64)

            else:
                raise ValueError(
                    f"Unsupported raw_data.position type: "
                    f"type={type(pos)}, dtype={getattr(pos, 'dtype', None)}"
                )

        else:
            raise ValueError(
                "No coordinate columns found in cbMatrix. "
                f"obs columns={list(obs.columns)}"
            )

        self._cell_pd["n_counts"] = self.cbm.cell_MID_counts

        if "area" not in self._cell_pd.columns:
            self._cell_pd["area"] = self.get_cellarea(self.cellmask)

        self._cell_pd = self._split_chipRegion()

        df_list = []
        for i in range(self.split_num):
            for j in range(self.split_num):
                sub = self._cell_pd[
                    (self._cell_pd["x_area"] == i) & (self._cell_pd["y_area"] == j)
                ].copy()
                if len(sub) == 0:
                    continue
                df_list.append(
                    self._cal_filtercell(
                        sub,
                        key="area",
                        para=3,
                        save_key="not_singlecell_area"
                    )
                )

        if len(df_list) == 0:
            raise ValueError("No cells assigned to chip regions when filtering by area.")

        _cell_pd = pd.concat(df_list, axis=0)

        df_list = []
        for i in range(self.split_num):
            for j in range(self.split_num):
                sub = _cell_pd[
                    (_cell_pd["x_area"] == i) & (_cell_pd["y_area"] == j)
                ].copy()
                if len(sub) == 0:
                    continue
                df_list.append(
                    self._cal_filtercell(
                        sub,
                        key="n_counts",
                        para=3.5,
                        save_key="not_singlecell_n_counts"
                    )
                )

        if len(df_list) == 0:
            raise ValueError("No cells assigned to chip regions when filtering by n_counts.")

        self._cell_pd = pd.concat(df_list, axis=0)

        self._cell_pd["not_singlecell_two_factor"] = (
            self._cell_pd["not_singlecell_n_counts"] &
            self._cell_pd["not_singlecell_area"]
        ).astype(np.int64)

        return self._cell_pd["not_singlecell_two_factor"].to_frame()

    def create_filtered_mask(self, output_path=None):
        """
        mark the multi-cell masks in the cell mask
        """
        if self._cell_pd.empty:
            raise ValueError("filter_doublecells() not run yet")

        double_cells_set = set(
            self._cell_pd.loc[
                self._cell_pd["not_singlecell_two_factor"] == 1,
                "cell_id_int"
            ].astype(np.int64).tolist()
        )
        print(f"number of multi-cell masks: {len(double_cells_set)}")

        from skimage import measure

        if not hasattr(self.cellmask, "dtype"):
            if isinstance(self.cellmask, (str, Path)):
                self.cellmask = tifffile.imread(str(self.cellmask))

        labeled_mask = measure.label(self.cellmask, connectivity=2)

        cell_to_region = {}
        region_to_cell = {}

        for _, row in self._cell_pd.iterrows():
            cell_id = int(row["cell_id_int"])
            x = int(row["x"])
            y = int(row["y"])
            if 0 <= y < labeled_mask.shape[0] and 0 <= x < labeled_mask.shape[1]:
                region_label = labeled_mask[y, x]
                if region_label > 0:
                    cell_to_region[cell_id] = region_label
                    if region_label not in region_to_cell:
                        region_to_cell[region_label] = []
                    region_to_cell[region_label].append(cell_id)

        print(f"projected {len(cell_to_region)} cell area")

        max_region = labeled_mask.max()
        valid_array = np.zeros(max_region + 1, dtype=bool)

        for region_label, cell_ids in region_to_cell.items():
            is_valid = True
            for cell_id in cell_ids:
                if cell_id in double_cells_set:
                    is_valid = False
                    break
            if is_valid:
                valid_array[region_label] = True

        valid_array[0] = False
        filtered_mask = valid_array[labeled_mask].astype(np.uint8)

        if output_path:
            cbimwrite(output_path, filtered_mask)

        return filtered_mask

    def _split_chipRegion(self):
        self._cell_pd = self._cell_pd.copy()

        splitx = np.linspace(0, self._cell_pd["x"].max(), self.split_num + 1)
        splity = np.linspace(0, self._cell_pd["y"].max(), self.split_num + 1)

        x_dic = {}
        y_dic = {}
        for i in np.arange(self.split_num):
            x_dic[i] = [splitx[i], splitx[i + 1]]
            y_dic[i] = [splity[i], splity[i + 1]]

        self._cell_pd["x_area"] = -1
        self._cell_pd["y_area"] = -1

        for key in x_dic.keys():
            self._cell_pd.loc[
                (self._cell_pd["x"] >= x_dic[key][0]) & (self._cell_pd["x"] <= x_dic[key][1]),
                "x_area"
            ] = key

        for key in y_dic.keys():
            self._cell_pd.loc[
                (self._cell_pd["y"] >= y_dic[key][0]) & (self._cell_pd["y"] <= y_dic[key][1]),
                "y_area"
            ] = key

        return self._cell_pd

    def result_to_txt(self, save_path=""):
        """
        Output filter result txt:
            cell_id
            x
            y
            delete_flag

        x/y are always written (sourced from cbMatrix); downstream
        delete_doublet_from_expanded_mask reads them from this txt.
        """
        if self._cell_pd.empty:
            raise ValueError("filter_doublecells() not run yet")

        required_cols = ["cell_id_int", "x", "y", "not_singlecell_two_factor"]
        for col in required_cols:
            if col not in self._cell_pd.columns:
                raise ValueError(f"{col} not found. Please run filter_doublecells() first.")

        out_df = self._cell_pd[["cell_id_int", "x", "y", "not_singlecell_two_factor"]].copy()
        out_df = out_df.rename(
            columns={
                "cell_id_int": "cell_id",
                "not_singlecell_two_factor": "delete_flag",
            }
        )

        out_df.to_csv(save_path, sep="\t", index=False)

    def get_cellarea(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
        cellarea_list = [cv2.contourArea(contours[i]) for i in range(len(contours))]
        return cellarea_list

    @classmethod
    def filter_to_gef(cls, filter_df: pd.DataFrame, cbm: cbMatrix, save_path="", key="not_singlecell_two_factor"):
        """
        filter original .gef or .gem file with filtered result file
        """
        contain_cell = filter_df[filter_df[key] == 0]
        cbm.raw_data.tl.filter_cells(cell_list=contain_cell, inplace=True)
        from stereo.io import write_mid_gef
        write_mid_gef(cbm.raw_data, save_path)
        return cbm


def filter_pipline(geffile, cellmask, output_filter_file="", output_filter_gef=""):
    """
    filter algorithm pipeline

    geffile: adjusted.cellbin.gef
    cellmask: original segmentation mask
    """
    print("GEF:", geffile)
    print("MASK:", cellmask)
    print("OUTPUT_FILTER_FILE:", output_filter_file)
    print("OUTPUT_FILTER_GEF:", output_filter_gef)

    fc = FilterCells(geffile, mask=cellmask)
    fc.filter_doublecells()

    filtered_mask = fc.create_filtered_mask()
    filtered_mask = np.where(filtered_mask > 0, 1, 0).astype(np.uint8)

    path = os.path.dirname(cellmask)
    out_mask = os.path.join(path, "filtered_cellmask.tif")
    cbimwrite(out_mask, filtered_mask)
    print("filtered mask saved to:", out_mask)

    if output_filter_file:
        fc.result_to_txt(output_filter_file)
        print("filter txt saved to:", output_filter_file)

    # Optional: write filtered GEF
    # if output_filter_gef:
    #     df = pd.read_csv(output_filter_file, header=0, sep="\t", index_col=0)
    #     cbm = cbMatrix(geffile)
    #     FilterCells.filter_to_gef(df, cbm, output_filter_gef)
    #     print("filtered gef saved to:", output_filter_gef)


def main():
    path = r"/"

    geffile = os.path.join(path, "sn_Transcriptomics.adjusted.cellbin.gef")
    cellmask = os.path.join(path, "sn_mask.tif")

    filter_pipline(
        geffile=geffile,
        cellmask=cellmask,
        output_filter_file=os.path.join(path, "sn.filter.txt"),
        output_filter_gef=os.path.join(path, "sn.filter.gef"),
    )


if __name__ == "__main__":
    main()