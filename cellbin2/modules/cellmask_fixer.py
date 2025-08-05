import rasterio
import rasterio.mask
import numpy as np
import pandas as pd
import tifffile as tif
import shapely
from cellbin2.utils import clog
import cv2
import rasterio.features
import geopandas as gpd
from scipy import ndimage
from rasterio.transform import from_origin
#from cellbin.dnn.cseg.cell_geo import mask2geopdf
import os
import warnings
warnings.filterwarnings('ignore')
from shapely.ops import linemerge, unary_union, polygonize
from shapely import MultiPoint,Point,LineString,MultiPolygon,Polygon

def mask2geopdf(src):
    lab, n = ndimage.label(src.read(1))
    results=({"id":i,"properties":{"label":v},"geometry":p} for i,(p,v) in enumerate(rasterio.features.shapes(lab,mask=src.read(1),connectivity=8)))
    geoms=list(results)
    geodf=gpd.GeoDataFrame.from_features(geoms)
    return geodf

class CellMaskFixer:
    def __init__(self,source_imge,refer_image,sn=None):
        self.source_imge=source_imge
        self.refer_image=refer_image
        self._src_image=None
        self._ref_image=None
        ### creat rasterio dataset objects
        self._src_dataset=None
        self._ref_dataset=None
        self.sn=sn

        ##
        self._res_mask=None



    # 设置name属性值
    def set_name(self, sn):
        self.sn = sn

    def _get_rasread_imge(self,image):
        if isinstance(image, str):
            img = rasterio.open(image)
        elif isinstance(image, np.ndarray):
            transform = from_origin(0, image.shape[0], 1, 1)
            with rasterio.io.MemoryFile() as memfile:
                with memfile.open(
                        driver='GTiff',
                        height=image.shape[0],
                        width=image.shape[1],
                        count=1,
                        dtype=image.dtype,
                        transform=transform,
                ) as img:
                    img.write(image, 1)
        else:
            img = None
            clog.info("DAPI file format error.")
        return img
    def _get_tifread_imge(self,image):
        if isinstance(image, str):
            img = tif.imread(image)
        elif isinstance(image, np.ndarray):
            img = image
        else:
            img = None
            clog.info("DAPI file format error.")
        return img

    @property
    def src_dataset(self):
        if self._src_dataset==None:
            self._src_dataset=self._get_rasread_imge(self.source_imge)
        return self._src_dataset
    @property
    def ref_dataset(self):
        if self._ref_dataset==None:
            self._ref_dataset=self._get_rasread_imge(self.refer_image)
        return self._ref_dataset
    @property
    def src_image(self):
        if self._src_image==None:
            self._src_image=self._get_tifread_imge(self.source_imge)
        return self._src_image

    @property
    def ref_image(self):
        if self._ref_image==None:
            self._ref_image=self._get_tifread_imge(self.refer_image)
        return self._ref_image





    def _get_need_fix_cell_gdf(self):
        src_gdf = mask2geopdf(self.src_dataset)
        ref_gdf = mask2geopdf(self.ref_dataset)
        ref_gdf["temp_center"] = ref_gdf.centroid
        ref_gdf["center_ref"] = ref_gdf.centroid
        ref_gdf=ref_gdf.set_geometry("temp_center")
        res_gdf = gpd.sjoin(src_gdf, ref_gdf, lsuffix="src", rsuffix="ref", how="left",predicate="contains")

        singlecell_gdf = res_gdf[res_gdf["label_src"].isin(
            res_gdf["label_src"].value_counts()[res_gdf.label_src.value_counts() < 2].index.tolist())]
        need_fix_cell_gdf = res_gdf[res_gdf["label_src"].isin(
            res_gdf["label_src"].value_counts()[res_gdf.label_src.value_counts() > 1].index.tolist())]

        return need_fix_cell_gdf, singlecell_gdf

    def del_notsinglecell2mask(self,save=True,out_path="./",sn="temp"):
        _,singlecell_gdf=self._get_need_fix_cell_gdf()
        ##### geodf 2 mask
        singlecell_mask, _ = rasterio.mask.mask(self.src_dataset, singlecell_gdf.geometry_src, invert=False, nodata=0,
                                              all_touched=True)

        if not self.sn==None:
            _sn=self.sn
        else:
            _sn=sn

        if save:
            tif.imwrite(os.path.join(out_path, f"{_sn}_fixed_cell_mask.tif"), singlecell_mask, compression='zlib')
        return singlecell_mask





    def fix_notsinglecell2mask(self,save=True,out_path="./",sn="temp"):
        def _get_multipoint(x):
            return shapely.MultiPoint(list(x))

        def _cut_polygon_by_line(polygon, line):
            merged = line.buffer(2).union(polygon.boundary.buffer(2))
            borders = unary_union(merged)
            polygons = polygonize(borders)
            return polygons

        need_fix_cell_gdf,singlecell_gdf = self._get_need_fix_cell_gdf()

        ### fix double-cells
        # need_fix_cell_gdf

        center_incell_df=need_fix_cell_gdf.groupby(["label_src"])[["geometry_src","center_ref"]].agg({'geometry_src': 'first',"center_ref":_get_multipoint})
        cut_list = [shapely.voronoi_polygons(i, only_edges=True, extend_to=j) for i, j in
                    zip(center_incell_df["center_ref"].tolist(), center_incell_df["geometry_src"].tolist())]
        center_incell_df["cut_line"] = cut_list



        vor = []
        for i, j in zip(center_incell_df["geometry_src"].tolist(), center_incell_df["cut_line"].tolist()):
            vor.append(_cut_polygon_by_line(i, j)[1:])
        center_incell_df["voronoi"] = vor
        center_incell_df["voronoi_multi"] = [MultiPolygon(i) for i in list(vor)]
        fixedcell_mask, _ = rasterio.mask.mask(self.src_dataset, center_incell_df["voronoi_multi"].tolist(), invert=False,
                                                   nodata=0,
                                                   all_touched=True)
        singlecell_mask, _ = rasterio.mask.mask(self.src_dataset, singlecell_gdf.geometry_src, invert=False, nodata=0,
                                                all_touched=True)
        res_mask = fixedcell_mask +singlecell_mask
        res_mask[res_mask>255]=255

        if not self.sn==None:
            _sn=self.sn
        else:
            _sn=sn

        if save:
            tif.imwrite(os.path.join(out_path, f"{_sn}_fixed_cell_mask.tif"), res_mask, compression='zlib')
        return res_mask



if __name__ == '__main__':
    #### del muti-cells in single cell
    cmf=CellMaskFixer(source_imge=r'D:/cellbin_data/result/chip/Q00327K8/Q00327K8_Transcriptomics_mask.tif',refer_image=r'D:/cellbin_data/result/chip/Q00327K8/Q00327K8_DAPI_mask_raw.tif',sn='Q00327K8')
    # cmf.del_notsinglecell2mask(out_path="Z:\data")

    ### fix muti-cells in single cell
    res_mask = cmf.fix_notsinglecell2mask(out_path="D:/cellbin_data/result/chip/Q00327K8/")
    print(sum(sum(sum(res_mask))))



