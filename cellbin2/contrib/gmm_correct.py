import tifffile as tifi
import cv2
import argparse
import os
import math
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from warnings import filterwarnings
from multiprocessing import Pool
from cellbin2.utils import clog
from cellbin2.image import cbimread, cbimwrite
from scipy.spatial import ConvexHull
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import h5py
filterwarnings('ignore')

def create_binary_mask_from_txt(txt_file, dilate_size=3, erode_size=3):
    """
    Process cell by cell: dilate to connect points, then erode the outermost ring
    """
    # 1. Read
    df = pd.read_csv(txt_file, sep='\t')
    
    # 2. Filter data with CellID
    df_cells = df[df['CellID'] > 0]
    if len(df_cells) == 0:
        return None

    # 3. Create blank mask (no padding, use max coordinates directly)
    x_max = df_cells['x'].max() + 1
    y_max = df_cells['y'].max() + 1
    mask = np.zeros((y_max, x_max), dtype=np.uint8)

    # 4. Mark all points at once (vectorized)
    x_coords = df_cells['x'].values.astype(int)
    y_coords = df_cells['y'].values.astype(int)
    cell_ids = df_cells['CellID'].values

    # 5. Process by CellID (use groupby to accelerate)
    unique_cells = np.unique(cell_ids)

    for cell_id in unique_cells:
        # Vectorized acquisition of coordinates for this cell
        idx = cell_ids == cell_id
        if np.sum(idx) < 3:  # Skip if too few points
            continue

        # Create cell mask
        cell_mask = np.zeros_like(mask)
        cell_mask[y_coords[idx], x_coords[idx]] = 255

        # Morphological operations
        if dilate_size > 0:
            kernel_dilate = np.ones((dilate_size, dilate_size), np.uint8)
            cell_mask = cv2.dilate(cell_mask, kernel_dilate)

        if erode_size > 0:
            kernel_erode = np.ones((erode_size, erode_size), np.uint8)
            cell_mask = cv2.erode(cell_mask, kernel_erode)

        # Merge
        mask = np.maximum(mask, cell_mask)

    return mask


def read_gef(gef_path):
    """
    Read raw.gef bin1 expression.
    Return dataframe:
    geneID, x, y, MIDCount
    """

    clog.info("Reading raw GEF...")

    with h5py.File(gef_path, "r") as f:
        exp = f["/geneExp/bin1/expression"][:]

    gem = pd.DataFrame({
        "geneID": "unknown",
        "x": exp["x"].astype(np.uint32),
        "y": exp["y"].astype(np.uint32),
        "MIDCount": exp["count"].astype(np.uint32),
    })

    clog.info(
        f"Loaded GEF molecules: {len(gem)}"
    )

    return gem


def row(gem_path):
    """Simplified version, directly returns 0 to skip row detection"""
    # If it's a GEF file, return 0 directly
    if gem_path.endswith('.gz'):
        import gzip
        f = gzip.open(gem_path, 'rb')
    else:
        f = open(gem_path, 'rb')
    
    try:
        i = 0
        while True:
            line_bytes = f.readline()
            if not line_bytes:
                break
                
            try:
                line = bytes.decode(line_bytes, 'utf-8')
            except:
                line = bytes.decode(line_bytes, 'latin-1')
            
            # Check if there are still comment lines
            if line.startswith('#'):
                i += 1
            else:
                # Check if reached data row (usually column names)
                if 'geneID' in line or '\t' in line:
                    break
                else:
                    i += 1
    finally:
        f.close()
    
    return i

def creat_cell_gxp(mask_path,gem_path,outpath='./',fileName='cellbin_gmm.txt'):
    clog.info("Loading mask file...")
    mask = tifi.imread(mask_path)
    num_labels, maskImg, _, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clog.info("Reading data..")
    if gem_path.endswith(".gef"):
        gem = read_gef(gem_path)
    else:
        gem = pd.read_csv(
            gem_path,
            sep='\t',
            skiprows=row(gem_path)
        )
    max_y,min_y,max_x,min_x = centroids[:,1].max(),centroids[:,1].min(),centroids[:,0].max(),centroids[:,0].min()
    gem = gem[(gem['x'] < max_x+100) & (gem['x']>min_x-100) & (gem['y']<max_y+100) & (gem['y']>min_y-100)]
    if "UMICount" in gem.columns:
        gem = gem.rename(columns={'UMICount':'MIDCount'})
    if "MIDCounts" in gem.columns:
        gem = gem.rename(columns={'MIDCounts':'MIDCount'})

    gem['CellID'] = maskImg[gem['y'],gem['x']]
    gem.to_csv(os.path.join(outpath, fileName), sep='\t', index=False)

    del maskImg
    return gem, centroids


class GMM(object):
    def __init__(self,  mask_file, gem_file, out_path, threshold, process):
        self.mask_file = mask_file
        self.gem_file = gem_file
        self.out_path = out_path
        self.threshold = threshold
        self.process = process
        self.radius = 50  # keep original radius

    def _split_gem(self,gem, centroids, chunks=10):
        centroids = centroids[1:].astype(int)
        centroids_list = np.array_split(centroids, chunks)
        sub_gem_list = []
        for i in centroids_list:
            min_y, max_y = i[:,1].min(), i[:,1].max()
            sub_gem = gem[(gem['y']<=max_y+30) & (gem['y']>=min_y-30)]
            sub_gem_list.append(sub_gem)

        del gem

        res = []
        for idx,(centroid,sub_gem) in tqdm(enumerate(zip(centroids_list, sub_gem_list)), total=chunks):
            if idx == 0:
                start_idx = 0
            else:
                start_idx = 0
                for i in range(idx):
                    start_idx += len(centroids_list[i])
            p = self._GMM_correction(centroid,sub_gem, start_idx)
            res.append(p)

        # Merge all results
        correct_data = pd.concat(res, ignore_index=True)

        # !!! New: Handle conflicts !!!
        # Find molecules assigned to multiple cells
        correct_data['key'] = (
            correct_data['x'].astype(str)
            + '_'
            + correct_data['y'].astype(str)
        )

        # Group by gene-coordinate, if there are multiple CellIDs, there is a conflict
        conflict_mask = (correct_data['CellID'] > 0) & (correct_data.duplicated(subset=['key'],keep=False))
        conflict_data = correct_data[conflict_mask].copy()
        unique_data = correct_data[~conflict_mask].copy()

        if len(conflict_data) > 0:
            print(f"Found {len(conflict_data)} conflicting molecules, resolving...")

            # For each conflicting gene-coordinate, select the one with the highest score
            # Note: The original code did not save scores, we need to return scores in _correction

            # Here we assume _correction already returns scores
            # vectorized conflict resolving
            if 'score' in conflict_data.columns:
                conflict_data = (
                    conflict_data
                    .sort_values(
                        'score',
                        ascending=True
                    )
                    .drop_duplicates(
                        subset=['key'],
                        keep='first'
                    )
                )
            else:
                conflict_data = conflict_data.drop_duplicates(
                    subset=['key'],
                    keep='first'
                )

            correct_data = pd.concat(
                [
                    unique_data,
                    conflict_data
                ],
                ignore_index=True
            )

        # Remove temporary key column
        if 'key' in correct_data.columns:
            correct_data = correct_data.drop('key', axis=1)

        # Continue with original processing
        correct_data = correct_data.drop_duplicates(
            subset=[
                'x',
                'y',
                'MIDCount',
                'CellID'
            ],
            keep='first'
        )
        correct_data = correct_data[correct_data['CellID']>0]
        correct_data.to_csv(os.path.join(self.out_path, 'cell_mask_profile.txt'), sep='\t', index=False)

    def __creat_gxp_data(self, ):
        gem, centroids = creat_cell_gxp(self.mask_file, self.gem_file, outpath=self.out_path,
                               fileName='nuclei_mask_profile.txt')
        return gem, centroids

    def _correction(self,cell_df,bg_df,cell_id):
        try:
            clf = GaussianMixture(n_components=3, covariance_type='spherical')
            clf.fit(cell_df[['x', 'y', 'MIDCount']].values)
            bg_group = bg_df.groupby(['x', 'y']).agg(MID_max=('MIDCount', 'max')).reset_index()
            cell_test_bg = pd.merge(bg_df, bg_group, on=['x', 'y'])
            score = pd.Series(-clf.score_samples(cell_test_bg[['x', 'y', 'MID_max']].values))
            
            # !!! New: Save scores !!!
            cell_test_bg['score'] = score
            cell_test_bg['CellID'] = np.where(score < self.threshold, cell_id , 0)
            cell_test_bg.drop('MID_max', axis=1, inplace=True)

            # Separate points assigned to this cell
            assigned = cell_test_bg[cell_test_bg['CellID']==cell_id].copy()

            # Merge cell points and assigned background points
            cell_gmm_df = pd.concat([assigned, cell_df], ignore_index=True)
            return cell_gmm_df
        except:
            return cell_df

    def _GMM_correction(self,centroids,gem, start_idx):
        cell_gem = gem[(gem['CellID']>0)]
        bg_gem = gem[(gem['CellID']==0)]
        gmm_gem = []
        for cell_id,(x,y) in enumerate(centroids):
            try:
                cell_id = cell_id + start_idx + 1
                cell_df = cell_gem[(cell_gem['x']<x+self.radius) & (cell_gem['x']>x-self.radius) & (cell_gem['y']<y+self.radius) & (cell_gem['y']>y-self.radius)]
                cell_df = cell_df[cell_df['CellID'] == cell_id]
                bg_df = bg_gem[(bg_gem['x']<x+self.radius) & (bg_gem['x']>x-self.radius) & (bg_gem['y']<y+self.radius) & (bg_gem['y']>y-self.radius)]
                cell_gmm_df = self._correction(cell_df,bg_df,cell_id)
                gmm_gem.append(cell_gmm_df)
            except Exception as e:
                clog.info(e)

        return pd.concat(gmm_gem, ignore_index=True)

    def cell_correct(self, ):
        t0 = time.time()
        gem, centroids = self.__creat_gxp_data()
        clog.info(gem.head())
        t1 = time.time()
        t2 = time.time()
        self._split_gem(gem, centroids, chunks=self.process*32)
        t3 = time.time()
        clog.info(f'Correct : {t3-t2}')
        clog.info(f'Total : {t3-t0}')

def args_parse():
    usage = """ Usage: %s Cell expression file (with background) path, single-process """
    arg = argparse.ArgumentParser(usage=usage)
    arg.add_argument('-m','--mask_path',help='cell mask', default="")
    arg.add_argument('-g', '--gem_path', help='gem file', default="")
    arg.add_argument('-o','--out_path',help='output path',default='')
    arg.add_argument('-p','--process',help='n process',type=int,default=8)
    arg.add_argument('-t','--threshold',help='threshold',type=int,default=30)

    return arg.parse_args()


def main():
    args = args_parse()
    correction = GMM(args.mask_path, args.gem_path, args.out_path, args.threshold, args.process)
    correction.cell_correct()

if __name__ == '__main__':
    main()
