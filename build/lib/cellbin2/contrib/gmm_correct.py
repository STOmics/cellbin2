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
filterwarnings('ignore')


def row(gem_path):
    if gem_path.endswith('.gz'):
        import gzip
        with gzip.open(gem_path,'rb') as f:
            first_line = bytes.decode(f.readline())
            if '#' in first_line:
                rows = 6
            else:
                rows = 0
    else:
        with open(gem_path,'rb') as f:
            first_line = bytes.decode(f.readline())
            if '#' in first_line:
                rows = 6
            else:
                rows = 0
    return rows

def creat_cell_gxp(mask_path,gem_path,outpath='./',fileName='cellbin_gmm.txt'):
    clog.info("Loading mask file...")
    mask = tifi.imread(mask_path)
    num_labels, maskImg, _, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clog.info("Reading data..")
    gem = pd.read_csv(gem_path, sep='\t', skiprows=row(gem_path))
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
        self.radius = 50

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

        correct_data = pd.concat(res)
        correct_data = correct_data.drop_duplicates(subset=['geneID', 'x', 'y', 'MIDCount'], keep='first')
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
            cell_test_bg['CellID'] = np.where(score < self.threshold, cell_id , 0)
            cell_test_bg.drop('MID_max', axis=1, inplace=True)
            cell_test_bg = cell_test_bg[cell_test_bg['CellID']==cell_id]
            cell_gmm_df = pd.concat([cell_test_bg,cell_df])
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

        return pd.concat(gmm_gem)

    def cell_correct(self, ):
        t0 = time.time()
        gem, centroids = self.__creat_gxp_data()
        clog.info(gem.head())
        t1 = time.time()
        clog.info('Load data :', (t1 - t0))
        t2 = time.time()
        self._split_gem(gem, centroids, chunks=self.process*32)
        t3 = time.time()
        clog.info('Correct :', (t3 - t2))
        clog.info('Total :', (t3 - t0))


def args_parse():
    usage = """ Usage: %s Cell expression file (with background) path, single-process """
    arg = argparse.ArgumentParser(usage=usage)
    arg.add_argument('-m','--mask_path',help='cell mask', default=r"D:\Cell_Bin\data\B02113D1\B02113D1_cell_mask_filter.tif")
    arg.add_argument('-g', '--gem_path', help='gem file', default=r"D:\Cell_Bin\data\B02113D1\B02113D1.gem.gz")
    arg.add_argument('-o','--out_path',help='output path',default='./')
    arg.add_argument('-p','--process',help='n process',type=int,default=8)
    arg.add_argument('-t','--threshold',help='threshold',type=int,default=20)

    return arg.parse_args()


def main():
    args = args_parse()
    correction = GMM(args.mask_path, args.gem_path, args.out_path, args.threshold, args.process)
    correction.cell_correct()


if __name__ == '__main__':
    main()
