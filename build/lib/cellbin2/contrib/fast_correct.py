import datetime
import cv2
import numpy as np
import tifffile
from scipy import ndimage
from collections import deque
import multiprocessing as mp
from tqdm import tqdm

from cellbin2.utils import clog
from cellbin2.utils.pro_monitor import process_decorator
from cellbin2.image import cbimread

CROP_BLOCK = 2000


class Fast:
    def __init__(self, mask, distance=10, process=8):
        """
        Fast class to generate corrected mask.
        Args:
            mask: cell mask
            distance: correction distance
            process: number of cores to use
        """
        super().__init__()
        self.processes = process
        self.distance = distance
        self.mask = mask.copy()
        num_cpus = mp.cpu_count()
        if self.processes > num_cpus:
            clog.info(f"cpu counts on current machine: {num_cpus}")
            self.processes = int(num_cpus // 2)
            clog.info(f"adjust process to: {self.processes}")

    @staticmethod
    def getNeighborLabels8(label, x, y, width, height):
        lastLabel = None
        for xx in range(max(x - 1, 0), min(height - 1, x + 2), 1):
            for yy in range(max(y - 1, 0), min(width - 1, y + 2), 1):
                if xx == x and yy == y:
                    continue
                l = label[xx, yy]
                if l != 0:
                    if not lastLabel:
                        lastLabel = l
                    elif lastLabel != l:
                        return None
        return lastLabel

    @staticmethod
    def addNeighboursToQueue8(queued, queue, x, y, width, height):
        try:
            if queued[x * width + (y - 1)] == 0 and x >= 0 and x < height and y >= 0 and y < width:
                queued[x * width + (y - 1)] = 1
                queue.append((x, y - 1))
            if queued[(x - 1) * width + y] == 0 and x >= 0 and x < height and y >= 0 and y < width:
                queued[(x - 1) * width + y] = 1
                queue.append((x - 1, y))
            if queued[(x + 1) * width + y] == 0 and x >= 0 and x < height and y >= 0 and y < width:
                queued[(x + 1) * width + y] = 1
                queue.append((x + 1, y))
            if queued[x * width + (y + 1)] == 0 and x >= 0 and x < height and y >= 0 and y < width:
                queued[x * width + (y + 1)] = 1
                queue.append((x, y + 1))
            if queued[(x - 1) * width + (y - 1)] == 0 and x >= 0 and x < height and y >= 0 and y < width:
                queued[(x - 1) * width + (y - 1)] = 1
                queue.append((x - 1, y - 1))
            if queued[(x - 1) * width + (y + 1)] == 0 and x >= 0 and x < height and y >= 0 and y < width:
                queued[(x - 1) * width + (y + 1)] = 1
                queue.append((x - 1, y + 1))
            if queued[(x + 1) * width + (y - 1)] == 0 and x >= 0 and x < height and y >= 0 and y < width:
                queued[(x + 1) * width + (y - 1)] = 1
                queue.append((x + 1, y - 1))
            if queued[(x + 1) * width + (y + 1)] == 0 and x >= 0 and x < height and y >= 0 and y < width:
                queued[(x + 1) * width + (y + 1)] = 1
                queue.append((x + 1, y + 1))
        except:
            pass

    @staticmethod
    def crop_mask(mask):
        x, y = np.where(mask > 0)
        start_x, start_y, end_x, end_y = max(np.min(x) - 100, 0), max(np.min(y) - 100, 0), min(np.max(x) + 100,
                                                                                               mask.shape[0]), min(
            np.max(y) + 100, mask.shape[1])
        start = (start_x, start_y)
        end = (end_x, end_y)
        cropmask = mask[start_x:end_x, start_y:end_y]
        return start, end, cropmask

    @staticmethod
    def array_to_block(a, p, q, step=100):
        '''
        Divides array a into subarrays of size p-by-q
        p: block row size
        q: block column size
        '''
        m = a.shape[0]  # image row size
        n = a.shape[1]  # image column size
        # pad array with NaNs so it can be divided by p row-wise and by q column-wise
        bpr = ((m - 1) // p + 1)  # blocks per row
        bpc = ((n - 1) // q + 1)  # blocks per column
        M = p * bpr
        N = q * bpc
        A = np.nan * np.ones([M, N], dtype=np.half)
        A[:a.shape[0], :a.shape[1]] = a
        block_list = []
        previous_row = 0
        for row_block in range(bpr):
            previous_row = row_block * p
            previous_column = 0
            for column_block in range(bpc):
                previous_column = column_block * q
                if previous_row == 0:
                    if previous_column == 0:
                        block = A[previous_row:previous_row + p, previous_column:previous_column + q]
                    elif previous_column == (bpc - 1) * q:
                        block = A[previous_row:previous_row + p, previous_column - (step * column_block):]
                    else:
                        block = A[previous_row:previous_row + p,
                                previous_column - (step * column_block):previous_column - (step * column_block) + q]
                elif previous_row == (bpr - 1) * p:
                    if previous_column == 0:
                        block = A[previous_row - (step * row_block):, previous_column:previous_column + q]
                    elif previous_column == (bpc - 1) * q:
                        block = A[previous_row - (step * row_block):, previous_column - (step * column_block):]
                    else:
                        block = A[previous_row - (step * row_block):,
                                previous_column - (step * column_block):previous_column - (step * column_block) + q]
                else:
                    if previous_column == 0:
                        block = A[previous_row - (step * row_block):previous_row - (step * row_block) + p,
                                previous_column:previous_column + q]
                    elif previous_column == (bpc - 1) * q:
                        block = A[previous_row - (step * row_block):previous_row - (step * row_block) + p,
                                previous_column - (step * column_block):]
                    else:
                        block = A[previous_row - (step * row_block): previous_row - (step * row_block) + p,
                                previous_column - (step * column_block): previous_column - (step * column_block) + q]
                        # remove nan columns and nan rows
                nan_cols = np.all(np.isnan(block), axis=0)
                block = block[:, ~nan_cols]
                nan_rows = np.all(np.isnan(block), axis=1)
                block = block[~nan_rows, :]
                # append
                if block.size:
                    block_list.append(block.astype(np.uint8))
        return block_list, (bpr, bpc)

    @staticmethod
    def create_edm_label(mask):
        """

        Args:
            mask: cropped cell mask

        Returns:
            edm: distance map of input mask
            maskImg: connected components of input mask

        """
        _, maskImg = cv2.connectedComponents(mask, connectivity=8)
        mask[mask > 0] = 255
        mask = cv2.bitwise_not(mask)
        edm = ndimage.distance_transform_edt(mask)
        edm[edm > 255] = 255
        edm = edm.astype(np.uint8)
        return edm, maskImg

    def process_queue(self, queued, queue, label, width, height):
        # print (f'start to iterate queue at {datetime.datetime.now()}', flush = True)
        while queue:
            x, y = queue.popleft()
            l = self.getNeighborLabels8(label, x, y, width, height)
            if not l:
                continue
            label[x, y] = l
            self.addNeighboursToQueue8(queued, queue, x, y, width, height)
        return label

    def correct(self, mask, dis, idx):
        edm, label = self.create_edm_label(mask)
        height, width = edm.shape
        queued = [0] * width * height
        queue = deque()
        # point = namedtuple('Points',['x','y','label'])
        for i in range(0, height, 1):
            for j in range(0, width, 1):
                val = edm[i, j].astype(int)
                if val > dis:
                    queued[i * width + j] = 1
                    continue
                l = label[i, j].astype(int)
                if l != 0:
                    queued[i * width + j] = 1
                    continue
                else:
                    if i > 0 and i < height - 1:
                        if label[i - 1, j] != 0 or label[i + 1, j] != 0:
                            queued[i * width + j] = 1
                            queue.append((i, j))
                    if j > 0 and j < width - 1:
                        if label[i, j - 1] != 0 or label[i, j + 1] != 0:
                            queued[i * width + j] = 1
                            queue.append((i, j))
        label = self.process_queue(queued, queue, label, width, height)
        label[label > 0] = 1
        label = label.astype(np.uint8)
        return label, idx

    @staticmethod
    def handle_error(error):
        clog.error(error)

    @staticmethod
    def merge_by_row(arr, loc, step=100):
        r, c = loc
        half_step = step // 2
        full_img = arr[0][:-half_step]
        for rr in range(1, r):
            if rr == r - 1:
                full_img = np.concatenate((full_img, arr[rr][half_step:]), axis=0)
            else:
                full_img = np.concatenate((full_img, arr[rr][half_step:-half_step]), axis=0)
        return full_img

    @staticmethod
    def merge_by_col(final_result, loc, step=100):
        r, c = loc
        row_list = []
        half_step = step // 2
        for rr in range(r):
            row_img = final_result[rr * c][:, :-half_step]
            for cc in range(1, c):
                if cc == c - 1:
                    row_img = np.concatenate((row_img, final_result[rr * c + cc][:, half_step:]), axis=1)
                else:
                    row_img = np.concatenate((row_img, final_result[rr * c + cc][:, half_step:-half_step]), axis=1)
            row_list.append(row_img)
        return row_list

    def process(self):
        clog.info(f"start of cell correct")
        if np.sum(self.mask) == 0:
            clog.info(f"The image is all black. Fast Labeling failed. Return the input as it is")
            return
        clog.info(f"Fast Labeling using {self.processes} processes")
        clog.info(f"Fast Labeling distance: {self.distance}")
        pool = mp.Pool(processes=self.processes)
        start, end, cropmask = self.crop_mask(self.mask)
        masks, loc = self.array_to_block(cropmask, CROP_BLOCK, CROP_BLOCK, step=100)
        final_result = []
        processes = []
        num_tasks = len(masks)
        pbar = tqdm(total=num_tasks, file=clog.tqdm_out, mininterval=60)
        pbar.set_description('fast correct')
        update = lambda *args: pbar.update()
        for i, ma in enumerate(masks):
            result = pool.apply_async(
                self.correct,
                (ma, self.distance, i,),
                error_callback=self.handle_error,
                callback=update
            )
            processes.append(result)
        pool.close()
        pool.join()
        for p in processes:
            final_result.append(p.get())
        final_result = sorted(final_result, key=lambda x: x[1])
        final_result = [arr for arr, i in final_result]
        if len(final_result) == 1:
            row_list = final_result
        else:
            row_list = self.merge_by_col(final_result, loc, step=100)
        if len(row_list) == 1:
            final_img = row_list[0]
        else:
            final_img = self.merge_by_row(row_list, loc, step=100)

        self.mask[start[0]:end[0], start[1]:end[1]] = final_img
        clog.info(f"end of cell correct")

    def get_mask_fast(self):
        """

        Returns: corrected mask, dtype is np.ndarray

        """
        return self.mask


@process_decorator("MiB")
def run_fast_correct(
        mask_path,
        distance=10,
        n_jobs=5
):
    """
    import os
    save_dir = "/media/Data/dzh/data/fast_correct_test/after"
    # all black
    m1 = "/media/Data/dzh/data/single_cell/debug_for_qiuying/result/C03427C3_mask.tif"
    # roi (1658x 2501)
    m2 = "/media/Data/dzh/data/cellbin/debug_cell_cor/ssDNA_D02266C2_regist_cellseg_D182_RA1_8_masks.tif"
    # roi (233, 230)
    m3 = "/media/Data/dzh/data/single_cell/wq_issue/result/C03433F3_mask.tif"

    # normal data
    m4 = "/media/Data/dzh/data/cellbin/FF-HE-C-Seg-Upgrade/TEST-all/cellseg_bcdu_H_240823_tf_deploy_test_tc_cls_2/ztron_output/B04372C214_SC_20240925_145336_4.1.0-beta.25.tif"

    # roi (1609, 14957)
    m5 = "/media/Data/dzh/data/fast_correct_test_data/data/Y00935N4_ssDNA_mask(2).tif"

    # roi (1658, 2051)
    m6 = "/media/Data/dzh/data/fast_correct_test_data/data/D02266C2_mask.tif"
    test_data = [m6]
    for i in test_data:
        i_no_ext, ext = os.path.splitext(os.path.basename(i))
        cur_save = os.path.join(save_dir, i_no_ext + "_fast" + ext)
        mask = run_fast_correct(mask_path=i)
        tifffile.imwrite(cur_save, mask, compression=True)
    """
    if not isinstance(mask_path, np.ndarray):
        mask_path = cbimread(mask_path, only_np=True)
    f_cor = Fast(
        mask_path,
        distance,
        n_jobs
    )
    f_cor.process()
    re_mask = f_cor.get_mask_fast()
    return re_mask


def fast_main():
    import argparse
    from cellbin2.image import cbimwrite
    usage_str = f"python {__file__} \n"
    parser = argparse.ArgumentParser(
        description="Fast correct script",
        usage=usage_str
    )
    parser.add_argument("-i", "--input_image", action="store", type=str, required=True,
                        help="Cell mask path")
    parser.add_argument("-o", "--output_path", action="store", type=str, required=True,
                        help="Save path")
    parser.add_argument("-d", "--distance", action="store", type=int, default=10,
                        help="Distance (radius)")
    parser.add_argument("-p", "--process", action="store", type=int, default=10,
                        help="# of process")
    (para, args) = parser.parse_known_args()
    mask_path = para.input_image
    distance = para.distance
    process = para.process
    save_path = para.output_path

    f_mask = run_fast_correct(
        mask_path=mask_path,
        distance=distance,
        n_jobs=process
    )
    cbimwrite(save_path, files=f_mask)


if __name__ == '__main__':
    fast_main()

