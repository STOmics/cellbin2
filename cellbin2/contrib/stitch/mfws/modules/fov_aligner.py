"""
Matcher provide the way to get overlap/offset from neighbor FOV.
"""
import glog
import copy
import tifffile
import itertools

import cv2 as cv
import numpy as np
from tqdm import tqdm

from multiprocessing import Manager, Pool
try:
    from .jitter_correct import JitterCorrect
except ImportError:
    from jitter_correct import JitterCorrect


def rc_key(row: int, col: int):
    return '{}_{}'.format(str(row).zfill(4), str(col).zfill(4))


class FOVAligner(object):
    """
    Calculate the offset of all images
    """
    def __init__(self, images_path, rows, cols, mode='FFT', multi=True,
                 channel=0, overlap=[0.1, 0.1], i_shape=[None, None]):
        self.images_path = None
        self.channel = 0
        self.rows = self.cols = None

        self.set_size(rows, cols)
        self.set_images_path(images_path)
        self.set_channel(channel)
        self.multi = multi

        self.num = 5
        self.overlap_x, self.overlap_y = overlap
        self.__init_value = 999
        self.i_shape = i_shape

        _overlap = max(self.overlap_x, self.overlap_y)
        self.matcher = FFTMatcher(_overlap) if mode == 'FFT' else SIFTMatcher(_overlap)

        self.horizontal_jitter = np.zeros((self.rows, self.cols, 2), dtype=int) + self.__init_value
        self.vertical_jitter = np.zeros((self.rows, self.cols, 2), dtype=int) + self.__init_value
        self.fov_mask = np.zeros((self.rows, self.cols, 3))

    def set_overlap(self, overlap):
        # assert isinstance(overlap, float), "Overlap type error."
        self.overlap_x, self.overlap_y = overlap

    def set_channel(self, channel):
        assert isinstance(channel, int), "Channel type error."
        self.channel = channel

    def set_images_path(self, images_path):
        # assert isinstance(images_path, dict), "Image path type error."
        self.images_path = images_path

    def set_size(self, rows, cols):
        assert isinstance(rows, int) or isinstance(rows, float), "Rows type error."
        assert isinstance(cols, int) or isinstance(cols, float), "Cols type error."
        self.rows = rows
        self.cols = cols

    def set_process(self, p):
        self.num = p
        if self.num <= 1:
            self.multi = False  # @lizepeng add on 2023/05/17

    @staticmethod
    def _cluster_num(arr, cn=10):
        cluster_list = list()
        for n in arr:
            add_flag = False
            for k_index, k in enumerate(arr):

                if k == n:
                    continue
                elif np.abs(k - n) <= cn:

                    if len(cluster_list) == 0:
                        cluster_list.append([n, k])
                        add_flag = True
                    else:
                        for index, cl in enumerate(cluster_list):
                            if k in cl or n in cl:
                                cluster_list[index].append(n)
                                add_flag = True
                                break
                        else:
                            cluster_list.append([n, k])
                            add_flag = True
            if not add_flag:
                cluster_list.append([n])

        return [list(set(i)) for i in cluster_list]

    def jitter_correct(self, jitter, connect_num=10, overlap_dis=None):
        x_set, x_count = np.unique(jitter[:, :, 0], return_counts=True)
        y_set, y_count = np.unique(jitter[:, :, 1], return_counts=True)

        x_set_cluster = self._cluster_num(x_set, connect_num)
        y_set_cluster = self._cluster_num(y_set, connect_num)

        x_dict = dict()
        for k, xc in enumerate(x_set_cluster):
            count = 0
            flag = False
            for i in xc:
                if min(overlap_dis[0]) < i < max(overlap_dis[0]): flag = True
                index = list(x_set).index(i)
                count += x_count[index]
            if flag:
                x_dict[k] = (xc, count)
        x_list = max(x_dict.items(), key=lambda x: x[1][1])[1][0]

        y_dict = dict()
        for k, yc in enumerate(y_set_cluster):
            count = 0
            flag = False
            for i in yc:
                if min(overlap_dis[1]) < i < max(overlap_dis[1]): flag = True
                index = list(y_set).index(i)
                count += y_count[index]
            if flag: y_dict[k] = (yc, count)
        y_list = max(y_dict.items(), key=lambda x: x[1][1])[1][0]

        for i in range(self.rows):
            for j in range(self.cols):
                if jitter[i, j, 0] not in x_list or jitter[i, j, 1] not in y_list:
                    jitter[i, j] = [-self.__init_value, -self.__init_value]
        return jitter

    def create_jitter(self, overlap_dis=0.05):
        """
        Scan image and create offset.
        """
        confi_mask = np.zeros((self.rows, self.cols, 2)) - 1
        ncc_confi_mask = np.zeros((self.rows, self.cols, 2)) - 1

        if self.multi:
            tesk_list = list()
            for row_index in range(self.rows):
                tesk_list.append({'row': row_index})
            for col_index in range(self.cols):
                tesk_list.append({'col': col_index})

            with Manager() as manager:
                h_j = manager.dict()
                v_j = manager.dict()
                confi_h = manager.dict()
                confi_v = manager.dict()
                # process_list = []
                # num = int(cpu_count() / 2)
                # num = 10
                glog.info(f"Stitch using {self.num} process")
                pool = Pool(processes=self.num)

                pb = tqdm(total=len(tesk_list), colour='green', unit='row', ncols=100)
                pb.set_description('Calculate jitter')

                for tesk in tesk_list:
                    if list(tesk.keys())[0] == 'row':
                        row_index = tesk['row']
                        pool.apply_async(
                            func=self._multi_jitter,
                            args=(h_j, confi_h, row_index, None, 0),
                            callback=lambda *args: pb.update(1)
                        )

                    elif list(tesk.keys())[0] == 'col':
                        col_index = tesk['col']
                        pool.apply_async(
                            func=self._multi_jitter,
                            args=(v_j, confi_v, None, col_index, 1),
                            callback=lambda *args: pb.update(1)
                        )

                    else:
                        pass

                pool.close()
                pool.join()

                for key in h_j.keys():
                    row, col = [int(i) for i in key.split('_')]
                    self.horizontal_jitter[row, col, :] = h_j[key]
                    confi_mask[row, col, 0] = confi_h[key]

                for key in v_j.keys():
                    row, col = [int(i) for i in key.split('_')]
                    self.vertical_jitter[row, col, :] = v_j[key]
                    confi_mask[row, col, 1] = confi_v[key]
        else:
            glog.info("Scan Row by Row.")
            for i in tqdm(range(self.rows), desc='RowByRow', colour='green', unit='row', ncols=100):
                train = self._get_image(i, 0)
                for j in range(1, self.cols):
                    query = self._get_image(i, j)
                    if (train is not None) and (query is not None):
                        b = self.matcher.neighbor_match(train, query, 0)
                        if b is not None:
                            self.horizontal_jitter[i, j, :] = b[:2]
                            confi_mask[i, j, 0] = b[2]
                            ncc_confi_mask[i, j, 0] = b[3]
                    train = query

            glog.info("Scan Col by Col.")
            for j in tqdm(range(self.cols), desc='ColByCol', colour='green', unit='col', ncols=100):
                train = self._get_image(0, j)
                for i in range(1, self.rows):
                    query = self._get_image(i, j)
                    if (train is not None) and (query is not None):
                        b = self.matcher.neighbor_match(train, query, 1)
                        if b is not None:
                            self.vertical_jitter[i, j, :] = b[:2]
                            confi_mask[i, j, 1] = b[2]
                            ncc_confi_mask[i, j, 1] = b[3]
                    train = query

        self.horizontal_jitter, confi_mask[:, :, 0], _ = self.filter_abnormal_offset(self.horizontal_jitter,
                                                                                     confi_mask[:, :, 0],
                                                                                     thread=5)

        self.vertical_jitter, confi_mask[:, :, 1], _ = self.filter_abnormal_offset(self.vertical_jitter,
                                                                                   confi_mask[:, :, 1],
                                                                                   thread=5)

        if self.overlap_x == 0:
            self.horizontal_jitter[:, 1:] = 0

        if self.overlap_y == 0:
            self.vertical_jitter[1:, :] = 0

        self.fov_mask[:, :, :2] = confi_mask
        self.fov_mask[:, :, 2] = np.max(confi_mask, axis=2)
        self.fov_mask[:, :, 2][np.where(self.fov_mask[:, :, 2] >= 99)] = 99

        jc = JitterCorrect([self.horizontal_jitter, self.vertical_jitter], image_size = self.i_shape)
        self.horizontal_jitter, self.vertical_jitter = jc.correct()

        # glog.info("Scan image done")

    def offset_eval(self, height, width, overlap=[0.1, 0.1]):
        """
        :param height: fov image height
        :param width: fov image width
        :param overlap:
        :return: offset matrix r * c * 1
        """
        h_j = copy.deepcopy(self.horizontal_jitter)
        v_j = copy.deepcopy(self.vertical_jitter)
        overlap_x, overlap_y = overlap

        for r in range(self.rows):
            for c in range(self.cols):
                if h_j[r, c, 0] != -self.__init_value and \
                        h_j[r, c, 0] != self.__init_value:
                    h_j[r, c, 0] += int(width * overlap_x)
                if v_j[r, c, 1] != -self.__init_value and \
                        v_j[r, c, 1] != self.__init_value:
                    v_j[r, c, 1] += int(height * overlap_y)

        h_e = (h_j[:, :, 0] ** 2 + h_j[:, :, 1] ** 2) ** 0.5
        h_e[np.where(h_e >= self.__init_value)] = -1

        v_e = (v_j[:, :, 0] ** 2 + v_j[:, :, 1] ** 2) ** 0.5
        v_e[np.where(v_e >= self.__init_value)] = -1

        e = np.max(np.stack((h_e.astype(int), v_e.astype(int)), axis=2), axis=2)

        return e, h_e, v_e

    def _multi_jitter(self, jitter, confi, row=None, col=None, axis=0):
        if axis == 0:
            train = self._get_image(row, 0)
            for col_index in range(1, self.cols):
                query = self._get_image(row, col_index)
                if (train is not None) and (query is not None):
                    b = self.matcher.neighbor_match(train, query, 0)
                    jitter[rc_key(row, col_index)] = b[:2]
                    confi[rc_key(row, col_index)] = b[2]
                train = query

        elif axis == 1:
            train = self._get_image(0, col)
            for row_index in range(1, self.rows):
                query = self._get_image(row_index, col)
                if (train is not None) and (query is not None):
                    b = self.matcher.neighbor_match(train, query, 1)
                    jitter[rc_key(row_index, col)] = b[:2]
                    confi[rc_key(row_index, col)] = b[2]
                train = query
        else:
            glog.info("axis value error.")

    def _get_image(self, row, col):
        """
        row: fov in row
        col: fov in col
        """
        temp_index = list(self.images_path.keys())[0]
        if len(temp_index.split('_')[0]) == 4:
            key = rc_key(row, col)
        else: key = rc_key(row, col, source=True)
        if type(self.images_path) is dict:
            if key not in self.images_path.keys():
                return None
            img = self.images_path[key]
            arr = img.get_image()
            if arr.ndim == 3:
                arr = arr[:, :, self.channel]
            return arr

    @staticmethod
    def filter_abnormal_offset(offset: np.array, stitch_confi_mask: np.ndarray, thread: int = 2):
        """
        Filter out abnormal offsets
        :param offset: W*H*C
        :return:
        """
        c = offset.shape[2]
        max_thread = 0
        for i in range(c):
            offset_1 = offset[:, :, i]
            valid_offset = offset_1[np.where((offset_1 != -999) & (stitch_confi_mask > thread))]
            # Remove the maximum and minimum values and take the values between
            # [0.2, 0.8] to calculate the mean and variance
            if len(valid_offset) > 3:
                percentile = np.percentile(valid_offset, (25, 50, 75), interpolation='midpoint')
                Q1, Q3 = percentile[0], percentile[2]
                IQR = Q3 - Q1
                IQR = IQR if IQR > 5 else IQR + 3

                thread_max = Q3 + 1.5 * IQR
                thread_min = Q1 - 1.5 * IQR

                error_fov = np.zeros(shape=(0, 2), dtype=np.int32)
                max_error_fov = np.array(np.where((offset_1 > thread_max))).T
                min_error_fov = np.array(np.where((offset_1 < thread_min) & (offset_1 != -999))).T
                error_fov = np.vstack([error_fov, max_error_fov, min_error_fov])
                if len(error_fov) > 0:
                    for row, col in error_fov:
                        if stitch_confi_mask[row, col] > thread * 3:  # High confidence, relaxed threshold
                            thread_max += IQR * 0.6
                            thread_min -= IQR * 0.6
                        if offset_1[row, col] > thread_max or offset_1[row, col] < thread_min:
                            offset[row, col] = -999
                            stitch_confi_mask[row, col] = 0

                if thread_max > max_thread:
                    max_thread = thread_max
        return offset, stitch_confi_mask, max_thread


class Matcher(object):

    def __init__(self, overlap=0.1):
        self.overlap = overlap
        self.horizontal = True

    def get_roi(self, arr0, arr1):
        """
        get overlap area
        :param arr0: fov 1
        :param arr1: fov 2
        :return: overlap area
        """
        h0, w0 = arr0.shape
        h1, w1 = arr1.shape
        w = int(min(w0, w1) * self.overlap)
        h = int(min(h0, h1) * self.overlap)

        if self.horizontal:
            x0, y0, x1, y1 = [w0 - w, 0, w0, h0]
            x0_, y0_, x1_, y1_ = [0, 0, w, h1]
        else:
            x0, y0, x1, y1 = [0, h0 - h, w0, h0]
            x0_, y0_, x1_, y1_ = [0, 0, w1, h]
        return arr0[y0: y1, x0: x1], arr1[y0_: y1_, x0_: x1_]

    @staticmethod
    def slice_image(image: np.ndarray, slice_width: int, slice_height: int,
                    overlap_height_ratio=0.2, overlap_width_ratio=0.2):
        """
        Image is cropped and stacked
        """
        if image.ndim == 3:
            image_height, image_width, _ = image.shape
        else:
            image_height, image_width = image.shape

        y_overlap = int(overlap_height_ratio * slice_height)
        x_overlap = int(overlap_width_ratio * slice_width)

        slice_bboxes = []
        y_max = y_min = 0
        while y_max < image_height:
            x_min = x_max = 0
            y_max = y_min + slice_height
            while x_max < image_width:
                x_max = x_min + slice_width
                if y_max > image_height or x_max > image_width:
                    xmax = min(image_width, x_max)
                    ymax = min(image_height, y_max)
                    xmin = max(0, xmax - slice_width)
                    ymin = max(0, ymax - slice_height)
                    slice_bboxes.append([xmin, ymin, xmax, ymax])
                else:
                    slice_bboxes.append([x_min, y_min, x_max, y_max])
                x_min = x_max - x_overlap
            y_min = y_max - y_overlap

        n_ims = 0
        sliced_image_result = []
        assert len(slice_bboxes) > 0, "slice image must > 0"
        for slice_bbox in slice_bboxes:
            n_ims += 1
            # extract image
            tlx = slice_bbox[0]
            tly = slice_bbox[1]
            brx = slice_bbox[2]
            bry = slice_bbox[3]
            image_pil_slice = image[tly:bry, tlx:brx]
            sliced_image_result.append(image_pil_slice)
        return sliced_image_result


class SIFTMatcher(Matcher):
    def __init__(self, overlap = 0.1):
        super(SIFTMatcher, self).__init__(overlap = overlap)
        self.sift = cv.SIFT_create()
        self.__matcher = self.matcher()

    def neighbor_match(self, train, query, axis=0):
        if axis == 0:
            self.horizontal = True
        else: self.horizontal = False

        train_local, query_local = self.get_roi(train, query)
        train_local = np.array(train_local, dtype=np.uint8)
        query_local = np.array(query_local, dtype=np.uint8)
        return self.sift_match(train_local, query_local)

    def sift_match(self, train, query):
        psd_kp1, psd_des1 = self.sift.detectAndCompute(train, None)
        kps1 = np.float32([kp.pt for kp in psd_kp1])
        psd_kp2, psd_des2 = self.sift.detectAndCompute(query, None)
        kps2 = np.float32([kp.pt for kp in psd_kp2])
        if len(kps1) == 0 or len(kps2) == 0:
            return None
        try:
            matches = self.__matcher.knnMatch(psd_des1, psd_des2, k=2)
        except:
            return None
        local_match = list()

        for m, n in matches:
            if m.distance < 0.5 * n.distance: local_match.append((m.trainIdx, n.queryIdx))

        if len(local_match) == 0:
            return None
        else:
            pts_a = np.float32([kps1[i] for (_, i) in local_match])
            pts_b = np.float32([kps2[i] for (i, _) in local_match])
            if self.horizontal:
                pts_b[:, 0] += train.shape[1]
            else:
                pts_b[:, 1] += train.shape[0]
            offset = np.median(pts_a - pts_b, axis=0)
            return [offset[0], offset[1], 100]

    @staticmethod
    def matcher():
        flann_index_kd_tree = 1
        index_params = dict(algorithm=flann_index_kd_tree, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        return flann


class FFTMatcher(Matcher):
    def __init__(self, overlap = 0.1):
        super(FFTMatcher, self).__init__(overlap = overlap)

    def pcm_peak(self, image1: np.ndarray, image2: np.ndarray):
        """
        Compute peak correlation matrix for two images and Interpret the translation to find the translation with heighest ncc
        :param image1:the first image (the dimension must be 2)
        :param image2:the second image (the dimension must be 2)
        :return:
        _ncc : the highest ncc
        x : the selected x position
        y : the selected y position
        sub_image1: the overlapping area of image1
        sub_image2: the overlapping area of image2
        """
        assert image1.ndim == 2, "the dimension must be 2"
        assert image2.ndim == 2, "the dimension must be 2"
        size_y, size_x = image1.shape[:2]
        pcm = self._pcm(image1, image2).real
        yins, xins, pcm_ = self._multi_peak_max(pcm)
        lims = np.array([[-size_y, size_y], [-size_x, size_x]])
        max_peak = self._interpret_translation(
            image1, image2, yins, xins, *lims[0], *lims[1]
        )  # opposite to inputted position parameter 
        ncc, offset_y, offset_x, sub_dst, sub_src = max_peak
        return ncc, offset_y, offset_x, sub_dst, sub_src

    def neighbor_match(self, src_img, dst_img, axis=0):
        """
        Calculate the relative stitch offsets of adjacent FOVs
        :param src_img: [NxM], image of source, A is number of image to calculate cross-power specturm
        :param dst_img:[NxM], image of destination
        :return: offset, confidence
        """
        if axis == 0:
            self.horizontal = True
        else: self.horizontal = False

        src_img, dst_img = self.get_roi(src_img, dst_img)
        ncc, offset_y, offset_x, sub_dst, sub_src = self.pcm_peak(src_img, dst_img)
        # Local image confirmation
        if sub_dst is not None:
            sub_dst_crop, sub_src_crop = self._slice_images(sub_dst, sub_src)
        else:
            return None
        if len(sub_src_crop) > 0:
            (y0, x0), confidence0, fft_image = self._stack_peak_pc(sub_dst_crop, sub_src_crop)
            offset_x += x0
            offset_y += y0
        else:
            confidence0 = ncc

        size_y, size_x = dst_img.shape[:2]
        if self.horizontal:
            offset_x = offset_x - size_x
        else:
            offset_y = offset_y - size_y

        return [offset_x, offset_y, confidence0 * 100, ncc * 100]

    @staticmethod
    def ncc(image1, image2):
        """Compute the normalized cross correlation for two images.
        Parameters
        ---------
        image1 : the first image (the dimension must be 2)
        image2 : the second image (the dimension must be 2)
        Returns
        -------
        ncc : the normalized cross correlation
        """
        assert image1.ndim == 2
        assert image2.ndim == 2
        assert np.array_equal(image1.shape, image2.shape)
        image1 = image1.flatten()
        image2 = image2.flatten()
        n = np.dot(image1 - np.mean(image1), image2 - np.mean(image2))
        d = np.linalg.norm(image1) * np.linalg.norm(image2)
        if d < 0.0001:
            return 0
        return n / d

    @staticmethod
    def extract_overlap_subregion(image, y: int, x: int):
        """Extract the overlapping subregion of the image.
        Parameters
        ---------
        image : the image (the dimension must be 2)
        y : the y (second last dim.) position
        x : the x (last dim.) position
        Returns
        subimage : the extracted subimage
        """
        sizeY = image.shape[0]
        sizeX = image.shape[1]
        assert (np.abs(y) < sizeY) and (np.abs(x) < sizeX)
        # clip x to (0, size_Y)
        h_start = int(max(0, min(y, sizeY, key=int), key=int))
        # clip x+sizeY to (0, size_Y)
        h_end = int(max(0, min(y + sizeY, sizeY, key=int), key=int))
        w_start = int(max(0, min(x, sizeX, key=int), key=int))
        w_end = int(max(0, min(x + sizeX, sizeX, key=int), key=int))
        return image[h_start:h_end, w_start:w_end]

    def _slice_images(self, src_img, dst_img):
        """
        crop and stack images
        :param src_img: template image
        :param dst_img: background image
        :return: flod images
        """
        ratio = 0.7
        overlap_size = min(dst_img.shape[:2])
        width = overlap_size if self.horizontal else int(overlap_size * ratio)
        height = int(overlap_size * ratio) if self.horizontal else overlap_size
        src_slice_img = self.slice_image(src_img, slice_width=width, slice_height=height,
                                         overlap_height_ratio=0.2, overlap_width_ratio=0.2, )
        dst_slice_img = self.slice_image(dst_img, slice_width=width, slice_height=height,
                                         overlap_height_ratio=0.2, overlap_width_ratio=0.2, )
        src_slice_img, dst_slice_img = self._fliter_null_information_image(src_slice_img, dst_slice_img)
        return np.array(src_slice_img), np.array(dst_slice_img)

    def _stack_peak_pc(self, stack_im0, stack_im1):
        """
        Computes phase correlation between stack im0 and stack im1 and Interpret the translation to find the translation
        Args:
            stack_im0: the first stack image, N*W*H
            stack_im1: the second stack image, N*W*H
        Returns:
            ret: location
            success: matcher degree
            scps_result: phase correlation image
        """
        # TODO: Implement some form of high-pass filtering of PHASE correlation
        stack_im0 = [self._apodize(im) for im in stack_im0]
        stack_im1 = [self._apodize(im) for im in stack_im1]
        f0, f1 = [np.fft.fft2(arr) for arr in (stack_im0, stack_im1)]
        # spectrum can be filtered (already),
        # so we have to take precaution against dividing by 0
        eps = abs(f1).max() * 1e-15
        cps = abs(np.fft.ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1) + eps)))
        # scps = shifted cps
        scps = np.fft.fftshift(cps)
        scps_result = np.zeros_like(scps[0])
        sum_weights = 0 + 0.00000001
        for i, scp in enumerate(scps):
            scp = cv.GaussianBlur(scp, (7, 7), 1)
            thread = np.percentile(scp, 95)
            scp[np.where(scp < thread)] = 0
            (_, _), success = self._argmax_translation(scp)
            scps_result += scp * success
            sum_weights += success
        scps_result = scps_result / sum_weights
        (t0, t1), success = self._argmax_translation(scps_result)  # (y,x)
        if success < 0.05:
            scps_result = cv.GaussianBlur(scps_result, (7, 7), 1)
            win = cv.createHanningWindow((7, 7), cv.CV_32F)
            scps_result = cv.filter2D(scps_result, -1, win)
            (t0, t1), _ = self._argmax_translation(scps_result)  # (y,x)

        ret = np.array((np.round(t0), np.round(t1)))

        # _compensate_fftshift is not appropriate here, this is OK.
        t0 -= scps[0].shape[0] // 2
        t1 -= scps[0].shape[1] // 2

        ret -= np.array(scps[0].shape, int) // 2
        return ret, success, scps_result

    def _interpret_translation(self,
                               image1: np.ndarray,
                               image2: np.ndarray,
                               yins: np.ndarray,
                               xins: np.ndarray,
                               ymin: int,
                               ymax: int,
                               xmin: int,
                               xmax: int,
                               n: int = 2,
                               ):
        """Interpret the translation to find the translation with heighest ncc.
        Parameters
        ---------
        image1 : the first image (the dimension must be 2)
        image2 : the second image (the dimension must be 2)
        yins : the y positions estimated by PCM
        xins : the x positions estimated by PCM
        ymin : the minimum value of y (second last dim.)
        ymax : the maximum value of y (second last dim.)
        xmin : the minimum value of x (last dim.)
        xmax : the maximum value of x (last dim.)
        n : the number of peaks to check, default is 2.
        Returns
        _ncc : the highest ncc
        x : the selected x position
        y : the selected y position
        """
        assert image1.ndim == 2
        assert image2.ndim == 2
        assert np.array_equal(image1.shape, image2.shape)
        sizeY = image1.shape[0]
        sizeX = image1.shape[1]
        assert np.all(0 <= yins) and np.all(yins < sizeY)
        assert np.all(0 <= xins) and np.all(xins < sizeX)

        _ncc = -np.infty
        x, y = 0, 0
        overlap_img_1 = None
        overlap_img_2 = None

        ymagss = [yins, sizeY - yins]
        ymagss[1][ymagss[0] == 0] = 0
        xmagss = [xins, sizeX - xins]
        xmagss[1][xmagss[0] == 0] = 0

        # concatenate all the candidates
        _poss = []
        for ymags, xmags, ysign, xsign in itertools.product(
                ymagss, xmagss, [-1, +1], [-1, +1]
        ):
            yvals = ymags * ysign
            xvals = xmags * xsign
            _poss.append([yvals, xvals])
        poss = np.array(_poss)
        valid_ind = (
                (ymin <= poss[:, 0, :])
                & (poss[:, 0, :] <= ymax)
                & (xmin <= poss[:, 1, :])
                & (poss[:, 1, :] <= xmax)
        )
        assert np.any(valid_ind)
        valid_ind = np.any(valid_ind, axis=0)
        for pos in np.moveaxis(poss[:, :, valid_ind], -1, 0)[: int(n)]:
            for yval, xval in pos:
                if (ymin <= yval) and (yval <= ymax) and (xmin <= xval) and (xval <= xmax):
                    subI1 = self.extract_overlap_subregion(image1, yval, xval)
                    subI2 = self.extract_overlap_subregion(image2, -yval, -xval)
                    if subI1.size / (sizeX * sizeY) > 0.05 and min(subI1.shape) > 40:  # In the overlap area, at least 10% overlap area
                        ncc_val = self.ncc(subI1, subI2)
                        if ncc_val > _ncc:
                            _ncc = float(ncc_val)
                            y = int(yval)
                            x = int(xval)
                            overlap_img_1 = subI1
                            overlap_img_2 = subI2
        return _ncc, y, x, overlap_img_1, overlap_img_2

    def _get_success(self, array, coord, radius=2):
        """
        Given a coord, examine the array around it and return a number signifying
        how good is the "match".

        Args:
            radius: Get the success as a sum of neighbor of coord of this radius
            coord: Coordinates of the maximum. Float numbers are allowed
                (and converted to int inside)

        Returns:
            Success as float between 0 and 1 (can get slightly higher than 1).
            The meaning of the number is loose, but the higher the better.
        """
        coord = np.round(coord).astype(int)
        coord = tuple(coord)

        subarr = self._get_subarr(array, coord, radius)

        theval = subarr.sum()
        theval2 = array[coord]
        # bigval = np.percentile(array, 97)
        # success = theval / bigval
        # TODO: Think this out
        success = np.sqrt(theval * theval2)
        return success

    def _argmax_translation(self, array):
        # We want to keep the original and here is obvious that
        # it won't get changed inadvertently
        array_orig = array.copy()
        ashape = np.array(array.shape, int)
        mask = np.ones(ashape, float)
        array *= mask

        # WE ARE FFTSHIFTED already.
        # ban translations that are too big
        aporad = (ashape // 6).min()
        mask2 = self._get_apofield(ashape, aporad, corner=True)
        array *= mask2
        # Find what we look for
        tvec = self._argmax_ext(array, 'inf')
        tvec = self._interpolate(array_orig, tvec)

        # If we use constraints or min filter,
        # array_orig[tvec] may not be the maximum
        success = self._get_success(array_orig, tuple(tvec), 2)

        return tvec, success

    def _interpolate(self, array, rough, rad=2):
        """
        Returns index that is in the array after being rounded.

        The result index tuple is in each of its components between zero and the
        array's shape.
        """
        rough = np.round(rough).astype(int)
        surroundings = self._get_subarr(array, rough, rad)
        com = self._argmax_ext(surroundings, 1)
        offset = com - rad
        ret = rough + offset
        # similar to win.wrap, so
        # -0.2 becomes 0.3 and then again -0.2, which is rounded to 0
        # -0.8 becomes - 0.3 -> len() - 0.3 and then len() - 0.8,
        # which is rounded to len() - 1. Yeah!
        ret += 0.5
        ret %= np.array(array.shape).astype(int)
        ret -= 0.5
        return ret

    def _apodize(self, what, aporad=None):
        """
        Given an image, it apodizes it (so it becomes quasi-seamless).
        When ``ratio`` is None, color near the edges will converge
        to the same colour, whereas when ratio is a float number, a blurred
        original image will serve as background.

        Args:
            what: The original image
            aporad (int): Radius [px], width of the band near the edges
                that will get modified
        Returns:
            The apodized image
        """
        if aporad is None:
            mindim = min(what.shape)
            aporad = int(mindim * 0.12)
        apofield = self._get_apofield(what.shape, aporad)
        res = what * apofield
        bg = self._get_borderval(what, aporad // 2)
        res += bg * (1 - apofield)
        return res

    @staticmethod
    def _get_borderval(img, radius=None):
        """
        Given an image and a radius, examine the average value of the image
        at most radius pixels from the edge
        """
        if radius is None:
            mindim = min(img.shape)
            radius = max(1, mindim // 20)
        mask = np.zeros_like(img, dtype=bool)
        mask[:, :radius] = True
        mask[:, -radius:] = True
        mask[radius, :] = True
        mask[-radius:, :] = True

        mean = np.median(img[mask])
        return mean

    @staticmethod
    def _get_apofield(shape, aporad, corner=False):
        """
        Returns an array between 0 and 1 that goes to zero close to the edges.
        """
        if aporad == 0:
            return np.ones(shape, dtype=float)
        apos = np.hanning(aporad * 2)
        vecs = []
        for dim in shape:
            assert dim > aporad * 2, \
                "Apodization radius %d too big for shape dim. %d" % (aporad, dim)
            toapp = np.ones(dim)
            toapp[:aporad] = apos[:aporad]
            toapp[-aporad:] = apos[-aporad:]
            vecs.append(toapp)
        apofield = np.outer(vecs[0], vecs[1])
        if corner:
            apofield[aporad:-aporad] = 1
            apofield[:, aporad:-aporad] = 1

        return apofield

    @staticmethod
    def _multi_peak_max(PCM):
        """Find the first to n th largest peaks in PCM.
        PCM : the peak correlation matrix
        Returns
        rows : the row indices for the peaks
        cols : the column indices for the peaks
        vals : the values of the peaks
        """
        row, col = np.unravel_index(np.argsort(PCM.ravel()), PCM.shape)
        vals = PCM[row[::-1], col[::-1]]
        return row[::-1], col[::-1], vals

    @staticmethod
    def _pcm(image1, image2):
        """Compute peak correlation matrix for two images.
        image1 :  the first image (the dimension must be 2)
        image2 : the second image (the dimension must be 2)
        Returns
        PCM : the peak correlation matrix
        """
        assert image1.ndim == 2
        assert image2.ndim == 2
        assert np.array_equal(image1.shape, image2.shape)
        F1 = np.fft.fft2(image1)
        F2 = np.fft.fft2(image2)
        FC = F1 * np.conjugate(F2)
        return np.fft.ifft2(FC / (np.abs(FC) + 0.0000000001)).real.astype(np.float32)

    @staticmethod
    def _fliter_null_information_image(src_imgs, dst_imgs):
        """
        Filter out sub img that has no information
        :param src_imgs: first crop and stack images
        :param dst_imgs: second crop and stack images
        :return: valid sub images
        """
        src_lists = []
        dst_lists = []
        fov_deepth = 8 if src_imgs[0].dtype == np.uint8 else 16
        img_grad_thread = 0.08 * (2 ** fov_deepth / 256) + 0.12
        for src, dst in zip(src_imgs, dst_imgs):
            src_grad = cv.Sobel(src, -1, 1, 1).std()
            dst_grad = cv.Sobel(dst, -1, 1, 1).std()
            if max(src_grad, dst_grad) > img_grad_thread:
                src_lists.append(src)
                dst_lists.append(dst)
        return src_lists, dst_lists

    @staticmethod
    def _argmax_ext(array, exponent):
        """
        Calculate coordinates of the COM (center of mass) of the provided array.
        Args:
            array (ndarray): The array to be examined.
            exponent (float or 'inf'): The exponent we power the array with. If the
                value 'inf' is given, the coordinage of the array maximum is taken.
        Returns:
            np.ndarray: The COM coordinate tuple, float values are allowed!
        """

        # When using an integer exponent for _argmax_ext, it is good to have the
        # neutral rotation/scale in the center rather near the edges
        if exponent == "inf":
            amax = np.argmax(array)
            ret = list(np.unravel_index(amax, array.shape))
        else:
            col = np.arange(array.shape[0])[:, np.newaxis]
            row = np.arange(array.shape[1])[np.newaxis, :]

            arr2 = array ** exponent
            arrsum = arr2.sum()
            if arrsum == 0:
                # We have to return SOMETHING, so let's go for (0, 0)
                return np.zeros(2)
            arrprody = np.sum(arr2 * col) / arrsum
            arrprodx = np.sum(arr2 * row) / arrsum
            ret = [arrprody, arrprodx]
            # We don't use it, but it still tells us about value distribution

        return np.array(ret)

    @staticmethod
    def _get_subarr(array, center, rad):
        """
        Args:
            array (ndarray): The array to search
            center (2-tuple): The point in the array to search around
            rad (int): Search radius, no radius (i.e. get the single point)
                implies rad == 0
        """
        dim = 1 + 2 * rad
        subarr = np.zeros((dim,) * 2)
        corner = np.array(center) - rad
        for ii in range(dim):
            yidx = corner[0] + ii
            yidx %= array.shape[0]
            for jj in range(dim):
                xidx = corner[1] + jj
                xidx %= array.shape[1]
                subarr[ii, jj] = array[yidx, xidx]
        return subarr

    # @staticmethod
    # def _show_3d(image):
    #     from matplotlib import cm
    #     import matplotlib.pyplot as plt
    #     h, w = image.shape
    #     X = np.arange(0, w, 1)
    #     Y = np.arange(0, h, 1)
    #     X, Y = np.meshgrid(X, Y)
    #     fig = plt.figure()
    #     ax = fig.add_subplot(projection='3d')
    #     ax.plot_surface(X, Y, image, cmap=cm.coolwarm)
    #     plt.show()


if __name__ == '__main__':
    dst_path = r"G:\0423\output\images\0005_0005.tif"
    src_path = r"G:\0423\output\images\0005_0006.tif"

    dst_img = tifffile.imread(dst_path)
    src_img = tifffile.imread(src_path)

    match = FFTMatcher()
    match.overlap = 0.1
    result0 = match.neighbor_match(dst_img, src_img, axis=0)
    # result = match.neighbor_match_v1(dst_img, src_img)
    glog.info('ok')
    pass
