import math
import cv2 as cv
import numpy as np
from scipy.special import softmax
from scipy.stats import rankdata
from scipy.spatial.distance import cdist
from numba import njit, prange, get_num_threads

from cellbin2.utils import clog


def rotate(ptx, pty, angle, original_shape, new_shape):
    px, py = ptx, pty
    ori_h, ori_w = original_shape
    new_h, new_w = new_shape
    cx = ori_w / 2
    cy = ori_h / 2
    rad = math.radians(angle)
    new_px = cx + (px - cx) * math.cos(rad) + (py - cy) * math.sin(rad)
    new_py = cy + -((px - cx) * math.sin(rad)) + (py - cy) * math.cos(rad)
    x_offset, y_offset = (new_w - ori_w) / 2, (new_h - ori_h) / 2
    new_px += x_offset
    new_py += y_offset
    return new_px, new_py


@njit(parallel=True)
def multiply_sum(a, b):
    """
    2023/09/20 @fxzhao calculate the cumulative sum after matrix multiplication
    """
    res = 0
    (h, w) = a.shape
    for i in prange(h):
        for j in range(w):
            res += a[i][j] * b[i][j]
    return res

@njit(fastmath=True, cache=True)
def _mi_from_hist(hist):
    """
    Calculate mutual information from a joint histogram;
    """
    bins = hist.shape[0]
    total = 0.0
    px = np.zeros(bins, dtype=np.float64)
    py = np.zeros(bins, dtype=np.float64)

    for i in range(bins):
        px_sum = 0.0
        for j in range(bins):
            v = hist[i, j]
            total += v
            px_sum += v
            py[j] += v
        px[i] = px_sum

    if total == 0.0:
        return 0.0

    inv_total = 1.0 / total
    mi = 0.0
    for i in range(bins):
        px_i = px[i] * inv_total
        if px_i == 0.0:
            continue
        for j in range(bins):
            py_j = py[j] * inv_total
            if py_j == 0.0:
                continue
            p = hist[i, j] * inv_total
            if p > 0.0:
                mi += p * math.log(p / (px_i * py_j))
    return mi


@njit(parallel=True, fastmath=True)
def _find_min_max(img, n_threads):
    """
    Parallel min/max: find the minimum and maximum in a single pass;
    """
    h, w = img.shape
    local_min = np.full(n_threads, np.inf, dtype=np.float64)
    local_max = np.full(n_threads, -np.inf, dtype=np.float64)

    for t in prange(n_threads):
        mn = np.inf
        mx = -np.inf
        for i in range(t, h, n_threads):
            for j in range(w):
                v = img[i, j]
                if v < mn: mn = v
                if v > mx: mx = v
        local_min[t] = mn
        local_max[t] = mx

    return local_min.min(), local_max.max()


@njit(parallel=True, fastmath=True, cache=True)
def _bin_and_hist_2d(img1, img2, bins, n_threads,
                     min1, max1, min2, max2):
    """
    Fused binning + joint histogram construction: completed in a single parallel pass;
    """
    h, w = img1.shape
    scale1 = (bins - 1) / (max1 - min1)
    scale2 = (bins - 1) / (max2 - min2)
    local = np.zeros((n_threads, bins, bins), dtype=np.int64)

    for t in prange(n_threads):
        for i in range(t, h, n_threads):
            for j in range(w):
                b1 = int((img1[i, j] - min1) * scale1)
                if b1 < 0: b1 = 0
                elif b1 >= bins: b1 = bins - 1
                b2 = int((img2[i, j] - min2) * scale2)
                if b2 < 0: b2 = 0
                elif b2 >= bins: b2 = bins - 1
                local[t, b1, b2] += 1

    hist = np.zeros((bins, bins), dtype=np.int64)
    for t in range(n_threads):
        for i in range(bins):
            for j in range(bins):
                hist[i, j] += local[t, i, j]
    return hist


def mutual_information(image1, image2, bins=64):
    """
    Compute the mutual information between two images (larger values indicate better alignment);
    """
    if image1.shape != image2.shape:
        raise ValueError("images must have the same dimensions")

    if image1.size < 100:
        return 0.0

    n_threads = max(1, get_num_threads())

    min1, max1 = _find_min_max(image1, n_threads)
    min2, max2 = _find_min_max(image2, n_threads)

    if max1 == min1: max1 = min1 + 1e-10
    if max2 == min2: max2 = min2 + 1e-10
    hist = _bin_and_hist_2d(image1, image2, bins, n_threads,
                            min1, max1, min2, max2)
    return _mi_from_hist(hist)


def compute_weighted_rank_score(scores, alpha=0.5, temperature=1.0):
    scores = np.array(scores)
    n = len(scores)
    
    # 1. Ranking score
    ranks = rankdata(-scores)
    rank_score = (n - ranks + 1) / n
    
    # 2. softmax (Margin ratio) 
    diff = scores.max() - scores.min()
    if diff < 1e-10:
        softmax_diff = np.zeros(n)
    else:
        diff_ratio = (scores - scores.min()) / diff
        softmax_diff = softmax(diff_ratio / temperature)
        softmax_diff = (softmax_diff - softmax_diff.min()) / (softmax_diff.max() - softmax_diff.min() + 1e-10)
    
    # 3. rank_score × (1 + α × softmax(diff_ratio))
    score = rank_score * (1 + alpha * softmax_diff)
    return score

def softmax_rank_fusion(scores_sim, scores_mi, alpha=0.5, temperature=1.0):

    scores_sim = np.array(scores_sim)
    scores_mi = np.array(scores_mi)
    
    score_sim = compute_weighted_rank_score(scores_sim, alpha, temperature)
    score_mi = compute_weighted_rank_score(scores_mi, alpha, temperature)
    
    w_sim, w_mi = 0.5,0.5
    
    final_scores = w_sim * score_sim + w_mi * score_mi
    best_idx = np.argmax(final_scores)
    
    return final_scores, best_idx, {'s_sim': score_sim, 's_mi': score_mi}


class AlignByTrack:
    def __init__(self):
        """
        Vision image: generated based on gene expression matrix
        Transformed image obtained based on
            - stitched image
            - scale and rotation

        Transformed image should be in the same scale compared to vision image

        Transformed image and vision image only contain:
            - n times 90 degree rotation (e.g. 90, 180, 270, etc.)
            - x, y direction offsets
        """
        self.search_angle_set = (0, 90, 180, 270)
        self.search_range_x = [-2, -1, 0, 1, 2]
        self.search_range_y = [-2, -1, 0, 1, 2]

        self.x_template = None
        self.y_template = None
        self.fov_size = None
        self.dist_thresh = None
        self.transformed_shape = None
        self.transformed_mass = None
        self.vision_shape = None
        self.vision_mass = None
        self.vision_img = None
        self.transformed_image = None
        self.transformed_cfov_pts = None
        self.vision_cfov_pts = None

        # mi calculate-bin-level
        self.mi_bins = 64
        # downsampling step（SN chip size：0.5→2, 1→5, >1→8）
        self.down_step = 5

    def set_chip_template(self, chip_template):
        self.x_template = chip_template[0]
        self.y_template = chip_template[1]
        self.fov_size = np.sum(self.x_template)
        self.dist_thresh = np.sum(self.x_template) * 2 / 3

    @staticmethod
    def get_mass(image):
        if image.ndim == 3:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        M = cv.moments(image)
        cx_cv = round(M['m10'] / M['m00'])
        cy_cv = round(M['m01'] / M['m00'])
        return cx_cv, cy_cv

    @staticmethod
    def adjust_cross(stitch_template, scale_x, scale_y, fov_stitched_shape,
                     new_shape, chip_template, rotation, flip=True):
        scale_shape = np.array([fov_stitched_shape[0] * scale_y, fov_stitched_shape[1] * scale_x])

        stitch_template[:, 0] = stitch_template[:, 0] * scale_x
        stitch_template[:, 1] = stitch_template[:, 1] * scale_y

        pts = stitch_template[:, :2]
        ids = stitch_template[:, 2:]

        new_px, new_py = rotate(
            pts[:, 0:1],
            pts[:, 1:2],
            rotation,
            original_shape=scale_shape,
            new_shape=new_shape,
        )

        if flip:
            # flip, note that the shape of image is used here, so -1 is required for index calculation
            new_px = new_shape[1] - 1 - new_px
            chip_xlen, chip_ylen = [len(chip_template[0]), len(chip_template[1])]
            ids[:, 0] = chip_xlen - 1 - ids[:, 0]
        pts_ids = np.hstack((new_px, new_py, ids))
        return pts_ids

    @staticmethod
    def flip_points(points, shape, chip_template, axis=0):
        """

        Args:
            points: [x, y]
            shape: [h, w] == [y, x]
            axis:
                - 0: x
                - 1: y
        """
        if axis == 0:
            points[:, 0] = shape[1] - 1 - points[:, 0]
            points[:, 2] = len(chip_template[0]) - 1 - points[:, 2]
        else:
            points[:, 1] = shape[0] - 1 - points[:, 1]
            points[:, 3] = len(chip_template[1]) - 1 - points[:, 3]
        return points

    @staticmethod
    def cal_score(transformed_image, vision_image, offset, mi_bins=64):
        x, y = 0, 0
        x0, y0 = 0, 0
        if offset[0] < 0:
            x = int(round(abs(offset[0])))
            x0 = 0
        else:
            x = 0
            x0 = int(round(abs(offset[0])))
        if offset[1] < 0:
            y = int(round(abs(offset[1])))
            y0 = 0
        else:
            y = 0
            y0 = int(round(abs(offset[1])))
        shape_vision = np.shape(vision_image)
        shape_white = np.shape(transformed_image)
        h = min(shape_vision[0] - y0, shape_white[0] - y)
        w = min(shape_vision[1] - x0, shape_white[1] - x)

        patch_vision = vision_image[y0:y0 + h, x0:x0 + w]
        patch_white = transformed_image[y:y + h, x:x + w]

        score_sim = multiply_sum(patch_vision, patch_white)
        score_mi = mutual_information(patch_white, patch_vision, bins=mi_bins)

        return score_sim, score_mi

    @staticmethod
    def get_pts_based_on_ids(pts_ids, keep_ids=(4, 4)):
        keep = (pts_ids[:, 2] == keep_ids[0]) & (pts_ids[:, 3] == keep_ids[1])
        selected_pts = pts_ids[keep][:, :2]
        return selected_pts

    @staticmethod
    def get_new_shape(old_shape, angle):
        angle = math.radians(angle)
        angle_sin = math.sin(angle)
        angle_cos = math.cos(angle)
        h, w = old_shape
        new_w = round(h * math.fabs(angle_sin) + w * math.fabs(angle_cos))
        new_h = round(w * math.fabs(angle_sin) + h * math.fabs(angle_cos))
        new_shape = (new_h, new_w)
        return new_shape

    @staticmethod
    def get_rough_offset(offset_guess, rot_guess, old_shape, new_shape, transformed_pts, vision_pts, dist_thresh):
        rot_x, rot_y = rotate(
            transformed_pts[:, 0:1],
            transformed_pts[:, 1:2],
            rot_guess,
            old_shape,
            new_shape
        )
        transformed_pts_temp = np.hstack((rot_x, rot_y))

        # get qualified pts
        dist = cdist(transformed_pts_temp + offset_guess, vision_pts)
        qualified = np.min(dist, axis=1) <= dist_thresh
        if not np.any(qualified):
            qualified = np.min(dist, axis=1) <= (dist_thresh + 390)
        transformed_pts_qualified = transformed_pts_temp[qualified]
        dist_qualified = dist[qualified]
        vision_pt_qualified = vision_pts[np.argmin(dist_qualified, axis=1)]

        if len(transformed_pts_qualified) > 0:
            x_offset = -np.median(np.array(transformed_pts_qualified.T[0] - vision_pt_qualified.T[0]))
            y_offset = -np.median(np.array(transformed_pts_qualified.T[1] - vision_pt_qualified.T[1]))
        else:
            x_offset, y_offset = 0, 0
        return x_offset, y_offset

    def search_fov(self, offset_ori, angle):
        white_image = self.transformed_image
        white_image = np.rot90(white_image, angle // 90)
        vision_image = self.vision_img

        offset_list = []
        scores_sim_list = []
        scores_mi_list = []

        for row in self.search_range_x:
            for col in self.search_range_y:
                offset_temp = [
                    offset_ori[0] + col * self.fov_size,
                    offset_ori[1] + row * self.fov_size
                ]
                score_sim, score_mi = self.cal_score(
                    white_image, vision_image,
                    np.array(offset_temp) / self.down_step,
                    mi_bins=self.mi_bins,
                )
                offset_list.append(offset_temp)
                scores_sim_list.append(score_sim)
                scores_mi_list.append(score_mi)
            
                # clog.info(f"Angle: {angle}, offset: {offset_temp}, sim: {score_sim:.6f}, mi: {score_mi:.6f}")

        return offset_list,scores_sim_list,scores_mi_list

    def get_best_in_all_angles_offsets(self):
        all_offsets = []
        all_sim = []
        all_mi = []
        all_angles = []
        for num, angle in enumerate(self.search_angle_set):
            old_shape = self.transformed_shape
            new_shape = self.get_new_shape(old_shape, angle)
            rot_mass_x, rot_mass_y = rotate(
                self.transformed_mass[0],
                self.transformed_mass[1],
                angle,
                self.transformed_shape,
                new_shape,
            )

            white_mass_temp = np.array([rot_mass_x, rot_mass_y])
            offset_temp = self.vision_mass - white_mass_temp
            rough_x_offset, rough_y_offset = self.get_rough_offset(
                offset_guess=offset_temp,
                rot_guess=angle,
                old_shape=old_shape,
                new_shape=new_shape,
                transformed_pts=self.transformed_cfov_pts,
                vision_pts=self.vision_cfov_pts,
                dist_thresh=self.dist_thresh
            )

            if rough_x_offset != 0 and rough_y_offset != 0:
                offsets, sims, mis = self.search_fov(
                    np.array([rough_x_offset, rough_y_offset]), angle
                )
                all_offsets.extend(offsets)
                all_sim.extend(sims)
                all_mi.extend(mis)
                all_angles.extend([num] * len(offsets))

        final_scores, best_idx, details = softmax_rank_fusion(
            np.array(all_sim),
            np.array(all_mi),
            alpha=0.5,
            temperature=0.5
        )

        best_offset = all_offsets[best_idx]
        best_rot_type = all_angles[best_idx]
        best_score = final_scores[best_idx]

        # 调试输出每个配准参数对应计算的分数
        # for angle in self.search_angle_set:
        #     indices = [i for i, a in enumerate(all_angles) if a == angle]
        #     if indices:
        #         max_score = max(final_scores[i] for i in indices)
        #         clog.info(f"Angle: {angle}, Best Final Score: {max_score:.6f}")

        return best_offset, best_rot_type, best_score

    @staticmethod
    def _chip_specif_to_step(chip_specif):
        """Derive the downsampling stride from SN chip_specif [w, h]:
        - min == 0.5 (small chip) → 3
        - max > 1 (large chip)    → 8
        - otherwise (1x1) -> 5
        """
        if not chip_specif:
            return 5
        if min(chip_specif) == 0.5:
            return 3
        if max(chip_specif) > 1:
            return 8
        return 5

    @staticmethod
    def _enhance_vision_image(vision_img, down_size=20, clip_limit=3.0, tile_grid=(16, 16), step=5):
        """MI vision pre-process:
        20x20 block-sum → p1/p99 clipping → CLAHE → resize to H/step, W/step;
        the step must match the DAPI downsampling stride to ensure both images are at the same scale;
        """
        h, w = vision_img.shape
        new_h = h // down_size
        new_w = w // down_size
        cropped = vision_img[:new_h * down_size, :new_w * down_size].astype(np.float32)
        sum_image = cropped.reshape(new_h, down_size, new_w, down_size).sum(axis=(1, 3))
        nz = sum_image[sum_image > 0]
        if nz.size > 0:
            p1, p99 = np.percentile(nz, [1, 99])
        else:
            p1, p99 = 0.0, 1.0
        if p99 <= p1:
            p99 = p1 + 1e-6
        stretched = np.clip(
            (sum_image - p1) / (p99 - p1) * 255.0, 0, 255
        ).astype(np.uint8)
        clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        boosted = clahe.apply(stretched)
        return cv.resize(boosted, (w // step, h // step), interpolation=cv.INTER_LINEAR)


    @staticmethod
    def _down_sample_normalize(img, clip_limit=3.0, tile_grid=(16, 16), step=5):
        """MI image pre-process:
        downsample by step → p1/p99 clipping → CLAHE, output uint8 0-255;
        """
        img = img[::step, ::step]
        if img.dtype != np.uint8:
            img_u8 = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        else:
            img_u8 = img
        p1, p99 = np.percentile(img_u8, [1, 99])
        if p99 <= p1:
            p99 = p1 + 1
        stretched = np.clip(
            (img_u8.astype(np.float32) - p1) / (p99 - p1) * 255.0, 0, 255
        ).astype(np.uint8)
        clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        return clahe.apply(stretched)


    def run(self,
            transformed_image,
            vision_img,
            vision_cp,
            stitch_tc,
            flip,
            rot90_flag,
            chip_specif=None
            ):
        if not rot90_flag:
            self.search_angle_set = tuple([0])

        if transformed_image.ndim == 3:
            transformed_image = cv.cvtColor(transformed_image, cv.COLOR_BGR2GRAY)

        self.down_step = self._chip_specif_to_step(chip_specif)
        self.transformed_shape = transformed_image.shape
        self.vision_shape = vision_img.shape

        if flip:
            self.transformed_image = np.fliplr(transformed_image)
            stitch_tc = self.flip_points(
                stitch_tc, self.transformed_shape,
                [self.x_template, self.y_template]
            )
        else:
            self.transformed_image = transformed_image

        self.transformed_mass = self.get_mass(self.transformed_image)
        self.vision_mass = self.get_mass(vision_img)

        self.transformed_image = self._down_sample_normalize(self.transformed_image, step=self.down_step)
        self.vision_img = self._enhance_vision_image(vision_img, step=self.down_step)

        self.transformed_cfov_pts = self.get_pts_based_on_ids(stitch_tc)
        self.vision_cfov_pts = self.get_pts_based_on_ids(vision_cp)

        offset, rot_type, score = self.get_best_in_all_angles_offsets()
        return offset, rot_type, score


if __name__ == '__main__':
    pass
