# -*- coding: utf-8 -*-
import sys
from pathlib import Path
from typing import Union

import tifffile
import cv2 as cv
import numpy as np

from cellbin2.utils.common import TechType
from cellbin2.contrib.alignment import ChipFeature
from cellbin2.contrib.alignment.basic import Alignment, AlignMode, ChipBoxInfo
from cellbin2.utils import clog
from cellbin2.image import CBImage
from cellbin2.image import cbimread, cbimwrite
from cellbin2.contrib.chip_detector import ChipParam, detect_chip


class MatrixBoxDetector(object):
    def __init__(self, binsize=21, down_size=4, morph_size=9, gene_base_size=9996):
        self.gene_base_size = gene_base_size
        self.binsize = binsize
        self.down_size = down_size
        self.morph_size = morph_size
        self.image: np.ndarray = np.array([])

    @staticmethod
    def _walks_image(dst, box_size, begin_xy=None, end_xy=None):
        if begin_xy is not None:
            bx, by = begin_xy
        else:
            bx = by = 0

        if end_xy is not None:
            ex, ey = end_xy
        else:
            ey, ex = np.array(dst.shape) - box_size

        max_value = 0
        max_x = max_y = 0

        for i in range(by, ey):
            for j in range(bx, ex):
                _dst = dst[i: i + box_size[0], j: j + box_size[1]]
                s = np.sum(_dst)
                if s > max_value:
                    max_value = s
                    max_x = j
                    max_y = i

        return max_x, max_y

    def get_box4maximize_area(self, dst, box_size, min_size=256, step=5):
        if min(dst.shape) < min_size:
            x, y = self._walks_image(dst, box_size)
            return x, y
        else:
            _dst = cv.pyrDown(dst)
            x, y = self.get_box4maximize_area(_dst, box_size=box_size // 2)
            x, y = map(lambda k: k * 2, (x, y))

            x, y = self._walks_image(
                dst,
                box_size=box_size,
                begin_xy=[x - step, y - step],
                end_xy=[x + step, y + step]
            )
            return x, y

    def detect(self, matrix: np.ndarray, chip_size: Union[int, float, list, tuple] = 1.0):
        if isinstance(chip_size, (int, float)):
            chip_size = [chip_size, chip_size]

        gene_size = np.array([
            int(self.gene_base_size * chip_size[0]),
            int(self.gene_base_size * chip_size[1])
        ])

        self.image = matrix
        if self.image.ndim == 3:
            self.image = self.image[:, :, 0]

        gene_image = cv.filter2D(self.image, -1, np.ones((self.binsize, self.binsize), np.float32))
        _, gene_image = cv.threshold(gene_image, 0, 255, cv.THRESH_BINARY)

        gene_image_s = cv.resize(
            gene_image,
            (gene_image.shape[1] // self.down_size, gene_image.shape[0] // self.down_size)
        )

        x, y = self.get_box4maximize_area(gene_image_s, box_size=gene_size // self.down_size)

        x = x * self.down_size + gene_size[1] / 2
        y = y * self.down_size + gene_size[0] / 2
        center = [x, y]

        new_box = [[center[0] + i * gene_size[1] / 2, center[1] + j * gene_size[0] / 2]
                   for i in [-1, 1] for j in [-1, 1]]
        new_box = np.array(new_box)[(0, 1, 3, 2), :]

        return new_box


def detect_chip_box(
        matrix: np.ndarray,
        chip_size: Union[int, float, list, tuple] = 1.0
) -> ChipBoxInfo:
    mbd = MatrixBoxDetector()

    if isinstance(chip_size, (list, tuple)):
        chip_size = chip_size[::-1]

    box = mbd.detect(matrix, chip_size)
    cbi = ChipBoxInfo(
        LeftTop=box[0],
        LeftBottom=box[1],
        RightBottom=box[2],
        RightTop=box[3]
    )
    return cbi


class ChipAlignment(Alignment):
    """
    coarse alignment:
        flip + rot90 + uniform scale + translation
        (NO corner-pulling affine)

    fine alignment:
        imreg_dft.similarity: angle / scale
        hierarchical offset search
        center 2000x2000 masked out
    """

    def __init__(
            self,
            flip_flag: bool = False,
            rot90_flag: bool = True
    ):
        super(ChipAlignment, self).__init__()

        self._pre_transform = {
            "scale": 1.0,
            "rotation": 0.0,
            "applied": False,
        }

        self.register_matrix: np.matrix = None
        self.transforms_matrix: np.matrix = None

        self.rot90_flag = rot90_flag
        self._hflip = flip_flag
        self.no_trans_flag = False

        self.max_length = 9996

        self._fixed_image = None
        self._flip_state = 0
        self._rot90 = 0
        self._offset = (0, 0)
        self._score = 0.0

        self._fine_scale = 1.0
        self._fine_angle = 0.0
        self._fine_offset = (0, 0)
        self._fine_alignment = None

        self._coarse_registered_full = None
        self._final_registered_full = None

        self._coarse_scale = 1.0
        self._coarse_translation = (0.0, 0.0)

    def registration_image(self, file: Union[str, np.ndarray, CBImage]):
        """
        return final numpy array
        """
        if self._final_registered_full is not None:
            return self._final_registered_full

        if not isinstance(file, CBImage):
            image = cbimread(file)
        else:
            image = file

        if self.no_trans_flag:
            return None

        s = max(self._scale_x, self._scale_y)

        result = image.trans_image(
            scale=[1 / s, 1 / s],
            rotate=self._rotation,
            rot90=self._rot90,
            offset=self._offset,
            dst_size=self._fixed_image.mat.shape,
            flip_lr=bool(self._flip_state)
        )

        if hasattr(result, "_image"):
            return result._image
        if hasattr(result, "image"):
            return result.image
        return result

    @staticmethod
    def _fill_image(image: np.ndarray, chip_box: np.ndarray):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        poly = np.int32(chip_box).reshape(-1, 1, 2)
        cv.drawContours(mask, [poly], -1, 1, cv.FILLED)

        if image.ndim == 3:
            return image * mask[:, :, None]
        return image * mask

    @staticmethod
    def _to_gray(image: np.ndarray):
        if image.ndim == 3:
            return image[:, :, 0]
        return image

    def _crop_roi_by_box(self, image: np.ndarray, chip_box: np.ndarray):
        image_2d = self._to_gray(image)

        lu_x, lu_y = np.floor(chip_box[0]).astype(int)
        rd_x, rd_y = np.ceil(chip_box[2]).astype(int)

        h, w = image_2d.shape[:2]
        lu_x = max(0, min(lu_x, w - 1))
        rd_x = max(lu_x + 1, min(rd_x, w))
        lu_y = max(0, min(lu_y, h - 1))
        rd_y = max(lu_y + 1, min(rd_y, h))

        roi = image_2d[lu_y:rd_y, lu_x:rd_x]
        return roi, (lu_x, lu_y, rd_x, rd_y)

    @staticmethod
    def _center_crop(image: np.ndarray, crop_size: int = 2000):
        h, w = image.shape[:2]
        crop_h = min(crop_size, h)
        crop_w = min(crop_size, w)

        cy = h // 2
        cx = w // 2

        y0 = max(0, cy - crop_h // 2)
        x0 = max(0, cx - crop_w // 2)
        y1 = min(h, y0 + crop_h)
        x1 = min(w, x0 + crop_w)

        return image[y0:y1, x0:x1], (x0, y0, x1, y1)

    @staticmethod
    def _make_center_hole_mask(shape, hole_size=2500):
        """
        外围=1，中间 hole=0
        """
        h, w = shape[:2]
        mask = np.ones((h, w), dtype=np.float32)

        hole_h = min(hole_size, h)
        hole_w = min(hole_size, w)

        cy = h // 2
        cx = w // 2

        y0 = max(0, cy - hole_h // 2)
        x0 = max(0, cx - hole_w // 2)
        y1 = min(h, y0 + hole_h)
        x1 = min(w, x0 + hole_w)

        mask[y0:y1, x0:x1] = 0.0
        return mask

    @staticmethod
    def _apply_mask(image: np.ndarray, mask: np.ndarray):
        if image.ndim == 3:
            return image * mask[:, :, None]
        return image * mask

    def _estimate_angle_scale_with_imreg(self, fixed_roi, moving_roi, center_hole_size=2500):
        import imreg_dft

        fixed_roi = fixed_roi.astype(np.float32)
        moving_roi = moving_roi.astype(np.float32)

        mask = self._make_center_hole_mask(fixed_roi.shape, hole_size=center_hole_size)
        fixed_roi = fixed_roi * mask
        moving_roi = moving_roi * mask

        down_size = 4000
        max_hw = max(fixed_roi.shape[:2])
        down_scale = max(1.0, max_hw / down_size)

        if down_scale > 1.0:
            new_w = int(round(fixed_roi.shape[1] / down_scale))
            new_h = int(round(fixed_roi.shape[0] / down_scale))

            fixed_roi_small = cv.resize(
                fixed_roi,
                (new_w, new_h),
                interpolation=cv.INTER_AREA
            )
            moving_roi_small = cv.resize(
                moving_roi,
                (new_w, new_h),
                interpolation=cv.INTER_AREA
            )
        else:
            fixed_roi_small = fixed_roi
            moving_roi_small = moving_roi

        ret = imreg_dft.similarity(
            fixed_roi_small,
            moving_roi_small
        )

        angle = float(ret.get("angle", 0.0))
        scale = float(ret.get("scale", 1.0))
        return angle, scale, ret

    @staticmethod
    def _apply_angle_scale_only(img: np.ndarray, angle: float, scale: float):
        h, w = img.shape[:2]
        center = (w / 2.0, h / 2.0)
        M = cv.getRotationMatrix2D(center, angle, scale)
        out = cv.warpAffine(
            img,
            M,
            (w, h),
            flags=cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT,
            borderValue=0
        )
        return out

    @staticmethod
    def _shift_image(img: np.ndarray, dx: int, dy: int):
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv.warpAffine(
            img,
            M,
            (img.shape[1], img.shape[0]),
            flags=cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT,
            borderValue=0
        )

    @staticmethod
    def _box_center(box: np.ndarray):
        return np.mean(box, axis=0).astype(np.float32)

    @staticmethod
    def _box_size(box: np.ndarray):
        box = np.asarray(box, dtype=np.float32)
        top_w = np.linalg.norm(box[3] - box[0])
        bottom_w = np.linalg.norm(box[2] - box[1])
        left_h = np.linalg.norm(box[1] - box[0])
        right_h = np.linalg.norm(box[2] - box[3])

        width = (top_w + bottom_w) / 2.0
        height = (left_h + right_h) / 2.0
        return float(width), float(height)

    @staticmethod
    def _flip_lr_points(points: np.ndarray, shape_hw):
        h, w = shape_hw[:2]
        pts = points.copy().astype(np.float32)
        pts[:, 0] = (w - 1) - pts[:, 0]

        order = np.argsort(pts[:, 0] + pts[:, 1] * 1e-3)
        left_two = pts[order[:2]]
        right_two = pts[order[2:]]

        left_two = left_two[np.argsort(left_two[:, 1])]
        right_two = right_two[np.argsort(right_two[:, 1])]

        lt, lb = left_two
        rt, rb = right_two
        return np.array([lt, lb, rb, rt], dtype=np.float32)

    @staticmethod
    def _rot90_points_once_ccw(points: np.ndarray, shape_hw):
        """
        对应 np.rot90(img, 1)，逆时针90°
        (x, y) -> (y, w-1-x)
        """
        h, w = shape_hw[:2]
        pts = points.copy().astype(np.float32)
        new_pts = np.zeros_like(pts)
        new_pts[:, 0] = pts[:, 1]
        new_pts[:, 1] = (w - 1) - pts[:, 0]

        new_shape = (w, h)

        order = np.argsort(new_pts[:, 0] + new_pts[:, 1] * 1e-3)
        left_two = new_pts[order[:2]]
        right_two = new_pts[order[2:]]

        left_two = left_two[np.argsort(left_two[:, 1])]
        right_two = right_two[np.argsort(right_two[:, 1])]

        lt, lb = left_two
        rt, rb = right_two
        new_pts = np.array([lt, lb, rb, rt], dtype=np.float32)

        return new_pts, new_shape

    def _apply_flip_rot_to_image_and_box(self, image: np.ndarray, box: np.ndarray, flip_state: int, rot_index: int):
        """
        只做 flip + rot90，不做仿射
        """
        out_img = image.copy()
        out_box = box.copy().astype(np.float32)
        cur_shape = out_img.shape[:2]

        if flip_state == 1:
            out_img = np.fliplr(out_img)
            out_box = self._flip_lr_points(out_box, cur_shape)
            cur_shape = out_img.shape[:2]

        k = rot_index % 4
        for _ in range(k):
            out_img = np.rot90(out_img, 1)
            out_box, cur_shape = self._rot90_points_once_ccw(out_box, cur_shape)

        return np.ascontiguousarray(out_img), out_box

    def _build_similarity_matrix_from_boxes(self, moving_box: np.ndarray, fixed_box: np.ndarray):
        """
        uniform scale + translation
        不做角点强拉，不做 shear
        """
        mc = self._box_center(moving_box)
        fc = self._box_center(fixed_box)

        mw, mh = self._box_size(moving_box)
        fw, fh = self._box_size(fixed_box)

        moving_diag = np.sqrt(mw * mh)
        fixed_diag = np.sqrt(fw * fh)

        if moving_diag < 1e-6:
            scale = 1.0
        else:
            scale = fixed_diag / moving_diag

        tx = fc[0] - scale * mc[0]
        ty = fc[1] - scale * mc[1]

        M = np.array([
            [scale, 0.0, tx],
            [0.0, scale, ty]
        ], dtype=np.float32)

        return M, float(scale), (float(tx), float(ty))

    @staticmethod
    def _apply_affine_to_points(points: np.ndarray, M: np.ndarray):
        pts = np.asarray(points, dtype=np.float32)
        ones = np.ones((pts.shape[0], 1), dtype=np.float32)
        pts_h = np.hstack([pts, ones])
        out = (M @ pts_h.T).T
        return out.astype(np.float32)

    def _search_offset_by_multiply_sum(
            self,
            fixed_roi,
            moving_roi_after_angle_scale,
            shift_range=100,
            coarse_downsample=4,
            coarse_step=8,
            mid_step=2,
            fine_span=3,
            center_hole_size=2500
    ):
        """
        分层 offset 搜索 + 中心 2500x2500 挖空
        """
        fixed_roi = fixed_roi.astype(np.float32)
        moving_roi_after_angle_scale = moving_roi_after_angle_scale.astype(np.float32)

        mask = self._make_center_hole_mask(fixed_roi.shape, hole_size=center_hole_size)
        fixed_roi = fixed_roi * mask
        moving_roi_after_angle_scale = moving_roi_after_angle_scale * mask

        if coarse_downsample > 1:
            h, w = fixed_roi.shape[:2]
            ds_w = max(32, w // coarse_downsample)
            ds_h = max(32, h // coarse_downsample)

            fixed_ds = cv.resize(fixed_roi, (ds_w, ds_h), interpolation=cv.INTER_AREA)
            moving_ds = cv.resize(moving_roi_after_angle_scale, (ds_w, ds_h), interpolation=cv.INTER_AREA)

            ds_range = max(1, int(round(shift_range / coarse_downsample)))
        else:
            fixed_ds = fixed_roi
            moving_ds = moving_roi_after_angle_scale
            ds_range = shift_range

        best = {"score": -np.inf, "dx": 0, "dy": 0}

        # stage 1
        for dx_ds in range(-ds_range, ds_range + 1, coarse_step):
            for dy_ds in range(-ds_range, ds_range + 1, coarse_step):
                moved = self._shift_image(moving_ds, dx_ds, dy_ds)
                score = self.multiply_sum(moved, fixed_ds)
                if score > best["score"]:
                    best = {
                        "score": score,
                        "dx": int(dx_ds * coarse_downsample),
                        "dy": int(dy_ds * coarse_downsample)
                    }

        # stage 2
        dx0, dy0 = best["dx"], best["dy"]
        mid_best = {"score": -np.inf, "dx": dx0, "dy": dy0}

        mid_span = coarse_downsample * coarse_step + mid_step
        for dx in range(dx0 - mid_span, dx0 + mid_span + 1, mid_step):
            for dy in range(dy0 - mid_span, dy0 + mid_span + 1, mid_step):
                moved = self._shift_image(moving_roi_after_angle_scale, dx, dy)
                score = self.multiply_sum(moved, fixed_roi)
                if score > mid_best["score"]:
                    mid_best = {"score": score, "dx": dx, "dy": dy}

        # stage 3
        dx1, dy1 = mid_best["dx"], mid_best["dy"]
        fine_best = {"score": -np.inf, "dx": dx1, "dy": dy1}

        for dx in range(dx1 - fine_span, dx1 + fine_span + 1):
            for dy in range(dy1 - fine_span, dy1 + fine_span + 1):
                moved = self._shift_image(moving_roi_after_angle_scale, dx, dy)
                score = self.multiply_sum(moved, fixed_roi)
                if score > fine_best["score"]:
                    fine_best = {"score": score, "dx": dx, "dy": dy}

        return fine_best

    @staticmethod
    def _apply_fine_transform_about_point(image: np.ndarray, angle=0.0, scale=1.0, offset=(0, 0), center_xy=None):
        """
        apply angle + scale + offset to coarse_registered_full
        """
        h, w = image.shape[:2]
        if center_xy is None:
            center_xy = (w / 2.0, h / 2.0)

        cx, cy = center_xy
        M = cv.getRotationMatrix2D((float(cx), float(cy)), float(angle), float(scale))
        M[0, 2] += float(offset[0])
        M[1, 2] += float(offset[1])

        warped = cv.warpAffine(
            image,
            M,
            (w, h),
            flags=cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT,
            borderValue=0
        )
        return warped, M

    def align_stitched(
            self,
            fixed_image: ChipFeature,
            moving_image: ChipFeature
    ):
        self._rotation = moving_image.chip_box.Rotation
        self._scale_x = moving_image.chip_box.ScaleX
        self._scale_y = moving_image.chip_box.ScaleY
        self._fixed_image = fixed_image

        if self.no_trans_flag:
            self.align_transformed(fixed_image, moving_image)
        else:
            s = max(self._scale_x, self._scale_y)

            self._pre_transform = {
                "scale": 1 / s,
                "rotation": self._rotation,
                "applied": True,
            }

            transformed_image = moving_image.mat.trans_image(
                scale=[1 / s, 1 / s],
                rotate=self._rotation
            )

            transformed_feature = ChipFeature()
            transformed_feature.set_mat(transformed_image)

            trans_mat = self.get_coordinate_transformation_matrix(
                moving_image.mat.shape,
                [1 / s, 1 / s],
                self._rotation
            )

            trans_points = self.get_points_by_matrix(moving_image.chip_box.chip_box, trans_mat)
            transformed_feature.chip_box.set_chip_box(trans_points)

            self.align_transformed(fixed_image, transformed_feature)

    def _run_fine_for_candidate(
            self,
            fixed_image: ChipFeature,
            moving_image: ChipFeature,
            fixed_box: np.ndarray,
            flip_state: int,
            rot_index: int,
    ) -> dict:
        """Run the full fine alignment for one coarse (flip, rot) candidate.

        Rebuilds the full-resolution coarse warp for this flip/rot, then runs
        imreg angle/scale + hierarchical offset search + the final warp. Returns
        a dict carrying the final registered image, fine params, the
        ``fine_alignment`` dict, and a ``final_score`` (multiply_sum over the
        chip-box ROI with the center hole masked) used to disambiguate
        180°-tied coarse candidates.
        """
        res = {
            "flip_state": flip_state,
            "rot_index": rot_index,
            "coarse_registered_full": None,
            "final_registered_full": None,
            "fine_angle": 0.0,
            "fine_scale": 1.0,
            "fine_offset": (0, 0),
            "coarse_scale": 1.0,
            "coarse_translation": (0.0, 0.0),
            "fine_alignment": {},
            "final_score": -np.inf,
            "ok": False,
            "err": None,
        }
        try:
            # ---------- full-res coarse warp for this flip/rot ----------
            coarse_input, coarse_box = self._apply_flip_rot_to_image_and_box(
                moving_image.mat.image,
                moving_image.chip_box.chip_box.astype(np.float32),
                flip_state=flip_state,
                rot_index=rot_index,
            )
            coarse_input_gray = self._to_gray(coarse_input)
            coarse_M, coarse_scale, coarse_trans = self._build_similarity_matrix_from_boxes(
                coarse_box,
                fixed_image.chip_box.chip_box.astype(np.float32),
            )
            coarse_registered_full = cv.warpAffine(
                coarse_input_gray,
                coarse_M,
                (fixed_image.mat.shape[1], fixed_image.mat.shape[0]),
                flags=cv.INTER_LINEAR,
                borderMode=cv.BORDER_CONSTANT,
                borderValue=0,
            )
            res["coarse_registered_full"] = coarse_registered_full

            # ---------- ROI ----------
            fixed_roi, roi_bbox = self._crop_roi_by_box(
                fixed_image.mat.image, fixed_box,
            )
            moving_roi, _ = self._crop_roi_by_box(
                coarse_registered_full, fixed_box,
            )

            # ---------- fine: angle / scale (imreg) ----------
            fine_angle, fine_scale, _ = self._estimate_angle_scale_with_imreg(
                fixed_roi, moving_roi,
            )

            # A large angle here means the ROI is 180°-flipped relative to the
            # fixed image (the wrong 180° branch). Clamp so this candidate stays
            # mis-aligned and scores low, letting the correct branch win the
            # disambiguation comparison below.
            if abs(fine_angle) > 5:
                clog.info(
                    f"[cand flip{flip_state}/rot{rot_index * 90}] "
                    f"angle {fine_angle:.3f}° -> clamp 0"
                )
                fine_angle = 0.0
            if abs(fine_scale - 1.0) > 0.05:
                clog.info(
                    f"[cand flip{flip_state}/rot{rot_index * 90}] "
                    f"scale {fine_scale:.6f} -> clamp 1.0"
                )
                fine_scale = 1.0

            moving_roi_rs = self._apply_angle_scale_only(
                moving_roi, fine_angle, fine_scale,
            )

            # ---------- fine: offset ----------
            offset_ret = self._search_offset_by_multiply_sum(
                fixed_roi=fixed_roi,
                moving_roi_after_angle_scale=moving_roi_rs,
                shift_range=200,
                coarse_downsample=4,
                coarse_step=8,
                mid_step=2,
                fine_span=3,
            )
            fine_dx = int(offset_ret["dx"])
            fine_dy = int(offset_ret["dy"])

            # ---------- apply fine to coarse result ----------
            chip_center = tuple(np.mean(fixed_box, axis=0))
            final_registered_full, _ = self._apply_fine_transform_about_point(
                coarse_registered_full,
                angle=fine_angle,
                scale=fine_scale,
                offset=(fine_dx, fine_dy),
                center_xy=chip_center,
            )
            res["final_registered_full"] = final_registered_full

            # ---------- final similarity score (chip-box ROI, masked) ----------
            final_roi, _ = self._crop_roi_by_box(final_registered_full, fixed_box)
            f_roi = fixed_roi.astype(np.float32)
            m_roi = final_roi.astype(np.float32)
            h = min(f_roi.shape[0], m_roi.shape[0])
            w = min(f_roi.shape[1], m_roi.shape[1])
            f_roi = f_roi[:h, :w]
            m_roi = m_roi[:h, :w]
            mask = self._make_center_hole_mask(f_roi.shape, hole_size=2500)
            final_score = float(self.multiply_sum(m_roi * mask, f_roi * mask))

            res.update({
                "fine_angle": fine_angle,
                "fine_scale": fine_scale,
                "fine_offset": (fine_dx, fine_dy),
                "coarse_scale": float(coarse_scale),
                "coarse_translation": tuple(coarse_trans),
                "fine_alignment": {
                    "pre_transform": self._pre_transform,
                    "coarse_scale": float(coarse_scale),
                    "coarse_translation": tuple(coarse_trans),
                    "offset": (fine_dx, fine_dy),
                    "scale": fine_scale,
                    "angle": fine_angle,
                    "roi_bbox": roi_bbox,
                    "fine_center": chip_center,
                },
                "final_score": final_score,
                "ok": True,
            })
        except Exception as e:
            res["err"] = str(e)
            clog.warning(
                f"[cand flip{flip_state}/rot{rot_index * 90}] fine failed: {e}"
            )
        return res

    def align_transformed(
            self,
            fixed_image: ChipFeature,
            moving_image: ChipFeature
    ):
        self._fixed_image = fixed_image
        moving_image.set_mat(
            self._fill_image(moving_image.mat.image, moving_image.chip_box.chip_box)
        )

        register_info = dict()
        down_size = max(1, max(fixed_image.mat.shape) // self.max_length)

        fixed_gray = self._to_gray(fixed_image.mat.image)
        fixed_box = fixed_image.chip_box.chip_box.astype(np.float32)

        fixed_gray_ds = fixed_gray[::down_size, ::down_size]
        fixed_box_ds = fixed_box / down_size

        # ---------- coarse: flip + rot90 + uniform scale + translation ----------
        for flip_state in [0, 1]:
            base_img = moving_image.mat.image
            base_box = moving_image.chip_box.chip_box.astype(np.float32)

            for rot_index in range(4):
                img_r, box_r = self._apply_flip_rot_to_image_and_box(
                    base_img,
                    base_box,
                    flip_state=flip_state,
                    rot_index=rot_index
                )

                img_r_gray = self._to_gray(img_r)
                img_r_gray_ds = img_r_gray[::down_size, ::down_size]
                box_r_ds = box_r / down_size

                M_ds, scale_ds, trans_ds = self._build_similarity_matrix_from_boxes(
                    box_r_ds,
                    fixed_box_ds
                )

                reg_ds = cv.warpAffine(
                    img_r_gray_ds,
                    M_ds,
                    (fixed_gray_ds.shape[1], fixed_gray_ds.shape[0]),
                    flags=cv.INTER_LINEAR,
                    borderMode=cv.BORDER_CONSTANT,
                    borderValue=0
                )

                lu_x, lu_y = map(int, fixed_box_ds[0])
                rd_x, rd_y = map(int, fixed_box_ds[2])

                _wsi_image = reg_ds[lu_y: rd_y, lu_x: rd_x]
                _gene_image = fixed_gray_ds[lu_y: rd_y, lu_x: rd_x]

                ms = self.multiply_sum(_wsi_image, _gene_image)

                transform_key = f"flip_{flip_state}_rot_{rot_index}"
                '''clog.info(
                    f"Flip: {flip_state}, Rot: {rot_index * 90}°, "
                    f"Scale: {scale_ds:.6f}, Score: {ms}"
                )'''

                register_info[transform_key] = {
                    "score": ms,
                    "flip_state": flip_state,
                    "rot_index": rot_index,
                    "scale": scale_ds,
                    "translation": trans_ds,
                    "box_after_flip_rot": box_r.copy(),
                }

        # ---------- rank coarse candidates ----------
        ranked = sorted(
            register_info.items(),
            key=lambda x: x[1]["score"],
            reverse=True,
        )
        top_score = ranked[0][1]["score"]

        # multiply_sum (a dot-product metric) cannot distinguish orientations
        # differing by 180°, so the correct branch and its 180° twin near-tie
        # in the coarse score. Take every candidate within a relative margin
        # of the top score (at least 2) and resolve the tie by running the full
        # fine alignment on each and comparing the final similarity.
        rel_margin = 0.85
        selected = [it for it in ranked if it[1]["score"] >= rel_margin * top_score]
        if len(selected) < 2:
            selected = ranked[:2]
        # The 180° twin is always the coarse top-2 (it near-ties the leader);
        # ranks 3+ are strictly lower and never the 180°-ambiguous element, so
        # they cannot change the winner. Cap at 2 to halve the fine-alignment
        # cost without losing disambiguation power.
        selected = selected[:2]

        clog.info(
            "Coarse candidate(s) entering 180° disambiguation: "
            + ", ".join(f"{k}={v['score']:.3e}" for k, v in selected)
        )

        # ---------- full fine alignment on each selected candidate ----------
        fine_results = []
        for key, info in selected:
            fr = self._run_fine_for_candidate(
                fixed_image=fixed_image,
                moving_image=moving_image,
                fixed_box=fixed_box,
                flip_state=info["flip_state"],
                rot_index=info["rot_index"],
            )
            fr["coarse_key"] = key
            fr["coarse_score"] = float(info["score"])
            fine_results.append(fr)
            clog.info(
                f"[cand {key}] coarse={fr['coarse_score']:.3e} "
                f"angle={fr['fine_angle']:.4f}° scale={fr['fine_scale']:.6f} "
                f"offset={fr['fine_offset']} final_score={fr['final_score']:.3e} "
                f"ok={fr['ok']}"
            )

        # ---------- pick winner by final similarity score ----------
        fine_results.sort(key=lambda r: r["final_score"], reverse=True)
        winner = fine_results[0]

        # all-failed fallback: use the best coarse candidate's coarse-only warp
        if not winner["ok"] or winner["final_registered_full"] is None:
            clog.warning(
                "All fine-alignment candidates failed; falling back to coarse-only "
                f"for best coarse candidate {ranked[0][0]}"
            )
            fb_info = ranked[0][1]
            coarse_input, coarse_box = self._apply_flip_rot_to_image_and_box(
                moving_image.mat.image,
                moving_image.chip_box.chip_box.astype(np.float32),
                flip_state=fb_info["flip_state"],
                rot_index=fb_info["rot_index"],
            )
            coarse_input_gray = self._to_gray(coarse_input)
            coarse_M, coarse_scale, coarse_trans = self._build_similarity_matrix_from_boxes(
                coarse_box,
                fixed_image.chip_box.chip_box.astype(np.float32),
            )
            coarse_registered_full = cv.warpAffine(
                coarse_input_gray,
                coarse_M,
                (fixed_image.mat.shape[1], fixed_image.mat.shape[0]),
                flags=cv.INTER_LINEAR,
                borderMode=cv.BORDER_CONSTANT,
                borderValue=0,
            )
            chip_center = tuple(np.mean(fixed_box, axis=0))
            winner = {
                "flip_state": fb_info["flip_state"],
                "rot_index": fb_info["rot_index"],
                "coarse_registered_full": coarse_registered_full,
                "final_registered_full": coarse_registered_full,
                "fine_angle": 0.0,
                "fine_scale": 1.0,
                "fine_offset": (0, 0),
                "coarse_scale": float(coarse_scale),
                "coarse_translation": tuple(coarse_trans),
                "fine_alignment": {
                    "pre_transform": self._pre_transform,
                    "coarse_scale": float(coarse_scale),
                    "coarse_translation": tuple(coarse_trans),
                    "offset": (0, 0),
                    "scale": 1.0,
                    "angle": 0.0,
                    "roi_bbox": None,
                    "fine_center": chip_center,
                },
                "final_score": -1.0,
                "ok": False,
                "coarse_key": ranked[0][0],
                "coarse_score": float(fb_info["score"]),
            }

        # ---------- commit winner to self ----------
        self._flip_state = winner["flip_state"]
        self._rot90 = winner["rot_index"]
        self._coarse_scale = winner["coarse_scale"]
        self._coarse_translation = winner["coarse_translation"]
        self._coarse_registered_full = winner["coarse_registered_full"]
        self._final_registered_full = winner["final_registered_full"]
        self._fine_scale = winner["fine_scale"]
        self._fine_angle = winner["fine_angle"]
        self._fine_offset = winner["fine_offset"]
        self._offset = winner["fine_offset"]
        self._fine_alignment = winner["fine_alignment"]
        # register_score: final similarity is the honest quality measure
        self._score = (
            float(winner["final_score"]) if winner["ok"]
            else float(winner["coarse_score"])
        )

        clog.info(
            f"Winner - Flip: {self._flip_state}, Rot: {self._rot90 * 90}°, "
            f"angle={self._fine_angle:.4f}°, scale={self._fine_scale:.6f}, "
            f"offset={self._offset}, final_score={self._score:.3e}, "
            f"coarse_score={winner['coarse_score']:.3e}"
        )

        return register_info


def chip_align(
        moving_image: ChipFeature,
        fixed_image: ChipFeature,
        from_stitched: bool = True,
        flip_flag: bool = True,
        rot90_flag: bool = True
):
    ca = ChipAlignment(flip_flag=flip_flag, rot90_flag=rot90_flag)

    if moving_image.tech_type is TechType.HE:
        from cellbin2.image.augmentation import f_rgb2hsv
        moving_image.set_mat(mat=f_rgb2hsv(moving_image.mat.image, channel=1, need_not=False))

    if from_stitched:
        ca.align_stitched(fixed_image=fixed_image, moving_image=moving_image)
    else:
        ca._pre_transform = {
            "scale": 1.0,
            "rotation": 0.0,
            "applied": False,
        }
        ca.align_transformed(fixed_image=fixed_image, moving_image=moving_image)

    info = {
        'offset': tuple(ca._offset),
        'counter_rot90': ca._rot90,
        'flip': bool(ca._flip_state),
        'register_score': ca._score,
        'register_mat': ca.registration_image(moving_image.mat),
        'dst_shape': (fixed_image.mat.shape[0], fixed_image.mat.shape[1]),
        'method': AlignMode.ChipBox,
        'fine_alignment': ca._fine_alignment
    }

    return info

def apply_chip_box_transform_to_array(
        image: np.ndarray,
        registration_info: dict,
        interpolation=cv.INTER_LINEAR,
):
    """
    Apply chip_box_scell full registration transform to image or mask.

    registration_info should contain:
    {
        "flip": bool,
        "counter_rot90": int,
        "dst_shape": (h, w),
        "fine_alignment": {
            "coarse_scale": float,
            "coarse_translation": (tx, ty),
            "offset": (dx, dy),
            "scale": float,
            "angle": float,
            "roi_bbox": (x0, y0, x1, y1)
        }
    }
    """

    img = image

    pre = registration_info["fine_alignment"].get("pre_transform", None)
    if pre is not None and pre.get("applied", False):
        pre_scale = float(pre.get("scale", 1.0))
        pre_rotation = float(pre.get("rotation", 0.0))

        pre_img = CBImage(img)
        img = pre_img.trans_image(
            scale=[pre_scale, pre_scale],
            rotate=pre_rotation
        )

        if hasattr(img, "image"):
            img = img.image
        elif hasattr(img, "_image"):
            img = img._image

    # 1. flip + rot90
    if bool(registration_info.get("flip", False)):
        img = np.fliplr(img)

    rot_index = int(registration_info.get("counter_rot90", 0)) % 4
    for _ in range(rot_index):
        img = np.rot90(img, 1)

    img = np.ascontiguousarray(img)

    fa = registration_info["fine_alignment"]

    dst_h, dst_w = registration_info["dst_shape"]

    # 2. coarse scale + translation
    coarse_scale = float(fa["coarse_scale"])
    tx, ty = fa["coarse_translation"]

    coarse_M = np.array(
        [
            [coarse_scale, 0.0, float(tx)],
            [0.0, coarse_scale, float(ty)]
        ],
        dtype=np.float32
    )

    coarse_registered = cv.warpAffine(
        img,
        coarse_M,
        (int(dst_w), int(dst_h)),
        flags=interpolation,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0
    )

    # 3. fine angle + scale + offset
    fine_angle = float(fa.get("angle", 0.0))
    fine_scale = float(fa.get("scale", 1.0))
    fine_dx, fine_dy = fa.get("offset", (0, 0))

    if fa.get("fine_center", None) is not None:
        cx, cy = fa["fine_center"]
    else:
        # fallback: use roi_bbox center
        roi_bbox = fa.get("roi_bbox", None)
        if roi_bbox is not None:
            x0, y0, x1, y1 = roi_bbox
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
        else:
            cx = int(dst_w) / 2.0
            cy = int(dst_h) / 2.0

    fine_M = cv.getRotationMatrix2D(
        (float(cx), float(cy)),
        fine_angle,
        fine_scale
    )
    fine_M[0, 2] += float(fine_dx)
    fine_M[1, 2] += float(fine_dy)

    final_registered = cv.warpAffine(
        coarse_registered,
        fine_M,
        (int(dst_w), int(dst_h)),
        flags=interpolation,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0
    )

    return final_registered

def chip_box_register_from_paths(
        moving_image_path: str,
        fixed_image_path: str,
        chip_detect_cfg: ChipParam,
        actual_size=(9996, 9996),
        moving_stain_type=TechType.DAPI,
        fixed_chip_size=(1, 1),
        from_stitched: bool = False,
        flip_flag: bool = True,
        rot90_flag: bool = True,
        is_debug: bool = False,
):
    """
    Run chip-box registration directly from moving/fixed image paths.

    Parameters
    ----------
    moving_image_path:
        Path of moving image, usually transformed DAPI image.
    fixed_image_path:
        Path of fixed matrix heatmap image.
    chip_detect_cfg:
        ChipParam for detecting moving image chip box.
    actual_size:
        Physical chip size used by detect_chip.
    moving_stain_type:
        Stain type for moving image chip detection.
    fixed_chip_size:
        Chip size used by matrix chip box detection.
    from_stitched:
        Passed to chip_align. In this scell branch, usually False.
    flip_flag:
        Whether to search horizontal flip.
    rot90_flag:
        Whether to search rot90.
    is_debug:
        Passed to detect_chip.

    Returns
    -------
    dict
        {
            "registration": result dict returned by chip_align,
            "moving_chip_box": moving chip box used by registration,
            "fixed_chip_box": fixed chip box used by registration,
            "moving_image": moving ChipFeature,
            "fixed_image": fixed ChipFeature,
        }
    """

    # ---------------- moving image ----------------
    moving_image = ChipFeature()
    moving_mat = cbimread(moving_image_path)
    moving_image.set_mat(moving_mat)

    cfg = chip_detect_cfg

    moving_chip_box = detect_chip(
        moving_mat.image,
        cfg=cfg,
        actual_size=(9996, 9996),
        is_debug=is_debug,
        stain_type=TechType.DAPI
    )[0]
    moving_image.set_chip_box(moving_chip_box)

    # ---------------- fixed image ----------------
    fixed_image = ChipFeature()
    fixed_image.tech_type = TechType.Transcriptomics
    fixed_image.set_mat(fixed_image_path)

    fixed_chip_box = detect_chip_box(fixed_image.mat.image, chip_size=(1, 1))
    fixed_image.set_chip_box(fixed_chip_box)

    # ---------------- align ----------------
    result = chip_align(moving_image, fixed_image)

    print(result["fine_alignment"])
    return {
        "registration": result,
        "moving_chip_box": moving_chip_box,
        "fixed_chip_box": fixed_chip_box,
        "moving_image": moving_image,
        "fixed_image": fixed_image,
    }
    

def main():
    # ---------------- moving image ----------------
    moving_image = ChipFeature()
    moving_mat = cbimread(rf"D:\cellbin_data\芯片单细胞\manlin_data\染色图\20260416拍照\6WGA不封片\正常全图\正常全图.tif")
    moving_image.set_mat(moving_mat)

    cfg = ChipParam(
        **{
            "stage1_weights_path": r"D:\cellbin_data\models\chip_detect_11obbn_640_stage1_20250402_pytorch.onnx",
            "stage2_weights_path": r"D:\cellbin_data\models\chip_detect_yolo11x_1024_stage2_20250411_2e3_equ_pytorch.onnx"
        }
    )

    m_info = detect_chip(
        moving_mat.image,
        cfg=cfg,
        actual_size=(9996, 9996),
        is_debug=False,
        stain_type=TechType.DAPI
    )[0]
    moving_image.set_chip_box(m_info)

    # ---------------- fixed image ----------------
    fixed_image = ChipFeature()
    fixed_image.tech_type = TechType.Transcriptomics
    fixed_image.set_mat(rf"D:\cellbin_data\芯片单细胞\manlin_data\染色图\20260416拍照\6WGA不封片\Y10116JA-2_Transcriptomics.tif")

    f_info = detect_chip_box(fixed_image.mat.image, chip_size=(1, 1))
    fixed_image.set_chip_box(f_info)

    # ---------------- align ----------------
    result = chip_align(moving_image, fixed_image)

    print(result["fine_alignment"])

    cbimwrite(
        rf"D:\cellbin_data\芯片单细胞\manlin_data\染色图\20260416拍照\6WGA不封片\正常全图_new_align.tif",
        result["register_mat"]
    )


if __name__ == '__main__':
    main()