# -*- coding: utf-8 -*-
"""
🌟 Create Time  : 2025/3/18 10:00
🌟 Author  : CB🐂🐎 - lizepeng
🌟 File  : manual_registration.py
🌟 Description  : 
"""

import os
import argparse

import cv2 as cv
import numpy as np
import tifffile as tif
import matplotlib.pyplot as plt

from typing import Union, List
from scipy.optimize import minimize
from itertools import combinations


# plt.switch_backend('TkAgg')


class PointsMarker:
    """
    Map points2 to points1
    """
    def __init__(self):
        self._points1 = Union[List[List[int]], np.ndarray]
        self._points2 = Union[List[List[int]], np.ndarray]

        self._scale_x = self._scale_y = 1
        self._rotate = 0

        self.trans_matrix: np.matrix = np.eye(3)

    def set_points(self, pts1, pts2):
        """

        Args:
            pts1:
            pts2:

        Returns:

        """
        if pts1.shape != pts2.shape:
            raise ValueError("Pair points need same shape.")

        self._points1 = pts1
        self._points2 = pts2

    @staticmethod
    def get_offset_matrix(offset):
        """

        Args:
            offset:

        Returns:

        """
        m = np.eye(3)
        m[:2, 2] = offset
        return np.matrix(m)

    @staticmethod
    def get_matrix(scale_x = 1.0, scale_y = 1.0, rotate = 0.0, ox = 0, oy = 0, rf = True):
        """

        Args:
            scale_x:
            scale_y:
            rotate:
            ox:
            oy:
            rf:

        Returns:

        """
        mat_scale = np.mat([[scale_x, 0, 0],
                            [0, scale_y, 0],
                            [0, 0, 1]])
        mat_rotate = np.mat([[np.cos(np.radians(rotate)), -np.sin(np.radians(rotate)), 0],
                             [np.sin(np.radians(rotate)), np.cos(np.radians(rotate)), 0],
                             [0, 0, 1]])
        mat_offset = np.mat([[1, 0, ox],
                             [0, 1, oy],
                             [0, 0, 1]])

        if rf:
            mat = mat_offset * mat_scale * mat_rotate
        else:
            mat = mat_offset * mat_rotate * mat_scale

        return mat

    @staticmethod
    def inv_mat2scale_rotate(m):
        """

        Args:
            m:

        Returns:

        """

        degree_x = - np.degrees(np.arctan(m[0, 1] / m[0, 0]))
        degree_y = np.degrees(np.arctan(m[1, 0] / m[1, 1]))
        scale_x = m[0, 0] / np.cos(np.radians(degree_x))
        scale_y = m[1, 0] / np.sin(np.radians(degree_y))

        return scale_x, scale_y, degree_x, degree_y

    @staticmethod
    def centered_points(pts, norm=False):
        """

        Returns:

        """
        if norm:
            pts = pts / np.linalg.norm(pts, axis=1)[:, np.newaxis]

        centroid = np.mean(pts, axis=0)

        return pts - centroid, centroid

    @staticmethod
    def get_points_by_matrix(points, mat):
        """

        Args:
            points:
            mat:

        Returns:

        """
        if points.ndim == 1:
            _points = np.array([points])
        else:
            _points = points

        _points = _points[:, :2]

        new_points = mat[:2, :] @ np.concatenate([
            _points, np.ones((_points.shape[0], 1))],
            axis=1
        ).transpose(1, 0)

        return np.array(new_points).squeeze().transpose()

    def iter_points(self, num: int):
        """
        Args:
            num:

        Returns:

        """
        size = len(self._points1)
        c_list = []
        for c in combinations(range(size), num):
            c_list.append(c)

        new_list = []
        for c in c_list:
            p1 = self._points1[c, :]
            p2 = self._points2[c, :]
            new_list.append([p1, p2])

        return new_list, c_list

    def eval_error(self, method = np.sum):
        """

        Returns:

        """
        return method(
            np.linalg.norm(
                self._points1 - self.get_points_by_matrix(self._points2, self.trans_matrix),
                axis = 1
            )
        )

    def paints_error(self, save_path: str = None):
        """

        Returns:

        """
        plt.cla()
        error = np.linalg.norm(
                self._points1 - self.get_points_by_matrix(self._points2, self.trans_matrix),
                axis = 1
        )

        pts = [f"pt{i}" for i in range(len(self._points1))]

        plt.bar(pts, error, align = 'center', color = 'steelblue', alpha = 0.8)
        plt.ylabel('error')
        plt.title('All points error')
        for x, y in enumerate(error):
            plt.text(x, y, '{:.2f}'.format(y), ha = 'center')

        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()

    @staticmethod
    def paint_points(pts1, pts2):
        plt.figure()
        plt.scatter(pts1[:, 0], pts1[:, 1], c = 'r', marker = 'o')
        plt.scatter(pts2[:, 0], pts2[:, 1], c = 'b', marker = 'o')

        plt.show()

    @property
    def scale_x(self):
        return self._scale_x

    @property
    def scale_y(self):
        return self._scale_y

    @property
    def rotate(self):
        return self._rotate

    def fit_scale_and_rotation(self,):
        """

        Returns:

        """
        cp1, c1 = self.centered_points(self._points1)
        cp2, c2 = self.centered_points(self._points2)

        H = cp1.T @ cp2

        U, S, Vt = np.linalg.svd(H)

        R = Vt.T @ U.T

        sb_x = sb_y = 1
        if np.linalg.det(R) < 0:
            if R[0, 0] < 0:
                R[:2, 0] *= -1
                sb_x = -1
            else:
                R[:2, 1] *= -1
                sb_y = -1

        self._rotate = -np.degrees(np.arctan2(R[1, 0], R[0, 0]))

        _pc = cp2 @ R

        self._scale_x = sb_x * np.linalg.norm(cp1[:, 0]) / np.linalg.norm(_pc[:, 0])
        self._scale_y = sb_y * np.linalg.norm(cp1[:, 1]) / np.linalg.norm(_pc[:, 1])

        new_trans_matrix = self.get_matrix(
            self._scale_x, self._scale_y, self._rotate
        )

        self.trans_matrix = self.get_offset_matrix(-c1).I @ new_trans_matrix @ self.get_offset_matrix(-c2)


class PointsFit(PointsMarker):
    def __init__(self):
        super().__init__()

    def fit_scale_and_rotation(self, ):
        min_dist = np.Inf
        last_matrix = None
        for index_p in range(len(self._points1)):
            cp1 = self._points1 - self._points1[index_p, :]
            cp2 = self._points2 - self._points2[index_p, :]

            para, rotate_first = self.fitting_points(cp1, cp2)

            x, y, r = para.x[0], para.x[1], para.x[2]
            trans_matrix = self.get_matrix(x, y, r, rf = rotate_first)

            result_pts = self.get_points_by_matrix(self._points2[index_p, :], trans_matrix.I)
            offset = self._points1[index_p, :] - result_pts

            self.trans_matrix = self.get_offset_matrix(offset) * trans_matrix.I
            _dist = self.eval_error(np.sum)
            if _dist < min_dist:
                self._scale_x, self._scale_y, self._rotate = 1 / para.x[0], 1 / para.x[1], -para.x[2]
                last_matrix = np.copy(self.trans_matrix)
                min_dist = _dist
        self.trans_matrix = last_matrix

    @staticmethod
    def fitting_points(point_re, point_qc, method = "BFGS"):
        """

        Args:
            point_re:
            point_qc:
            method:

        Returns:

        """

        for k, point in enumerate(point_re):
            if point[0] == 0 and point[1] == 0:
                point_re = np.delete(point_re, k, axis = 0)
                point_qc = np.delete(point_qc, k, axis = 0)
                break

        if len(point_re) == 0 or len(point_qc) == 0:
            return None

        def _error(p, pr, pq, rf = True):
            _scale_x, _scale_y, _rotate = p

            mat_scale = np.mat([[_scale_x, 0, 0],
                                [0, _scale_y, 0],
                                [0, 0, 1]])

            mat_rotate = np.mat([[np.cos(np.radians(_rotate)), -np.sin(np.radians(_rotate)), 0],
                                 [np.sin(np.radians(_rotate)), np.cos(np.radians(_rotate)), 0],
                                 [0, 0, 1]])

            if rf:
                mat = mat_scale * mat_rotate
            else:
                mat = mat_rotate * mat_scale

            new_points = mat[:2, :] @ np.concatenate([
                pr, np.ones((pr.shape[0], 1))],
                axis = 1
            ).transpose(1, 0)

            new_points = np.array(new_points).squeeze().transpose()

            error = np.sqrt((pq[:, 0] - new_points[:, 0]) ** 2 + (pq[:, 1] - new_points[:, 1]) ** 2)

            return np.sum(error)

        para1 = minimize(
            _error, x0 = np.array([1.0, 1.0, 0.]),
            args = (point_re, point_qc, False), method = method
        )
        para2 = minimize(
            _error, x0 = np.array([1.0, 1.0, 0.]),
            args = (point_re, point_qc, True), method = method
        )

        para, rotate_first = (para1, False) if para1.fun < para2.fun else (para2, True)

        return para, rotate_first


class PointsMatrix(PointsMarker):
    """
    """
    def __init__(self):
        super().__init__()

    def fit_scale_and_rotation(self, ):
        pts_list, c_list = self.iter_points(num = 3)
        self.outlier_removal(pts_list)
        self._scale_x, self._scale_y, r1, r2 = self.inv_mat2scale_rotate(self.trans_matrix)
        self._rotate = (r1 + r2) / 2

    def outlier_removal(self, pts: list):
        """
        Args:
            pts:
        """
        m_list = list()
        dist_list = list()
        for pts1, pts2 in pts:
            m = cv.getAffineTransform(
                pts1.astype(np.float32),
                pts2.astype(np.float32)
            )
            _m = np.eye(3)
            if np.sum(m) != 0:
                _m[:2, :] = m
            m_list.append(_m)

            _p = self.get_points_by_matrix(self._points1, _m)

            dist_list.append(np.sum(np.linalg.norm(_p - self._points2, axis = 1)))

        index = dist_list.index(min(dist_list))
        self.trans_matrix = np.matrix(m_list[index]).I
