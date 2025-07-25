# -*- coding: utf-8 -*-
"""
🌟 Create Time  : 2025/1/17 11:07
🌟 Author  : CB🐂🐎 - lizepeng
🌟 File  : jitter_correct.py
🌟 Description  : using any method to correct jitter offset
"""

import numpy as np

from enum import Enum
from typing import Union


class CorrectMethods(Enum):
    MethodNone = 0
    LineScanHorizontal = 1
    LineScanVertical = 2


class JitterCorrect:
    """

    """
    def __init__(
            self,
            jitter: list,
            overlap: Union[float, list, tuple, np.ndarray] = 0.1,
            image_size: Union[list, tuple, np.ndarray] = None,
            method: Union[CorrectMethods, int] = CorrectMethods.MethodNone,
            placeholder: int = -999
    ):
        """

        Args:
            jitter: h_jitter, v_jitter -- Horizontal, Vertical
            overlap: between 0 ~ 1
                -- float: h_overlap = v_overlap = overlap
                -- list, tuple, np.array : h_overlap, v_overlap = overlap
            image_size: image shape -- [h, w]
            method: Enum -- CorrectMethods
            placeholder:

        Examples:

        """
        self._h_overlap: int = 0
        self._v_overlap: int = 0

        self.set_overlap(overlap)

        self._jitter = jitter
        self._h, self._w = image_size

        if isinstance(method, CorrectMethods):
            self._method = method.value

        self._placeholder = placeholder

    def correct(self) -> tuple:

        hxo, hyo = int(self._w * self._h_overlap), 0
        vxo, vyo = 0, int(self._h * self._v_overlap)

        h_jitter = self.fill_value(self._jitter[0], hxo, hyo)
        v_jitter = self.fill_value(self._jitter[1], vxo, vyo)

        return h_jitter, v_jitter

    def fill_value(self, jit: np.ndarray, xo: int, yo: int) -> np.ndarray:
        """

        Args:
            jit:
            xo:
            yo:

        Returns:

        """
        if xo == 0 and yo == 0:
            return jit

        x_jit: np.ndarray = jit[:, :, 0]
        y_jit: np.ndarray = jit[:, :, 1]

        x_jit = self._delete_outline_value(x_jit)
        y_jit = self._delete_outline_value(y_jit)

        # if xo != 0:
        #     pass
        #
        # if yo != 0:
        #     pass

        new_jit = np.concatenate([x_jit[:, :, None], y_jit[:, :, None]], axis=2)
        for r in range(new_jit.shape[0]):
            for c in range(new_jit.shape[1]):
                if new_jit[r, c, 0] == self._placeholder or \
                        new_jit[r, c, 1] == self._placeholder:
                    new_jit[r, c] = self._placeholder

        return new_jit

    def _delete_outline_value(self, jit: np.ndarray):
        _jit = jit.copy()
        rx, cx = _jit.shape
        _x = list(_jit.ravel())
        _x = [i for i in _x if i != self._placeholder]
        # _x_mean = round(np.mean(np.percentile(_x, [25, 75])))

        x_dict = self._cluster_num(
            *np.unique(_jit[:, :], return_counts = True)
        )
        x_list = sorted(x_dict.items(), key = lambda x: x[1][1], reverse = True)

        usable_list = list()
        for xl in x_list:
            if xl[1][0][0] == self._placeholder: continue

            if xl[1][1] >= len(_x) * 0.1:
                usable_list.extend(xl[1][0])

        for r in range(rx):
            for c in range(cx):
                if _jit[r, c] not in usable_list:
                    _jit[r, c] = self._placeholder

        return _jit

    def _mean_fill(self):
        pass

    def _extremum_fill(self):
        pass

    def set_overlap(self, overlap):
        if isinstance(overlap, float):
            self._h_overlap = self._v_overlap = overlap
        else:
            self._h_overlap, self._v_overlap = overlap

    @property
    def overlap(self):
        return self._h_overlap, self._v_overlap

    @staticmethod
    def _cluster_num(arr, count_list, cn=10):
        cluster_list = list()
        for n in arr:
            add_flag = False
            for k_index, k in enumerate(arr):

                if k == n: continue
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

        x_set_cluster = [list(set(i)) for i in cluster_list]

        x_dict = dict()
        for k, xc in enumerate(x_set_cluster):
            count = 0
            for i in xc:
                index = list(arr).index(i)
                count += count_list[index]
            x_dict[k] = (xc, count)
        #  x_list = max(x_dict.items(), key=lambda x: x[1][1])[1][0]

        return x_dict  # , x_list


if __name__ == "__main__":
    aaa = np.load(r"D:\02.data\temp\temp\x.npy")
    bbb = np.load(r"D:\02.data\temp\temp\y.npy")
    jc = JitterCorrect([aaa, bbb], image_size = [6105, 4608])
    aaa, bbb = jc.correct()
    print(1)
