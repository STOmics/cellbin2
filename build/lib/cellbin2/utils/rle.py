"""
Implement Run-Length Encode for storing CellMask and TissueMask
"""
import copy

import numpy as np


class RLEncode:
    """
    Reference: https://en.wikipedia.org/wiki/Run-length_encoding
    """

    def __init__(self):
        self.size = 0
        self.step = 0
        self.batch = 40

    def _encode(self, binary_mask, idx_start):
        """
        Encode input mask with the starting index
        """
        binary_mask[binary_mask >= 1] = 1
        pixels = binary_mask.flatten()
        # All index add 1 because we add 0 on the head
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        runs = np.reshape(runs, (-1, 2))
        runs[:, 0] += idx_start
        return runs

    def _merge(self, a, b):
        """
        Merge two fragments with encoded
        """
        if b.shape[0] == 0:
            return a
        if a.shape[0] == 0:
            return b

        remove_start = False
        if (a[-1][0] + a[-1][1] - 1) % (self.step * self.size) == 0:
            # the last number is 1
            if b[0][0] == 1:
                # double 1
                a[-1][1] += b[0][1]
                remove_start = True
        else:
            # the last number is 0
            if b[0][0] == 0:
                # double 0
                a[-1][1] += b[0][1]
                remove_start = True
        if remove_start:
            return np.concatenate([a, b[1:]])
        return np.concatenate([a, b])

    def encode(self, arr):
        """
        arr: numpy array, 1 - mask, 0 - background
        Returns run length econde data
        """
        row, col = arr.shape
        self.size = row
        self.step = self.size // self.batch
        if self.size % self.batch:
            self.step += 1
        if self.step == 1:
            return self._encode(arr, 0)

        res = None
        for i in range(self.batch):
            start = i * self.step
            end = min((i + 1) * self.step, row)
            e = self._encode(arr[start:end, ], start * col)
            if res is None:
                res = e
            else:
                res = self._merge(res, e)

        return res

    def decode(self, mask_rle, shape):
        """
        mask_rle: run-length as string formated (start length)
        shape: (height,width) of array to return
        Returns numpy array, 1 - mask, 0 - background
        """
        mask = copy.deepcopy(mask_rle)
        starts = mask[:, 0]
        lengths = mask[:, 1]
        starts -= 1
        ends = starts + lengths
        binary_mask = np.zeros(int(shape[0]) * int(shape[1]), dtype=np.uint8)
        binary_mask.fill(0)
        for start, end in zip(starts, ends):
            binary_mask[start:end] = 1
        return binary_mask.reshape(shape)
