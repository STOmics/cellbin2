# 支撑功能
import numpy as np
import h5py
import os
import json
from typing import Any


def obj2dict(obj):
    dct = {}
    for k, v in vars(obj).items():
        if type(v) in [int, str, float, bool, list, np.int64, np.int32, np.float64, np.ndarray, np.bool_, tuple]:
            dct[k] = v
        else:
            dct[k] = obj2dict(v)

    return dct


def dict2h5(dct, h5):
    for k, v in dct.items():
        if type(v) is dict:
            sub = h5.create_group(k)
            dict2h5(v, sub)
        elif type(v) is np.ndarray:
            h5.create_dataset(k, data=v, compression="lzf")  # TODO: 压缩后h5 view看不到，等不需要debug了再开
        else:
            h5.attrs[k] = v


def h52dict(h5, dct: dict):
    for k, v in h5.items():
        if isinstance(v, h5py._hl.group.Group):
            dct[k] = {}
            h52dict(v, dct[k])
        elif isinstance(v, h5py._hl.dataset.Dataset):
            dct[k] = v[::]
        else:
            pass

    for k, v in h5.attrs.items():
        dct[k] = v


class HDF5(object):
    def __init__(self):
        pass

    def read(self, file_path: str):
        dct = {}
        with h5py.File(file_path, 'r') as fd:
            h52dict(fd, dct)

        for k, v in dct.items():
            if not hasattr(self, k):
                dtype = type(v)
                if dtype in [int, str, float, bool, list, np.bool_, np.int64, np.float64, np.int32, np.ndarray]:
                    if dtype is np.bool_:
                        v = bool(v)
                    if dtype is np.int64:
                        v = int(v)
                    if dtype is np.float64:
                        v = float(v)
                    if dtype is np.int32:
                        v = int(v)
                    else:
                        pass

                    if not hasattr(self, k): self.__setattr__(k, v)
                else:
                    obj1 = Dict2Obj(v)
                    self.__setattr__(k, obj1)

    def write(self, file_path: str, extra: dict):
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        for k, v in extra.items():
            if not hasattr(self, k):
                self.__setattr__(k, v)
        dct = obj2dict(self)

        with h5py.File(file_path, 'w') as fd:
            dict2h5(dct, fd)


class Dict2Obj(object):
    def __init__(self, dct: dict):
        for k, v in dct.items():
            if not hasattr(self, k):
                dtype = type(v)
                if dtype in [int, str, float, bool, list, np.bool_, np.int64, np.float64, np.int32, np.ndarray]:
                    if dtype is np.bool_: v = bool(v)
                    if dtype is np.int64: v = int(v)
                    if dtype is np.float64: v = float(v)
                    if dtype is np.int32:
                        v = int(v)
                    else:
                        pass

                    if not hasattr(self, k): self.__setattr__(k, v)
                else:
                    obj1 = Dict2Obj(v)
                    self.__setattr__(k, obj1)


class DictEncoder(json.JSONEncoder):
    """ 适配json各种数据类型 """

    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, (np.bool_,)):
            return bool(o)
        elif isinstance(o, (np.int32, np.int64)):
            return int(o)
        elif isinstance(o, (np.str_,)):
            return str(o)

        return json.JSONEncoder.default(self, o)


def dict2json(data: dict, json_path: str, indent: int = 2):
    with open(json_path, 'w') as fd:
        json.dump(data, fd, cls=DictEncoder, indent=indent)


def main():
    file_path = r'test.json'
    dct = {'matrix': np.ones((3, 3), dtype=int)}

    dict2json(data=dct, json_path=file_path)


if __name__ == '__main__':
    main()
