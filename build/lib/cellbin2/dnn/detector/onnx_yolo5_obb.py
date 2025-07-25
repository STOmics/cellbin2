from typing import Optional, Tuple, List

import numpy as np
import onnxruntime
import os
from cellbin2.dnn.detector.util import letterbox
from cellbin2.dnn.detector.util import scale_polys
from cellbin2.dnn.detector.util import rbox2poly
from cellbin2.dnn.detector.util import non_max_suppression_obb_np
from cellbin2.image import CBImage
from cellbin2.utils import clog

def init_session(model_path, gpu=-1, providers=['CPUExecutionProvider'],
                 providers_id=[{'device_id': '-1'}], num_threads = 0):
    if os.path.exists(model_path):
        sessionOptions = onnxruntime.SessionOptions()
        try:
            if (gpu < 0) and (num_threads > 0):
                sessionOptions.intra_op_num_threads = num_threads
            sess = onnxruntime.InferenceSession(model_path, providers=providers,
                                                provider_options=providers_id,
                                                sessionOptions=sessionOptions)
            # if gpu<0:
            #     clog.info(f'onnx work on cpu, threads {num_threads}')
            # else:
            #     clog.info(f'onnx work on gpu {gpu}')
        except:
            if num_threads>0:
                sessionOptions.intra_op_num_threads = num_threads
            sess = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'],
                                                sessionOptions=[{'device_id': '-1'}],
                                                sess_options=sessionOptions)
            # clog.info(f'onnx work on cpu, threads {num_threads}')

        return sess

class PickableInferenceSession:  # This is a wrapper to make the current InferenceSession class pickable.
    def __init__(self, model_path, gpu, providers, providers_id, num_threads):
        self.gpu = gpu
        self.providers = providers
        self.providers_id = providers_id
        self.num_threads = num_threads
        self.model_path = model_path
        self.sess = init_session(self.model_path, gpu, providers, providers_id, num_threads)

    def run(self, input_name, data):
        return self.sess.run(None, {f"{input_name}": data})

    def __getstate__(self):
        return {'model_path': self.model_path}

    def __setstate__(self, values):
        self.model_path = values['model_path']
        self.sess = init_session(self.model_path)


class OBB5Detector(object):
    def __init__(
            self,
            conf_thresh=0.25,
            iou_thresh=0.1,
            gpu="-1",
            num_threads=0,
            img_func=None
    ):
        self.conf_thresh: float = conf_thresh
        self.iou_thresh: float = iou_thresh
        self.gpu: int = int(gpu)
        self.num_threads: int = num_threads

        self.providers = ['CPUExecutionProvider']
        self.providers_id = [{'device_id': '-1'}]
        # if img_func is None:
        #     self.img_func = pt_enhance_method.get(TechType.ssDNA.value)  # default using ssDNA enhance
        # else:
        self.img_func = img_func

        # constant
        self._input_name: str = 'images'
        self.img_size = (1024, 1024)

        self.model: Optional[PickableInferenceSession] = None
        self._f_init()

    def _f_init(self):
        if self.gpu > -1:
            self.providers = ['CUDAExecutionProvider']
            self.providers_id = [{'device_id': str(self.gpu)}]


    def load_model(self, weight_path):
        self.model = PickableInferenceSession(weight_path, self.gpu, self.providers, self.providers_id, self.num_threads)
        self.img_size = self.model.sess.get_inputs()[0].shape[-2:]

    def set_func(self, fun):
        self.img_func = fun

    def preprocess(self, img: CBImage) -> Tuple[np.ndarray, np.ndarray]:
        enhance_img = self.img_func(img)  # returned image in BGR format
        ori_shape = enhance_img.shape
        padded_img = letterbox(enhance_img, self.img_size)
        # Convert
        adjust_img = padded_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        adjust_img = np.ascontiguousarray(adjust_img)  # convert to a C-contiguous array (memory contiguous) for faster processing 
        adjust_img = np.float32(adjust_img)  # must do this for onnx
        adjust_img /= 255
        return ori_shape, adjust_img

    @staticmethod
    def postprocess(det, w, h, tol=15):
        angles = np.median(np.degrees(det[:, -3]))
        if angles >= 0:
            img_angle = 90 - (180 - 45 + angles)
            img_angle = -img_angle
        else:
            img_angle = 90 - (180 - (-angles + 45))
        mid_1 = (det[:, 4: 6] + det[:, 0: 2]) / 2
        mid_2 = (det[:, 6: 8] + det[:, 2: 4]) / 2
        mid_final = (mid_1 + mid_2) / 2.0
        xy = np.concatenate((mid_final, det[:, -2].reshape(-1, 1),), axis=1)
        x, y = xy[:, 0], xy[:, 1]
        remain = ~np.logical_or.reduce(
            np.concatenate(
                ((x - tol < 0).reshape(-1, 1),
                 (x + tol > w).reshape(-1, 1),
                 (y - tol < 0).reshape(-1, 1),
                 (y + tol > h).reshape(-1, 1)),
                axis=1
            )
            , axis=1
        )
        xy = xy[remain]
        return xy.tolist(), img_angle

    def predict(self, img: CBImage) -> Tuple[List[List[float]], Optional[float]]:
        ori_shape, img = self.preprocess(img)
        cp, angle = list(), None
        h, w = ori_shape[:2]
        if len(img.shape) == 3:
            img = img[None]

        # Inference
        pred = self.model.run(input_name=self._input_name, data=img)[0]  # list*(n, [cx, cy, l, s, θ, conf, cls]) θ ∈ [-pi/2, pi/2)
        pred = non_max_suppression_obb_np(
            prediction=pred,
            conf_thres=self.conf_thresh,
            iou_thres=self.iou_thresh,
            multi_label=True,
            max_det=200,
        )

        det = pred[0]
        # for i, det in enumerate(pred):  # per image
        pred_poly = rbox2poly(det[:, :5])  # (n, [x1 y1 x2 y2 x3 y3 x4 y4])
        pred_poly = scale_polys(img.shape[2:], pred_poly, ori_shape)
        det = np.concatenate((pred_poly, pred[0][:, -3:]), axis=1)  # pred[0][:, -3:] -> [θ, conf, cls]
        if len(det):
            cp, angle = self.postprocess(det, w, h)
        return cp, angle


if __name__ == '__main__':
    import cv2 as cv
    from cellbin2.image import cbimread
    from cellbin2.image.augmentation import f_gray2bgr, f_ij_16_to_8_v2

    def test_enhance(img): return cv.cvtColor(img.image, cv.COLOR_BGR2RGB)

    # model_path = r"E:\03.users\liuhuanlin\01.data\cellbin2\weights\chip_detect_yolov5obb_SSDNA_20241001_pytorch.onnx"
    model_path = r"E:\03.users\liuhuanlin\01.data\cellbin2\weights\last.onnx"
    # model_path = r"E:\03.users\liuhuanlin\01.data\cellbin2\weights\points_detect_yolov5obb_SSDNA_20220513_pytorch.onnx"
    ci = OBB5Detector(conf_thresh=0.25,
            iou_thresh=0.1,
            gpu=-1,
            num_threads=0,
            img_func=None)
    ci.load_model(model_path)
    ci.set_func(test_enhance)

    # img_path = r"E:/03.users/liuhuanlin/01.data/cellbin2/other/controlD1_0004_0006.tif"
    img_path = r"E:/03.users/liuhuanlin/01.data/cellbin2/other/Y00038K4.tif"
    img = cbimread(img_path)
    # img = f_ij_16_to_8_v2(img.image)
    img = f_gray2bgr(img.image)
    img = cbimread(img)
    # # ori_shape, adjust_img = ci.preprocess(img)
    cp, angle = ci.predict(img)
    print(cp, angle)
    print("asd")
