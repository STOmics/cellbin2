# -*- coding: utf-8 -*-
"""
🌟 Create Time  : 2024/9/19 10:40
🌟 Author  : CB🐂🐎 - lizepeng
🌟 File  : test.py
🌟 Description  :
"""
import argparse
import copy

import cv2
import numpy as np
import onnxruntime as ort

import cellbin2.image as cbi

from cellbin2.image.augmentation import f_ij_16_to_8
from cellbin2.utils import clog


class Yolo8Detector(object):
    """YOLOv8 object detection model class for handling inference and visualization."""

    def __init__(self, onnx_model, confidence_thres=0.5, iou_thres=0.5, gpu="-1", num_threads=0):
        """
        Initializes an instance of the YOLOv8 class.

        Args:
            onnx_model: Path to the ONNX model.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
        """
        self.source_image = None

        self.onnx_model = onnx_model

        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        self.providers = ["CPUExecutionProvider"]
        self.predictor_id = [{'device': "-1"}]
        self.gpu = int(gpu)
        self.num_threads = num_threads

        self.preprocess_func = None
        self.postprocess_func = None

        self.session = None

        self.f_init()
        self._init_session()

    def f_init(self):
        if self.gpu > -1:
            self.providers = ["CUDAExecutionProvider"]
            self.predictor_id = [{'device_id': str(self.gpu)}]

    def _init_session(self):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.

        """
        # Create an inference session using the ONNX model and specify execution providers

        clog.info(f'loading weight from {self.onnx_model}')
        sessionOptions = ort.SessionOptions()
        try:
            if (self.gpu < 0) and (self.num_threads > 0):
                sessionOptions.intra_op_num_threads = self.num_threads
            self.session = ort.InferenceSession(self.onnx_model,
                                           providers=self.providers,
                                           provider_options=self.predictor_id,
                                           sess_options = sessionOptions)
            active_provider = self.session.get_providers()[0]
            expected_provider = self.providers[0]
            if active_provider == expected_provider:
                if self.gpu < 0:
                    clog.info(f'onnx work on cpu,threads {self.num_threads}')
                else:
                    clog.info(f'onnx work on gpu {self.gpu}')
            else:
                # ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
                clog.warning(f'Warning!!! expected: {expected_provider}, active: {active_provider}')
                if active_provider == 'CPUExecutionProvider':
                    clog.info(f'Warning!!! GPU call failed, onnx work on cpu,threads {self.num_threads}')
                if active_provider == 'CUDAExecutionProvider':
                    clog.info(f'onnx work on gpu')
        except:
            if self.num_threads > 0:
                sessionOptions.intra_op_num_threads = self.num_threads
            session = ort.InferenceSession(self.onnx_model,
                                           providers=['CPUExecutionProvider'],
                                           provider_options=[{'device_id': '-1'}],
                                           sess_options=sessionOptions)
            clog.info(f'Warning!!! GPU call failed, onnx work on cpu,threads {self.num_threads}')

    def parse_image(self, img):
        # Read the input image using CBI
        self.source_image = cbi.cbimread(img, only_np = True)
        self.source_image = f_ij_16_to_8(self.source_image)

    def set_preprocess_func(self, func):
        """

        Args:
            func:

        Returns:

        """
        self.preprocess_func = func

    def set_postprocess_func(self, func):
        """

        Args:
            func:

        Returns:

        """
        self.postprocess_func = func

    @staticmethod
    def xywhr2xyxyxyxy(x):
        """
        Convert batched Oriented Bounding Boxes (OBB) from [xywh, rotation]
        to [xy1, xy2, xy3, xy4]. Rotation values should be in radians from 0 to pi/2.

        Args:
            x (numpy.ndarray | torch.Tensor): Boxes in [cx, cy, w, h, rotation] format of shape (n, 5) or (b, n, 5).

        Returns:
            (numpy.ndarray | torch.Tensor): Converted corner points of shape (n, 4, 2) or (b, n, 4, 2).
        """
        cos, sin, cat, stack = (np.cos, np.sin, np.concatenate, np.stack)

        ctr = x[..., :2]
        w, h, angle = (x[..., i: i + 1] for i in range(2, 5))
        cos_value, sin_value = cos(angle), sin(angle)
        vec1 = [w / 2 * cos_value, w / 2 * sin_value]
        vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
        vec1 = cat(vec1, -1)
        vec2 = cat(vec2, -1)
        pt1 = ctr + vec1 + vec2
        pt2 = ctr + vec1 - vec2
        pt3 = ctr - vec1 - vec2
        pt4 = ctr - vec1 + vec2
        return stack([pt1, pt2, pt3, pt4], -2)

    def preprocess(self, img, input_shape):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        if self.preprocess_func is None:
            _img = copy.deepcopy(img)
            if _img.ndim == 2:
                _img = cv2.cvtColor(_img, cv2.COLOR_GRAY2RGB)
        else:
            # Return RGB image color
            _img = self.preprocess_func(img)

        x_factor = _img.shape[1] / input_shape[1]
        y_factor = _img.shape[0] / input_shape[0]

        # Resize the image to match the input shape
        _img = cv2.resize(_img, (input_shape[1], input_shape[0]))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(_img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data, x_factor, y_factor

    def postprocess(self, output, x_factor, y_factor):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            output (numpy.ndarray): The output of the model.
            x_factor:
            y_factor:
            obb:

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """
        # Transpose and squeeze the output to match the expected shape
        if self.postprocess_func is None:
            outputs = np.transpose(np.squeeze(output[0]))
            index = np.argmax(outputs[:, 4:], axis = 0)

            if isinstance(index, np.ndarray):
                score = list()
                for i in range(index.shape[0]):
                    _s = outputs[index[i], 4 + i]
                    score.append(_s)
                _i = score.index(max(score))
                index = index[_i]

            x, y, w, h = outputs[index][0], outputs[index][1], outputs[index][2], outputs[index][3]

            points = self.xywhr2xyxyxyxy(np.array([x * x_factor, y * y_factor, w * x_factor, h * y_factor, 0]))
        else:
            points = self.postprocess_func(output, x_factor, y_factor)

        return points

    def run(self, input_image):

        self.parse_image(input_image)
        model_inputs = self.session.get_inputs()

        # Store the shape of the input for later use
        input_shape = model_inputs[0].shape
        input_width = input_shape[2]
        input_height = input_shape[3]

        # Preprocess the image data
        img_data, x_factor, y_factor = self.preprocess(self.source_image, (input_height, input_width))

        # Run inference using the preprocessed image data
        outputs = self.session.run(None, {model_inputs[0].name: img_data})

        # Perform post-processing on the outputs to obtain output image.
        points = self.postprocess(outputs, x_factor, y_factor)

        return points


if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8n.onnx", help="Input your ONNX model.")
    parser.add_argument("--img", type=str, default="", help="Path to input image.")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="NMS IoU threshold")
    args = parser.parse_args()

    # Create an instance of the YOLOv8 class with the specified arguments
    detection = Yolo8Detector(args.model, args.conf_thres, args.iou_thres)

    # Perform object detection and obtain the output image
    output_image = detection.run(args.img)
