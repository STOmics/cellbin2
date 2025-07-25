# -*- coding: utf-8 -*-
"""
🌟 Create Time  : 2024/9/19 10:40
🌟 Author  : CB🐂🐎 - lizepeng
🌟 File  : test.py
🌟 Description  :
"""
import argparse
import numpy as np

from .onnx_yolo8 import Yolo8Detector


class OBB8Detector(Yolo8Detector):
    """YOLOv8 object detection model class for handling inference and visualization."""

    def __init__(self, onnx_model, confidence_thres=0.5, iou_thres=0.5, gpu="-1", num_threads=0):
        super().__init__(onnx_model, confidence_thres, iou_thres, gpu, num_threads)

    def postprocess(self, output, x_factor, y_factor):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            output (numpy.ndarray): The output of the model.
            x_factor:
            y_factor:

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """
        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))
        index = np.argmax(outputs[:, 4:-1])
        r = outputs[index][-1]

        x, y, w, h = outputs[index][0], outputs[index][1], outputs[index][2], outputs[index][3]

        corner_points = self.xywhr2xyxyxyxy(np.array([x * x_factor, y * y_factor, w * x_factor, h * y_factor, -r]))

        return corner_points


if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8n.onnx", help="Input your ONNX model.")
    parser.add_argument("--img", type=str, default="", help="Path to input image.")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="NMS IoU threshold")
    args = parser.parse_args()

    # Create an instance of the YOLOv8 class with the specified arguments
    detection = OBB8Detector(args.model, args.conf_thres, args.iou_thres)

    # Perform object detection and obtain the output image
    output_image = detection.run(args.img)
