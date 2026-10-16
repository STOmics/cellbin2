import os

# 设置最大像素数为 2^40 (约 1 万亿像素)，可根据实际图片大小调整
# 注意：必须在 import cv2 之前执行
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**40)

import tifffile
import cv2
import numpy as np
import argparse
from pathlib import Path

def resize_grayscale_image(input_path, ratio, output_path):
    """
    缩小灰度图像
    
    参数:
        input_path: 原图像路径
        fx: X 方向缩小倍数 (例如 0.5 表示缩小到一半)
        fy: Y 方向缩小倍数 (例如 0.5 表示缩小到一半)
        output_path: 输出图像路径
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在：{input_path}")

    # 读取灰度图像 (使用 cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"无法读取图像文件或不是有效的图像格式：{input_path}")

    # 获取原始尺寸
    height, width = img.shape[:2]
    print(f"原始图像尺寸: {width} x {height}")

    # 计算新尺寸
    new_width = int(width * ratio)
    new_height = int(height * ratio)

    # 确保新尺寸至少为 1
    new_width = max(1, new_width)
    new_height = max(1, new_height)

    print(f"缩小倍数: ratio={ratio}")
    print(f"新图像尺寸: {new_width} x {new_height}")

    # 使用 INTER_AREA 插值方法进行缩小 (适合图像缩小，效果较好)
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存缩小后的图像
    cv2.imwrite(output_path, resized_img)
    print(f"缩小后的图像已保存到：{output_path}")

    return resized_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="缩小 TIF 图像")
    parser.add_argument("-i", "--input", required=True, help="输入tiff路径")
    parser.add_argument("-o", "--output", required=True, help="输出tiff路径")
    parser.add_argument("-ratio", type=float, default=0.235, help="X 方向缩小倍数 (默认: 0.235)")

    args = parser.parse_args()

    resize_grayscale_image(args.input, args.ratio, args.output)
