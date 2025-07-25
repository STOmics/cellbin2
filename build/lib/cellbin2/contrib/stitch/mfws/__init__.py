# -*- coding: utf-8 -*-
"""
🌟 Create Time  : 2024/12/27 15:14
🌟 Author  : CB🐂🐎 - lizepeng
🌟 File  : __init__.py.py
🌟 Description  : 
"""
try:
    from .image_stitch import stitch_image
except ImportError:
    from image_stitch import stitch_image

try:
    from .modules.stitching import Stitching
except ImportError:
    from modules.stitching import Stitching

version = '0.0.1'
