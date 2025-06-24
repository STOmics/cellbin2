# -*- coding: utf-8 -*-
"""
🌟 Create Time  : 2025/3/18 10:06
🌟 Author  : CB🐂🐎 - lizepeng
🌟 File  : __init__.py
🌟 Description  : 
"""
try:
    from manual_registration import \
        PointsMarker, PointsFit, PointsMatrix
except ImportError:
    from .manual_registration import \
        PointsMarker, PointsFit, PointsMatrix
