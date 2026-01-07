"""
LossLandscape - 通用 Loss Landscape 自动化分析平台

一个极简的SDK，用于在训练中或训练后自动生成高维Loss Landscape的3D数据文件。
"""

from .core import Explorer, LandscapeStorage

__version__ = "0.1.0"
__all__ = ["Explorer", "LandscapeStorage"]

