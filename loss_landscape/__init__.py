"""
LossLandscape - 通用 Loss Landscape 自动化分析平台

一个极简的 SDK，用于在训练中或训练后自动生成高维 Loss Landscape 数据。

快速开始：
    # 方式 1: 一行代码（便捷函数）
    from loss_landscape import sample_landscape
    sample_landscape(model, data_loader, loss_fn, "./landscape.json")

    # 方式 2: Writer 接口（推荐）
    from loss_landscape import LossLandscapeWriter

    writer = LossLandscapeWriter("./runs/exp1")
    writer.sample_landscape(model, data_loader, loss_fn)
    writer.close()
"""

from .core import (
    # 推荐使用
    LossLandscapeWriter,
    sample_landscape,
    # 高级用户
    Explorer,
    LandscapeStorage,
    # 别名
    Writer,
)

__version__ = "0.1.0"

__all__ = [
    # 推荐使用
    "LossLandscapeWriter",
    "sample_landscape",
    # 高级用户
    "Explorer",
    "LandscapeStorage",
    # 别名
    "Writer",
]
