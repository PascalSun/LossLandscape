"""
LossLandscape Core Modules

核心功能模块：
- LossLandscapeWriter: 高级接口，类似 TensorBoard 的简单 API
- sample_landscape: 便捷函数，一次性使用
- Explorer: 底层计算引擎（高级用户）
- LandscapeStorage: 数据存储层（高级用户）
"""

from .explorer import Explorer
from .storage import LandscapeStorage
from .writer import LossLandscapeWriter, sample_landscape

# 别名
Writer = LossLandscapeWriter

__all__ = [
    # 推荐使用
    "LossLandscapeWriter",  # 主要接口
    "sample_landscape",  # 便捷函数
    # 高级用户
    "Explorer",
    "LandscapeStorage",
    # 别名
    "Writer",
]
