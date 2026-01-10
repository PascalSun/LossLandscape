"""
LossLandscapeWriter - Loss Landscape 记录器

职责：
1. 管理存储路径和随机种子
2. 管理训练检查点（用于轨迹）
3. 协调采样并存储结果
4. 导出数据供前端查看

设计原则：
- Writer 只是"记录器"，不持有 Loss 计算逻辑
- 所有与"这次采样"相关的参数都在方法调用时传入
"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from .explorer import Explorer
from .storage import LandscapeStorage


class LossLandscapeWriter:
    """
    Loss Landscape 记录器。

    使用示例：

    1. 基本用法：
        ```python
        from loss_landscape import LossLandscapeWriter
        import torch.nn as nn

        # loss_fn 签名: (model, inputs, targets) -> loss
        def loss_fn(model, inputs, targets):
            return nn.MSELoss()(model(inputs), targets)

        writer = LossLandscapeWriter("./runs/exp1")
        writer.sample_landscape(model, data_loader, loss_fn)
        writer.close()  # 自动导出 JSON
        ```

    2. 带正则化：
        ```python
        def loss_with_reg(model, inputs, targets):
            outputs = model(inputs)
            loss = nn.MSELoss()(outputs, targets)
            reg = 0.01 * sum(p.norm()**2 for p in model.parameters())
            return loss + reg

        writer = LossLandscapeWriter("./runs/exp2")
        writer.sample_landscape(model, data_loader, loss_with_reg)
        writer.close()
        ```

    3. Physics-Informed Loss：
        ```python
        def physics_loss(model, inputs, targets):
            outputs = model(inputs)
            data_loss = nn.MSELoss()(outputs, targets)
            physics_loss = compute_pde_residual(model, inputs)
            return data_loss + 0.1 * physics_loss

        writer = LossLandscapeWriter("./runs/pinn")
        writer.sample_landscape(model, data_loader, physics_loss)
        writer.close()
        ```

    4. 训练时记录轨迹：
        ```python
        writer = LossLandscapeWriter("./runs/training")

        for epoch in range(100):
            train_loss = train_one_epoch(model, ...)
            writer.record_checkpoint(model, epoch, train_loss=train_loss)

        # 构建轨迹
        writer.build_trajectory(model, data_loader, loss_fn)
        writer.close()
        ```

    5. 对比实验（共享方向）：
        ```python
        writer_a = LossLandscapeWriter("./runs/model_a", seed=42)
        writer_a.sample_landscape(model_a, loader, loss_fn)
        directions = writer_a.get_directions()
        writer_a.close()

        writer_b = LossLandscapeWriter("./runs/model_b")
        writer_b.sample_landscape(model_b, loader, loss_fn, directions=directions)
        writer_b.close()
        ```
    """

    def __init__(
        self,
        log_dir: Union[str, Path],
        seed: Optional[int] = 42,
        auto_export: bool = True,
    ):
        """
        初始化 Writer。

        Args:
            log_dir: 日志目录（类似 TensorBoard 的 log_dir）
            seed: 随机种子，确保方向可复现（默认42）。设为None则随机。
            auto_export: 是否在 close() 时自动导出 JSON
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.seed = seed
        self.auto_export = auto_export

        # 设置随机种子
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            np.random.seed(seed)

        # 创建存储
        db_path = self.log_dir / "landscape.duckdb"
        self.storage = LandscapeStorage(str(db_path), mode="create")

        # 方向向量（用于对比实验）
        self._landscape_directions = None

        # 训练检查点（用于轨迹）
        self._checkpoints = []

    def get_directions(self) -> Optional[Tuple[torch.Tensor, ...]]:
        """
        获取当前使用的方向向量。

        可用于在不同实验中复用相同方向，以便对比。

        Returns:
            方向向量元组 (dir1, dir2, [dir3]) 或 None
        """
        return self._landscape_directions

    def _compute_pca_directions(
        self, model: nn.Module, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从 checkpoints 中计算 PCA 方向。

        PCA 方向捕捉训练过程中参数变化最大的方向，
        比随机方向更能展示优化路径。
        """
        # 收集所有 checkpoint 的参数差异（相对于最后一个）
        final_state = self._checkpoints[-1]["model_state"]

        # 将参数展平为向量
        def flatten_params(state_dict):
            return torch.cat([p.flatten() for p in state_dict.values()])

        final_flat = flatten_params(final_state)

        # 计算每个 checkpoint 相对于最终状态的差异
        diffs = []
        for ckpt in self._checkpoints[:-1]:
            ckpt_flat = flatten_params(ckpt["model_state"])
            diff = ckpt_flat - final_flat
            diffs.append(diff)

        if len(diffs) < 2:
            # 只有一个差异向量，无法做 PCA，返回该向量和一个正交向量
            d1 = diffs[0]
            d1 = d1 / (d1.norm() + 1e-10)
            # 生成一个随机正交向量
            d2 = torch.randn_like(d1)
            d2 = d2 - (d2 @ d1) * d1  # Gram-Schmidt
            d2 = d2 / (d2.norm() + 1e-10)
            return (d1.to(device), d2.to(device))

        # Stack 成矩阵 [n_checkpoints-1, n_params]
        diff_matrix = torch.stack(diffs)

        # 中心化
        diff_matrix = diff_matrix - diff_matrix.mean(dim=0, keepdim=True)

        # SVD 获取主成分
        try:
            U, S, Vh = torch.linalg.svd(diff_matrix, full_matrices=False)
            # Vh[0] 是第一主成分方向，Vh[1] 是第二主成分方向
            d1 = Vh[0]
            d2 = Vh[1] if Vh.shape[0] > 1 else torch.randn_like(d1)
        except Exception:
            # SVD 失败，fallback 到前两个差异向量
            d1 = diffs[0]
            d2 = diffs[1] if len(diffs) > 1 else torch.randn_like(d1)

        # 归一化
        d1 = d1 / (d1.norm() + 1e-10)
        d2 = d2 - (d2 @ d1) * d1  # 确保正交
        d2 = d2 / (d2.norm() + 1e-10)

        return (d1.to(device), d2.to(device))

    def sample_landscape(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        loss_fn: Callable[[nn.Module, Any, Any], torch.Tensor],
        *,  # 以下参数必须使用关键字
        # 采样配置
        grid_size: int = 50,
        range_scale: float = 0.1,
        directions: Optional[Tuple[torch.Tensor, ...]] = None,
        direction_mode: str = "random",  # "random" or "pca"
        mode: str = "2d",  # "1d", "2d", "3d"
        # Loss 配置
        model_mode: str = "eval",
        max_batches: Optional[int] = 10,
        # 高级配置
        pre_batch_hook: Optional[Callable[[Any], Any]] = None,
        # 其他
        device: Optional[torch.device] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        采样 Loss Landscape。

        Args:
            model: PyTorch 模型
            data_loader: 数据加载器
            loss_fn: 损失函数，签名: (model, inputs, targets) -> loss
                     例如: lambda m, x, y: nn.MSELoss()(m(x), y)

            # 采样配置
            grid_size: 网格大小（采样点数）
            range_scale: 扰动范围（越大看得越远，但可能失去细节）
            directions: 方向向量。如果提供，直接使用；否则根据 direction_mode 生成。
            direction_mode: 方向生成模式
                - "random": 随机方向（默认，无需 checkpoints）
                - "pca": PCA 方向（需要先调用 record_checkpoint，推荐有轨迹时使用）
            mode: "1d", "2d", "3d"

            # Loss 配置
            model_mode: 模型模式 'eval' 或 'train'（影响 BatchNorm/Dropout）
            max_batches: 评估 Loss 时使用的最大 batch 数

            # 高级配置
            pre_batch_hook: batch 前处理（数据增强、Mixup 等）

            device: 计算设备（默认自动检测）
            verbose: 是否打印进度

        Returns:
            包含 landscape 数据的字典
        """
        device = device or next(model.parameters()).device

        # 处理方向：如果指定 PCA 且有 checkpoints，计算 PCA 方向
        if directions is None and direction_mode == "pca":
            if len(self._checkpoints) < 2:
                raise ValueError(
                    "direction_mode='pca' 需要至少 2 个 checkpoints。"
                    "请先调用 record_checkpoint()，或使用 direction_mode='random'"
                )
            directions = self._compute_pca_directions(model, device)
            if verbose:
                logger.info(f"使用 PCA 方向（基于 {len(self._checkpoints)} 个 checkpoints）")

        # 创建 Explorer 并采样
        with Explorer(
            model,
            loss_fn,
            data_loader,
            device=device,
            storage=self.storage,
            model_mode=model_mode,
            pre_batch_hook=pre_batch_hook,
            max_batches=max_batches,
        ) as explorer:
            if mode == "1d":
                direction_1d = directions[0] if directions else None
                result = explorer.build_line(
                    grid_size=grid_size,
                    range_scale=range_scale,
                    direction=direction_1d,
                    verbose=verbose,
                )

            elif mode == "2d":
                result = explorer.build_surface(
                    grid_size=grid_size,
                    range_scale=range_scale,
                    directions=directions,
                    verbose=verbose,
                )
                # 保存方向用于对比实验
                if directions is not None:
                    self._landscape_directions = directions
                elif explorer._fixed_directions is not None:
                    self._landscape_directions = explorer._fixed_directions

            elif mode == "3d":
                result = explorer.build_volume(
                    grid_size=grid_size,
                    range_scale=range_scale,
                    directions=directions,
                    verbose=verbose,
                )
                # 保存方向
                if directions is not None:
                    self._landscape_directions = directions[:2] if len(directions) >= 2 else None
                elif explorer._fixed_directions_3d is not None:
                    dir1, dir2, _ = explorer._fixed_directions_3d
                    self._landscape_directions = (dir1, dir2)
            else:
                raise ValueError(f"不支持的 mode: {mode}，必须是 '1d', '2d', 或 '3d'")

        return result

    def record_checkpoint(
        self,
        model: nn.Module,
        epoch: int,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
    ):
        """
        记录模型检查点（用于构建训练轨迹）。

        Args:
            model: 当前模型
            epoch: 当前 epoch
            train_loss: 训练 loss（可选）
            val_loss: 验证 loss（可选）
        """
        model_state = {
            name: param.data.cpu().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        self._checkpoints.append(
            {
                "epoch": epoch,
                "model_state": model_state,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

    def build_trajectory(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        loss_fn: Callable[[nn.Module, Any, Any], torch.Tensor],
        *,
        mode: str = "pca",  # "pca" 或 "fixed"
        directions: Optional[Tuple[torch.Tensor, ...]] = None,
        # Loss 配置
        model_mode: str = "eval",
        max_batches: Optional[int] = 10,
        # 高级配置
        pre_batch_hook: Optional[Callable[[Any], Any]] = None,
        device: Optional[torch.device] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        基于已记录的检查点构建训练轨迹。

        Args:
            model: 基准模型（通常是最终训练好的模型）
            data_loader: 数据加载器
            loss_fn: 损失函数，签名: (model, inputs, targets) -> loss

            mode: 轨迹模式
                - "pca": 使用 PCA 降维（推荐，自动选择主方向）
                - "fixed": 使用固定方向（需要先调用 sample_landscape 或提供 directions）
            directions: 固定模式下的方向向量

            # Loss 配置
            model_mode: 模型模式 'eval' 或 'train'
            max_batches: 评估时使用的最大 batch 数

            # 高级配置
            pre_batch_hook: batch 前处理（数据增强、Mixup 等）

            device: 计算设备
            verbose: 是否打印进度

        Returns:
            包含轨迹数据的字典
        """
        if len(self._checkpoints) == 0:
            raise ValueError("没有记录的检查点。请先调用 record_checkpoint()")

        device = device or next(model.parameters()).device

        # 保存基准模型权重
        base_state_dict = {
            name: param.data.cpu().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        # 创建 Explorer
        with Explorer(
            model,
            loss_fn,
            data_loader,
            device=device,
            storage=self.storage,
            model_mode=model_mode,
            max_batches=max_batches,
            pre_batch_hook=pre_batch_hook,
        ) as explorer:
            # 设置轨迹模式
            explorer.set_trajectory_mode("pca" if mode == "pca" else "fixed")

            # 记录所有检查点
            for checkpoint in self._checkpoints:
                checkpoint_state = checkpoint["model_state"]
                for name, param in model.named_parameters():
                    if param.requires_grad and name in checkpoint_state:
                        param.data.copy_(checkpoint_state[name].to(device))

                explorer.log_position(
                    epoch=checkpoint["epoch"],
                    loss=checkpoint["train_loss"],
                    val_loss=checkpoint["val_loss"],
                    verbose=verbose,
                )

            # 恢复基准模型权重
            for name, param in model.named_parameters():
                if param.requires_grad and name in base_state_dict:
                    param.data.copy_(base_state_dict[name].to(device))

            # 构建轨迹
            if mode == "fixed":
                if directions is None:
                    if self._landscape_directions is not None:
                        dir1, dir2 = self._landscape_directions
                        dir3 = explorer._generate_third_direction(dir1, dir2)
                        directions = (dir1, dir2, dir3)
                    else:
                        raise ValueError(
                            "fixed 模式需要方向向量。请先调用 sample_landscape() 或提供 directions 参数"
                        )
                result = explorer.build_trajectory(mode="fixed", directions=directions)
            else:
                result = explorer.build_trajectory(mode="pca")

        return result

    def export(self, output_path: Optional[Union[str, Path]] = None) -> str:
        """
        导出 JSON 文件供前端查看。

        Args:
            output_path: 输出路径（默认：log_dir/landscape.json）

        Returns:
            导出文件的路径
        """
        if output_path is None:
            output_path = self.log_dir / "landscape.json"
        else:
            output_path = Path(output_path)

        self.storage.export_for_frontend(str(output_path))
        logger.info(f"已导出 landscape 数据到: {output_path}")
        return str(output_path)

    def close(self):
        """关闭 writer，自动导出 JSON（如果启用）"""
        if self.auto_export:
            self.export()
        self.storage.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# ============================================================================
# 便捷函数：一次性使用，不需要创建 Writer
# ============================================================================


def sample_landscape(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: Callable[[nn.Module, Any, Any], torch.Tensor],
    output_path: Union[str, Path] = "./landscape.json",
    *,
    grid_size: int = 50,
    range_scale: float = 0.1,
    mode: str = "2d",
    seed: int = 42,
    verbose: bool = True,
) -> str:
    """
    一次性采样 Loss Landscape 并导出。

    这是一个便捷函数，适合快速使用。如果需要更多控制，请使用 LossLandscapeWriter。

    Args:
        model: PyTorch 模型
        data_loader: 数据加载器
        loss_fn: 损失函数，签名: (model, inputs, targets) -> loss
                 例如: lambda m, x, y: nn.MSELoss()(m(x), y)
        output_path: 输出 JSON 路径
        grid_size: 网格大小
        range_scale: 扰动范围
        mode: "1d", "2d", "3d"
        seed: 随机种子
        verbose: 是否打印进度

    Returns:
        导出文件的路径

    Example:
        >>> def loss_fn(model, x, y):
        ...     return nn.MSELoss()(model(x), y)
        >>> sample_landscape(model, loader, loss_fn, "./my_landscape.json")
    """
    output_path = Path(output_path)
    log_dir = output_path.parent / f".{output_path.stem}_tmp"

    writer = LossLandscapeWriter(log_dir, seed=seed, auto_export=False)
    writer.sample_landscape(
        model,
        data_loader,
        loss_fn,
        grid_size=grid_size,
        range_scale=range_scale,
        mode=mode,
        verbose=verbose,
    )
    result_path = writer.export(output_path)
    writer.storage.close()

    # 清理临时目录
    import shutil

    try:
        shutil.rmtree(log_dir)
    except Exception:
        pass

    return result_path
