"""
Explorer - 核心上下文管理器，用于安全地扰动模型参数并计算Loss Landscape
"""

import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Optional, Tuple, Dict, Any, TYPE_CHECKING
from contextlib import contextmanager
import warnings
from loguru import logger

if TYPE_CHECKING:
    from .storage import LandscapeStorage

from .hessian import HessianCalculator


class Explorer:
    """
    用于生成Loss Landscape的上下文管理器。
    
    使用示例:
        ```python
        model = MyModel()
        loss_fn = nn.CrossEntropyLoss()
        
        # 基本用法
        with Explorer(model, loss_fn, data_loader) as explorer:
            explorer.build_surface(grid_size=50, range_scale=0.1)
        
        # 带正则化项的用法
        def l2_regularizer(model):
            return sum(p.norm()**2 for p in model.parameters())
        
        with Explorer(
            model, 
            loss_fn, 
            data_loader,
            regularizer=l2_regularizer,
            regularizer_weight=0.01  # lambda
        ) as explorer:
            explorer.build_surface(grid_size=50, range_scale=0.1)
        
        # 完全自定义loss计算（支持数据增强、Mixup等）
        def custom_loss(model, inputs, targets):
            # 可以在这里做数据增强、Mixup等
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            # 添加正则化
            reg = 0.01 * sum(p.norm()**2 for p in model.parameters())
            return loss + reg
        
        with Explorer(
            model,
            loss_fn,
            data_loader,
            custom_loss_fn=custom_loss  # 完全自定义
        ) as explorer:
            explorer.build_surface(grid_size=50, range_scale=0.1)
        ```
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        data_loader: Optional[torch.utils.data.DataLoader] = None,
        device: Optional[torch.device] = None,
        storage: Optional['LandscapeStorage'] = None,
        regularizer: Optional[Callable[[nn.Module], torch.Tensor]] = None,
        regularizer_weight: float = 1.0,
        custom_loss_fn: Optional[Callable[[nn.Module, Any, Any], torch.Tensor]] = None,
        model_mode: str = 'eval',
        pre_batch_hook: Optional[Callable[[Any], Any]] = None,
        post_batch_hook: Optional[Callable[[torch.Tensor, Any, torch.Tensor], torch.Tensor]] = None,
        max_batches: Optional[int] = 10,
    ):
        """
        初始化Explorer。
        
        Args:
            model: PyTorch模型
            loss_fn: 损失函数，接受(model_output, target)并返回标量
            data_loader: 数据加载器，用于评估loss。如果为None，需要手动提供数据
            device: 计算设备，如果为None则自动检测
            storage: LandscapeStorage实例，用于持久化数据。如果为None，数据仅保存在内存中
            regularizer: 正则化函数，接受model作为参数，返回正则化项的标量Tensor。
                        例如: lambda m: sum(p.norm()**2 for p in m.parameters())
            regularizer_weight: 正则化项的权重（lambda），默认1.0
            custom_loss_fn: 自定义loss计算函数，完全覆盖默认的loss计算逻辑。
                           接受(model, inputs, targets)并返回loss标量Tensor。
                           如果提供，将忽略loss_fn和regularizer，使用此函数计算总loss。
                           例如: lambda m, x, y: loss_fn(m(x), y) + 0.01 * sum(p.norm()**2 for p in m.parameters())
            model_mode: 模型模式，'eval'或'train'。影响BatchNorm和Dropout的行为，默认'eval'
            pre_batch_hook: 在每个batch处理前的钩子函数，接受batch数据，返回处理后的batch。
                            可用于数据增强、Mixup等。例如: lambda batch: mixup(batch, alpha=0.2)
            post_batch_hook: 在每个batch处理后的钩子函数，接受(outputs, inputs, targets)，返回修改后的loss。
                             可用于label smoothing、temperature scaling等。
        """
        self.model = model
        self.loss_fn = loss_fn
        self.data_loader = data_loader
        self.device = device or next(model.parameters()).device
        self.regularizer = regularizer
        self.regularizer_weight = regularizer_weight
        self.custom_loss_fn = custom_loss_fn
        self.model_mode = model_mode
        self.pre_batch_hook = pre_batch_hook
        self.post_batch_hook = post_batch_hook
        # Limit evaluation batches for speed when building landscapes/volumes.
        # Set to None to evaluate the entire loader (more accurate, slower).
        self.max_batches = max_batches
        
        # 备份原始权重
        self._original_state = None
        self._flattened_params = None
        self._param_shapes = None
        self._param_names = None
        
        # 存储实例
        self.storage = storage
        
        # 轨迹记录
        self._trajectory_weights = []
        self._trajectory_epochs = []
        # Optional per-point scalar losses (e.g., true train loss per epoch).
        # If provided, they will be exported alongside trajectory coordinates.
        self._trajectory_losses = []
        # Optional validation losses per epoch
        self._trajectory_val_losses = []
        self._trajectory_mode = 'fixed'  # 'fixed' or 'pca'
        self._fixed_directions = None  # 2D directions (dir1, dir2)
        self._fixed_directions_3d = None  # 3D directions (dir1, dir2, dir3)
        
        # 用于PCA模式
        self._all_weights_for_pca = []
        
    def __enter__(self):
        """进入上下文时备份模型参数"""
        self._backup_parameters()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时恢复模型参数"""
        self._restore_parameters()
        return False
    
    def _backup_parameters(self):
        """备份模型的所有参数"""
        self._original_state = copy.deepcopy(self.model.state_dict())
        
        # 展平参数并记录形状
        params = []
        shapes = []
        names = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params.append(param.data.flatten().cpu().clone())
                shapes.append(param.data.shape)
                names.append(name)
        
        self._flattened_params = torch.cat(params)
        self._param_shapes = shapes
        self._param_names = names
    
    def _restore_parameters(self):
        """恢复模型参数到备份状态"""
        if self._original_state is not None:
            self.model.load_state_dict(self._original_state)
    
    def _flatten_to_model(self, flat_vector: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        将展平的向量重新组装为模型参数字典。
        
        Args:
            flat_vector: 展平的参数向量
            
        Returns:
            参数字典，格式与state_dict相同
        """
        state_dict = {}
        idx = 0
        
        for name, shape in zip(self._param_names, self._param_shapes):
            size = np.prod(shape)
            param_data = flat_vector[idx:idx+size].reshape(shape)
            state_dict[name] = param_data
            idx += size
        
        return state_dict
    
    def _apply_perturbation(self, direction1: torch.Tensor, direction2: torch.Tensor, 
                           alpha: float, beta: float):
        """
        应用扰动到模型参数。
        
        Args:
            direction1: 第一个方向向量（已归一化）
            direction2: 第二个方向向量（已归一化）
            alpha: 第一个方向的扰动强度
            beta: 第二个方向的扰动强度
        """
        # 确保所有张量都在同一设备上
        base_params = self._flattened_params.to(self.device)
        dir1 = direction1.to(self.device)
        dir2 = direction2.to(self.device)
        
        # 计算扰动后的参数
        perturbed = base_params + alpha * dir1 + beta * dir2
        
        # 重新组装并加载到模型
        state_dict = self._flatten_to_model(perturbed)
        self.model.load_state_dict(state_dict, strict=False)
    
    def _normalize_direction_filterwise(self, direction: torch.Tensor) -> torch.Tensor:
        """
        实现 Filter-wise Normalization (Li et al., 2018)。
        
        不仅对方向向量进行单位化，还会根据该层参数的范数进行缩放。
        d_new = d / ||d|| * ||w||
        这样使得扰动的相对尺度与参数的尺度一致。
        
        Args:
            direction: 展平的扰动方向向量
            
        Returns:
            归一化并缩放后的方向向量
        """
        normalized_parts = []
        idx = 0
        
        # 确保有基准参数用于计算范数
        # self._flattened_params 是在 _backup_parameters() 中初始化的 (on CPU)
        if self._flattened_params is None:
             # 如果没有基准参数，退化为简单的单位归一化 (Log warning?)
            base_params = None
        else:
            base_params = self._flattened_params
            
        for shape in self._param_shapes:
            size = np.prod(shape)
            layer_direction = direction[idx:idx+size]
            
            # 计算方向向量的范数
            layer_d = layer_direction.reshape(shape)
            d_norm = torch.norm(layer_d, p='fro')
            
            scale = 1.0
            if d_norm > 1e-8:
                scale = 1.0 / d_norm
                
                # 如果有基准参数，应用 Li et al. 的 scaling: * ||w||
                if base_params is not None:
                    # base_params 在 CPU，需要移动到 device
                    layer_w = base_params[idx:idx+size].to(layer_direction.device).reshape(shape)
                    w_norm = torch.norm(layer_w, p='fro')
                    scale *= w_norm
            
            normalized_layer = layer_direction * scale
            
            normalized_parts.append(normalized_layer)
            idx += size
        
        return torch.cat(normalized_parts)
    
    def _generate_random_directions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成两个随机方向向量（已归一化）。
        
        Returns:
            (direction1, direction2): 两个已归一化的方向向量
        """
        # 生成随机向量
        rand1 = torch.randn_like(self._flattened_params)
        rand2 = torch.randn_like(self._flattened_params)
        
        # Filter-wise归一化
        dir1 = self._normalize_direction_filterwise(rand1)
        dir2 = self._normalize_direction_filterwise(rand2)
        
        # 确保两个方向正交（Gram-Schmidt）
        dir2 = dir2 - torch.dot(dir1, dir2) * dir1
        dir2 = self._normalize_direction_filterwise(dir2)
        
        return dir1, dir2

    def _generate_third_direction(self, dir1: torch.Tensor, dir2: torch.Tensor) -> torch.Tensor:
        """
        给定已归一化且近似正交的 dir1/dir2，再生成一个与它们正交的第三方向。
        """
        device = dir1.device
        rand = torch.randn_like(self._flattened_params, device=device)
        dir3 = self._normalize_direction_filterwise(rand.to(device)) # 确保在同一设备
        
        # 确保dir3在正确设备
        if dir3.device != device:
            dir3 = dir3.to(device)
            
        dir3 = dir3 - torch.dot(dir1, dir3) * dir1 - torch.dot(dir2, dir3) * dir2
        dir3 = self._normalize_direction_filterwise(dir3)
        return dir3

    def _generate_random_directions_3d(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        生成三个随机且近似正交的方向向量（已归一化），用于3D体积采样。
        
        Returns:
            (direction1, direction2, direction3)
        """
        rand1 = torch.randn_like(self._flattened_params)
        rand2 = torch.randn_like(self._flattened_params)
        rand3 = torch.randn_like(self._flattened_params)

        # 第一条方向
        dir1 = self._normalize_direction_filterwise(rand1)

        # 第二条：对dir1做Gram-Schmidt正交
        dir2 = self._normalize_direction_filterwise(rand2)
        dir2 = dir2 - torch.dot(dir1, dir2) * dir1
        dir2 = self._normalize_direction_filterwise(dir2)

        # 第三条：对dir1, dir2做Gram-Schmidt正交
        dir3 = self._normalize_direction_filterwise(rand3)
        dir3 = dir3 - torch.dot(dir1, dir3) * dir1 - torch.dot(dir2, dir3) * dir2
        dir3 = self._normalize_direction_filterwise(dir3)

        return dir1, dir2, dir3
    
    def _evaluate_loss(self, data_batch: Optional[Tuple] = None) -> float:
        """
        评估当前模型参数的loss（包括正则化项和自定义逻辑）。
        
        Args:
            data_batch: 可选的数据批次。如果为None，使用self.data_loader
            
        Returns:
            loss值（标量），包括数据loss和正则化项
        """
        # 设置模型模式
        if self.model_mode == 'train':
            self.model.train()
        else:
            self.model.eval()
        
        # 如果提供了自定义loss函数，使用它（完全覆盖默认逻辑）
        if self.custom_loss_fn is not None:
            if data_batch is not None:
                inputs, targets = data_batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                with torch.no_grad():
                    loss = self.custom_loss_fn(self.model, inputs, targets)
                    return loss.item() if isinstance(loss, torch.Tensor) else float(loss)
            
            elif self.data_loader is not None:
                total_loss = 0.0
                num_batches = 0
                
                with torch.no_grad():
                    for batch in self.data_loader:
                        # 应用pre_batch_hook（数据增强等）
                        if self.pre_batch_hook is not None:
                            batch = self.pre_batch_hook(batch)
                        
                        if isinstance(batch, (list, tuple)) and len(batch) == 2:
                            inputs, targets = batch
                        else:
                            inputs = batch
                            targets = None
                        
                        inputs = inputs.to(self.device)
                        if targets is not None:
                            targets = targets.to(self.device)
                        
                        loss = self.custom_loss_fn(self.model, inputs, targets)
                        total_loss += loss.item() if isinstance(loss, torch.Tensor) else float(loss)
                        num_batches += 1

                        if self.max_batches is not None and num_batches >= self.max_batches:
                            break
                
                return total_loss / num_batches if num_batches > 0 else 0.0
            else:
                raise ValueError("需要提供data_loader或data_batch来评估loss")
        
        # 默认loss计算逻辑
        data_loss = 0.0
        
        if data_batch is not None:
            inputs, targets = data_batch
            
            # 应用pre_batch_hook
            if self.pre_batch_hook is not None:
                batch = self.pre_batch_hook((inputs, targets))
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, targets = batch
            
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                
                # 应用post_batch_hook
                if self.post_batch_hook is not None:
                    loss = self.post_batch_hook(outputs, inputs, targets)
                
                data_loss = loss.item()
        
        elif self.data_loader is not None:
            total_loss = 0.0
            num_batches = 0
            
            with torch.no_grad():
                for batch in self.data_loader:
                    # 应用pre_batch_hook（数据增强、Mixup等）
                    if self.pre_batch_hook is not None:
                        batch = self.pre_batch_hook(batch)
                    
                    if isinstance(batch, (list, tuple)) and len(batch) == 2:
                        inputs, targets = batch
                    else:
                        inputs = batch
                        targets = None
                    
                    inputs = inputs.to(self.device)
                    if targets is not None:
                        targets = targets.to(self.device)
                    
                    outputs = self.model(inputs)
                    
                    if targets is not None:
                        loss = self.loss_fn(outputs, targets)
                    else:
                        # 如果loss_fn不需要targets
                        loss = self.loss_fn(outputs)
                    
                    # 应用post_batch_hook（label smoothing、temperature scaling等）
                    if self.post_batch_hook is not None:
                        loss = self.post_batch_hook(outputs, inputs, targets)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # 限制批次数以提高速度（可选）
                    if self.max_batches is not None and num_batches >= self.max_batches:
                        break
            
            data_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        else:
            raise ValueError("需要提供data_loader或data_batch来评估loss")
        
        # 计算正则化项
        reg_loss = 0.0
        if self.regularizer is not None:
            with torch.no_grad():
                reg_value = self.regularizer(self.model)
                if isinstance(reg_value, torch.Tensor):
                    reg_loss = reg_value.item()
                else:
                    reg_loss = float(reg_value)
        
        # 总loss = 数据loss + lambda * 正则化项
        total_loss = data_loss + self.regularizer_weight * reg_loss
        
        return total_loss
    
    def build_surface(
        self,
        grid_size: int = 50,
        range_scale: float = 0.1,
        directions: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        静态生成模式：在当前权重周围采样网格点，生成Loss Landscape表面。
        
        Args:
            grid_size: 网格大小（grid_size x grid_size）
            range_scale: 扰动范围缩放因子
            directions: 可选的两个方向向量。如果为None，自动生成随机方向
            verbose: 是否打印进度
            
        Returns:
            包含X, Y, loss_grid等数据的字典
        """
        if directions is None:
            dir1, dir2 = self._generate_random_directions()
        else:
            dir1, dir2 = directions
            # 确保方向已归一化
            dir1 = self._normalize_direction_filterwise(dir1)
            dir2 = self._normalize_direction_filterwise(dir2)
        
        # 存储方向（用于后续轨迹可视化）
        self._fixed_directions = (dir1, dir2)
        # 生成一个与上述方向正交的第三方向，便于默认提供3D轨迹投影
        self._fixed_directions_3d = (dir1, dir2, self._generate_third_direction(dir1, dir2))
        
        # 生成网格
        alpha_range = np.linspace(-range_scale, range_scale, grid_size)
        beta_range = np.linspace(-range_scale, range_scale, grid_size)
        # Ensure 0 is sampled (helps keep a near-baseline region even when grid_size is even).
        if grid_size > 1:
            mid = grid_size // 2
            if not np.any(np.isclose(alpha_range, 0.0)):
                alpha_range[mid] = 0.0
            if not np.any(np.isclose(beta_range, 0.0)):
                beta_range[mid] = 0.0
        
        X = np.zeros((grid_size, grid_size))
        Y = np.zeros((grid_size, grid_size))
        loss_grid = np.zeros((grid_size, grid_size))
        
        # 计算baseline loss（原点）
        self._restore_parameters()
        baseline_loss = self._evaluate_loss()
        
        if verbose:
            logger.info(f"Baseline loss: {baseline_loss:.6f}")
            logger.info(f"Computing {grid_size}x{grid_size} grid...")
        
        # 双重循环计算loss
        total_points = grid_size * grid_size
        computed = 0
        
        for i, alpha in enumerate(alpha_range):
            for j, beta in enumerate(beta_range):
                # 应用扰动
                self._apply_perturbation(dir1, dir2, alpha, beta)
                
                # 评估loss
                loss = self._evaluate_loss()
                
                X[i, j] = alpha
                Y[i, j] = beta
                loss_grid[i, j] = loss
                
                computed += 1
                if verbose and computed % (total_points // 10) == 0:
                    logger.info(f"Progress: {computed}/{total_points} ({100*computed/total_points:.1f}%)")
        
        # 恢复原始参数
        self._restore_parameters()
        
        # 准备返回数据
        result = {
            'X': X.tolist(),
            'Y': Y.tolist(),
            'loss_grid_2d': loss_grid.tolist(),
            'baseline_loss': float(baseline_loss),
            'grid_size': grid_size,
            'range_scale': range_scale,
            'mode': '2d',
        }
        
        # 保存到storage（如果提供）
        if self.storage is not None:
            self.storage.save_surface(result)
        
        if verbose:
            logger.info(f"Surface generation completed. Loss range: [{loss_grid.min():.6f}, {loss_grid.max():.6f}]")
        
        return result

    def _apply_perturbation_1d(self, direction: torch.Tensor, alpha: float):
        """
        应用一维扰动到模型参数。
        
        Args:
            direction: 方向向量（已归一化）
            alpha: 扰动强度
        """
        base_params = self._flattened_params.to(self.device)
        dir1 = direction.to(self.device)
        
        # 计算扰动后的参数
        perturbed = base_params + alpha * dir1
        
        # 重新组装并加载到模型
        state_dict = self._flatten_to_model(perturbed)
        self.model.load_state_dict(state_dict, strict=False)

    def build_line(
        self,
        grid_size: int = 100,
        range_scale: float = 0.1,
        direction: Optional[torch.Tensor] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        生成1D Loss Landscape：在当前权重周围沿单一方向采样，生成一条loss曲线。
        
        Args:
            grid_size: 采样点数量
            range_scale: 扰动范围缩放因子
            direction: 可选的方向向量。如果为None，自动生成随机方向
            verbose: 是否打印进度
            
        Returns:
            包含X, loss_line等数据的字典
        """
        if direction is None:
            # 生成随机方向
            rand = torch.randn_like(self._flattened_params)
            dir1 = self._normalize_direction_filterwise(rand)
        else:
            dir1 = direction
            # 确保方向已归一化
            dir1 = self._normalize_direction_filterwise(dir1)
        
        # 存储方向（用于后续轨迹可视化）
        self._fixed_directions = (dir1, dir1)  # 对于1D，两个方向相同
        
        # 生成采样点
        alpha_range = np.linspace(-range_scale, range_scale, grid_size)
        # Ensure 0 is sampled
        if grid_size > 1:
            mid = grid_size // 2
            if not np.any(np.isclose(alpha_range, 0.0)):
                alpha_range[mid] = 0.0
        
        X = np.zeros(grid_size)
        loss_line = np.zeros(grid_size)
        
        # 计算baseline loss（原点）
        self._restore_parameters()
        baseline_loss = self._evaluate_loss()
        
        if verbose:
            logger.info(f"[1D] Baseline loss: {baseline_loss:.6f}")
            logger.info(f"[1D] Computing {grid_size} points...")
        
        # 计算loss
        for i, alpha in enumerate(alpha_range):
            # 应用扰动
            self._apply_perturbation_1d(dir1, alpha)
            
            # 评估loss
            loss = self._evaluate_loss()
            
            X[i] = alpha
            loss_line[i] = loss
            
            if verbose and (i + 1) % (grid_size // 10) == 0:
                logger.info(f"[1D] Progress: {i+1}/{grid_size} ({100*(i+1)/grid_size:.1f}%)")
        
        # 恢复原始参数
        self._restore_parameters()
        
        # 准备返回数据
        result = {
            'X': X.tolist(),
            'loss_line_1d': loss_line.tolist(),
            'baseline_loss': float(baseline_loss),
            'grid_size': grid_size,
            'range_scale': range_scale,
            'mode': '1d',
        }
        
        # 保存到storage（如果提供）
        if self.storage is not None:
            self.storage.save_line(result)
        
        if verbose:
            logger.info(f"[1D] Line generation completed. Loss range: [{loss_line.min():.6f}, {loss_line.max():.6f}]")
        
        return result

    def _apply_perturbation_3d(
        self,
        direction1: torch.Tensor,
        direction2: torch.Tensor,
        direction3: torch.Tensor,
        alpha: float,
        beta: float,
        gamma: float,
    ):
        """
        应用三维扰动到模型参数（用于3D体积采样）。
        """
        base_params = self._flattened_params.to(self.device)
        dir1 = direction1.to(self.device)
        dir2 = direction2.to(self.device)
        dir3 = direction3.to(self.device)

        perturbed = base_params + alpha * dir1 + beta * dir2 + gamma * dir3
        state_dict = self._flatten_to_model(perturbed)
        self.model.load_state_dict(state_dict, strict=False)

    def build_volume(
        self,
        grid_size: int = 20,
        range_scale: float = 0.1,
        directions: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        3D体积模式：在当前权重周围沿三条方向采样，生成Loss Volume。
        
        与build_surface类似，但会返回:
            - X, Y, Z: 形状为 (N, N, N) 的坐标网格（对应alpha/beta/gamma）
            - loss_grid_3d: 形状为 (N, N, N) 的loss体积
        
        Args:
            grid_size: 每个维度上的网格大小（N），总体采样点数为 N^3
            range_scale: 扰动范围缩放因子
            directions: 可选的三个方向向量(已展平)。如果为None，自动生成随机正交方向
            verbose: 是否打印进度
        """
        if directions is None:
            dir1, dir2, dir3 = self._generate_random_directions_3d()
        else:
            dir1, dir2, dir3 = directions
            dir1 = self._normalize_direction_filterwise(dir1)
            dir2 = self._normalize_direction_filterwise(dir2)
            dir3 = self._normalize_direction_filterwise(dir3)

        # 生成网格
        alpha_range = np.linspace(-range_scale, range_scale, grid_size)
        beta_range = np.linspace(-range_scale, range_scale, grid_size)
        gamma_range = np.linspace(-range_scale, range_scale, grid_size)
        # Ensure 0 is sampled (helps baseline visibility and numerical stability in exports).
        if grid_size > 1:
            mid = grid_size // 2
            if not np.any(np.isclose(alpha_range, 0.0)):
                alpha_range[mid] = 0.0
            if not np.any(np.isclose(beta_range, 0.0)):
                beta_range[mid] = 0.0
            if not np.any(np.isclose(gamma_range, 0.0)):
                gamma_range[mid] = 0.0

        X = np.zeros((grid_size, grid_size, grid_size))
        Y = np.zeros((grid_size, grid_size, grid_size))
        Z = np.zeros((grid_size, grid_size, grid_size))
        loss_grid = np.zeros((grid_size, grid_size, grid_size))

        # baseline loss（原点）
        self._restore_parameters()
        baseline_loss = self._evaluate_loss()

        if verbose:
            logger.info(f"[3D] Baseline loss: {baseline_loss:.6f}")
            logger.info(f"[3D] Computing {grid_size}x{grid_size}x{grid_size} volume...")

        total_points = grid_size * grid_size * grid_size
        computed = 0

        for i, alpha in enumerate(alpha_range):
            for j, beta in enumerate(beta_range):
                for k, gamma in enumerate(gamma_range):
                    # 应用三维扰动
                    self._apply_perturbation_3d(dir1, dir2, dir3, alpha, beta, gamma)

                    loss = self._evaluate_loss()

                    X[i, j, k] = alpha
                    Y[i, j, k] = beta
                    Z[i, j, k] = gamma
                    loss_grid[i, j, k] = loss

                    computed += 1
                    if verbose and total_points >= 10 and computed % max(1, total_points // 10) == 0:
                        logger.info(
                            f"[3D] Progress: {computed}/{total_points} ({100 * computed / total_points:.1f}%)"
                        )

        # 恢复原始参数
        self._restore_parameters()

        result = {
            "X": X.tolist(),
            "Y": Y.tolist(),
            "Z": Z.tolist(),
            "loss_grid_3d": loss_grid.tolist(),
            "baseline_loss": float(baseline_loss),
            "grid_size": grid_size,
            "range_scale": range_scale,
            "mode": "3d",
        }

        # 持久化（如果提供storage）
        if self.storage is not None and hasattr(self.storage, "save_volume"):
            self.storage.save_volume(result)

        # 保存3D方向（用于轨迹投影）
        self._fixed_directions_3d = (dir1, dir2, dir3)

        if verbose:
            logger.info(
                f"[3D] Volume generation completed. Loss range: "
                f"[{loss_grid.min():.6f}, {loss_grid.max():.6f}]"
            )

        return result
    
    def log_position(
        self,
        epoch: int,
        verbose: bool = False,
        loss: Optional[float] = None,
        compute_loss: bool = False,
        val_loss: Optional[float] = None,
    ):
        """
        动态轨迹模式：记录当前epoch的权重位置。
        
        Args:
            epoch: 当前epoch编号
            verbose: 是否打印信息
            loss: 可选。当前点的“真实loss”（例如训练循环中计算的epoch平均train loss）。
            compute_loss: 若为True且loss未提供，则用Explorer内部的loss逻辑在self.data_loader上评估一次loss（默认最多10个batch）。
        """
        # 获取当前权重
        current_params = []
        for param in self.model.parameters():
            if param.requires_grad:
                current_params.append(param.data.flatten().cpu().clone())
        
        current_flat = torch.cat(current_params)
        
        # 计算相对于原始权重的偏移
        offset = current_flat - self._flattened_params
        
        self._trajectory_weights.append(offset)
        self._trajectory_epochs.append(epoch)

        # Record optional scalar loss for this point.
        if loss is None and compute_loss:
            try:
                loss = float(self._evaluate_loss())
            except Exception as e:
                warnings.warn(f"Failed to compute loss at epoch={epoch}: {e}")
                loss = None
        self._trajectory_losses.append(loss)
        
        # Record optional validation loss
        self._trajectory_val_losses.append(val_loss)
        
        # 如果使用PCA模式，收集所有权重用于后续PCA
        if self._trajectory_mode == 'pca':
            self._all_weights_for_pca.append(current_flat)
        
        if verbose:
            logger.debug(f"Logged position for epoch {epoch}")
    
    def build_trajectory(
        self,
        mode: str = 'fixed',
        directions: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        基于记录的轨迹点生成轨迹数据。
        
        Args:
            mode: 'fixed' 使用固定方向，'pca' 使用PCA降维
            directions: 固定模式下的方向向量（如果为None，使用build_surface时生成的方向）
            
        Returns:
            包含轨迹数据的字典
        """
        if len(self._trajectory_weights) == 0:
            raise ValueError("没有记录的轨迹点。请先调用log_position()")
        
        if mode == 'fixed':
            if directions is None:
                if self._fixed_directions_3d is not None:
                    dir1, dir2, dir3 = self._fixed_directions_3d
                elif self._fixed_directions is not None:
                    dir1, dir2 = self._fixed_directions
                    dir3 = self._generate_third_direction(dir1, dir2)
                    self._fixed_directions_3d = (dir1, dir2, dir3)
                else:
                    raise ValueError("固定模式需要方向向量。请先调用build_surface()/build_volume或提供directions参数")
            else:
                if len(directions) == 3:
                    dir1, dir2, dir3 = directions
                else:
                    dir1, dir2 = directions  # type: ignore
                    dir3 = self._generate_third_direction(dir1, dir2)
                dir1 = self._normalize_direction_filterwise(dir1)
                dir2 = self._normalize_direction_filterwise(dir2)
                dir3 = self._normalize_direction_filterwise(dir3)
                self._fixed_directions_3d = (dir1, dir2, dir3)
            
            # 投影到固定方向（包含第三个方向，便于3D轨迹）
            # 确保方向向量在CPU上（因为 _trajectory_weights 在CPU上）
            dir1_cpu = dir1.cpu()
            dir2_cpu = dir2.cpu()
            dir3_cpu = dir3.cpu()

            # IMPORTANT:
            # Explorer._normalize_direction_filterwise scales directions per-layer by ||w||, so directions are
            # generally NOT unit vectors. If we use plain dot-products as "coordinates", we effectively mix
            # coordinate scale with basis magnitude and can end up with extremely large trajectory coords and
            # range_scales. That makes the sampled grid far from the true trajectory and produces wildly
            # mis-scaled "landscape loss" values.
            #
            # Correct projection coefficient onto a (possibly non-unit) direction d is:
            #   alpha = <x, d> / <d, d>
            # so that alpha * d is the least-squares projection of x onto span(d).
            def _proj_coeff(v: torch.Tensor, d: torch.Tensor) -> float:
                denom = torch.dot(d, d).item()
                if denom <= 1e-12:
                    return 0.0
                return (torch.dot(v, d).item() / denom)

            traj_1 = [_proj_coeff(weight, dir1_cpu) for weight in self._trajectory_weights]
            traj_2 = [_proj_coeff(weight, dir2_cpu) for weight in self._trajectory_weights]
            traj_3 = [_proj_coeff(weight, dir3_cpu) for weight in self._trajectory_weights]
        
        elif mode == 'pca':
            # PCA降维
            if len(self._all_weights_for_pca) == 0:
                raise ValueError("PCA模式需要收集权重。请确保在log_position之前设置trajectory_mode='pca'")
            
            # 堆叠所有权重
            weights_matrix = torch.stack(self._all_weights_for_pca).numpy()
            
            # 计算PCA
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            weights_2d = pca.fit_transform(weights_matrix)
            
            traj_1 = weights_2d[:, 0].tolist()
            traj_2 = weights_2d[:, 1].tolist()
            traj_3 = None
        
        else:
            raise ValueError(f"未知的轨迹模式: {mode}")
        
        result = {
            'traj_1': traj_1,
            'traj_2': traj_2,
            'traj_3': traj_3,
            'epochs': self._trajectory_epochs.copy(),
        }

        if len(self._trajectory_losses) == len(self._trajectory_epochs):
            result["losses"] = self._trajectory_losses.copy()
        
        # Add validation losses if available
        if len(self._trajectory_val_losses) == len(self._trajectory_epochs):
            result["val_losses"] = self._trajectory_val_losses.copy()
        
        # 保存到storage（如果提供）
        if self.storage is not None:
            self.storage.save_trajectory(result)
        
        return result
    
    def set_trajectory_mode(self, mode: str):
        """
        设置轨迹记录模式。
        
        Args:
            mode: 'fixed' 或 'pca'
        """
        if mode not in ['fixed', 'pca']:
            raise ValueError(f"模式必须是'fixed'或'pca'，得到: {mode}")
        self._trajectory_mode = mode

    def build_hessian_trajectory(
        self,
        top_k: int = 5,
        compute_trace: bool = True,
        max_batches: Optional[int] = 5,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        沿轨迹计算Hessian特征。
        
        Args:
            top_k: 计算前k个特征值 (used as fallback or parameter for older methods)
            compute_trace: 是否计算Hessian迹
            max_batches: 计算Hessian时使用的最大batch数（减少计算量）
            verbose: 是否打印进度
            
        Returns:
            hessian_data 字典
        """
        if len(self._trajectory_weights) == 0:
            raise ValueError("没有记录的轨迹点。")
            
        # 准备 HessianCalculator
        hessian_calc = HessianCalculator(
            self.model,
            self.loss_fn,
            self.data_loader,
            device=self.device,
            max_batches=max_batches
        )
        
        results = {
            'epochs': [],
            'max_eigenvalue': [],
            'trace': [],
            'top_eigenvalues': []
        }
        
        total = len(self._trajectory_weights)
        
        if verbose:
            logger.info(f"Computing Hessian metrics for {total} points...")
        
        # 备份当前参数（虽然restore会覆盖，但好习惯）
        # self._backup_parameters() # Assume already backed up or managed by context
        
        # self._trajectory_weights 存储的是 offset = W_t - W_base
        # self._flattened_params 存储的是 W_base
        
        base_params = self._flattened_params.to(self.device)
        
        for i, (epoch, weight_offset) in enumerate(zip(self._trajectory_epochs, self._trajectory_weights)):
            if verbose:
                logger.info(f"Hessian analysis: Epoch {epoch} ({i+1}/{total})")
            
            # 恢复该epoch的参数
            current_flat = base_params + weight_offset.to(self.device)
            state_dict = self._flatten_to_model(current_flat)
            self.model.load_state_dict(state_dict, strict=False)
            
            # 计算 Top-K / Spectrum
            # Use Lanczos for better density estimation (default k=40)
            try:
                # k=40, max_iter=40 gives us ~40 Ritz values approximating the spectrum
                eigs = hessian_calc.compute_spectrum_lanczos(k=40, max_iter=40)
            except Exception as e:
                logger.warning(f"Lanczos failed at epoch {epoch}: {e}, falling back to Power Iteration")
                eigs = hessian_calc.compute_top_eigenvalues(k=top_k)

            # Max Eigenvalue (Sharpness)
            if eigs:
                # Lanczos returns algebraic sorted. Power iteration returns sorted magnitudes.
                # To be safe:
                max_eig = max(abs(min(eigs)), abs(max(eigs)))
            else:
                max_eig = 0.0
            
            # 计算 Trace
            trace_val = 0.0
            if compute_trace:
                trace_val = hessian_calc.compute_trace()
                
            results['epochs'].append(epoch)
            results['max_eigenvalue'].append(max_eig)
            results['trace'].append(trace_val)
            results['top_eigenvalues'].append(eigs)
            
        # 恢复原始参数
        self._restore_parameters()
        
        # 保存
        if self.storage is not None and hasattr(self.storage, 'save_hessian'):
            self.storage.save_hessian(results)
            
        return results

    def build_hessian_snapshot(
        self,
        epoch: int = 0,
        top_k: int = 40,
        compute_trace: bool = True,
        max_batches: Optional[int] = 5,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        在“当前权重点”计算一次 Hessian 特征（无训练轨迹时使用）。

        这会导出一个单点的 Hessian 数据，便于前端展示为 Snapshot：
          - epochs: [epoch]
          - max_eigenvalue: [λmax]
          - trace: [tr(H)]
          - top_eigenvalues: [spectrum]
        """
        if self.data_loader is None:
            raise ValueError("build_hessian_snapshot 需要 data_loader 来评估 loss/HVP")

        hessian_calc = HessianCalculator(
            self.model,
            self.loss_fn,
            self.data_loader,
            device=self.device,
            max_batches=max_batches,
        )

        if verbose:
            logger.info(f"Hessian snapshot at epoch={epoch} ...")

        # Spectrum: prefer Lanczos (gives a density-friendly set of Ritz values)
        try:
            eigs = hessian_calc.compute_spectrum_lanczos(k=top_k, max_iter=max(top_k, 40))
        except Exception as e:
            logger.warning(f"Lanczos failed for snapshot: {e}, falling back to Power Iteration")
            eigs = hessian_calc.compute_top_eigenvalues(k=min(5, top_k))

        if eigs:
            max_eig = max(abs(min(eigs)), abs(max(eigs)))
        else:
            max_eig = 0.0

        trace_val = 0.0
        if compute_trace:
            trace_val = float(hessian_calc.compute_trace())

        results = {
            "epochs": [int(epoch)],
            "max_eigenvalue": [float(max_eig)],
            "trace": [float(trace_val)],
            "top_eigenvalues": [eigs],
        }

        if self.storage is not None and hasattr(self.storage, "save_hessian"):
            self.storage.save_hessian(results)

        return results
