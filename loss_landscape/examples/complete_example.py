"""
1D Burgers方程示例：比较MSE Loss和Physics Loss的Loss Landscape

使用1D Burgers方程 u_t + u * u_x = ν * u_xx 生成数据。
这是一个非线性PDE，包含：
- 时间导数项 u_t
- 非线性对流项 u * u_x（会产生激波，使loss landscape非常复杂）
- 扩散项 ν * u_xx

这个问题的physics loss会涉及一阶和二阶偏导数，以及非线性项，
会产生极其复杂、坑坑洼洼的loss landscape，包含多个局部最小值、鞍点和狭窄的谷底。

本示例会导出 4 个结果（4 个 run 目录）：
1) physics loss (with train trajectory)
2) physics loss (no train / no trajectory)
3) mse loss (with train trajectory)
4) mse loss (no train / no trajectory)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from loss_landscape import Explorer, LandscapeStorage
from loguru import logger
import numpy as np
import os
from datetime import datetime
import platform
try:
    import psutil
except ImportError:
    psutil = None


class BurgersEquationDataset(Dataset):
    """基于1D Burgers方程的数据集"""
    def __init__(self, num_samples=2000, L=2.0, nu=0.05, x_range=(0, 2), t_range=(0, 1)):
        """
        Args:
            num_samples: 样本数量
            L: 空间域长度
            nu: 粘性系数（扩散系数）。注意：nu越小激波越陡，训练越难。
            x_range: x坐标范围
            t_range: t坐标范围
        """
        self.num_samples = num_samples
        self.L = L
        self.nu = nu
        
        # 生成随机采样点
        x = np.random.uniform(x_range[0], x_range[1], num_samples)
        t = np.random.uniform(t_range[0], t_range[1], num_samples)
        
        # 使用Burgers方程的精确行波解（Traveling Wave Solution）
        # u(x, t) = 1 - tanh((x - t - x0) / (2*nu))
        # 这是一个向右移动的激波
        x0 = 0.5  # 激波初始位置
        
        # 计算解析解
        # argument of tanh
        arg = (x - t - x0) / (2 * nu)
        u_true = 1.0 - np.tanh(arg)
        
        # 添加少量噪声（模拟真实观测数据）
        noise = np.random.normal(0, 0.01, num_samples)
        u_true = u_true + noise
        
        self.inputs = torch.FloatTensor(np.column_stack([x, t]))
        self.targets = torch.FloatTensor(u_true.reshape(-1, 1))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class BurgersEquationModel(nn.Module):
    """神经网络模型，用于预测Burgers方程的解"""
    def __init__(self, hidden_sizes=[256, 256, 128, 64]):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc5 = nn.Linear(hidden_sizes[3], 1)
        # 使用Swish/SiLU激活函数，对非线性PDE效果更好
        self.activation = nn.SiLU()
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.fc5(x)
        return x


def physics_loss_fn(model, inputs, targets, nu=0.01, eps=1e-3):
    """
    计算physics-informed loss for Burgers方程
    Burgers方程: u_t + u * u_x = ν * u_xx
    
    这是一个非线性PDE，包含：
    - 时间导数项 u_t
    - 非线性对流项 u * u_x（会产生激波）
    - 扩散项 ν * u_xx
    
    Loss = MSE_data + lambda * (u_t + u*u_x - ν*u_xx)^2
    
    这个loss会产生极其复杂的landscape，因为：
    1. 非线性项 u*u_x 使得loss高度非凸
    2. 需要同时满足多个约束条件
    3. 激波区域梯度变化剧烈
    """
    predictions = model(inputs)
    mse_loss = nn.functional.mse_loss(predictions, targets)
    
    # 确保inputs需要梯度（用于自动微分）
    # 使用torch.enable_grad()确保在eval模式下也能计算梯度
    with torch.enable_grad():
        # 创建需要梯度的输入副本
        inputs_grad = inputs.clone().detach().requires_grad_(True)
        
        # 使用自动微分计算偏导数
        u = model(inputs_grad)
        
        # 计算对完整输入的梯度
        grad_u = torch.autograd.grad(
            outputs=u, inputs=inputs_grad,
            grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True,
            only_inputs=True
        )[0]
        
        # 提取 u_x (对空间的偏导数) - inputs_grad的第一列
        u_x = grad_u[:, 0:1]
        
        # 提取 u_t (对时间的偏导数) - inputs_grad的第二列
        u_t = grad_u[:, 1:2]
        
        # 计算 u_xx (对空间的二阶偏导数)
        # 需要对u_x再次求导，但只对x分量求导
        u_xx_grad = torch.autograd.grad(
            outputs=u_x, inputs=inputs_grad,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True,
            only_inputs=True
        )[0]
        u_xx = u_xx_grad[:, 0:1]  # 提取对x的二阶偏导数
        
        # Burgers方程残差: u_t + u * u_x - ν * u_xx = 0
        # 注意：u 和 u_x 需要从同一个点计算
        physics_residual = u_t + u * u_x - nu * u_xx
        
        # 计算physics constraint loss
        physics_constraint = torch.mean(physics_residual ** 2)
    
    # 组合loss，physics项的权重较大，因为这是主要约束
    # 非线性项使得landscape非常复杂，需要更大的权重来平衡
    # 降低权重到0.1，避免physics loss过大导致梯度不稳定
    total_loss = mse_loss + 0.1 * physics_constraint
    
    return total_loss


def train_model(model, train_loader, val_loader, optimizer, step_loss_fn, device, num_epochs=50, explorer=None, scheduler=None):
    """
    Train a model and (optionally) record its parameter trajectory via Explorer.log_position().

    step_loss_fn signature: step_loss_fn(model, inputs, targets) -> scalar tensor
    
    Returns:
        dict with training statistics
    """
    def eval_epoch_loss(data_loader):
        """
        Evaluate loss on a full dataloader using *fixed* weights.
        This is the quantity that should be comparable to loss-landscape evaluation
        when the landscape also uses the same dataloader + same loss definition.
        """
        model.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                loss = step_loss_fn(model, inputs, targets)
                total += float(loss.item())
                n += 1
        model.train()
        return (total / n) if n > 0 else 0.0

    model.train()
    training_stats = {
        # "loss_history" is what we export/plot as "Train Loss".
        # We make it an epoch-end evaluation loss on the full train_loader so it is directly comparable
        # to loss-landscape evaluation on the same train_loader.
        "loss_history": [],
        # Keep the traditional "running" epoch loss (average of per-batch training losses during updates)
        # for debugging/interpretation.
        "train_running_loss_history": [],
        "val_loss_history": [],
        "learning_rate_history": [],
        "final_loss": None,
        "final_val_loss": None,
        "initial_loss": None,
        "min_loss": float('inf'),
        "min_loss_epoch": None,
    }
    
    for epoch in range(num_epochs):
        # Training phase
        total_loss = 0.0
        num_batches = 0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            loss = step_loss_fn(model, inputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        
        if scheduler is not None:
            scheduler.step()

        avg_loss_running = total_loss / num_batches if num_batches > 0 else 0.0
        current_lr = optimizer.param_groups[0]['lr']

        # Epoch-end train loss with fixed weights (full train_loader).
        # This is the number we want to compare against the landscape.
        train_eval_loss = eval_epoch_loss(train_loader)
        
        # Validation phase
        val_loss = None
        if val_loader is not None:
            val_loss = eval_epoch_loss(val_loader)
        
        # Record statistics
        training_stats["loss_history"].append(train_eval_loss)
        training_stats["train_running_loss_history"].append(avg_loss_running)
        training_stats["learning_rate_history"].append(current_lr)
        if val_loss is not None:
            training_stats["val_loss_history"].append(val_loss)
        
        if epoch == 0:
            training_stats["initial_loss"] = train_eval_loss
        
        if train_eval_loss < training_stats["min_loss"]:
            training_stats["min_loss"] = train_eval_loss
            training_stats["min_loss_epoch"] = epoch
        
        training_stats["final_loss"] = train_eval_loss
        if val_loss is not None:
            training_stats["final_val_loss"] = val_loss

        # Record trajectory position + true (epoch-average) train loss and val loss for comparison in the UI.
        if explorer is not None:
            explorer.log_position(epoch=epoch, verbose=False, loss=train_eval_loss, val_loss=val_loss)
        
        if (epoch + 1) % 10 == 0:
            val_str = f", Val Loss: {val_loss:.6f}" if val_loss is not None else ""
            logger.info(
                f"  Epoch {epoch+1}/{num_epochs}, "
                f"Train Loss (epoch-end eval): {train_eval_loss:.6f}, "
                f"Train Loss (running): {avg_loss_running:.6f}"
                f"{val_str}, LR: {current_lr:.6f}"
            )
    
    return training_stats


def normalize_direction_filterwise(direction, model):
    """
    Match Explorer._normalize_direction_filterwise logic to ensure consistent scaling.
    Now implements Li et al. (2018) scaling: d_new = d / ||d|| * ||w||
    This ensures that perturbations are proportional to the magnitude of the weights in each layer.
    """
    normalized_parts = []
    idx = 0
    # Iterate parameters in the same order as flattened params
    params = [p for p in model.parameters() if p.requires_grad]
    
    for param in params:
        shape = param.shape
        size = np.prod(shape)
        layer_direction = direction[idx:idx+size]
        
        # Calculate Frobenius norm for this layer direction
        layer_d = layer_direction.reshape(shape)
        d_norm = torch.norm(layer_d, p='fro')
        
        scale = 1.0
        if d_norm > 1e-8:
            scale = 1.0 / d_norm
            
            # Apply weight norm scaling
            w_norm = torch.norm(param.data, p='fro')
            scale *= w_norm
            
        normalized_layer = layer_direction * scale
            
        normalized_parts.append(normalized_layer)
        idx += size
    
    return torch.cat(normalized_parts)


def compute_pca_directions(trajectory_weights, device, model):
    """
    从轨迹权重差值中计算PCA方向和合适的range_scale
    
    Args:
        trajectory_weights: list of tensors, 存储的是 (W_t - W_init)
        device: torch device
        model: model instance (used for parameter shapes)
        
    Returns:
        (dir1, dir2, dir3), range_scale
    """
    # 堆叠权重偏移: (N, D)
    # trajectory_weights 里存的是 W_t - W_init
    # 我们希望 Landscape 以 W_final 为中心 (0,0,0)
    # 所以我们需要的数据相对于 W_final 的偏移:
    # W_t - W_final = (W_t - W_init) - (W_final - W_init)
    
    offsets = torch.stack(trajectory_weights).to(device) # (N, D)
    final_offset = offsets[-1] # W_final - W_init
    
    # 重新中心化到 W_final
    centered_data = offsets - final_offset # (N, D)
    
    # 计算SVD: centered_data = U * S * Vh
    # Vh 的行就是主成分方向
    # 注意：如果 N < D，我们最多得到 N 个方向
    U, S, Vh = torch.linalg.svd(centered_data, full_matrices=False)
    
    # 取前3个主成分作为方向
    # 确保至少有3个方向（如果轨迹点少于3个，用随机方向补全，虽然这在训练中不太可能）
    k = Vh.shape[0]
    dirs = []
    for i in range(3):
        if i < k:
            d = Vh[i]
        else:
            d = torch.randn_like(Vh[0])
        d = d / d.norm() # 归一化
        dirs.append(d)
        
    # IMPORTANT: Normalize directions using the same logic as Explorer
    # Otherwise the coordinate system used for range_scale calculation (here)
    # will differ from the one used for grid generation (Explorer), causing out-of-bounds.
    norm_dirs = [normalize_direction_filterwise(d, model) for d in dirs]
    dir1, dir2, dir3 = norm_dirs
    
    # 计算投影范围以确定 range_scale
    # 投影 = centered_data @ Vh.T
    # 我们只关心前3个归一化后的方向的投影
    # Proj coordinates: P = <W, d> / <d, d> for each direction d.
    # IMPORTANT: Explorer uses filterwise-normalized directions which are generally NOT unit vectors,
    # so we must divide by ||d||^2 to get true projection coefficients. This matches
    # Explorer.build_trajectory's coordinate convention.
    D = torch.stack(norm_dirs)  # (3, D)
    denom = (D * D).sum(dim=1).clamp_min(1e-12)  # (3,)
    proj = torch.matmul(centered_data, D.T) / denom  # (N, 3)
    
    # 找到最大的绝对坐标值
    max_coord = proj.abs().max().item()
    
    # 设置 range_scale，留出 20% 的余量
    range_scale = max_coord * 1.2
    
    # 避免 scale 太小
    range_scale = max(range_scale, 0.1)
    
    logger.info(f"PCA Analysis:")
    logger.info(f"  Max projection coord (normalized basis): {max_coord:.4f}")
    logger.info(f"  Selected range_scale: {range_scale:.4f}")
    
    # Return original directions (pre-normalization)? 
    # Explorer applies normalization internally if directions are passed.
    # HOWEVER, we calculated range_scale based on `norm_dirs`.
    # If we pass `dirs` (un-normalized) to Explorer, Explorer will normalize them to `norm_dirs` (deterministically).
    # So we should pass the raw `dirs` (or `norm_dirs`, doesn't matter if logic is idempotent).
    # Explorer's normalization is idempotent (x/|x| / |x/|x|| = x/|x|).
    # So let's return `dirs` (the PCA unit vectors) to be safe, knowing Explorer will convert them to `norm_dirs`.
    return (dirs[0], dirs[1], dirs[2]), range_scale


def generate_landscape(model, data_loader, loss_fn, loss_name, output_dir, device, 
                      grid_size_1d=100, grid_size_2d=30, grid_size_3d=16, metadata_info=None,
                      trajectory_weights=None, trajectory_epochs=None, trajectory_base_weights=None,
                      pca_directions=None, pca_range_scale=None, training_stats=None):
    """生成loss landscape（包括1D、2D和3D）"""
    
    landscape_path = f"{output_dir}/complete_example.landscape"
    storage = LandscapeStorage(landscape_path, mode='create')
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Generating {loss_name} Loss Landscape (1D + 2D + 3D)...")
    logger.info(f"{'='*60}")
    
    if loss_name == "mse":
        loss_description = {
            "type": "MSE (Mean Squared Error)",
            "formula": "L = (1/n) * Σ(y_pred - y_true)²",
            "description": "标准均方误差损失函数。"
        }
        def custom_loss_fn(m, inputs, targets):
            outputs = m(inputs)
            return nn.functional.mse_loss(outputs, targets)
    elif loss_name == "physics":
        loss_description = {
            "type": "Physics-Informed Loss (Burgers Equation)",
            "formula": "L = MSE + λ * |u_t + u·u_x - ν·u_xx|²",
            "description": "结合数据拟合和Burgers方程物理约束的损失函数。包含非线性对流项u·u_x，会产生极其复杂、坑坑洼洼的loss landscape，包含多个局部最小值、鞍点和狭窄的谷底。"
        }
        def custom_loss_fn(m, inputs, targets):
            return physics_loss_fn(m, inputs, targets, nu=0.05)
    else:
        raise ValueError(f"Unknown loss name: {loss_name}")
    
    # 使用PCA计算出的参数，或者默认值
    range_scale = pca_range_scale if pca_range_scale is not None else 0.1
    direction_1d = pca_directions[0] if pca_directions else None
    directions_2d = (pca_directions[0], pca_directions[1]) if pca_directions else None
    directions_3d = pca_directions if pca_directions else None
    
    # 收集硬件和系统信息
    system_info = {
        "device": str(device),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
    }
    
    if psutil:
        system_info["cpu_count"] = psutil.cpu_count()
        system_info["memory_total_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
    
    if torch.cuda.is_available():
        system_info["cuda_available"] = True
        system_info["cuda_version"] = torch.version.cuda
        system_info["cudnn_version"] = torch.backends.cudnn.version()
        if torch.cuda.device_count() > 0:
            system_info["gpu_name"] = torch.cuda.get_device_name(0)
            system_info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    else:
        system_info["cuda_available"] = False
    
    # 收集landscape生成参数
    landscape_params = {
        "grid_size_1d": grid_size_1d,
        "grid_size_2d": grid_size_2d,
        "grid_size_3d": grid_size_3d,
        "method": "PCA-aligned" if pca_directions else "Random",
        "range_scale": float(range_scale),
        "has_trajectory": trajectory_weights is not None and len(trajectory_weights) > 0,
        "trajectory_points": len(trajectory_epochs) if trajectory_epochs else 0,
        "description": f"Grid centered at final weights. Scale={range_scale:.4f} covers full trajectory."
    }
    
    # 收集并保存详细的metadata
    if metadata_info:
        full_metadata = {
            **metadata_info,
            "loss_function": {
                "name": loss_name,
                **loss_description
            },
            "landscape_generation": landscape_params,
            "system": system_info,
        }
        
        # 添加训练统计信息（如果有）
        if training_stats:
            full_metadata["training_statistics"] = {
                "initial_loss": float(training_stats["initial_loss"]) if training_stats["initial_loss"] is not None else None,
                "final_loss": float(training_stats["final_loss"]) if training_stats["final_loss"] is not None else None,
                "min_loss": float(training_stats["min_loss"]) if training_stats["min_loss"] != float('inf') else None,
                "min_loss_epoch": training_stats["min_loss_epoch"],
                "loss_reduction": float(training_stats["initial_loss"] - training_stats["final_loss"]) if training_stats["initial_loss"] and training_stats["final_loss"] else None,
                "loss_reduction_percent": float((training_stats["initial_loss"] - training_stats["final_loss"]) / training_stats["initial_loss"] * 100) if training_stats["initial_loss"] and training_stats["final_loss"] and training_stats["initial_loss"] > 0 else None,
                # Export per-epoch LR so the frontend can draw the stability boundary 2/η.
                # (η is the learning rate.)
                "learning_rate_history": [float(x) for x in (training_stats.get("learning_rate_history") or [])],
                # Optional: keep val loss history for contextual overlays if needed
                "val_loss_history": [float(x) for x in (training_stats.get("val_loss_history") or [])],
            }
        
        storage.save_metadata(full_metadata)
    
    with Explorer(
        model,
        loss_fn,
        data_loader,
        device=device,
        storage=storage,
        custom_loss_fn=custom_loss_fn,
        model_mode='eval',
        # More accurate (but slower): evaluate full loader instead of only a few batches.
        max_batches=None,
    ) as explorer:
        
        # 1. 恢复轨迹数据（用于 build_trajectory）
        if trajectory_weights is not None and trajectory_epochs is not None and trajectory_base_weights is not None:
            # 计算当前权重(W_final)相对于训练基准(W_init)的偏移
            # model此时处于 eval 模式，且加载了训练后的权重 (W_final)
            current_flat = []
            for param in model.parameters():
                if param.requires_grad:
                    current_flat.append(param.data.flatten().cpu().clone())
            current_flat = torch.cat(current_flat)
            
            # W_init
            w_init = trajectory_base_weights
            
            # W_final - W_init
            base_offset = current_flat - w_init
            
            # 验证: trajectory_weights[-1] 应该是 W_final - W_init
            # 如果 trajectory_weights 是 [W_t - W_init]，那么最后一个点应该是 W_final - W_init
            last_traj_offset = trajectory_weights[-1]
            diff = (last_traj_offset - base_offset).abs().max().item()
            logger.info(f"Trajectory endpoint check: max diff between traj[-1] and (W_final-W_init) = {diff:.6f}")
            
            # 调整轨迹： Explorer 期望 _trajectory_weights 存储的是 (W_t - W_final)
            # 我们传入的 trajectory_weights 是 (W_t - W_init)
            # 所以 (W_t - W_final) = (W_t - W_init) - (W_final - W_init)
            adjusted_weights = [w - base_offset for w in trajectory_weights]
            
            # 再次验证: adjusted_weights[-1] 应该是 0 (因为 W_final - W_final = 0)
            final_adj_norm = adjusted_weights[-1].norm().item()
            logger.info(f"Adjusted trajectory endpoint norm (should be ~0): {final_adj_norm:.6f}")
            
            explorer._trajectory_weights = adjusted_weights
            explorer._trajectory_epochs = trajectory_epochs.copy()
            logger.info(f"Restored {len(trajectory_epochs)} trajectory points")

            # Restore per-epoch train/val losses so the frontend can draw curves.
            # `trajectory_epochs` includes -1 (init) and `train_epochs` (final marker), while
            # training_stats histories are typically epochs 0..N-1.
            if training_stats is not None:
                train_hist = training_stats.get("loss_history") or []
                val_hist = training_stats.get("val_loss_history") or []

                restored_train_losses = []
                restored_val_losses = []
                for ep in trajectory_epochs:
                    # Epoch -1 (init): use initial_loss if available
                    if ep == -1:
                        init_loss = training_stats.get("initial_loss")
                        restored_train_losses.append(float(init_loss) if init_loss is not None else None)
                        # For val, we don't usually have pre-init val loss unless we calculated it
                        restored_val_losses.append(None) 
                    elif isinstance(ep, int) and 0 <= ep < len(train_hist):
                        restored_train_losses.append(float(train_hist[ep]) if train_hist[ep] is not None else None)
                        restored_val_losses.append(float(val_hist[ep]) if ep < len(val_hist) and val_hist[ep] is not None else None)
                    elif ep == len(train_hist): # Final epoch marker might be equal to length
                         # Use final loss
                        final_loss = training_stats.get("final_loss")
                        restored_train_losses.append(float(final_loss) if final_loss is not None else None)
                        final_val = training_stats.get("final_val_loss")
                        restored_val_losses.append(float(final_val) if final_val is not None else None)
                    else:
                        restored_train_losses.append(None)
                        restored_val_losses.append(None)

                explorer._trajectory_losses = restored_train_losses
                explorer._trajectory_val_losses = restored_val_losses

        # If no trajectory is provided, we intentionally do NOT create a dummy trajectory.
        
        # 1. 生成1D Line
        direction_desc = "PCA first direction" if direction_1d is not None else "random direction"
        logger.info(f"\n[1/5] Generating 1D Loss Landscape line (using {direction_desc})...")
        result_1d = explorer.build_line(
            grid_size=grid_size_1d,
            range_scale=range_scale,
            direction=direction_1d,
            verbose=True,
        )
        
        # 2. 生成2D Surface
        directions_desc = "PCA first two directions" if directions_2d is not None else "random directions"
        logger.info(f"\n[2/5] Generating 2D Loss Landscape surface (using {directions_desc})...")
        result_2d = explorer.build_surface(
            grid_size=grid_size_2d,
            range_scale=range_scale,
            directions=directions_2d,
            verbose=True,
        )
        
        # 3. 生成3D Volume
        directions_3d_desc = "PCA three directions" if directions_3d is not None else "random directions"
        logger.info(f"\n[3/5] Generating 3D Loss Volume (using {directions_3d_desc})...")
        result_3d = explorer.build_volume(
            grid_size=grid_size_3d,
            range_scale=range_scale,
            directions=directions_3d,
            verbose=True,
        )
        
        # 4. 构建轨迹 (使用相同的方向)
        # 注意：这里必须显式传入 directions，否则 build_trajectory 可能会自己生成或者使用缓存的
        # 传入我们用于生成 landscape 的相同 PCA 方向，确保坐标一致
        if trajectory_weights is not None and len(explorer._trajectory_weights) > 1:
            logger.info(f"\n[4/6] Building trajectory...")
            trajectory = explorer.build_trajectory(mode='fixed', directions=directions_3d)
            logger.info(f"✓ Trajectory generated: {len(trajectory['epochs'])} points")
            
            # 5. 计算 Hessian (仅当有轨迹时)
            logger.info(f"\n[5/6] Analyzing Hessian along trajectory (Top-5 Eigs + Trace)...")
            hessian_data = explorer.build_hessian_trajectory(top_k=5, max_batches=5) # Limit batches for speed in demo
            logger.info(f"✓ Hessian analysis completed")
            
        else:
            logger.info(f"\n[4/6] Skipping trajectory (no training data)")
            # No-train mode: still compute a single Hessian snapshot at the current weights.
            logger.info(f"\n[5/6] Analyzing Hessian snapshot (no trajectory)...")
            _ = explorer.build_hessian_snapshot(epoch=0, top_k=40, max_batches=5)
            logger.info("✓ Hessian snapshot completed")
    
    # 导出数据
    logger.info(f"\n[6/6] Exporting data...")
    json_path = f"{output_dir}/complete_example.json"
    storage.export_for_frontend(json_path)
    storage.close()
    
    # Remove intermediate .landscape file as we only need the JSON output
    if os.path.exists(landscape_path):
        try:
            os.remove(landscape_path)
            logger.info(f"Removed intermediate file: {landscape_path}")
        except OSError as e:
            logger.warning(f"Error removing intermediate file {landscape_path}: {e}")
    
    return result_1d, result_2d, result_3d, json_path


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    torch.manual_seed(42)
    np.random.seed(42)
    
    # ==================== 数据准备 ====================
    # 使用 nu=0.05 避免过于尖锐的梯度，保证数值稳定性
    logger.info("Generating Burgers equation dataset...")
    nu_val = 0.05
    train_dataset = BurgersEquationDataset(num_samples=2000, L=2.0, nu=nu_val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    # Use a deterministic loader for landscape evaluation to avoid introducing visual "noise"
    # from shuffled mini-batch order (especially if max_batches is limited in other configs).
    landscape_train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)
    eval_dataset = BurgersEquationDataset(num_samples=500, L=2.0, nu=nu_val)
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # ==================== 模型初始化 ====================
    base_model = BurgersEquationModel(hidden_sizes=[256, 256, 128, 64]).to(device)
    base_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
    total_params = sum(p.numel() for p in base_model.parameters())
    
    # ==================== Helper: train + record + PCA ====================
    mse_loss_fn = nn.MSELoss()

    def train_and_record(loss_key: str, train_epochs: int = 100):
        """
        Train a fresh model starting from the same initialization, record trajectory,
        then compute PCA directions aligned with that trajectory.
        
        Returns:
            (model, trajectory_base_weights, trajectory_weights, trajectory_epochs, 
             pca_directions, pca_range_scale, training_stats)
        """
        logger.info("\n" + "=" * 60)
        logger.info(f"Training model and recording trajectory ({loss_key})")
        logger.info("=" * 60)

        model = BurgersEquationModel(hidden_sizes=[256, 256, 128, 64]).to(device)
        model.load_state_dict(base_state, strict=True)

        if loss_key == "mse":
            step_loss_fn = lambda m, x, y: nn.functional.mse_loss(m(x), y)
            lr = 1e-3  # MSE可以稍微大一点
        elif loss_key == "physics":
            step_loss_fn = lambda m, x, y: physics_loss_fn(m, x, y, nu=nu_val)
            lr = 5e-4  # Physics loss由于有高阶导数，梯度可能较大，LR小一点更稳
        else:
            raise ValueError(loss_key)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

        training_stats = None
        
        # IMPORTANT: Do NOT use 'with Explorer(...)', because __exit__ restores the model to its original state!
        # We want the trained model weights to persist after this block so we can generate the landscape around the solution.
        temp_explorer = Explorer(model, mse_loss_fn, eval_loader, device=device, storage=None, model_mode='train')
        temp_explorer._backup_parameters()  # Initialize internal state (flattened params) from current (init) model
        
        temp_explorer.log_position(epoch=-1, verbose=True)  # init
        training_stats = train_model(
            model, train_loader, eval_loader, optimizer, step_loss_fn, device,
            num_epochs=train_epochs, explorer=temp_explorer, scheduler=scheduler
        )
        temp_explorer.log_position(epoch=train_epochs, verbose=True)  # final

        trajectory_base_weights = temp_explorer._flattened_params.clone()  # W_init
        trajectory_weights = [w.clone() for w in temp_explorer._trajectory_weights]  # W_t - W_init
        trajectory_epochs = temp_explorer._trajectory_epochs.copy()

        logger.info("\n" + "=" * 60)
        logger.info(f"Computing PCA directions from trajectory ({loss_key})")
        logger.info("=" * 60)
        pca_directions, pca_range_scale = compute_pca_directions(trajectory_weights, device, model)

        return model, trajectory_base_weights, trajectory_weights, trajectory_epochs, pca_directions, pca_range_scale, training_stats

    # ==================== Generate and export 4 runs ====================
    # We intentionally generate 4 *separate* run directories under outputs/ so the web UI can
    # treat each one as an independent selectable run.
    base_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def run_dir(loss_key: str, train_mode: str) -> str:
        # e.g. outputs/run_20260108_110000_mse_with_train
        return os.path.join("outputs", f"run_{base_timestamp}_{loss_key}_{train_mode}")
    
    base_metadata = {
        "summary": "Compare MSE Loss and Physics-Informed Loss on a 1D Burgers equation dataset. This is a nonlinear PDE with convective term that produces shock waves, resulting in an extremely complex, bumpy loss landscape with multiple local minima, saddle points, and narrow valleys. Training uses Adam optimizer (lr=3e-4) for 100 epochs with StepLR scheduler.",
        "experiment": {
            "name": "1D Burgers Equation Loss Landscape Comparison",
            "description": "比较MSE Loss和Physics Loss在1D Burgers方程预测任务上的Loss Landscape差异。Burgers方程包含非线性对流项u·u_x，会产生极其复杂、坑坑洼洼的landscape，包含多个局部最小值、鞍点和狭窄的谷底。",
            "created_at": datetime.now().isoformat(),
        },
        "dataset": {
            "name": "1D Burgers Equation Synthetic Dataset",
            "formula": "PDE: u_t + u·u_x = ν·u_xx (nonlinear convection-diffusion equation)",
            "parameters": {
                "L": "2.0 (spatial domain length)",
                "ν (nu)": "0.01 (viscosity/diffusion coefficient)",
                "x_range": "[0, 2]",
                "t_range": "[0, 1]",
                "features": "Nonlinear convective term u·u_x creates shock waves, making the problem highly challenging"
            },
            "noise": "Gaussian (mean=0, std=0.03)",
            "samples": {
                "train": len(train_dataset),
                "eval": len(eval_dataset)
            }
        },
        "model": {
            "name": "BurgersEquationModel (MLP)",
            "architecture": "Input(2) -> Linear(256) -> SiLU -> Linear(256) -> SiLU -> Linear(128) -> SiLU -> Linear(64) -> SiLU -> Linear(1)",
            "total_params": total_params,
            "trainable_params": total_params 
        },
        "training": {
            "epochs": 100, 
            "optimizer": {
                "name": "Adam",
                "learning_rate": 0.0003,
                "weight_decay": 0.0,
                "description": "Adam optimizer with initial lr=3e-4 (smaller LR needed due to nonlinearity)"
            },
            "scheduler": {
                "name": "StepLR",
                "step_size": 20,
                "gamma": 0.5
            },
            "regularization": {
                "type": "Physics-Informed Regularization (Complex)",
                "description": "Physics loss includes nonlinear PDE residual: |u_t + u·u_x - ν·u_xx|². The nonlinear convective term u·u_x makes the loss landscape highly non-convex with multiple local minima, saddle points, and narrow valleys. This creates an extremely complex and bumpy landscape that is much more challenging than linear PDEs."
            },
            "description": "PCA-aligned landscape based on full training trajectory. The physics loss landscape is expected to be extremely complex and bumpy due to the nonlinear nature of Burgers equation, with shock waves creating regions of high gradient variability."
        }
    }

    # --- 1) MSE with train ---
    mse_trained_model, mse_base_w, mse_traj_w, mse_traj_ep, mse_dirs, mse_scale, mse_stats = train_and_record("mse", train_epochs=100)
    mse_with_dir = run_dir("mse", "with_train")
    os.makedirs(mse_with_dir, exist_ok=True)
    generate_landscape(
        model=mse_trained_model,
        # Use deterministic train loader so Train Loss (epoch-end eval) and Landscape Loss are comparable
        # without extra noise from batch shuffling.
        data_loader=landscape_train_loader,
        loss_fn=mse_loss_fn,
        loss_name="mse",
        output_dir=mse_with_dir,
        device=device,
        grid_size_1d=100,
        grid_size_2d=30,
        grid_size_3d=16,
        metadata_info={**base_metadata, "loss_type": "MSE", "train_mode": "with_train"},
        trajectory_weights=mse_traj_w,
        trajectory_epochs=mse_traj_ep,
        trajectory_base_weights=mse_base_w,
        pca_directions=mse_dirs,
        pca_range_scale=mse_scale,
        training_stats=mse_stats
    )

    # --- 2) MSE no train ---
    mse_no_model = BurgersEquationModel(hidden_sizes=[256, 256, 128, 64]).to(device)
    mse_no_model.load_state_dict(base_state, strict=True)
    mse_no_dir = run_dir("mse", "no_train")
    os.makedirs(mse_no_dir, exist_ok=True)
    generate_landscape(
        model=mse_no_model,
        data_loader=landscape_train_loader,
        loss_fn=mse_loss_fn,
        loss_name="mse",
        output_dir=mse_no_dir,
        device=device,
        grid_size_1d=100,
        grid_size_2d=30,
        grid_size_3d=16,
        metadata_info={**base_metadata, "loss_type": "MSE", "train_mode": "no_train"},
        trajectory_weights=None,
        trajectory_epochs=None,
        trajectory_base_weights=None,
        pca_directions=None,
        pca_range_scale=None
    )

    # --- 3) Physics with train ---
    phy_trained_model, phy_base_w, phy_traj_w, phy_traj_ep, phy_dirs, phy_scale, phy_stats = train_and_record("physics", train_epochs=100)
    phy_with_dir = run_dir("physics", "with_train")
    os.makedirs(phy_with_dir, exist_ok=True)
    generate_landscape(
        model=phy_trained_model,
        data_loader=landscape_train_loader,
        loss_fn=mse_loss_fn,
        loss_name="physics",
        output_dir=phy_with_dir,
        device=device,
        grid_size_1d=100,
        grid_size_2d=30,
        grid_size_3d=16,
        metadata_info={**base_metadata, "loss_type": "Physics", "train_mode": "with_train"},
        trajectory_weights=phy_traj_w,
        trajectory_epochs=phy_traj_ep,
        trajectory_base_weights=phy_base_w,
        pca_directions=phy_dirs,
        pca_range_scale=phy_scale,
        training_stats=phy_stats
    )

    # --- 4) Physics no train ---
    phy_no_model = BurgersEquationModel(hidden_sizes=[256, 256, 128, 64]).to(device)
    phy_no_model.load_state_dict(base_state, strict=True)
    phy_no_dir = run_dir("physics", "no_train")
    os.makedirs(phy_no_dir, exist_ok=True)
    generate_landscape(
        model=phy_no_model,
        data_loader=landscape_train_loader,
        loss_fn=mse_loss_fn,
        loss_name="physics",
        output_dir=phy_no_dir,
        device=device,
        grid_size_1d=100,
        grid_size_2d=30,
        grid_size_3d=16,
        metadata_info={**base_metadata, "loss_type": "Physics", "train_mode": "no_train"},
        trajectory_weights=None,
        trajectory_epochs=None,
        trajectory_base_weights=None,
        pca_directions=None,
        pca_range_scale=None
    )
    
    logger.info("\nAll Done! Check outputs/")
    logger.info("Generated 4 runs:")
    logger.info(f" - {mse_with_dir}")
    logger.info(f" - {mse_no_dir}")
    logger.info(f" - {phy_with_dir}")
    logger.info(f" - {phy_no_dir}")

if __name__ == "__main__":
    main()
