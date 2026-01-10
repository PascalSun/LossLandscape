# Loss Landscape 可视化平台

[English](./README.md) | 中文

一个用于可视化和分析深度学习模型损失曲面的综合平台。通过交互式 1D、2D 和 3D 可视化来理解模型的优化行为。

## ✨ 功能特性

- **多维度可视化**：支持 1D（曲线）、2D（曲面）和 3D（体积）损失曲面
- **训练轨迹追踪**：记录并可视化训练过程中的优化路径
- **Hessian 分析**：计算特征值谱、迹和锐度指标
- **PCA 对齐方向**：基于训练轨迹自动选择方向
- **交互式 Web 界面**：美观、现代的可视化界面
- **简洁的 API**：只需几行代码即可生成损失曲面

## 🚀 快速开始

### 安装

```bash
# 安装 Python SDK
pip install -e .

# 或使用 uv
uv pip install -e .
```

### 基本用法

```python
import torch.nn as nn
from loss_landscape import sample_landscape

# 定义损失函数: (model, inputs, targets) -> loss
def loss_fn(model, inputs, targets):
    return nn.MSELoss()(model(inputs), targets)

# 一行代码生成损失曲面
sample_landscape(model, data_loader, loss_fn, "./landscape.json")
```

### 使用 Writer 接口（推荐）

```python
import torch.nn as nn
from loss_landscape import LossLandscapeWriter

# 定义损失函数: (model, inputs, targets) -> loss
def loss_fn(model, inputs, targets):
    return nn.MSELoss()(model(inputs), targets)

# 创建 writer
writer = LossLandscapeWriter("./runs/experiment1")

# 生成 2D 损失曲面
writer.sample_landscape(model, data_loader, loss_fn, grid_size=50)

# 关闭并导出
writer.close()
```

### 记录训练轨迹

```python
from loss_landscape import LossLandscapeWriter

writer = LossLandscapeWriter("./runs/training")

# 训练循环
for epoch in range(100):
    train_loss = train_one_epoch(model, ...)
    writer.record_checkpoint(model, epoch, train_loss=train_loss)

# 构建轨迹可视化
writer.build_trajectory(model, data_loader, loss_fn)
writer.sample_landscape(model, data_loader, loss_fn)
writer.close()
```

### 带正则化的损失函数

```python
import torch.nn as nn
from loss_landscape import LossLandscapeWriter

def loss_with_reg(model, inputs, targets):
    outputs = model(inputs)
    data_loss = nn.MSELoss()(outputs, targets)
    l2_reg = 0.01 * sum(p.norm()**2 for p in model.parameters())
    return data_loss + l2_reg

writer = LossLandscapeWriter("./runs/regularized")
writer.sample_landscape(model, data_loader, loss_with_reg)
writer.close()
```

### Physics-Informed 损失函数

```python
import torch.nn as nn
from loss_landscape import LossLandscapeWriter

def physics_loss(model, inputs, targets):
    outputs = model(inputs)
    data_loss = nn.MSELoss()(outputs, targets)
    physics_residual = compute_pde_residual(model, inputs)
    return data_loss + 0.1 * physics_residual

writer = LossLandscapeWriter("./runs/pinn")
writer.sample_landscape(model, data_loader, physics_loss)
writer.close()
```

## 🖥️ Web 可视化

### 启动开发服务器

```bash
cd web
npm install
npm run dev
```

打开 http://localhost:3000 查看交互式可视化界面。

### 功能

- **曲面图**：可旋转缩放的交互式 3D 曲面
- **等高线图**：带轨迹叠加的 2D 等高线可视化
- **Hessian 分析**：特征值谱密度和锐度指标
- **多语言支持**：中英文界面

## 📁 项目结构

```
LossLandscape/
├── loss_landscape/          # Python SDK
│   ├── core/               # 核心模块
│   │   ├── explorer.py     # 损失曲面计算
│   │   ├── storage.py      # 数据持久化（DuckDB）
│   │   ├── writer.py       # 高级 API
│   │   └── hessian.py      # Hessian 分析
│   ├── examples/           # 示例脚本
│   └── cli.py              # 命令行接口
├── web/                    # Next.js 前端
│   └── src/
│       ├── app/            # React 组件
│       └── lib/            # 工具函数
└── pyproject.toml          # Python 包配置
```

## 🔧 CLI 命令

```bash
# 查看损失曲面数据信息
losslandscape view -i ./landscape.json

# 运行完整示例
losslandscape example
```

## 📊 输出格式

生成的 JSON 文件包含以下结构：

```json
{
  // === 2D 曲面数据（主要可视化） ===
  "X": [[0.0, 0.0, ...], [0.1, 0.1, ...]],           // X 坐标网格 (grid_size x grid_size)
  "Y": [[0.0, 0.1, ...], [0.0, 0.1, ...]],           // Y 坐标网格 (grid_size x grid_size)
  "loss_grid_2d": [[1.2, 1.1, ...], [1.3, 1.0, ...]], // Loss 值 (grid_size x grid_size)
  "baseline_loss": 0.5,                              // 原点处的 Loss（当前模型权重）
  "grid_size": 50,                                   // 网格分辨率
  "mode": "1d+2d",                                   // 数据模式: "1d", "2d", "1d+2d"

  // === 1D 线条数据（可选） ===
  "X_1d": [-0.5, -0.4, ..., 0.4, 0.5],              // 1D 线条的 X 坐标
  "loss_line_1d": [2.1, 1.8, ..., 1.9, 2.2],        // 沿 1D 线条的 Loss 值
  "baseline_loss_1d": 0.5,                          // 1D 基准 Loss
  "grid_size_1d": 100,                              // 1D 网格分辨率

  // === 3D 体积数据（可选） ===
  "Z": [[[...]]],                                   // Z 坐标 (nx x ny x nz)
  "loss_grid_3d": [[[...]]],                        // 3D Loss 体积 (nx x ny x nz)
  "volume_x": [-0.5, -0.4, ...],                    // X 轴值
  "volume_y": [-0.5, -0.4, ...],                    // Y 轴值
  "volume_z": [-0.5, -0.4, ...],                    // Z 轴值

  // === 训练轨迹（可选） ===
  "trajectory_data": {
    "traj_1": [0.0, 0.1, 0.15, ...],               // 轨迹的 X 坐标
    "traj_2": [0.0, -0.05, -0.1, ...],             // 轨迹的 Y 坐标
    "traj_3": [0.0, 0.02, 0.03, ...],              // Z 坐标（3D 用）
    "epochs": [0, 1, 2, ...],                       // Epoch 编号
    "losses": [1.5, 1.2, 0.9, ...],                // 每个 epoch 的训练 Loss
    "val_losses": [1.6, 1.3, 1.0, ...]             // 验证 Loss（可选）
  },

  // === Hessian 分析（可选） ===
  "hessian": {
    "epochs": [0, 10, 20, ...],                     // 计算 Hessian 的 epoch
    "max_eigenvalue": [150.5, 80.2, 45.1, ...],    // 最大特征值（锐度）
    "trace": [1200.0, 800.0, 500.0, ...],          // Hessian 迹
    "top_eigenvalues": [[150.5, 120.3, ...], ...]  // 每个 epoch 的 Top-k 特征值
  },

  // === 元数据（可选） ===
  "metadata": {
    "model_info": { ... },                          // 模型架构信息
    "training_statistics": {                        // 训练统计
      "initial_loss": 1.5,
      "final_loss": 0.1,
      "min_loss": 0.08,
      "min_loss_epoch": 95,
      "learning_rate_history": [0.001, 0.001, ...]
    },
    "loss_description": { ... },                    // Loss 函数描述
    "system_info": { ... }                          // 硬件/系统信息
  }
}
```

## 📚 参考文献

- Li et al., "Visualizing the Loss Landscape of Neural Nets" (NeurIPS 2018)
- Ghorbani et al., "An Investigation into Neural Net Optimization via Hessian Eigenvalue Density" (ICML 2019)

## 📄 许可证

MIT 许可证
