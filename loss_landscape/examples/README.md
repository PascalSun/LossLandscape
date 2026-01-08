# LossLandscape 完整示例（1D Burgers方程）

本目录包含一个综合示例，展示 Loss Landscape 的核心功能，并对比 **MSE loss** 与 **Physics-Informed loss** 的 landscape 差异。

**注意**：本示例使用1D Burgers方程（非线性PDE，包含对流项），会产生**极其复杂、坑坑洼洼**的physics loss landscape。Burgers方程包含非线性项u·u_x，会产生激波，使得loss landscape高度非凸，包含多个局部最小值、鞍点和狭窄的谷底，比简单的波动方程或重力公式更有代表性。

## 快速开始

运行完整示例（推荐方式）：

```bash
lossvis example
```

或者使用 Python 模块方式：

```bash
uv run python -m loss_landscape.examples.complete_example
```

或者直接运行文件：

```bash
uv run python loss_landscape/examples/complete_example.py
```

## 示例功能

`complete_example.py` 展示了以下所有功能：

### 1. 2D Loss Landscape 表面
- 生成 30x30 的 2D Loss Landscape 表面（α–β）

### 2. 3D Loss Volume 体积
- 生成 16x16x16 的 3D Loss Volume
- 支持 3D 切片和体积渲染

### 3. 训练轨迹记录
- 对 **with_train** 的 run：
  - 记录初始位置（epoch=-1，训练前）
  - 记录每个训练 epoch 的位置
  - 用轨迹做 PCA，得到与轨迹对齐的 α/β/γ 方向
- 对 **no_train** 的 run：
  - 不训练，不导出轨迹数据（用于对比：仅看初始权重附近的 landscape）

### 4. 自定义 Loss 函数
- MSE loss：标准均方误差
- Physics loss：MSE + Burgers方程残差项（\(u_t + u \cdot u_x - \nu \cdot u_{xx} = 0\)），使用自动微分计算偏导数
  - **极其复杂**：非线性对流项u·u_x使得loss landscape高度非凸
  - **多个局部最小值**：激波区域产生多个局部最优解
  - **鞍点和狭窄谷底**：非线性项导致梯度变化剧烈
  - **比线性PDE复杂得多**：这是真正的非线性PDE，loss landscape会非常坑坑洼洼

### 5. PCA 模式（可选）
- 如需使用 PCA 模式，在训练前设置：`explorer.set_trajectory_mode('pca')`
- 然后重新训练并调用：`explorer.build_trajectory(mode='pca')`

## 模型与数据集

- **任务**：用神经网络求解1D Burgers方程 \(u_t + u \cdot u_x = \nu \cdot u_{xx}\)
- **输入**：\([x, t]\) (空间坐标和时间)
- **输出**：\(u(x,t)\) (Burgers方程的解)
- **真实解**：使用行波激波解 \(u(x, t) = 1 - \tanh((x - t - x_0) / (2\nu))\) 生成数据，确保数据满足方程
- **PDE特点**：
  - **非线性对流项** \(u \cdot u_x\)：会产生激波，使问题高度非线性
  - **扩散项** \(\nu \cdot u_{xx}\)：粘性项，\(\nu = 0.05\)
  - **时间导数** \(u_t\)：时间演化
- **训练集/评估集**：示例中默认 train=2000, eval=500
- **模型架构**：更大的网络 [256, 256, 128, 64]，使用SiLU激活函数
- **特点**：
  - Physics loss涉及一阶和二阶偏导数，以及**非线性项**
  - 会产生**极其复杂、坑坑洼洼**的loss landscape
  - 包含多个局部最小值、鞍点、狭窄谷底

## 输出文件

示例会在 `outputs/` 下生成 4 个 **独立 run 目录**：

- `run_YYYYMMDD_HHMMSS_mse_with_train/`
- `run_YYYYMMDD_HHMMSS_mse_no_train/`
- `run_YYYYMMDD_HHMMSS_physics_with_train/`
- `run_YYYYMMDD_HHMMSS_physics_no_train/`

每个目录里包含：
- `complete_example.landscape`：DuckDB 格式的数据文件
- `complete_example.json`：前端可用的 JSON（包含 2D、3D；with_train 还包含轨迹数据）

## 运行要求

确保已安装所有依赖：

```bash
uv sync
```

## 注意事项

1. **计算时间**：
   - 2D surface (30x30)：约 1-2 分钟（CPU）
   - 3D volume (16x16x16)：约 5-10 分钟（CPU）
   - GPU 会显著加快速度
2. **输出目录**：输出文件会保存在项目根目录的 `outputs/` 目录

## 自定义

你可以修改示例代码来：
- 使用不同的模型架构
- 使用不同的数据集
- 调整网格大小（`grid_size`）
- 调整扰动范围（`range_scale`）
- 修改正则化权重（`regularizer_weight`）
- 调整训练 epoch 数量
- 使用不同的优化器和学习率
