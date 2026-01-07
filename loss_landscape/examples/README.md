# LossLandscape 示例代码

本目录包含使用 MNIST 数据集和 MLP 模型的完整示例。

## 示例列表

### 1. `basic_usage.py` - 基本使用示例

最简单的使用示例，展示如何生成 Loss Landscape 表面。

```bash
uv run python examples/basic_usage.py
```

**功能：**
- 加载 MNIST 测试集
- 创建 MLP 模型
- 生成 30x30 的 Loss Landscape 表面
- 保存为 `.landscape` 和 `.json` 文件

### 2. `with_regularizer.py` - 正则化示例

展示如何使用 L2 正则化，并比较有无正则化的差异。

```bash
uv run python examples/with_regularizer.py
```

**功能：**
- 使用 L2 正则化生成 landscape
- 不使用正则化生成 landscape
- 比较两者的差异

### 3. `training_trajectory.py` - 训练轨迹示例

在训练过程中记录权重轨迹，展示训练路径在 Loss Landscape 上的移动。

```bash
uv run python examples/training_trajectory.py
```

**功能：**
- 训练 MLP 模型 10 个 epoch
- 记录每个 epoch 的权重位置
- 生成训练轨迹数据

### 4. `pca_trajectory.py` - PCA 降维轨迹示例

使用 PCA 自动找到权重变动最大的两个维度，用于轨迹可视化。

```bash
uv run python examples/pca_trajectory.py
```

**功能：**
- 训练模型 15 个 epoch
- 使用 PCA 降维找到主要变动方向
- 生成基于 PCA 的轨迹数据

### 5. `advanced_loss_computation.py` - 高级 Loss 计算示例

展示各种高级功能：Mixup、Label Smoothing、Temperature Scaling 等。

```bash
uv run python examples/advanced_loss_computation.py
```

**功能：**
- Mixup 数据增强
- Label Smoothing
- Temperature Scaling + 正则化
- Train/Eval 模式对比

## 模型架构

所有示例使用相同的 MLP 架构：

```python
MLP(
    input_size=784,      # MNIST 图像: 28x28 = 784
    hidden_sizes=[128, 64],
    num_classes=10        # MNIST 有 10 个类别 (0-9)
)
```

## 数据集

所有示例使用 **MNIST** 手写数字数据集：
- 训练集：60,000 张图像
- 测试集：10,000 张图像
- 图像大小：28x28 灰度图像
- 类别数：10（数字 0-9）

示例中为了加快计算速度，只使用了数据集的一部分：
- 训练示例：使用 2000 个训练样本
- 评估示例：使用 1000 个测试样本

## 输出文件

所有示例会在 `output/` 目录下生成：
- `.landscape` 文件：DuckDB 格式的数据文件
- `.json` 文件：前端可用的 JSON 格式

## 运行方式

直接运行Python模块：

```bash
uv run python -m loss_landscape.examples.basic_usage
uv run python -m loss_landscape.examples.with_regularizer
uv run python -m loss_landscape.examples.training_trajectory
uv run python -m loss_landscape.examples.pca_trajectory
uv run python -m loss_landscape.examples.advanced_loss_computation
```

## 运行要求

确保已安装所有依赖：

```bash
uv sync
```

## 注意事项

1. **首次运行**：MNIST 数据集会自动下载到项目根目录的 `data/` 目录
2. **计算时间**：30x30 网格在 CPU 上大约需要几分钟，GPU 会快很多
3. **内存使用**：MLP 模型较小，内存占用不大
4. **输出目录**：输出文件会保存在项目根目录的 `output/` 目录

## 自定义

你可以修改示例代码来：
- 使用不同的模型架构
- 使用不同的数据集
- 调整网格大小（`grid_size`）
- 调整扰动范围（`range_scale`）
- 添加自定义的正则化函数
- 使用自定义的 loss 计算逻辑

