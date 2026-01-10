"""
简单示例：展示 Loss Landscape 的使用

演示三种场景：
1. 一行代码快速生成
2. Writer 接口（带正则化）
3. 训练时记录轨迹
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger


# ========== 示例 1: 一行代码 ==========


def example_1_quick_sample():
    """最简单的用法：一行代码生成 Landscape"""
    from loss_landscape import sample_landscape

    # 创建简单模型和数据
    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))

    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    loader = DataLoader(TensorDataset(X, y), batch_size=32)

    # loss_fn 签名: (model, inputs, targets) -> loss
    def loss_fn(model, inputs, targets):
        return nn.MSELoss()(model(inputs), targets)

    # 一行代码！
    sample_landscape(model, loader, loss_fn, "./outputs/quick_landscape.json")

    logger.info("完成！输出: ./outputs/quick_landscape.json")


# ========== 示例 2: Writer 接口（带正则化）==========


def example_2_with_regularizer():
    """使用 Writer 接口，并添加正则化"""
    from loss_landscape import LossLandscapeWriter

    # 创建模型和数据
    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))

    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    loader = DataLoader(TensorDataset(X, y), batch_size=32)

    # loss_fn 包含正则化
    def loss_with_reg(model, inputs, targets):
        outputs = model(inputs)
        data_loss = nn.MSELoss()(outputs, targets)
        l2_reg = 0.01 * sum(p.norm() ** 2 for p in model.parameters())
        return data_loss + l2_reg

    # 使用 Writer
    writer = LossLandscapeWriter("./outputs/runs/with_reg")

    # 采样
    writer.sample_landscape(model, loader, loss_with_reg, grid_size=30)

    writer.close()
    logger.info("完成！输出: ./outputs/runs/with_reg/landscape.json")


# ========== 示例 3: Physics-Informed Loss ==========


def example_3_custom_loss():
    """使用 Physics-Informed Loss 函数"""
    from loss_landscape import LossLandscapeWriter
    import torch.nn.functional as F

    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))

    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    loader = DataLoader(TensorDataset(X, y), batch_size=32)

    # Physics-Informed Loss：数据 Loss + 物理约束
    def physics_loss(model, inputs, targets):
        outputs = model(inputs)
        data_loss = F.mse_loss(outputs, targets)
        # 模拟物理约束（实际使用中替换为真实的 PDE 残差）
        physics_constraint = 0.01 * outputs.mean() ** 2
        return data_loss + physics_constraint

    writer = LossLandscapeWriter("./outputs/runs/physics_loss")

    # 直接传入 loss_fn
    writer.sample_landscape(model, loader, physics_loss, grid_size=30)

    writer.close()
    logger.info("完成！输出: ./outputs/runs/physics_loss/landscape.json")


# ========== 示例 4: 训练时记录轨迹 ==========


def example_4_training_trajectory():
    """训练时记录轨迹"""
    from loss_landscape import LossLandscapeWriter

    # 创建模型和数据
    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))

    X_train = torch.randn(200, 10)
    y_train = torch.randn(200, 1)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32)

    # loss_fn 签名: (model, inputs, targets) -> loss
    def loss_fn(model, inputs, targets):
        return nn.MSELoss()(model(inputs), targets)

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 创建 Writer
    writer = LossLandscapeWriter("./outputs/runs/training")

    # 训练循环
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = nn.MSELoss()(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}")

        # 记录检查点
        writer.record_checkpoint(model, epoch, train_loss=train_loss)

    # 训练结束后，构建轨迹
    logger.info("构建训练轨迹（PCA 模式）...")
    writer.build_trajectory(model, train_loader, loss_fn)

    # 也可以生成 Landscape
    logger.info("生成 Landscape...")
    writer.sample_landscape(model, train_loader, loss_fn, grid_size=20)

    writer.close()
    logger.info("完成！输出: ./outputs/runs/training/landscape.json")


# ========== 示例 5: 对比实验 ==========


def example_5_comparison():
    """对比两个模型的 Landscape（使用相同方向）"""
    from loss_landscape import LossLandscapeWriter

    # 模型 A: 较浅的网络
    model_a = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 1))

    # 模型 B: 较深的网络
    model_b = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    loader = DataLoader(TensorDataset(X, y), batch_size=32)

    # loss_fn 签名: (model, inputs, targets) -> loss
    def loss_fn(model, inputs, targets):
        return nn.MSELoss()(model(inputs), targets)

    # 实验 A（生成方向）
    writer_a = LossLandscapeWriter("./outputs/runs/comparison/model_a", seed=42)
    writer_a.sample_landscape(model_a, loader, loss_fn, grid_size=20)
    _ = writer_a.get_directions()  # 获取方向（可用于传递给其他实验）
    writer_a.close()
    logger.info("模型 A 完成")

    # 实验 B（使用相同方向）
    # 注意：由于模型参数维度不同，这里的方向不能直接复用
    # 这个示例仅用于演示 API，实际对比需要相同架构的模型
    writer_b = LossLandscapeWriter("./outputs/runs/comparison/model_b")
    writer_b.sample_landscape(model_b, loader, loss_fn, grid_size=20)
    # 如果要复用方向：writer_b.sample_landscape(..., directions=directions)
    writer_b.close()
    logger.info("模型 B 完成")

    logger.info("对比实验完成！")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("示例 1: 一行代码")
    logger.info("=" * 60)
    example_1_quick_sample()

    logger.info("\n" + "=" * 60)
    logger.info("示例 2: Writer + 正则化")
    logger.info("=" * 60)
    example_2_with_regularizer()

    logger.info("\n" + "=" * 60)
    logger.info("示例 3: Physics-Informed Loss")
    logger.info("=" * 60)
    example_3_custom_loss()

    logger.info("\n" + "=" * 60)
    logger.info("示例 4: 训练轨迹")
    logger.info("=" * 60)
    example_4_training_trajectory()

    logger.info("\n" + "=" * 60)
    logger.info("示例 5: 对比实验")
    logger.info("=" * 60)
    example_5_comparison()
