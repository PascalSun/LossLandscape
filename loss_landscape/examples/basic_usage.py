"""
基本使用示例：使用MNIST数据集和MLP模型生成Loss Landscape表面
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from loss_landscape import Explorer, LandscapeStorage
from loguru import logger


# 定义MLP模型用于MNIST分类
class MLP(nn.Module):
    """多层感知机，用于MNIST手写数字分类"""
    def __init__(self, input_size=784, hidden_sizes=[128, 64], num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # Flatten the input (batch_size, 1, 28, 28) -> (batch_size, 784)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 加载MNIST数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # 使用测试集的一部分来快速计算loss landscape
    # 在实际应用中，可以使用训练集或验证集
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # 只使用前1000个样本以加快计算速度
    indices = torch.arange(1000)
    subset_dataset = torch.utils.data.Subset(test_dataset, indices)
    data_loader = DataLoader(
        subset_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f"Loaded {len(subset_dataset)} samples from MNIST test set")
    
    # 创建MLP模型
    model = MLP(input_size=784, hidden_sizes=[128, 64], num_classes=10)
    model = model.to(device)
    
    # 初始化模型权重（可以加载预训练权重）
    # 这里使用随机初始化的权重作为示例
    logger.info("Model initialized with random weights")
    
    # 定义损失函数
    loss_fn = nn.CrossEntropyLoss()
    
    # 创建输出目录结构：outputs/run_xxx/
    import os
    from datetime import datetime
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = f"outputs/{run_name}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # 创建存储实例
    landscape_path = f"{output_dir}/mnist_basic_example.landscape"
    storage = LandscapeStorage(landscape_path, mode="create")
    
    # 使用Explorer生成2D表面 + 3D体积
    logger.info("Generating 2D Loss Landscape surface for MNIST MLP...")
    with Explorer(model, loss_fn, data_loader, device=device, storage=storage) as explorer:
        result_2d = explorer.build_surface(
            grid_size=30,  # 30x30网格（较小以便快速测试）
            range_scale=0.1,
            verbose=True,
        )

        logger.info("Generating 3D Loss Volume (smaller grid for speed)...")
        # 3D体积计算开销较大，这里用更小的grid_size示例
        result_3d = explorer.build_volume(
            grid_size=16,
            range_scale=0.1,
            verbose=True,
        )
    
    # 导出为前端格式（在关闭连接之前），会同时包含2D和3D数据
    json_path = f"{output_dir}/mnist_basic_example.json"
    frontend_data = storage.export_for_frontend(json_path)
    
    # 关闭存储连接
    storage.close()
    
    logger.info("=" * 60)
    logger.info("Generation completed!")
    logger.info(f"[2D] Grid size: {result_2d['grid_size']}x{result_2d['grid_size']}")
    logger.info(f"[2D] Baseline loss: {result_2d['baseline_loss']:.6f}")
    loss_min = min(map(min, result_2d['loss_grid_2d']))
    loss_max = max(map(max, result_2d['loss_grid_2d']))
    logger.info(f"[2D] Loss range: [{loss_min:.6f}, {loss_max:.6f}]")
    logger.info(f"[3D] Grid size: {result_3d['grid_size']}^3")
    logger.info(f"[3D] Baseline loss: {result_3d['baseline_loss']:.6f}")
    logger.info(f"Data saved to: {landscape_path}")
    logger.info(f"Frontend format (2D+3D) saved to: {json_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
