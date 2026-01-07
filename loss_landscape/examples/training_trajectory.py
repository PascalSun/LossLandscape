"""
训练轨迹示例：在MNIST MLP训练过程中记录权重轨迹
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from loss_landscape import Explorer, LandscapeStorage
from loguru import logger


class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[128, 64], num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_one_epoch(model, data_loader, optimizer, loss_fn, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 加载MNIST数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # 使用训练集的一部分
    indices = torch.arange(2000)
    subset_dataset = torch.utils.data.Subset(train_dataset, indices)
    train_loader = DataLoader(
        subset_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0
    )
    
    # 用于评估的测试集
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    test_indices = torch.arange(500)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)
    eval_loader = DataLoader(
        test_subset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f"Loaded {len(subset_dataset)} training samples and {len(test_subset)} test samples")
    
    # 创建MLP模型
    model = MLP(input_size=784, hidden_sizes=[128, 64], num_classes=10)
    model = model.to(device)
    
    # 定义损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # 创建输出目录结构：outputs/run_xxx/
    import os
    from datetime import datetime
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = f"outputs/{run_name}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    landscape_path = f"{output_dir}/mnist_trajectory_example.landscape"
    storage = LandscapeStorage(landscape_path, mode='create')
    
    # 使用Explorer记录训练轨迹
    logger.info("Training model and recording trajectory...")
    with Explorer(model, loss_fn, eval_loader, device=device, storage=storage) as explorer:
        # 先生成表面（确定方向）
        logger.info("Generating initial 2D surface...")
        result_2d = explorer.build_surface(grid_size=30, range_scale=0.1, verbose=False)
        
        # 生成3D体积
        logger.info("Generating 3D Loss Volume (smaller grid for speed)...")
        # 3D体积计算开销较大，这里用更小的grid_size示例
        result_3d = explorer.build_volume(
            grid_size=16,
            range_scale=0.1,
            verbose=True,
        )
        
        # 训练循环
        num_epochs = 10
        for epoch in range(num_epochs):
            # 训练
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            
            # 记录权重位置
            explorer.log_position(epoch=epoch, verbose=False)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}")
        
        # 生成轨迹数据
        logger.info("\nBuilding trajectory...")
        trajectory = explorer.build_trajectory(mode='fixed')
        
        logger.info(f"Trajectory generated with {len(trajectory['epochs'])} points")
    
    # 导出为前端格式（在关闭连接之前）
    json_path = f"{output_dir}/mnist_trajectory_example.json"
    frontend_data = storage.export_for_frontend(json_path)
    storage.close()
    
    logger.info("\n" + "="*60)
    logger.info("Training completed!")
    logger.info(f"[2D] Grid size: {result_2d['grid_size']}x{result_2d['grid_size']}")
    logger.info(f"[2D] Baseline loss: {result_2d['baseline_loss']:.6f}")
    loss_min = min(map(min, result_2d['loss_grid_2d']))
    loss_max = max(map(max, result_2d['loss_grid_2d']))
    logger.info(f"[2D] Loss range: [{loss_min:.6f}, {loss_max:.6f}]")
    logger.info(f"[3D] Grid size: {result_3d['grid_size']}^3")
    logger.info(f"[3D] Baseline loss: {result_3d['baseline_loss']:.6f}")
    logger.info(f"Data saved to: {landscape_path}")
    logger.info(f"Frontend format (2D+3D+trajectory) saved to: {json_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
