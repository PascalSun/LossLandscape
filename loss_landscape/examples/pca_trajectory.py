"""
PCA降维轨迹示例：使用PCA自动找到MNIST MLP权重变动最大的维度
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
    
    indices = torch.arange(2000)
    subset_dataset = torch.utils.data.Subset(train_dataset, indices)
    train_loader = DataLoader(
        subset_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0
    )
    
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
    
    # 创建存储实例
    storage = LandscapeStorage("output/mnist_pca_trajectory_example.landscape", mode='create')
    
    # 使用Explorer记录训练轨迹（PCA模式）
    logger.info("Training model and recording trajectory (PCA mode)...")
    with Explorer(model, loss_fn, eval_loader, device=device, storage=storage) as explorer:
        # 设置PCA模式
        explorer.set_trajectory_mode('pca')
        
        # 训练循环
        num_epochs = 15
        for epoch in range(num_epochs):
            # 训练
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            
            # 记录权重位置（PCA模式会自动收集所有权重）
            explorer.log_position(epoch=epoch, verbose=False)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}")
        
        # 使用PCA降维生成轨迹
        logger.info("\nBuilding trajectory with PCA...")
        trajectory = explorer.build_trajectory(mode='pca')
        
        logger.info(f"Trajectory generated with {len(trajectory['epochs'])} points")
        logger.info("PCA found the two dimensions with maximum variance")
    
    storage.close()
    
    # 导出为前端格式
    frontend_data = storage.export_for_frontend("output/mnist_pca_trajectory_example.json")
    
    logger.info("\n" + "="*60)
    logger.info("Training completed!")
    logger.info(f"Data saved to: output/mnist_pca_trajectory_example.landscape")
    logger.info(f"Frontend format saved to: output/mnist_pca_trajectory_example.json")
    logger.info("="*60)


if __name__ == "__main__":
    import os
    os.makedirs("output", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    main()
