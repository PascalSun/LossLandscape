"""
正则化示例：使用MNIST和MLP，展示L2正则化对Loss Landscape的影响
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from loss_landscape import Explorer, LandscapeStorage
from loguru import logger


# 定义MLP模型
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


def l2_regularizer(model: nn.Module) -> torch.Tensor:
    """
    L2正则化：计算所有参数的L2范数的平方和
    
    Args:
        model: PyTorch模型
        
    Returns:
        正则化项的标量Tensor
    """
    return sum(p.norm()**2 for p in model.parameters() if p.requires_grad)


def l1_regularizer(model: nn.Module) -> torch.Tensor:
    """
    L1正则化：计算所有参数的L1范数的和
    
    Args:
        model: PyTorch模型
        
    Returns:
        正则化项的标量Tensor
    """
    return sum(p.abs().sum() for p in model.parameters() if p.requires_grad)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 加载MNIST数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
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
    
    # 定义损失函数
    loss_fn = nn.CrossEntropyLoss()
    
    # 创建存储实例
    storage = LandscapeStorage("output/mnist_regularizer_example.landscape", mode='create')
    
    # 使用Explorer生成表面（带L2正则化）
    logger.info("Generating Loss Landscape surface with L2 regularizer...")
    
    # 定义正则化权重（lambda）
    lambda_reg = 0.01
    
    with Explorer(
        model, 
        loss_fn, 
        data_loader, 
        device=device,
        storage=storage,
        regularizer=l2_regularizer,
        regularizer_weight=lambda_reg
    ) as explorer:
        result = explorer.build_surface(
            grid_size=30,  # 30x30网格
            range_scale=0.1,
            verbose=True
        )
    
    storage.close()
    
    # 导出为前端格式
    frontend_data = storage.export_for_frontend("output/mnist_regularizer_example.json")
    
    logger.info("Generation completed!")
    logger.info(f"Grid size: {result['grid_size']}x{result['grid_size']}")
    logger.info(f"Baseline loss (with L2 reg, lambda={lambda_reg}): {result['baseline_loss']:.6f}")
    loss_min = min(map(min, result['loss_grid_2d']))
    loss_max = max(map(max, result['loss_grid_2d']))
    logger.info(f"Loss range: [{loss_min:.6f}, {loss_max:.6f}]")
    logger.info(f"Data saved to: output/mnist_regularizer_example.landscape")
    logger.info(f"Frontend format saved to: output/mnist_regularizer_example.json")
    
    # 比较：不带正则化的情况
    logger.info("\n" + "="*60)
    logger.info("Comparing with no regularizer...")
    
    storage_no_reg = LandscapeStorage("output/mnist_no_regularizer_example.landscape", mode='create')
    
    with Explorer(model, loss_fn, data_loader, device=device, storage=storage_no_reg) as explorer:
        result_no_reg = explorer.build_surface(
            grid_size=30,
            range_scale=0.1,
            verbose=False
        )
    
    storage_no_reg.close()
    
    logger.info(f"Baseline loss (no regularizer): {result_no_reg['baseline_loss']:.6f}")
    logger.info(f"Difference: {result['baseline_loss'] - result_no_reg['baseline_loss']:.6f}")
    logger.info("="*60)


if __name__ == "__main__":
    import os
    os.makedirs("output", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    main()
