"""
高级Loss计算示例：使用MNIST和MLP，展示自定义loss函数、数据增强、Mixup等功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from loss_landscape import Explorer, LandscapeStorage
from loguru import logger
import numpy as np


class MLP(nn.Module):
    """多层感知机，用于MNIST分类"""
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


def mixup_data(x, y, alpha=0.2):
    """Mixup数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup损失计算"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def label_smoothing_loss(outputs, targets, smoothing=0.1):
    """Label smoothing"""
    log_probs = F.log_softmax(outputs, dim=1)
    n_classes = outputs.size(1)
    with torch.no_grad():
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(smoothing / (n_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
    return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


def temperature_scaling(outputs, temperature=2.0):
    """Temperature scaling"""
    return outputs / temperature


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
    
    # 定义基础损失函数
    base_loss_fn = nn.CrossEntropyLoss()
    
    # 示例1: 使用pre_batch_hook进行Mixup数据增强
    logger.info("Example 1: Using Mixup data augmentation...")
    
    def mixup_hook(batch):
        """Mixup钩子函数"""
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, targets = batch
            mixed_x, y_a, y_b, lam = mixup_data(inputs, targets, alpha=0.2)
            return mixed_x, (y_a, y_b, lam)
        return batch
    
    def custom_loss_with_mixup(model, inputs, targets_tuple):
        """自定义loss函数，支持Mixup"""
        outputs = model(inputs)
        y_a, y_b, lam = targets_tuple
        return mixup_criterion(base_loss_fn, outputs, y_a, y_b, lam)
    
    storage = LandscapeStorage("output/mnist_mixup_example.landscape", mode='create')
    
    with Explorer(
        model,
        base_loss_fn,
        data_loader,
        device=device,
        storage=storage,
        custom_loss_fn=custom_loss_with_mixup,
        pre_batch_hook=mixup_hook,
        model_mode='train'  # Mixup通常在train模式下使用
    ) as explorer:
        result1 = explorer.build_surface(
            grid_size=20,
            range_scale=0.1,
            verbose=False
        )
    
    storage.close()
    logger.info(f"Mixup baseline loss: {result1['baseline_loss']:.6f}")
    
    # 示例2: 使用post_batch_hook进行Label Smoothing
    logger.info("\nExample 2: Using Label Smoothing...")
    
    def label_smoothing_hook(outputs, inputs, targets):
        """Label smoothing钩子"""
        return label_smoothing_loss(outputs, targets, smoothing=0.1)
    
    storage_ls = LandscapeStorage("output/mnist_label_smoothing_example.landscape", mode='create')
    
    with Explorer(
        model,
        base_loss_fn,
        data_loader,
        device=device,
        storage=storage_ls,
        post_batch_hook=label_smoothing_hook
    ) as explorer:
        result2 = explorer.build_surface(
            grid_size=20,
            range_scale=0.1,
            verbose=False
        )
    
    storage_ls.close()
    logger.info(f"Label Smoothing baseline loss: {result2['baseline_loss']:.6f}")
    
    # 示例3: 使用custom_loss_fn完全自定义（包含正则化、温度缩放等）
    logger.info("\nExample 3: Fully custom loss function...")
    
    def fully_custom_loss(model, inputs, targets):
        """完全自定义loss：包含温度缩放、正则化等"""
        outputs = model(inputs)
        
        # 温度缩放
        outputs_scaled = temperature_scaling(outputs, temperature=2.0)
        
        # 基础loss
        loss = base_loss_fn(outputs_scaled, targets)
        
        # L2正则化
        l2_reg = 0.01 * sum(p.norm()**2 for p in model.parameters() if p.requires_grad)
        
        return loss + l2_reg
    
    storage_custom = LandscapeStorage("output/mnist_custom_loss_example.landscape", mode='create')
    
    with Explorer(
        model,
        base_loss_fn,
        data_loader,
        device=device,
        storage=storage_custom,
        custom_loss_fn=fully_custom_loss
    ) as explorer:
        result3 = explorer.build_surface(
            grid_size=20,
            range_scale=0.1,
            verbose=False
        )
    
    storage_custom.close()
    logger.info(f"Custom loss baseline: {result3['baseline_loss']:.6f}")
    
    # 示例4: 使用train模式（影响Dropout和BatchNorm）
    logger.info("\nExample 4: Using train mode (affects Dropout/BatchNorm)...")
    
    storage_train = LandscapeStorage("output/mnist_train_mode_example.landscape", mode='create')
    
    with Explorer(
        model,
        base_loss_fn,
        data_loader,
        device=device,
        storage=storage_train,
        model_mode='train'  # 使用train模式
    ) as explorer:
        result4 = explorer.build_surface(
            grid_size=20,
            range_scale=0.1,
            verbose=False
        )
    
    storage_train.close()
    logger.info(f"Train mode baseline loss: {result4['baseline_loss']:.6f}")
    
    # 比较所有方法
    logger.info("\n" + "="*60)
    logger.info("Comparison:")
    logger.info(f"  Mixup:           {result1['baseline_loss']:.6f}")
    logger.info(f"  Label Smoothing: {result2['baseline_loss']:.6f}")
    logger.info(f"  Custom Loss:     {result3['baseline_loss']:.6f}")
    logger.info(f"  Train Mode:      {result4['baseline_loss']:.6f}")
    logger.info("="*60)


if __name__ == "__main__":
    import os
    os.makedirs("output", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    main()
