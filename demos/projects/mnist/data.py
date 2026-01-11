"""
Data loading utilities for MNIST.
"""

from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from demos.base import registry


@registry.register_dataset("mnist")
def get_mnist_loaders(
    config: Dict[str, Any],
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """
    Get MNIST data loaders.

    Args:
        config: Configuration dictionary with dataset settings

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    dataset_config = config.get("dataset", {})

    batch_size = dataset_config.get("batch_size", 128)
    num_workers = dataset_config.get("num_workers", 4)
    data_dir = dataset_config.get("data_dir", "./data")
    val_split = dataset_config.get("val_split", 0.1)  # 10% for validation
    subset = dataset_config.get("subset", None)  # Optional: use subset of data

    # Transforms
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
        ]
    )

    # Download and load datasets
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)

    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    # Optional: use subset for faster experiments
    if subset is not None and subset < len(train_dataset):
        indices = torch.randperm(len(train_dataset))[:subset]
        train_dataset = Subset(train_dataset, indices)

    # Split train into train/val
    val_loader = None
    if val_split > 0:
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def get_mnist_info() -> Dict[str, Any]:
    """Get information about the MNIST dataset."""
    return {
        "name": "MNIST",
        "description": "Handwritten digit classification (0-9)",
        "num_classes": 10,
        "input_shape": (1, 28, 28),
        "train_samples": 60000,
        "test_samples": 10000,
        "mean": 0.1307,
        "std": 0.3081,
    }
