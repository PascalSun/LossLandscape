"""
Data loading utilities for simple regression.
"""

from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from demos.base import registry


@registry.register_dataset("regression")
def get_regression_loaders(
    config: Dict[str, Any],
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """
    Get synthetic regression data loaders.

    Creates random input-output pairs with optional noise.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    dataset_config = config.get("dataset", {})
    seed = config.get("experiment", {}).get("seed", 42)

    num_samples = dataset_config.get("num_samples", 200)
    input_size = dataset_config.get("input_size", 10)
    output_size = dataset_config.get("output_size", 1)
    batch_size = dataset_config.get("batch_size", 32)
    num_workers = dataset_config.get("num_workers", 0)
    noise_std = dataset_config.get("noise_std", 0.1)
    val_split = dataset_config.get("val_split", 0.1)

    torch.manual_seed(seed)

    # Generate synthetic data
    # y = Wx + b + noise
    X = torch.randn(num_samples, input_size)
    W = torch.randn(input_size, output_size) * 0.5
    b = torch.randn(output_size) * 0.1
    y = X @ W + b + torch.randn(num_samples, output_size) * noise_std

    # Create dataset
    train_dataset = TensorDataset(X, y)

    # Generate test data (separate samples)
    torch.manual_seed(seed + 1000)
    X_test = torch.randn(num_samples // 4, input_size)
    y_test = X_test @ W + b + torch.randn(num_samples // 4, output_size) * noise_std
    test_dataset = TensorDataset(X_test, y_test)

    # Split train into train/val
    val_loader = None
    if val_split > 0:
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
