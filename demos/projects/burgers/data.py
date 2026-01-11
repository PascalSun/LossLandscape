"""
Data loading utilities for Burgers equation.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from demos.base import registry


class BurgersEquationDataset(Dataset):
    """
    Dataset based on 1D Burgers equation analytical solution.

    Uses the traveling wave solution:
    u(x, t) = 1 - tanh((x - t - x0) / (2*nu))

    This represents a shock wave moving to the right.
    """

    def __init__(
        self,
        num_samples: int = 2000,
        nu: float = 0.05,
        x_range: Tuple[float, float] = (0, 2),
        t_range: Tuple[float, float] = (0, 1),
        noise_std: float = 0.01,
        seed: Optional[int] = None,
    ):
        """
        Args:
            num_samples: Number of samples
            nu: Viscosity coefficient (smaller = sharper shock, harder to train)
            x_range: Spatial domain range
            t_range: Time domain range
            noise_std: Gaussian noise standard deviation
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.nu = nu

        if seed is not None:
            np.random.seed(seed)

        # Generate random sampling points
        x = np.random.uniform(x_range[0], x_range[1], num_samples)
        t = np.random.uniform(t_range[0], t_range[1], num_samples)

        # Analytical solution (traveling wave)
        x0 = 0.5  # Initial shock position
        arg = (x - t - x0) / (2 * nu)
        u_true = 1.0 - np.tanh(arg)

        # Add noise
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, num_samples)
            u_true = u_true + noise

        self.inputs = torch.FloatTensor(np.column_stack([x, t]))
        self.targets = torch.FloatTensor(u_true.reshape(-1, 1))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


@registry.register_dataset("burgers")
def get_burgers_loaders(
    config: Dict[str, Any],
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """
    Get Burgers equation data loaders.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    dataset_config = config.get("dataset", {})

    num_samples = dataset_config.get("num_samples", 2000)
    batch_size = dataset_config.get("batch_size", 64)
    num_workers = dataset_config.get("num_workers", 0)
    nu = dataset_config.get("nu", 0.05)
    noise_std = dataset_config.get("noise_std", 0.01)
    val_split = dataset_config.get("val_split", 0.1)
    seed = config.get("experiment", {}).get("seed", 42)

    # Create train dataset
    train_dataset = BurgersEquationDataset(
        num_samples=num_samples,
        nu=nu,
        noise_std=noise_std,
        seed=seed,
    )

    # Create test dataset (separate samples)
    test_dataset = BurgersEquationDataset(
        num_samples=num_samples // 4,
        nu=nu,
        noise_std=noise_std,
        seed=seed + 1000,
    )

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


def get_burgers_info() -> Dict[str, Any]:
    """Get information about the Burgers equation dataset."""
    return {
        "name": "Burgers Equation",
        "description": "1D Burgers equation: u_t + u * u_x = ν * u_xx",
        "pde": "u_t + u * u_x = ν * u_xx",
        "input_dim": 2,
        "output_dim": 1,
        "solution_type": "Traveling wave (shock)",
    }
