"""
Model definitions for simple regression.
"""

from typing import List

import torch
import torch.nn as nn

from demos.base import registry


@registry.register_model("regression", "mlp")
class SimpleNet(nn.Module):
    """
    Simple MLP for regression.

    Args:
        input_size: Input feature dimension
        hidden_sizes: List of hidden layer sizes
        output_size: Output dimension
        activation: Activation function
    """

    def __init__(
        self,
        input_size: int = 10,
        hidden_sizes: List[int] = [32, 16],
        output_size: int = 1,
        activation: str = "relu",
    ):
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self._get_activation(activation))
            prev_size = hidden_size

        self.features = nn.Sequential(*layers)
        self.output = nn.Linear(prev_size, output_size)

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "silu": nn.SiLU(),
            "gelu": nn.GELU(),
        }
        return activations.get(name.lower(), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.output(x)
        return x


@registry.register_model("regression", "shallow")
class ShallowNet(nn.Module):
    """Single hidden layer network."""

    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 16,
        output_size: int = 1,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@registry.register_model("regression", "deep")
class DeepNet(nn.Module):
    """Deeper network for comparison."""

    def __init__(
        self,
        input_size: int = 10,
        output_size: int = 1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
