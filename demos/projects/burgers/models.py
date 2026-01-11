"""
Model definitions for Burgers equation prediction.
"""

from typing import List

import torch
import torch.nn as nn

from demos.base import registry


@registry.register_model("burgers", "mlp")
class BurgersNet(nn.Module):
    """
    Neural network for predicting Burgers equation solution.

    Input: (x, t) coordinates
    Output: u(x, t) solution value

    Args:
        hidden_sizes: List of hidden layer sizes
        activation: Activation function ("silu", "relu", "tanh", "gelu")
    """

    def __init__(
        self,
        hidden_sizes: List[int] = [256, 256, 128, 64],
        activation: str = "silu",
    ):
        super().__init__()

        self.activation_fn = self._get_activation(activation)

        # Build layers
        layers = []
        prev_size = 2  # Input: (x, t)

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self._get_activation(activation))
            prev_size = hidden_size

        self.features = nn.Sequential(*layers)
        self.output = nn.Linear(prev_size, 1)

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "silu": nn.SiLU(),
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
        }
        return activations.get(name.lower(), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.output(x)
        return x


@registry.register_model("burgers", "mlp_small")
class BurgersNetSmall(BurgersNet):
    """Smaller network for faster experiments."""

    def __init__(self, activation: str = "silu"):
        super().__init__(hidden_sizes=[128, 64], activation=activation)


@registry.register_model("burgers", "mlp_large")
class BurgersNetLarge(BurgersNet):
    """Larger network for better accuracy."""

    def __init__(self, activation: str = "silu"):
        super().__init__(hidden_sizes=[512, 256, 256, 128, 64], activation=activation)
