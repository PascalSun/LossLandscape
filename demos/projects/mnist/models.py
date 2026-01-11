"""
Model definitions for MNIST classification.

Available models:
- MLP: Multi-Layer Perceptron
- CNN: Convolutional Neural Network
- LeNet: Classic LeNet-5 style architecture
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from demos.base import registry


@registry.register_model("mnist", "mlp")
class MLP(nn.Module):
    """
    Multi-Layer Perceptron for MNIST.

    Args:
        hidden_sizes: List of hidden layer sizes
        activation: Activation function ("relu", "silu", "tanh", "gelu")
        dropout: Dropout probability (0 to disable)
        input_size: Input feature size (default: 784 = 28*28)
        num_classes: Number of output classes (default: 10)
    """

    def __init__(
        self,
        hidden_sizes: List[int] = [256, 128],
        activation: str = "relu",
        dropout: float = 0.0,
        input_size: int = 784,
        num_classes: int = 10,
    ):
        super().__init__()

        self.input_size = input_size
        self.flatten = nn.Flatten()

        # Build layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self._get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_size, num_classes)

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU(0.1),
        }
        return activations.get(name.lower(), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.features(x)
        x = self.classifier(x)
        return x


@registry.register_model("mnist", "cnn")
class CNN(nn.Module):
    """
    Convolutional Neural Network for MNIST.

    Args:
        channels: List of channel sizes for conv layers
        kernel_size: Convolution kernel size
        hidden_size: Size of the hidden FC layer
        activation: Activation function
        dropout: Dropout probability
        batch_norm: Whether to use batch normalization
        num_classes: Number of output classes
    """

    def __init__(
        self,
        channels: List[int] = [32, 64],
        kernel_size: int = 3,
        hidden_size: int = 128,
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
        num_classes: int = 10,
    ):
        super().__init__()

        self.activation = self._get_activation(activation)

        # Build conv layers
        conv_layers = []
        in_channels = 1  # MNIST is grayscale

        for out_channels in channels:
            conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
            )
            if batch_norm:
                conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(self._get_activation(activation))
            conv_layers.append(nn.MaxPool2d(2, 2))
            if dropout > 0:
                conv_layers.append(nn.Dropout2d(dropout))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # Calculate flatten size (28 -> 14 -> 7 for 2 pool layers)
        # For n pool layers with 2x2: 28 / (2^n)
        final_size = 28 // (2 ** len(channels))
        flatten_size = channels[-1] * final_size * final_size

        # FC layers
        self.flatten = nn.Flatten()
        fc_layers = [
            nn.Linear(flatten_size, hidden_size),
            self._get_activation(activation),
        ]
        if dropout > 0:
            fc_layers.append(nn.Dropout(dropout))
        fc_layers.append(nn.Linear(hidden_size, num_classes))

        self.fc_layers = nn.Sequential(*fc_layers)

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU(0.1),
        }
        return activations.get(name.lower(), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x


@registry.register_model("mnist", "lenet")
class LeNet(nn.Module):
    """
    LeNet-5 style architecture for MNIST.

    A classic CNN architecture with:
    - 2 conv layers with average pooling
    - 3 fully connected layers

    Args:
        num_classes: Number of output classes
        activation: Activation function
    """

    def __init__(
        self,
        num_classes: int = 10,
        activation: str = "relu",
    ):
        super().__init__()

        act = self._get_activation(activation)

        # Feature extraction
        self.features = nn.Sequential(
            # Conv1: 1x28x28 -> 6x28x28 -> 6x14x14
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            act,
            nn.AvgPool2d(kernel_size=2, stride=2),
            # Conv2: 6x14x14 -> 16x10x10 -> 16x5x5
            nn.Conv2d(6, 16, kernel_size=5),
            act,
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            act,
            nn.Linear(120, 84),
            act,
            nn.Linear(84, num_classes),
        )

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
        }
        return activations.get(name.lower(), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


@registry.register_model("mnist", "mlp_bn")
class MLPBN(nn.Module):
    """
    MLP with Batch Normalization for MNIST.

    Args:
        hidden_sizes: List of hidden layer sizes
        activation: Activation function
        dropout: Dropout probability
        num_classes: Number of output classes
    """

    def __init__(
        self,
        hidden_sizes: List[int] = [256, 128],
        activation: str = "relu",
        dropout: float = 0.0,
        num_classes: int = 10,
    ):
        super().__init__()

        self.flatten = nn.Flatten()

        # Build layers
        layers = []
        prev_size = 784

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(self._get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_size, num_classes)

    def _get_activation(self, name: str) -> nn.Module:
        return {"relu": nn.ReLU(), "silu": nn.SiLU(), "gelu": nn.GELU()}.get(
            name.lower(), nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.features(x)
        x = self.classifier(x)
        return x
