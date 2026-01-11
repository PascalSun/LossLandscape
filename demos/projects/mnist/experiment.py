"""
MNIST Experiment class.
"""

from typing import Any, Dict, Optional, Tuple

import torch.nn as nn
from torch.utils.data import DataLoader

from demos.base import BaseExperiment, registry

from .data import get_mnist_loaders
from .models import CNN, MLP, LeNet, MLPBN


class MNISTExperiment(BaseExperiment):
    """
    Experiment class for MNIST classification.

    Supports multiple model architectures:
    - mlp: Multi-Layer Perceptron
    - cnn: Convolutional Neural Network
    - lenet: LeNet-5 style architecture
    - mlp_bn: MLP with Batch Normalization
    """

    def setup_data(self) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
        """Setup MNIST data loaders."""
        return get_mnist_loaders(self.config)

    def setup_model(self) -> nn.Module:
        """Setup model based on config."""
        model_config = self.config.get("model", {})
        model_name = model_config.get("name", "mlp")
        model_params = model_config.get("params", {})

        # Get model class from registry
        model_cls = registry.get_model("mnist", model_name)

        # Create model instance
        model = model_cls(**model_params)

        return model
