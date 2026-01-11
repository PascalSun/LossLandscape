"""
MNIST Classification Project

A classic image classification task using the MNIST dataset.
Demonstrates loss landscape analysis for different model architectures:
- MLP (Multi-Layer Perceptron)
- CNN (Convolutional Neural Network)
- LeNet-style architecture

Available configs:
- mlp_small: Small MLP with 2 hidden layers
- mlp_large: Larger MLP with 3 hidden layers
- cnn_simple: Simple CNN with 2 conv layers
- cnn_deep: Deeper CNN with batch normalization
"""

from demos.base import registry

from .data import get_mnist_loaders
from .experiment import MNISTExperiment
from .models import CNN, MLP, LeNet

# Register the project
registry._projects["mnist"] = {
    "name": "mnist",
    "description": "MNIST handwritten digit classification",
    "experiment_class": MNISTExperiment,
}

__all__ = [
    "MNISTExperiment",
    "get_mnist_loaders",
    "MLP",
    "CNN",
    "LeNet",
]
