"""
Registry for models, datasets, and losses.

Provides a decorator-based registration system for easy extensibility.

Usage:
    from demos.base import registry

    @registry.register_model("mnist", "mlp")
    class MLP(nn.Module):
        ...

    @registry.register_dataset("mnist")
    def get_mnist_loaders(config):
        ...

    # Later, retrieve registered components
    model_cls = registry.get_model("mnist", "mlp")
    get_loaders = registry.get_dataset("mnist")
"""

from typing import Any, Callable, Dict, Optional, Type

import torch.nn as nn


class Registry:
    """Central registry for models, datasets, and loss functions."""

    def __init__(self):
        self._models: Dict[str, Dict[str, Type[nn.Module]]] = {}
        self._datasets: Dict[str, Callable] = {}
        self._losses: Dict[str, Callable] = {}
        self._projects: Dict[str, Any] = {}

    # ==================== Model Registration ====================

    def register_model(self, project: str, name: str):
        """
        Decorator to register a model class.

        Args:
            project: Project name (e.g., "mnist", "cifar10")
            name: Model name (e.g., "mlp", "cnn")

        Example:
            @registry.register_model("mnist", "mlp")
            class MLP(nn.Module):
                def __init__(self, config):
                    ...
        """

        def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
            if project not in self._models:
                self._models[project] = {}
            self._models[project][name] = cls
            return cls

        return decorator

    def get_model(self, project: str, name: str) -> Type[nn.Module]:
        """Get a registered model class."""
        if project not in self._models:
            raise KeyError(f"Project '{project}' not found. Available: {list(self._models.keys())}")
        if name not in self._models[project]:
            raise KeyError(
                f"Model '{name}' not found in project '{project}'. "
                f"Available: {list(self._models[project].keys())}"
            )
        return self._models[project][name]

    def list_models(self, project: Optional[str] = None) -> Dict[str, list]:
        """List all registered models, optionally filtered by project."""
        if project:
            return {project: list(self._models.get(project, {}).keys())}
        return {p: list(models.keys()) for p, models in self._models.items()}

    # ==================== Dataset Registration ====================

    def register_dataset(self, name: str):
        """
        Decorator to register a dataset loader function.

        Args:
            name: Dataset name (e.g., "mnist", "cifar10")

        Example:
            @registry.register_dataset("mnist")
            def get_mnist_loaders(config):
                return train_loader, val_loader, test_loader
        """

        def decorator(func: Callable) -> Callable:
            self._datasets[name] = func
            return func

        return decorator

    def get_dataset(self, name: str) -> Callable:
        """Get a registered dataset loader function."""
        if name not in self._datasets:
            raise KeyError(f"Dataset '{name}' not found. Available: {list(self._datasets.keys())}")
        return self._datasets[name]

    def list_datasets(self) -> list:
        """List all registered datasets."""
        return list(self._datasets.keys())

    # ==================== Loss Registration ====================

    def register_loss(self, name: str):
        """
        Decorator to register a loss function.

        Args:
            name: Loss name (e.g., "cross_entropy", "mse", "focal")

        Example:
            @registry.register_loss("focal")
            def focal_loss(model, inputs, targets, gamma=2.0):
                ...
        """

        def decorator(func: Callable) -> Callable:
            self._losses[name] = func
            return func

        return decorator

    def get_loss(self, name: str) -> Callable:
        """Get a registered loss function."""
        if name not in self._losses:
            raise KeyError(f"Loss '{name}' not found. Available: {list(self._losses.keys())}")
        return self._losses[name]

    def list_losses(self) -> list:
        """List all registered loss functions."""
        return list(self._losses.keys())

    # ==================== Project Registration ====================

    def register_project(self, name: str):
        """
        Decorator to register a project module.

        Args:
            name: Project name (e.g., "mnist", "cifar10")
        """

        def decorator(module: Any) -> Any:
            self._projects[name] = module
            return module

        return decorator

    def get_project(self, name: str) -> Any:
        """Get a registered project module."""
        if name not in self._projects:
            raise KeyError(f"Project '{name}' not found. Available: {list(self._projects.keys())}")
        return self._projects[name]

    def list_projects(self) -> list:
        """List all registered projects."""
        return list(self._projects.keys())

    # ==================== Utility ====================

    def summary(self) -> Dict[str, Any]:
        """Get a summary of all registered components."""
        return {
            "projects": self.list_projects(),
            "models": self.list_models(),
            "datasets": self.list_datasets(),
            "losses": self.list_losses(),
        }


# Global registry instance
registry = Registry()


# ==================== Built-in Loss Functions ====================
# These are common losses that can be used across projects


@registry.register_loss("cross_entropy")
def cross_entropy_loss(model, inputs, targets):
    """Standard cross-entropy loss for classification."""
    import torch.nn.functional as F

    outputs = model(inputs)
    return F.cross_entropy(outputs, targets)


@registry.register_loss("mse")
def mse_loss(model, inputs, targets):
    """Mean squared error loss for regression."""
    import torch.nn.functional as F

    outputs = model(inputs)
    return F.mse_loss(outputs, targets)


@registry.register_loss("nll")
def nll_loss(model, inputs, targets):
    """Negative log likelihood loss (expects log-softmax outputs)."""
    import torch.nn.functional as F

    outputs = model(inputs)
    return F.nll_loss(outputs, targets)


@registry.register_loss("bce")
def bce_loss(model, inputs, targets):
    """Binary cross-entropy loss."""
    import torch.nn.functional as F

    outputs = model(inputs)
    return F.binary_cross_entropy_with_logits(outputs, targets.float())
