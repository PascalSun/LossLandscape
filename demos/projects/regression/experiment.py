"""
Simple Regression Experiment class.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import torch.nn as nn
from torch.utils.data import DataLoader

from demos.base import BaseExperiment, registry

from .data import get_regression_loaders
from .losses import create_l2_loss, regression_mse_loss
from .models import SimpleNet


class RegressionExperiment(BaseExperiment):
    """
    Experiment class for simple regression tasks.

    Supports:
    - mse: Pure MSE loss
    - l2: MSE with L2 regularization
    """

    def setup_data(self) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
        """Setup regression data loaders."""
        return get_regression_loaders(self.config)

    def setup_model(self) -> nn.Module:
        """Setup model based on config."""
        model_config = self.config.get("model", {})
        model_name = model_config.get("name", "mlp")
        model_params = model_config.get("params", {})

        # Get model class from registry
        model_cls = registry.get_model("regression", model_name)

        # Inject input/output size from dataset config
        dataset_config = self.config.get("dataset", {})
        if "input_size" not in model_params:
            model_params["input_size"] = dataset_config.get("input_size", 10)
        if "output_size" not in model_params:
            model_params["output_size"] = dataset_config.get("output_size", 1)

        return model_cls(**model_params)

    def _setup_loss(self) -> Callable:
        """Setup loss function based on config."""
        loss_name = self.config.get("training", {}).get("loss", "regression_mse")
        loss_config = self.config.get("training", {}).get("loss_params", {})

        if loss_name == "regression_l2" or loss_name == "l2":
            weight_decay = loss_config.get("weight_decay", 0.01)
            return create_l2_loss(weight_decay=weight_decay)

        # Default to MSE
        return regression_mse_loss
