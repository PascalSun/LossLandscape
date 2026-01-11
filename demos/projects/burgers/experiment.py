"""
Burgers Equation Experiment class.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import torch.nn as nn
from torch.utils.data import DataLoader

from demos.base import BaseExperiment, registry

from .data import get_burgers_loaders
from .losses import burgers_mse_loss, create_physics_loss
from .models import BurgersNet


class BurgersExperiment(BaseExperiment):
    """
    Experiment class for Burgers equation PINN.

    Supports two loss types:
    - mse: Pure data fitting (MSE loss)
    - physics: Physics-informed (MSE + PDE residual)
    """

    def setup_data(self) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
        """Setup Burgers equation data loaders."""
        return get_burgers_loaders(self.config)

    def setup_model(self) -> nn.Module:
        """Setup model based on config."""
        model_config = self.config.get("model", {})
        model_name = model_config.get("name", "mlp")
        model_params = model_config.get("params", {})

        # Get model class from registry
        model_cls = registry.get_model("burgers", model_name)

        # Create model instance
        return model_cls(**model_params)

    def _setup_loss(self) -> Callable:
        """Setup loss function based on config."""
        loss_name = self.config.get("training", {}).get("loss", "burgers_mse")
        loss_config = self.config.get("training", {}).get("loss_params", {})

        if loss_name == "burgers_physics" or loss_name == "physics":
            nu = self.config.get("dataset", {}).get("nu", 0.05)
            physics_weight = loss_config.get("physics_weight", 0.1)
            return create_physics_loss(nu=nu, physics_weight=physics_weight)

        # Default to MSE
        return burgers_mse_loss
