"""
Burgers Equation PINN Project

Physics-Informed Neural Network for 1D Burgers equation.
Demonstrates loss landscape analysis comparing MSE vs Physics-informed loss.

The Burgers equation is: u_t + u * u_x = ν * u_xx
- Time derivative term: u_t
- Nonlinear convection term: u * u_x (creates shock waves)
- Diffusion term: ν * u_xx

This project shows how physics constraints affect the loss landscape,
creating complex, bumpy landscapes with multiple local minima, saddle points,
and narrow valleys.

Available configs:
- mse: Pure MSE loss (data fitting only)
- physics: Physics-informed loss (MSE + PDE residual)
- quick_test: Fast testing configuration
"""

from demos.base import registry

from .data import get_burgers_loaders
from .experiment import BurgersExperiment
from .models import BurgersNet

# Register the project
registry._projects["burgers"] = {
    "name": "burgers",
    "description": "1D Burgers equation PINN - comparing MSE vs Physics loss landscapes",
    "experiment_class": BurgersExperiment,
}

__all__ = [
    "BurgersExperiment",
    "get_burgers_loaders",
    "BurgersNet",
]
