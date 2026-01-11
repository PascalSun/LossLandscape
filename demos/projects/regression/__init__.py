"""
Simple Regression Project

Basic regression tasks using synthetic data.
Demonstrates loss landscape analysis for simple cases:
- MSE loss
- MSE with L2 regularization
- Custom loss functions

This project is useful for understanding loss landscape basics
before moving to more complex tasks.

Available configs:
- mse: Basic MSE regression
- regularized: MSE with L2 regularization
- quick_test: Fast testing configuration
"""

from demos.base import registry

from .data import get_regression_loaders
from .experiment import RegressionExperiment
from .models import SimpleNet

# Register the project
registry._projects["regression"] = {
    "name": "regression",
    "description": "Simple regression with synthetic data",
    "experiment_class": RegressionExperiment,
}

__all__ = [
    "RegressionExperiment",
    "get_regression_loaders",
    "SimpleNet",
]
