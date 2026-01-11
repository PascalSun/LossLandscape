"""
Loss functions for regression tasks.
"""

import torch.nn.functional as F

from demos.base import registry


@registry.register_loss("regression_mse")
def regression_mse_loss(model, inputs, targets):
    """Standard MSE loss."""
    outputs = model(inputs)
    return F.mse_loss(outputs, targets)


@registry.register_loss("regression_l2")
def regression_l2_loss(model, inputs, targets, weight_decay: float = 0.01):
    """MSE loss with L2 regularization."""
    outputs = model(inputs)
    data_loss = F.mse_loss(outputs, targets)
    l2_reg = weight_decay * sum(p.norm() ** 2 for p in model.parameters())
    return data_loss + l2_reg


def create_l2_loss(weight_decay: float = 0.01):
    """Factory function to create L2 regularized loss."""

    def loss_fn(model, inputs, targets):
        return regression_l2_loss(model, inputs, targets, weight_decay=weight_decay)

    return loss_fn
