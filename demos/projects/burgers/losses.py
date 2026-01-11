"""
Loss functions for Burgers equation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from demos.base import registry


@registry.register_loss("burgers_mse")
def burgers_mse_loss(model, inputs, targets):
    """Standard MSE loss for data fitting."""
    outputs = model(inputs)
    return F.mse_loss(outputs, targets)


@registry.register_loss("burgers_physics")
def burgers_physics_loss(model, inputs, targets, nu: float = 0.05, physics_weight: float = 0.1):
    """
    Physics-informed loss for Burgers equation.

    Loss = MSE_data + lambda * |u_t + u*u_x - ν*u_xx|²

    The physics residual involves:
    - u_t: time derivative
    - u*u_x: nonlinear convection term
    - ν*u_xx: diffusion term

    This creates a highly complex loss landscape due to:
    1. Nonlinear term u*u_x making the loss non-convex
    2. Multiple constraints to satisfy simultaneously
    3. Sharp gradients in shock regions
    """
    # Data loss
    predictions = model(inputs)
    mse_loss = F.mse_loss(predictions, targets)

    # Physics constraint using autograd
    with torch.enable_grad():
        inputs_grad = inputs.clone().detach().requires_grad_(True)
        u = model(inputs_grad)

        # Compute gradients: du/d(x,t)
        grad_u = torch.autograd.grad(
            outputs=u,
            inputs=inputs_grad,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]

        # u_x (spatial derivative) - first column
        u_x = grad_u[:, 0:1]
        # u_t (time derivative) - second column
        u_t = grad_u[:, 1:2]

        # Compute u_xx (second spatial derivative)
        u_xx_grad = torch.autograd.grad(
            outputs=u_x,
            inputs=inputs_grad,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True,
        )[0]
        u_xx = u_xx_grad[:, 0:1]

        # Burgers equation residual: u_t + u * u_x - ν * u_xx = 0
        physics_residual = u_t + u * u_x - nu * u_xx
        physics_constraint = torch.mean(physics_residual**2)

    return mse_loss + physics_weight * physics_constraint


def create_physics_loss(nu: float = 0.05, physics_weight: float = 0.1):
    """Factory function to create physics loss with specific parameters."""

    def loss_fn(model, inputs, targets):
        return burgers_physics_loss(model, inputs, targets, nu=nu, physics_weight=physics_weight)

    return loss_fn
