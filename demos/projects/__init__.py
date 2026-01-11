"""
Demo projects for loss landscape analysis.

Each project is a self-contained ML experiment with:
- Data loading
- Model definitions
- Configuration files
- Integration with the loss landscape framework

Available projects:
- mnist: MNIST handwritten digit classification
- burgers: 1D Burgers equation PINN (physics-informed)
- regression: Simple regression with synthetic data
"""

# Import projects to register them
from . import burgers, mnist, regression

__all__ = ["mnist", "burgers", "regression"]
