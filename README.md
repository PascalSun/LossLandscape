# Loss Landscape Visualization Platform

English | [中文](./README.zh-CN.md)

A comprehensive platform for visualizing and analyzing loss landscapes of deep learning models. Understand your model's optimization behavior through interactive 1D, 2D, and 3D visualizations.

## ✨ Features

- **Multi-dimensional Visualization**: View loss landscapes in 1D (line), 2D (surface), and 3D (volume)
- **Training Trajectory Tracking**: Record and visualize the optimization path during training
- **Hessian Analysis**: Compute eigenvalue spectra, traces, and sharpness metrics
- **PCA-aligned Directions**: Automatic direction selection based on training trajectory
- **Interactive Web UI**: Beautiful, modern visualization interface
- **Simple API**: Just a few lines of code to generate landscapes

## 🚀 Quick Start

### Installation

```bash
# Install the Python SDK
pip install -e .

# Or using uv
uv pip install -e .
```

### Basic Usage

```python
import torch.nn as nn
from loss_landscape import sample_landscape

# Define loss function: (model, inputs, targets) -> loss
def loss_fn(model, inputs, targets):
    return nn.MSELoss()(model(inputs), targets)

# One-line landscape generation
sample_landscape(model, data_loader, loss_fn, "./landscape.json")
```

### Using the Writer Interface (Recommended)

```python
import torch.nn as nn
from loss_landscape import LossLandscapeWriter

# Define loss function: (model, inputs, targets) -> loss
def loss_fn(model, inputs, targets):
    return nn.MSELoss()(model(inputs), targets)

# Create a writer
writer = LossLandscapeWriter("./runs/experiment1")

# Generate 2D landscape
writer.sample_landscape(model, data_loader, loss_fn, grid_size=50)

# Close and export
writer.close()
```

### Recording Training Trajectory

```python
from loss_landscape import LossLandscapeWriter

writer = LossLandscapeWriter("./runs/training")

# Training loop
for epoch in range(100):
    train_loss = train_one_epoch(model, ...)
    writer.record_checkpoint(model, epoch, train_loss=train_loss)

# Build trajectory visualization
writer.build_trajectory(model, data_loader, loss_fn)
writer.sample_landscape(model, data_loader, loss_fn)
writer.close()
```

### Loss Function with Regularization

```python
import torch.nn as nn
from loss_landscape import LossLandscapeWriter

def loss_with_reg(model, inputs, targets):
    outputs = model(inputs)
    data_loss = nn.MSELoss()(outputs, targets)
    l2_reg = 0.01 * sum(p.norm()**2 for p in model.parameters())
    return data_loss + l2_reg

writer = LossLandscapeWriter("./runs/regularized")
writer.sample_landscape(model, data_loader, loss_with_reg)
writer.close()
```

### Physics-Informed Loss

```python
import torch.nn as nn
from loss_landscape import LossLandscapeWriter

def physics_loss(model, inputs, targets):
    outputs = model(inputs)
    data_loss = nn.MSELoss()(outputs, targets)
    physics_residual = compute_pde_residual(model, inputs)
    return data_loss + 0.1 * physics_residual

writer = LossLandscapeWriter("./runs/pinn")
writer.sample_landscape(model, data_loader, physics_loss)
writer.close()
```

## 🖥️ Web Visualization

### Start the Development Server

```bash
cd web
npm install
npm run dev
```

Open http://localhost:3000 to view the interactive visualization.

### Features

- **Surface Plot**: Interactive 3D surface with rotation and zoom
- **Contour Plot**: 2D contour visualization with trajectory overlay
- **Hessian Analysis**: Eigenvalue spectrum density and sharpness metrics
- **Multi-language Support**: English and Chinese interface

## 📁 Project Structure

```
LossLandscape/
├── loss_landscape/          # Python SDK
│   ├── core/               # Core modules
│   │   ├── explorer.py     # Loss landscape computation
│   │   ├── storage.py      # Data persistence (DuckDB)
│   │   ├── writer.py       # High-level API
│   │   └── hessian.py      # Hessian analysis
│   ├── examples/           # Example scripts
│   └── cli.py              # Command-line interface
├── web/                    # Next.js frontend
│   └── src/
│       ├── app/            # React components
│       └── lib/            # Utilities
└── pyproject.toml          # Python package config
```

## 🔧 CLI Commands

```bash
# View landscape data info
losslandscape view -i ./landscape.json

# Run the complete example
losslandscape example
```

## 📊 Output Format

The generated JSON file contains the following structure:

```json
{
  // === 2D Surface Data (primary visualization) ===
  "X": [[0.0, 0.0, ...], [0.1, 0.1, ...]],           // X coordinates grid (grid_size x grid_size)
  "Y": [[0.0, 0.1, ...], [0.0, 0.1, ...]],           // Y coordinates grid (grid_size x grid_size)
  "loss_grid_2d": [[1.2, 1.1, ...], [1.3, 1.0, ...]], // Loss values (grid_size x grid_size)
  "baseline_loss": 0.5,                              // Loss at origin (current model weights)
  "grid_size": 50,                                   // Grid resolution
  "mode": "1d+2d",                                   // Data mode: "1d", "2d", "1d+2d"

  // === 1D Line Data (optional) ===
  "X_1d": [-0.5, -0.4, ..., 0.4, 0.5],              // X coordinates for 1D line
  "loss_line_1d": [2.1, 1.8, ..., 1.9, 2.2],        // Loss values along 1D line
  "baseline_loss_1d": 0.5,                          // Baseline loss for 1D
  "grid_size_1d": 100,                              // 1D grid resolution

  // === 3D Volume Data (optional) ===
  "Z": [[[...]]],                                   // Z coordinates (nx x ny x nz)
  "loss_grid_3d": [[[...]]],                        // 3D loss volume (nx x ny x nz)
  "volume_x": [-0.5, -0.4, ...],                    // X axis values
  "volume_y": [-0.5, -0.4, ...],                    // Y axis values
  "volume_z": [-0.5, -0.4, ...],                    // Z axis values

  // === Training Trajectory (optional) ===
  "trajectory_data": {
    "traj_1": [0.0, 0.1, 0.15, ...],               // X coordinates of trajectory
    "traj_2": [0.0, -0.05, -0.1, ...],             // Y coordinates of trajectory
    "traj_3": [0.0, 0.02, 0.03, ...],              // Z coordinates (for 3D)
    "epochs": [0, 1, 2, ...],                       // Epoch numbers
    "losses": [1.5, 1.2, 0.9, ...],                // Training loss at each epoch
    "val_losses": [1.6, 1.3, 1.0, ...]             // Validation loss (optional)
  },

  // === Hessian Analysis (optional) ===
  "hessian": {
    "epochs": [0, 10, 20, ...],                     // Epochs where Hessian was computed
    "max_eigenvalue": [150.5, 80.2, 45.1, ...],    // Maximum eigenvalue (sharpness)
    "trace": [1200.0, 800.0, 500.0, ...],          // Trace of Hessian
    "top_eigenvalues": [[150.5, 120.3, ...], ...]  // Top-k eigenvalues per epoch
  },

  // === Metadata (optional) ===
  "metadata": {
    "model_info": { ... },                          // Model architecture info
    "training_statistics": {                        // Training stats
      "initial_loss": 1.5,
      "final_loss": 0.1,
      "min_loss": 0.08,
      "min_loss_epoch": 95,
      "learning_rate_history": [0.001, 0.001, ...]
    },
    "loss_description": { ... },                    // Loss function description
    "system_info": { ... }                          // Hardware/system info
  }
}
```

## 📚 References

- Li et al., "Visualizing the Loss Landscape of Neural Nets" (NeurIPS 2018)
- Ghorbani et al., "An Investigation into Neural Net Optimization via Hessian Eigenvalue Density" (ICML 2019)

## 📄 License

MIT License
