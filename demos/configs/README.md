# Experiment Configurations

This folder contains pre-defined experiment configurations organized by comparison type.

## Quick Start

```bash
# Run all experiments in a folder
losslandscape demo run demos/configs/burgers_comparison/

# Run with custom parameters
losslandscape demo run demos/configs/mnist_comparison/ -o training.epochs=20

# Run a specific config
losslandscape demo run demos/configs/burgers_comparison/mse_with_train.yaml
```

## Available Experiment Sets

### `burgers_comparison/`
Compares MSE vs Physics-informed loss on 1D Burgers equation.

| Config | Description |
|--------|-------------|
| `mse_with_train.yaml` | MSE loss with training trajectory |
| `mse_no_train.yaml` | MSE loss, no training (random init) |
| `physics_with_train.yaml` | Physics loss with training trajectory |
| `physics_no_train.yaml` | Physics loss, no training |

```bash
losslandscape demo run demos/configs/burgers_comparison/
```

### `mnist_comparison/`
Compares different architectures on MNIST classification.

| Config | Description |
|--------|-------------|
| `mlp_small.yaml` | Small MLP (2 layers) |
| `mlp_large.yaml` | Large MLP (4 layers) |
| `cnn_simple.yaml` | Simple CNN (2 conv layers) |
| `lenet.yaml` | LeNet-5 style |

```bash
losslandscape demo run demos/configs/mnist_comparison/
```

### `regression_comparison/`
Compares MSE vs L2 regularized loss on synthetic regression.

| Config | Description |
|--------|-------------|
| `mse.yaml` | Pure MSE loss |
| `regularized.yaml` | MSE + L2 regularization |

```bash
losslandscape demo run demos/configs/regression_comparison/
```

## Quick Test Mode

For faster testing, use override flags to reduce computation:

```bash
# Reduce grid sizes and epochs
losslandscape demo run demos/configs/burgers_comparison/ \
    -o training.epochs=10 \
    -o landscape.grid_size_1d=50 \
    -o landscape.grid_size_2d=15 \
    -o landscape.grid_size_3d=8
```

## Creating New Experiment Sets

1. Create a new subfolder: `mkdir demos/configs/my_experiment/`
2. Add YAML config files with `experiment.project` field
3. Run: `losslandscape demo run demos/configs/my_experiment/`
