# Loss Landscape

## 🎯 Why: Three Questions This Tool Answers

### ❓ Q1: Is my setup hard to train?

| Landscape Shape | Meaning | Action |
|-----------------|---------|--------|
| 🏔️ Rugged, many peaks | Hard to optimize | Larger batch, lower LR, different optimizer |
| 🏞️ Smooth, single valley | Easy to optimize | Current setup is good |
| 🕳️ Sharp narrow minimum | Overfitting risk | Add regularization |
| 🏖️ Flat wide minimum | Good generalization | This is the goal! |

### ❓ Q2: Can I improve further?

| Current Position | Meaning | Action |
|------------------|---------|--------|
| Bottom of valley | Converged well | Done, or try different architecture |
| On a slope | Not converged | Train longer |
| Local minimum (lower areas visible) | Stuck | Restart with different init |
| Flat plateau everywhere | Vanishing gradients | Check architecture |

### ❓ Q3: What happened during training?

| Trajectory Shape | Meaning | Action |
|------------------|---------|--------|
| 📈 Smooth descent | Healthy training | Good! |
| 🔄 Oscillating | LR too high | Reduce learning rate |
| 📉 Stuck, no movement | LR too low | Increase LR |
| 🌀 Spiraling in | Momentum working | Normal |

---

## 🔬 What: Five Types of Analysis

```
                                    ┌─────────────────────────────┐
                                    │  1. 1D Line Profile         │
                                    │  2. 2D Surface              │
High-dimensional    ──────────►     │  3. 3D Volume               │
Parameter Space θ                   │  4. Training Trajectory     │
   (millions)                       │  5. Hessian Eigenvalues     │
                                    └─────────────────────────────┘
```

### 1. 1D Line Profile
Loss along one direction. Quick sanity check.
```python
writer.sample_landscape(model, loader, loss_fn, mode="1d")
```

### 2. 2D Surface
Loss surface on a 2D plane. **Main visualization**.
```python
writer.sample_landscape(model, loader, loss_fn, mode="2d")
```

### 3. 3D Volume
Loss in 3D parameter space. For complex analysis.
```python
writer.sample_landscape(model, loader, loss_fn, mode="3d")
```

### 4. Training Trajectory
Path taken during training on the landscape.
```python
for epoch in range(100):
    train_loss = train(...)
    writer.record_checkpoint(model, epoch, train_loss=train_loss)
writer.build_trajectory(model, loader, loss_fn)
```

### 5. Hessian Analysis
Curvature information: sharpness/flatness metrics.
```python
with Explorer(model, loss_fn, loader) as explorer:
    hessian_data = explorer.compute_hessian_metrics()
```

---

## 📐 How: The Math Behind It

### Step 1: Project to Low Dimensions

```
θ(α, β) = θ₀ + α·d₁ + β·d₂

where:
  θ₀  = current model parameters (center point)
  d₁  = direction vector 1
  d₂  = direction vector 2
  α,β = coordinates on the 2D plane
```

**Direction types:**
- **Random** (`direction_mode="random"`): Quick exploration, no trajectory needed
- **PCA** ⭐ (`direction_mode="pca"`): Captures actual optimization path, most informative
- **Custom** (`directions=(d1, d2)`): User-defined for specific analysis

> 💡 **Which to use?**
> - No checkpoints → `direction_mode="random"` (only option)
> - Have checkpoints → `direction_mode="pca"` ⭐ (recommended, shows where params actually changed)

### Step 2: Sample the Surface

```python
for α in linspace(-range, +range, grid_size):
    for β in linspace(-range, +range, grid_size):
        θ_perturbed = θ₀ + α * d₁ + β * d₂
        surface[α, β] = loss(model(θ_perturbed), data)
```

### Step 3: Compute Hessian (Optional)

```
H = ∂²L/∂θ²

Metrics:
  - Eigenvalues λᵢ → Curvature in each direction
  - Trace(H) → Total curvature
  - λ_max → Sharpest direction (generalization indicator)
```

---

## 🔧 Factors: What Affects the Landscape

### A. Factors That Determine Landscape Shape

| Factor | Parameter | Example |
|--------|-----------|---------|
| **Data** | `data_loader` | Training/validation set |
| **Data Preprocessing** | `pre_batch_hook` | Mixup, augmentation |
| **Evaluation Size** | `max_batches` | 10 (fast) vs None (accurate) |
| **Model** | `model` | Network architecture |
| **Model State** | `model_mode` | `'eval'` or `'train'` |
| **Loss Function** | `loss_fn` | MSE, CrossEntropy |
| **Regularization** | `regularizer` | L1, L2 functions |
| **Reg Weight** | `regularizer_weight` | λ value |
| **Custom Loss** | `custom_loss_fn` | Physics-informed, etc. |
| **Post-processing** | `post_batch_hook` | Label smoothing |

### B. Factors That Determine Viewing Angle (Not the landscape itself)

| Factor | Parameter | Effect |
|--------|-----------|--------|
| **Directions** | `directions` | Which plane to view |
| **Range** | `range_scale` | How far to look (0.1 = ±10%) |
| **Resolution** | `grid_size` | How detailed (50 = 2500 points) |
| **Seed** | `seed` | Reproducibility |

---

## 🚀 Usage

### Quick Start: One Line

```python
from loss_landscape import sample_landscape

sample_landscape(model, loader, loss_fn, "./landscape.json")
```

### Standard: Writer API

```python
from loss_landscape import LossLandscapeWriter

with LossLandscapeWriter("./runs/exp1") as writer:
    writer.sample_landscape(model, loader, loss_fn)
# Auto-exports JSON on close
```

### With Regularization

```python
def l2_reg(model):
    return sum(p.norm()**2 for p in model.parameters())

writer.sample_landscape(
    model, loader, loss_fn,
    regularizer=l2_reg,
    regularizer_weight=0.01
)
```

### Custom Loss (e.g., Physics-Informed)

```python
def pinn_loss(model, inputs, targets):
    pred = model(inputs)
    data_loss = F.mse_loss(pred, targets)
    pde_loss = compute_pde_residual(model, inputs)
    return data_loss + 0.1 * pde_loss

writer.sample_landscape(model, loader, custom_loss_fn=pinn_loss)
```

### Training Trajectory

```python
writer = LossLandscapeWriter("./runs/training")

for epoch in range(100):
    loss = train_one_epoch(model, ...)
    writer.record_checkpoint(model, epoch, train_loss=loss)

writer.build_trajectory(model, loader, loss_fn, mode="pca")
writer.close()
```

### Landscape with PCA Directions (Recommended when you have checkpoints)

```python
writer = LossLandscapeWriter("./runs/pca_landscape")

# Record checkpoints during training
for epoch in range(100):
    loss = train_one_epoch(model, ...)
    writer.record_checkpoint(model, epoch, train_loss=loss)

# Sample landscape using PCA directions ⭐
writer.sample_landscape(
    model, loader, loss_fn,
    direction_mode="pca",  # Use PCA instead of random!
)
writer.close()
```

### Compare Models (Same Direction)

```python
# Model A
writer_a = LossLandscapeWriter("./runs/a", seed=42)
writer_a.sample_landscape(model_a, loader, loss_fn)
directions = writer_a.get_directions()
writer_a.close()

# Model B (same viewing angle)
writer_b = LossLandscapeWriter("./runs/b")
writer_b.sample_landscape(model_b, loader, loss_fn, directions=directions)
writer_b.close()
```

### Low-Level: Explorer + Hessian

```python
from loss_landscape import Explorer, LandscapeStorage

storage = LandscapeStorage("landscape.duckdb")
with Explorer(model, loss_fn, loader, storage=storage) as explorer:
    explorer.build_surface(grid_size=50, range_scale=0.1)
    hessian = explorer.compute_hessian_metrics()
```

---

## 📋 API Reference

### LossLandscapeWriter

```python
writer = LossLandscapeWriter(
    log_dir="./runs/exp1",  # Output directory
    seed=42,                 # Random seed (reproducibility)
    auto_export=True,        # Export JSON on close()
)
```

### sample_landscape()

```python
writer.sample_landscape(
    # Required
    model,                   # PyTorch model
    data_loader,             # DataLoader
    loss_fn,                 # Loss function (or use custom_loss_fn)
    
    # Sampling
    mode="2d",               # "1d", "2d", "3d"
    grid_size=50,            # Points per dimension
    range_scale=0.1,         # Perturbation range
    directions=None,         # Explicit directions (overrides direction_mode)
    direction_mode="random", # "random" or "pca" (pca needs checkpoints)
    
    # Loss calculation
    regularizer=None,        # model -> scalar
    regularizer_weight=1.0,  # λ
    custom_loss_fn=None,     # (model, x, y) -> scalar
    model_mode='eval',       # 'eval' or 'train'
    max_batches=10,          # None = use all data
    
    # Hooks
    pre_batch_hook=None,     # batch -> batch
    post_batch_hook=None,    # (out, x, y) -> loss
)
```

### build_trajectory()

```python
writer.build_trajectory(
    model, data_loader, loss_fn,
    mode="pca",              # "pca" or "fixed"
    directions=None,         # For "fixed" mode
    # ... same loss params as sample_landscape
)
```

---

## ⚠️ Gotchas

### Weight Decay ≠ Regularizer

```python
# ❌ This weight_decay won't appear in loss landscape!
optimizer = Adam(model.parameters(), weight_decay=0.01)

# ✅ Use regularizer to visualize it
writer.sample_landscape(
    ...,
    regularizer=lambda m: sum(p.norm()**2 for p in m.parameters()),
    regularizer_weight=0.01
)
```

### Data Shuffle → Variance

```python
# ✅ Use shuffle=False for consistent evaluation
eval_loader = DataLoader(dataset, shuffle=False)
```

### max_batches Trade-off

```python
max_batches=10    # Fast (default), good for exploration
max_batches=None  # Slow but accurate, for final results
```

### Random Seed for Comparison

```python
# Different seeds → different directions → incomparable results!
# Always use same seed or share directions explicitly
directions = writer_a.get_directions()
writer_b.sample_landscape(..., directions=directions)
```

---

## 🎮 Quick Reference: Answering the Three Questions

| Question | Code | Key Settings |
|----------|------|--------------|
| **Q1: Hard to train?** | `sample_landscape()` | `range_scale=0.5` (see far) |
| **Q2: Can improve?** | `sample_landscape()` | `range_scale=0.1, grid_size=100` (zoom in) |
| **Q3: Training path?** | `record_checkpoint()` + `build_trajectory()` | `mode="pca"` |

> 💡 **Pro tip:** If you have checkpoints, always use `direction_mode="pca"` for more meaningful landscapes!

```bash
# View results
loss_landscape view ./runs/exp1/landscape.json
```

---

## 📁 File Structure

```
core/
├── __init__.py      # Exports
├── writer.py        # High-level API
├── explorer.py      # Computation engine
├── storage.py       # DuckDB persistence
└── hessian.py       # Hessian computation
```
