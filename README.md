# Loss Landscape — Quick Start

## Setup
```bash
cd /home/pascal/AI4WA/LossLandscape
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Generate Landscapes (Python)
```bash
# Basic 2D + 3D example (MNIST MLP), outputs to outputs/run_*/ 
python -m loss_landscape.examples.basic_usage

# Training trajectory example, outputs to outputs/run_*/
python -m loss_landscape.examples.training_trajectory
```

## CLI (installed via `pip install -e .`)
```bash
# Show help (any alias works: loss_landscape / losslandscape / loss-landscape / lossvis)
loss_landscape --help

# View information about exported JSON data
loss_landscape view -i outputs/run_xxx/foo.json
```

## Frontend (Next.js app)
```bash
cd app
npm install
npm run dev
# Open http://localhost:3000 and load a run directory (e.g., outputs/run_xxx/)
```

## Where outputs go
- Examples write to `outputs/run_YYYYMMDD_HHMMSS/`
- Each run contains:
  - `<name>.landscape` (DuckDB)
  - `<name>.json` (frontend-ready; includes 2D + 3D + trajectory)

## Notes
- 3D volume generation is heavier; reduce `grid_size` if needed.
- Trajectory now records 3D (`traj_3`) by default for slice/volume views.

