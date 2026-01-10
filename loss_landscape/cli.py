"""
Command Line Interface for LossLandscape
"""

import json

import click
import numpy as np
from loguru import logger


@click.group()
@click.version_option(version="0.1.0", prog_name="loss_landscape")
def cli():
    """LossLandscape - 通用 Loss Landscape 自动化分析平台"""
    pass


@cli.command()
@click.option(
    "--input",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Input JSON file exported from LossLandscape",
)
def view(input: str):
    """View information about exported Loss Landscape JSON data"""
    logger.info(f"Reading JSON file: {input}...")

    try:
        with open(input, "r") as f:
            data = json.load(f)

        logger.info("=" * 60)
        logger.info(f"File: {input}")

        # Basic info
        grid_size = data.get("grid_size", 0)
        baseline_loss = data.get("baseline_loss", 0.0)
        mode = data.get("mode", "unknown")

        logger.info(f"Mode: {mode}")
        logger.info(f"Grid size: {grid_size}")
        logger.info(f"Baseline loss: {baseline_loss:.6f}")

        # 1D data
        loss_line_1d = data.get("loss_line_1d", [])
        if loss_line_1d:
            loss_array = np.array(loss_line_1d)
            loss_min = float(loss_array.min())
            loss_max = float(loss_array.max())
            loss_mean = float(loss_array.mean())
            logger.info(f"[1D] Loss range: [{loss_min:.6f}, {loss_max:.6f}]")
            logger.info(f"[1D] Average loss: {loss_mean:.6f}")
            logger.info(f"[1D] Points: {len(loss_line_1d)}")

        # 2D data
        loss_grid_2d = data.get("loss_grid_2d", [])
        if loss_grid_2d:
            loss_array = np.array(loss_grid_2d)
            loss_min = float(loss_array.min())
            loss_max = float(loss_array.max())
            loss_mean = float(loss_array.mean())
            logger.info(f"[2D] Loss range: [{loss_min:.6f}, {loss_max:.6f}]")
            logger.info(f"[2D] Average loss: {loss_mean:.6f}")
            logger.info(f"[2D] Grid shape: {loss_array.shape}")

        # 3D data
        loss_grid_3d = data.get("loss_grid_3d", [])
        if loss_grid_3d:
            loss_array = np.array(loss_grid_3d)
            loss_min = float(loss_array.min())
            loss_max = float(loss_array.max())
            loss_mean = float(loss_array.mean())
            logger.info(f"[3D] Loss range: [{loss_min:.6f}, {loss_max:.6f}]")
            logger.info(f"[3D] Average loss: {loss_mean:.6f}")
            logger.info(f"[3D] Grid shape: {loss_array.shape}")

        # Trajectory data
        trajectory_data = data.get("trajectory_data")
        if trajectory_data:
            epochs = trajectory_data.get("epochs", [])
            if epochs:
                epoch_min = min(epochs)
                epoch_max = max(epochs)
                logger.info(f"Trajectory points: {len(epochs)}")
                logger.info(f"Epoch range: {epoch_min} - {epoch_max}")

        logger.info("=" * 60)

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        raise click.ClickException(f"Invalid JSON file: {e}")
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        raise click.ClickException(f"View failed: {e}")


@cli.command()
def example():
    """Run the complete example demonstrating all Loss Landscape features"""
    import subprocess
    import sys
    from pathlib import Path

    # Get the path to complete_example.py
    # __file__ is loss_landscape/cli.py, so parent is loss_landscape/, and examples is in loss_landscape/examples/
    examples_dir = Path(__file__).parent / "examples"
    example_file = examples_dir / "complete_example.py"

    if not example_file.exists():
        logger.error(f"Example file not found: {example_file}")
        raise click.ClickException(f"Example file not found: {example_file}")

    logger.info("Running complete example...")
    logger.info(f"Example file: {example_file}")
    logger.info("=" * 60)

    # Run the example
    try:
        subprocess.run([sys.executable, str(example_file)], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Example failed with exit code {e.returncode}")
        raise click.ClickException(f"Example failed: {e}")
    except Exception as e:
        logger.error(f"Failed to run example: {e}")
        raise click.ClickException(f"Failed to run example: {e}")
