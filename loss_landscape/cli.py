"""
Command Line Interface for LossLandscape
"""

import click
import json
from pathlib import Path
from typing import Optional
from loguru import logger
import numpy as np


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
    help="Input JSON file exported from LossLandscape"
)
def view(input: str):
    """View information about exported Loss Landscape JSON data"""
    logger.info(f"Reading JSON file: {input}...")
    
    try:
        with open(input, 'r') as f:
            data = json.load(f)
        
        logger.info("=" * 60)
        logger.info(f"File: {input}")
        
        # Basic info
        grid_size = data.get('grid_size', 0)
        baseline_loss = data.get('baseline_loss', 0.0)
        mode = data.get('mode', 'unknown')
        
        logger.info(f"Mode: {mode}")
        logger.info(f"Grid size: {grid_size}")
        logger.info(f"Baseline loss: {baseline_loss:.6f}")
        
        # 2D data
        loss_grid_2d = data.get('loss_grid_2d', [])
        if loss_grid_2d:
            loss_array = np.array(loss_grid_2d)
            loss_min = float(loss_array.min())
            loss_max = float(loss_array.max())
            loss_mean = float(loss_array.mean())
            logger.info(f"[2D] Loss range: [{loss_min:.6f}, {loss_max:.6f}]")
            logger.info(f"[2D] Average loss: {loss_mean:.6f}")
            logger.info(f"[2D] Grid shape: {loss_array.shape}")
        
        # 3D data
        loss_grid_3d = data.get('loss_grid_3d', [])
        if loss_grid_3d:
            loss_array = np.array(loss_grid_3d)
            loss_min = float(loss_array.min())
            loss_max = float(loss_array.max())
            loss_mean = float(loss_array.mean())
            logger.info(f"[3D] Loss range: [{loss_min:.6f}, {loss_max:.6f}]")
            logger.info(f"[3D] Average loss: {loss_mean:.6f}")
            logger.info(f"[3D] Grid shape: {loss_array.shape}")
        
        # Trajectory data
        trajectory_data = data.get('trajectory_data')
        if trajectory_data:
            epochs = trajectory_data.get('epochs', [])
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
@click.option(
    "--input",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Input .landscape (DuckDB) file path"
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="Output JSON file path"
)
def export(input: str, output: str):
    """Export .landscape (DuckDB) file to JSON format for frontend"""
    from .core import LandscapeStorage
    import json
    import time
    from datetime import datetime
    
    output_path = Path(output)
    status_file = output_path.with_suffix('.export.meta.json')
    
    # Write status file to indicate export is in progress
    status_data = {
        'status': 'exporting',
        'started_at': datetime.now().isoformat(),
        'input_file': str(input),
        'output_file': str(output),
    }
    try:
        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to write status file: {e}")
    
    logger.info(f"Exporting landscape file: {input} -> {output}")
    start_time = time.time()
    
    try:
        storage = LandscapeStorage(input)
        data = storage.export_for_frontend(output_path=output)
        
        export_time = time.time() - start_time
        
        # Update status file with success information
        status_data.update({
            'status': 'completed',
            'completed_at': datetime.now().isoformat(),
            'export_duration_seconds': round(export_time, 2),
            'metadata': {
                'grid_size_2d': data.get('grid_size', None),
                'grid_size_3d': data.get('grid_size_3d', None),
                'trajectory_points': len(data.get('trajectory_data', {}).get('epochs', [])),
                'baseline_loss': data.get('baseline_loss', None),
                'mode': data.get('mode', None),
            }
        })
        
        try:
            with open(status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to update status file: {e}")
        
        logger.info(f"✓ Successfully exported to: {output}")
        logger.info(f"  - 2D grid size: {data.get('grid_size', 'N/A')}")
        logger.info(f"  - 3D grid size: {data.get('grid_size_3d', 'N/A')}")
        logger.info(f"  - Trajectory points: {len(data.get('trajectory_data', {}).get('epochs', []))}")
        logger.info(f"  - Export time: {export_time:.2f}s")
    except Exception as e:
        # Update status file with error information
        status_data.update({
            'status': 'failed',
            'failed_at': datetime.now().isoformat(),
            'error': str(e),
        })
        try:
            with open(status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
        except Exception:
            pass
        
        logger.error(f"Failed to export: {e}")
        raise click.ClickException(f"Export failed: {e}")


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

