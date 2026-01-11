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


# ==================== Demo Subcommand Group ====================


@cli.group()
def demo():
    """Run ML experiment demo with loss landscape analysis.

    Examples:
        losslandscape demo list
        losslandscape demo run config.yaml
    """
    pass


@demo.command("list")
@click.option("--project", "-p", help="Show details for specific project")
def demo_list(project):
    """List available projects and configurations."""
    from demos.runner import discover_projects, load_project

    projects = discover_projects()

    if not projects:
        click.echo("No projects found.")
        return

    if project:
        if project not in projects:
            click.echo(f"Project '{project}' not found.")
            click.echo(f"Available projects: {list(projects.keys())}")
            return

        info = projects[project]
        click.echo(f"\n{'='*60}")
        click.echo(f"Project: {project}")
        click.echo(f"{'='*60}")
        click.echo(f"Path: {info['path']}")
        click.echo("\nAvailable configs:")
        for config in info["configs"]:
            click.echo(f"  - {config}")

        try:
            module = load_project(project)
            if module.__doc__:
                click.echo("\nDescription:")
                click.echo(module.__doc__.strip())
        except Exception:
            pass
    else:
        click.echo(f"\n{'='*60}")
        click.echo("Available Projects")
        click.echo(f"{'='*60}\n")

        for name, info in sorted(projects.items()):
            click.echo(f"📁 {name}")
            click.echo(f"   Configs: {', '.join(info['configs'])}")
            click.echo()

        click.echo("Use 'losslandscape demo list --project <name>' for more details.")


@demo.command("run")
@click.argument(
    "files",
    nargs=-1,
    type=click.Path(exists=True),
)
@click.option(
    "--override",
    "-o",
    multiple=True,
    help="Config overrides (e.g., training.epochs=100)",
)
@click.option("--output-dir", "-d", help="Output directory override")
def demo_run(files, override, output_dir):
    """Run experiment(s) from YAML config file(s).

    FILES: YAML config file(s) or folder(s) containing configs

    Examples:

        # Run from a single YAML file
        losslandscape demo run config.yaml

        # Run from multiple YAML files
        losslandscape demo run config1.yaml config2.yaml

        # Run all configs in a folder
        losslandscape demo run ./configs/

        # With overrides
        losslandscape demo run config.yaml -o training.epochs=50
    """
    from pathlib import Path

    from demos.runner import run_from_file

    if not files:
        raise click.ClickException("Please provide at least one YAML config file or folder.")

    # Collect all config files to run
    config_files = []

    for f in files:
        p = Path(f)
        if p.is_dir():
            # Collect all YAML files in directory
            config_files.extend(sorted(p.glob("*.yaml")))
            config_files.extend(sorted(p.glob("*.yml")))
        else:
            config_files.append(p)

    if not config_files:
        raise click.ClickException(
            "No YAML config files found. Please provide valid file(s) or folder(s)."
        )

    # Single file - use detailed output format
    if len(config_files) == 1:
        config_path = config_files[0]
        click.echo(f"\n{'='*60}")
        click.echo(f"Running experiment: {config_path.name}")
        click.echo(f"{'='*60}\n")

        try:
            result = run_from_file(
                config_path=str(config_path),
                overrides=list(override) if override else None,
                output_dir=output_dir,
            )

            click.echo(f"\n{'='*60}")
            click.echo("Experiment Complete!")
            click.echo(f"{'='*60}")
            click.echo(f"Output: {result['output_dir']}")
            if result.get("landscape_path"):
                click.echo(f"Landscape: {result['landscape_path']}")
            if result.get("final_results"):
                fr = result["final_results"]
                if "test_accuracy" in fr:
                    click.echo(f"Test Accuracy: {fr['test_accuracy']:.2f}%")

        except Exception as e:
            logger.exception("Experiment failed")
            raise click.ClickException(str(e))
    else:
        # Multiple files - use batch format
        click.echo(f"\n{'='*60}")
        click.echo(f"Running {len(config_files)} experiment(s) from file(s)")
        click.echo(f"{'='*60}\n")

        # Check if all config files are from the same directory
        # If so, create a unified output directory
        parent_dirs = {f.parent for f in config_files}
        unified_output_base = None

        if len(parent_dirs) == 1 and not output_dir:
            # All files from the same directory - create unified output folder
            parent_dir = parent_dirs.pop()
            from datetime import datetime
            from pathlib import Path

            # Create unified output directory based on parent folder name
            folder_name = parent_dir.name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unified_output_base = Path("outputs") / folder_name / f"batch_{timestamp}"
            unified_output_base.mkdir(parents=True, exist_ok=True)
            logger.info(
                f"All configs from same folder - using unified output: {unified_output_base}"
            )

        results = []
        for i, config_path in enumerate(config_files, 1):
            click.echo(f"\n[{i}/{len(config_files)}] Running: {config_path.name}")
            click.echo("-" * 40)

            try:
                # If unified output, create subdirectory for each config
                config_output_dir = output_dir
                if unified_output_base and not output_dir:
                    # Create subdirectory based on config file name (without extension)
                    config_output_dir = str(unified_output_base / config_path.stem)

                result = run_from_file(
                    config_path=str(config_path),
                    overrides=list(override) if override else None,
                    output_dir=config_output_dir,
                )
                results.append({"file": str(config_path), "status": "success", "result": result})
                click.echo(f"✓ {config_path.name} completed")
                click.echo(f"  Output: {result['output_dir']}")
            except Exception as e:
                results.append({"file": str(config_path), "status": "failed", "error": str(e)})
                click.echo(f"✗ {config_path.name} failed: {e}")
                logger.exception(f"Failed to run {config_path}")

        # Summary
        click.echo(f"\n{'='*60}")
        click.echo("Batch Summary")
        click.echo(f"{'='*60}")

        success = sum(1 for r in results if r["status"] == "success")
        failed = len(results) - success

        click.echo(f"Total: {len(results)}, Success: {success}, Failed: {failed}")

        for r in results:
            status = "✓" if r["status"] == "success" else "✗"
            click.echo(f"  {status} {Path(r['file']).name}")

        # Show unified output directory if used
        if unified_output_base:
            click.echo(f"\n{'='*60}")
            click.echo(f"Unified Output Directory: {unified_output_base}")
            click.echo(f"{'='*60}")
            click.echo("All experiment outputs are organized in the above directory.")


@cli.command()
@click.argument(
    "input",
    type=click.Path(exists=True),
)
def view(input: str):
    """View information about exported Loss Landscape JSON data

    INPUT: JSON file exported from LossLandscape
    """
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
