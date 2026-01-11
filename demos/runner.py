"""
CLI Runner for Loss Landscape Demos.

Usage:
    # List available projects and configs
    python -m demos.runner list

    # Run a specific experiment
    python -m demos.runner run --project mnist --config mlp_small

    # Run with overrides
    python -m demos.runner run --project mnist --config mlp_small \
        --override training.epochs=100

    # Run multiple experiments (comma-separated config names)
    python -m demos.runner run --project mnist \
        --config mlp_small,mlp_large,cnn_simple
"""

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from loguru import logger

from .base.utils import apply_overrides, find_configs, load_config


def get_projects_dir() -> Path:
    """Get the projects directory path."""
    return Path(__file__).parent / "projects"


def discover_projects() -> Dict[str, Dict[str, Any]]:
    """
    Discover all available projects.

    Returns:
        Dictionary mapping project names to their info
    """
    projects_dir = get_projects_dir()
    projects = {}

    for project_path in projects_dir.iterdir():
        if project_path.is_dir() and not project_path.name.startswith("_"):
            # Check if it's a valid project (has __init__.py)
            if (project_path / "__init__.py").exists():
                configs = find_configs(project_path)
                projects[project_path.name] = {
                    "path": str(project_path),
                    "configs": [c.stem for c in configs],
                }

    return projects


def load_project(project_name: str):
    """
    Load and import a project module.

    Args:
        project_name: Name of the project (e.g., "mnist")

    Returns:
        The project module
    """
    try:
        module = importlib.import_module(f"demos.projects.{project_name}")
        return module
    except ImportError as e:
        raise click.ClickException(f"Failed to import project '{project_name}': {e}")


def get_experiment_class(project_name: str):
    """Get the experiment class for a project."""
    from .base import registry

    # Import the project to register it
    load_project(project_name)

    # Get from registry
    project_info = registry._projects.get(project_name)
    if not project_info:
        raise click.ClickException(f"Project '{project_name}' not registered")

    return project_info.get("experiment_class")


def run_experiment(
    project: str,
    config_name: str,
    overrides: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a single experiment.

    Args:
        project: Project name (e.g., "mnist")
        config_name: Config file name (without .yaml)
        overrides: List of config overrides
        output_dir: Optional output directory override

    Returns:
        Experiment results dictionary
    """
    projects_dir = get_projects_dir()
    project_path = projects_dir / project

    if not project_path.exists():
        raise click.ClickException(f"Project '{project}' not found at {project_path}")

    # Find config file
    config_path = project_path / "configs" / f"{config_name}.yaml"
    if not config_path.exists():
        # Try .yml extension
        config_path = project_path / "configs" / f"{config_name}.yml"
        if not config_path.exists():
            available = find_configs(project_path)
            available_names = [c.stem for c in available]
            raise click.ClickException(
                f"Config '{config_name}' not found. Available: {available_names}"
            )

    # Load config
    config = load_config(config_path)

    # Apply overrides
    if overrides:
        config = apply_overrides(config, overrides)

    # Get experiment class
    experiment_cls = get_experiment_class(project)
    if not experiment_cls:
        raise click.ClickException(f"No experiment class found for project '{project}'")

    # Create and run experiment
    exp = experiment_cls(config, output_dir=output_dir)
    results = exp.run()

    return results


def run_from_file(
    config_path: str,
    overrides: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run an experiment directly from a YAML config file.

    The config file must specify the project in experiment.project field.

    Args:
        config_path: Path to the YAML config file
        overrides: List of config overrides
        output_dir: Optional output directory override

    Returns:
        Experiment results dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise click.ClickException(f"Config file not found: {config_path}")

    # Load config
    config = load_config(config_path)

    # Apply overrides
    if overrides:
        config = apply_overrides(config, overrides)

    # Get project name from config
    project = config.get("experiment", {}).get("project")
    if not project:
        raise click.ClickException(
            f"Config file must specify 'experiment.project' field: {config_path}"
        )

    # Get experiment class
    experiment_cls = get_experiment_class(project)
    if not experiment_cls:
        raise click.ClickException(f"No experiment class found for project '{project}'")

    # Create and run experiment
    exp = experiment_cls(config, output_dir=output_dir)
    results = exp.run()

    return results


# ==================== CLI Commands ====================


@click.group()
def cli():
    """Loss Landscape Demo Runner - Run ML experiments with loss landscape analysis."""
    pass


@cli.command("list")
@click.option("--project", "-p", help="Show details for specific project")
def list_projects(project: Optional[str]):
    """List available projects and configurations."""
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
        click.echo(f"\nAvailable configs:")
        for config in info["configs"]:
            click.echo(f"  - {config}")

        # Try to load and show project description
        try:
            module = load_project(project)
            if module.__doc__:
                click.echo(f"\nDescription:")
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

        click.echo("Use 'demos list --project <name>' for more details.")


@cli.command("run")
@click.option("--project", "-p", required=True, help="Project name (e.g., mnist)")
@click.option(
    "--config",
    "-c",
    required=True,
    help="Config name(s) - single name or comma-separated (e.g., mlp_small or mlp_small,mlp_large)",
)
@click.option(
    "--override",
    "-o",
    multiple=True,
    help="Config overrides (e.g., training.epochs=100)",
)
@click.option("--output-dir", "-d", help="Output directory override")
def run_cmd(project: str, config: str, override: tuple, output_dir: Optional[str]):
    """Run experiment(s) from project config(s)."""
    # Parse comma-separated config names
    config_list = [c.strip() for c in config.split(",")]

    # Single config - use original single-experiment output format
    if len(config_list) == 1:
        config_name = config_list[0]
        click.echo(f"\n{'='*60}")
        click.echo(f"Running experiment: {project}/{config_name}")
        click.echo(f"{'='*60}\n")

        try:
            results = run_experiment(
                project=project,
                config_name=config_name,
                overrides=list(override) if override else None,
                output_dir=output_dir,
            )

            click.echo(f"\n{'='*60}")
            click.echo("Experiment Complete!")
            click.echo(f"{'='*60}")
            click.echo(f"Output: {results['output_dir']}")
            if results.get("landscape_path"):
                click.echo(f"Landscape: {results['landscape_path']}")
            if results.get("final_results"):
                fr = results["final_results"]
                if "test_accuracy" in fr:
                    click.echo(f"Test Accuracy: {fr['test_accuracy']:.2f}%")

        except Exception as e:
            logger.exception("Experiment failed")
            raise click.ClickException(str(e))
    else:
        # Multiple configs - use batch format
        click.echo(f"\n{'='*60}")
        click.echo(f"Running batch: {len(config_list)} experiments")
        click.echo(f"{'='*60}\n")

        results = []
        for i, config_name in enumerate(config_list, 1):
            click.echo(f"\n[{i}/{len(config_list)}] Running: {config_name}")
            click.echo("-" * 40)

            try:
                result = run_experiment(
                    project=project,
                    config_name=config_name,
                    overrides=list(override) if override else None,
                    output_dir=output_dir,
                )
                results.append({"config": config_name, "status": "success", "result": result})
                click.echo(f"✓ {config_name} completed")
            except Exception as e:
                results.append({"config": config_name, "status": "failed", "error": str(e)})
                click.echo(f"✗ {config_name} failed: {e}")

        # Summary
        click.echo(f"\n{'='*60}")
        click.echo("Batch Summary")
        click.echo(f"{'='*60}")

        success = sum(1 for r in results if r["status"] == "success")
        failed = len(results) - success

        click.echo(f"Total: {len(results)}, Success: {success}, Failed: {failed}")

        for r in results:
            status = "✓" if r["status"] == "success" else "✗"
            click.echo(f"  {status} {r['config']}")


@cli.command("run-all")
@click.option("--project", "-p", required=True, help="Project name")
@click.option("--override", "-o", multiple=True, help="Config overrides (applied to all)")
def run_all(project: str, override: tuple):
    """Run all configurations for a project."""
    projects = discover_projects()

    if project not in projects:
        raise click.ClickException(f"Project '{project}' not found")

    config_list = projects[project]["configs"]

    if not config_list:
        raise click.ClickException(f"No configs found for project '{project}'")

    click.echo(f"\nRunning all {len(config_list)} configs for {project}...")

    # Use run command with comma-separated configs
    ctx = click.get_current_context()
    ctx.invoke(run_cmd, project=project, config=",".join(config_list), override=override)


def main():
    """Entry point for the runner."""
    cli()


if __name__ == "__main__":
    main()
