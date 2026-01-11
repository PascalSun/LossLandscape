"""
Utility functions for the demos framework.
"""

import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import yaml
from loguru import logger


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        config_path: Path to the YAML config file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two configuration dictionaries.

    Args:
        base: Base configuration
        override: Override configuration (takes precedence)

    Returns:
        Merged configuration
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def apply_overrides(config: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """
    Apply command-line overrides to a configuration.

    Args:
        config: Base configuration dictionary
        overrides: List of override strings like "training.epochs=100"

    Returns:
        Modified configuration
    """
    for override in overrides:
        if "=" not in override:
            logger.warning(f"Invalid override format: {override}. Expected 'key=value'")
            continue

        key_path, value_str = override.split("=", 1)
        keys = key_path.split(".")

        # Parse value
        value = parse_value(value_str)

        # Navigate to the nested key and set value
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    return config


def parse_value(value_str: str) -> Any:
    """
    Parse a string value into appropriate Python type.

    Handles: int, float, bool, list, None, str
    """
    value_str = value_str.strip()

    # None
    if value_str.lower() == "none":
        return None

    # Boolean
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False

    # List (simple format: [1,2,3] or [a,b,c])
    if value_str.startswith("[") and value_str.endswith("]"):
        inner = value_str[1:-1].strip()
        if not inner:
            return []
        items = [parse_value(item.strip()) for item in inner.split(",")]
        return items

    # Number
    try:
        if "." in value_str or "e" in value_str.lower():
            return float(value_str)
        return int(value_str)
    except ValueError:
        pass

    # String (remove quotes if present)
    if (value_str.startswith('"') and value_str.endswith('"')) or (
        value_str.startswith("'") and value_str.endswith("'")
    ):
        return value_str[1:-1]

    return value_str


def create_output_dir(
    base_dir: str = "outputs",
    project: str = "",
    config_name: str = "",
    timestamp: bool = True,
) -> Path:
    """
    Create a unique output directory for an experiment.

    Args:
        base_dir: Base output directory
        project: Project name
        config_name: Config name
        timestamp: Whether to include timestamp

    Returns:
        Path to the created directory
    """
    parts = [base_dir]

    if project:
        parts.append(project)

    name_parts = []
    if config_name:
        name_parts.append(config_name)
    if timestamp:
        name_parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))

    if name_parts:
        parts.append("_".join(name_parts))

    output_dir = Path(*parts)
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def save_config(config: Dict[str, Any], path: Union[str, Path]):
    """Save configuration to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def get_system_info() -> Dict[str, Any]:
    """Collect system information for reproducibility."""
    import platform

    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
        if torch.cuda.device_count() > 0:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()

    try:
        import psutil

        info["cpu_count"] = psutil.cpu_count()
        info["memory_total_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
    except ImportError:
        pass

    return info


def find_configs(project_dir: Union[str, Path]) -> List[Path]:
    """Find all YAML config files in a project directory."""
    project_dir = Path(project_dir)
    configs_dir = project_dir / "configs"

    if not configs_dir.exists():
        return []

    return sorted(configs_dir.glob("*.yaml")) + sorted(configs_dir.glob("*.yml"))


def get_default_config() -> Dict[str, Any]:
    """Get default configuration template."""
    return {
        "experiment": {
            "name": "unnamed",
            "description": "",
            "seed": 42,
            "tags": [],
        },
        "dataset": {
            "name": "",
            "batch_size": 128,
            "num_workers": 4,
            "subset": None,
        },
        "model": {
            "name": "",
            "params": {},
        },
        "training": {
            "epochs": 50,
            "optimizer": {
                "name": "adam",
                "lr": 0.001,
                "weight_decay": 0.0,
            },
            "scheduler": None,
            "loss": "cross_entropy",
        },
        "landscape": {
            "enabled": True,
            "grid_size_1d": 100,
            "grid_size_2d": 30,
            "grid_size_3d": 16,
            "record_trajectory": True,
            "trajectory_interval": 1,
            "compute_hessian": True,
            "hessian_top_k": 5,
            "directions": "pca",
        },
    }
