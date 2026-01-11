"""
Loss Landscape Analysis Demos

A collection of ML projects for systematic loss landscape analysis across
different datasets, model architectures, optimizers, and training configurations.

Usage:
    # List available projects and configs
    losslandscape demo list

    # Run a specific experiment from YAML file
    losslandscape demo run config.yaml

    # Run with overrides
    losslandscape demo run config.yaml -o training.epochs=100
"""

from .base import BaseExperiment, registry

__all__ = ["BaseExperiment", "registry"]
