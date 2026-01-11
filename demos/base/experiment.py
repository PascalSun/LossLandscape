"""
Base experiment class for running ML experiments with loss landscape analysis.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader

from loss_landscape import Explorer, LandscapeStorage

from .registry import registry
from .utils import (
    count_parameters,
    create_output_dir,
    get_device,
    get_system_info,
    save_config,
    set_seed,
)


class BaseExperiment:
    """
    Base class for running ML experiments with loss landscape analysis.

    Subclasses should implement:
        - setup_data(): Create data loaders
        - setup_model(): Create model instance

    Example:
        class MNISTExperiment(BaseExperiment):
            def setup_data(self):
                # Load MNIST data
                return train_loader, val_loader, test_loader

            def setup_model(self):
                # Create model
                return MyModel(self.config["model"]["params"])

        # Run experiment
        exp = MNISTExperiment(config)
        exp.run()
    """

    def __init__(self, config: Dict[str, Any], output_dir: Optional[Path] = None):
        """
        Initialize the experiment.

        Args:
            config: Experiment configuration dictionary
            output_dir: Output directory (auto-created if not provided)
        """
        self.config = config
        self.device = get_device()

        # Set random seed
        seed = config.get("experiment", {}).get("seed", 42)
        set_seed(seed)

        # Setup output directory
        if output_dir is None:
            output_dir = create_output_dir(
                base_dir="outputs",
                project=config.get("experiment", {}).get("project", ""),
                config_name=config.get("experiment", {}).get("name", "experiment"),
            )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config snapshot
        save_config(config, self.output_dir / "config.yaml")

        # Placeholders (initialized in setup)
        self.model: Optional[nn.Module] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.loss_fn: Optional[Callable] = None

        # Training state
        self.current_epoch = 0
        self.training_stats: Dict[str, Any] = {
            "loss_history": [],
            "val_loss_history": [],
            "accuracy_history": [],
            "val_accuracy_history": [],
            "learning_rate_history": [],
            "initial_loss": None,
            "final_loss": None,
            "min_loss": float("inf"),
            "min_loss_epoch": None,
        }

        # Trajectory recording
        self._trajectory_weights: List[torch.Tensor] = []
        self._trajectory_epochs: List[int] = []
        self._base_weights: Optional[torch.Tensor] = None
        self._explorer: Optional[Explorer] = None

        logger.info(f"Experiment initialized: {config.get('experiment', {}).get('name', 'unnamed')}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Device: {self.device}")

    # ==================== Abstract Methods (Override in Subclass) ====================

    def setup_data(self) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """
        Setup data loaders.

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
            val_loader and test_loader can be None
        """
        raise NotImplementedError("Subclasses must implement setup_data()")

    def setup_model(self) -> nn.Module:
        """
        Create and return the model.

        Returns:
            Model instance
        """
        raise NotImplementedError("Subclasses must implement setup_model()")

    # ==================== Setup Methods ====================

    def setup(self):
        """Full setup: data, model, optimizer, loss."""
        logger.info("Setting up experiment...")

        # Data
        self.train_loader, self.val_loader, self.test_loader = self.setup_data()
        logger.info(f"Data loaded: {len(self.train_loader)} train batches")
        if self.val_loader:
            logger.info(f"  {len(self.val_loader)} val batches")

        # Model
        self.model = self.setup_model().to(self.device)
        param_info = count_parameters(self.model)
        logger.info(f"Model created: {param_info['trainable']:,} trainable parameters")

        # Loss function
        self.loss_fn = self._setup_loss()

        # Optimizer
        self.optimizer = self._setup_optimizer()

        # Scheduler
        self.scheduler = self._setup_scheduler()

        # Initialize base weights for trajectory
        self._backup_weights()

        logger.info("Setup complete!")

    def _setup_loss(self) -> Callable:
        """Setup loss function from config."""
        loss_name = self.config.get("training", {}).get("loss", "cross_entropy")

        # Check if it's a registered loss
        try:
            return registry.get_loss(loss_name)
        except KeyError:
            # Try standard PyTorch losses
            loss_map = {
                "cross_entropy": lambda m, x, y: nn.functional.cross_entropy(m(x), y),
                "mse": lambda m, x, y: nn.functional.mse_loss(m(x), y),
                "nll": lambda m, x, y: nn.functional.nll_loss(m(x), y),
            }
            if loss_name in loss_map:
                return loss_map[loss_name]
            raise ValueError(f"Unknown loss: {loss_name}")

    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer from config."""
        opt_config = self.config.get("training", {}).get("optimizer", {})
        opt_name = opt_config.get("name", "adam").lower()
        lr = opt_config.get("lr", 0.001)
        weight_decay = opt_config.get("weight_decay", 0.0)

        if opt_name == "adam":
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == "adamw":
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == "sgd":
            momentum = opt_config.get("momentum", 0.9)
            return optim.SGD(
                self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
            )
        elif opt_name == "rmsprop":
            return optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

    def _setup_scheduler(self) -> Optional[Any]:
        """Setup learning rate scheduler from config."""
        sched_config = self.config.get("training", {}).get("scheduler")
        if not sched_config:
            return None

        sched_name = sched_config.get("name", "").lower()

        if sched_name == "step_lr":
            step_size = sched_config.get("step_size", 30)
            gamma = sched_config.get("gamma", 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif sched_name == "cosine":
            T_max = sched_config.get("T_max", self.config.get("training", {}).get("epochs", 100))
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif sched_name == "reduce_on_plateau":
            patience = sched_config.get("patience", 10)
            factor = sched_config.get("factor", 0.1)
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, patience=patience, factor=factor
            )
        elif sched_name == "exponential":
            gamma = sched_config.get("gamma", 0.95)
            return optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        else:
            logger.warning(f"Unknown scheduler: {sched_name}, skipping")
            return None

    # ==================== Weight Tracking ====================

    def _backup_weights(self):
        """Backup initial weights for trajectory computation."""
        flat_params = []
        for param in self.model.parameters():
            if param.requires_grad:
                flat_params.append(param.data.flatten().cpu().clone())
        self._base_weights = torch.cat(flat_params)

    def _get_weight_offset(self) -> torch.Tensor:
        """Get current weight offset from initial weights."""
        flat_params = []
        for param in self.model.parameters():
            if param.requires_grad:
                flat_params.append(param.data.flatten().cpu().clone())
        current = torch.cat(flat_params)
        return current - self._base_weights

    def _record_trajectory_point(self, epoch: int):
        """Record current weights as trajectory point."""
        offset = self._get_weight_offset()
        self._trajectory_weights.append(offset)
        self._trajectory_epochs.append(epoch)

    # ==================== Training ====================

    def train_epoch(self) -> Tuple[float, Optional[float]]:
        """
        Train for one epoch.

        Returns:
            Tuple of (train_loss, train_accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model, inputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Compute accuracy if classification
            with torch.no_grad():
                outputs = self.model(inputs)
                if outputs.dim() > 1 and outputs.size(1) > 1:  # Classification
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(targets).sum().item()
                    total += targets.size(0)

        avg_loss = total_loss / len(self.train_loader)
        accuracy = (correct / total * 100) if total > 0 else None

        return avg_loss, accuracy

    def evaluate(self, loader: DataLoader) -> Tuple[float, Optional[float]]:
        """
        Evaluate model on a data loader.

        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                loss = self.loss_fn(self.model, inputs, targets)
                total_loss += loss.item()

                outputs = self.model(inputs)
                if outputs.dim() > 1 and outputs.size(1) > 1:  # Classification
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(targets).sum().item()
                    total += targets.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = (correct / total * 100) if total > 0 else None

        return avg_loss, accuracy

    def train(self) -> Dict[str, Any]:
        """
        Full training loop.

        Returns:
            Training statistics dictionary
        """
        num_epochs = self.config.get("training", {}).get("epochs", 50)
        landscape_config = self.config.get("landscape", {})
        record_trajectory = landscape_config.get("record_trajectory", True)
        trajectory_interval = landscape_config.get("trajectory_interval", 1)

        # Skip training if epochs=0 (no_train mode)
        if num_epochs == 0:
            logger.info("Skipping training (epochs=0, no_train mode)")
            # Evaluate initial loss
            if self.train_loader:
                init_loss, init_acc = self.evaluate(self.train_loader)
                self.training_stats["initial_loss"] = init_loss
                self.training_stats["final_loss"] = init_loss
                logger.info(f"Initial loss: {init_loss:.4f}")
            return self.training_stats

        logger.info(f"Starting training for {num_epochs} epochs...")

        # Record initial point
        if record_trajectory:
            self._record_trajectory_point(-1)

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = None, None
            if self.val_loader:
                val_loss, val_acc = self.evaluate(self.val_loader)

            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if val_loss else train_loss)
                else:
                    self.scheduler.step()

            # Record stats
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.training_stats["loss_history"].append(train_loss)
            self.training_stats["learning_rate_history"].append(current_lr)

            if train_acc is not None:
                self.training_stats["accuracy_history"].append(train_acc)
            if val_loss is not None:
                self.training_stats["val_loss_history"].append(val_loss)
            if val_acc is not None:
                self.training_stats["val_accuracy_history"].append(val_acc)

            # Track best loss
            if epoch == 0:
                self.training_stats["initial_loss"] = train_loss
            if train_loss < self.training_stats["min_loss"]:
                self.training_stats["min_loss"] = train_loss
                self.training_stats["min_loss_epoch"] = epoch

            # Record trajectory
            if record_trajectory and (epoch + 1) % trajectory_interval == 0:
                self._record_trajectory_point(epoch)

            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                msg = f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.4f}"
                if train_acc is not None:
                    msg += f" | Acc: {train_acc:.2f}%"
                if val_loss is not None:
                    msg += f" | Val Loss: {val_loss:.4f}"
                if val_acc is not None:
                    msg += f" | Val Acc: {val_acc:.2f}%"
                msg += f" | LR: {current_lr:.6f}"
                logger.info(msg)

        # Final point
        if record_trajectory:
            self._record_trajectory_point(num_epochs)

        self.training_stats["final_loss"] = self.training_stats["loss_history"][-1]
        if self.training_stats["val_loss_history"]:
            self.training_stats["final_val_loss"] = self.training_stats["val_loss_history"][-1]

        logger.info("Training complete!")
        return self.training_stats

    # ==================== Loss Landscape Generation ====================

    def generate_landscape(self) -> str:
        """
        Generate loss landscape visualization.

        Returns:
            Path to the generated JSON file
        """
        landscape_config = self.config.get("landscape", {})
        if not landscape_config.get("enabled", True):
            logger.info("Landscape generation disabled in config")
            return ""

        logger.info("\n" + "=" * 60)
        logger.info("Generating Loss Landscape...")
        logger.info("=" * 60)

        grid_size_1d = landscape_config.get("grid_size_1d", 100)
        grid_size_2d = landscape_config.get("grid_size_2d", 30)
        grid_size_3d = landscape_config.get("grid_size_3d", 16)
        compute_hessian = landscape_config.get("compute_hessian", True)
        hessian_top_k = landscape_config.get("hessian_top_k", 5)
        directions_mode = landscape_config.get("directions", "pca")

        # Create storage
        landscape_path = self.output_dir / "landscape.landscape"
        storage = LandscapeStorage(str(landscape_path), mode="create")

        # Prepare metadata
        metadata = self._build_metadata()
        storage.save_metadata(metadata)

        # Use a deterministic loader for landscape
        landscape_loader = DataLoader(
            self.train_loader.dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Compute PCA directions if we have trajectory
        pca_directions = None
        range_scale = 0.1

        if self._trajectory_weights and len(self._trajectory_weights) > 1:
            pca_directions, range_scale = self._compute_pca_directions()
            logger.info(f"PCA range_scale: {range_scale:.4f}")

        # Create explorer
        with Explorer(
            self.model,
            self.loss_fn,
            landscape_loader,
            device=self.device,
            storage=storage,
            model_mode="eval",
            max_batches=None,
        ) as explorer:

            # Restore trajectory
            if self._trajectory_weights and len(self._trajectory_weights) > 1:
                self._restore_trajectory(explorer)

            # Direction setup
            dir_1d = pca_directions[0] if pca_directions else None
            dirs_2d = (pca_directions[0], pca_directions[1]) if pca_directions else None
            dirs_3d = pca_directions if pca_directions else None

            # 1D Line
            logger.info(f"[1/5] Generating 1D line (grid_size={grid_size_1d})...")
            explorer.build_line(
                grid_size=grid_size_1d,
                range_scale=range_scale,
                direction=dir_1d,
                verbose=True,
            )

            # 2D Surface
            logger.info(f"[2/5] Generating 2D surface (grid_size={grid_size_2d})...")
            explorer.build_surface(
                grid_size=grid_size_2d,
                range_scale=range_scale,
                directions=dirs_2d,
                verbose=True,
            )

            # 3D Volume
            logger.info(f"[3/5] Generating 3D volume (grid_size={grid_size_3d})...")
            explorer.build_volume(
                grid_size=grid_size_3d,
                range_scale=range_scale,
                directions=dirs_3d,
                verbose=True,
            )

            # Trajectory
            if self._trajectory_weights and len(explorer._trajectory_weights) > 1:
                logger.info("[4/5] Building trajectory...")
                explorer.build_trajectory(mode="fixed", directions=dirs_3d)

                # Hessian along trajectory
                if compute_hessian:
                    logger.info(f"[5/5] Computing Hessian (top_k={hessian_top_k})...")
                    explorer.build_hessian_trajectory(top_k=hessian_top_k, max_batches=5)
            else:
                logger.info("[4/5] Skipping trajectory (not enough points)")
                if compute_hessian:
                    logger.info("[5/5] Computing Hessian snapshot...")
                    explorer.build_hessian_snapshot(epoch=0, top_k=40, max_batches=5)

        # Export
        json_path = self.output_dir / "landscape.json"
        storage.export_for_frontend(str(json_path))
        storage.close()

        # Cleanup intermediate file
        if landscape_path.exists():
            try:
                os.remove(landscape_path)
            except OSError:
                pass

        logger.info(f"Landscape saved to: {json_path}")
        return str(json_path)

    def _compute_pca_directions(self) -> Tuple[Tuple[torch.Tensor, ...], float]:
        """Compute PCA directions from trajectory."""
        offsets = torch.stack(self._trajectory_weights).to(self.device)
        final_offset = offsets[-1]
        centered_data = offsets - final_offset

        U, S, Vh = torch.linalg.svd(centered_data, full_matrices=False)

        k = Vh.shape[0]
        dirs = []
        for i in range(3):
            if i < k:
                d = Vh[i]
            else:
                d = torch.randn_like(Vh[0])
            d = d / d.norm()
            dirs.append(d)

        # Normalize directions
        norm_dirs = [self._normalize_direction_filterwise(d) for d in dirs]

        # Compute range scale
        D = torch.stack(norm_dirs)
        denom = (D * D).sum(dim=1).clamp_min(1e-12)
        proj = torch.matmul(centered_data, D.T) / denom
        max_coord = proj.abs().max().item()
        range_scale = max(max_coord * 1.2, 0.1)

        return tuple(dirs), range_scale

    def _normalize_direction_filterwise(self, direction: torch.Tensor) -> torch.Tensor:
        """Normalize direction using filter-wise normalization."""
        import numpy as np

        normalized_parts = []
        idx = 0
        params = [p for p in self.model.parameters() if p.requires_grad]

        for param in params:
            shape = param.shape
            size = int(np.prod(shape))
            layer_direction = direction[idx : idx + size]

            layer_d = layer_direction.reshape(shape)
            d_norm = torch.norm(layer_d, p="fro")

            scale = 1.0
            if d_norm > 1e-8:
                scale = 1.0 / d_norm
                w_norm = torch.norm(param.data, p="fro")
                scale *= w_norm

            normalized_parts.append(layer_direction * scale)
            idx += size

        return torch.cat(normalized_parts)

    def _restore_trajectory(self, explorer: Explorer):
        """Restore trajectory data to explorer."""
        current_flat = []
        for param in self.model.parameters():
            if param.requires_grad:
                current_flat.append(param.data.flatten().cpu().clone())
        current_flat = torch.cat(current_flat)

        base_offset = current_flat - self._base_weights
        adjusted_weights = [w - base_offset for w in self._trajectory_weights]

        explorer._trajectory_weights = adjusted_weights
        explorer._trajectory_epochs = self._trajectory_epochs.copy()

        # Restore losses
        if self.training_stats.get("loss_history"):
            train_hist = self.training_stats["loss_history"]
            val_hist = self.training_stats.get("val_loss_history", [])

            train_losses = []
            val_losses = []

            for ep in self._trajectory_epochs:
                if ep == -1:
                    init_loss = self.training_stats.get("initial_loss")
                    train_losses.append(float(init_loss) if init_loss else None)
                    val_losses.append(None)
                elif 0 <= ep < len(train_hist):
                    train_losses.append(float(train_hist[ep]))
                    val_losses.append(float(val_hist[ep]) if ep < len(val_hist) else None)
                else:
                    train_losses.append(None)
                    val_losses.append(None)

            explorer._trajectory_losses = train_losses
            explorer._trajectory_val_losses = val_losses

        logger.info(f"Restored {len(self._trajectory_epochs)} trajectory points")

    def _build_metadata(self) -> Dict[str, Any]:
        """Build metadata for the landscape file."""
        config = self.config
        param_info = count_parameters(self.model)

        return {
            "experiment": config.get("experiment", {}),
            "dataset": config.get("dataset", {}),
            "model": {
                **config.get("model", {}),
                "total_params": param_info["total"],
                "trainable_params": param_info["trainable"],
            },
            "training": config.get("training", {}),
            "training_statistics": {
                "initial_loss": self.training_stats.get("initial_loss"),
                "final_loss": self.training_stats.get("final_loss"),
                "min_loss": (
                    self.training_stats["min_loss"]
                    if self.training_stats["min_loss"] != float("inf")
                    else None
                ),
                "min_loss_epoch": self.training_stats.get("min_loss_epoch"),
                "learning_rate_history": self.training_stats.get("learning_rate_history", []),
            },
            "landscape": config.get("landscape", {}),
            "system": get_system_info(),
            "created_at": datetime.now().isoformat(),
        }

    # ==================== Main Entry Point ====================

    def run(self) -> Dict[str, Any]:
        """
        Run the complete experiment: setup, train, generate landscape.

        Returns:
            Dictionary with results including paths to outputs
        """
        logger.info("\n" + "=" * 60)
        logger.info(f"Running Experiment: {self.config.get('experiment', {}).get('name', 'unnamed')}")
        logger.info("=" * 60 + "\n")

        # Setup
        self.setup()

        # Train
        stats = self.train()

        # Generate landscape
        landscape_path = self.generate_landscape()

        # Final evaluation
        final_results = {}
        if self.test_loader:
            test_loss, test_acc = self.evaluate(self.test_loader)
            final_results["test_loss"] = test_loss
            if test_acc is not None:
                final_results["test_accuracy"] = test_acc
                logger.info(f"\nTest Results: Loss={test_loss:.4f}, Accuracy={test_acc:.2f}%")
            else:
                logger.info(f"\nTest Results: Loss={test_loss:.4f}")

        results = {
            "output_dir": str(self.output_dir),
            "landscape_path": landscape_path,
            "training_stats": stats,
            "final_results": final_results,
        }

        # Save results
        save_config(results, self.output_dir / "results.yaml")

        logger.info(f"\nExperiment complete! Results saved to: {self.output_dir}")
        return results
