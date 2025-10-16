"""
Experiment tracking utilities for the Singer Classification project.
Supports both WandB and local logging.
"""

import os
import json
import time
import logging
import wandb
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from omegaconf import DictConfig, OmegaConf
import torch
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Unified experiment tracking interface supporting multiple backends."""

    def __init__(
        self,
        config: DictConfig,
        experiment_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        use_wandb: bool = False
    ):
        """
        Initialize experiment tracker.

        Args:
            config: Configuration object
            experiment_name: Name for the experiment
            tags: List of tags for organizing experiments
            notes: Description/notes for the experiment
            use_wandb: Whether to use WandB (requires API key)
        """
        self.config = config
        self.use_wandb = use_wandb and self._check_wandb_available()

        # Generate experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"{config.experiment.type}_{config.experiment.name}_{timestamp}"

        self.experiment_name = experiment_name
        self.tags = tags or []
        self.notes = notes or ""

        # Setup logging directories
        self.log_dir = Path(config.logging.log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tracking backends
        self.wandb_run = None
        self.local_metrics = []
        self.start_time = time.time()

        self._setup_logging()
        self._initialize_tracking()

    def _check_wandb_available(self) -> bool:
        """Check if WandB is available and configured."""
        try:
            # Check if wandb API key is set
            api_key = os.getenv('WANDB_API_KEY')
            if api_key:
                return True

            # Try to get key from wandb config
            try:
                import wandb
                wandb.login()
                return True
            except Exception:
                logger.warning("WandB not configured. Using local logging only.")
                return False

        except ImportError:
            logger.warning("WandB not installed. Using local logging only.")
            return False

    def _setup_logging(self):
        """Setup file logging for the experiment."""
        log_file = self.log_dir / "experiment.log"

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)

        # Add handler to logger
        experiment_logger = logging.getLogger('experiment')
        experiment_logger.addHandler(file_handler)
        experiment_logger.setLevel(logging.INFO)

        self.logger = experiment_logger

    def _initialize_tracking(self):
        """Initialize tracking backends."""
        # Initialize WandB if available
        if self.use_wandb:
            try:
                self.wandb_run = wandb.init(
                    project=self.config.wandb.project,
                    entity=self.config.wandb.get('entity'),
                    name=self.experiment_name,
                    tags=self.tags,
                    notes=self.notes,
                    config=OmegaConf.to_container(self.config, resolve=True),
                    dir=str(self.log_dir)
                )
                logger.info(f"WandB run initialized: {self.wandb_run.url}")
            except Exception as e:
                logger.warning(f"Failed to initialize WandB: {e}")
                self.use_wandb = False

        # Save config to local files
        config_file = self.log_dir / "config.yaml"
        with open(config_file, 'w') as f:
            OmegaConf.save(self.config, f)

        # Create experiment metadata
        metadata = {
            "experiment_name": self.experiment_name,
            "start_time": datetime.now().isoformat(),
            "tags": self.tags,
            "notes": self.notes,
            "config_path": str(config_file),
            "use_wandb": self.use_wandb,
            "wandb_url": self.wandb_run.url if self.wandb_run else None
        }

        metadata_file = self.log_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Experiment '{self.experiment_name}' initialized")
        self.logger.info(f"Log directory: {self.log_dir}")

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics to all tracking backends.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number (epoch, batch, etc.)
        """
        # Add timestamp
        timestamp = time.time()
        metrics_with_meta = {
            **metrics,
            "timestamp": timestamp,
            "elapsed_time": timestamp - self.start_time
        }

        if step is not None:
            metrics_with_meta["step"] = step

        # Log to WandB
        if self.use_wandb and self.wandb_run:
            try:
                self.wandb_run.log(metrics, step=step)
            except Exception as e:
                logger.warning(f"Failed to log to WandB: {e}")

        # Log to local storage
        self.local_metrics.append(metrics_with_meta)

        # Log to file
        metric_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in metrics.items()])
        self.logger.info(f"Step {step}: {metric_str}")

        # Save local metrics periodically
        if len(self.local_metrics) % 10 == 0:
            self._save_local_metrics()

    def log_model_info(self, model: torch.nn.Module, input_shape: Optional[tuple] = None):
        """
        Log model architecture information.

        Args:
            model: PyTorch model
            input_shape: Input shape for parameter counting
        """
        try:
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            model_info = {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
            }

            # Log model architecture as text
            model_summary = str(model)

            # Save model info
            if self.use_wandb and self.wandb_run:
                self.wandb_run.log(model_info)
                # Log model architecture as text
                self.wandb_run.log({"model_architecture": wandb.Html(f"<pre>{model_summary}</pre>")})

            # Save locally
            model_info_file = self.log_dir / "model_info.json"
            with open(model_info_file, 'w') as f:
                json.dump(model_info, f, indent=2)

            architecture_file = self.log_dir / "model_architecture.txt"
            with open(architecture_file, 'w') as f:
                f.write(model_summary)

            self.logger.info(f"Model info: {total_params:,} total parameters, {trainable_params:,} trainable")

        except Exception as e:
            logger.warning(f"Failed to log model info: {e}")

    def log_audio_sample(self, audio: torch.Tensor, sample_rate: int, caption: str = ""):
        """
        Log audio sample to WandB (if available).

        Args:
            audio: Audio tensor
            sample_rate: Sample rate
            caption: Caption for the audio
        """
        if self.use_wandb and self.wandb_run:
            try:
                # Convert to numpy if needed
                if isinstance(audio, torch.Tensor):
                    audio_np = audio.detach().cpu().numpy()
                else:
                    audio_np = audio

                self.wandb_run.log({
                    f"audio/{caption}": wandb.Audio(audio_np, sample_rate=sample_rate, caption=caption)
                })
            except Exception as e:
                logger.warning(f"Failed to log audio: {e}")

    def log_image(self, image: Union[plt.Figure, str, torch.Tensor], name: str, caption: str = ""):
        """
        Log image/plot to tracking backends.

        Args:
            image: Image as matplotlib figure, file path, or tensor
            name: Name for the image
            caption: Caption for the image
        """
        try:
            if self.use_wandb and self.wandb_run:
                if isinstance(image, plt.Figure):
                    self.wandb_run.log({name: wandb.Image(image, caption=caption)})
                elif isinstance(image, str) and Path(image).exists():
                    self.wandb_run.log({name: wandb.Image(image, caption=caption)})
                elif isinstance(image, torch.Tensor):
                    self.wandb_run.log({name: wandb.Image(image, caption=caption)})

            # Save locally
            if isinstance(image, plt.Figure):
                image_file = self.log_dir / f"{name}.png"
                image.savefig(image_file, dpi=150, bbox_inches='tight')
                self.logger.info(f"Saved image: {image_file}")

        except Exception as e:
            logger.warning(f"Failed to log image '{name}': {e}")

    def log_confusion_matrix(self, y_true: List, y_pred: List, class_names: List[str], title: str = "Confusion Matrix"):
        """
        Log confusion matrix as image and metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            title: Title for the plot
        """
        try:
            from sklearn.metrics import confusion_matrix, classification_report
            import seaborn as sns

            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred)

            # Create plot
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.title(title)
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()

            # Log the plot
            self.log_image(plt.gcf(), "confusion_matrix", title)
            plt.close()

            # Generate classification report
            report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

            # Log classification metrics
            if self.use_wandb and self.wandb_run:
                # Log per-class metrics
                for class_name in class_names:
                    if class_name in report:
                        class_metrics = report[class_name]
                        for metric, value in class_metrics.items():
                            if isinstance(value, (int, float)):
                                self.wandb_run.log({f"class_{class_name}_{metric}": value})

                # Log overall metrics
                for metric in ['accuracy', 'macro avg', 'weighted avg']:
                    if metric in report:
                        if isinstance(report[metric], dict):
                            for sub_metric, value in report[metric].items():
                                if isinstance(value, (int, float)):
                                    self.wandb_run.log({f"{metric}_{sub_metric}": value})
                        else:
                            self.wandb_run.log({metric: report[metric]})

            # Save classification report locally
            report_file = self.log_dir / "classification_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to log confusion matrix: {e}")

    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       epoch: int, metrics: Dict[str, float], filename: Optional[str] = None):
        """
        Save model checkpoint.

        Args:
            model: PyTorch model
            optimizer: Optimizer
            epoch: Current epoch
            metrics: Current metrics
            filename: Optional filename (auto-generated if None)
        """
        try:
            if filename is None:
                filename = f"checkpoint_epoch_{epoch:03d}.pt"

            checkpoint_path = self.log_dir / "checkpoints"
            checkpoint_path.mkdir(exist_ok=True)

            checkpoint_file = checkpoint_path / filename

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'config': OmegaConf.to_container(self.config, resolve=True),
                'experiment_name': self.experiment_name,
            }

            torch.save(checkpoint, checkpoint_file)

            # Save best model separately if this is the best so far
            if 'val_accuracy' in metrics:
                best_file = checkpoint_path / "best_model.pt"
                if not best_file.exists() or self._is_better_checkpoint(checkpoint_file, best_file):
                    torch.save(checkpoint, best_file)
                    self.logger.info(f"Saved new best model: {metrics.get('val_accuracy', 0):.4f}")

            # Log to WandB if enabled
            if self.use_wandb and self.wandb_run and self.config.wandb.log_model:
                artifact = wandb.Artifact(f"model-{self.experiment_name}", type="model")
                artifact.add_file(str(checkpoint_file))
                self.wandb_run.log_artifact(artifact)

            self.logger.info(f"Saved checkpoint: {checkpoint_file}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def _is_better_checkpoint(self, new_checkpoint_path: Path, best_checkpoint_path: Path) -> bool:
        """Check if new checkpoint is better than current best."""
        try:
            new_checkpoint = torch.load(new_checkpoint_path, map_location='cpu')
            best_checkpoint = torch.load(best_checkpoint_path, map_location='cpu')

            new_acc = new_checkpoint['metrics'].get('val_accuracy', 0)
            best_acc = best_checkpoint['metrics'].get('val_accuracy', 0)

            return new_acc > best_acc
        except:
            return True  # If we can't compare, assume new is better

    def _save_local_metrics(self):
        """Save local metrics to file."""
        try:
            metrics_file = self.log_dir / "metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.local_metrics, f, indent=2)

            # Also save as CSV for easy analysis
            if self.local_metrics:
                df = pd.DataFrame(self.local_metrics)
                csv_file = self.log_dir / "metrics.csv"
                df.to_csv(csv_file, index=False)

        except Exception as e:
            logger.warning(f"Failed to save local metrics: {e}")

    def create_summary(self) -> Dict[str, Any]:
        """Create experiment summary."""
        end_time = time.time()
        duration = end_time - self.start_time

        from datetime import timedelta

        summary = {
            "experiment_name": self.experiment_name,
            "duration_seconds": duration,
            "duration_formatted": str(timedelta(seconds=int(duration))),
            "total_steps": len(self.local_metrics),
            "config": OmegaConf.to_container(self.config, resolve=True),
            "log_directory": str(self.log_dir),
        }

        # Add final metrics if available
        if self.local_metrics:
            final_metrics = self.local_metrics[-1]
            summary["final_metrics"] = {k: v for k, v in final_metrics.items()
                                      if k not in ['timestamp', 'elapsed_time', 'step']}

        return summary

    def finish(self):
        """Finish the experiment and cleanup."""
        try:
            # Save final metrics
            self._save_local_metrics()

            # Create and save summary
            summary = self.create_summary()
            summary_file = self.log_dir / "summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

            # Finish WandB run
            if self.use_wandb and self.wandb_run:
                self.wandb_run.log({"experiment_summary": summary})
                self.wandb_run.finish()

            self.logger.info(f"Experiment '{self.experiment_name}' completed")
            self.logger.info(f"Duration: {summary['duration_formatted']}")
            self.logger.info(f"Results saved to: {self.log_dir}")

        except Exception as e:
            logger.error(f"Error finishing experiment: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()