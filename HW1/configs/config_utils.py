"""
Configuration utilities for the singer classification project.
Provides functions to load, validate, and merge configurations.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from omegaconf import OmegaConf, DictConfig
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """
    Load configuration from YAML file using OmegaConf.

    Args:
        config_path: Path to the configuration file

    Returns:
        DictConfig: Loaded configuration
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        config = OmegaConf.load(config_path)
        logger.info(f"Successfully loaded config from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        raise


def merge_configs(*configs: DictConfig) -> DictConfig:
    """
    Merge multiple configurations with later configs taking precedence.

    Args:
        *configs: Variable number of DictConfig objects to merge

    Returns:
        DictConfig: Merged configuration
    """
    if not configs:
        return OmegaConf.create({})

    merged = configs[0]
    for config in configs[1:]:
        merged = OmegaConf.merge(merged, config)

    return merged


def load_experiment_config(
    experiment_type: str,
    experiment_name: str,
    base_config_path: Optional[str] = None
) -> DictConfig:
    """
    Load experiment configuration by type and name.

    Args:
        experiment_type: Type of experiment ('traditional_ml' or 'deep_learning')
        experiment_name: Name of the experiment configuration
        base_config_path: Optional path to base config (defaults to configs/base_config.yaml)

    Returns:
        DictConfig: Complete experiment configuration
    """
    # Default paths
    if base_config_path is None:
        base_config_path = "configs/base_config.yaml"

    experiment_config_path = f"configs/{experiment_type}/{experiment_name}_config.yaml"

    # Load base configuration
    base_config = load_config(base_config_path)

    # Load experiment configuration
    experiment_config = load_config(experiment_config_path)

    # Merge configurations
    final_config = merge_configs(base_config, experiment_config)

    # Set experiment metadata
    final_config.experiment.type = experiment_type
    final_config.experiment.config_path = experiment_config_path

    logger.info(f"Loaded {experiment_type} experiment: {experiment_name}")

    return final_config


def validate_config(config: DictConfig) -> bool:
    """
    Validate configuration for required fields and logical consistency.

    Args:
        config: Configuration to validate

    Returns:
        bool: True if valid, raises exception if invalid
    """
    required_sections = [
        "dataset", "audio", "training", "evaluation",
        "logging", "device", "paths"
    ]

    # Check required sections
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")

    # Validate dataset paths
    dataset_config = config.dataset
    required_dataset_fields = ["train_json", "val_json", "test_dir", "num_classes"]

    for field in required_dataset_fields:
        if field not in dataset_config:
            raise ValueError(f"Missing required dataset field: {field}")

    # Validate paths exist
    paths_to_check = [
        dataset_config.train_json,
        dataset_config.val_json,
        dataset_config.test_dir
    ]

    for path in paths_to_check:
        if not os.path.exists(path):
            logger.warning(f"Path does not exist: {path}")

    # Validate audio settings
    audio_config = config.audio
    if audio_config.sample_rate <= 0:
        raise ValueError("Sample rate must be positive")

    if audio_config.n_fft <= 0:
        raise ValueError("n_fft must be positive")

    # Validate training settings
    training_config = config.training
    if training_config.batch_size is not None and training_config.batch_size <= 0:
        raise ValueError("Batch size must be positive")

    if training_config.learning_rate is not None and training_config.learning_rate <= 0:
        raise ValueError("Learning rate must be positive")

    logger.info("Configuration validation passed")
    return True


def save_config(config: DictConfig, save_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration to save
        save_path: Path where to save the configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(save_path, 'w') as f:
            OmegaConf.save(config, f)
        logger.info(f"Configuration saved to {save_path}")
    except Exception as e:
        logger.error(f"Error saving config to {save_path}: {e}")
        raise


def update_config_paths(config: DictConfig, root_dir: Optional[str] = None) -> DictConfig:
    """
    Update relative paths in configuration to absolute paths.

    Args:
        config: Configuration to update
        root_dir: Root directory for relative paths (defaults to current directory)

    Returns:
        DictConfig: Updated configuration
    """
    if root_dir is None:
        root_dir = os.getcwd()

    root_path = Path(root_dir)

    # Update dataset paths
    if not os.path.isabs(config.dataset.train_json):
        config.dataset.train_json = str(root_path / config.dataset.train_json)

    if not os.path.isabs(config.dataset.val_json):
        config.dataset.val_json = str(root_path / config.dataset.val_json)

    if not os.path.isabs(config.dataset.test_dir):
        config.dataset.test_dir = str(root_path / config.dataset.test_dir)

    # Update output paths
    for path_key in config.paths:
        if not os.path.isabs(config.paths[path_key]):
            config.paths[path_key] = str(root_path / config.paths[path_key])

    return config


def create_experiment_dirs(config: DictConfig) -> None:
    """
    Create necessary directories for the experiment.

    Args:
        config: Configuration containing path information
    """
    dirs_to_create = [
        config.paths.checkpoints,
        config.paths.predictions,
        config.paths.visualizations,
        config.paths.processed_data,
        config.paths.features,
        config.logging.log_dir
    ]

    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {dir_path}")

    logger.info("All experiment directories created")


def print_config(config: DictConfig, title: str = "Configuration") -> None:
    """
    Pretty print configuration for debugging.

    Args:
        config: Configuration to print
        title: Title for the printout
    """
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    print(OmegaConf.to_yaml(config))
    print(f"{'='*50}\n")


# Example usage and testing
if __name__ == "__main__":
    # Test configuration loading
    try:
        # Load traditional ML config
        ml_config = load_experiment_config("traditional_ml", "baseline")
        validate_config(ml_config)
        print_config(ml_config, "Traditional ML Config")

        # Load deep learning config
        dl_config = load_experiment_config("deep_learning", "baseline")
        validate_config(dl_config)
        print_config(dl_config, "Deep Learning Config")

        print("Configuration system test passed!")

    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        raise