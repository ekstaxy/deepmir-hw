"""
Configuration module for singer classification project.
"""

from .config_utils import (
    load_config,
    merge_configs,
    load_experiment_config,
    validate_config,
    save_config,
    update_config_paths,
    create_experiment_dirs,
    print_config
)
from .device_utils import (
    setup_device,
    get_optimal_batch_size,
    get_optimal_num_workers,
    print_device_info
)

__all__ = [
    'load_config',
    'merge_configs',
    'load_experiment_config',
    'validate_config',
    'save_config',
    'update_config_paths',
    'create_experiment_dirs',
    'print_config',
    'setup_device',
    'get_optimal_batch_size',
    'get_optimal_num_workers',
    'print_device_info'
]