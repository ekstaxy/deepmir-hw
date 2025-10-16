"""
Device utilities for handling GPU/CPU configuration across different environments.
"""

import torch
import logging
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def setup_device(config: DictConfig) -> torch.device:
    """
    Setup device (GPU/CPU) based on configuration and availability.

    Args:
        config: Configuration object with device settings

    Returns:
        torch.device: Device to use for training/inference
    """
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()

    # Handle different use_cuda settings
    use_cuda = config.device.use_cuda

    if config.device.force_cpu:
        device = torch.device("cpu")
        logger.info("Forced CPU usage via configuration")

    elif use_cuda == "auto":
        if cuda_available:
            device_id = config.device.get("device_id", 0)
            device = torch.device(f"cuda:{device_id}")
            logger.info(f"Auto-detected CUDA device: {device}")

            # Log GPU information
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(device_id)
                gpu_memory = torch.cuda.get_device_properties(device_id).total_memory / 1e9
                logger.info(f"GPU: {gpu_name}, Memory: {gpu_memory:.1f} GB")
        else:
            device = torch.device("cpu")
            logger.info("CUDA not available, using CPU")

    elif use_cuda is True:
        if cuda_available:
            device_id = config.device.get("device_id", 0)
            device = torch.device(f"cuda:{device_id}")
            logger.info(f"Using specified CUDA device: {device}")
        else:
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = torch.device("cpu")

    else:  # use_cuda is False
        device = torch.device("cpu")
        logger.info("Using CPU as specified in configuration")

    return device


def get_optimal_batch_size(config: DictConfig, device: torch.device) -> int:
    """
    Get optimal batch size based on device and available memory.

    Args:
        config: Configuration object
        device: Device being used

    Returns:
        int: Optimal batch size
    """
    base_batch_size = config.training.batch_size

    if device.type == "cpu":
        # Reduce batch size for CPU to avoid memory issues
        optimal_batch_size = min(base_batch_size, 16)
        if optimal_batch_size != base_batch_size:
            logger.info(f"Reduced batch size from {base_batch_size} to {optimal_batch_size} for CPU")
    else:
        # For GPU, check available memory
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(device).total_memory / 1e9

            if gpu_memory_gb < 8:  # Less than 8GB
                optimal_batch_size = min(base_batch_size, 16)
                if optimal_batch_size != base_batch_size:
                    logger.info(f"Reduced batch size from {base_batch_size} to {optimal_batch_size} for GPU with {gpu_memory_gb:.1f}GB memory")
            else:
                optimal_batch_size = base_batch_size
        else:
            optimal_batch_size = base_batch_size

    return optimal_batch_size


def get_optimal_num_workers(config: DictConfig, device: torch.device) -> int:
    """
    Get optimal number of workers for data loading.

    Args:
        config: Configuration object
        device: Device being used

    Returns:
        int: Optimal number of workers
    """
    base_num_workers = config.device.num_workers

    # Check if running in Colab
    try:
        import google.colab
        in_colab = True
    except ImportError:
        in_colab = False

    if in_colab:
        # Colab works better with fewer workers
        optimal_workers = min(base_num_workers, 2)
        if optimal_workers != base_num_workers:
            logger.info(f"Reduced num_workers from {base_num_workers} to {optimal_workers} for Colab")
    elif device.type == "cpu":
        # CPU can handle more workers for data loading
        optimal_workers = min(base_num_workers, 4)
    else:
        optimal_workers = base_num_workers

    return optimal_workers


def print_device_info(device: torch.device) -> None:
    """
    Print detailed device information.

    Args:
        device: Device to print information about
    """
    print("\n" + "="*50)
    print("DEVICE CONFIGURATION")
    print("="*50)

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Count: {torch.cuda.device_count()}")

            gpu_id = device.index if device.index is not None else 0
            if gpu_id < torch.cuda.device_count():
                props = torch.cuda.get_device_properties(gpu_id)
                print(f"GPU Name: {props.name}")
                print(f"GPU Memory: {props.total_memory / 1e9:.1f} GB")
                print(f"GPU Compute Capability: {props.major}.{props.minor}")

                # Check available memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # Clear cache
                    memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1e9
                    memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1e9
                    memory_free = (props.total_memory - torch.cuda.memory_reserved(gpu_id)) / 1e9

                    print(f"Memory Allocated: {memory_allocated:.1f} GB")
                    print(f"Memory Reserved: {memory_reserved:.1f} GB")
                    print(f"Memory Free: {memory_free:.1f} GB")
    else:
        print("Using CPU")
        print(f"PyTorch Version: {torch.__version__}")

    print("="*50)


def create_colab_setup_snippet() -> str:
    """
    Create a code snippet for Colab setup.

    Returns:
        str: Code snippet to run in Colab
    """
    snippet = '''
# Run this in Google Colab first cell
import torch
import sys

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU Memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")

# Install additional packages if needed
!pip install librosa soundfile wandb hydra-core omegaconf

# Mount Google Drive (optional, for saving results)
from google.colab import drive
drive.mount('/content/drive')

# Clone your repository
!git clone https://github.com/YOUR_USERNAME/DeepMIR-HW1.git
%cd DeepMIR-HW1

# Verify setup
from configs import load_experiment_config, setup_device
config = load_experiment_config("deep_learning", "baseline")
device = setup_device(config)
print(f"Ready to train on: {device}")
'''
    return snippet


def save_colab_setup():
    """Save Colab setup instructions to a markdown file."""
    setup_content = f"""# Google Colab Setup Instructions

## Quick Start

1. **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com)

2. **Enable GPU**:
   - Runtime → Change runtime type → Hardware accelerator → GPU → Save

3. **Setup Environment**: Run this in the first cell:

```python
{create_colab_setup_snippet()}
```

## Configuration for Colab

The project automatically detects Colab environment and adjusts:
- **Batch size**: Automatically reduced based on available GPU memory
- **Workers**: Set to 2 for optimal Colab performance
- **Device**: Auto-detects GPU availability

## Memory Management

For large datasets in Colab:
- Consider using smaller `max_duration` for audio (e.g., 20 seconds)
- Monitor GPU memory with `torch.cuda.memory_summary()`

## Saving Results

Save results to Google Drive:
```python
# In Colab, save to mounted drive
config.paths.checkpoints = "/content/drive/MyDrive/DeepMIR-HW1/checkpoints"
config.paths.predictions = "/content/drive/MyDrive/DeepMIR-HW1/predictions"
```

## Troubleshooting

- **Out of Memory**: Reduce batch_size in config
- **Slow Training**: Check if GPU is properly detected
- **Data Loading Issues**: Set num_workers=0 if multiprocessing fails
"""

    with open("COLAB_SETUP.md", "w", encoding="utf-8") as f:
        f.write(setup_content)

    logger.info("Created COLAB_SETUP.md with detailed instructions")


if __name__ == "__main__":
    # Demo device detection
    from configs import load_experiment_config

    config = load_experiment_config("deep_learning", "baseline")
    device = setup_device(config)
    print_device_info(device)

    # Create Colab setup file
    save_colab_setup()