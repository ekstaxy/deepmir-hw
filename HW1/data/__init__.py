"""
Data module for singer classification project.
Provides dataset classes, data loaders, and preprocessing utilities.
"""

from .datasets import Artist20Dataset, Artist20TestDataset
from .transforms import (
    AudioToMelSpectrogram,
    AudioNormalize,
    AudioResample,
    AudioPad,
    AudioToMono,
    ComposeTransforms
)
from .dataloaders import create_dataloaders, create_test_dataloader
from .utils import (
    get_artist_mapping,
    load_audio_file,
    validate_audio_file,
    create_label_encoder
)

__all__ = [
    # Datasets
    'Artist20Dataset',
    'Artist20TestDataset',

    # Transforms
    'AudioToMelSpectrogram',
    'AudioNormalize',
    'AudioResample',
    'AudioPad',
    'AudioToMono',
    'ComposeTransforms',

    # Data loaders
    'create_dataloaders',
    'create_test_dataloader',

    # Utilities
    'get_artist_mapping',
    'load_audio_file',
    'validate_audio_file',
    'create_label_encoder'
]