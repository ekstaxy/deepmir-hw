"""
Data loader utilities for the Singer Classification project.
"""

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import logging

from .datasets import Artist20Dataset, Artist20TestDataset
from .transforms import create_traditional_ml_transforms, create_deep_learning_transforms
from .utils import get_artist_mapping

logger = logging.getLogger(__name__)


def create_dataloaders(
    config,
    experiment_type: str = "deep_learning",
    use_weighted_sampling: bool = True
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """
    Create train and validation data loaders.

    Args:
        config: Configuration object with dataset and training parameters
        experiment_type: Type of experiment ('traditional_ml' or 'deep_learning')
        use_weighted_sampling: Use weighted sampling for imbalanced classes

    Returns:
        Tuple of (train_loader, val_loader, artist_to_id)
    """
    # Get paths from config
    train_json = config.dataset.train_json
    val_json = config.dataset.val_json
    root_dir = config.dataset.root_path

    # Create artist mapping
    artist_to_id = get_artist_mapping(train_json)

    # Create transforms based on experiment type
    if experiment_type == "traditional_ml":
        train_transform = create_traditional_ml_transforms(
            sample_rate=config.audio.sample_rate,
            target_duration=config.audio.max_duration
        )
        val_transform = create_traditional_ml_transforms(
            sample_rate=config.audio.sample_rate,
            target_duration=config.audio.max_duration
        )
    elif experiment_type == "deep_learning":
        train_transform = create_deep_learning_transforms(
            sample_rate=config.audio.sample_rate,
            training=True
        )
        val_transform = create_deep_learning_transforms(
            sample_rate=config.audio.sample_rate,
            training=False
        )
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    # Create datasets
    train_dataset = Artist20Dataset(
        json_file_path=train_json,
        root_dir=root_dir,
        artist_to_id=artist_to_id,
        transform=None,
        sample_rate=config.audio.sample_rate,
        max_duration=config.audio.max_duration,
        validate_files=True
    )

    val_dataset = Artist20Dataset(
        json_file_path=val_json,
        root_dir=root_dir,
        artist_to_id=artist_to_id,
        transform=None,
        sample_rate=config.audio.sample_rate,
        max_duration=config.audio.max_duration,
        validate_files=True
    )


    # Create samplers
    train_sampler = None
    if use_weighted_sampling and hasattr(train_dataset, 'get_class_weights'):
        class_weights = train_dataset.get_class_weights()

        # Create sample weights
        sample_weights = class_weights[train_dataset.labels]

        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        logger.info("Using weighted sampling for training")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        # shuffle=True,
        sampler=train_sampler,
        num_workers=config.device.num_workers,
        pin_memory=config.device.pin_memory,
        drop_last=True,
        collate_fn=_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.device.num_workers,
        pin_memory=config.device.pin_memory,
        drop_last=False,
        collate_fn=_collate_fn
    )

    logger.info(f"Created data loaders:")
    logger.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    logger.info(f"  Classes: {len(artist_to_id)}")

    return train_loader, val_loader, artist_to_id


def create_test_dataloader(
    config,
    experiment_type: str = "deep_learning"
) -> Tuple[DataLoader, Artist20TestDataset]:
    """
    Create test data loader.

    Args:
        config: Configuration object
        experiment_type: Type of experiment ('traditional_ml' or 'deep_learning')

    Returns:
        Tuple of (test_loader, test_dataset)
    """
    # Create transform based on experiment type
    if experiment_type == "traditional_ml":
        transform = create_traditional_ml_transforms(
            sample_rate=config.audio.sample_rate,
            target_duration=config.audio.max_duration
        )
    elif experiment_type == "deep_learning":
        transform = create_deep_learning_transforms(
            sample_rate=config.audio.sample_rate,
            training=False  # No augmentation for test
        )
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    # Create test dataset
    test_dataset = Artist20TestDataset(
        test_dir=config.dataset.test_dir,
        transform=None,
        sample_rate=config.audio.sample_rate,
        max_duration=config.audio.max_duration
    )

    # Create test data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.device.num_workers,
        pin_memory=config.device.pin_memory,
        drop_last=False,
        collate_fn=_collate_fn_test
    )

    logger.info(f"Created test loader: {len(test_dataset)} samples, {len(test_loader)} batches")

    return test_loader, test_dataset


def _collate_fn(batch):
    """
    Custom collate function for training/validation batches.

    Args:
        batch: List of (audio, label) tuples

    Returns:
        Tuple of (batched_audio, batched_labels)
    """
    audio_batch = []
    label_batch = []

    for audio, label in batch:
        audio_batch.append(audio)
        label_batch.append(label)

    # Stack audio tensors
    audio_tensor = torch.stack(audio_batch)
    label_tensor = torch.tensor(label_batch, dtype=torch.long)

    return audio_tensor, label_tensor


def _collate_fn_test(batch):
    """
    Custom collate function for test batches.

    Args:
        batch: List of (audio, test_id) tuples

    Returns:
        Tuple of (batched_audio, test_ids)
    """
    audio_batch = []
    id_batch = []

    for audio, test_id in batch:
        audio_batch.append(audio)
        id_batch.append(test_id)

    # Stack audio tensors
    audio_tensor = torch.stack(audio_batch)

    return audio_tensor, id_batch


def create_feature_dataloaders(
    features_train: torch.Tensor,
    labels_train: torch.Tensor,
    features_val: torch.Tensor,
    labels_val: torch.Tensor,
    batch_size: int = 32,
    use_weighted_sampling: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders from pre-extracted features (for traditional ML).

    Args:
        features_train: Training features tensor
        labels_train: Training labels tensor
        features_val: Validation features tensor
        labels_val: Validation labels tensor
        batch_size: Batch size
        use_weighted_sampling: Use weighted sampling for imbalanced classes
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        Tuple of (train_loader, val_loader)
    """
    from torch.utils.data import TensorDataset

    # Create tensor datasets
    train_dataset = TensorDataset(features_train, labels_train)
    val_dataset = TensorDataset(features_val, labels_val)

    # Create weighted sampler if requested
    train_sampler = None
    if use_weighted_sampling:
        # Compute class weights
        num_classes = len(torch.unique(labels_train))
        class_counts = torch.bincount(labels_train, minlength=num_classes)
        class_weights = len(labels_train) / (num_classes * class_counts.float())

        # Create sample weights
        sample_weights = class_weights[labels_train]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return train_loader, val_loader


def create_inference_dataloader(
    dataset: Union[Artist20Dataset, Artist20TestDataset],
    batch_size: int = 32,
    num_workers: int = 4
) -> DataLoader:
    """
    Create data loader for inference (no shuffling, no augmentation).

    Args:
        dataset: Dataset to create loader for
        batch_size: Batch size
        num_workers: Number of worker processes

    Returns:
        DataLoader for inference
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=_collate_fn if isinstance(dataset, Artist20Dataset) else _collate_fn_test
    )