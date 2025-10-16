"""
Utility functions for data loading and preprocessing.
"""

import os
import json
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)


def load_audio_file(
    file_path: Union[str, Path],
    sample_rate: int = 16000,
    mono: bool = True,
    duration: Optional[float] = None,
    offset: float = 0.0
) -> Tuple[np.ndarray, int]:
    """
    Load audio file using librosa.

    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate
        mono: Convert to mono if True
        duration: Maximum duration to load (None for full file)
        offset: Start offset in seconds

    Returns:
        Tuple of (audio_data, sample_rate)

    Raises:
        FileNotFoundError: If audio file doesn't exist
        Exception: If audio loading fails
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        # Load audio with librosa
        audio, sr = librosa.load(
            file_path,
            sr=sample_rate,
            mono=mono,
            duration=duration,
            offset=offset
        )

        logger.debug(f"Loaded audio: {file_path}, shape: {audio.shape}, sr: {sr}")
        return audio, sr

    except Exception as e:
        logger.error(f"Failed to load audio file {file_path}: {e}")
        raise


def validate_audio_file(file_path: Union[str, Path]) -> bool:
    """
    Validate if audio file exists and can be loaded.

    Args:
        file_path: Path to audio file

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            return False

        # Try to get basic info without loading full file
        info = sf.info(file_path)
        return info.frames > 0 and info.samplerate > 0

    except Exception as e:
        logger.warning(f"Audio validation failed for {file_path}: {e}")
        return False


def get_artist_mapping(train_json_path: Union[str, Path]) -> Dict[str, int]:
    """
    Extract artist names from training data and create label mapping.

    Args:
        train_json_path: Path to train.json file

    Returns:
        Dictionary mapping artist names to integer labels
    """
    with open(train_json_path, 'r') as f:
        train_files = json.load(f)

    # Extract artist names from file paths
    artists = set()
    for file_path in train_files:
        try:
            artist_name = extract_artist_from_path(file_path)
            artists.add(artist_name)
        except Exception as e:
            logger.warning(f"Failed to extract artist from {file_path}: {e}")

    # Sort for consistent mapping
    sorted_artists = sorted(list(artists))

    # Create mapping
    artist_to_id = {artist: idx for idx, artist in enumerate(sorted_artists)}

    logger.info(f"Found {len(artist_to_id)} artists: {list(artist_to_id.keys())}")
    return artist_to_id


def create_label_encoder(artist_to_id: Dict[str, int]) -> LabelEncoder:
    """
    Create sklearn LabelEncoder from artist mapping.

    Args:
        artist_to_id: Dictionary mapping artist names to IDs

    Returns:
        Fitted LabelEncoder
    """
    label_encoder = LabelEncoder()
    sorted_artists = [artist for artist, _ in sorted(artist_to_id.items(), key=lambda x: x[1])]
    label_encoder.fit(sorted_artists)

    return label_encoder


def extract_artist_from_path(file_path: Union[str, Path]) -> str:
    """
    Extract artist name from file path.

    Args:
        file_path: Path to audio file

    Returns:
        Artist name

    Raises:
        ValueError: If artist cannot be extracted from path
    """
    # Handle both string and Path inputs
    file_path_str = str(file_path)

    # Expected format: "./train_val/artist_name/album/song.mp3"
    if file_path_str.startswith('./'):
        file_path_str = file_path_str[2:]  # Remove './'

    parts = file_path_str.split('/')

    if len(parts) < 3:
        raise ValueError(f"Invalid path format: {file_path}")

    # Expected format: "train_val/artist_name/album/song.mp3"
    if parts[0] == "train_val":
        return parts[1]  # artist_name
    else:
        raise ValueError(f"Unexpected path format: {file_path}")


def pad_or_truncate_audio(
    audio: np.ndarray,
    target_length: int,
    mode: str = "constant"
) -> np.ndarray:
    """
    Pad or truncate audio to target length.

    Args:
        audio: Input audio array
        target_length: Target length in samples
        mode: Padding mode for numpy.pad

    Returns:
        Processed audio array
    """
    current_length = len(audio)

    if current_length == target_length:
        return audio
    elif current_length > target_length:
        # Truncate from center
        start = (current_length - target_length) // 2
        return audio[start:start + target_length]
    else:
        # Pad to target length
        pad_width = target_length - current_length
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        return np.pad(audio, (pad_left, pad_right), mode=mode)


def compute_audio_statistics(
    file_paths: List[str],
    sample_rate: int = 16000,
    max_files: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute basic statistics over a set of audio files.

    Args:
        file_paths: List of audio file paths
        sample_rate: Sample rate for loading
        max_files: Maximum number of files to analyze (None for all)

    Returns:
        Dictionary with audio statistics
    """
    durations = []
    amplitudes = []

    files_to_process = file_paths[:max_files] if max_files else file_paths

    for file_path in files_to_process:
        try:
            audio, sr = load_audio_file(file_path, sample_rate=sample_rate)
            duration = len(audio) / sr
            durations.append(duration)
            amplitudes.extend(audio.tolist())

        except Exception as e:
            logger.warning(f"Failed to process {file_path}: {e}")
            continue

    amplitudes = np.array(amplitudes)

    statistics = {
        "num_files": len(files_to_process),
        "mean_duration": np.mean(durations),
        "std_duration": np.std(durations),
        "min_duration": np.min(durations),
        "max_duration": np.max(durations),
        "mean_amplitude": np.mean(amplitudes),
        "std_amplitude": np.std(amplitudes),
        "rms_amplitude": np.sqrt(np.mean(amplitudes**2))
    }

    return statistics


def create_train_val_split_from_json(
    train_json_path: Union[str, Path],
    val_json_path: Union[str, Path]
) -> Tuple[List[str], List[str], List[int], List[int]]:
    """
    Load train/val splits from JSON files and extract labels.

    Args:
        train_json_path: Path to train.json
        val_json_path: Path to val.json

    Returns:
        Tuple of (train_files, val_files, train_labels, val_labels)
    """
    # Load file lists
    with open(train_json_path, 'r') as f:
        train_files = json.load(f)

    with open(val_json_path, 'r') as f:
        val_files = json.load(f)

    # Get artist mapping
    artist_to_id = get_artist_mapping(train_json_path)

    # Extract labels
    train_labels = []
    for file_path in train_files:
        artist = extract_artist_from_path(file_path)
        train_labels.append(artist_to_id[artist])

    val_labels = []
    for file_path in val_files:
        artist = extract_artist_from_path(file_path)
        val_labels.append(artist_to_id[artist])

    logger.info(f"Loaded {len(train_files)} training files, {len(val_files)} validation files")
    logger.info(f"Training label distribution: {np.bincount(train_labels)}")
    logger.info(f"Validation label distribution: {np.bincount(val_labels)}")

    return train_files, val_files, train_labels, val_labels


def get_test_files(test_dir: Union[str, Path]) -> List[str]:
    """
    Get list of test files in order.

    Args:
        test_dir: Directory containing test files

    Returns:
        Sorted list of test file paths
    """
    test_dir = Path(test_dir)
    test_files = []

    for i in range(1, 234):  # 001_vocals.mp3 to 233_vocals.mp3
        file_path = test_dir / f"{i:03d}_vocals.mp3"
        if file_path.exists():
            test_files.append(str(file_path))
        else:
            logger.warning(f"Test file not found: {file_path}")

    logger.info(f"Found {len(test_files)} test files")
    return test_files


def save_processed_features(
    features: np.ndarray,
    labels: np.ndarray,
    save_path: Union[str, Path],
    metadata: Optional[Dict] = None
) -> None:
    """
    Save processed features to disk.

    Args:
        features: Feature array
        labels: Label array
        save_path: Path to save file
        metadata: Optional metadata dictionary
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        'features': features,
        'labels': labels,
        'metadata': metadata or {}
    }

    np.savez_compressed(save_path, **save_data)
    logger.info(f"Saved features to {save_path}")


def load_processed_features(load_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load processed features from disk.

    Args:
        load_path: Path to load file

    Returns:
        Tuple of (features, labels, metadata)
    """
    data = np.load(load_path, allow_pickle=True)

    features = data['features']
    labels = data['labels']
    metadata = data.get('metadata', {}).item() if 'metadata' in data else {}

    logger.info(f"Loaded features from {load_path}")
    return features, labels, metadata
