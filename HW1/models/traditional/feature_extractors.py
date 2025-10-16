"""
Audio feature extraction for traditional machine learning models.
Template for implementing hand-crafted audio features.
"""

import numpy as np
import librosa
import torch
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MFCCExtractor:
    """Extract MFCC features from audio."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
        include_delta: bool = True,
        include_delta2: bool = True
    ):
        """
        Initialize MFCC extractor.

        Args:
            sample_rate: Audio sample rate
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length for STFT
            include_delta: Include delta (velocity) features
            include_delta2: Include delta-delta (acceleration) features
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.include_delta = include_delta
        self.include_delta2 = include_delta2
        

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from audio.

        Args:
            audio: Audio signal

        Returns:
            MFCC features of shape (n_features, n_frames)
        """
        mfcc_features = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        features = [mfcc_features]

        if self.include_delta:
            delta_features = librosa.feature.delta(mfcc_features)
            features.append(delta_features)

        if self.include_delta2:
            delta2_features = librosa.feature.delta(mfcc_features, order=2)
            features.append(delta2_features)

        return np.concatenate(features, axis=0)

class SpectralFeaturesExtractor:
    """Extract spectral features from audio."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 512
    ):
        """
        Initialize spectral features extractor.

        Args:
            sample_rate: Audio sample rate
            n_fft: FFT window size
            hop_length: Hop length for STFT
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract spectral features from audio.

        Args:
            audio: Audio signal

        Returns:
            Dictionary of spectral features
        """
        features = {}

        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(
            y=audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )

        return features

class ChromaExtractor:
    """Extract chroma features from audio."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_chroma: int = 12,
        n_fft: int = 2048,
        hop_length: int = 512
    ):
        """
        Initialize chroma extractor.

        Args:
            sample_rate: Audio sample rate
            n_chroma: Number of chroma bins
            n_fft: FFT window size
            hop_length: Hop length for STFT
        """
        self.sample_rate = sample_rate
        self.n_chroma = n_chroma
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract chroma features from audio.

        Args:
            audio: Audio signal

        Returns:
            Chroma features of shape (n_chroma, n_frames)
        """
        return librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            n_chroma=self.n_chroma,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

class TonnetzExtractor:
    """Extract tonnetz (tonal centroid) features from audio."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 512
    ):
        """
        Initialize tonnetz extractor.

        Args:
            sample_rate: Audio sample rate
            n_fft: FFT window size
            hop_length: Hop length for STFT
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract tonnetz features from audio.

        Args:
            audio: Audio signal

        Returns:
            Tonnetz features of shape (6, n_frames)
        """
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return librosa.feature.tonnetz(
            y=audio,
            sr=self.sample_rate,
            chroma=chroma
        )

class AudioFeatureExtractor:
    """Unified audio feature extractor for traditional ML."""

    def __init__(self, config):
        """
        Initialize feature extractor with configuration.

        Args:
            config: Configuration object with feature extraction parameters
        """
        self.config = config
        self.sample_rate = config.audio.sample_rate

        self.mfcc_extractor = MFCCExtractor(
            sample_rate=config.audio.sample_rate,
            n_mfcc=config.features.mfcc.n_mfcc,
            n_fft=config.features.mfcc.n_fft,
            hop_length=config.features.mfcc.hop_length,
            include_delta=config.features.mfcc.delta,
            include_delta2=config.features.mfcc.delta2
        )
        self.spectral_extractor = SpectralFeaturesExtractor(
            sample_rate=config.audio.sample_rate,
            n_fft=config.features.spectral.n_fft,
            hop_length=config.features.spectral.hop_length
        )
        self.chroma_extractor = ChromaExtractor(
            sample_rate=config.audio.sample_rate,
            n_chroma=config.features.chroma.n_chroma,
            n_fft=config.features.chroma.n_fft,
            hop_length=config.features.chroma.hop_length
        )
        self.tonnetz_extractor = TonnetzExtractor(
            sample_rate=config.audio.sample_rate,
            n_fft=config.features.tonnetz.n_fft,
            hop_length=config.features.tonnetz.hop_length
        )

    def _remove_silence(self, audio: np.ndarray) -> np.ndarray:
        """Remove silent segments from audio."""
        non_silent_intervals = librosa.effects.split(audio, top_db=self.config.preprocessing.silence_db_threshold)
        non_silent_audio = np.concatenate([audio[start:end] for start, end in non_silent_intervals])
        return non_silent_audio

    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract all configured features from audio after removing silence.

        Args:
            audio: Audio signal

        Returns:
            Concatenated feature vector
        """
        # Remove silence
        audio = self._remove_silence(audio)

        features = []

        if 'mfcc' in self.config.features.extract_features:
            mfcc_features = self.mfcc_extractor.extract(audio)
            features.append(mfcc_features)

        if 'spectral' in self.config.features.extract_features:
            spectral_features_dict = self.spectral_extractor.extract(audio)
            spectral_features = np.concatenate(list(spectral_features_dict.values()), axis=0)
            features.append(spectral_features)

        if 'chroma' in self.config.features.extract_features:
            chroma_features = self.chroma_extractor.extract(audio)
            features.append(chroma_features)

        if 'tonnetz' in self.config.features.extract_features:
            tonnetz_features = self.tonnetz_extractor.extract(audio)
            features.append(tonnetz_features)

        final_features = np.concatenate(features, axis=0)
        aggregated_features = self._aggregate_features(final_features)

        return aggregated_features

    def _aggregate_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize time-series features and aggregate them to fixed-size vectors.

        Args:
            features: Feature matrix of shape (n_features, n_frames)

        Returns:
            Aggregated feature vector
        """
        # Normalize features before aggregation
        mean = np.mean(features, axis=1, keepdims=True)
        std = np.std(features, axis=1, keepdims=True)
        normalized_features = (features - mean) / (std + 1e-8)  # Avoid division by zero

        aggregated = []
        methods = self.config.preprocessing.aggregation.methods

        for method in methods:
            if method == "mean":
                aggregated.append(np.mean(normalized_features, axis=1))
            elif method == "std":
                aggregated.append(np.std(normalized_features, axis=1))
            elif method == "min":
                aggregated.append(np.min(normalized_features, axis=1))
            elif method == "max":
                aggregated.append(np.max(normalized_features, axis=1))
            elif method == "median":
                aggregated.append(np.median(normalized_features, axis=1))
            else:
                logger.warning(f"Unknown aggregation method: {method}")

        return np.concatenate(aggregated)

    def extract_features_batch(self, audio_list: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from multiple audio samples.

        Args:
            audio_list: List of audio signals

        Returns:
            Feature matrix of shape (n_samples, n_features)
        """
        feature_list = []
        for audio in audio_list:
            features = self.extract_features(audio)
            feature_list.append(features)

        return np.stack(feature_list, axis=0)

    def save_features(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        save_path: Union[str, Path],
        metadata: Optional[Dict] = None
    ):
        """
        Save extracted features to disk.

        Args:
            features: Feature matrix
            labels: Corresponding labels
            save_path: Path to save features
            metadata: Optional metadata
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare metadata
        if metadata is None:
            metadata = {}

        metadata.update({
            'timestamp': str(Path(__file__).stat().st_mtime),
            'feature_shape': features.shape,
            'label_shape': labels.shape,
            'extraction_config': {
                'sample_rate': self.sample_rate,
                'aggregation_methods': self.config.preprocessing.aggregation.methods,
                'extracted_features': self.config.features.extract_features
            }
        })

        # Save features, labels, and metadata in compressed format
        np.savez_compressed(
            save_path,
            features=features,
            labels=labels,
            metadata=metadata
        )

        logger.info(f"Features saved to {save_path} (shape: {features.shape})")

    def load_features(self, load_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load features from disk.

        Args:
            load_path: Path to load features from

        Returns:
            Tuple of (features, labels, metadata)
        """
        load_path = Path(load_path)

        if not load_path.exists():
            raise FileNotFoundError(f"Feature file not found: {load_path}")

        # Load compressed numpy archive
        data = np.load(load_path, allow_pickle=True)

        try:
            features = data['features']
            labels = data['labels']
            metadata = data['metadata'].item()  # Convert 0-d array back to dict

            logger.info(f"Features loaded from {load_path} (shape: {features.shape})")

            return features, labels, metadata

        except KeyError as e:
            raise ValueError(f"Invalid feature file format. Missing key: {e}")
        finally:
            data.close()

def extract_features_from_dataset(dataset, config, save_path: Optional[str] = None):
    """
    Extract features from entire dataset.

    Args:
        dataset: Dataset object
        config: Configuration object
        save_path: Optional path to save features

    Returns:
        Tuple of (features, labels)
    """
    logger.info("Starting feature extraction from dataset...")

    # Initialize feature extractor
    feature_extractor = AudioFeatureExtractor(config)

    # Process dataset in 500-sample batches to prevent RAM overflow
    all_features = []
    all_labels = []
    batch_size = 100

    logger.info(f"Processing {len(dataset)} samples in batches of {batch_size}...")

    for batch_start in range(0, len(dataset), batch_size):
        batch_end = min(batch_start + batch_size, len(dataset))

        # Load current batch
        audio_list = []
        labels = []

        for i in range(batch_start, batch_end):
            excerpts, label = dataset[i]  # Stack of excerpts
            for excerpt in excerpts:
                # Convert tensor to numpy array if needed
                if isinstance(excerpt, torch.Tensor):
                    excerpt = excerpt.numpy()
                audio_list.append(excerpt)
                # Convert integer label back to artist name string
                artist_name = dataset.get_artist_name(label)
                labels.append(artist_name)

        logger.info(f"Loaded batch {batch_start//batch_size + 1}/{(len(dataset)-1)//batch_size + 1} ({batch_end - batch_start} samples)")

        # Convert labels and extract features for this batch
        labels = np.array(labels)
        batch_features = feature_extractor.extract_features_batch(audio_list)

        # Store results
        all_features.append(batch_features)
        all_labels.append(labels)

        # Clear batch memory
        del audio_list, labels, batch_features

    # Combine all batches
    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)

    logger.info(f"Feature extraction complete. Shape: {features.shape}")

    # Save features if path provided
    if save_path:
        metadata = {
            'dataset_size': len(dataset),
            'extraction_timestamp': str(np.datetime64('now'))
        }
        feature_extractor.save_features(features, labels, save_path, metadata)

    return features, labels


# Example usage template
if __name__ == "__main__":
    pass