"""
Audio preprocessing transforms for the singer classification project.
"""

import numpy as np
import librosa
import torch
import torchaudio
from typing import Optional, Union, Callable, List
import logging

logger = logging.getLogger(__name__)


class ComposeTransforms:
    """Compose multiple transforms together."""

    def __init__(self, transforms: List[Callable]):
        """
        Initialize composed transforms.

        Args:
            transforms: List of transform functions/objects
        """
        self.transforms = transforms

    def __call__(self, audio: np.ndarray) -> Union[np.ndarray, torch.Tensor]:
        """Apply all transforms in sequence."""
        for transform in self.transforms:
            audio = transform(audio)
        return audio

    def __repr__(self):
        transform_names = [t.__class__.__name__ for t in self.transforms]
        return f"ComposeTransforms({transform_names})"


class AudioResample:
    """Resample audio to target sample rate."""

    def __init__(self, target_sr: int = 16000, orig_sr: Optional[int] = None):
        """
        Initialize resampling transform.

        Args:
            target_sr: Target sample rate
            orig_sr: Original sample rate (if known)
        """
        self.target_sr = target_sr
        self.orig_sr = orig_sr

    def __call__(self, audio: np.ndarray, orig_sr: Optional[int] = None) -> np.ndarray:
        """
        Resample audio.

        Args:
            audio: Input audio array
            orig_sr: Original sample rate (overrides init value)

        Returns:
            Resampled audio
        """
        sr = orig_sr or self.orig_sr
        if sr is None:
            raise ValueError("Original sample rate must be provided")

        if sr == self.target_sr:
            return audio

        resampled = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
        return resampled


class AudioToMono:
    """Convert audio to mono."""

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert audio to mono.

        Args:
            audio: Input audio (1D or 2D array)

        Returns:
            Mono audio (1D array)
        """
        if audio.ndim == 1:
            return audio
        elif audio.ndim == 2:
            # Average across channels
            return np.mean(audio, axis=0)
        else:
            raise ValueError(f"Unsupported audio shape: {audio.shape}")


class AudioNormalize:
    """Normalize audio amplitude."""

    def __init__(self, method: str = "peak", target_db: float = -20.0):
        """
        Initialize normalization.

        Args:
            method: Normalization method ('peak', 'rms', 'lufs')
            target_db: Target level in dB (for RMS/LUFS)
        """
        self.method = method
        self.target_db = target_db

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio.

        Args:
            audio: Input audio array

        Returns:
            Normalized audio
        """
        if len(audio) == 0:
            return audio

        if self.method == "peak":
            # Peak normalization to [-1, 1]
            peak = np.max(np.abs(audio))
            if peak > 0:
                return audio / peak
            return audio

        elif self.method == "rms":
            # RMS normalization
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                target_rms = 10**(self.target_db / 20.0)
                return audio * (target_rms / rms)
            return audio

        else:
            raise ValueError(f"Unknown normalization method: {self.method}")


class AudioPad:
    """Pad or truncate audio to fixed length."""

    def __init__(
        self,
        target_length: int,
        mode: str = "constant",
        truncate_mode: str = "center"
    ):
        """
        Initialize padding transform.

        Args:
            target_length: Target length in samples
            mode: Padding mode for numpy.pad
            truncate_mode: How to truncate ('center', 'random', 'start', 'end')
        """
        self.target_length = target_length
        self.mode = mode
        self.truncate_mode = truncate_mode

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Pad or truncate audio.

        Args:
            audio: Input audio array

        Returns:
            Padded/truncated audio
        """
        current_length = len(audio)

        if current_length == self.target_length:
            return audio

        elif current_length > self.target_length:
            # Truncate
            if self.truncate_mode == "center":
                start = (current_length - self.target_length) // 2
                return audio[start:start + self.target_length]

            elif self.truncate_mode == "random":
                start = np.random.randint(0, current_length - self.target_length + 1)
                return audio[start:start + self.target_length]

            elif self.truncate_mode == "start":
                return audio[:self.target_length]

            elif self.truncate_mode == "end":
                return audio[-self.target_length:]

            else:
                raise ValueError(f"Unknown truncate mode: {self.truncate_mode}")

        else:
            # Pad
            pad_width = self.target_length - current_length
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            return np.pad(audio, (pad_left, pad_right), mode=self.mode)
class AudioToMelSpectrogram:
    """Convert audio to mel-spectrogram."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        power: float = 2.0,
        to_db: bool = True,
        ref: float = 1.0,
        amin: float = 1e-10,
        top_db: Optional[float] = 80.0
    ):
        """
        Initialize mel-spectrogram transform.

        Args:
            sample_rate: Sample rate of audio
            n_fft: Length of FFT window
            hop_length: Number of samples between frames
            n_mels: Number of mel frequency bins
            f_min: Minimum frequency
            f_max: Maximum frequency (None for sr/2)
            power: Exponent for magnitude spectrogram
            to_db: Convert to decibel scale
            ref: Reference value for dB conversion
            amin: Minimum value for dB conversion
            top_db: Maximum dB value (for clipping)
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate // 2
        self.power = power
        self.to_db = to_db
        self.ref = ref
        self.amin = amin
        self.top_db = top_db

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert audio to mel-spectrogram.

        Args:
            audio: Input audio array

        Returns:
            Mel-spectrogram array of shape (n_mels, time_frames)
        """
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
            power=self.power
        )

        if self.to_db:
            # Convert to dB scale
            mel_spec_db = librosa.power_to_db(
                mel_spec,
                ref=self.ref,
                amin=self.amin,
                top_db=self.top_db
            )
            return mel_spec_db
        else:
            return mel_spec

class AudioToTensor:
    """Convert audio numpy array to PyTorch tensor."""

    def __init__(self, dtype: torch.dtype = torch.float32):
        """
        Initialize tensor conversion.

        Args:
            dtype: Target tensor dtype
        """
        self.dtype = dtype

    def __call__(self, audio: np.ndarray) -> torch.Tensor:
        """
        Convert to tensor.

        Args:
            audio: Input audio array

        Returns:
            Audio tensor
        """
        return torch.from_numpy(audio).to(self.dtype)

class TimeShift:
    """Randomly shift audio in time (data augmentation)."""

    def __init__(self, max_shift: float = 0.1, probability: float = 0.5):
        """
        Initialize time shift augmentation.

        Args:
            max_shift: Maximum shift as fraction of total length
            probability: Probability of applying augmentation
        """
        self.max_shift = max_shift
        self.probability = probability

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply time shift.

        Args:
            audio: Input audio array

        Returns:
            Time-shifted audio
        """
        if np.random.random() > self.probability:
            return audio

        max_shift_samples = int(len(audio) * self.max_shift)
        shift_samples = np.random.randint(-max_shift_samples, max_shift_samples + 1)

        if shift_samples == 0:
            return audio

        if shift_samples > 0:
            # Shift right (delay)
            shifted = np.concatenate([np.zeros(shift_samples), audio[:-shift_samples]])
        else:
            # Shift left (advance)
            shifted = np.concatenate([audio[-shift_samples:], np.zeros(-shift_samples)])

        return shifted


class AddNoise:
    """Add white noise to audio (data augmentation)."""

    def __init__(self, noise_factor: float = 0.01, probability: float = 0.5):
        """
        Initialize noise augmentation.

        Args:
            noise_factor: Noise level as fraction of signal amplitude
            probability: Probability of applying augmentation
        """
        self.noise_factor = noise_factor
        self.probability = probability

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Add noise to audio.

        Args:
            audio: Input audio array

        Returns:
            Noisy audio
        """
        if np.random.random() > self.probability:
            return audio

        noise = np.random.normal(0, 1, audio.shape)
        noise_level = self.noise_factor * np.std(audio)
        return audio + noise_level * noise


class TimeStretch:
    """Time-stretch audio without changing pitch using librosa."""

    def __init__(self, min_rate: float = 0.95, max_rate: float = 1.10, probability: float = 0.3):
        """
        Initialize time-stretch augmentation.

        Args:
            min_rate: Minimum stretch factor (e.g., 0.8 -> slower)
            max_rate: Maximum stretch factor (e.g., 1.25 -> faster)
            probability: Probability of applying augmentation
        """
        if min_rate <= 0 or max_rate <= 0:
            raise ValueError("min_rate and max_rate must be > 0")
        if min_rate > max_rate:
            raise ValueError("min_rate must be <= max_rate")
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.probability = probability

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply time-stretch to audio.

        Args:
            audio: 1D numpy array of audio samples

        Returns:
            Time-stretched audio (1D numpy array)
        """
        if np.random.random() > self.probability:
            return audio

        # Pick a random rate in [min_rate, max_rate]
        rate = np.random.uniform(self.min_rate, self.max_rate)

        # librosa.effects.time_stretch expects a mono signal
        if audio.ndim != 1:
            # If stereo, average to mono first
            audio_mono = np.mean(audio, axis=0)
        else:
            audio_mono = audio

        try:
            stretched = librosa.effects.time_stretch(y=audio_mono.astype(float), rate=rate)
        except Exception as e:
            logger.warning(f"TimeStretch failed: {e}")
            return audio

        return stretched


def create_traditional_ml_transforms(
    sample_rate: int = 16000,
    target_duration: Optional[float] = None
) -> ComposeTransforms:
    """
    Create transform pipeline for traditional ML (feature extraction).

    Args:
        sample_rate: Target sample rate
        target_duration: Target duration in seconds (None for no padding)

    Returns:
        Composed transforms
    """
    transforms = [
        AudioToMono(),
        AudioNormalize(method="peak")
    ]

    if target_duration is not None:
        target_length = int(sample_rate * target_duration)
        transforms.append(AudioPad(target_length))

    return ComposeTransforms(transforms)


def create_deep_learning_transforms(
    sample_rate: int = 16000,
    training: bool = True
) -> ComposeTransforms:
    """
    Create transform pipeline for deep learning.

    Args:
        sample_rate: Target sample rate 
        training: Whether to include data augmentation

    Returns:
        Composed transforms
    """
    transforms = [
        AudioToMono(),
        AudioNormalize(method="peak"),
    ]

    # Add data augmentation for training
    if training:
        transforms.extend([
            TimeStretch(min_rate=0.95, max_rate=1.10, probability=0.3),
            AddNoise(noise_factor=0.005, probability=0.2)
        ])

    return ComposeTransforms(transforms)