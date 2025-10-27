import json
from pathlib import Path
import torchaudio
import numpy as np
import librosa
import scipy.signal as signal
from torchaudio import transforms as T

def extract_melody_one_hot(audio_path,
                           sr=44100,
                           cutoff=261.2, 
                           win_length=2048,
                           hop_length=256,
                           max_duration=None):
    """Extract a one-hot chromagram-based melody from an audio file."""
    audio, in_sr = torchaudio.load(str(audio_path))
    audio_mono = audio.mean(dim=0)

    if in_sr != sr:
        resample_tf = T.Resample(orig_freq=in_sr, new_freq=sr)
        audio_mono = resample_tf(audio_mono)

    # Trim to max_duration if specified
    if max_duration is not None:
        max_samples = int(max_duration * sr)
        audio_mono = audio_mono[:max_samples]

    y = audio_mono.numpy()

    nyquist = 0.5 * sr
    norm_cutoff = cutoff / nyquist
    b, a = signal.butter(N=2, Wn=norm_cutoff, btype='high', analog=False)
    y_hp = signal.filtfilt(b, a, y)

    chroma = librosa.feature.chroma_stft(
        y=y_hp,
        sr=sr,
        n_fft=win_length,
        win_length=win_length,
        hop_length=hop_length
    )

    pitch_class_idx = np.argmax(chroma, axis=0)
    one_hot_chroma = np.zeros_like(chroma)
    one_hot_chroma[pitch_class_idx, np.arange(chroma.shape[1])] = 1.0
    
    return one_hot_chroma


def calculate_melody_accuracy(target_path, generated_path, max_duration=None):
    """Calculate melody accuracy between two audio files."""
    gt_melody = extract_melody_one_hot(target_path, max_duration=max_duration)      
    gen_melody = extract_melody_one_hot(generated_path, max_duration=max_duration)
    
    min_len_melody = min(gen_melody.shape[1], gt_melody.shape[1])
    matches = ((gen_melody[:, :min_len_melody] == gt_melody[:, :min_len_melody]) & 
               (gen_melody[:, :min_len_melody] == 1)).sum()
    accuracy = matches / min_len_melody
    
    return float(accuracy)


def normalize_filename(filename):
    """Normalize filename by removing duplicate extensions."""
    name = filename
    while True:
        stem = Path(name).stem
        if stem == name:
            break
        name = stem
    return name


def find_audio_file(folder, target_name):
    """Find audio file in folder matching the name."""
    folder = Path(folder)
    normalized_target = normalize_filename(target_name)
    
    audio_files = []
    for ext in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
        audio_files.extend(folder.glob(f'*{ext}'))
    
    for audio_file in audio_files:
        normalized_name = normalize_filename(audio_file.name)
        if normalized_name == normalized_target:
            return audio_file
    
    return None


def evaluate_folder(target_folder, generated_folder, duration_sec):
    """Evaluate melody accuracy for one folder at specific duration."""
    target_path = Path(target_folder)
    generated_path = Path(generated_folder)
    
    results = {}
    
    target_files = []
    for ext in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
        target_files.extend(target_path.glob(f'*{ext}'))
    
    print(f"\nProcessing {duration_sec}s: {generated_path.name}")
    
    for target_file in target_files:
        normalized_name = normalize_filename(target_file.name)
        generated_file = find_audio_file(generated_path, normalized_name)
        
        if generated_file is None:
            print(f"  Missing: {normalized_name}")
            results[normalized_name] = None
            continue
        
        try:
            accuracy = calculate_melody_accuracy(target_file, generated_file, max_duration=duration_sec)
            results[normalized_name] = accuracy
            print(f"  {normalized_name}: {accuracy:.4f}")
        except Exception as e:
            print(f"  Error {normalized_name}: {e}")
            results[normalized_name] = None
    
    output_path = generated_path / f"melody_accuracy_{duration_sec}s.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"  Saved: {output_path}")
    return results


def main():
    target_folder = "HW2/data/target_music_list_60s"
    generated_folder = "HW2/results/musicgen-small_generated_music"  # Change this
    
    target_path = Path(target_folder)
    generated_path = Path(generated_folder)
    
    print(f"Target folder: {target_path}")
    print(f"Generated folder: {generated_path}")
    
    # Evaluate for 30s
    evaluate_folder(target_path, generated_path, duration_sec=30)
    
    # Evaluate for 60s
    evaluate_folder(target_path, generated_path, duration_sec=60)
    
    print(f"\n✓ Done")


if __name__ == "__main__":
    main()