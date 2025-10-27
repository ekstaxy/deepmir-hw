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
                           hop_length=256):
    """Extract a one-hot chromagram-based melody from an audio file."""
    audio, in_sr = torchaudio.load(str(audio_path))
    audio_mono = audio.mean(dim=0)

    if in_sr != sr:
        resample_tf = T.Resample(orig_freq=in_sr, new_freq=sr)
        audio_mono = resample_tf(audio_mono)

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


def calculate_melody_accuracy(target_path, generated_path):
    """Calculate melody accuracy between two audio files."""
    gt_melody = extract_melody_one_hot(target_path)      
    gen_melody = extract_melody_one_hot(generated_path)
    
    min_len_melody = min(gen_melody.shape[1], gt_melody.shape[1])
    matches = ((gen_melody[:, :min_len_melody] == gt_melody[:, :min_len_melody]) & 
               (gen_melody[:, :min_len_melody] == 1)).sum()
    accuracy = matches / min_len_melody
    
    return float(accuracy)


def normalize_filename(filename):
    """Normalize filename by removing duplicate extensions."""
    name = filename
    # Keep removing extensions until no more to remove
    while True:
        stem = Path(name).stem
        if stem == name:  # No extension removed
            break
        name = stem
    return name


def find_audio_file(folder, target_name):
    """Find audio file in folder matching the name (handles duplicate extensions)."""
    folder = Path(folder)
    
    # Normalize target name
    normalized_target = normalize_filename(target_name)
    
    # Get all audio files
    audio_files = []
    for ext in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
        audio_files.extend(folder.glob(f'*{ext}'))
    
    # Try to find match by normalizing each file's name
    for audio_file in audio_files:
        normalized_name = normalize_filename(audio_file.name)
        if normalized_name == normalized_target:
            return audio_file
    
    return None


def evaluate_folder(target_folder, generated_folder):
    """Evaluate melody accuracy for one generated folder."""
    target_path = Path(target_folder)
    generated_path = Path(generated_folder)
    
    results = {}
    
    # Get all audio files from target folder
    target_files = []
    for ext in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
        target_files.extend(target_path.glob(f'*{ext}'))
    
    print(f"\nProcessing: {generated_path.name}")
    
    for target_file in target_files:
        # Normalize the target filename
        normalized_name = normalize_filename(target_file.name)
        
        generated_file = find_audio_file(generated_path, normalized_name)
        
        if generated_file is None:
            print(f"  Missing: {normalized_name}")
            results[normalized_name] = None
            continue
        
        try:
            accuracy = calculate_melody_accuracy(target_file, generated_file)
            results[normalized_name] = accuracy
            print(f"  {normalized_name}: {accuracy:.4f}")
        except Exception as e:
            print(f"  Error {normalized_name}: {e}")
            results[normalized_name] = None
    
    # Save results
    output_path = generated_path / "melody_accuracy.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"  Saved: {output_path}")
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="HW2/data/target_music_list_60s", help="Target audio folder")
    parser.add_argument("--generated", default="HW2/results/musicControlLite_generated_music", help="Top directory with generated folders")
    args = parser.parse_args()
    
    target_path = Path(args.target)
    generated_top_path = Path(args.generated)
    
    # Find all subdirectories
    generated_folders = [f for f in generated_top_path.iterdir() if f.is_dir()]
    
    print(f"Target folder: {target_path}")
    print(f"Found {len(generated_folders)} folders")
    
    all_results = {}
    
    for folder in generated_folders:
        results = evaluate_folder(target_path, folder)
        all_results[folder.name] = results
    
    # Save overall results
    overall_path = generated_top_path / "all_melody_accuracy.json"
    with open(overall_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Overall results: {overall_path}")


if __name__ == "__main__":
    main()