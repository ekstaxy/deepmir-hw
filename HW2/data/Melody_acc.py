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


def calculate_melody_accuracy(target_path, reference_path):
    """Calculate melody accuracy between two audio files."""
    gt_melody = extract_melody_one_hot(target_path)      
    gen_melody = extract_melody_one_hot(reference_path)
    
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


def evaluate_retrieval_melody(retrieval_json, target_folder, reference_folder):
    """Evaluate melody accuracy for retrieved pairs."""
    
    with open(retrieval_json, 'r', encoding='utf-8') as f:
        retrieval_data = json.load(f)
    
    target_path = Path(target_folder)
    reference_path = Path(reference_folder)
    
    results = {}
    all_accuracies = []
    
    print("Calculating melody accuracy for retrieved pairs...")
    
    for target_name, retrieval_list in retrieval_data.items():
        print(f"\n  Target: {target_name}")
        
        # Find target audio file
        target_file = find_audio_file(target_path, target_name)
        if target_file is None:
            print(f"    ✗ Target not found")
            results[target_name] = None
            continue
        
        # Get the top reference (highest similarity)
        if not retrieval_list:
            print(f"    ✗ No retrievals")
            results[target_name] = None
            continue
        
        top_reference = retrieval_list[0]['reference']
        reference_file = find_audio_file(reference_path, top_reference)
        
        if reference_file is None:
            print(f"    ✗ Reference not found: {top_reference}")
            results[target_name] = None
            continue
        
        try:
            accuracy = calculate_melody_accuracy(target_file, reference_file)
            results[target_name] = accuracy
            all_accuracies.append(accuracy)
            print(f"    ✓ Top reference: {top_reference}")
            print(f"    Melody accuracy: {accuracy:.4f}")
        except Exception as e:
            print(f"    ✗ Error: {e}")
            results[target_name] = None
    
    # Calculate mean accuracy
    mean_accuracy = float(np.mean(all_accuracies)) if all_accuracies else None
    
    output_data = {
        "mean_melody_accuracy": mean_accuracy,
        "results": results
    }
    
    # Save results
    output_path = Path(retrieval_json).parent / "retrieval_melody_accuracy.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Mean melody accuracy: {mean_accuracy:.4f}")
    print(f"✓ Saved: {output_path}")
    
    return mean_accuracy


def main():
    retrieval_json = "HW2/results/retrieval_results.json"
    target_folder = "HW2/data/target_music_list_60s"
    reference_folder = "HW2/data/reference_music_list_60s"
    
    evaluate_retrieval_melody(retrieval_json, target_folder, reference_folder)
    
    print(f"\n✓ Done")


if __name__ == "__main__":
    main()