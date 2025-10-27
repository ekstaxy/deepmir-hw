import torch
import numpy as np
from pathlib import Path
import json
import torchaudio
from transformers import ClapModel, ClapProcessor
import msclap

def load_clap_model():
    """Load CLAP model using msclap."""
    model = msclap.CLAP(version='2023', use_cuda=True)
    return model

def load_audio(audio_path, target_sr=48000, max_duration=None):
    """Load and preprocess audio file."""
    audio, sr = torchaudio.load(audio_path)
    
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio)
    
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0)
    else:
        audio = audio[0]
    
    # Trim to max_duration if specified
    if max_duration is not None:
        max_samples = int(max_duration * target_sr)
        audio = audio[:max_samples]
    
    return audio.numpy(), target_sr

def calculate_audio_audio_similarity(model, processor, audio1, audio2, sr=48000):
    """Calculate similarity between two audio arrays."""
    # Trim to same length
    min_len = min(len(audio1), len(audio2))
    audio1 = audio1[:min_len]
    audio2 = audio2[:min_len]
    
    audio1_inputs = processor(audios=audio1, sampling_rate=sr, return_tensors="pt")
    audio2_inputs = processor(audios=audio2, sampling_rate=sr, return_tensors="pt")
    
    with torch.no_grad():
        audio1_embed = model.get_audio_features(**audio1_inputs)
        audio2_embed = model.get_audio_features(**audio2_inputs)
    
    audio1_embed = audio1_embed / audio1_embed.norm(dim=-1, keepdim=True)
    audio2_embed = audio2_embed / audio2_embed.norm(dim=-1, keepdim=True)
    
    similarity = (audio1_embed @ audio2_embed.T).item()
    return similarity

def calculate_text_audio_similarity(model, processor, text, audio, sr=48000):
    """Calculate similarity between text and audio."""
    text_inputs = processor(text=[text], return_tensors="pt")
    audio_inputs = processor(audios=audio, sampling_rate=sr, return_tensors="pt")
    
    with torch.no_grad():
        text_embed = model.get_text_features(**text_inputs)
        audio_embed = model.get_audio_features(**audio_inputs)
    
    text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
    audio_embed = audio_embed / audio_embed.norm(dim=-1, keepdim=True)
    
    similarity = (text_embed @ audio_embed.T).item()
    return similarity

def normalize_filename(filename):
    """Remove duplicate extensions."""
    name = filename
    while True:
        stem = Path(name).stem
        if stem == name:
            break
        name = stem
    return name

def find_audio_file(folder, target_name):
    """Find matching audio file."""
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

def evaluate_clap(input1, input2):
    """
    Evaluate CLAP similarity between two inputs.
    
    Args:
        input1: Audio folder path OR JSON file path
        input2: Audio folder path OR JSON file path
    
    Returns:
        Saves results to JSON file
    """
    input1_path = Path(input1)
    input2_path = Path(input2)
    
    # Load CLAP model
    print("Loading CLAP model...")
    model, processor = load_clap_model()
    print("✓ CLAP model loaded\n")
    
    # Determine mode: audio-audio or text-audio
    is_input1_json = input1_path.suffix == '.json'
    is_input2_json = input2_path.suffix == '.json'
    
    if is_input1_json and is_input2_json:
        print("Error: Both inputs are JSON files. Need one audio folder.")
        return
    
    if not is_input1_json and not is_input2_json:
        # Audio-Audio mode
        print("Mode: Audio-to-Audio comparison")
        audio_folder1 = input1_path
        audio_folder2 = input2_path
        
        # Get all audio files from folder1
        audio_files1 = []
        for ext in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
            audio_files1.extend(audio_folder1.glob(f'*{ext}'))
        
        results = {}
        
        for audio_file1 in audio_files1:
            normalized_name = normalize_filename(audio_file1.name)
            audio_file2 = find_audio_file(audio_folder2, normalized_name)
            
            if audio_file2 is None:
                print(f"  Missing: {normalized_name}")
                results[normalized_name] = None
                continue
            
            try:
                audio1, sr = load_audio(audio_file1)
                audio2, _ = load_audio(audio_file2)
                
                similarity = calculate_audio_audio_similarity(model, processor, audio1, audio2, sr)
                results[normalized_name] = similarity
                print(f"  {normalized_name}: {similarity:.4f}")
                
            except Exception as e:
                print(f"  Error {normalized_name}: {e}")
                results[normalized_name] = None
        
        # Save to folder1
        output_path = audio_folder1 / "clap_audio_similarities.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Saved: {output_path}")
    
    else:
        # Text-Audio mode
        print("Mode: Text-to-Audio comparison")
        
        if is_input1_json:
            text_json_path = input1_path
            audio_folder = input2_path
        else:
            text_json_path = input2_path
            audio_folder = input1_path
        
        # Load text JSON
        with open(text_json_path, 'r', encoding='utf-8') as f:
            text_data = json.load(f)
        
        results = {}
        
        for filename, text_prompt in text_data.items():
            normalized_name = normalize_filename(filename)
            audio_file = find_audio_file(audio_folder, normalized_name)
            
            if audio_file is None:
                print(f"  Missing: {normalized_name}")
                results[normalized_name] = None
                continue
            
            try:
                audio, sr = load_audio(audio_file)
                similarity = calculate_text_audio_similarity(model, processor, text_prompt, audio, sr)
                results[normalized_name] = similarity
                print(f"  {normalized_name}: {similarity:.4f}")
                
            except Exception as e:
                print(f"  Error {normalized_name}: {e}")
                results[normalized_name] = None
        
        # Save to same directory as JSON
        output_path = text_json_path.parent / "clap_text_audio_similarities.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Saved: {output_path}")


# Usage
if __name__ == "__main__":
    # Example 1: Text-Audio comparison
    input_1_path = [
        "HW2/results/musicgen-small_generated_music", 
        "HW2/results/flamingo3_audio_captioning.json",
        "HW2/results/musicControlLite_generated_music/musicControlLite_melody_generated_0_guidance",
        "HW2/results/musicControlLite_generated_music/musicControlLite_melody_generated_1_guidance",
        "HW2/results/musicControlLite_generated_music/musicControlLite_melody_generated_2_guidance",
        "HW2/results/musicControlLite_generated_music/musicControlLite_melody_generated_3_guidance",
        "HW2/results/musicControlLite_generated_music/musicControlLite_melody&rythym_generated_1_guidance",
        "HW2/results/musicControlLite_generated_music/musicControlLite_rythym_generated_1_guidance"
    ]
    for path in input_1_path:
        evaluate_clap(
            path,
            "HW2/data/target_music_list_60s"
        )
        