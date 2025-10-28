import msclap
import numpy as np
from pathlib import Path
import json
import librosa

def load_clap_model():
    """Load CLAP model using msclap."""
    model = msclap.CLAP(version='2023', use_cuda=True)
    return model

def calculate_audio_audio_similarity(model, audio_path1, audio_path2):
    """Calculate similarity between two audio files."""
    audio1_embed = model.get_audio_embeddings([str(audio_path1)])
    audio2_embed = model.get_audio_embeddings([str(audio_path2)])
    
    similarity = model.compute_similarity(audio1_embed, audio2_embed)
    return float(similarity[0][0])

def calculate_text_audio_similarity(model, text, audio_path):
    """Calculate similarity between text and audio file."""
    text_embed = model.get_text_embeddings([text])
    audio_embed = model.get_audio_embeddings([str(audio_path)])
    
    similarity = model.compute_similarity(audio_embed, text_embed)
    return float(similarity[0][0])

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
    """Evaluate CLAP similarity between two inputs."""
    input1_path = Path(input1)
    input2_path = Path(input2)
    
    print("Loading CLAP model...")
    model = load_clap_model()
    print("✓ CLAP model loaded\n")
    
    is_input1_json = input1_path.suffix == '.json'
    is_input2_json = input2_path.suffix == '.json'
    
    if is_input1_json and is_input2_json:
        print("Error: Both inputs are JSON files.")
        return
    
    if not is_input1_json and not is_input2_json:
        # Audio-Audio mode
        print("Mode: Audio-to-Audio comparison")
        audio_folder1 = input1_path
        audio_folder2 = input2_path
        
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
                similarity = calculate_audio_audio_similarity(model, audio_file1, audio_file2)
                results[normalized_name] = similarity
                print(f"  {normalized_name}: {similarity:.4f}")
                
            except Exception as e:
                print(f"  Error {normalized_name}: {e}")
                results[normalized_name] = None
        
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
                similarity = calculate_text_audio_similarity(model, text_prompt, audio_file)
                results[normalized_name] = similarity
                print(f"  {normalized_name}: {similarity:.4f}")
                
            except Exception as e:
                print(f"  Error {normalized_name}: {e}")
                results[normalized_name] = None

        output_path = audio_folder / "clap_text_audio_similarities.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Saved: {output_path}")


if __name__ == "__main__":

    input_1_path = [
        "HW2/results/musicgen-small_generated_music", 
        # "HW2/results/flamingo3_audio_captioning.json",
        "HW2/results/musicControlLite_generated_music/musicControlLite_melody_generated_0_guidance",
        "HW2/results/musicControlLite_generated_music/musicControlLite_melody_generated_1_guidance",
        "HW2/results/musicControlLite_generated_music/musicControlLite_melody_generated_2_guidance",
        "HW2/results/musicControlLite_generated_music/musicControlLite_melody_generated_3_guidance",
        "HW2/results/musicControlLite_generated_music/musicControlLite_melody&rythym_generated_1_guidance",
        "HW2/results/musicControlLite_generated_music/musicControlLite_rythym_generated_1_guidance"
    ]
    input_2_path = [
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
            "HW2/results/flamingo3_audio_captioning.json"
        )
    for path in input_2_path:
        evaluate_clap(
            path,
            "HW2/data/target_music_list_60s"
        )
    # evaluate_clap(
    #     "HW2/results/musicgen-small_generated_music", 
    #     "HW2/data/target_music_list_60s"
    # )