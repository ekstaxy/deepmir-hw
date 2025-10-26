import os
import json
import torch
import scipy.io.wavfile
import numpy as np
# !pip install transformer
from transformers import AutoProcessor, MusicgenForConditionalGeneration

# Setup
output_dir = '/kaggle/working/generated_music'
os.makedirs(output_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("/kaggle/input/musicgen/pytorch/small/1/")
model = MusicgenForConditionalGeneration.from_pretrained("/kaggle/input/musicgen/pytorch/small/1/").to(device)

# Load captions
with open('/kaggle/working/DeepMIR-HW/HW2/results/flamingo3_audio_captioning.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

filenames = list(data.keys())
captions = list(data.values())

# Generate longer audio by continuation
def generate_long_audio(caption, target_duration_seconds=60, chunk_duration_tokens=1500):
    """Generate audio longer than 30 seconds using continuation"""
    sampling_rate = model.config.audio_encoder.sampling_rate
    audio_chunks = []
    
    # Calculate how many chunks needed
    num_chunks = int(np.ceil(target_duration_seconds / 30))
    
    for i in range(num_chunks):
        print(f"  Generating chunk {i+1}/{num_chunks}...")
        
        inputs = processor(text=[caption], padding=True, return_tensors="pt").to(device)
        
        # Generate chunk (max 30 seconds = ~1500 tokens)
        audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=min(chunk_duration_tokens, 1500))
        
        audio_chunks.append(audio_values[0, 0].cpu().numpy())
    
    # Concatenate all chunks
    full_audio = np.concatenate(audio_chunks)
    
    return full_audio, sampling_rate

# Generate and save
for idx, caption in enumerate(captions):
    print(f"Generating {idx+1}/{len(captions)}: {filenames[idx]}")
    
    # Generate long audio (60 seconds)
    audio_data, sampling_rate = generate_long_audio(caption, target_duration_seconds=60)
    
    # Save
    output_path = os.path.join(output_dir, f'{filenames[idx].replace(".wav", "")}.wav')
    scipy.io.wavfile.write(output_path, rate=sampling_rate, data=audio_data)
    
    print(f"Saved: {output_path}")