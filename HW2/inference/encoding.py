import sys

# Get token from command line argument
if len(sys.argv) > 1:
    HF_TOKEN = sys.argv[1]
else:
    HF_TOKEN = ""  # ← Or paste your token here

if not HF_TOKEN:
    print("❌ Please provide token: python script.py YOUR_TOKEN")
    sys.exit(1)

# Login
from huggingface_hub import login
login(token=HF_TOKEN)

import torch
import torchaudio
from diffusers import AutoencoderOobleck
import os
from pathlib import Path
import numpy as np

# Load encoder
print("Loading encoder...")
vae = AutoencoderOobleck.from_pretrained("stabilityai/stable-audio-open-1.0", subfolder="vae")
device = "cuda" if torch.cuda.is_available() else "cpu"
vae = vae.to(device)
vae.eval()
print(f"Encoder loaded on {device}")

# Paths
reference_music_dir = "HW2/data/reference_music_list_60s"
target_music_dir = "HW2/data/target_music_list_60s"
output_dir = "HW2/data/latents"

os.makedirs(f"{output_dir}/reference", exist_ok=True)
os.makedirs(f"{output_dir}/target", exist_ok=True)

def encode_audio_file(audio_path, target_length=44100*60):
    waveform, sample_rate = torchaudio.load(audio_path)
    
    if sample_rate != 44100:
        resampler = torchaudio.transforms.Resample(sample_rate, 44100)
        waveform = resampler(waveform)
    
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    
    if waveform.shape[1] < target_length:
        num_repeats = (target_length // waveform.shape[1]) + 1
        waveform = waveform.repeat(1, num_repeats)
        waveform = waveform[:, :target_length]
    else:
        waveform = waveform[:, :target_length]
    
    waveform = waveform.unsqueeze(0).to(device)
    
    with torch.no_grad():
        latent = vae.encode(waveform).latent_dist.sample()
    
    return latent.cpu().numpy()

def process_directory(input_dir, output_subdir, dir_name):
    audio_files = []
    for ext in ['.wav', '.mp3', '.flac', '.ogg']:
        audio_files.extend(Path(input_dir).glob(f'*{ext}'))
    
    print(f"\nProcessing {dir_name}: {len(audio_files)} files")
    
    for i, audio_file in enumerate(audio_files):
        print(f"  [{i+1}/{len(audio_files)}] {audio_file.name}...")
        latent = encode_audio_file(str(audio_file))
        output_path = f"{output_dir}/{output_subdir}/{audio_file.stem}.npy"
        np.save(output_path, latent)
        print(f"    → Saved (shape: {latent.shape})")

process_directory(reference_music_dir, "reference", "Reference")
process_directory(target_music_dir, "target", "Target")

print("\n✅ DONE!")