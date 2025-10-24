# Login to Hugging Face Hub
from huggingface_hub import login
login(token="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")

import torch
import torchaudio
from diffusers import AutoencoderOobleck
import os
from pathlib import Path
import numpy as np

# Load the encoder
print("Loading encoder...")
vae = AutoencoderOobleck.from_pretrained(
    "stabilityai/stable-audio-open-1.0", 
    subfolder="vae"
)
device = "cuda" if torch.cuda.is_available() else "cpu"
vae = vae.to(device)
vae.eval()
print(f"Encoder loaded on {device}")

# Define paths - MODIFY THESE TO YOUR LOCAL PATHS
reference_music_dir = "HW2/data/reference_music_list_60s"  # ← Change this to your path
target_music_dir = "HW2/data/target_music_list_60s"        # ← Change this to your path
output_dir = "HW2/data/latents"                             # ← Where to save latents

# Create output directory
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/reference", exist_ok=True)
os.makedirs(f"{output_dir}/target", exist_ok=True)

def encode_audio_file(audio_path, target_length=44100*60):
    """Load audio file and encode to latent"""
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Resample to 44.1kHz if needed
    if sample_rate != 44100:
        resampler = torchaudio.transforms.Resample(sample_rate, 44100)
        waveform = resampler(waveform)
    
    # Convert to stereo if mono
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    
    # Repeat or trim to target length
    if waveform.shape[1] < target_length:
        # Calculate how many times to repeat
        num_repeats = (target_length // waveform.shape[1]) + 1
        # Repeat the waveform
        waveform = waveform.repeat(1, num_repeats)
        # Trim to exact target length
        waveform = waveform[:, :target_length]
    else:
        waveform = waveform[:, :target_length]
    
    # Add batch dimension and move to device
    waveform = waveform.unsqueeze(0).to(device)
    
    # Encode
    with torch.no_grad():
        latent = vae.encode(waveform).latent_dist.sample()
    
    return latent.cpu().numpy()

def process_directory(input_dir, output_subdir, dir_name):
    """Process all audio files in a directory"""
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(Path(input_dir).glob(f'*{ext}'))
    
    print(f"\nProcessing {dir_name}: {len(audio_files)} files")
    
    for i, audio_file in enumerate(audio_files):
        try:
            print(f"  [{i+1}/{len(audio_files)}] Encoding {audio_file.name}...")
            
            # Encode audio
            latent = encode_audio_file(str(audio_file))
            
            # Save latent
            output_path = f"{output_dir}/{output_subdir}/{audio_file.stem}.npy"
            np.save(output_path, latent)
            
            print(f"    → Saved to {output_path} (shape: {latent.shape})")
            
        except Exception as e:
            print(f"    ✗ Error processing {audio_file.name}: {e}")
    
    print(f"✓ Completed {dir_name}")

# Process reference music
print("\n" + "="*60)
print("ENCODING REFERENCE MUSIC")
print("="*60)
process_directory(reference_music_dir, "reference", "Reference Music")

# Process target music
print("\n" + "="*60)
print("ENCODING TARGET MUSIC")
print("="*60)
process_directory(target_music_dir, "target", "Target Music")

print("\n" + "="*60)
print("✅ ALL ENCODING COMPLETE!")
print("="*60)
print(f"Latents saved to: {output_dir}")