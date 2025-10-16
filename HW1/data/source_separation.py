# Import necessary libraries
import os
import glob
import torch
import torchaudio
import matplotlib.pyplot as plt

from IPython.display import Audio, display

import utils
from pathlib import Path

from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.transforms import Fade, Resample
from torchaudio.utils  import download_asset

# Construct pipeline
bundle = HDEMUCS_HIGH_MUSDB_PLUS
model = bundle.get_model().eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
sample_rate = bundle.sample_rate

def separate_audio(
        file_path, 
        output_dir="data/processed/test",
        segment=10,
        overlap=0.20
        ):
    """
    Separates the audio file into its components using the pre-trained model.
    
    Args:
        file_path (str): Path to the input audio file.
        output_dir (str): Directory to save the separated audio files.
    """
    # Load and preprocess audio
    waveform, sr = torchaudio.load(file_path)
    if sr != sample_rate:
        waveform = Resample(orig_freq=sr, new_freq=sample_rate)(waveform)
    
    # Move to device
    waveform = waveform.to(device)

    chuck_len = int(segment * sample_rate * (1 + overlap))
    start = 0
    end = chuck_len
    overlap_frame = int(overlap * sample_rate)
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frame), fade_shape='linear')
    # Batch size 1, 4 sources, 2 channels, time
    final = torch.zeros(1, 4, 2, waveform.size(1))

    while start < waveform.size(1)-overlap_frame:
        chunk = waveform[:, start:end]
        # Pad chunk if it's shorter than chuck_len
        if chunk.size(1) < chuck_len:
            padding = torch.zeros(waveform.size(0), chuck_len - chunk.size(1)).to(device)
            chunk = torch.cat((chunk, padding), dim=1)
        
        # Ensure chunk has 2 channels
        if chunk.size(0) == 1:
            chunk = chunk.repeat(2, 1)
        # Separate chunk
        with torch.inference_mode():
            separated_chunk = model(chunk.unsqueeze(0))

        separated_chunk = fade(separated_chunk)
        separated_chunk = separated_chunk[:, :, :, :end-start]
        final[:, :, :, start:end] += separated_chunk
        # Overlap-add
        if start == 0:
            fade.fade_in_len = int(overlap_frame)
            start += chuck_len - overlap_frame
        else:
            start += chuck_len
        
        end += chuck_len
        if end > waveform.size(1):
            end = waveform.size(1)
    
    # Save separated sources
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    source_names = ['drums', 'bass', 'other', 'vocals']
    vocals = final[0][3]

    output_path = os.path.join(output_dir, f"{Path(file_path).stem}_vocals.mp3")
    vocals = Resample(orig_freq=sample_rate, new_freq=sr)(vocals)
    if vocals.dim() == 2 and vocals.size(0) == 2:
        vocals_mono = vocals.mean(dim=0, keepdim=True)
    else:
        vocals_mono = vocals
    torchaudio.save(output_path, vocals_mono.cpu(), sr, format="mp3")
    print(f"Saved vocals to {output_path}")
    
    display(Audio(vocals_mono.cpu().numpy(), rate=sr))
    return output_path, vocals_mono, sample_rate

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    """
    Plots the waveform of the audio signal.
    
    Args:
        waveform (Tensor): The audio waveform tensor.
        sample_rate (int): The sample rate of the audio.
        title (str): Title of the plot.
        xlim (tuple): Limits for the x-axis.
        ylim (tuple): Limits for the y-axis.
    """
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    plt.figure(figsize=(15, 5))
    for c in range(num_channels):
        plt.plot(time_axis, waveform[c], label=f'Channel {c+1}')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.legend()
    plt.grid()
    plt.show()

def main(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".mp3") or file.endswith(".wav"):
                file_path = os.path.join(root, file)
                # Compute relative path from input_dir
                rel_path = os.path.relpath(root, input_dir)
                # Create corresponding output subfolder
                out_subdir = os.path.join(output_dir, rel_path)
                os.makedirs(out_subdir, exist_ok=True)
                separate_audio(file_path, output_dir=out_subdir)

if __name__ == "__main__":
    # For test set
    # main('data/raw/artist20/test', output_dir='data/processed/test')
    # For train_val set
    main('data/raw/artist20/train_val', output_dir='data/processed/train_val')