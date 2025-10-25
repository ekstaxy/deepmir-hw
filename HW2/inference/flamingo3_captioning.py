"""
Simplest Version - Process Audio Files to JSON
"""

from gradio_client import Client, handle_file
from pathlib import Path
import json

# 1. Setup
client = Client("nvidia/audio-flamingo-3")
folder = "/kaggle/working/DeepMIR-HW/HW2/data/target_music_list_60s"  # Change this
results = {}

# 2. Loop through files
for audio_file in Path(folder).glob("*.wav"):
    print(f"Processing: {audio_file.name}")
    
    caption = client.predict(
        audio_file=handle_file(str(audio_file)),
        prompt_text="Describe this music",
        api_name="/single_turn_infer"
    )
    
    results[audio_file.name] = caption

# 3. Also process .mp3 files
for audio_file in Path(folder).glob("*.mp3"):
    print(f"Processing: {audio_file.name}")
    
    caption = client.predict(
        audio_file=handle_file(str(audio_file)),
        prompt_text="Describe this music",
        api_name="/single_turn_infer"
    )
    
    results[audio_file.name] = caption

# 4. Save to JSON
with open("captions.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✓ Done! Processed {len(results)} files")
print("Results saved to: captions.json")