from miditok import REMI, CPWord, TokenizerConfig
from pathlib import Path
import shutil

# Your tokenizer setup
TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": {(0, 4): 8, (4, 12): 4, (12, 16): 8},
    "num_velocities": 64,
    "use_velocities": True,
    "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
    "use_note_duration_program": True,
    "use_chords": True,
    "chord_tokens_with_root_note": True,
    "use_rests": True,
    "use_tempos": True,
    "use_time_signatures": False,
    "use_programs": False,
    "num_tempos": 32,
    "tempo_range": (50, 200),
}
config = TokenizerConfig(**TOKENIZER_PARAMS)
tokenizer = REMI(config)

# Get MIDI files
midi_dir = Path("/kaggle/input/pop1k7/Pop1K7/midi_analyzed/src_001")
midi_files = list(midi_dir.glob("*.mid")) + list(midi_dir.glob("*.midi"))
print(f"Found {len(midi_files)} MIDI files")

# Tokenize
tokenizer.tokenize_dataset(
    files_paths=midi_files, 
    out_dir="/kaggle/working/midi_analyzed_tokenized/",
    overwrite_mode=True
)

# Create zip file
output_dir = Path("/kaggle/working/midi_analyzed_tokenized")
zip_path = "/kaggle/working/tokenized_data"

print("Creating zip file...")
shutil.make_archive(zip_path, 'zip', output_dir)
print(f"Zip file created: {zip_path}.zip")

# Download using Kaggle's file system
from IPython.display import FileLink
FileLink(f"{zip_path}.zip")