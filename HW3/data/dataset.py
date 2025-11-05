import torch
from torch.utils.data import Dataset, DataLoader
from miditok import REMI, TokenizerConfig
from symusic import Score
from pathlib import Path
import random
import argparse

class Dataset_Pop1K7(Dataset):
    """Dataset that tokenizes MIDI on-the-fly with augmentation and 32-bar chunking"""
    
    def __init__(
        self, 
        midi_files, 
        tokenizer, 
        bars_per_chunk=32,
        pitch_augment_range=(-5, 5),  # Range for pitch augmentation
        velocity_augment_range=(-4, 4),  # Range for velocity augmentation
        augment_prob=0.5,  # Probability of applying augmentation
    ):
        self.midi_files = midi_files
        self.tokenizer = tokenizer
        self.bars_per_chunk = bars_per_chunk
        self.pitch_augment_range = pitch_augment_range
        self.velocity_augment_range = velocity_augment_range
        self.augment_prob = augment_prob
        
        # Special tokens
        self.bos_id = tokenizer["BOS_None"]
        self.eos_id = tokenizer["EOS_None"]
        self.pad_id = tokenizer["PAD_None"]
        self.bar_token = "Bar_None"
        
        # Pre-compute which MIDI files have enough bars
        print("Analyzing MIDI files...")
        self.valid_chunks = []  # List of (file_idx, start_bar, num_bars)
        
        for file_idx, midi_file in enumerate(midi_files):
            try:
                # Quick tokenization to count bars
                tokens = tokenizer(midi_file)
                if isinstance(tokens, list):
                    tokens = tokens[0]  # Get first track
                
                # Count bars
                bar_positions = [
                    i for i, token in enumerate(tokens.tokens) 
                    if token == self.bar_token
                ]
                
                num_bars = len(bar_positions)
                
                # Create chunks of 32 bars from this file
                for start_bar in range(0, num_bars, bars_per_chunk - 8):
                    chunk_bars = min(bars_per_chunk, num_bars - start_bar)
                    if chunk_bars >= 8:  # Only keep chunks with at least 8 bars
                        self.valid_chunks.append((file_idx, start_bar, chunk_bars))
                
            except Exception as e:
                print(f"Error processing {midi_file}: {e}")
                continue
        
        print(f"Found {len(self.valid_chunks)} valid 32-bar chunks from {len(midi_files)} MIDI files")
    
    def __len__(self):
        return len(self.valid_chunks)
    
    def _augment_midi(self, score):
        """Apply pitch and velocity augmentation to MIDI"""
        # Random pitch shift
        if random.random() < self.augment_prob and self.pitch_augment_range:
            pitch_offset = random.randint(*self.pitch_augment_range)
            if pitch_offset != 0:
                for track in score.tracks:
                    for note in track.notes:
                        note.pitch = max(0, min(127, note.pitch + pitch_offset))
        
        # Random velocity shift
        if random.random() < self.augment_prob and self.velocity_augment_range:
            velocity_offset = random.randint(*self.velocity_augment_range)
            if velocity_offset != 0:
                for track in score.tracks:
                    for note in track.notes:
                        note.velocity = max(1, min(127, note.velocity + velocity_offset))
        
        return score
    
    def __getitem__(self, idx):
        file_idx, start_bar, num_bars = self.valid_chunks[idx]
        midi_file = self.midi_files[file_idx]
        
        # Load MIDI
        score = Score(str(midi_file))
        
        # Apply augmentation
        score = self._augment_midi(score)
        
        # Tokenize
        tokens = self.tokenizer(score)
        if isinstance(tokens, list):
            tokens = tokens[0]  # Get first track
        
        token_ids = tokens.ids
        token_strings = tokens.tokens
        
        # Find bar positions
        bar_positions = [
            i for i, token in enumerate(token_strings) 
            if token == self.bar_token
        ]
        
        # Extract the specific 32-bar chunk
        if start_bar < len(bar_positions):
            start_idx = bar_positions[start_bar]
            
            # Find end position
            end_bar = min(start_bar + self.bars_per_chunk, len(bar_positions))
            if end_bar < len(bar_positions):
                end_idx = bar_positions[end_bar]
            else:
                end_idx = len(token_ids)
            
            chunk = token_ids[start_idx:end_idx]
        else:
            # Fallback: take first 32 bars
            if len(bar_positions) >= self.bars_per_chunk:
                end_idx = bar_positions[self.bars_per_chunk]
            else:
                end_idx = len(token_ids)
            chunk = token_ids[:end_idx]
        
        # Add BOS and EOS
        sequence = [self.bos_id] + chunk + [self.eos_id]
        
        return torch.tensor(sequence, dtype=torch.long)


def collate_fn_dynamic(batch, pad_id):
    """Collate function with dynamic padding"""
    # Find max length in batch
    max_len = max(len(seq) for seq in batch)
    
    padded_sequences = []
    attention_masks = []
    
    for seq in batch:
        seq_len = len(seq)
        attention_mask = [1] * seq_len
        
        # Pad
        if seq_len < max_len:
            padding = [pad_id] * (max_len - seq_len)
            seq = torch.cat([seq, torch.tensor(padding, dtype=torch.long)])
            attention_mask = attention_mask + [0] * (max_len - seq_len)
        
        padded_sequences.append(seq)
        attention_masks.append(torch.tensor(attention_mask, dtype=torch.long))
    
    # Stack
    input_ids = torch.stack(padded_sequences)
    attention_mask = torch.stack(attention_masks)
    
    # Create labels (shift by 1)
    labels = input_ids[:, 1:].clone()
    input_ids = input_ids[:, :-1]
    attention_mask = attention_mask[:, :-1]
    
    # Mask padding in labels
    labels[attention_mask == 0] = -100
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask
    }

# Dataset Testing
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Dataset preparation script")
    parser.add_argument(
        "-d", "--data", 
        type=str, 
        action="append", 
        required=True, 
        help="Path to MIDI file directories (can be specified multiple times)"
    )

    args = parser.parse_args()

    midi_files = []
    for directory in args.data:
        midi_files.extend(Path(directory).rglob("*.mid"))  # Recursively find all MIDI files

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

    dataset = Dataset_Pop1K7(
        midi_files=midi_files,
        tokenizer=REMI(config),
    )