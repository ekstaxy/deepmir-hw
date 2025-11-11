import torch
from torch.utils.data import Dataset, DataLoader
from miditok import REMI, CPWord, TokenizerConfig
from symusic import Score
from pathlib import Path
import random
import argparse
import numpy as np

class Dataset_Pop1K7(Dataset):
    """Dataset that tokenizes MIDI on-the-fly with augmentation and 32-bar chunking"""
    
    def __init__(
        self, 
        midi_files, 
        tokenizer, 
        tokenizer_type='REMI',
        bars_per_chunk=32,
        pitch_augment_range=(-5, 5),
        velocity_augment_range=(-4, 4),
        augment_prob=0.5,
    ):
        self.midi_files = midi_files
        self.tokenizer = tokenizer
        self.tokenizer_type = tokenizer_type
        self.bars_per_chunk = bars_per_chunk
        self.pitch_augment_range = pitch_augment_range
        self.velocity_augment_range = velocity_augment_range
        self.augment_prob = augment_prob
        
        if tokenizer_type == 'REMI':
            self.bos_id = tokenizer["BOS_None"]
            self.eos_id = tokenizer["EOS_None"]
            self.pad_id = tokenizer["PAD_None"]
            self.bar_token = "Bar_None"
        elif tokenizer_type == 'CPWORD':
            self.bos_token = self._get_cpword_special_token('BOS')
            self.eos_token = self._get_cpword_special_token('EOS')
            self.pad_token = self._get_cpword_special_token('PAD')
            self.bar_type_id = tokenizer.vocab[1]['Bar_None']
        
        print("Analyzing MIDI files...")
        self.valid_chunks = []
        
        for file_idx, midi_file in enumerate(midi_files):
            try:
                tokens = tokenizer(midi_file)
                if isinstance(tokens, list):
                    tokens = tokens[0]
                
                if tokenizer_type == 'REMI':
                    bar_positions = [
                        i for i, token in enumerate(tokens.tokens) 
                        if token == self.bar_token
                    ]
                else:
                    bar_positions = [
                        i for i, token_compound in enumerate(tokens.ids)
                        if token_compound[1] == self.bar_type_id
                    ]
                
                num_bars = len(bar_positions)
                
                for start_bar in range(0, num_bars, bars_per_chunk - 8):
                    chunk_bars = min(bars_per_chunk, num_bars - start_bar)
                    if chunk_bars >= 8:
                        self.valid_chunks.append((file_idx, start_bar, chunk_bars))

            except Exception as e:
                print(f"Error processing {midi_file}: {e}")
                continue
        
        print(f"Found {len(self.valid_chunks)} valid 32-bar chunks from {len(midi_files)} MIDI files")
    
    def _get_cpword_special_token(self, token_type):
        token_str = f"{token_type}_None"
        return [
            self.tokenizer.vocab[i][token_str]
            for i in range(8)
        ]
        
    def __len__(self):
        return len(self.valid_chunks)
    
    def _augment_midi(self, score):
        if random.random() < self.augment_prob and self.pitch_augment_range:
            pitch_offset = random.randint(*self.pitch_augment_range)
            if pitch_offset != 0:
                for track in score.tracks:
                    for note in track.notes:
                        note.pitch = max(0, min(127, note.pitch + pitch_offset))
        
        if random.random() < self.augment_prob and self.velocity_augment_range:
            for track in score.tracks:
                for note in track.notes:
                    velocity_offset = random.randint(*self.velocity_augment_range)
                    note.velocity = max(1, min(127, note.velocity + velocity_offset))
        
        return score
    
    def __getitem__(self, idx):
        file_idx, start_bar, num_bars = self.valid_chunks[idx]
        midi_file = self.midi_files[file_idx]

        score = Score(str(midi_file))
        score = self._augment_midi(score)

        tokens = self.tokenizer(score)
        if isinstance(tokens, list):
            tokens = tokens[0]

        if self.tokenizer_type == 'REMI':
            return self._process_remi_tokens(tokens, idx)
        else:
            return self._process_cpword_tokens(tokens, idx)

    def _process_remi_tokens(self, tokens, chunk_idx):
        max_token_id = len(self.tokenizer) - 1
        token_ids = tokens.ids
        token_strings = tokens.tokens

        for i, t in enumerate(token_ids):
            if t >= len(self.tokenizer):
                token_ids[i] = 0
                token_strings[i] = "PAD_None"

        bar_positions = [
            i for i, token in enumerate(token_strings)
            if token == self.bar_token
        ]

        chunk = self._extract_chunk(token_ids, bar_positions, chunk_idx)
        sequence = [self.bos_id] + chunk + [self.eos_id]

        return torch.tensor(sequence, dtype=torch.long)

    def _process_cpword_tokens(self, tokens, chunk_idx):
        token_ids = np.array(tokens.ids)

        for i in range(8):
            vocab_size = len(self.tokenizer.vocab[i])
            mask = token_ids[:, i] >= vocab_size
            if mask.any():
                token_ids[mask, i] = 0

        bar_positions = [
            i for i in range(len(token_ids))
            if token_ids[i, 1] == self.bar_type_id
        ]

        chunk = self._extract_chunk(token_ids, bar_positions, chunk_idx)

        if len(chunk) > 3072:
            chunk = chunk[:3072]

        sequence = np.vstack([
            self.bos_token,
            chunk,
            self.eos_token
        ])

        return torch.tensor(sequence, dtype=torch.long)

    def _extract_chunk(self, token_ids, bar_positions, chunk_idx):
        """
        Extract a chunk of tokens based on bar positions

        Args:
            token_ids: Token sequence (1D for REMI, 2D for CPWord)
            bar_positions: List of indices where bars occur
            chunk_idx: Index into self.valid_chunks to get chunk info

        Returns:
            Extracted chunk of tokens
        """
        file_idx, start_bar, num_bars = self.valid_chunks[chunk_idx]

        # Handle case where start_bar is within available bars
        if start_bar < len(bar_positions):
            start_idx = bar_positions[start_bar]

            # Calculate end position (start_bar + desired chunk size, or end of file)
            end_bar = min(start_bar + self.bars_per_chunk, len(bar_positions))
            if end_bar < len(bar_positions):
                end_idx = bar_positions[end_bar]
            else:
                end_idx = len(token_ids)

            chunk = token_ids[start_idx:end_idx]
        else:
            # Fallback: start_bar >= available bars
            # This should rarely happen if valid_chunks was built correctly
            # Take from beginning up to bars_per_chunk
            if len(bar_positions) > 0:
                end_bar = min(self.bars_per_chunk, len(bar_positions))
                if end_bar < len(bar_positions):
                    end_idx = bar_positions[end_bar]
                else:
                    end_idx = len(token_ids)
                chunk = token_ids[:end_idx]
            else:
                # No bars found - take first 3072 tokens as safety
                if len(token_ids) > 3072:
                    chunk = token_ids[:3072]
                else:
                    chunk = token_ids

        return chunk


def collate_fn_dynamic(batch, pad_id, tokenizer_type='REMI'):
    if tokenizer_type == 'REMI':
        return _collate_remi(batch, pad_id)
    else:
        return _collate_cpword(batch, pad_id)


def _collate_remi(batch, pad_id):
    # max_len = max(len(seq) for seq in batch)
    max_len = 3072 + 2
    
    padded_sequences = []
    attention_masks = []
    
    for seq in batch:
        seq_len = len(seq)
        attention_mask = [1] * seq_len
        
        if seq_len < max_len:
            padding = [pad_id] * (max_len - seq_len)
            seq = torch.cat([seq, torch.tensor(padding, dtype=torch.long)])
            attention_mask = attention_mask + [0] * (max_len - seq_len)
        
        padded_sequences.append(seq)
        attention_masks.append(torch.tensor(attention_mask, dtype=torch.long))
    
    input_ids = torch.stack(padded_sequences)
    attention_mask = torch.stack(attention_masks)
    
    labels = input_ids[:, 1:].clone()
    input_ids = input_ids[:, :-1]
    attention_mask = attention_mask[:, :-1]
    
    labels[attention_mask == 0] = -100
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask
    }


def _collate_cpword(batch, pad_token):
    max_len = max(seq.shape[0] for seq in batch)
    max_len  = 3072 + 2
    
    padded_sequences = []
    labels = []
    loss_masks = []
    
    for seq in batch:
        seq_len = seq.shape[0]
        loss_mask = [1] * (seq_len - 1)
        
        if seq_len < max_len:
            padding_len = max_len - seq_len
            padding = torch.tensor([pad_token] * padding_len, dtype=torch.long)
            seq = torch.cat([seq, padding], dim=0)
            loss_mask = loss_mask + [0] * padding_len
        
        padded_sequences.append(seq[:-1])
        labels.append(seq[1:])
        loss_masks.append(torch.tensor(loss_mask, dtype=torch.float))
    
    x = torch.stack(padded_sequences)
    target = torch.stack(labels)
    loss_mask = torch.stack(loss_masks)
    
    return {
        'x': x,
        'target': target,
        'loss_mask': loss_mask
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset preparation script")
    parser.add_argument(
        "-d", "--data", 
        type=str, 
        action="append", 
        required=True, 
        help="Path to MIDI file directories"
    )
    parser.add_argument(
        "-t", "--tokenizer_type",
        type=str,
        default="REMI",
        choices=["REMI", "CPWORD"],
        help="Tokenizer type"
    )

    args = parser.parse_args()

    midi_files = []
    for directory in args.data:
        midi_files.extend(Path(directory).rglob("*.mid"))

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

    if args.tokenizer_type == "REMI":
        tokenizer = REMI(config)
    else:
        tokenizer = CPWord(config)

    dataset = Dataset_Pop1K7(
        midi_files=midi_files,
        tokenizer=tokenizer,
        tokenizer_type=args.tokenizer_type,
        bars_per_chunk=32,
        pitch_augment_range=(-5, 5),
        velocity_augment_range=(-10, 10),
        augment_prob=0.5,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_dynamic(
            batch,
            dataset.pad_token if args.tokenizer_type == 'CPWORD' else dataset.pad_id,
            args.tokenizer_type
        )
    )
    
    for batch in dataloader:
        print(f"Batch keys: {batch.keys()}")
        if args.tokenizer_type == 'REMI':
            print(f"Input shape: {batch['input_ids'].shape}")
            print(f"Labels shape: {batch['labels'].shape}")
        else:
            print(f"X shape: {batch['x'].shape}")
            print(f"Target shape: {batch['target'].shape}")
        break