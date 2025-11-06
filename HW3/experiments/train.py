import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import torch
from torch.utils.data import DataLoader
from data.dataset import Dataset_Pop1K7, collate_fn_dynamic
from model.model_transformers import GPT2, TransformerXL, CPWordModel
from transformers import get_cosine_schedule_with_warmup
from miditok import REMI, CPWord, TokenizerConfig
from miditok import TokSequence 

from functools import partial
import matplotlib.pyplot as plt
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train music generation model')
    
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps for scheduler')
    
    parser.add_argument('--model_type', type=str, default='cpword', choices=['gpt2', 'transformer-xl', 'cpword'])
    parser.add_argument('--n_layer', type=int, default=12, help='Number of transformer layers')
    parser.add_argument('--n_embd', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--n_head', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--max_seq_len', type=int, default=1024, help='Maximum sequence length')
    
    parser.add_argument('--bars_per_chunk', type=int, default=32, help='Bars per training chunk')
    parser.add_argument('--midi_dir', type=str, default='/kaggle/input/pop1k7/Pop1K7/midi_analyzed')
    parser.add_argument('--num_files', type=int, default=None, help='Limit number of MIDI files')
    
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--save_every', type=int, default=20, help='Save checkpoint every N epochs')
    
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    return parser.parse_args()


def create_dataloader(dataset, batch_size=8, shuffle=True, num_workers=2, tokenizer=None, tokenizer_type='REMI'):
    """Create dataloader with appropriate collate function"""
    if tokenizer_type == 'REMI':
        collate_fn = partial(collate_fn_dynamic, pad_id=tokenizer["PAD_None"], tokenizer_type='REMI')
    else:
        # For CPWord, get PAD tokens for all 8 vocabularies
        pad_token = [tokenizer.vocab[i].get("PAD_None", 0) for i in range(len(tokenizer.vocab))]
        collate_fn = partial(collate_fn_dynamic, pad_id=pad_token, tokenizer_type='CPWORD')
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )


def create_optimizer(model, lr=5e-4, weight_decay=0.01):
    return torch.optim.AdamW(
        model.parameters(), 
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay
    )


def create_scheduler(optimizer, num_training_steps, num_warmup_steps=1000):
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return scheduler


def create_model(model_type='cpword', tokenizer=None, **kwargs):
    """Create model based on type"""
    if model_type == 'gpt2':
        model = GPT2(**kwargs)
    elif model_type == 'transformer-xl':
        model = TransformerXL(**kwargs)
    elif model_type == 'cpword':
        # FIXED: Pass tokenizer to auto-extract vocab sizes
        model = CPWordModel(tokenizer=tokenizer, is_training=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model


def plot_loss_curve(losses, save_path='./checkpoints/loss_curve.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss vs Epoch', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss curve saved to {save_path}")


def train_epoch_remi(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


def train_epoch_cpword(model, dataloader, optimizer, scheduler, device):
    """
    FIXED: Training epoch for CPWord model with 8 vocabularies
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Track individual losses for monitoring
    loss_accumulator = {
        'family': 0.0, 'bar': 0.0, 'pitch': 0.0, 'velocity': 0.0,
        'duration': 0.0, 'chord': 0.0, 'rest': 0.0, 'tempo': 0.0
    }
    
    for batch_idx, batch in enumerate(dataloader):
        x = batch['x'].to(device)
        target = batch['target'].to(device)
        loss_mask = batch['loss_mask'].to(device)
        
        # Verify shapes
        assert x.shape[2] == 8, f"X has wrong shape: {x.shape}"
        assert target.shape[2] == 8, f"Target has wrong shape: {target.shape}"
        
        # Get losses for all 8 vocabularies
        loss_family, loss_bar, loss_pitch, loss_velocity, loss_duration, \
        loss_chord, loss_rest, loss_tempo = model.train_step(x, target, loss_mask)
        
        # Total loss is sum of all vocabulary losses
        loss = loss_family + loss_bar + loss_pitch + loss_velocity + \
               loss_duration + loss_chord + loss_rest + loss_tempo
        
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Accumulate individual losses
        loss_accumulator['family'] += loss_family.item()
        loss_accumulator['bar'] += loss_bar.item()
        loss_accumulator['pitch'] += loss_pitch.item()
        loss_accumulator['velocity'] += loss_velocity.item()
        loss_accumulator['duration'] += loss_duration.item()
        loss_accumulator['chord'] += loss_chord.item()
        loss_accumulator['rest'] += loss_rest.item()
        loss_accumulator['tempo'] += loss_tempo.item()
        
        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"    Batch {batch_idx + 1}/{len(dataloader)}: loss={loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    
    # Print average losses for each vocabulary
    print(f"  Average losses per vocabulary:")
    for key in loss_accumulator:
        loss_accumulator[key] /= num_batches
        print(f"    {key:12s}: {loss_accumulator[key]:.4f}")
    
    return avg_loss


def inference_cpword_32bars(model, tokenizer, device, temperature=1.0, target_bars=32):
    """
    Generate a 32-bar sequence using CPWord model
    
    Args:
        model: CPWordModel instance
        tokenizer: CPWord tokenizer
        device: torch device
        temperature: sampling temperature
        target_bars: number of bars to generate (default 32)
    
    Returns:
        numpy array of shape [seq_len, 8] containing generated tokens
    """
    model.eval()
    
    # Get BOS token for CPWord (8-dimensional)
    bos_token = np.array([[
        tokenizer.vocab[i].get("BOS_None", 0) 
        for i in range(8)
    ]])
    
    # Get bar token ID from vocabulary 1 (bar vocabulary)
    bar_token_id = tokenizer.vocab[1].get("Bar_None", None)
    if bar_token_id is None:
        # Try to find any Bar token
        for key in tokenizer.vocab[1].keys():
            if "Bar" in key:
                bar_token_id = tokenizer.vocab[1][key]
                break
    
    with torch.no_grad():
        final_res = []
        memory = None
        h = None
        
        cnt_bar = 0
        init_t = torch.from_numpy(bos_token).long().to(device)
        
        # Process BOS token
        input_ = init_t[0, :].unsqueeze(0).unsqueeze(0)
        final_res.append(bos_token[0, :][None, ...])
        h, y_family, memory = model.forward_hidden(input_, memory, is_training=False)
        
        # Generate tokens
        max_len = 3000  # Safety limit
        for gen_step in range(max_len):
            # Sample next token
            next_arr = model.forward_output_sampling(h, y_family)
            final_res.append(next_arr[None, ...])
            
            # Count bars (check vocabulary index 1)
            if bar_token_id is not None and next_arr[1] == bar_token_id:
                cnt_bar += 1
            
            # Check if we've generated enough bars
            if cnt_bar >= target_bars:
                break
            
            # Continue generation
            input_ = torch.from_numpy(next_arr).long().to(device)
            input_ = input_.unsqueeze(0).unsqueeze(0)
            h, y_family, memory = model.forward_hidden(input_, memory, is_training=False)
    
    final_res = np.concatenate(final_res, axis=0)
    print(final_res)
    print(f"  Generated sequence shape: {final_res.shape}, Bars: {cnt_bar}")
    
    return final_res


def save_cpword_tokens_as_midi(tokens, tokenizer, output_path):
    """
    Convert CPWord tokens to MIDI and save
    
    Args:
        tokens: numpy array of shape [seq_len, 8]
        tokenizer: CPWord tokenizer
        output_path: path to save MIDI file
    
    Returns:
        bool: True if successful, False otherwise
    """

    # Convert NaN or invalid values to padding tokens
    if np.isnan(tokens).any():
        print(f"  Warning: Found NaN values, replacing with PAD tokens")
        pad_tokens = np.array([tokenizer.vocab[i].get("PAD_None", 0) for i in range(8)])
        nan_mask = np.isnan(tokens)
        for i in range(8):
            tokens[nan_mask[:, i], i] = pad_tokens[i]
    
    # Ensure all values are integers
    tokens = tokens.astype(np.int64)
    
    # Clip values to valid range for each vocabulary
    tokens = tokens.astype(np.int32)
    for i in range(8):
        tokens[:, i] = np.clip(tokens[:, i], 0, len(tokenizer.vocab[i]) - 1)

    print(tokens)
    # Convert to MIDI
    midi = tokenizer([tokens])
    midi.dump_midi(output_path)
    print(f"  ✓ MIDI saved to: {output_path}")
    return True


def train(
    model,
    train_dataset,
    num_epochs=100,
    batch_size=8,
    lr=5e-4,
    device='cuda',
    checkpoint_dir='./checkpoints',
    save_every=20,
    warmup_steps=1000,
    tokenizer=None,
    model_type='cpword'
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create results directory for inference outputs
    results_dir = os.path.join(checkpoint_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    tokenizer_type = 'CPWORD' if model_type == 'cpword' else 'REMI'
    dataloader = create_dataloader(
        train_dataset, 
        batch_size=batch_size, 
        tokenizer=tokenizer,
        tokenizer_type=tokenizer_type
    )
    
    optimizer = create_optimizer(model, lr=lr)
    
    num_training_steps = len(dataloader) * num_epochs
    scheduler = create_scheduler(optimizer, num_training_steps, num_warmup_steps=warmup_steps)
    
    model = model.to(device)
    
    losses = []
    
    print(f"\n{'='*70}")
    print(f"Training Configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Device: {device}")
    print(f"  Model type: {model_type}")
    print(f"  Total training steps: {num_training_steps} ({len(dataloader)} batches/epoch)")
    print(f"  Results directory: {results_dir}")
    print(f"{'='*70}\n")
    
    train_epoch_fn = train_epoch_cpword if model_type == 'cpword' else train_epoch_remi
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        avg_loss = train_epoch_fn(model, dataloader, optimizer, scheduler, device)
        losses.append(avg_loss)
        
        current_lr = scheduler.get_last_lr()[0]
        print(f"  Total Loss: {avg_loss:.4f} | LR: {current_lr:.2e}\n")
        
        if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'losses': losses,
            }, checkpoint_path)
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")
            
            plot_loss_curve(losses, save_path=os.path.join(checkpoint_dir, 'loss_curve.png'))
            
            # =====================================================================
            # NEW: Generate inference sample at checkpoint
            # =====================================================================
            if model_type == 'cpword':
                print(f"\n  Generating 32-bar inference sample...")
                generated_tokens = inference_cpword_32bars(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    temperature=1.0,
                    target_bars=32
                )
                
                # Save as MIDI
                midi_filename = f'inference_epoch_{epoch+1}.mid'
                midi_path = os.path.join(results_dir, midi_filename)
                save_cpword_tokens_as_midi(generated_tokens, tokenizer, midi_path)

                print(f"  ✓ Inference MIDI saved: {midi_path}")
            # =====================================================================

    np.save(os.path.join(checkpoint_dir, 'training_losses.npy'), np.array(losses))
    
    return model, losses


def main():
    args = parse_args()
    
    print(f"\n{'='*70}")
    print(f"CPWord Music Generation Training")
    print(f"{'='*70}\n")
    
    # Tokenizer configuration
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
        "tempo_range": (40, 220),
    }
    
    # Load MIDI files
    midi_dir = Path(args.midi_dir)
    all_midi_files = list(midi_dir.glob("**/*.mid")) + list(midi_dir.glob("**/*.midi"))
    
    if args.num_files is not None:
        midi_files = all_midi_files[:args.num_files]
        print(f"TEST MODE: Using {len(midi_files)}/{len(all_midi_files)} MIDI files")
    else:
        midi_files = all_midi_files
        print(f"Found {len(midi_files)} MIDI files")
    
    # Create tokenizer
    if args.model_type == 'cpword':
        tokenizer = CPWord(TokenizerConfig(**TOKENIZER_PARAMS))
        print(f"\nCPWord Tokenizer created:")
        print(f"  Number of vocabularies: {len(tokenizer.vocab)}")
        print(f"  Vocabulary sizes: {[len(tokenizer.vocab[i]) for i in range(len(tokenizer.vocab))]}")
        
        # Verify it's 8 vocabularies
        if len(tokenizer.vocab) != 8:
            print(f"\n⚠️  WARNING: Expected 8 vocabularies, got {len(tokenizer.vocab)}")
            print(f"  Model expects: family/bar/pitch/velocity/duration/chord/rest/tempo")
            print(f"  Check your tokenizer configuration!")
    else:
        tokenizer = REMI(TokenizerConfig(**TOKENIZER_PARAMS))
        print(f"REMI Tokenizer created: {len(tokenizer.vocab)} tokens")
    
    # Save tokenizer
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    tokenizer.save_params(os.path.join(args.checkpoint_dir, 'tokenizer_config.json'))
    print(f"✓ Tokenizer config saved")
    
    # Create dataset
    print(f"\nCreating dataset (bars_per_chunk={args.bars_per_chunk})...")
    dataset = Dataset_Pop1K7(
        midi_files, 
        tokenizer,
        tokenizer_type='CPWORD' if args.model_type == 'cpword' else 'REMI',
        bars_per_chunk=args.bars_per_chunk,
        pitch_augment_range=(-5, 5),
        velocity_augment_range=(-10, 10),
        augment_prob=0.5,
    )
    
    print(f"✓ Dataset created: {len(dataset)} chunks")
    
    # Check sample lengths
    sample_lengths = []
    for i in range(min(100, len(dataset))):
        sample = dataset[i]
        sample_lengths.append(sample.shape[0])
    
    print(f"\nSequence length statistics:")
    print(f"  Min: {min(sample_lengths)}, Max: {max(sample_lengths)}, Mean: {np.mean(sample_lengths):.1f}")
    
    # Test a sample
    print(f"\nTesting sample shape...")
    test_sample = dataset[0]
    print(f"  Sample shape: {test_sample.shape}")
    if args.model_type == 'cpword':
        assert test_sample.shape[1] == 8, f"Expected 8 vocabularies, got {test_sample.shape[1]}"
        print(f"  ✓ Correct shape for CPWord!")
    
    # Create model
    print(f"\nCreating model: {args.model_type}")
    
    if args.model_type == 'cpword':
        # FIXED: Pass tokenizer to get vocab sizes automatically
        model = create_model(
            model_type=args.model_type,
            tokenizer=tokenizer
        )
    else:
        model = create_model(
            model_type=args.model_type,
            vocab_size=len(tokenizer.vocab),
            max_seq_len=args.max_seq_len,
            n_layer=args.n_layer,
            n_embd=args.n_embd,
            n_head=args.n_head,
            bos_token_id=tokenizer["BOS_None"],
            eos_token_id=tokenizer["EOS_None"],
            pad_token_id=tokenizer["PAD_None"],
        )
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created: {total_params:,} trainable parameters")
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("⚠️  WARNING: Training on CPU (this will be slow!)")
    else:
        print(f"✓ Using device: {device}")
    
    # Train
    print(f"\n{'='*70}")
    print(f"Starting training...")
    print(f"{'='*70}\n")
    
    trained_model, losses = train(
        model=model,
        train_dataset=dataset,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        warmup_steps=args.warmup_steps,
        tokenizer=tokenizer,
        model_type=args.model_type
    )
    
    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Checkpoints saved to: {args.checkpoint_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()