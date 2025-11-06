import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import numpy as np
import torch
from torch.utils.data import DataLoader
from data.dataset import Dataset_Pop1K7, collate_fn_dynamic
from model.model_transformers import GPT2, TransformerXL, CPWordModel
from transformers import get_cosine_schedule_with_warmup
from miditok import REMI, CPWord, TokenizerConfig
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
    if tokenizer_type == 'REMI':
        collate_fn = partial(collate_fn_dynamic, pad_id=tokenizer["PAD_None"], tokenizer_type='REMI')
    else:
        pad_token = [tokenizer.vocab["PAD_None"] for i in range(8)]
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


def create_model(model_type='gpt2', tokenizer=None, **kwargs):
    if model_type == 'gpt2':
        model = GPT2(**kwargs)
    elif model_type == 'transformer-xl':
        model = TransformerXL(**kwargs)
    elif model_type == 'cpword':
        n_token = [len(tokenizer.vocab[i]) for i in range(8)]
        model = CPWordModel(n_token=n_token, is_training=True)
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
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        x = batch['x'].to(device)
        print(x)
        target = batch['target'].to(device)
        loss_mask = batch['loss_mask'].to(device)
        
        loss_tempo, loss_chord, loss_barbeat, loss_type, loss_pitch, loss_duration, loss_velocity = model.train_step(
            x, target, loss_mask
        )
        
        loss = loss_tempo + loss_chord + loss_barbeat + loss_type + loss_pitch + loss_duration + loss_velocity
        
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


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
    model_type='gpt2'
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
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
    
    print(f"Training Configuration: epochs={num_epochs}, batch_size={batch_size}, lr={lr}, warmup_steps={warmup_steps}, device={device}, model_type={model_type}")
    print(f"Total training steps: {num_training_steps} ({len(dataloader)} batches/epoch)")
    
    train_epoch_fn = train_epoch_cpword if model_type == 'cpword' else train_epoch_remi
    
    for epoch in range(num_epochs):
        avg_loss = train_epoch_fn(model, dataloader, optimizer, scheduler, device)
        losses.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.2e}")
        
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
            print(f"Checkpoint saved: {checkpoint_path}")
            
            plot_loss_curve(losses, save_path=os.path.join(checkpoint_dir, 'loss_curve.png'))
    
    np.save(os.path.join(checkpoint_dir, 'training_losses.npy'), np.array(losses))
    
    return model, losses


def main():
    args = parse_args()
    
    TOKENIZER_PARAMS = {
        "pitch_range": (1, 127),
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
    
    midi_dir = Path(args.midi_dir)
    all_midi_files = list(midi_dir.glob("**/*.mid")) + list(midi_dir.glob("**/*.midi"))
    
    if args.num_files is not None:
        midi_files = all_midi_files[:args.num_files]
        print(f"TEST MODE: Using {len(midi_files)}/{len(all_midi_files)} MIDI files")
    else:
        midi_files = all_midi_files
        print(f"Found {len(midi_files)} MIDI files")
    
    if args.model_type == 'cpword':
        tokenizer = CPWord(TokenizerConfig(**TOKENIZER_PARAMS))
    else:
        tokenizer = REMI(TokenizerConfig(**TOKENIZER_PARAMS))
    
    print(f"Vocabulary size: {len(tokenizer.vocab) if args.model_type != 'cpword' else [len(tokenizer.vocab[i]) for i in range(8)]}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    tokenizer.save_params(os.path.join(args.checkpoint_dir, 'tokenizer_config.json'))
    
    print(f"Creating dataset: bars_per_chunk={args.bars_per_chunk}")
    dataset = Dataset_Pop1K7(
        midi_files, 
        tokenizer,
        tokenizer_type='CPWORD' if args.model_type == 'cpword' else 'REMI',
        bars_per_chunk=args.bars_per_chunk,
        pitch_augment_range=(-5, 5),
        velocity_augment_range=(-10, 10),
        augment_prob=0.5,
    )
    
    print(f"Dataset created: {len(dataset)} chunks")
    
    sample_lengths = []
    for i in range(min(100, len(dataset))):
        sample = dataset[i]
        if args.model_type == 'cpword':
            sample_lengths.append(sample.shape[0])
        else:
            sample_lengths.append(len(sample))
    
    print(f"Sequence length stats: min={min(sample_lengths)}, max={max(sample_lengths)}, mean={np.mean(sample_lengths):.1f}, model_max={args.max_seq_len}")
    
    print(f"Creating model: type={args.model_type}, layers={args.n_layer}, embd={args.n_embd}, heads={args.n_head}")
    
    if args.model_type == 'cpword':
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
    print(f"Model created: {total_params:,} parameters")
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("WARNING: Training on CPU")
    
    print("Starting training...")
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
    
    print(f"Training complete. Final loss: {losses[-1]:.4f}")
    print(f"Checkpoints saved: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()