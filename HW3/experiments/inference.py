"""
inference.py - Music generation inference
"""

import torch
import argparse
import os
from pathlib import Path
from miditok import REMI
from model.model_transformers import GPT2, TransformerXL
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Generate music')
    
    # Generation parameters
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--tokenizer_config', type=str, required=True, help='Path to tokenizer config')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of samples to generate')
    parser.add_argument('--n_bars', type=int, default=32, help='Number of bars to generate')
    
    # Sampling parameters (3 configurations for homework)
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.95, help='Nucleus sampling')
    parser.add_argument('--sampling_method', type=str, default='top_k', 
                       choices=['greedy', 'top_k', 'nucleus'], help='Sampling method')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='gpt2', choices=['gpt2', 'transformer-xl'])
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    return parser.parse_args()


def load_model(checkpoint_path, tokenizer, model_type='gpt2', device='cuda'):
    """Load trained model from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config from checkpoint or recreate
    if model_type == 'gpt2':
        model = GPT2(
            vocab_size=len(tokenizer.vocab),
            max_seq_len=1024,
            n_embd=512,
            n_layer=12,
            n_head=8,
            bos_token_id=tokenizer["BOS_None"],
            eos_token_id=tokenizer["EOS_None"],
            pad_token_id=tokenizer["PAD_None"],
        )
    elif model_type == 'transformer-xl':
        model = TransformerXL(
            vocab_size=len(tokenizer.vocab),
            d_model=512,
            n_layer=12,
            n_head=8,
            bos_token_id=tokenizer["BOS_None"],
            eos_token_id=tokenizer["EOS_None"],
            pad_token_id=tokenizer["PAD_None"],
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Loaded model from epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.4f}")
    return model


def generate_unconditional(
    model,
    tokenizer,
    num_samples=20,
    target_bars=32,
    temperature=1.0,
    top_k=50,
    top_p=0.95,
    sampling_method='top_k',
    device='cuda'
):
    """
    Unconditional generation - start from BOS token
    
    Uses sampling during inference (not teacher forcing!)
    ✅ Greedy / Top-K / Nucleus sampling
    ✅ Temperature scaling
    """
    generated_sequences = []
    
    bos_id = tokenizer["BOS_None"]
    eos_id = tokenizer["EOS_None"]
    bar_id = tokenizer["Bar_None"]
    
    print(f"\nGenerating {num_samples} samples with {sampling_method} sampling...")
    print(f"  Temperature: {temperature}")
    if sampling_method == 'top_k':
        print(f"  Top-K: {top_k}")
    elif sampling_method == 'nucleus':
        print(f"  Top-P: {top_p}")
    
    for i in range(num_samples):
        print(f"  Sample {i+1}/{num_samples}...", end='\r')
        
        # Start with BOS token
        input_ids = torch.tensor([[bos_id]]).to(device)
        generated = [bos_id]
        bar_count = 0
        max_length = 2048  # Safety limit
        
        with torch.no_grad():
            while bar_count < target_bars and len(generated) < max_length:
                # Get model predictions
                outputs = model(input_ids=input_ids)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply sampling method
                if sampling_method == 'greedy':
                    next_token = torch.argmax(next_token_logits).item()
                
                elif sampling_method == 'top_k':
                    # Top-K sampling
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    probs = torch.softmax(top_k_logits, dim=-1)
                    next_token_idx = torch.multinomial(probs, 1).item()
                    next_token = top_k_indices[next_token_idx].item()
                
                elif sampling_method == 'nucleus':
                    # Nucleus (top-p) sampling
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                    
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                
                generated.append(next_token)
                
                # Count bars
                if next_token == bar_id:
                    bar_count += 1
                
                # Stop if EOS
                if next_token == eos_id:
                    break
                
                # Update input
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]]).to(device)], dim=1)
        
        # Remove BOS/EOS/PAD
        clean_tokens = [t for t in generated if t not in [bos_id, eos_id, tokenizer["PAD_None"]]]
        generated_sequences.append(clean_tokens)
        
    print(f"\n✅ Generated {len(generated_sequences)} sequences")
    return generated_sequences


def tokens_to_midi(token_ids, tokenizer, output_path):
    """Convert token IDs back to MIDI file using miditok"""
    try:
        # Convert tokens to MIDI
        midi = tokenizer.tokens_to_midi([token_ids])
        
        # Save MIDI
        midi.dump_midi(output_path)
        return True
    except Exception as e:
        print(f"Error converting tokens to MIDI: {e}")
        return False


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_config}...")
    tokenizer = REMI(params=args.tokenizer_config)
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    
    # Load model
    model = load_model(args.checkpoint, tokenizer, args.model_type, args.device)
    
    # Generate music
    sequences = generate_unconditional(
        model=model,
        tokenizer=tokenizer,
        num_samples=args.num_samples,
        target_bars=args.n_bars,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        sampling_method=args.sampling_method,
        device=args.device
    )
    
    # Save MIDI files
    print(f"\nSaving MIDI files to {args.output_dir}...")
    checkpoint_name = Path(args.checkpoint).stem
    success_count = 0
    
    for i, tokens in enumerate(sequences):
        output_file = os.path.join(
            args.output_dir,
            f"{checkpoint_name}_{args.sampling_method}_temp{args.temperature}_{i:03d}.mid"
        )
        if tokens_to_midi(tokens, tokenizer, output_file):
            success_count += 1
    
    print(f"✅ Successfully saved {success_count}/{len(sequences)} MIDI files")
    
    # Convert to WAV (optional - requires fluidsynth)
    print("\nTo convert to WAV, run:")
    print(f"  fluidsynth -ni soundfont.sf2 {args.output_dir}/*.mid -F output.wav -r 44100")


if __name__ == "__main__":
    main()