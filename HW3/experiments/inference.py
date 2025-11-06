"""
inference.py - Music generation inference with CPWord support and continuation
"""

import torch
import argparse
import os
from pathlib import Path
from miditok import REMI, CPWord
from model.model_transformers import GPT2, TransformerXL, CPWordModel
import numpy as np
from symusic import Score


def parse_args():
    parser = argparse.ArgumentParser(description='Generate music')
    
    # Generation mode
    parser.add_argument('--mode', type=str, default='unconditional', 
                       choices=['unconditional', 'continuation'],
                       help='Generation mode: unconditional or continuation')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--tokenizer_config', type=str, required=True, help='Path to tokenizer config')
    parser.add_argument('--model_type', type=str, default='cpword', 
                       choices=['gpt2', 'transformer-xl', 'cpword'])
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of samples to generate (unconditional mode)')
    parser.add_argument('--n_bars', type=int, default=32, help='Number of bars to generate')
    
    # Continuation mode parameters
    parser.add_argument('--prompt_dir', type=str, default=None, 
                       help='Directory containing prompt MIDI files (continuation mode)')
    parser.add_argument('--prompt_bars', type=int, default=8, 
                       help='Number of bars to use as prompt (continuation mode)')
    
    # Sampling parameters
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.95, help='Nucleus sampling')
    parser.add_argument('--sampling_method', type=str, default='top_k', 
                       choices=['greedy', 'top_k', 'nucleus'], help='Sampling method')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--tokenizer_type', type=str, default='CPWORD',
                       choices=['REMI', 'CPWORD'], help='Tokenizer type')
    
    return parser.parse_args()


def load_tokenizer(config_path, tokenizer_type='CPWORD'):
    """Load tokenizer from config"""
    print(f"Loading {tokenizer_type} tokenizer from {config_path}...")
    if tokenizer_type == 'CPWORD':
        tokenizer = CPWord(params=config_path)
    else:
        tokenizer = REMI(params=config_path)
    
    if tokenizer_type == 'CPWORD':
        print(f"  Vocabulary sizes: {[len(tokenizer.vocab[i]) for i in range(len(tokenizer.vocab))]}")
    else:
        print(f"  Vocabulary size: {len(tokenizer.vocab)}")
    return tokenizer


def load_model_cpword(checkpoint_path, tokenizer, device='cuda'):
    """Load CPWord model from checkpoint"""
    print(f"Loading CPWord checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = CPWordModel(tokenizer=tokenizer, is_training=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.4f}")
    return model


def load_model_remi(checkpoint_path, tokenizer, model_type='gpt2', device='cuda'):
    """Load REMI-based model from checkpoint"""
    print(f"Loading {model_type} checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
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
    
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.4f}")
    return model


def generate_unconditional_cpword(
    model,
    tokenizer,
    num_samples=20,
    target_bars=32,
    device='cuda'
):
    """
    Unconditional generation for CPWord model
    
    Args:
        model: CPWordModel instance
        tokenizer: CPWord tokenizer
        num_samples: number of sequences to generate
        target_bars: number of bars to generate
        device: torch device
    
    Returns:
        List of generated token sequences (each is numpy array of shape [seq_len, 8])
    """
    generated_sequences = []
    
    # Get BOS token (8-dimensional)
    bos_token = np.array([[
        tokenizer.vocab[i].get("BOS_None", 0) 
        for i in range(8)
    ]])
    
    # Get bar token ID from vocabulary 1 (bar vocabulary)
    bar_token_id = tokenizer.vocab[1].get("Bar_None", None)
    if bar_token_id is None:
        for key in tokenizer.vocab[1].keys():
            if "Bar" in key:
                bar_token_id = tokenizer.vocab[1][key]
                break
    
    print(f"\nGenerating {num_samples} samples unconditionally...")
    print(f"  Target bars: {target_bars}")
    
    for i in range(num_samples):
        print(f"  Sample {i+1}/{num_samples}...", end='\r')
        
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
            max_len = 3000
            for gen_step in range(max_len):
                next_arr = model.forward_output_sampling(h, y_family)
                final_res.append(next_arr[None, ...])
                
                # Count bars
                if bar_token_id is not None and next_arr[1] == bar_token_id:
                    cnt_bar += 1
                
                if cnt_bar >= target_bars:
                    break
                
                # Continue generation
                input_ = torch.from_numpy(next_arr).long().to(device)
                input_ = input_.unsqueeze(0).unsqueeze(0)
                h, y_family, memory = model.forward_hidden(input_, memory, is_training=False)
        
        final_res = np.concatenate(final_res, axis=0)
        generated_sequences.append(final_res)
    
    print(f"\n✓ Generated {len(generated_sequences)} sequences")
    return generated_sequences


def extract_prompt_tokens_cpword(midi_path, tokenizer, prompt_bars=8):
    """
    Extract first N bars from a MIDI file as prompt tokens for CPWord
    
    Args:
        midi_path: path to MIDI file
        tokenizer: CPWord tokenizer
        prompt_bars: number of bars to extract
    
    Returns:
        numpy array of shape [seq_len, 8] containing prompt tokens
    """
    # Load and tokenize MIDI
    score = Score(str(midi_path))
    tokens = tokenizer(score)
    
    if isinstance(tokens, list):
        tokens = tokens[0]
    
    # Convert to numpy array
    token_ids = np.array(tokens.ids)
    
    # Find bar positions (vocabulary index 1)
    bar_token_id = tokenizer.vocab[1].get("Bar_None", None)
    if bar_token_id is None:
        for key in tokenizer.vocab[1].keys():
            if "Bar" in key:
                bar_token_id = tokenizer.vocab[1][key]
                break
    
    bar_positions = []
    for i in range(len(token_ids)):
        if token_ids[i, 1] == bar_token_id:
            bar_positions.append(i)
    
    # Extract first N bars
    if len(bar_positions) >= prompt_bars:
        end_idx = bar_positions[prompt_bars]
        prompt_tokens = token_ids[:end_idx]
    else:
        # If less than N bars, use all available
        prompt_tokens = token_ids
    
    return prompt_tokens


def generate_continuation_cpword(
    model,
    tokenizer,
    prompt_tokens,
    target_bars=32,
    device='cuda'
):
    """
    Generate continuation given prompt tokens
    
    Args:
        model: CPWordModel instance
        tokenizer: CPWord tokenizer
        prompt_tokens: numpy array of shape [seq_len, 8]
        target_bars: total number of bars (including prompt)
        device: torch device
    
    Returns:
        numpy array of shape [seq_len, 8] - concatenated prompt + generation
    """
    # Count bars in prompt
    bar_token_id = tokenizer.vocab[1].get("Bar_None", None)
    if bar_token_id is None:
        for key in tokenizer.vocab[1].keys():
            if "Bar" in key:
                bar_token_id = tokenizer.vocab[1][key]
                break
    
    prompt_bar_count = 0
    for i in range(len(prompt_tokens)):
        if prompt_tokens[i, 1] == bar_token_id:
            prompt_bar_count += 1
    
    bars_to_generate = target_bars - prompt_bar_count
    
    print(f"  Prompt bars: {prompt_bar_count}, Generating: {bars_to_generate} more bars")
    
    with torch.no_grad():
        final_res = []
        memory = None
        h = None
        
        # Process prompt tokens
        for step in range(len(prompt_tokens)):
            token = prompt_tokens[step]
            input_ = torch.from_numpy(token).long().to(device).unsqueeze(0).unsqueeze(0)
            final_res.append(token[None, ...])
            h, y_family, memory = model.forward_hidden(input_, memory, is_training=False)
        
        # Generate continuation
        cnt_bar = prompt_bar_count
        max_len = 3000
        for gen_step in range(max_len):
            next_arr = model.forward_output_sampling(h, y_family)
            final_res.append(next_arr[None, ...])
            
            # Count bars
            if bar_token_id is not None and next_arr[1] == bar_token_id:
                cnt_bar += 1
            
            if cnt_bar >= target_bars:
                break
            
            # Continue generation
            input_ = torch.from_numpy(next_arr).long().to(device)
            input_ = input_.unsqueeze(0).unsqueeze(0)
            h, y_family, memory = model.forward_hidden(input_, memory, is_training=False)
    
    final_res = np.concatenate(final_res, axis=0)
    print(f"  Total bars in output: {cnt_bar}")
    
    return final_res


def save_cpword_tokens_as_midi(tokens, tokenizer, output_path):
    """
    Convert CPWord tokens to MIDI and save
    
    Args:
        tokens: numpy array of shape [seq_len, 8]
        tokenizer: CPWord tokenizer
        output_path: path to save MIDI file
    
    Returns:
        bool: True if successful
    """
    try:
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
        for i in range(8):
            vocab_size = len(tokenizer.vocab[i])
            tokens[:, i] = np.clip(tokens[:, i], 0, vocab_size - 1)
            
        # Convert to MIDI
        midi = tokenizer.decode([tokens])

        # Save MIDI file
        midi.dump_midi(output_path)
        return True
        
    except Exception as e:
        print(f"  Error converting tokens to MIDI: {e}")
        return False


def generate_unconditional_remi(
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
    Unconditional generation for REMI-based models (GPT2, TransformerXL)
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
        
        input_ids = torch.tensor([[bos_id]]).to(device)
        generated = [bos_id]
        bar_count = 0
        max_length = 2048
        
        with torch.no_grad():
            while bar_count < target_bars and len(generated) < max_length:
                outputs = model(input_ids=input_ids)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                next_token_logits = logits[0, -1, :] / temperature
                
                if sampling_method == 'greedy':
                    next_token = torch.argmax(next_token_logits).item()
                
                elif sampling_method == 'top_k':
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    probs = torch.softmax(top_k_logits, dim=-1)
                    next_token_idx = torch.multinomial(probs, 1).item()
                    next_token = top_k_indices[next_token_idx].item()
                
                elif sampling_method == 'nucleus':
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                    
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                
                generated.append(next_token)
                
                if next_token == bar_id:
                    bar_count += 1
                
                if next_token == eos_id:
                    break
                
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]]).to(device)], dim=1)
        
        clean_tokens = [t for t in generated if t not in [bos_id, eos_id, tokenizer["PAD_None"]]]
        generated_sequences.append(clean_tokens)
        
    print(f"\n✓ Generated {len(generated_sequences)} sequences")
    return generated_sequences


def save_remi_tokens_as_midi(token_ids, tokenizer, output_path):
    """Convert REMI token IDs to MIDI file"""
    try:
        midi = tokenizer.decode([token_ids])
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
    tokenizer = load_tokenizer(args.tokenizer_config, args.tokenizer_type)
    
    # Load model
    if args.tokenizer_type == 'CPWORD':
        model = load_model_cpword(args.checkpoint, tokenizer, args.device)
    else:
        model = load_model_remi(args.checkpoint, tokenizer, args.model_type, args.device)
    
    # =========================================================================
    # Generation Mode: Unconditional
    # =========================================================================
    if args.mode == 'unconditional':
        print(f"\n{'='*70}")
        print(f"UNCONDITIONAL GENERATION MODE")
        print(f"{'='*70}")
        
        if args.tokenizer_type == 'CPWORD':
            sequences = generate_unconditional_cpword(
                model=model,
                tokenizer=tokenizer,
                num_samples=args.num_samples,
                target_bars=args.n_bars,
                device=args.device
            )
            
            # Save MIDI files
            print(f"\nSaving MIDI files to {args.output_dir}...")
            checkpoint_name = Path(args.checkpoint).stem
            success_count = 0
            
            for i, tokens in enumerate(sequences):
                output_file = os.path.join(
                    args.output_dir,
                    f"{checkpoint_name}_unconditional_{i:03d}.mid"
                )
                if save_cpword_tokens_as_midi(tokens, tokenizer, output_file):
                    success_count += 1
            
            print(f"✓ Successfully saved {success_count}/{len(sequences)} MIDI files")
        
        else:  # REMI
            sequences = generate_unconditional_remi(
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
                if save_remi_tokens_as_midi(tokens, tokenizer, output_file):
                    success_count += 1
            
            print(f"✓ Successfully saved {success_count}/{len(sequences)} MIDI files")
    
    # =========================================================================
    # Generation Mode: Continuation
    # =========================================================================
    elif args.mode == 'continuation':
        print(f"\n{'='*70}")
        print(f"CONTINUATION GENERATION MODE")
        print(f"{'='*70}")
        
        if args.prompt_dir is None:
            raise ValueError("--prompt_dir must be specified for continuation mode")
        
        # Get all MIDI files from prompt directory
        prompt_dir = Path(args.prompt_dir)
        midi_files = list(prompt_dir.glob("*.mid")) + list(prompt_dir.glob("*.midi"))
        
        if len(midi_files) == 0:
            raise ValueError(f"No MIDI files found in {prompt_dir}")
        
        print(f"\nFound {len(midi_files)} prompt MIDI files")
        print(f"Prompt bars: {args.prompt_bars}, Target total: {args.n_bars} bars")
        
        if args.tokenizer_type == 'CPWORD':
            checkpoint_name = Path(args.checkpoint).stem
            success_count = 0
            
            for i, midi_file in enumerate(midi_files):
                print(f"\n[{i+1}/{len(midi_files)}] Processing: {midi_file.name}")
                
                try:
                    # Extract prompt tokens
                    prompt_tokens = extract_prompt_tokens_cpword(
                        midi_file, tokenizer, args.prompt_bars
                    )
                    
                    # Generate continuation
                    full_sequence = generate_continuation_cpword(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_tokens=prompt_tokens,
                        target_bars=args.n_bars,
                        device=args.device
                    )
                    
                    # Save result
                    output_file = os.path.join(
                        args.output_dir,
                        f"{checkpoint_name}_continuation_{midi_file.stem}.mid"
                    )
                    
                    if save_cpword_tokens_as_midi(full_sequence, tokenizer, output_file):
                        success_count += 1
                        print(f"  ✓ Saved: {output_file}")
                    
                except Exception as e:
                    print(f"  ✗ Error processing {midi_file.name}: {e}")
                    continue
            
            print(f"\n✓ Successfully generated {success_count}/{len(midi_files)} continuations")
        
        else:
            print("ERROR: Continuation mode currently only supports CPWORD tokenizer")
            return
    
    print(f"\n{'='*70}")
    print(f"Generation complete!")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()