"""
Simple Inference Pipeline for Music Generation
Generates MIDI files and converts them to audio using FluidSynth
"""
import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from inference import (
    load_tokenizer,
    load_model_cpword,
    generate_unconditional_cpword,
    save_cpword_tokens_as_midi
)

try:
    from midi2audio import FluidSynth
    FLUIDSYNTH_AVAILABLE = True
except ImportError:
    print("Warning: midi2audio not installed. Audio conversion will be skipped.")
    print("Install with: pip install midi2audio")
    FLUIDSYNTH_AVAILABLE = False


def convert_midi_to_audio(midi_path, audio_path, soundfont=None):
    """
    Convert MIDI file to audio using FluidSynth

    Args:
        midi_path: path to input MIDI file
        audio_path: path to output audio file (WAV)
        soundfont: optional path to soundfont file (.sf2)

    Returns:
        bool: True if successful, False otherwise
    """
    if not FLUIDSYNTH_AVAILABLE:
        print(f"  Skipping audio conversion (FluidSynth not available)")
        return False

    try:
        if soundfont:
            fs = FluidSynth(sound_font=soundfont)
        else:
            # Use default soundfont
            fs = FluidSynth()

        fs.midi_to_audio(str(midi_path), str(audio_path))
        print(f"  ✓ Audio saved: {audio_path}")
        return True

    except Exception as e:
        print(f"  ✗ Error converting to audio: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Simple Music Generation Inference Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 5 samples with default settings
  python simple_inference.py --checkpoint checkpoints/checkpoint_epoch_500.pt --num_samples 5

  # Generate samples in custom output directory
  python simple_inference.py --checkpoint checkpoints/checkpoint_epoch_500.pt --output_dir my_samples

  # Generate with custom soundfont
  python simple_inference.py --checkpoint checkpoints/checkpoint_epoch_500.pt --soundfont path/to/soundfont.sf2
        """
    )

    # Required arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.pt file)'
    )

    # Optional arguments
    parser.add_argument(
        '--tokenizer_config',
        type=str,
        default=None,
        help='Path to tokenizer config (default: auto-detect from checkpoint dir)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./inference_results',
        help='Output directory for generated files (default: ./inference_results)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='Number of samples to generate (default: 5)'
    )
    parser.add_argument(
        '--n_bars',
        type=int,
        default=32,
        help='Number of bars to generate (default: 32)'
    )
    parser.add_argument(
        '--soundfont',
        type=str,
        default=None,
        help='Path to SoundFont file (.sf2) for audio synthesis'
    )
    parser.add_argument(
        '--no_audio',
        action='store_true',
        help='Skip audio conversion, only generate MIDI files'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu, default: cuda)'
    )

    args = parser.parse_args()

    # Auto-detect tokenizer config if not provided
    if args.tokenizer_config is None:
        checkpoint_dir = Path(args.checkpoint).parent
        tokenizer_config = checkpoint_dir / 'tokenizer_config.json'
        if tokenizer_config.exists():
            args.tokenizer_config = str(tokenizer_config)
            print(f"Auto-detected tokenizer config: {args.tokenizer_config}")
        else:
            print("ERROR: Could not find tokenizer_config.json")
            print(f"Please specify --tokenizer_config or place tokenizer_config.json in {checkpoint_dir}")
            return

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    midi_dir = Path(args.output_dir) / 'midi'
    audio_dir = Path(args.output_dir) / 'audio'
    os.makedirs(midi_dir, exist_ok=True)
    if not args.no_audio:
        os.makedirs(audio_dir, exist_ok=True)

    print("\n" + "="*70)
    print("Simple Music Generation Inference Pipeline")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Tokenizer: {args.tokenizer_config}")
    print(f"Output dir: {args.output_dir}")
    print(f"Samples: {args.num_samples}")
    print(f"Bars per sample: {args.n_bars}")
    print(f"Device: {args.device}")
    if not args.no_audio:
        print(f"Audio conversion: {'Enabled' if FLUIDSYNTH_AVAILABLE else 'Disabled (midi2audio not installed)'}")
        if args.soundfont:
            print(f"SoundFont: {args.soundfont}")
    print("="*70 + "\n")

    # Load tokenizer and model
    tokenizer = load_tokenizer(args.tokenizer_config, tokenizer_type='CPWORD')
    model = load_model_cpword(args.checkpoint, tokenizer, args.device)

    # Generate samples
    print("\nGenerating music samples...")
    sequences = generate_unconditional_cpword(
        model=model,
        tokenizer=tokenizer,
        num_samples=args.num_samples,
        target_bars=args.n_bars,
        device=args.device
    )

    # Save MIDI files and convert to audio
    print(f"\nSaving results...")
    checkpoint_name = Path(args.checkpoint).stem
    midi_success = 0
    audio_success = 0

    for i, tokens in enumerate(sequences):
        sample_name = f"{checkpoint_name}_sample_{i:03d}"

        # Save MIDI
        midi_file = midi_dir / f"{sample_name}.mid"
        if save_cpword_tokens_as_midi(tokens, tokenizer, str(midi_file)):
            midi_success += 1

            # Convert to audio
            if not args.no_audio:
                audio_file = audio_dir / f"{sample_name}.wav"
                if convert_midi_to_audio(midi_file, audio_file, args.soundfont):
                    audio_success += 1

    # Summary
    print("\n" + "="*70)
    print("Generation Complete!")
    print("="*70)
    print(f"MIDI files: {midi_success}/{args.num_samples} saved to {midi_dir}")
    if not args.no_audio:
        print(f"Audio files: {audio_success}/{args.num_samples} saved to {audio_dir}")
    print(f"\nOutput directory: {args.output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
