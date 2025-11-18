# CPWord Inference Guide

## Overview
Your inference now supports **configurable sampling parameters** for each vocabulary in the CPWord model!

## Basic Usage

```bash
python experiments/inference.py \
  --checkpoint ./checkpoints/checkpoint_epoch_100.pt \
  --tokenizer_config ./checkpoints/tokenizer_config.json \
  --output_dir ./results \
  --num_samples 20 \
  --n_bars 32
```

## Advanced: Tuning Sampling Parameters

### Temperature Parameters (`--temp_*`)
**Higher temperature = More random/creative**
**Lower temperature = More conservative/predictable**

```bash
# More creative generation (higher temps)
python experiments/inference.py \
  --checkpoint ./checkpoints/model.pt \
  --tokenizer_config ./checkpoints/tokenizer_config.json \
  --temp_pitch 1.5 \      # More varied melodies
  --temp_velocity 7.0 \   # More dynamic variation
  --temp_duration 2.5     # More rhythm variety

# More conservative generation (lower temps)
python experiments/inference.py \
  --checkpoint ./checkpoints/model.pt \
  --tokenizer_config ./checkpoints/tokenizer_config.json \
  --temp_pitch 0.7 \      # Safer melodies
  --temp_velocity 3.0 \   # Less dynamic range
  --temp_duration 1.5     # More regular rhythms
```

### Nucleus Sampling Parameters (`--p_*`)
**Higher p = More diversity** (samples from larger probability mass)
**Lower p = More focused** (samples from highest probability tokens only)

```bash
# More diverse harmonies
python experiments/inference.py \
  --checkpoint ./checkpoints/model.pt \
  --tokenizer_config ./checkpoints/tokenizer_config.json \
  --p_chord 0.95 \        # More chord variety
  --p_pitch 0.95          # More pitch variety

# More predictable/safe output
python experiments/inference.py \
  --checkpoint ./checkpoints/model.pt \
  --tokenizer_config ./checkpoints/tokenizer_config.json \
  --p_chord 0.99 \        # Conservative chords
  --p_pitch 0.85          # Stay close to likely pitches
```

## All Available Parameters

### CPWord-Specific Sampling
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--temp_family` | 1.0 | Temperature for event type (Metric/Note) |
| `--temp_bar` | 1.2 | Temperature for bar positions |
| `--temp_pitch` | 1.0 | Temperature for note pitches (C2-C8) |
| `--temp_velocity` | 5.0 | Temperature for note dynamics |
| `--temp_duration` | 2.0 | Temperature for note durations |
| `--temp_chord` | 1.0 | Temperature for chord selection |
| `--temp_rest` | 1.2 | Temperature for rests/time-shifts |
| `--temp_tempo` | 1.2 | Temperature for tempo changes |
| `--p_family` | 0.90 | Nucleus p for event type |
| `--p_pitch` | 0.9 | Nucleus p for pitches |
| `--p_duration` | 0.9 | Nucleus p for durations |
| `--p_chord` | 0.99 | Nucleus p for chords |
| `--p_tempo` | 0.9 | Nucleus p for tempo |

## Example Presets

### Preset 1: Jazz-like (More Freedom)
```bash
python experiments/inference.py \
  --checkpoint ./checkpoints/model.pt \
  --tokenizer_config ./checkpoints/tokenizer_config.json \
  --temp_pitch 1.3 \
  --temp_velocity 6.0 \
  --temp_duration 2.5 \
  --temp_chord 1.5 \
  --p_chord 0.95
```

### Preset 2: Classical (Conservative)
```bash
python experiments/inference.py \
  --checkpoint ./checkpoints/model.pt \
  --tokenizer_config ./checkpoints/tokenizer_config.json \
  --temp_pitch 0.8 \
  --temp_velocity 3.0 \
  --temp_duration 1.5 \
  --temp_chord 0.9 \
  --p_chord 0.99 \
  --p_pitch 0.85
```

### Preset 3: Experimental
```bash
python experiments/inference.py \
  --checkpoint ./checkpoints/model.pt \
  --tokenizer_config ./checkpoints/tokenizer_config.json \
  --temp_pitch 2.0 \
  --temp_velocity 8.0 \
  --temp_duration 3.0 \
  --temp_chord 2.0 \
  --p_chord 0.90 \
  --p_pitch 0.95
```

## Understanding the Output

During generation, you'll see:
```
Generating 20 samples unconditionally...
  Target bars: 32
  Bar token ID: 5
  Sampling parameters:
    Family: t=1.00, p=0.90
    Pitch:  t=1.00, p=0.90
    Velocity: t=5.00
    Duration: t=2.00, p=0.90
    Chord:  t=1.00, p=0.99
    Tempo:  t=1.20, p=0.90
```

## Tips for Experimentation

1. **Start with defaults** - They work well!
2. **Adjust one parameter at a time** - Easier to understand effects
3. **Pitch range is constrained** - C2-C8 (MIDI 36-108) for musical output
4. **Higher velocity temp** - More dynamic expression
5. **Lower chord p** - More harmonic variety
6. **Save your presets** - Create bash scripts for reproducible results

## Constraints (Always Active)

These are built into the model and cannot be disabled:
- ✅ **Family**: Only Metric (4) or Note (5) - no special tokens
- ✅ **Pitch**: C2-C8 range only (vocab indices 20-92)
- ✅ **All vocabularies**: Special tokens (PAD/BOS/EOS/MASK/Ignore) blocked
- ✅ **Metric events**: Use Bar/Chord/Rest/Tempo, ignore Pitch/Velocity/Duration
- ✅ **Note events**: Use Pitch/Velocity/Duration, ignore Bar/Chord/Rest/Tempo

Happy generating! 🎵
