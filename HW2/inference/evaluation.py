import os
import sys
import json
from pathlib import Path
import numpy as np
import torch
import torchaudio
from audiobox_aesthetics.infer import initialize_predictor

def load_evaluation_data(retrieval_json_path, data_dir="HW2/data"):
    """Load audio pairs based on retrieval_results.json

    Args:
        retrieval_json_path: Path to the retrieval_results.json file
        data_dir: Base directory containing audio files

    Returns:
        List of tuples: [(target_path, reference_path, similarity_score), ...]
    """
    # Load retrieval results
    with open(retrieval_json_path, 'r', encoding='utf-8') as f:
        retrieval_results = json.load(f)

    data_path = Path(data_dir)
    target_dir = data_path / "target_music_list_60s"
    reference_dir = data_path / "reference_music_list_60s"

    audio_pairs = []

    for target_name, matches in retrieval_results.items():
        # Find target audio file (try .wav first, then .mp3)
        target_path = target_dir / f"{target_name}.wav"
        if not target_path.exists():
            target_path = target_dir / f"{target_name}.mp3"

        if not target_path.exists():
            print(f"Warning: Target file not found: {target_name} (.wav or .mp3)")
            continue

        # Get top match (or iterate through all matches if needed)
        for match in matches:
            reference_name = match["reference"]
            similarity_score = match["similarity_score"]

            # Find reference audio file (try .wav first, then .mp3)
            reference_path = reference_dir / f"{reference_name}.wav"
            if not reference_path.exists():
                reference_path = reference_dir / f"{reference_name}.mp3"

            if not reference_path.exists():
                print(f"Warning: Reference file not found: {reference_name} (.wav or .mp3)")
                continue

            audio_pairs.append((target_path, reference_path, similarity_score))

    print(f"Loaded {len(audio_pairs)} audio pairs for evaluation.")
    return audio_pairs

def evaluate_aesthetic_scores(audio_pairs):
    """Evaluate aesthetic scores for given audio pairs

    Args:
        audio_pairs: List of tuples: [(target_path, reference_path, similarity_score), ...]
    Returns:
        List of dicts with evaluation results
    """
    results = []
    predictor = initialize_predictor()

    for i, (target_path, reference_path, similarity_score) in enumerate(audio_pairs, 1):
        print(f"  [{i}/{len(audio_pairs)}] Evaluating {target_path.name}...")

        # Load audio files
        target_audio, target_sr = torchaudio.load(target_path)
        reference_audio, reference_sr = torchaudio.load(reference_path)

        # Get aesthetic scores using forward() method with proper format
        target_scores = predictor.forward([{"path": target_audio, "sample_rate": target_sr}])
        reference_scores = predictor.forward([{"path": reference_audio, "sample_rate": reference_sr}])

        # Extract first result (since we only pass one audio at a time)
        target_aesthetic = target_scores[0] if isinstance(target_scores, list) else target_scores
        reference_aesthetic = reference_scores[0] if isinstance(reference_scores, list) else reference_scores

        results.append({
            "target": target_path.name,
            "reference": reference_path.name,
            "similarity_score": similarity_score,
            "target_aesthetic": target_aesthetic,
            "reference_aesthetic": reference_aesthetic
        })

    return results

def main():
    # Define paths
    retrieval_json_path = "HW2/results/retrieval_results.json"
    output_path = "HW2/results/aesthetic_evaluation_results.json"

    print("="*60)
    print("AESTHETIC EVALUATION OF RETRIEVED AUDIO PAIRS")
    print("="*60)

    # Load evaluation data
    audio_pairs = load_evaluation_data(retrieval_json_path)

    # Evaluate aesthetic scores
    print("\nEvaluating aesthetic scores...")
    evaluation_results = evaluate_aesthetic_scores(audio_pairs)

    # Save results to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Evaluation results saved to: {output_path}")

    print("\n" + "="*60)
    print("✅ AESTHETIC EVALUATION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()