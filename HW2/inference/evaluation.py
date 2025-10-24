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
        # Find target audio file
        target_path = target_dir / f"{target_name}.wav"

        if not target_path.exists():
            print(f"Warning: Target file not found: {target_path}")
            continue

        # Get top match (or iterate through all matches if needed)
        for match in matches:
            reference_name = match["reference"]
            similarity_score = match["similarity_score"]

            # Find reference audio file
            reference_path = reference_dir / f"{reference_name}.wav"

            if not reference_path.exists():
                print(f"Warning: Reference file not found: {reference_path}")
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

    for target_path, reference_path, similarity_score in audio_pairs:
        # Load audio files
        target_audio, _ = torchaudio.load(target_path)
        reference_audio, _ = torchaudio.load(reference_path)

        # Get aesthetic scores
        target_score = predictor(target_audio)
        reference_score = predictor(reference_audio)

        results.append({
            "target": target_path.name,
            "reference": reference_path.name,
            "similarity_score": similarity_score,
            "target_aesthetic": target_score,
            "reference_aesthetic": reference_score
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