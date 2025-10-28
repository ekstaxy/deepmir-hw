import os
import numpy as np
import json
from pathlib import Path
import msclap
from sklearn.metrics.pairwise import cosine_similarity

def load_clap_model():
    """Load CLAP model using msclap."""
    model = msclap.CLAP(version='2023', use_cuda=True)
    return model

def load_latents(latent_dir, subdir):
    """Load all latent files from a subdirectory"""
    latent_path = Path(latent_dir) / subdir
    latents = {}

    for npy_file in latent_path.glob("*.npy"):
        latent = np.load(npy_file)
        # Flatten latent to 1D vector for cosine similarity
        latent_flat = latent.flatten()
        latents[npy_file.stem] = latent_flat

    print(f"Loaded {len(latents)} latents from {subdir}/")
    return latents


def calculate_similarity(target_latent, reference_latents, top_k=3):
    """Calculate cosine similarity and return top-k matches"""
    similarities = {}

    # Reshape target for sklearn cosine_similarity (needs 2D array)
    target_2d = target_latent.reshape(1, -1)

    for ref_name, ref_latent in reference_latents.items():
        ref_2d = ref_latent.reshape(1, -1)
        # Calculate cosine similarity
        similarity = cosine_similarity(target_2d, ref_2d)[0][0]
        similarities[ref_name] = float(similarity)

    # Sort by similarity score (highest first) and get top-k
    top_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]

    return top_matches


def create_retrieval_results(target_latents, reference_latents, top_k=3):
    """Create retrieval results for all target latents"""
    results = {}

    for i, (target_name, target_latent) in enumerate(target_latents.items(), 1):
        print(f"  [{i}/{len(target_latents)}] Processing {target_name}...")

        # Get top-k similar references
        top_matches = calculate_similarity(target_latent, reference_latents, top_k)

        # Format results
        results[target_name] = [
            {
                "reference": ref_name,
                "similarity_score": score
            }
            for ref_name, score in top_matches
        ]

    return results


def save_json(results, output_path):
    """Save results to JSON file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to: {output_path}")


def main():
    # Define paths
    latent_dir = "HW2/data/latents"
    output_path = "HW2/results/retrieval_results.json"
    top_k = 3

    print("="*60)
    print("MUSIC RETRIEVAL - COSINE SIMILARITY")
    print("="*60)

    # Load latents
    print("\nLoading latents...")
    target_latents = load_latents(latent_dir, "target")
    reference_latents = load_latents(latent_dir, "reference")

    # Calculate similarities and create results
    print(f"\nCalculating top-{top_k} similar references for each target...")
    results = create_retrieval_results(target_latents, reference_latents, top_k)

    # Save to JSON
    save_json(results, output_path)

    print("\n" + "="*60)
    print("✅ RETRIEVAL COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
