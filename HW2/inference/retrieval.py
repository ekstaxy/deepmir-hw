import os
import json
from pathlib import Path
import msclap

def load_clap_model():
    """Load CLAP model using msclap."""
    model = msclap.CLAP(version='2023', use_cuda=True)
    return model

def get_audio_files(audio_dir):
    """Get all audio files from directory"""
    audio_path = Path(audio_dir)
    audio_files = {}
    
    for ext in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
        for audio_file in audio_path.glob(f'*{ext}'):
            audio_files[audio_file.stem] = str(audio_file)
    
    print(f"Found {len(audio_files)} audio files in {audio_dir}")
    return audio_files

def calculate_similarity(model, target_path, reference_files, top_k=3):
    """Calculate CLAP similarity and return top-k matches"""
    similarities = {}
    
    target_embed = model.get_audio_embeddings([target_path])
    
    for ref_name, ref_path in reference_files.items():
        ref_embed = model.get_audio_embeddings([ref_path])
        similarity = model.compute_similarity(target_embed, ref_embed)
        similarities[ref_name] = float(similarity[0][0])
    
    top_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return top_matches

def create_retrieval_results(model, target_files, reference_files, top_k=3):
    """Create retrieval results for all target files"""
    results = {}
    
    for i, (target_name, target_path) in enumerate(target_files.items(), 1):
        print(f"  [{i}/{len(target_files)}] Processing {target_name}...")
        
        top_matches = calculate_similarity(model, target_path, reference_files, top_k)
        
        results[target_name] = [
            {
                "reference": ref_name,
                "similarity_score": score
            }
            for ref_name, score in top_matches
        ]
    
    return results

def main():
    target_dir = "HW2/data/target_music_list_60s"
    reference_dir = "HW2/data/reference_music_list_60s"
    output_path = "HW2/results/retrieval_results.json"
    top_k = 3
    
    print("="*60)
    print("MUSIC RETRIEVAL - CLAP SIMILARITY")
    print("="*60)
    
    print("\nLoading CLAP model...")
    model = load_clap_model()
    print("✓ CLAP model loaded")
    
    print("\nLoading audio files...")
    target_files = get_audio_files(target_dir)
    reference_files = get_audio_files(reference_dir)
    
    print(f"\nCalculating top-{top_k} similar references for each target...")
    results = create_retrieval_results(model, target_files, reference_files, top_k)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {output_path}")
    print("\n" + "="*60)
    print("✅ RETRIEVAL COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()