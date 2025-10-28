import json
from pathlib import Path
import msclap

def load_clap_model():
    """Load CLAP model."""
    model = msclap.CLAP(version='2023', use_cuda=True)
    return model

def normalize_filename(filename):
    """Remove extensions."""
    name = filename
    while True:
        stem = Path(name).stem
        if stem == name:
            break
        name = stem
    return name

def find_audio_file(folder, name):
    """Find audio file by name."""
    folder = Path(folder)
    normalized_target = normalize_filename(name)
    
    for ext in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
        for audio_file in folder.glob(f'*{ext}'):
            if normalize_filename(audio_file.name) == normalized_target:
                return audio_file
    return None

def calculate_clap_for_retrievals(retrieval_json, target_folder, reference_folder):
    """Calculate CLAP similarity for retrieved results."""
    
    # Load retrieval results
    with open(retrieval_json, 'r', encoding='utf-8') as f:
        retrieval_data = json.load(f)
    
    # Load CLAP model
    print("Loading CLAP model...")
    model = load_clap_model()
    print("✓ CLAP model loaded\n")
    
    results = {}
    
    for i, (target_name, retrieval_list) in enumerate(retrieval_data.items(), 1):
        print(f"[{i}/{len(retrieval_data)}] Processing: {target_name}")
        
        # Find target audio file
        target_file = find_audio_file(target_folder, target_name)
        if target_file is None:
            print(f"  ✗ Target not found")
            results[target_name] = None
            continue
        
        # Get target embedding
        target_embed = model.get_audio_embeddings([str(target_file)])
        
        # Calculate CLAP similarity for each reference
        clap_results = []
        for item in retrieval_list:
            ref_name = item['reference']
            ref_file = find_audio_file(reference_folder, ref_name)
            
            if ref_file is None:
                print(f"  ✗ Reference not found: {ref_name}")
                clap_results.append({
                    "reference": ref_name,
                    "cosine_similarity": item['similarity_score'],
                    "clap_similarity": None
                })
                continue
            
            # Get reference embedding and calculate CLAP similarity
            ref_embed = model.get_audio_embeddings([str(ref_file)])
            clap_sim = model.compute_similarity(target_embed, ref_embed)
            
            clap_results.append({
                "reference": ref_name,
                "cosine_similarity": item['similarity_score'],
                "clap_similarity": float(clap_sim[0][0])
            })
            print(f"  ✓ {ref_name}: cosine={item['similarity_score']:.4f}, clap={float(clap_sim[0][0]):.4f}")
        
        results[target_name] = clap_results
    
    # Save results
    output_path = Path(retrieval_json).parent / "retrieval_clap_similarities.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved: {output_path}")

def main():
    retrieval_json = "HW2/results/retrieval_results.json"
    target_folder = "HW2/data/target_music_list_60s"
    reference_folder = "HW2/data/reference_music_list_60s"
    
    calculate_clap_for_retrievals(retrieval_json, target_folder, reference_folder)
    print("\n✓ Done")

if __name__ == "__main__":
    main()