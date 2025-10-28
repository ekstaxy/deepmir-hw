import json
from pathlib import Path

def normalize_filename(filename):
    """Remove extensions and normalize."""
    name = filename
    while True:
        stem = Path(name).stem
        if stem == name:
            break
        name = stem
    return name

def combine_retrieval_results(folder_path):
    """Combine retrieval and aesthetic evaluation results."""
    folder = Path(folder_path)
    
    # Load JSON files
    retrieval_file = folder / "retrieval_results.json"
    aesthetic_file = folder / "aesthetic_evaluation_results.json"
    
    with open(retrieval_file, 'r', encoding='utf-8') as f:
        retrieval_data = json.load(f)
    
    with open(aesthetic_file, 'r', encoding='utf-8') as f:
        aesthetic_data = json.load(f)
    
    # Create combined results
    combined = []
    
    for target_name, retrieval_list in retrieval_data.items():
        normalized_target = normalize_filename(target_name)
        
        # Find matching aesthetic data
        aesthetic_entry = None
        for entry in aesthetic_data:
            if normalize_filename(entry['target']) == normalized_target:
                aesthetic_entry = entry
                break
        
        if aesthetic_entry:
            combined.append({
                "target": target_name,
                "target_aesthetic": aesthetic_entry['target_aesthetic'],
                "top_retrievals": [
                    {
                        "reference": item['reference'],
                        "similarity_score": item['similarity_score'],
                        "reference_aesthetic": next(
                            (e['reference_aesthetic'] for e in aesthetic_data 
                             if normalize_filename(e['reference']) == normalize_filename(item['reference'])),
                            None
                        )
                    }
                    for item in retrieval_list
                ]
            })
    
    # Save combined results
    output_file = folder / "retrieved_combine_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Combined results saved")


def main():
    folder = Path("HW2/results")
    
    combine_retrieval_results(folder)
    
    print(f"\n✓ Done")


if __name__ == "__main__":
    main()