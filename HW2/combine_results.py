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

def combine_retrieval_with_melody(final_retrieval_path, melody_accuracy_path):
    """Combine final retrieval results with melody accuracy."""
    
    # Load JSON files
    with open(final_retrieval_path, 'r', encoding='utf-8') as f:
        retrieval_data = json.load(f)
    
    with open(melody_accuracy_path, 'r', encoding='utf-8') as f:
        melody_data = json.load(f)
    
    melody_results = melody_data['results']
    
    # Add melody accuracy to each target
    for entry in retrieval_data:
        target_name = entry['target']
        normalized_target = normalize_filename(target_name)
        
        # Find matching melody accuracy
        melody_acc = None
        for key, value in melody_results.items():
            if normalize_filename(key) == normalized_target:
                melody_acc = value
                break
        
        # Add melody accuracy field
        entry['melody_accuracy'] = melody_acc
    
    # Save updated results
    output_file = Path(final_retrieval_path).parent / "final_retrieval_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(retrieval_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Updated final retrieval results with melody accuracy")
    print(f"✓ Saved to: {output_file}")


def main():
    final_retrieval = "HW2/results/final_retrieval_results.json"
    melody_accuracy = "HW2/results/retrieval_melody_accuracy.json"
    
    combine_retrieval_with_melody(final_retrieval, melody_accuracy)
    
    print(f"\n✓ Done")


if __name__ == "__main__":
    main()