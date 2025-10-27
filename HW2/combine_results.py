import json
from pathlib import Path

def combine_json_files(folder_path, duration_sec=60):
    """Combine aesthetic and melody accuracy JSON files in a folder."""
    folder = Path(folder_path)
    
    aesthetic_file = folder / "aesthetic_evaluation.json"
    melody_file = folder / f"melody_accuracy_{duration_sec}s.json"
    
    # Load both JSON files
    with open(aesthetic_file, 'r', encoding='utf-8') as f:
        aesthetic_data = json.load(f)
    
    with open(melody_file, 'r', encoding='utf-8') as f:
        melody_data = json.load(f)
    
    # Create combined results
    combined = []
    
    for file_info in aesthetic_data['files']:
        filename = file_info['filename']
        # Remove extension to match melody_data keys
        name_key = Path(filename).stem
        
        # Find matching melody accuracy
        melody_acc = None
        for key in melody_data:
            if key in filename or filename.startswith(key):
                melody_acc = melody_data[key]
                break
        
        combined.append({
            "filename": filename,
            "aesthetic_score": file_info.get('raw_scores', {}),
            "melody_accuracy": melody_acc
        })
    
    # Save combined results
    output_file = folder / f"combined_evaluation_{duration_sec}s.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    
    print(f"✓ {folder.name} - {duration_sec}s")


def main():
    # Single folder mode
    folder = Path("HW2/results/musicgen-small_generated_music")  # Change this
    
    # Combine 30s
    combine_json_files(folder, duration_sec=30)
    
    # Combine 60s
    combine_json_files(folder, duration_sec=60)
    
    print(f"\n✓ Done")


if __name__ == "__main__":
    main()