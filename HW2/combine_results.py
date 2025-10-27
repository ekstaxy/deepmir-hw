import json
from pathlib import Path

def combine_json_files(folder_path):
    """Combine aesthetic and melody accuracy JSON files in a folder."""
    folder = Path(folder_path)
    
    aesthetic_file = folder / "aesthetic_evaluation.json"
    melody_file = folder / "melody_accuracy.json"
    
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
    output_file = folder / "combined_evaluation.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    
    print(f"✓ {folder.name}")


def main():
    top_dir = Path("HW2/results/musicControlLite_generated_music")
    
    folders = [f for f in top_dir.iterdir() if f.is_dir()]
    
    for folder in folders:
        combine_json_files(folder)
    
    print(f"\n✓ Combined {len(folders)} folders")


if __name__ == "__main__":
    main()