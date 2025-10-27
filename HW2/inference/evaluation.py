import os
import sys
import json
from pathlib import Path
import numpy as np
import torch
import torchaudio
from audiobox_aesthetics.infer import initialize_predictor
from tqdm import tqdm

def find_audio_folders(top_dir):
    """Find all folders containing audio files under top directory
    
    Args:
        top_dir: Top-level directory to search
        
    Returns:
        List of Path objects for folders containing audio files
    """
    top_path = Path(top_dir)
    audio_folders = []
    
    # Walk through directory tree
    for folder in top_path.rglob('*'):
        if folder.is_dir():
            # Check if folder contains any audio files
            audio_files = list(folder.glob('*.mp3')) + list(folder.glob('*.wav'))
            if audio_files:
                audio_folders.append(folder)
    
    return audio_folders

def get_audio_files(folder_path):
    """Get all audio files (mp3, wav) in a folder
    
    Args:
        folder_path: Path to folder
        
    Returns:
        List of audio file paths
    """
    folder = Path(folder_path)
    audio_files = []
    
    # Get all mp3 files
    audio_files.extend(folder.glob('*.mp3'))
    # Get all wav files
    audio_files.extend(folder.glob('*.wav'))
    
    return sorted(audio_files)

def evaluate_audio_file(predictor, audio_path):
    """Evaluate aesthetic score for a single audio file"""
    try:
        audio, sr = torchaudio.load(audio_path)
        scores = predictor.forward([{"path": audio, "sample_rate": sr}])
        
        # Extract score
        aesthetic_result = scores[0] if isinstance(scores, list) else scores
        
        # Handle dict or numeric result
        if isinstance(aesthetic_result, dict):
            # Store the full dict and calculate mean of numeric values
            raw_scores = aesthetic_result
            numeric_values = [float(v) for v in aesthetic_result.values() 
                            if isinstance(v, (int, float))]
            aesthetic_score = np.mean(numeric_values) if numeric_values else raw_scores
        else:
            raw_scores = None
            aesthetic_score = float(aesthetic_result)
        
        result = {
            "filename": audio_path.name,
            "aesthetic_score": aesthetic_score,
            "duration_seconds": float(audio.shape[1] / sr),
            "sample_rate": int(sr),
            "channels": int(audio.shape[0])
        }
        
        if raw_scores:
            result["raw_scores"] = raw_scores
        
        return result
        
    except Exception as e:
        print(f"    ✗ Error: {audio_path.name}: {str(e)}")
        return None

def evaluate_folder(predictor, folder_path, output_filename="aesthetic_evaluation.json"):
    """Evaluate all audio files in a folder and save results
    
    Args:
        predictor: Initialized aesthetic predictor
        folder_path: Path to folder containing audio files
        output_filename: Name of output JSON file
        
    Returns:
        dict with summary statistics
    """
    folder = Path(folder_path)
    audio_files = get_audio_files(folder)
    
    if not audio_files:
        print(f"  ⚠ No audio files found in {folder}")
        return None
    
    print(f"\n  Processing folder: {folder.name}")
    print(f"  Found {len(audio_files)} audio files")
    
    results = []
    successful = 0
    failed = 0
    
    # Evaluate each audio file
    for audio_file in tqdm(audio_files, desc=f"  Evaluating", unit="file"):
        result = evaluate_audio_file(predictor, audio_file)
        if result:
            results.append(result)
            successful += 1
        else:
            failed += 1
    
    # Calculate statistics
    if results:
        aesthetic_scores = [r["aesthetic_score"] for r in results]
        summary = {
            "folder": str(folder.relative_to(folder.parent.parent)),
            "total_files": len(audio_files),
            "successful_evaluations": successful,
            "failed_evaluations": failed,
            "statistics": {
                "mean_aesthetic_score": float(np.mean(aesthetic_scores)),
                "median_aesthetic_score": float(np.median(aesthetic_scores)),
                "std_aesthetic_score": float(np.std(aesthetic_scores)),
                "min_aesthetic_score": float(np.min(aesthetic_scores)),
                "max_aesthetic_score": float(np.max(aesthetic_scores))
            },
            "files": results
        }
        
        # Save results to folder
        output_path = folder / output_filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Results saved to: {output_path}")
        print(f"  ✓ Mean aesthetic score: {summary['statistics']['mean_aesthetic_score']:.4f}")
        
        return summary
    else:
        print(f"  ✗ No successful evaluations in {folder}")
        return None

def main():
    top_dir = "HW2/results/musicgen-small_generated_music"  # Change this to your folder path
    
    print("="*80)
    print("AESTHETIC EVALUATION OF AUDIO FILES")
    print("="*80)
    
    # Initialize predictor once
    print("\nInitializing aesthetic predictor...")
    predictor = initialize_predictor()
    print("✓ Predictor initialized")
    
    # Evaluate the single folder
    folder = Path(top_dir)
    summary = evaluate_folder(predictor, folder, "aesthetic_evaluation.json")
    
    if summary:
        print("\n" + "="*80)
        print("✅ EVALUATION COMPLETE!")
        print("="*80)
    else:
        print("\n✗ No successful evaluations completed")

if __name__ == "__main__":
    main()