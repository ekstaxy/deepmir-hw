import json
from pathlib import Path
import numpy as np

def calculate_aesthetic_scores(aesthetic_dict):
    """Calculate mean of aesthetic scores and individual components."""
    if aesthetic_dict is None:
        return None
    
    ce = aesthetic_dict.get('CE', 0)
    cu = aesthetic_dict.get('CU', 0)
    pc = aesthetic_dict.get('PC', 0)
    pq = aesthetic_dict.get('PQ', 0)
    
    mean = float(np.mean([ce, cu, pc, pq]))
    
    return {
        "CE": float(ce),
        "CU": float(cu),
        "PC": float(pc),
        "PQ": float(pq),
        "mean": mean
    }

def analyze_target(target_json_path):
    """Analyze target audio."""
    with open(target_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    ce_scores = []
    cu_scores = []
    pc_scores = []
    pq_scores = []
    clap_text_sims = []
    
    for entry in data:
        scores = calculate_aesthetic_scores(entry['target_aesthetic'])
        if scores is not None:
            ce_scores.append(scores['CE'])
            cu_scores.append(scores['CU'])
            pc_scores.append(scores['PC'])
            pq_scores.append(scores['PQ'])
        
        if entry.get('clap_text_similarity') is not None:
            clap_text_sims.append(entry['clap_text_similarity'])
    
    return {
        "type": "target",
        "mean_aesthetic": {
            "CE": float(np.mean(ce_scores)) if ce_scores else None,
            "CU": float(np.mean(cu_scores)) if cu_scores else None,
            "PC": float(np.mean(pc_scores)) if pc_scores else None,
            "PQ": float(np.mean(pq_scores)) if pq_scores else None,
            "overall_mean": float(np.mean([np.mean(ce_scores), np.mean(cu_scores), np.mean(pc_scores), np.mean(pq_scores)])) if ce_scores else None
        },
        "mean_clap_text_similarity": float(np.mean(clap_text_sims)) if clap_text_sims else None
    }

def analyze_retrieved(retrieval_json_path):
    """Analyze retrieved references."""
    with open(retrieval_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    target_ce, target_cu, target_pc, target_pq = [], [], [], []
    ref_ce, ref_cu, ref_pc, ref_pq = [], [], [], []
    clap_sims = []
    
    for entry in data:
        # Target aesthetic
        target_scores = calculate_aesthetic_scores(entry['target_aesthetic'])
        if target_scores is not None:
            target_ce.append(target_scores['CE'])
            target_cu.append(target_scores['CU'])
            target_pc.append(target_scores['PC'])
            target_pq.append(target_scores['PQ'])
        
        # Reference aesthetics and CLAP
        for retrieval in entry['top_retrievals']:
            if retrieval['reference_aesthetic'] is not None:
                ref_scores = calculate_aesthetic_scores(retrieval['reference_aesthetic'])
                ref_ce.append(ref_scores['CE'])
                ref_cu.append(ref_scores['CU'])
                ref_pc.append(ref_scores['PC'])
                ref_pq.append(ref_scores['PQ'])
            
            if retrieval.get('clap_similarity') is not None:
                clap_sims.append(retrieval['clap_similarity'])
    
    return {
        "type": "retrieved",
        "mean_target_aesthetic": {
            "CE": float(np.mean(target_ce)) if target_ce else None,
            "CU": float(np.mean(target_cu)) if target_cu else None,
            "PC": float(np.mean(target_pc)) if target_pc else None,
            "PQ": float(np.mean(target_pq)) if target_pq else None,
            "overall_mean": float(np.mean([np.mean(target_ce), np.mean(target_cu), np.mean(target_pc), np.mean(target_pq)])) if target_ce else None
        },
        "mean_reference_aesthetic": {
            "CE": float(np.mean(ref_ce)) if ref_ce else None,
            "CU": float(np.mean(ref_cu)) if ref_cu else None,
            "PC": float(np.mean(ref_pc)) if ref_pc else None,
            "PQ": float(np.mean(ref_pq)) if ref_pq else None,
            "overall_mean": float(np.mean([np.mean(ref_ce), np.mean(ref_cu), np.mean(ref_pc), np.mean(ref_pq)])) if ref_ce else None
        },
        "mean_clap_audio_similarity": float(np.mean(clap_sims)) if clap_sims else None
    }

def analyze_generated(combined_json_path):
    """Analyze generated audio."""
    with open(combined_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    ce_scores, cu_scores, pc_scores, pq_scores = [], [], [], []
    melody_accs = []
    clap_audio_sims = []
    clap_text_sims = []
    
    for entry in data:
        # Aesthetic
        scores = calculate_aesthetic_scores(entry['aesthetic_score'])
        if scores is not None:
            ce_scores.append(scores['CE'])
            cu_scores.append(scores['CU'])
            pc_scores.append(scores['PC'])
            pq_scores.append(scores['PQ'])
        
        # Melody accuracy
        if entry.get('melody_accuracy') is not None:
            melody_accs.append(entry['melody_accuracy'])
        
        # CLAP similarities
        if entry.get('clap_audio_similarity') is not None:
            clap_audio_sims.append(entry['clap_audio_similarity'])
        
        if entry.get('clap_text_similarity') is not None:
            clap_text_sims.append(entry['clap_text_similarity'])
    
    folder_name = Path(combined_json_path).parent.name
    
    return {
        "type": "generated",
        "folder": folder_name,
        "mean_aesthetic": {
            "CE": float(np.mean(ce_scores)) if ce_scores else None,
            "CU": float(np.mean(cu_scores)) if cu_scores else None,
            "PC": float(np.mean(pc_scores)) if pc_scores else None,
            "PQ": float(np.mean(pq_scores)) if pq_scores else None,
            "overall_mean": float(np.mean([np.mean(ce_scores), np.mean(cu_scores), np.mean(pc_scores), np.mean(pq_scores)])) if ce_scores else None
        },
        "mean_melody_accuracy": float(np.mean(melody_accs)) if melody_accs else None,
        "mean_clap_audio_similarity": float(np.mean(clap_audio_sims)) if clap_audio_sims else None,
        "mean_clap_text_similarity": float(np.mean(clap_text_sims)) if clap_text_sims else None
    }

def main():
    target_json = "HW2/results/target_combined_results.json"
    retrieval_json = "HW2/results/final_retrieval_results.json"
    generated_top_dir = "HW2/results/museControlLite_generated_music"
    
    results = []
    
    print("Analyzing target audio...")
    target_stats = analyze_target(target_json)
    results.append(target_stats)
    print(f"  Overall mean aesthetic: {target_stats['mean_aesthetic']['overall_mean']:.4f}")
    
    print("\nAnalyzing retrieved audio...")
    retrieved_stats = analyze_retrieved(retrieval_json)
    results.append(retrieved_stats)
    print(f"  Target overall mean: {retrieved_stats['mean_target_aesthetic']['overall_mean']:.4f}")
    print(f"  Reference overall mean: {retrieved_stats['mean_reference_aesthetic']['overall_mean']:.4f}")
    
    print("\nAnalyzing generated audio folders...")
    generated_top_path = Path(generated_top_dir)
    
    for folder in sorted(generated_top_path.iterdir()):
        if folder.is_dir():
            combined_file = folder / "combined_evaluation.json"
            if combined_file.exists():
                print(f"\n  {folder.name}:")
                generated_stats = analyze_generated(combined_file)
                results.append(generated_stats)
                print(f"    Overall aesthetic: {generated_stats['mean_aesthetic']['overall_mean']:.4f}")
                print(f"    Melody Accuracy: {generated_stats['mean_melody_accuracy']:.4f}")
    
    output_path = Path(generated_top_dir).parent / "all_metrics_summary.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Summary saved to: {output_path}")

if __name__ == "__main__":
    main()