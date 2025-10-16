"""
Visualization utilities for experiment tracking and analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import json

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")


def plot_training_curves(
    metrics_file: Union[str, Path],
    metrics: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Plot training curves from metrics file.

    Args:
        metrics_file: Path to metrics JSON file
        metrics: List of metrics to plot (if None, plots all)
        save_path: Path to save the plot
        show_plot: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    # Load metrics
    with open(metrics_file, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Filter out non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'step' in numeric_cols:
        numeric_cols.remove('step')

    if metrics is None:
        metrics = [col for col in numeric_cols if not col.startswith('timestamp')]

    # Create subplots
    n_metrics = len(metrics)
    if n_metrics == 0:
        raise ValueError("No metrics found to plot")

    fig, axes = plt.subplots(
        nrows=(n_metrics + 1) // 2,
        ncols=2 if n_metrics > 1 else 1,
        figsize=(12, 4 * ((n_metrics + 1) // 2))
    )

    if n_metrics == 1:
        axes = [axes]
    elif n_metrics > 1:
        axes = axes.flatten()

    # Plot each metric
    for i, metric in enumerate(metrics):
        if metric in df.columns:
            axes[i].plot(df.index, df[metric], linewidth=2, label=metric)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_xlabel('Step')
            axes[i].set_ylabel(metric)
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()

    # Hide extra subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show_plot:
        plt.show()

    return fig


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    title: str = "Confusion Matrix",
    normalize: bool = False,
    save_path: Optional[Path] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        title: Plot title
        normalize: Whether to normalize the matrix
        save_path: Path to save the plot
        show_plot: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    from sklearn.metrics import confusion_matrix

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title += ' (Normalized)'
    else:
        fmt = 'd'

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )

    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show_plot:
        plt.show()

    return fig


def plot_class_distribution(
    labels: List[int],
    class_names: List[str],
    title: str = "Class Distribution",
    save_path: Optional[Path] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Plot class distribution as bar chart.

    Args:
        labels: List of labels
        class_names: List of class names
        title: Plot title
        save_path: Path to save the plot
        show_plot: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    # Count occurrences
    unique, counts = np.unique(labels, return_counts=True)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(range(len(unique)), counts, color=sns.color_palette("husl", len(unique)))

    ax.set_title(title)
    ax.set_xlabel('Artist')
    ax.set_ylabel('Number of Samples')
    ax.set_xticks(range(len(unique)))
    ax.set_xticklabels([class_names[i] for i in unique], rotation=45, ha='right')

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                str(count), ha='center', va='bottom')

    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show_plot:
        plt.show()

    return fig


def plot_audio_features(
    features: np.ndarray,
    feature_names: List[str],
    title: str = "Audio Features",
    save_path: Optional[Path] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Plot audio features as heatmap.

    Args:
        features: Feature matrix (time x features)
        feature_names: Names of features
        title: Plot title
        save_path: Path to save the plot
        show_plot: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.heatmap(
        features.T,
        yticklabels=feature_names,
        cmap='viridis',
        ax=ax
    )

    ax.set_title(title)
    ax.set_xlabel('Time Frame')
    ax.set_ylabel('Feature')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show_plot:
        plt.show()

    return fig


def plot_mel_spectrogram(
    mel_spec: np.ndarray,
    sample_rate: int = 16000,
    hop_length: int = 512,
    title: str = "Mel Spectrogram",
    save_path: Optional[Path] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Plot mel spectrogram.

    Args:
        mel_spec: Mel spectrogram array
        sample_rate: Sample rate
        hop_length: Hop length used for STFT
        title: Plot title
        save_path: Path to save the plot
        show_plot: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    import librosa

    fig, ax = plt.subplots(figsize=(12, 6))

    img = librosa.display.specshow(
        mel_spec,
        x_axis='time',
        y_axis='mel',
        sr=sample_rate,
        hop_length=hop_length,
        ax=ax
    )

    ax.set_title(title)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show_plot:
        plt.show()

    return fig


def create_experiment_dashboard(
    experiment_dir: Path,
    output_file: Optional[Path] = None
) -> str:
    """
    Create an HTML dashboard for experiment results.

    Args:
        experiment_dir: Directory containing experiment results
        output_file: Output HTML file path

    Returns:
        Path to created HTML file
    """
    experiment_dir = Path(experiment_dir)

    if output_file is None:
        output_file = experiment_dir / "dashboard.html"

    # Load experiment metadata
    metadata_file = experiment_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}

    # Load metrics if available
    metrics_file = experiment_dir / "metrics.json"
    metrics_data = {}
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)

    # HTML template
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Experiment Dashboard: {experiment_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            .metrics {{ display: flex; flex-wrap: wrap; gap: 20px; }}
            .metric {{ background-color: #e8f4fd; padding: 10px; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Experiment Dashboard</h1>
            <h2>{experiment_name}</h2>
            <p><strong>Started:</strong> {start_time}</p>
            <p><strong>Duration:</strong> {duration}</p>
            <p><strong>Tags:</strong> {tags}</p>
        </div>

        <div class="section">
            <h3>Configuration</h3>
            <pre>{config}</pre>
        </div>

        <div class="section">
            <h3>Final Metrics</h3>
            <div class="metrics">
                {metrics_html}
            </div>
        </div>

        <div class="section">
            <h3>Files</h3>
            <ul>
                {files_html}
            </ul>
        </div>
    </body>
    </html>
    """

    # Format data
    experiment_name = metadata.get('experiment_name', 'Unknown')
    start_time = metadata.get('start_time', 'Unknown')
    duration = metadata.get('duration_formatted', 'Unknown')
    tags = ', '.join(metadata.get('tags', []))

    # Format configuration
    config_str = json.dumps(metadata.get('config', {}), indent=2)

    # Format metrics
    metrics_html = ""
    if metrics_data and len(metrics_data) > 0:
        final_metrics = metrics_data[-1]  # Get last metrics
        for key, value in final_metrics.items():
            if key not in ['timestamp', 'elapsed_time', 'step']:
                if isinstance(value, float):
                    metrics_html += f'<div class="metric"><strong>{key}:</strong> {value:.4f}</div>'
                else:
                    metrics_html += f'<div class="metric"><strong>{key}:</strong> {value}</div>'

    # List files
    files_html = ""
    for file_path in experiment_dir.rglob('*'):
        if file_path.is_file():
            rel_path = file_path.relative_to(experiment_dir)
            files_html += f'<li><a href="{rel_path}">{rel_path}</a></li>'

    # Generate HTML
    html_content = html_template.format(
        experiment_name=experiment_name,
        start_time=start_time,
        duration=duration,
        tags=tags,
        config=config_str,
        metrics_html=metrics_html,
        files_html=files_html
    )

    # Write file
    with open(output_file, 'w') as f:
        f.write(html_content)

    return str(output_file)