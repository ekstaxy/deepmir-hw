#!/usr/bin/env python3
"""
Main training script for traditional ML models.
Train k-NN, SVM, and Random Forest classifiers for singer classification.
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import your ML models
from models.traditional.ml_models import (
    create_model, ModelEvaluator, HyperparameterTuner, generate_test_predictions
)
from models.traditional.feature_extractors import AudioFeatureExtractor, extract_features_from_dataset
from data.datasets import Artist20Dataset
from data.utils import load_audio_file, get_artist_mapping
from experiments.tracking import ExperimentTracker

logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

# Set seed at the start of the script
set_seed(42)  # Replace 42 with your desired seed value

def load_datasets(config):
    """Load training and validation datasets."""
    logger.info("Loading datasets...")

    # Load datasets with memory-efficient settings
    train_dataset = Artist20Dataset(
        config.dataset.train_json,
        root_dir=config.dataset.root_path,
        sample_rate=config.dataset.sample_rate,
        validate_files=False,
        return_full_audio=True, 
        max_duration=config.audio.max_duration
    )
    val_dataset = Artist20Dataset(
        config.dataset.val_json,
        root_dir=config.dataset.root_path,
        sample_rate=config.dataset.sample_rate,
        validate_files=False,
        return_full_audio=True,
        max_duration=config.audio.max_duration
    )

    # Removed test files loading
    artist_to_id = get_artist_mapping(config.dataset.train_json)

    logger.info(f"Loaded {len(train_dataset)} training samples")
    logger.info(f"Loaded {len(val_dataset)} validation samples")

    return train_dataset, val_dataset, artist_to_id

def extract_features(datasets, config):
    """Extract audio features for all datasets."""
    logger.info("Extracting features...")

    train_dataset, val_dataset, artist_to_id = datasets

    feature_extractor = AudioFeatureExtractor(config)

    # Extract features from training and validation datasets
    X_train, y_train = extract_features_from_dataset(train_dataset, config)
    X_val, y_val = extract_features_from_dataset(val_dataset, config)

    logger.info(f"Training features shape: {X_train.shape}")
    logger.info(f"Validation features shape: {X_val.shape}")
    logger.info(f"Number of unique classes: {len(set(y_train))}")

    return (X_train, y_train), (X_val, y_val)

def train_model(model_type, config, train_data, val_data, artist_to_id):
    """Train a single ML model."""
    logger.info(f"Training {model_type} model...")

    X_train, y_train = train_data
    X_val, y_val = val_data

    model = create_model(model_type, config)
    logger.info(f"Created {model_type} model with config: {getattr(config.models, model_type)}")

    # Train the model
    model.fit(X_train, y_train)
    logger.info(f"Training completed for {model_type}")

    # Evaluate the model
    evaluator = ModelEvaluator(config)
    metrics = evaluator.evaluate_model(model, X_val, y_val)

    # Retrieve class names from artist_to_id mapping
    class_names = list(artist_to_id.keys())

    # Plot confusion matrix
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, model_type)

    return model, metrics

def plot_confusion_matrix(confusion_matrix, class_names, save_dir):
    """Plot and save confusion matrix using matplotlib and seaborn."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()

    # Ensure save directory exists
    os.makedirs("results/visualizations", exist_ok=True)
    save_path = os.path.join("results/visualizations", f"confusion_matrix_{save_dir}.png")
    plt.savefig(save_path)
    plt.close()

def hyperparameter_tuning(model_type, config, train_data):
    """Perform hyperparameter tuning for a model and return updated config."""
    logger.info(f"Tuning hyperparameters for {model_type}...")

    X_train, y_train = train_data
    tuner = HyperparameterTuner(config)

    if model_type == 'knn':
        results = tuner.tune_knn(X_train, y_train)
    elif model_type == 'svm':
        results = tuner.tune_svm(X_train, y_train)
    elif model_type == 'random_forest':
        results = tuner.tune_random_forest(X_train, y_train)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info(f"Best parameters for {model_type}: {results['best_params']}")
    logger.info(f"Best score for {model_type}: {results['best_score']:.4f}")

    # Update the config with the best parameters
    for param, value in results['best_params'].items():
        setattr(getattr(config.models, model_type), param, value)

    return config

def save_model(model, model_type, config):
    """Save trained model to disk."""
    logger.info(f"Saving {model_type} model...")

    model_dir = Path(config.paths.checkpoints) / "traditional_ml"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{model_type}_model.pkl"
    model.save_model(model_path)

    logger.info(f"Model saved to: {model_path}")
    return model_path

def generate_test_predictions_for_best_model(models, test_data, config):
    """Generate predictions using the best performing model."""
    logger.info("Generating test predictions...")

    # Find best model based on final score
    best_model_type = None
    best_score = -1
    best_model = None

    for model_type, results in models.items():
        score = results['metrics']['final_score']
        if score > best_score:
            best_score = score
            best_model_type = model_type
            best_model = results['model']

    logger.info(f"Best model: {best_model_type} with score: {best_score:.4f}")

    X_test, test_filenames = test_data

    # Generate predictions
    predictions_dir = Path(config.paths.predictions)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    predictions_path = predictions_dir / "test_predictions.json"

    predictions = generate_test_predictions(best_model, X_test, test_filenames, predictions_path)

    return predictions_path

@hydra.main(version_base=None, config_path="../../configs", config_name="traditional_ml/baseline_config")
def main(cfg: DictConfig):
    """Main training function."""
    if 'traditional_ml' in cfg:
        config = cfg.traditional_ml
    else:
        config = cfg
    # Setup logging and tracking
    experiment_name = f"traditional_ml_{config.experiment.name}"
    logger.info("Starting traditional ML training...")

    # Initialize experiment tracker
    tracker = ExperimentTracker(
        config=config,
        experiment_name=f"traditional_ml_{config.experiment.name}",
        tags=["traditional_ml", "classification"],
        notes="Training k-NN, SVM, and Random Forest classifiers"
    )

    try:
        # Step 1: Load datasets
        datasets = load_datasets(config)

        # Step 2: Extract features
        train_data, val_data = extract_features(datasets, config)

        # Step 3: Train models
        models = {}
        model_types = ['knn', 'svm', 'random_forest']

        for model_type in model_types:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {model_type.upper()} Model")
            logger.info(f"{'='*50}")

            try:
                # Optional: Run hyperparameter tuning
                # config = hyperparameter_tuning(model_type, config, train_data)

                # Train model
                model, metrics = train_model(model_type, config, train_data, val_data, datasets[2])

                # Save model
                save_model(model, model_type, config)

                # Store results
                models[model_type] = {
                    'model': model,
                    'metrics': metrics
                }

                # Log metrics
                logger.info(f"Results for {model_type}:")
                logger.info(f"  Top-1 Accuracy: {metrics['top1_accuracy']:.4f}")
                logger.info(f"  Top-3 Accuracy: {metrics['top3_accuracy']:.4f}")
                logger.info(f"  Final Score: {metrics['final_score']:.4f}")

                # Log to tracker
                tracker.log_metrics({
                    f'{model_type}/top1_accuracy': metrics['top1_accuracy'],
                    f'{model_type}/top3_accuracy': metrics['top3_accuracy'],
                    f'{model_type}/final_score': metrics['final_score']
                })

            except Exception as e:
                logger.error(f"Failed to train {model_type}: {e}")
                continue

        # # Step 4: Generate final predictions
        # if models:
        #     predictions_path = generate_test_predictions_for_best_model(models, test_data, config)
        #     logger.info(f"Test predictions saved to: {predictions_path}")

        # Step 5: Print final summary
        logger.info(f"\n{'='*50}")
        logger.info("TRAINING COMPLETE")
        logger.info(f"{'='*50}")

        for model_type, results in models.items():
            metrics = results['metrics']
            logger.info(f"{model_type.upper()}:")
            logger.info(f"  Final Score: {metrics['final_score']:.4f}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    finally:
        # Finish experiment tracking
        tracker.finish()

if __name__ == "__main__":
    main()