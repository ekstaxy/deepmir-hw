# HW1 Singer Classification - Requirements Document

## Project Overview
Build a complete singer classification system using the Artist20 dataset with both traditional machine learning and deep learning approaches. This project serves as practice for building a full ML research project with proper experiment tracking, reproducible results, and systematic evaluation.

## Dataset Specifications
- **Dataset**: Artist20 
- **Classes**: 20 artists
- **Data Split**: Album-level split (4:1:1 ratio)
  - Training: 949 tracks
  - Validation: 231 tracks  
  - Testing: 233 tracks
- **Audio Format**: MP3, 16kHz, Mono, Full song length
- **Size**: 1.28GB

## Core Tasks

### Task 1: Traditional Machine Learning Model
**Objective**: Train a classical ML classifier using hand-crafted audio features

**Requirements**:
- Extract audio features using librosa or torchaudio
- Train traditional ML models (k-NN, SVM, Random Forest, etc.)
- Apply proper preprocessing (standardization, normalization, pooling)
- Report validation results with confusion matrix, top-1 and top-3 accuracy
- Document feature extraction process and model implementation clearly

### Task 2: Deep Learning Model  
**Objective**: Train an end-to-end deep learning model from scratch

**Requirements**:
- Design and implement a neural network architecture
- Train from scratch (no pre-trained models for core requirement)
- Use proper train/validation split (never touch test data during development)
- Report validation results with confusion matrix, top-1 and top-3 accuracy
- Generate test set predictions for submission
- Document model architecture and implementation details
- Cite any referenced papers or code appropriately

## Technical Implementation Requirements

### Project Structure
```
singer_classification/
├── configs/                 # Configuration files
│   ├── base_config.yaml
│   ├── traditional_ml/      # ML model configs
│   └── deep_learning/       # DL model configs
├── data/                    # Data management
│   ├── raw/                 # Original dataset
│   ├── processed/           # Preprocessed features
│   └── dataloaders/         # Data loading utilities
├── models/                  # Model implementations
│   ├── traditional/         # ML models
│   ├── deep_learning/       # Neural networks
│   └── utils/               # Model utilities
├── experiments/             # Experiment scripts
│   ├── traditional_ml/      # ML training scripts
│   ├── deep_learning/       # DL training scripts
│   └── evaluation/          # Evaluation scripts
├── results/                 # Results storage
│   ├── checkpoints/         # Model checkpoints
│   ├── logs/               # Training logs
│   ├── predictions/        # Test predictions
│   └── visualizations/     # Plots and figures
├── notebooks/              # Exploratory analysis
└── requirements.txt        # Dependencies
```

### Development Workflow
1. **Data Exploration**: Analyze dataset characteristics and class distribution
2. **Feature Engineering**: Design and extract meaningful audio features
3. **Baseline Implementation**: Start with simple models for both tasks
4. **Experiment Tracking**: Use systematic approach to track all experiments
5. **Model Development**: Iteratively improve models based on validation results
6. **Final Evaluation**: Test best models on test set (only once)

### Code Requirements
- **Modularity**: Separate data loading, feature extraction, model training, and evaluation
- **Reproducibility**: Set random seeds, save exact configurations
- **Documentation**: Clear docstrings and comments
- **Error Handling**: Robust error handling for data loading and training
- **Logging**: Comprehensive logging of training progress and results

## Experiment Tracking and Management

### Required Tracking
- Model hyperparameters and architecture details
- Feature extraction parameters  
- Training/validation loss curves
- Accuracy metrics (top-1, top-3)
- Confusion matrices
- Training time and computational resources
- Audio feature statistics and distributions

### Recommended Tools
- **Configuration Management**: Hydra for managing experiment configs
- **Experiment Tracking**: WandB or MLflow for logging experiments
- **Version Control**: Git for code versioning
- **Environment Management**: Requirements.txt for reproducibility

## Evaluation Criteria

### Validation Metrics (Required)
- Confusion matrix visualization
- Top-1 accuracy
- Top-3 accuracy  
- Per-class precision/recall analysis

### Test Submission Format
```json
{
  "001": ["artist_name_1", "artist_name_2", "artist_name_3"],
  "002": ["artist_name_1", "artist_name_2", "artist_name_3"],
  ...
  "233": ["artist_name_1", "artist_name_2", "artist_name_3"]
}
```

## Deliverables

### 1. Report (50% of grade)
- **Format**: PDF in presentation style (16:9 aspect ratio)
- **Length**: ~10 pages (flexible based on content quality)
- **Content**: 
  - Problem description and approach
  - Feature engineering process (Task 1)
  - Model architecture and implementation (Task 2)
  - Experimental results with visualizations
  - Analysis and discussion of results
  - References to any used papers/code

### 2. Code Submission
- **Cloud Storage**: Upload to accessible cloud drive
- **Contents**: 
  - Complete source code
  - Trained model checkpoints
  - requirements.txt
  - README with inference instructions
- **Inference**: Code must run successfully to reproduce test predictions

### 3. Test Predictions (50% of grade)
- **Format**: JSON file with top-3 predictions per test sample
- **Naming**: studentID.json
- **Evaluation**: top1_accuracy + 0.5 * top3_accuracy

## Timeline
- **Start Date**: September 11, 2025
- **Deadline**: October 8, 2025 (23:59)
- **Late Penalty**: -20% per day up to 3 days

## Optional Enhancements

### Advanced Features (Bonus)
- Data augmentation using source separation (spleeter, demucs)
- t-SNE visualization of learned embeddings
- Personal audio testing with mel-spectrogram analysis
- Pre-trained model comparison (for reference, not core requirement)

### Research Extensions
- Compare with state-of-the-art singer identification methods
- Implement attention mechanisms or advanced architectures
- Analyze model interpretability and feature importance

## Success Metrics
- **Technical**: Both models train successfully and produce reasonable validation accuracy
- **Reproducibility**: Code runs without errors and produces consistent results  
- **Documentation**: Clear explanation of methodology and results
- **Analysis**: Thoughtful comparison between traditional ML and DL approaches
- **Best Practices**: Proper train/validation/test splits and evaluation procedures

## Getting Started Checklist
- [ ] Set up project structure and version control
- [ ] Download and explore Artist20 dataset
- [ ] Implement basic data loading and preprocessing
- [ ] Extract baseline audio features for traditional ML
- [ ] Implement simple baseline models for both tasks
- [ ] Set up experiment tracking system
- [ ] Create evaluation and visualization utilities
- [ ] Plan systematic hyperparameter exploration