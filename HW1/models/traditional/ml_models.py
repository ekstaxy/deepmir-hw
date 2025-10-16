"""
Traditional machine learning models for singer classification.
Implements k-NN, SVM, and Random Forest classifiers with hyperparameter tuning.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import joblib
import json

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import wandb

logger = logging.getLogger(__name__)

class TraditionalMLModel:
    """Base class for traditional ML models with preprocessing and label encoding."""

    def __init__(self, config):

        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def preprocess_features(self, X, fit_scaler=False):

        if fit_scaler:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)

    def encode_labels(self, y, fit_encoder=False):

        if fit_encoder:
            return self.label_encoder.fit_transform(y)
        else:
            return self.label_encoder.transform(y)

    def decode_labels(self, y_encoded):

        return self.label_encoder.inverse_transform(y_encoded)


    def fit(self, X, y):
        """
        Abstract method - fit the model.
        Should be implemented by subclasses.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
        """
        
        raise NotImplementedError

    def predict(self, X):
        """
        Abstract method - make predictions.
        Should be implemented by subclasses.

        Args:
            X: Test features (n_samples, n_features)

        Returns:
            Predicted labels as strings
        """
        raise NotImplementedError

    def predict_proba(self, X):
        """
        Abstract method - predict probabilities.
        Should be implemented by subclasses.

        Args:
            X: Test features (n_samples, n_features)

        Returns:
            Class probabilities (n_samples, n_classes)
        """
        raise NotImplementedError

    def save_model(self, save_path):
        """
        Save trained model to disk.

        Args:
            save_path: Path to save model (str or Path object)
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'config': self.config,
            'is_fitted': self.is_fitted
        }

        joblib.dump(model_data, save_path)
        logger.info(f"Model saved to {save_path}")

    def load_model(self, load_path):
        """
        Load trained model from disk.

        Args:
            load_path: Path to load model from (str or Path object)
        """
        load_path = Path(load_path)

        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        model_data = joblib.load(load_path)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.is_fitted = model_data['is_fitted']

        logger.info(f"Model loaded from {load_path}")

class KNNClassifier(TraditionalMLModel):
    """k-Nearest Neighbors classifier."""

    def __init__(self, config):
        """Initialize k-NN classifier with config parameters."""
        super().__init__(config)
        self.model = KNeighborsClassifier(
            n_neighbors=config.models.knn.n_neighbors[0],
            weights=config.models.knn.weights[0],
            metric=config.models.knn.metric[0],
            n_jobs=-1
        )

    def fit(self, X, y):
        """Fit k-NN classifier."""
        X_scaled = self.preprocess_features(X, fit_scaler=True)
        y_encoded = self.encode_labels(y, fit_encoder=True)
        self.model.fit(X_scaled, y_encoded)
        self.is_fitted = True
        logger.info(f"k-NN model fitted with {len(X)} samples")

        return self

    def predict(self, X):
        """Make predictions with k-NN."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.preprocess_features(X, fit_scaler=False)
        y_pred_encoded = self.model.predict(X_scaled)
        y_pred = self.decode_labels(y_pred_encoded)

        return y_pred


    def predict_proba(self, X):
        """Predict probabilities with k-NN."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.preprocess_features(X, fit_scaler=False)
        probabilities = self.model.predict_proba(X_scaled)

        return probabilities

class SVMClassifier(TraditionalMLModel):
    """Support Vector Machine classifier."""

    def __init__(self, config):
        """Initialize SVM classifier with config parameters."""
        super().__init__(config)
        self.model = SVC(
            C=config.models.svm.C[0],
            kernel=config.models.svm.kernel[0],
            gamma=config.models.svm.gamma[0],
            probability=True,
            random_state=config.project.seed
        )

    def fit(self, X, y):
        """Fit SVM classifier."""
        X_scaled = self.preprocess_features(X, fit_scaler=True)
        y_encoded = self.encode_labels(y, fit_encoder=True)
        self.model.fit(X_scaled, y_encoded)
        self.is_fitted = True
        logger.info(f"SVM model fitted with {len(X)} samples")
        return self

    def predict(self, X):
        """Make predictions with SVM."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.preprocess_features(X, fit_scaler=False)
        y_pred_encoded = self.model.predict(X_scaled)
        y_pred = self.decode_labels(y_pred_encoded)

        return y_pred

    def predict_proba(self, X):
        """Predict probabilities with SVM."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.preprocess_features(X, fit_scaler=False)
        probabilities = self.model.predict_proba(X_scaled)

        return probabilities

class RandomForestClassifier(TraditionalMLModel):
    """Random Forest classifier."""

    def __init__(self, config):
        """Initialize Random Forest classifier with config parameters."""
        super().__init__(config)
        self.model = SklearnRandomForestClassifier(
            n_estimators=config.models.random_forest.n_estimators[0],
            max_depth=config.models.random_forest.max_depth[0],
            min_samples_split=config.models.random_forest.min_samples_split[0],
            min_samples_leaf=config.models.random_forest.min_samples_leaf[0],
            random_state=config.project.seed,
            n_jobs=-1
        )

    def fit(self, X, y):
        """Fit Random Forest classifier."""
        X_scaled = self.preprocess_features(X, fit_scaler=True)
        y_encoded = self.encode_labels(y, fit_encoder=True)
        self.model.fit(X_scaled, y_encoded)
        self.is_fitted = True
        logger.info(f"Random Forest model fitted with {len(X)} samples")
        return self

    def predict(self, X):
        """Make predictions with Random Forest."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.preprocess_features(X, fit_scaler=False)
        y_pred_encoded = self.model.predict(X_scaled)
        y_pred = self.decode_labels(y_pred_encoded)

        return y_pred

    def predict_proba(self, X):
        """Predict probabilities with Random Forest."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.preprocess_features(X, fit_scaler=False)
        probabilities = self.model.predict_proba(X_scaled)

        return probabilities

class ModelEvaluator:
    """
    Evaluate traditional ML models with standard and ranking-based metrics.
    Calculates top-1, top-3 accuracy and final score for model comparison.
    """

    def __init__(self, config):
        """Initialize evaluator with configuration."""
        self.config = config

    def evaluate_model(self, model, X_val, y_val):
        """
        Evaluate trained model with comprehensive metrics.
        Returns top-1, top-3 accuracy and final score.
        """
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        conf_matrix = confusion_matrix(y_val, y_pred)
        class_report = classification_report(y_val, y_pred)
        top3_acc = self._calculate_top_k_accuracy(y_val, y_proba, model.label_encoder, k=3)
        final_score = accuracy + 0.5 * top3_acc
        return {
            'top1_accuracy': accuracy,
            'top3_accuracy': top3_acc,
            'final_score': final_score,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'accuracy': accuracy  # alias for compatibility
        }

    def _calculate_top_k_accuracy(self, y_true, y_proba, label_encoder, k=3):
        """
        Calculate top-k accuracy - checks if true label appears in top-k predictions.
        """
        top_k_indices = np.argsort(y_proba, axis=1)[:, -k:]
        y_true_encoded = label_encoder.transform(y_true)
        correct_count = 0
        for i in range(len(y_true_encoded)):
            if y_true_encoded[i] in top_k_indices[i]:
                correct_count += 1
        return correct_count / len(y_true_encoded)

    def cross_validate_model(self, model, X, y, cv_folds=5):
        """
        Perform k-fold cross-validation for robust performance assessment.
        Returns mean, std, and individual fold scores.
        """
        X_scaled = model.scaler.fit_transform(X)
        y_encoded = model.label_encoder.fit_transform(y)
        scores = cross_val_score(
            model.model, X_scaled, y_encoded, cv=cv_folds, scoring='accuracy'
        )
        mean_score = scores.mean()
        std_score = scores.std()

        return {
            'scores': scores,
            'mean': mean_score,
            'std': std_score
        } 

class HyperparameterTuner:
    """Hyperparameter tuning for traditional ML models using GridSearchCV."""

    def __init__(self, config):
        """Initialize tuner with config."""
        self.config = config

    def tune_knn(self, X, y):
        """
        Tune k-NN hyperparameters using GridSearchCV.
        Returns best parameters, score, and fitted model.
        """
        scaler = StandardScaler()
        label_encoder = LabelEncoder()
        X_scaled = scaler.fit_transform(X)
        y_encoded = label_encoder.fit_transform(y)

        # Dynamically adjust n_neighbors based on the smallest fold size
        smallest_fold_size = len(y) // 5  # Assuming 5-fold cross-validation
        max_neighbors = max(1, smallest_fold_size)  # Ensure at least 1 neighbor

        param_grid = {
            'n_neighbors': list(range(1, max_neighbors + 1)),
            'weights': self.config.models.knn.weights,
            'metric': self.config.models.knn.metric
        }

        knn = KNeighborsClassifier(n_jobs=-1)

        # Use StratifiedKFold for balanced class distribution
        stratified_cv = StratifiedKFold(n_splits=5)
        grid_search = GridSearchCV(
            knn, param_grid, cv=stratified_cv, scoring='accuracy', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_scaled, y_encoded)

        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': grid_search.best_estimator_,
            'cv_results': grid_search.cv_results_
        }

    def tune_svm(self, X, y):
        """
        Tune SVM hyperparameters using GridSearchCV.
        Returns best parameters, score, and fitted model.
        """
        # Preprocess features and encode labels
        scaler = StandardScaler()
        label_encoder = LabelEncoder()
        X_scaled = scaler.fit_transform(X)
        y_encoded = label_encoder.fit_transform(y)

        # Define parameter grid from config
        param_grid = {
            'C': self.config.models.svm.C,
            'kernel': self.config.models.svm.kernel,
            'gamma': self.config.models.svm.gamma
        }

        svm = SVC(probability=True, random_state=self.config.project.seed)
        grid_search = GridSearchCV(
            svm, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_scaled, y_encoded)

        # Validate grid_search results
        best_params = grid_search.best_params_ if isinstance(grid_search.best_params_, dict) else {}
        best_score = grid_search.best_score_ if isinstance(grid_search.best_score_, (float, int)) else None

        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_model': grid_search.best_estimator_,
            'cv_results': grid_search.cv_results_
        }

    def tune_random_forest(self, X, y):
        """
        Tune Random Forest hyperparameters using GridSearchCV.
        Returns best parameters, score, and fitted model.
        """
        # Preprocess features and encode labels
        scaler = StandardScaler()
        label_encoder = LabelEncoder()
        X_scaled = scaler.fit_transform(X)
        y_encoded = label_encoder.fit_transform(y)

        # Define parameter grid from config
        param_grid = {
            'n_estimators': self.config.models.random_forest.n_estimators,
            'max_depth': self.config.models.random_forest.max_depth,
            'min_samples_split': self.config.models.random_forest.min_samples_split,
            'min_samples_leaf': self.config.models.random_forest.min_samples_leaf
        }

        rf = SklearnRandomForestClassifier(
            random_state=self.config.project.seed, n_jobs=-1
        )

        # Use StratifiedKFold for balanced class distribution
        stratified_cv = StratifiedKFold(n_splits=3)
        grid_search = GridSearchCV(
            rf, param_grid, cv=stratified_cv, scoring='accuracy', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_scaled, y_encoded)

        # Validate grid_search results
        best_params = grid_search.best_params_ if isinstance(grid_search.best_params_, dict) else {}
        best_score = grid_search.best_score_ if isinstance(grid_search.best_score_, (float, int)) else None

        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_model': grid_search.best_estimator_,
            'cv_results': grid_search.cv_results_
        }

def create_model(model_type, config):
    """Factory function to create ML models."""
    if model_type == 'knn':
        return KNNClassifier(config)
    elif model_type == 'svm':
        return SVMClassifier(config)
    elif model_type == 'random_forest':
        return RandomForestClassifier(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Supported types: 'knn', 'svm', 'random_forest'")

def generate_test_predictions(model, X_test, test_filenames, save_path):
    """Generate test predictions in required JSON format."""
    y_proba = model.predict_proba(X_test)
    top3_indices = np.argsort(y_proba, axis=1)[:, -3:]
    top3_indices = top3_indices[:, ::-1]  # Descending order
    top3_artists = model.decode_labels(top3_indices.flatten()).reshape(top3_indices.shape)

    predictions = {}
    for i, filename in enumerate(test_filenames):
        file_number = str(int(Path(filename).stem))
        predictions[file_number] = top3_artists[i].tolist()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(predictions, f, indent=2)

    logger.info(f"Test predictions saved to {save_path}")

    return predictions

if __name__ == "__main__":
    """
    Example usage of traditional ML models.
    """
    pass