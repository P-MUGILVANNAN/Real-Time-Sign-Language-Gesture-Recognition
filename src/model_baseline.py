"""
Conventional Baseline Classifiers for sequence comparison (Random Forest & Statistical ML).
Computes temporal statistical landmark aggregations (mean, std, min, max, velocity delta)
to provide a rigorous non-recurrent benchmark.
"""

from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.utils import setup_logger

logger = setup_logger("ModelBaseline")


def extract_temporal_statistical_features(sequence_3d: np.ndarray) -> np.ndarray:
    """
    Transforms a batch of 3D landmark sequences (N, T, D) into a 2D statistical feature matrix (N, 5*D).

    Aggregations computed per feature channel:
    1. Temporal Mean: (D,)
    2. Temporal Standard Deviation: (D,)
    3. Temporal Min: (D,)
    4. Temporal Max: (D,)
    5. Mean Absolute Velocity Delta (|X_t - X_{t-1}|): (D,)

    Returns:
        np.ndarray: Feature matrix of shape (N, 5*D).
    """
    if sequence_3d.ndim == 2:
        sequence_3d = np.expand_dims(sequence_3d, axis=0)

    n, t, d = sequence_3d.shape

    mean_feat = np.mean(sequence_3d, axis=1)
    std_feat = np.std(sequence_3d, axis=1)
    min_feat = np.min(sequence_3d, axis=1)
    max_feat = np.max(sequence_3d, axis=1)

    # Velocity / Difference between adjacent frames
    diff = np.diff(sequence_3d, axis=1) if t > 1 else np.zeros((n, 1, d), dtype=np.float32)
    velocity_feat = np.mean(np.abs(diff), axis=1)

    aggregated = np.hstack([mean_feat, std_feat, min_feat, max_feat, velocity_feat])
    return aggregated.astype(np.float32)


class BaselineRandomForestModel:
    """
    Random Forest temporal gesture classifier based on engineered landmark statistics.
    """
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=15,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train_labels: Union[List[str], np.ndarray]):
        """
        Fits Random Forest on statistical landmark representations.
        """
        feats = extract_temporal_statistical_features(X_train)
        if y_train_labels.ndim > 1:
            y_train_labels = np.argmax(y_train_labels, axis=1)

        self.classifier.fit(feats, y_train_labels)
        self.is_fitted = True
        train_acc = self.classifier.score(feats, y_train_labels)
        logger.info(f"Fitted Baseline Random Forest on {len(X_train)} samples. Train Acc: {train_acc:.3f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        feats = extract_temporal_statistical_features(X)
        return self.classifier.predict(feats)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        feats = extract_temporal_statistical_features(X)
        return self.classifier.predict_proba(feats)

    def save(self, filepath: str):
        joblib.dump(self.classifier, filepath)
        logger.info(f"Saved Random Forest baseline model to {filepath}")

    def load(self, filepath: str):
        self.classifier = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Loaded Random Forest baseline model from {filepath}")
