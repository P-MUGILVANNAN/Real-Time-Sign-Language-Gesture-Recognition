"""
Data preprocessing pipeline for sign language landmark sequences.
Handles temporal resampling, length standardization, NaN sanitization,
label encoding, and shape validation.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.interpolate import interp1d

from src.utils import setup_logger

logger = setup_logger("Preprocessing")


def resample_sequence(sequence: np.ndarray, target_length: int = 30) -> np.ndarray:
    """
    Resamples a temporal sequence of shape (T, D) to (target_length, D) using linear interpolation.

    Ensures:
    1. Dynamic gesture speed variation is normalized to a fixed temporal window.
    2. Temporal ordering is strictly preserved.
    3. Handles sequences that are shorter or longer than target_length.

    Parameters:
        sequence (np.ndarray): Input sequence of shape (T, D).
        target_length (int): Desired output number of frames (default 30).

    Returns:
        np.ndarray: Resampled sequence of shape (target_length, D).
    """
    if not isinstance(sequence, np.ndarray):
        sequence = np.array(sequence, dtype=np.float32)

    if sequence.ndim != 2:
        raise ValueError(f"Expected 2D array of shape (T, D), got ndim={sequence.ndim} with shape {sequence.shape}")

    t_in, d = sequence.shape

    if t_in == 0:
        return np.zeros((target_length, d), dtype=np.float32)

    if t_in == target_length:
        return sequence.astype(np.float32)

    if t_in == 1:
        # Replicate the single static frame across all time steps
        return np.repeat(sequence, target_length, axis=0).astype(np.float32)

    # Linear interpolation over normalized time [0, 1]
    original_time = np.linspace(0.0, 1.0, num=t_in, endpoint=True)
    target_time = np.linspace(0.0, 1.0, num=target_length, endpoint=True)

    interpolator = interp1d(original_time, sequence, axis=0, kind='linear', fill_value="extrapolate")
    resampled = interpolator(target_time)

    # Sanitize any unexpected non-finite values
    resampled = np.nan_to_num(resampled, nan=0.0, posinf=0.0, neginf=0.0)

    return resampled.astype(np.float32)


def pad_or_truncate_sequence(sequence: np.ndarray, target_length: int = 30) -> np.ndarray:
    """
    Alternative length standardization using zero-padding or truncation.
    """
    t_in, d = sequence.shape
    if t_in == target_length:
        return sequence.astype(np.float32)
    elif t_in > target_length:
        return sequence[:target_length].astype(np.float32)
    else:
        pad_width = target_length - t_in
        padding = np.zeros((pad_width, d), dtype=np.float32)
        return np.vstack([sequence, padding]).astype(np.float32)


class GestureLabelEncoder:
    """
    Encodes categorical gesture string labels into integers and one-hot vectors,
    with exact bidirectional mapping and unknown class handling.
    """
    def __init__(self, classes: List[str]):
        self.classes: List[str] = list(classes)
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class: Dict[int, str] = {i: c for i, c in enumerate(self.classes)}
        self.num_classes: int = len(self.classes)

    def fit_classes(self, classes: List[str]):
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = {i: c for i, c in enumerate(self.classes)}
        self.num_classes = len(self.classes)

    def transform_label(self, label: str) -> int:
        if label not in self.class_to_idx:
            raise KeyError(f"Gesture class '{label}' is not in encoder classes: {self.classes}")
        return self.class_to_idx[label]

    def inverse_transform_idx(self, idx: int) -> str:
        if idx not in self.idx_to_class:
            return "Unknown"
        return self.idx_to_class[idx]

    def to_one_hot(self, labels: Union[List[str], np.ndarray]) -> np.ndarray:
        indices = [self.transform_label(lbl) for lbl in labels]
        one_hot = np.zeros((len(indices), self.num_classes), dtype=np.float32)
        for i, idx in enumerate(indices):
            one_hot[i, idx] = 1.0
        return one_hot

    def from_probabilities(self, probs: np.ndarray, threshold: float = 0.70) -> Tuple[str, float]:
        """
        Interprets model output softmax vector:
        Returns predicted class and confidence.
        If max probability is below threshold, returns ("No confident gesture detected", confidence).
        """
        probs = np.asarray(probs, dtype=np.float32).flatten()
        max_idx = int(np.argmax(probs))
        confidence = float(probs[max_idx])

        if confidence < threshold:
            return "No confident gesture detected", confidence

        return self.inverse_transform_idx(max_idx), confidence


class PreprocessingPipeline:
    """
    End-to-end preprocessing pipeline for sign gesture sequences.
    """
    def __init__(
        self,
        sequence_length: int = 30,
        feature_dim: int = 63,
        classes: Optional[List[str]] = None
    ):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.encoder = GestureLabelEncoder(classes or [])

    def process_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Resamples and validates a single sequence to (sequence_length, feature_dim).
        """
        sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)

        if sequence.shape[1] != self.feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.feature_dim}, got {sequence.shape[1]}"
            )

        resampled = resample_sequence(sequence, target_length=self.sequence_length)
        return resampled

    def process_batch(
        self,
        sequences: List[np.ndarray],
        labels: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Processes a collection of sequences into an array of shape (N, sequence_length, feature_dim).
        Optionally encodes labels to one-hot format (N, num_classes).
        """
        processed_seqs = []
        valid_labels = []

        for i, seq in enumerate(sequences):
            try:
                proc = self.process_sequence(seq)
                processed_seqs.append(proc)
                if labels is not None:
                    valid_labels.append(labels[i])
            except Exception as e:
                logger.warning(f"Skipping sequence index {i} due to preprocessing error: {e}")

        X = np.array(processed_seqs, dtype=np.float32)

        y = None
        if labels is not None:
            y = self.encoder.to_one_hot(valid_labels)

        return X, y
