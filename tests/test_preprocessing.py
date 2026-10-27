"""
Unit tests for data preprocessing, normalization, and label encoding.
"""

import numpy as np
import pytest

from src.landmark_extractor import normalize_hand_landmarks
from src.preprocessing import (
    GestureLabelEncoder,
    PreprocessingPipeline,
    pad_or_truncate_sequence,
    resample_sequence
)


def test_normalize_hand_landmarks_shape_and_origin():
    """Verify that landmark normalization creates a 63-dim vector with wrist centered at origin."""
    raw_landmarks = np.random.uniform(0.1, 0.9, size=(21, 3)).astype(np.float32)
    norm = normalize_hand_landmarks(raw_landmarks)

    assert norm.shape == (63,)
    assert norm.dtype == np.float32
    # Wrist (landmark 0) should be at [0, 0, 0]
    assert np.allclose(norm[:3], [0.0, 0.0, 0.0], atol=1e-5)


def test_normalize_hand_landmarks_all_zeros():
    """Verify that an all-zero input (e.g. no hand detected) cleanly returns all zeros."""
    empty_lm = np.zeros((21, 3), dtype=np.float32)
    norm = normalize_hand_landmarks(empty_lm)
    assert norm.shape == (63,)
    assert np.all(norm == 0.0)


def test_normalize_hand_landmarks_nan_inf_handling():
    """Verify that NaN or Infinite values are safely sanitized."""
    corrupt_lm = np.random.uniform(0.1, 0.9, size=(21, 3)).astype(np.float32)
    corrupt_lm[5, 1] = np.nan
    corrupt_lm[10, 2] = np.inf
    norm = normalize_hand_landmarks(corrupt_lm)
    assert norm.shape == (63,)
    assert np.isfinite(norm).all()


def test_resample_sequence_expansion_and_compression():
    """Verify that variable length sequences (e.g. 10 or 60 frames) interpolate to exactly 30 frames."""
    # Test short sequence (15 frames -> 30)
    seq_short = np.random.randn(15, 63).astype(np.float32)
    res_short = resample_sequence(seq_short, target_length=30)
    assert res_short.shape == (30, 63)
    assert np.isfinite(res_short).all()

    # Test long sequence (50 frames -> 30)
    seq_long = np.random.randn(50, 63).astype(np.float32)
    res_long = resample_sequence(seq_long, target_length=30)
    assert res_long.shape == (30, 63)
    assert np.isfinite(res_long).all()

    # Test exact sequence (30 -> 30)
    seq_exact = np.random.randn(30, 63).astype(np.float32)
    res_exact = resample_sequence(seq_exact, target_length=30)
    assert np.allclose(res_exact, seq_exact)


def test_label_encoder_bidirectional():
    """Verify forward and backward categorical label encoding."""
    classes = ["hello", "thank_you", "yes", "no"]
    encoder = GestureLabelEncoder(classes)

    assert encoder.transform_label("hello") == 0
    assert encoder.transform_label("no") == 3
    assert encoder.inverse_transform_idx(1) == "thank_you"

    # Test one-hot
    labels = ["hello", "yes", "thank_you"]
    one_hot = encoder.to_one_hot(labels)
    assert one_hot.shape == (3, 4)
    assert np.array_equal(one_hot[0], [1, 0, 0, 0])
    assert np.array_equal(one_hot[1], [0, 0, 1, 0])


def test_label_encoder_threshold_unknown():
    """Verify low-confidence probabilities return unknown state."""
    classes = ["hello", "thank_you", "yes", "no"]
    encoder = GestureLabelEncoder(classes)

    # Low confidence case (max probability = 0.40)
    probs_low = np.array([0.40, 0.30, 0.20, 0.10], dtype=np.float32)
    label, conf = encoder.from_probabilities(probs_low, threshold=0.70)
    assert label == "No confident gesture detected"
    assert conf == pytest.approx(0.40, rel=1e-4)

    # High confidence case (max probability = 0.85)
    probs_high = np.array([0.05, 0.85, 0.05, 0.05], dtype=np.float32)
    label, conf = encoder.from_probabilities(probs_high, threshold=0.70)
    assert label == "thank_you"
    assert conf == pytest.approx(0.85, rel=1e-4)
