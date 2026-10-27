"""
Unit tests for landmark extraction and normalization invariant checks.
"""

import numpy as np
import pytest

from src.landmark_extractor import LandmarkExtractor, normalize_hand_landmarks


def test_landmark_normalization_scale_invariance():
    """Verify that scaling the hand (e.g. closer or further from camera) yields equivalent normalized features."""
    # Create arbitrary hand landmarks
    base_hand = np.random.uniform(0.2, 0.8, size=(21, 3)).astype(np.float32)

    # Hand 2 is scaled 2.5x larger and shifted
    scaled_shifted_hand = (base_hand * 2.5) + np.array([0.5, -0.3, 0.1])

    norm_base = normalize_hand_landmarks(base_hand)
    norm_scaled = normalize_hand_landmarks(scaled_shifted_hand)

    # Invariance check: normalized coordinates should match closely
    assert np.allclose(norm_base, norm_scaled, atol=1e-4)


def test_extractor_offline_empty_frame():
    """Verify landmark extractor handles empty/None frame gracefully without crashing."""
    extractor = LandmarkExtractor(mode="single_hand")
    features, detected, info = extractor.extract_from_frame(None)

    assert features.shape == (63,)
    assert detected is False
    assert np.all(features == 0)

    # Synthetic black frame
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    features, detected, info = extractor.extract_from_frame(black_frame)
    assert features.shape == (63,)
    assert detected is False
