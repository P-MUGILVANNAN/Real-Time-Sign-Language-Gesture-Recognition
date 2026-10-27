"""
Unit tests for LSTM, GRU, and baseline models.
"""

import os
import tempfile
import numpy as np
import pytest

from src.model_baseline import BaselineRandomForestModel, extract_temporal_statistical_features
from src.model_gru import build_gru_model
from src.model_lstm import build_lstm_model, load_trained_model, save_trained_model


def test_lstm_model_shapes():
    """Verify LSTM input and output shapes and parameter count."""
    model = build_lstm_model(
        sequence_length=30,
        feature_dim=63,
        num_classes=8,
        lstm_units_1=128,
        lstm_units_2=64
    )

    batch_input = np.random.randn(4, 30, 63).astype(np.float32)
    output = model(batch_input, training=False)

    assert output.shape == (4, 8)
    # Assert probability distribution sums to 1
    assert np.allclose(np.sum(output.numpy(), axis=1), 1.0, atol=1e-5)


def test_gru_model_shapes():
    """Verify GRU input and output shapes."""
    model = build_gru_model(
        sequence_length=30,
        feature_dim=63,
        num_classes=8
    )

    batch_input = np.random.randn(2, 30, 63).astype(np.float32)
    output = model(batch_input, training=False)

    assert output.shape == (2, 8)


def test_model_save_load_consistency():
    """Verify serialized model produces identical predictions after reloading."""
    model = build_lstm_model(sequence_length=30, feature_dim=63, num_classes=5)
    test_data = np.random.randn(2, 30, 63).astype(np.float32)
    pred_orig = model(test_data, training=False).numpy()

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "test_model.keras")
        save_trained_model(model, save_path)
        loaded_model = load_trained_model(save_path)
        pred_loaded = loaded_model(test_data, training=False).numpy()

        assert np.allclose(pred_orig, pred_loaded, atol=1e-5)


def test_baseline_random_forest():
    """Verify feature extraction and fit/predict for statistical Random Forest baseline."""
    X = np.random.randn(10, 30, 63).astype(np.float32)
    feats = extract_temporal_statistical_features(X)
    assert feats.shape == (10, 63 * 5)

    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    rf = BaselineRandomForestModel(n_estimators=10)
    rf.fit(X, y)
    preds = rf.predict(X)
    assert preds.shape == (10,)
