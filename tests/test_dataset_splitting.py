"""
Unit tests for signer-independent dataset splitting and data leakage prevention.
"""

import numpy as np
import pandas as pd
import pytest

from src.dataset_loader import DatasetLoader
from src.dataset_split import (
    split_dataset_signer_independent,
    split_dataset_stratified,
    verify_split_leakage
)


def test_signer_independent_splitting_zero_leakage():
    """Verify that signer-independent splitting produces strictly disjoint signer groups."""
    X, y, df = DatasetLoader.generate_synthetic_benchmark_dataset(
        num_samples_per_class=10,
        classes=["hello", "thank_you", "yes", "no"],
        num_signers=4,
        seed=123
    )

    (X_tr, y_tr, meta_tr), (X_val, y_val, meta_val), (X_te, y_te, meta_te) = split_dataset_signer_independent(
        X, y, df, test_size=0.25, val_size=0.25, random_state=42
    )

    # Check non-empty partitions
    assert len(X_tr) > 0
    assert len(X_val) > 0
    assert len(X_te) > 0
    assert len(X_tr) + len(X_val) + len(X_te) == len(X)

    # Verify zero group leakage
    report = verify_split_leakage(meta_tr, meta_val, meta_te, group_col="signer_id")
    assert report["is_leakage_free"] is True
    assert len(report["train_test_overlap"]) == 0
    assert len(report["train_val_overlap"]) == 0
    assert len(report["val_test_overlap"]) == 0


def test_stratified_split_preserves_classes():
    """Verify stratified split maintains class balance across partitions."""
    X, y, df = DatasetLoader.generate_synthetic_benchmark_dataset(
        num_samples_per_class=12,
        classes=["hello", "thank_you", "yes", "no"],
        seed=42
    )

    (X_tr, y_tr, _), (X_val, y_val, _), (X_te, y_te, _) = split_dataset_stratified(
        X, y, df, test_size=0.25, val_size=0.25, random_state=42
    )

    # Assert all classes appear in all splits
    assert len(np.unique(np.argmax(y_tr, axis=1))) == 4
    assert len(np.unique(np.argmax(y_val, axis=1))) == 4
    assert len(np.unique(np.argmax(y_te, axis=1))) == 4
