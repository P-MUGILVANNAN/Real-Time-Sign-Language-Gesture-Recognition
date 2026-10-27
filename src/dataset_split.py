"""
Leakage-safe dataset splitting module.
Implements signer-independent and session-independent group splitting,
and stratified splitting with rigorous verification against data leakage.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

from src.utils import setup_logger

logger = setup_logger("DatasetSplit")


def verify_split_leakage(
    meta_train: pd.DataFrame,
    meta_val: pd.DataFrame,
    meta_test: pd.DataFrame,
    group_col: str = "signer_id"
) -> Dict[str, any]:
    """
    Formally inspects partitions for group overlap / data leakage.

    Returns:
        verification_report (dict): Contains leakage detection flags and group intersection sets.
    """
    train_groups = set(meta_train[group_col].dropna().unique()) if group_col in meta_train else set()
    val_groups = set(meta_val[group_col].dropna().unique()) if group_col in meta_val else set()
    test_groups = set(meta_test[group_col].dropna().unique()) if group_col in meta_test else set()

    train_val_overlap = train_groups.intersection(val_groups)
    train_test_overlap = train_groups.intersection(test_groups)
    val_test_overlap = val_groups.intersection(test_groups)

    has_leakage = bool(train_val_overlap or train_test_overlap or val_test_overlap)

    report = {
        "group_column": group_col,
        "is_leakage_free": not has_leakage,
        "train_groups": list(train_groups),
        "val_groups": list(val_groups),
        "test_groups": list(test_groups),
        "train_val_overlap": list(train_val_overlap),
        "train_test_overlap": list(train_test_overlap),
        "val_test_overlap": list(val_test_overlap),
    }

    if has_leakage:
        logger.warning(f"Data leakage detected! Overlapping {group_col}: {train_test_overlap}")
    else:
        logger.info(f"Zero {group_col} leakage confirmed across train, val, and test splits.")

    return report


def split_dataset_signer_independent(
    X: np.ndarray,
    y: np.ndarray,
    metadata_df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, pd.DataFrame],
    Tuple[np.ndarray, np.ndarray, pd.DataFrame],
    Tuple[np.ndarray, np.ndarray, pd.DataFrame]
]:
    """
    Performs rigorous signer-independent group splitting.
    Guarantees that test signers never appear in the training or validation sets.
    """
    if "signer_id" not in metadata_df.columns:
        logger.warning("'signer_id' column not found. Falling back to stratified split.")
        return split_dataset_stratified(X, y, metadata_df, test_size, val_size, random_state)

    unique_signers = metadata_df["signer_id"].nunique()
    if unique_signers < 3:
        logger.warning(f"Only {unique_signers} unique signer(s) present. Cannot form 3 disjoint signer groups.")
        return split_dataset_stratified(X, y, metadata_df, test_size, val_size, random_state)

    groups = metadata_df["signer_id"].values

    # Step 1: Split into Train+Val vs Test
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(X, y, groups))

    X_train_val = X[train_val_idx]
    y_train_val = y[train_val_idx]
    meta_train_val = metadata_df.iloc[train_val_idx].reset_index(drop=True)

    X_test = X[test_idx]
    y_test = y[test_idx]
    meta_test = metadata_df.iloc[test_idx].reset_index(drop=True)

    # Step 2: Split Train+Val into Train vs Val
    val_relative_size = val_size / (1.0 - test_size)
    groups_train_val = meta_train_val["signer_id"].values

    if meta_train_val["signer_id"].nunique() >= 2:
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_relative_size, random_state=random_state)
        train_idx, val_idx = next(gss_val.split(X_train_val, y_train_val, groups_train_val))
    else:
        # If remaining signers cannot be grouped, use stratified for val
        labels = np.argmax(y_train_val, axis=1)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_relative_size, random_state=random_state)
        train_idx, val_idx = next(sss.split(X_train_val, labels))

    X_train = X_train_val[train_idx]
    y_train = y_train_val[train_idx]
    meta_train = meta_train_val.iloc[train_idx].reset_index(drop=True)

    X_val = X_train_val[val_idx]
    y_val = y_train_val[val_idx]
    meta_val = meta_train_val.iloc[val_idx].reset_index(drop=True)

    verify_split_leakage(meta_train, meta_val, meta_test, group_col="signer_id")

    logger.info(
        f"Signer-independent split: Train={len(X_train)} samples, "
        f"Val={len(X_val)} samples, Test={len(X_test)} samples."
    )
    return (X_train, y_train, meta_train), (X_val, y_val, meta_val), (X_test, y_test, meta_test)


def split_dataset_stratified(
    X: np.ndarray,
    y: np.ndarray,
    metadata_df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, pd.DataFrame],
    Tuple[np.ndarray, np.ndarray, pd.DataFrame],
    Tuple[np.ndarray, np.ndarray, pd.DataFrame]
]:
    """
    Performs standard stratified train/val/test splitting preserving class proportions.
    """
    labels = np.argmax(y, axis=1)

    # Step 1: Train+Val vs Test
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(sss_test.split(X, labels))

    X_train_val = X[train_val_idx]
    y_train_val = y[train_val_idx]
    meta_train_val = metadata_df.iloc[train_val_idx].reset_index(drop=True)
    labels_train_val = labels[train_val_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]
    meta_test = metadata_df.iloc[test_idx].reset_index(drop=True)

    # Step 2: Train vs Val
    val_relative_size = val_size / (1.0 - test_size)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_relative_size, random_state=random_state)
    train_idx, val_idx = next(sss_val.split(X_train_val, labels_train_val))

    X_train = X_train_val[train_idx]
    y_train = y_train_val[train_idx]
    meta_train = meta_train_val.iloc[train_idx].reset_index(drop=True)

    X_val = X_train_val[val_idx]
    y_val = y_train_val[val_idx]
    meta_val = meta_train_val.iloc[val_idx].reset_index(drop=True)

    logger.info(
        f"Stratified split: Train={len(X_train)} samples, "
        f"Val={len(X_val)} samples, Test={len(X_test)} samples."
    )
    return (X_train, y_train, meta_train), (X_val, y_val, meta_val), (X_test, y_test, meta_test)
