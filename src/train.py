"""
Model training pipeline for SignVision AI.
Includes callbacks (EarlyStopping, Checkpoint, ReduceLROnPlateau), class weighting,
model serialization, metadata export, and evaluation visualization.
"""

import datetime
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src.dataset_loader import DatasetLoader
from src.dataset_split import split_dataset_signer_independent, split_dataset_stratified
from src.metrics import (
    compute_classification_metrics,
    plot_and_save_confusion_matrix,
    plot_and_save_training_history
)
from src.model_lstm import build_lstm_model, save_trained_model
from src.utils import ensure_dir, load_classes, load_config, setup_logger

logger = setup_logger("Trainer")


def run_training_pipeline(
    config_path: str = "config/config.yaml",
    classes_path: str = "config/classes.json",
    use_synthetic_if_empty: bool = True
) -> Dict:
    """
    Executes the end-to-end model training workflow.

    Returns:
        results (dict): Training history, test evaluation metrics, and artifact paths.
    """
    config = load_config(config_path)
    classes = load_classes(classes_path)

    seq_len = config["features"]["sequence_length"]
    feat_dim = config["features"]["single_hand_features"]
    num_classes = len(classes)

    logger.info(f"Initiating training for {num_classes} classes: {classes}")

    # Step 1: Load Dataset
    loader = DatasetLoader(
        custom_data_dir=config["data"]["custom_dir"],
        manifest_path=config["data"]["metadata_csv"],
        classes=classes,
        sequence_length=seq_len,
        feature_dim=feat_dim
    )
    X, y, meta_df = loader.load_custom_dataset()

    if len(X) == 0:
        if use_synthetic_if_empty:
            logger.warning("No custom dataset found! Generating synthetic benchmark sequences for pipeline calibration.")
            X, y, meta_df = DatasetLoader.generate_synthetic_benchmark_dataset(
                num_samples_per_class=20,
                classes=classes,
                sequence_length=seq_len,
                feature_dim=feat_dim,
                num_signers=4
            )
        else:
            raise ValueError("Dataset is empty. Record custom samples or prepare public dataset before training.")

    # Step 2: Leakage-Safe Splitting
    test_ratio = config["experiments"].get("test_split_ratio", 0.2)
    val_ratio = config["experiments"].get("val_split_ratio", 0.15)

    if meta_df["signer_id"].nunique() >= 3:
        (X_tr, y_tr, meta_tr), (X_val, y_val, meta_val), (X_te, y_te, meta_te) = split_dataset_signer_independent(
            X, y, meta_df, test_size=test_ratio, val_size=val_ratio, random_state=config["model"]["random_seed"]
        )
        split_type = "signer_independent"
    else:
        (X_tr, y_tr, meta_tr), (X_val, y_val, meta_val), (X_te, y_te, meta_te) = split_dataset_stratified(
            X, y, meta_df, test_size=test_ratio, val_size=val_ratio, random_state=config["model"]["random_seed"]
        )
        split_type = "stratified"

    # Step 3: Handle Class Imbalance
    y_tr_integers = np.argmax(y_tr, axis=1)
    present_classes = np.unique(y_tr_integers)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=y_tr_integers
    )
    class_weight_dict = {cls_idx: float(w) for cls_idx, w in zip(present_classes, weights)}

    # Step 4: Build Architecture
    m_cfg = config["model"]
    model = build_lstm_model(
        sequence_length=seq_len,
        feature_dim=feat_dim,
        num_classes=num_classes,
        lstm_units_1=m_cfg.get("lstm_units_1", 128),
        lstm_units_2=m_cfg.get("lstm_units_2", 64),
        dropout_rate=m_cfg.get("dropout_rate", 0.3),
        dense_units=m_cfg.get("dense_units", 64),
        learning_rate=m_cfg.get("learning_rate", 0.001)
    )

    # Step 5: Setup Callbacks & Paths
    save_dir = ensure_dir(m_cfg.get("save_dir", "models"))
    best_model_path = str(save_dir / "signvision_lstm_best.keras")
    checkpoint_path = str(save_dir / "checkpoints" / "lstm_checkpoint.keras")
    ensure_dir(str(Path(checkpoint_path).parent))

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=m_cfg.get("early_stopping_patience", 12),
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_categorical_accuracy",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=m_cfg.get("reduce_lr_patience", 6),
            min_lr=1e-5,
            verbose=1
        )
    ]

    # Step 6: Train Model
    logger.info("Starting model training...")
    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=m_cfg.get("epochs", 50),
        batch_size=m_cfg.get("batch_size", 16),
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # Save Best Model
    save_trained_model(model, best_model_path)

    # Step 7: Visualizations
    hist_dict = history.history
    curves_plot = plot_and_save_training_history(hist_dict, output_path="artifacts/plots/training_curves.png")

    # Step 8: Test Set Evaluation
    y_pred_probs = model.predict(X_te)
    y_pred_indices = np.argmax(y_pred_probs, axis=1)
    y_true_indices = np.argmax(y_te, axis=1)

    eval_metrics = compute_classification_metrics(y_true_indices, y_pred_indices, class_names=classes)
    cm_plot = plot_and_save_confusion_matrix(
        y_true_indices,
        y_pred_indices,
        class_names=classes,
        output_path="artifacts/plots/confusion_matrix.png"
    )

    # Step 9: Save Comprehensive Metadata
    metadata = {
        "model_name": m_cfg.get("name", "signvision_lstm"),
        "architecture": "2-Layer LSTM with Dropout & Dense Projection",
        "classes": classes,
        "num_classes": num_classes,
        "sequence_length": seq_len,
        "feature_dim": feat_dim,
        "total_parameters": int(model.count_params()),
        "trained_timestamp": datetime.datetime.now().isoformat(),
        "split_type": split_type,
        "training_samples": len(X_tr),
        "validation_samples": len(X_val),
        "test_samples": len(X_te),
        "test_accuracy": eval_metrics["accuracy"],
        "macro_f1": eval_metrics["macro_f1"],
        "macro_precision": eval_metrics["macro_precision"],
        "macro_recall": eval_metrics["macro_recall"],
        "model_file": best_model_path,
        "training_curves_plot": curves_plot,
        "confusion_matrix_plot": cm_plot
    }

    meta_path = str(save_dir / "model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Model metadata exported to {meta_path}")
    logger.info(f"Training Complete! Test Accuracy: {eval_metrics['accuracy']:.2%}, Macro-F1: {eval_metrics['macro_f1']:.3f}")

    return {
        "metadata": metadata,
        "history": hist_dict,
        "eval_metrics": eval_metrics
    }


if __name__ == "__main__":
    run_training_pipeline()
