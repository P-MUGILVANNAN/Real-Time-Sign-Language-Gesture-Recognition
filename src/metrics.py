"""
Comprehensive evaluation metrics, confusion matrix visualization,
and latency benchmarking for gesture classification.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

from src.utils import ensure_dir, setup_logger

logger = setup_logger("Metrics")


def compute_classification_metrics(
    y_true_indices: np.ndarray,
    y_pred_indices: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Union[float, Dict]]:
    """
    Computes rigorous classification metrics:
    - Overall Accuracy
    - Macro and Weighted Precision, Recall, F1
    - Per-class precision, recall, f1, support
    """
    acc = float(accuracy_score(y_true_indices, y_pred_indices))
    macro_p = float(precision_score(y_true_indices, y_pred_indices, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true_indices, y_pred_indices, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true_indices, y_pred_indices, average="macro", zero_division=0))

    weighted_p = float(precision_score(y_true_indices, y_pred_indices, average="weighted", zero_division=0))
    weighted_r = float(recall_score(y_true_indices, y_pred_indices, average="weighted", zero_division=0))
    weighted_f1 = float(f1_score(y_true_indices, y_pred_indices, average="weighted", zero_division=0))

    report_dict = classification_report(
        y_true_indices,
        y_pred_indices,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    return {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "classification_report": report_dict
    }


def plot_and_save_confusion_matrix(
    y_true_indices: np.ndarray,
    y_pred_indices: np.ndarray,
    class_names: List[str],
    output_path: str = "artifacts/plots/confusion_matrix.png",
    normalize: bool = True,
    title: str = "Normalized Confusion Matrix"
) -> str:
    """
    Renders an publication-quality Seaborn heatmap confusion matrix.
    """
    ensure_dir(str(Path(output_path).parent))

    cm = confusion_matrix(y_true_indices, y_pred_indices)
    if normalize:
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
        cm_display = cm_norm
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
        linewidths=0.5,
        linecolor="#e0e0e0"
    )

    ax.set_title(title, fontsize=14, pad=12, fontweight="bold")
    ax.set_xlabel("Predicted Gesture Class", fontsize=11, labelpad=8)
    ax.set_ylabel("True Gesture Class", fontsize=11, labelpad=8)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved confusion matrix plot to {output_path}")
    return output_path


def plot_and_save_training_history(
    history_dict: Dict,
    output_path: str = "artifacts/plots/training_curves.png"
) -> str:
    """
    Generates side-by-side loss and accuracy curves for training and validation.
    """
    ensure_dir(str(Path(output_path).parent))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    epochs = range(1, len(history_dict.get("loss", [])) + 1)

    # Accuracy subplot
    ax1.plot(epochs, history_dict.get("categorical_accuracy", []), label="Train Accuracy", color="#2563EB", lw=2)
    if "val_categorical_accuracy" in history_dict:
        ax1.plot(epochs, history_dict.get("val_categorical_accuracy", []), label="Val Accuracy", color="#059669", lw=2, linestyle="--")
    ax1.set_title("Model Accuracy Across Epochs", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Loss subplot
    ax2.plot(epochs, history_dict.get("loss", []), label="Train Loss", color="#DC2626", lw=2)
    if "val_loss" in history_dict:
        ax2.plot(epochs, history_dict.get("val_loss", []), label="Val Loss", color="#D97706", lw=2, linestyle="--")
    ax2.set_title("Categorical Cross-Entropy Loss Across Epochs", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved training curves plot to {output_path}")
    return output_path
