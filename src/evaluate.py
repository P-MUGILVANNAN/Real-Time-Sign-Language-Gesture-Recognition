"""
Independent model evaluation and benchmarking engine.
Loads serialized models and generates detailed scientific reports,
classification metrics, confusion matrices, and latency statistics.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf

from src.dataset_loader import DatasetLoader
from src.metrics import compute_classification_metrics, plot_and_save_confusion_matrix
from src.model_lstm import load_trained_model
from src.utils import ensure_dir, load_classes, load_config, setup_logger

logger = setup_logger("Evaluator")


def benchmark_model_latency(
    model: tf.keras.Model,
    input_shape: Tuple[int, int] = (30, 63),
    num_warmup: int = 10,
    num_iterations: int = 50
) -> Dict[str, float]:
    """
    Measures warm-up and forward-pass inference latency under CPU conditions.
    """
    dummy_input = np.random.randn(1, *input_shape).astype(np.float32)

    # Warmup
    for _ in range(num_warmup):
        _ = model(dummy_input, training=False)

    latencies_ms = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        _ = model(dummy_input, training=False)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    mean_lat = float(np.mean(latencies_ms))
    median_lat = float(np.median(latencies_ms))
    p95_lat = float(np.percentile(latencies_ms, 95))
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    return {
        "mean_latency_ms": round(mean_lat, 2),
        "median_latency_ms": round(median_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "throughput_fps": round(fps, 1)
    }


def evaluate_saved_model(
    model_path: str = "models/signvision_lstm_best.keras",
    config_path: str = "config/config.yaml",
    classes_path: str = "config/classes.json",
    output_report_path: str = "artifacts/reports/evaluation_report.md"
) -> Dict:
    """
    Evaluates a saved model and produces a structured research evaluation report.
    """
    config = load_config(config_path)
    classes = load_classes(classes_path)
    seq_len = config["features"]["sequence_length"]
    feat_dim = config["features"]["single_hand_features"]

    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Model file does not exist at {model_path}. Train the model first.")

    model = load_trained_model(model_path)

    # Load dataset
    loader = DatasetLoader(
        custom_data_dir=config["data"]["custom_dir"],
        manifest_path=config["data"]["metadata_csv"],
        classes=classes,
        sequence_length=seq_len,
        feature_dim=feat_dim
    )
    X, y, meta_df = loader.load_custom_dataset()

    if len(X) == 0:
        logger.warning("Empty dataset! Generating synthetic test sample for evaluation demonstration.")
        X, y, meta_df = DatasetLoader.generate_synthetic_benchmark_dataset(
            num_samples_per_class=10,
            classes=classes,
            sequence_length=seq_len,
            feature_dim=feat_dim
        )

    # Predict
    y_pred_probs = model.predict(X)
    y_pred_idx = np.argmax(y_pred_probs, axis=1)
    y_true_idx = np.argmax(y, axis=1)

    metrics = compute_classification_metrics(y_true_idx, y_pred_idx, class_names=classes)
    latency_stats = benchmark_model_latency(model, input_shape=(seq_len, feat_dim))

    cm_path = plot_and_save_confusion_matrix(
        y_true_idx,
        y_pred_idx,
        class_names=classes,
        output_path="artifacts/plots/eval_confusion_matrix.png"
    )

    # Markdown Report Generation
    ensure_dir(str(Path(output_report_path).parent))
    report_lines = [
        "# SignVision AI — Model Evaluation Report",
        "",
        "## 1. Executive Summary",
        f"- **Model Path**: `{model_path}`",
        f"- **Evaluated Samples**: {len(X)} across {len(classes)} classes",
        f"- **Test Accuracy**: {metrics['accuracy']:.2%}",
        f"- **Macro-F1 Score**: {metrics['macro_f1']:.3f}",
        f"- **Macro Precision**: {metrics['macro_precision']:.3f}",
        f"- **Macro Recall**: {metrics['macro_recall']:.3f}",
        f"- **Weighted F1**: {metrics['weighted_f1']:.3f}",
        "",
        "## 2. Real-Time Inference Latency",
        f"- **Mean Latency**: {latency_stats['mean_latency_ms']} ms",
        f"- **Median Latency**: {latency_stats['median_latency_ms']} ms",
        f"- **95th Percentile Latency**: {latency_stats['p95_latency_ms']} ms",
        f"- **Inference Throughput**: {latency_stats['throughput_fps']} FPS",
        "",
        "## 3. Per-Class Classification Report",
        "| Gesture Class | Precision | Recall | F1-Score | Support |",
        "| :--- | :---: | :---: | :---: | :---: |"
    ]

    for cls in classes:
        cls_data = metrics["classification_report"].get(cls, {})
        if cls_data:
            report_lines.append(
                f"| `{cls}` | {cls_data.get('precision', 0):.2f} | "
                f"{cls_data.get('recall', 0):.2f} | {cls_data.get('f1-score', 0):.2f} | "
                f"{cls_data.get('support', 0)} |"
            )

    report_lines.extend([
        "",
        "## 4. Visualizations",
        f"![Confusion Matrix]({cm_path})",
        "",
        "---",
        "*Report auto-generated by SignVision AI Evaluation Engine.*"
    ])

    report_text = "\n".join(report_lines)
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info(f"Evaluation report written to {output_report_path}")
    return {
        "metrics": metrics,
        "latency": latency_stats,
        "report_file": output_report_path,
        "confusion_matrix_file": cm_path
    }


if __name__ == "__main__":
    evaluate_saved_model()
