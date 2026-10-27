"""
SignVision AI — Model Evaluation & Performance Benchmark Dashboard.
"""

import json
from pathlib import Path
import pandas as pd
import streamlit as st

from src.evaluate import evaluate_saved_model
from src.utils import load_classes, load_config

st.set_page_config(page_title="Model Evaluation — SignVision AI", page_icon="📊", layout="wide")

st.title("📊 Model Evaluation & Benchmarks")
st.markdown("Quantitative assessment of temporal classification performance, latency, and confusion matrices.")

config = load_config("config/config.yaml")
classes = load_classes("config/classes.json")
metadata_path = Path("models/model_metadata.json")
cm_plot_path = Path("artifacts/plots/eval_confusion_matrix.png")
if not cm_plot_path.is_file():
    cm_plot_path = Path("artifacts/plots/confusion_matrix.png")

eval_report_path = Path("artifacts/reports/evaluation_report.md")

# Top Controls
top_col1, top_col2 = st.columns([3, 1])
with top_col2:
    if st.button("🔄 Run Full Evaluation", type="primary", use_container_width=True):
        with st.spinner("Evaluating model on test partition..."):
            try:
                evaluate_saved_model()
                st.success("Evaluation complete!")
                st.rerun()
            except Exception as e:
                st.error(f"Evaluation failed: {e}")

# Load Metadata
if metadata_path.is_file():
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Metric Banner Cards
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Accuracy", f"{meta.get('test_accuracy', 0):.1%}")
    m2.metric("Macro-F1", f"{meta.get('macro_f1', 0):.3f}")
    m3.metric("Macro Precision", f"{meta.get('macro_precision', 0):.3f}")
    m4.metric("Macro Recall", f"{meta.get('macro_recall', 0):.3f}")
    m5.metric("Split Strategy", meta.get("split_type", "signer_independent").replace("_", " ").title())
    m6.metric("Test Samples", meta.get("test_samples", 0))

    st.write("")

    # Visualizations & Per-Class Details
    col_cm, col_report = st.columns([1.1, 1.2])

    with col_cm:
        st.subheader("Normalized Confusion Matrix")
        if cm_plot_path.is_file():
            st.image(str(cm_plot_path), caption="Normalized Test Set Confusion Matrix", use_container_width=True)
        else:
            st.info("Confusion matrix plot pending generation.")

    with col_report:
        st.subheader("Per-Class Classification Metrics")
        if eval_report_path.is_file():
            with open(eval_report_path, "r", encoding="utf-8") as f:
                report_content = f.read()

            # Render markdown table from report
            st.markdown(report_content)
        else:
            st.info("Click 'Run Full Evaluation' to generate per-class breakdown.")

    st.divider()
    st.subheader("Inference Latency & Production Efficiency")
    st.markdown("""
    SignVision AI is engineered for edge CPU webcam inference with zero GPU dependency.
    The lightweight 2-layer LSTM architecture operates well within real-time streaming constraints (30 FPS requires < 33.3 ms per frame).
    """)

else:
    st.warning("No model evaluation metadata found. Please train the model in the Model Training page first.")
