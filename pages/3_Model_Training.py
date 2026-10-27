"""
SignVision AI — Model Training Interface.
"""

import json
from pathlib import Path
import streamlit as st

from src.train import run_training_pipeline
from src.utils import load_classes, load_config

st.set_page_config(page_title="Model Training — SignVision AI", page_icon="⚡", layout="wide")

st.title("⚡ LSTM Model Training Pipeline")
st.markdown("Configure hyperparameters, initiate training jobs, and inspect convergence curves.")

config = load_config("config/config.yaml")
classes = load_classes("config/classes.json")
metadata_path = Path(config["model"]["metadata_path"])
curves_plot_path = Path("artifacts/plots/training_curves.png")

t_col1, t_col2 = st.columns([1, 1.2])

with t_col1:
    st.subheader("🛠️ Hyperparameter Settings")
    with st.form("training_hyperparams_form"):
        epochs = st.slider("Training Epochs", min_value=10, max_value=100, value=config["model"]["epochs"], step=5)
        batch_size = st.select_slider("Batch Size", options=[8, 16, 32, 64], value=config["model"]["batch_size"])
        learning_rate = st.select_slider("Learning Rate", options=[0.01, 0.005, 0.001, 0.0005, 0.0001], value=config["model"]["learning_rate"])

        st.markdown("**Architecture Parameters**")
        lstm_units_1 = st.slider("LSTM Layer 1 Units", 64, 256, config["model"]["lstm_units_1"], step=32)
        lstm_units_2 = st.slider("LSTM Layer 2 Units", 32, 128, config["model"]["lstm_units_2"], step=16)
        dropout_rate = st.slider("Dropout Rate", 0.1, 0.5, config["model"]["dropout_rate"], step=0.05)

        st.caption("Training will employ EarlyStopping (patience=12) and ReduceLROnPlateau (patience=6).")
        submit_train = st.form_submit_button("🚀 Start Training Pipeline", type="primary", use_container_width=True)

    if submit_train:
        # Update config file in-place temporarily
        config["model"]["epochs"] = epochs
        config["model"]["batch_size"] = batch_size
        config["model"]["learning_rate"] = learning_rate
        config["model"]["lstm_units_1"] = lstm_units_1
        config["model"]["lstm_units_2"] = lstm_units_2
        config["model"]["dropout_rate"] = dropout_rate

        with st.status("Executing training pipeline...", expanded=True) as status:
            st.write("1. Loading and preprocessing landmark sequences...")
            st.write("2. Verifying zero group leakage (signer-independent partitioning)...")
            st.write("3. Fitting 2-Layer LSTM with categorical cross-entropy & class weights...")
            try:
                results = run_training_pipeline(use_synthetic_if_empty=True)
                status.update(label="Training Completed Successfully!", state="complete", expanded=False)
                st.success("Model trained, best checkpoint saved, and metadata exported.")
                st.rerun()
            except Exception as e:
                status.update(label=f"Training Failed: {e}", state="error")
                st.error(f"Error during training execution: {e}")

with t_col2:
    st.subheader("📈 Convergence Curves & Status")
    if curves_plot_path.is_file():
        st.image(str(curves_plot_path), caption="Training vs Validation Accuracy & Loss Curves", use_container_width=True)
    else:
        st.info("No training curves found. Run a training session to generate accuracy and loss curves.")

    st.subheader("📋 Active Model Metadata")
    if metadata_path.is_file():
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        m1, m2, m3 = st.columns(3)
        m1.metric("Test Accuracy", f"{meta.get('test_accuracy', 0):.1%}")
        m2.metric("Macro-F1", f"{meta.get('macro_f1', 0):.3f}")
        m3.metric("Total Parameters", f"{meta.get('total_parameters', 0):,}")

        st.json(meta)
    else:
        st.warning("No model metadata found. Model has not been trained yet.")
