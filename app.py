"""
SignVision AI — Real-Time Sign Language Recognition System
Home Dashboard Application.
"""

import json
import os
from pathlib import Path
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="SignVision AI — Sign Language Recognition",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        font-size: 1.15rem;
        color: #94A3B8;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(99, 102, 241, 0.5);
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.2;
    }

    .metric-label {
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-top: 0.3rem;
    }

    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }

    .badge-cyan { background: rgba(6, 182, 212, 0.15); color: #22D3EE; border: 1px solid rgba(6, 182, 212, 0.3); }
    .badge-purple { background: rgba(168, 85, 247, 0.15); color: #C084FC; border: 1px solid rgba(168, 85, 247, 0.3); }
    .badge-emerald { background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-amber { background: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3); }

    .feature-box {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to check model and dataset status
def get_system_status():
    manifest_path = Path("data/metadata/dataset_manifest.csv")
    classes_path = Path("config/classes.json")
    model_path = Path("models/signvision_lstm_best.keras")
    meta_path = Path("models/model_metadata.json")

    total_classes = 8
    classes_list = []
    if classes_path.is_file():
        with open(classes_path, "r", encoding="utf-8") as f:
            c_data = json.load(f)
            classes_list = c_data.get("classes", [])
            total_classes = len(classes_list)

    total_samples = 0
    if manifest_path.is_file():
        try:
            df = pd.read_csv(manifest_path)
            total_samples = len(df)
        except Exception:
            total_samples = 0

    model_trained = model_path.is_file()
    model_metadata = {}
    if meta_path.is_file():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                model_metadata = json.load(f)
        except Exception:
            pass

    return {
        "classes_list": classes_list,
        "total_classes": total_classes,
        "total_samples": total_samples,
        "model_trained": model_trained,
        "model_metadata": model_metadata
    }

status = get_system_status()

# Hero Header
st.markdown('<h1 class="main-title">SignVision AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Real-Time Sign Language Gesture Recognition using Computer Vision & LSTM-Based Deep Learning</p>', unsafe_allow_html=True)

# Technology Stack Pills
st.markdown("""
<div style="margin-bottom: 1.8rem;">
    <span class="badge-pill badge-cyan">OpenCV 4.8+</span>
    <span class="badge-pill badge-purple">MediaPipe Hands</span>
    <span class="badge-pill badge-emerald">TensorFlow / Keras 3</span>
    <span class="badge-pill badge-amber">2-Layer LSTM</span>
    <span class="badge-pill badge-cyan">Streamlit</span>
    <span class="badge-pill badge-purple">Scikit-Learn</span>
</div>
""", unsafe_allow_html=True)

# Primary Metrics Grid
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{status['total_classes']}</div>
        <div class="metric-label">Supported Gestures</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{status['total_samples']}</div>
        <div class="metric-label">Captured Sequences</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    status_label = "TRAINED & READY" if status["model_trained"] else "UNTRAINED"
    color_val = "#10B981" if status["model_trained"] else "#F59E0B"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color: {color_val}; font-size: 1.7rem;">{status_label}</div>
        <div class="metric-label">Deep Learning Model</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    test_acc = status["model_metadata"].get("test_accuracy")
    acc_str = f"{test_acc:.1%}" if test_acc is not None else "N/A"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color: #60A5FA;">{acc_str}</div>
        <div class="metric-label">Test Evaluation Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# Navigation Section
st.subheader("Explore Modules")
n_col1, n_col2, n_col3 = st.columns(3)

with n_col1:
    st.markdown("""
    <div class="feature-box">
        <h4>🎥 Live Recognition</h4>
        <p style="color: #94A3B8; font-size: 0.95rem;">
            Real-time webcam inference with MediaPipe hand landmark tracking, rolling sequence buffer,
            and sentence construction.
        </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Launch Live Camera ➜", key="nav_live", use_container_width=True):
        st.switch_page("pages/1_Live_Recognition.py")

    st.markdown("""
    <div class="feature-box" style="margin-top: 1rem;">
        <h4>📊 Model Evaluation</h4>
        <p style="color: #94A3B8; font-size: 0.95rem;">
            Examine classification reports, per-class F1-scores, normalized confusion matrices,
            and CPU inference latency.
        </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("View Evaluation ➜", key="nav_eval", use_container_width=True):
        st.switch_page("pages/4_Model_Evaluation.py")

with n_col2:
    st.markdown("""
    <div class="feature-box">
        <h4>📁 Dataset Management</h4>
        <p style="color: #94A3B8; font-size: 0.95rem;">
            Record custom sign gesture samples via webcam, view manifest distributions,
            and explore WLASL public dataset guides.
        </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Manage Dataset ➜", key="nav_data", use_container_width=True):
        st.switch_page("pages/2_Dataset_Management.py")

    st.markdown("""
    <div class="feature-box" style="margin-top: 1rem;">
        <h4>🔬 Research Experiments</h4>
        <p style="color: #94A3B8; font-size: 0.95rem;">
            Inspect cross-user generalization gaps, background robustness, speed sensitivity,
            and LSTM vs GRU vs Random Forest baselines.
        </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Inspect Experiments ➜", key="nav_exp", use_container_width=True):
        st.switch_page("pages/5_Research_Experiments.py")

with n_col3:
    st.markdown("""
    <div class="feature-box">
        <h4>⚡ Model Training</h4>
        <p style="color: #94A3B8; font-size: 0.95rem;">
            Configure hyperparameters, run training with early stopping,
            and generate training history curves and checkpoints.
        </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Start Training ➜", key="nav_train", use_container_width=True):
        st.switch_page("pages/3_Model_Training.py")

    st.markdown("""
    <div class="feature-box" style="margin-top: 1rem;">
        <h4>📖 Architecture & Docs</h4>
        <p style="color: #94A3B8; font-size: 0.95rem;">
            Detailed architectural specifications, MediaPipe coordinate normalization math,
            reproducibility guide, and references.
        </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Read Documentation ➜", key="nav_docs", use_container_width=True):
        st.switch_page("pages/6_About.py")

# System Specifications Footer
st.divider()
st.markdown("""
<div style="display: flex; justify-content: space-between; color: #64748B; font-size: 0.85rem;">
    <span>Topology: 21 Landmarks × 3 Coordinates = 63 Features per Frame</span>
    <span>Temporal Window: 30 Frames @ 30 FPS (1.0 sec window)</span>
    <span>Inference Gating: Double-Tier Smoothing & Debounce Buffer</span>
</div>
""", unsafe_allow_html=True)
