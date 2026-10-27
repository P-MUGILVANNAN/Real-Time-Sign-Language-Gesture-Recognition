"""
SignVision AI — Live Webcam Recognition & Sentence Builder.
"""

import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.inference import RealTimeInferenceEngine
from src.utils import FPSCounter, load_classes, load_config

st.set_page_config(page_title="Live Recognition — SignVision AI", page_icon="🎥", layout="wide")

# Custom CSS for Live Recognition
st.markdown("""
<style>
    .prediction-banner {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
        border: 2px solid #38BDF8;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 0 25px rgba(56, 189, 248, 0.2);
    }
    .gesture-label {
        font-size: 2.5rem;
        font-weight: 800;
        color: #38BDF8;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }
    .sentence-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .sentence-text {
        font-size: 1.4rem;
        font-weight: 600;
        color: #F8FAFC;
        min-height: 2.2rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎥 Live Sign Language Recognition")
st.markdown("Real-time temporal gesture classification via MediaPipe landmark sequences and LSTM.")

# Initialize Engine in session_state so it is not re-instantiated on every rerun
if "inference_engine" not in st.session_state:
    with st.spinner("Initializing MediaPipe Landmark Extractor & LSTM Engine..."):
        st.session_state.inference_engine = RealTimeInferenceEngine()
        st.session_state.fps_counter = FPSCounter(window_size=15)
        st.session_state.history_records = []

engine: RealTimeInferenceEngine = st.session_state.inference_engine
fps_counter: FPSCounter = st.session_state.fps_counter

# Sidebar Controls
st.sidebar.header("⚙️ Recognition Settings")
draw_skeleton = st.sidebar.checkbox("Draw Hand Landmarks Skeleton", value=True)
conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.40, max_value=0.95, value=0.70, step=0.05)
smoothing_window = st.sidebar.slider("Smoothing Window Size", min_value=3, max_value=12, value=5, step=1)
camera_index = st.sidebar.number_input("Webcam Hardware Index", min_value=0, max_value=4, value=0, step=1)

# Apply dynamic threshold adjustments to engine
engine.smoother.confidence_threshold = conf_threshold
engine.smoother.window_size = smoothing_window

# Main Layout
col_video, col_results = st.columns([1.3, 1.0])

with col_video:
    st.subheader("Webcam Stream")
    camera_mode = st.radio("Camera Capture Mode", ["OpenCV Live Stream (Recommended)", "Browser Snapshot Mode"], horizontal=True)

    stream_container = st.empty()
    status_alert = st.empty()

    btn_col1, btn_col2 = st.columns(2)
    start_cam = btn_col1.button("▶️ Start Camera Stream", use_container_width=True, type="primary")
    stop_cam = btn_col2.button("⏹️ Stop Camera Stream", use_container_width=True)

    if "camera_active" not in st.session_state:
        st.session_state.camera_active = False

    if start_cam:
        st.session_state.camera_active = True
    if stop_cam:
        st.session_state.camera_active = False
        engine.reset_buffer()

with col_results:
    st.subheader("Predicted Gesture")
    banner_placeholder = st.empty()

    # Metrics row
    m_col1, m_col2, m_col3 = st.columns(3)
    fps_metric = m_col1.empty()
    latency_metric = m_col2.empty()
    buffer_metric = m_col3.empty()

    st.write("")
    st.markdown("### 💬 Sentence Accumulator")
    sentence_box = st.empty()

    s_btn1, s_btn2, s_btn3 = st.columns(3)
    if s_btn1.button("↩️ Undo Token", use_container_width=True):
        engine.sentence_builder.undo_last_token()
    if s_btn2.button("🗑️ Clear Sentence", use_container_width=True):
        engine.sentence_builder.clear()
    export_csv = s_btn3.button("📥 Export CSV", use_container_width=True)

    st.write("")
    st.markdown("### 📊 Gesture Probabilities")
    prob_bars_container = st.empty()

# Render default initial state
banner_placeholder.markdown("""
<div class="prediction-banner">
    <div style="font-size: 0.9rem; color: #94A3B8;">Awaiting camera input</div>
    <div class="gesture-label" style="font-size: 1.8rem; color: #64748B;">Camera Standby</div>
    <div style="color: #94A3B8; font-size: 0.85rem;">Click 'Start Camera Stream' to begin live recognition</div>
</div>
""", unsafe_allow_html=True)

sentence_box.markdown(f"""
<div class="sentence-card">
    <div style="font-size: 0.8rem; text-transform: uppercase; color: #94A3B8; margin-bottom: 0.2rem;">Recognized Sentence</div>
    <div class="sentence-text">{engine.sentence_builder.get_sentence() or '<span style=\"color:#64748B;\">No words accumulated yet</span>'}</div>
</div>
""", unsafe_allow_html=True)

fps_metric.metric("FPS", "0.0")
latency_metric.metric("Latency", "0.0 ms")
buffer_metric.metric("Buffer", f"0 / {engine.seq_len}")

# Live OpenCV Camera Loop
if st.session_state.camera_active and camera_mode == "OpenCV Live Stream (Recommended)":
    cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(int(camera_index))

    if not cap.isOpened():
        status_alert.error(
            f"Unable to access camera index {camera_index}. "
            "Please check permissions or switch to Browser Snapshot Mode."
        )
        st.session_state.camera_active = False
    else:
        status_alert.success("Live Camera Feed Active. Perform signs in front of the lens.")
        try:
            while st.session_state.camera_active:
                ret, frame = cap.read()
                if not ret or frame is None:
                    status_alert.warning("Failed to grab video frame.")
                    break

                # Flip horizontally for natural mirror feel
                frame = cv2.flip(frame, 1)

                # Process frame through inference engine
                annotated, result = engine.process_frame(frame, draw_landmarks=draw_skeleton)
                current_fps = fps_counter.update()

                # Display video
                rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                stream_container.image(rgb_frame, channels="RGB", use_container_width=True)

                # Update Prediction Banner
                pred_class = result["gesture_class"]
                conf_val = result["confidence"]
                is_confident = pred_class != engine.config["inference"]["unknown_class_label"]
                accent_color = "#38BDF8" if is_confident else "#64748B"

                banner_placeholder.markdown(f"""
                <div class="prediction-banner" style="border-color: {accent_color};">
                    <div style="font-size: 0.9rem; color: #94A3B8;">Detected Gesture Label</div>
                    <div class="gesture-label" style="color: {accent_color};">{pred_class}</div>
                    <div style="color: #E2E8F0; font-size: 1rem; font-weight: 600;">
                        Confidence: {conf_val:.1%} &nbsp;|&nbsp; Threshold: {conf_threshold:.0%}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Update Metrics
                fps_metric.metric("FPS", f"{current_fps:.1f}")
                latency_metric.metric("Inference Latency", f"{result['latency_ms']:.1f} ms")
                buffer_metric.metric("Sequence Buffer", f"{result['buffer_frames']} / {engine.seq_len}")

                # Update Sentence
                sentence_box.markdown(f"""
                <div class="sentence-card">
                    <div style="font-size: 0.8rem; text-transform: uppercase; color: #94A3B8; margin-bottom: 0.2rem;">Recognized Sentence</div>
                    <div class="sentence-text">{result['sentence'] or '<span style=\"color:#64748B;\">No words accumulated yet</span>'}</div>
                </div>
                """, unsafe_allow_html=True)

                # Update Probability Bars
                if result["probabilities"]:
                    with prob_bars_container.container():
                        for cls_name, prob in result["probabilities"].items():
                            st.write(f"**{cls_name.replace('_', ' ').capitalize()}** ({prob:.1%})")
                            st.progress(float(min(max(prob, 0.0), 1.0)))

                # Record history
                if is_confident and result.get("is_new_trigger", False):
                    st.session_state.history_records.append({
                        "timestamp": time.strftime("%H:%M:%S"),
                        "gesture": pred_class,
                        "confidence": f"{conf_val:.2%}",
                        "latency_ms": f"{result['latency_ms']:.1f}"
                    })

                time.sleep(0.01)

        finally:
            cap.release()

# Browser Snapshot Mode (st.camera_input fallback)
elif camera_mode == "Browser Snapshot Mode":
    picture = st.camera_input("Take a snapshot for hand landmark verification")
    if picture is not None:
        bytes_data = picture.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        annotated, result = engine.process_frame(cv2_img, draw_landmarks=draw_skeleton)
        stream_container.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        st.success(f"Snapshot Processed. Hand detected: {result['hand_detected']}. Buffer primed: {result['buffer_frames']}/{engine.seq_len}")

# Prediction History Log
if st.session_state.history_records:
    st.write("")
    st.subheader("📜 Recent Recognized Gestures History")
    hist_df = pd.DataFrame(st.session_state.history_records)
    st.dataframe(hist_df.tail(10), use_container_width=True)
    if st.button("Clear History Log"):
        st.session_state.history_records.clear()
        st.rerun()
