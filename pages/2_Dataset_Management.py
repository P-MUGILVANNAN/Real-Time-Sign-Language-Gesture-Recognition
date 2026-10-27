"""
SignVision AI — Gesture Dataset Management & Webcam Data Collector.
"""

import json
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_collection import DataCollector
from src.landmark_extractor import LandmarkExtractor
from src.utils import ensure_dir, load_classes, load_config

st.set_page_config(page_title="Dataset Management — SignVision AI", page_icon="📁", layout="wide")

st.title("📁 Gesture Dataset Management")
st.markdown("Record custom gesture sequences, inspect class distributions, and manage data manifests.")

config = load_config("config/config.yaml")
classes = load_classes("config/classes.json")
manifest_path = Path(config["data"]["metadata_csv"])

# Initialize Collector in session_state
if "data_collector" not in st.session_state:
    st.session_state.data_collector = DataCollector(
        base_dir=config["data"]["custom_dir"],
        manifest_path=config["data"]["metadata_csv"],
        sequence_length=config["features"]["sequence_length"],
        feature_dim=config["features"]["single_hand_features"]
    )

collector: DataCollector = st.session_state.data_collector

# Tabs
tab_overview, tab_record, tab_manage, tab_wlasl = st.tabs([
    "📊 Dataset Overview",
    "🎥 Record Custom Gesture",
    "🛠️ Sample Management",
    "🌐 Public Dataset (WLASL)"
])

# ----------------- Tab 1: Overview -----------------
with tab_overview:
    stats = collector.get_dataset_statistics()
    st.subheader(f"Dataset Manifest Summary ({stats['total_samples']} Total Sequences)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Captured Samples", stats["total_samples"])
    col2.metric("Gesture Classes", len(stats["class_counts"]))
    col3.metric("Signers Represented", len(stats["signer_counts"]))
    col4.metric("Background Variations", len(stats["background_counts"]))

    if stats["total_samples"] > 0:
        c1, c2 = st.columns(2)
        with c1:
            df_class = pd.DataFrame(list(stats["class_counts"].items()), columns=["Gesture", "Count"])
            fig_cls = px.bar(
                df_class,
                x="Gesture",
                y="Count",
                color="Gesture",
                title="Sample Distribution by Gesture Class",
                template="plotly_dark"
            )
            st.plotly_chart(fig_cls, use_container_width=True)

        with c2:
            df_signer = pd.DataFrame(list(stats["signer_counts"].items()), columns=["Signer", "Count"])
            fig_sig = px.pie(
                df_signer,
                names="Signer",
                values="Count",
                title="Distribution Across Signers (Independence Analysis)",
                template="plotly_dark",
                hole=0.4
            )
            st.plotly_chart(fig_sig, use_container_width=True)

        st.subheader("Gesture Vocabulary Reference")
        with open("config/classes.json", "r", encoding="utf-8") as f:
            c_info = json.load(f)

        vocab_rows = []
        for c in classes:
            desc = c_info.get("class_descriptions", {}).get(c, "Standard dynamic gesture")
            samples = stats["class_counts"].get(c, 0)
            vocab_rows.append({"Class Label": c, "Description": desc, "Available Samples": samples})

        st.dataframe(pd.DataFrame(vocab_rows), use_container_width=True)
    else:
        st.info("No recorded gesture samples found. Use the 'Record Custom Gesture' tab to capture your first samples.")

# ----------------- Tab 2: Custom Webcam Recording -----------------
with tab_record:
    st.subheader("Record New Gesture Sequence")
    st.markdown("Capture a 30-frame dynamic landmark sequence using your webcam.")

    r_col1, r_col2 = st.columns([1, 1])

    with r_col1:
        target_gesture = st.selectbox("Target Gesture Class", classes)
        signer_id = st.text_input("Signer Identifier", value="signer_user_1")
        session_id = st.text_input("Session Identifier", value="session_01")
        background_env = st.selectbox("Background Condition", ["plain_wall", "office_desk", "living_room", "dim_light"])
        speed_var = st.selectbox("Performance Speed", ["normal", "slow", "fast"])
        cam_idx = st.number_input("Webcam Hardware Index", min_value=0, max_value=3, value=0)

        record_btn = st.button("🔴 Start Recording Session", type="primary", use_container_width=True)

    with r_col2:
        preview_box = st.empty()
        rec_status = st.empty()

    if record_btn:
        cap = cv2.VideoCapture(int(cam_idx), cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(cam_idx))

        if not cap.isOpened():
            rec_status.error(f"Failed to open camera index {cam_idx}.")
        else:
            try:
                # 3-Second Countdown
                for sec in [3, 2, 1]:
                    rec_status.warning(f"⏳ Prepare gesture '{target_gesture}'... Recording starts in {sec}s")
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.flip(frame, 1)
                        cv2.putText(frame, f"GET READY: {sec}", (160, 240), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 215, 255), 4)
                        preview_box.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                    time.sleep(1.0)

                rec_status.error(f"🔴 RECORDING '{target_gesture.upper()}' NOW! PERFORM GESTURE...")

                captured_features = []
                detection_flags = []
                target_frames = config["features"]["sequence_length"]

                for frame_no in range(target_frames):
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break

                    frame = cv2.flip(frame, 1)
                    features, detected, info = collector.extractor.extract_from_frame(frame)
                    captured_features.append(features)
                    detection_flags.append(detected)

                    annotated = collector.extractor.draw_landmarks_on_image(frame, info)
                    cv2.putText(
                        annotated,
                        f"RECORDING: {frame_no+1}/{target_frames}",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2
                    )
                    preview_box.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                    time.sleep(0.02)

                # Persist sample
                success, msg, metadata = collector.save_sequence_sample(
                    sequence_frames=captured_features,
                    gesture_class=target_gesture,
                    signer_id=signer_id,
                    session_id=session_id,
                    background=background_env,
                    speed=speed_var,
                    detection_flags=detection_flags
                )

                if success:
                    rec_status.success(f"✅ {msg}")
                    st.json(metadata)
                else:
                    rec_status.error(f"❌ {msg}")

            finally:
                cap.release()

# ----------------- Tab 3: Sample Management -----------------
with tab_manage:
    st.subheader("Dataset Manifest Explorer & Cleanup")
    if manifest_path.is_file():
        df_manifest = pd.read_csv(manifest_path)
        if not df_manifest.empty:
            st.dataframe(df_manifest, use_container_width=True)

            st.write("")
            st.markdown("#### Delete Invalid Sample")
            del_id = st.selectbox("Select Sample ID to Delete", df_manifest["sample_id"].tolist())
            if st.button("🗑️ Delete Selected Sample", type="secondary"):
                del_ok, del_msg = collector.delete_sample(del_id)
                if del_ok:
                    st.success(del_msg)
                    st.rerun()
                else:
                    st.error(del_msg)
        else:
            st.info("Manifest exists but is currently empty.")
    else:
        st.info("No manifest file found.")

# ----------------- Tab 4: Public Dataset (WLASL) -----------------
with tab_wlasl:
    st.subheader("🌐 WLASL (Word-Level American Sign Language) Integration")
    st.markdown("""
    **SignVision AI** natively supports integration with the **WLASL dataset**
    (2,000 word-level sign video instances).

    ### Dataset Information & Compliance
    - **Authors**: Dongxu Li, Cristian Rodriguez, Xin Yu, Hongdong Li (WACV 2020)
    - **License**: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
    - **Distribution Restriction**: Raw videos are copyrighted by their original creators across YouTube,
      DeafSign, and ASL-LEX. They must be downloaded via official WLASL scripts.

    ### Pre-processing Pipeline for WLASL
    1. Place `WLASL_v0.3.json` into `data/raw/wlasl/`.
    2. Download videos for target words: `hello`, `thank you`, `yes`, `no`, `please`, `help`, `sorry`.
    3. Run the SignVision video feature extractor:
    ```bash
    python scripts/extract_landmarks.py --input_dir data/raw/wlasl/videos/hello --output_dir data/custom/hello --label hello
    ```
    4. SignVision will automatically detect MediaPipe 3D coordinates, normalize to wrist-relative scale,
       and resample to 30 frames for LSTM training!
    """)
