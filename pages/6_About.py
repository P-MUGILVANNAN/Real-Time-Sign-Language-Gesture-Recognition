"""
SignVision AI — Architecture, Documentation, and Scientific References.
"""

import streamlit as st

st.set_page_config(page_title="About & Docs — SignVision AI", page_icon="📖", layout="wide")

st.title("📖 System Architecture & Research Documentation")
st.markdown("Mathematical formulations, model topology, reproducibility guide, and scientific references.")

tab_arch, tab_norm, tab_limits, tab_refs = st.tabs([
    "🏗️ System Architecture",
    "📐 Normalization Formulations",
    "⚠️ Scope & Known Limitations",
    "📚 Scientific References"
])

with tab_arch:
    st.subheader("SignVision AI Pipeline Architecture")
    st.markdown("""
    ```
    ┌───────────────────────┐
    │ Live Webcam / Video   │ (30 FPS RGB Stream)
    └──────────┬────────────┘
               │
               ▼
    ┌───────────────────────┐
    │ MediaPipe HandLandmark│ 21 3D Landmarks per Hand (x, y, z)
    └──────────┬────────────┘
               │
               ▼
    ┌───────────────────────┐
    │ Invariant Transform   │ Wrist Centering: P_i' = P_i - P_0
    │ & Normalization       │ Scale Invariance: P_i'' = P_i' / ||P_9 - P_0||
    └──────────┬────────────┘
               │
               ▼
    ┌───────────────────────┐
    │ Rolling Sequence Buffer│ Deque(maxlen=30) -> Tensor Shape: (1, 30, 63)
    └──────────┬────────────┘
               │
               ▼
    ┌───────────────────────┐
    │ 2-Layer LSTM Network  │ LSTM(128, return_seq=True) -> Dropout(0.3)
    │                       │ -> LSTM(64) -> Dropout(0.3) -> Dense(64) -> Softmax(K)
    └──────────┬────────────┘
               │
               ▼
    ┌───────────────────────┐
    │ Temporal Debouncing   │ 1. Rolling probability averaging (window=5)
    │ & Confidence Gating   │ 2. Confidence thresholding (threshold >= 0.70)
    │                       │ 3. Consecutive stability filter (>= 4 frames)
    └──────────┬────────────┘
               │
               ▼
    ┌───────────────────────┐
    │ Real-Time UI & Output │ Label, Confidence %, FPS, Latency, Sentence Accumulator
    └───────────────────────┘
    ```
    """)

with tab_norm:
    st.subheader("Mathematical Landmark Preprocessing")
    st.markdown("""
    To ensure reliable recognition across varying camera distances, user heights, and lens perspectives:

    #### 1. Translation Centering
    Raw coordinates are translated so that the **wrist landmark ($P_0$)** serves as the coordinate origin $(0, 0, 0)$:
    $$P_i' = P_i - P_0, \quad \forall i \in \{0, \dots, 20\}$$

    #### 2. Scale Invariant Normalization
    Scale variation (distance from the webcam) is normalized by calculating the Euclidean distance between the wrist ($P_0$) and the middle finger metacarpophalangeal (MCP) joint ($P_9$):
    $$d_{\\text{ref}} = \\|P_9 - P_0\\|_2 = \\sqrt{(x_9 - x_0)^2 + (y_9 - y_0)^2 + (z_9 - z_0)^2}$$

    The normalized coordinate vector $P_i''$ is computed as:
    $$P_i'' = \\frac{P_i'}{d_{\\text{ref}}}$$

    This invariant representation guarantees that a user standing 1 meter away generates feature vectors geometrically consistent with a user standing 2 meters away.
    """)

with tab_limits:
    st.subheader("Scope, Ethical Considerations & Known Limitations")
    st.markdown("""
    > [!IMPORTANT]
    > **Distinction Between Dynamic Gesture Recognition and Sign Language Translation:**
    > This application recognizes **isolated dynamic hand gesture sequences** from a predefined vocabulary.
    > It is **not** a full grammatical sign language translation system. Full sign languages (such as ASL, BSL, or ISL) possess rich syntactic structures, facial grammars (non-manual markers), spatial reference frames, and mouth morphemes that cannot be translated word-for-word without a continuous translation model and facial landmark pipeline.

    ### Known Limitations:
    1. **Rapid Hand Motion Blur**: Extreme hand velocity can cause optical motion blur in standard webcams, leading to temporary tracking dropouts.
    2. **Severe Occlusions**: When fingers cross directly behind one another relative to the camera lens, Z-depth estimation relies on model priors.
    3. **Lighting & Complex Backgrounds**: While MediaPipe is robust to varying backgrounds, extreme backlighting or low-light conditions impair hand segmentation.
    """)

with tab_refs:
    st.subheader("Scientific References")
    st.markdown("""
    1. **Lugaresi, C., et al.** (2019). *MediaPipe: A Framework for Building Perception Pipelines*. arXiv preprint arXiv:1906.08172.
    2. **Li, D., Rodriguez, C., Yu, X., & Li, H.** (2020). *Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison*. The IEEE Winter Conference on Applications of Computer Vision (WACV 2020), pp. 1459-1469.
    3. **Hochreiter, S., & Schmidhuber, J.** (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735-1780.
    4. **Cho, K., et al.** (2014). *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation*. EMNLP 2014.
    5. **Zhang, F., et al.** (2020). *MediaPipe Hands: On-device Real-time Hand Tracking*. CVPR 2020 Workshop.
    """)
