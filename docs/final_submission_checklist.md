# SignVision AI — Final Submission Checklist

- [x] **Module 1: Real-Time Webcam Recognition**
  - [x] Access webcam using OpenCV (`cv2.VideoCapture`).
  - [x] Live camera feed rendered in Streamlit interface.
  - [x] Hand detection and real-time landmark tracking via MediaPipe.
  - [x] Normalized landmark coordinates (wrist-relative translation and scale invariance).
  - [x] Rolling sequence buffer (30 frames deque).
  - [x] LSTM inference forward pass on temporal window.
  - [x] Predicted gesture label display with confidence percentage.
  - [x] Confidence thresholding and neutral "No confident gesture detected" state.
  - [x] Multi-frame prediction smoothing and consecutive stability gating.
  - [x] Real-time FPS counter and inference latency metrics.
  - [x] Start/Stop camera stream controls and graceful error handling.

- [x] **Module 2: Hand Landmark Extraction**
  - [x] MediaPipe Tasks HandLandmarker (`hand_landmarker.task`).
  - [x] 21 anatomical landmarks per hand with X, Y, Z coordinates.
  - [x] Handedness detection.
  - [x] Single-hand ($21 \times 3 = 63$ features) and two-hand ($126$ features) schemas.
  - [x] Reproducible wrist-centered and scale-invariant transformation.

- [x] **Module 3: Gesture Dataset Management**
  - [x] Option A: Public dataset (WLASL) parser, index reader, and download guide.
  - [x] Option B: Custom webcam recording interface with countdown, sequence validation, and metadata manifest.
  - [x] Deletion of invalid samples from disk and manifest.

- [x] **Module 4: Data Preprocessing**
  - [x] Linear interpolation resampling to 30 fixed frames (`resample_sequence`).
  - [x] Zero-padding and truncation utilities.
  - [x] NaN and Inf sanitization.
  - [x] Bidirectional categorical label encoding (`GestureLabelEncoder`).
  - [x] Preserved temporal sequence ordering (no frame shuffling within sequence).

- [x] **Module 5: LSTM Deep Learning Model**
  - [x] Input shape $(30, 63)$.
  - [x] LSTM 1 (128 units, `return_sequences=True`) + Dropout (0.3).
  - [x] LSTM 2 (64 units, `return_sequences=False`) + Dropout (0.3).
  - [x] Dense (64 units, ReLU).
  - [x] Output Dense ($K$ classes, Softmax).
  - [x] Categorical Cross-Entropy, Adam optimizer.
  - [x] Model saving and loading in native Keras format.

- [x] **Module 6: Model Training Pipeline**
  - [x] Dataset loading and dimension validation.
  - [x] Signer-independent disjoint group splitting (`GroupShuffleSplit`).
  - [x] Class balance weighting.
  - [x] Callbacks: `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`.
  - [x] Training history curves exported to `artifacts/plots/`.
  - [x] Test set evaluation and confusion matrix heatmap.
  - [x] Export of `models/model_metadata.json`.

- [x] **Module 7: Real-Time Gesture Prediction**
  - [x] `RealTimeInferenceEngine` class with rolling deque buffer.
  - [x] Prediction smoothing and debouncing.
  - [x] Neutral/unknown abstention state.

- [x] **Module 8: Streamlit Web Application**
  - [x] Home Dashboard (`app.py`).
  - [x] Page 1: Live Recognition (`pages/1_Live_Recognition.py`).
  - [x] Page 2: Gesture Dataset (`pages/2_Dataset_Management.py`).
  - [x] Page 3: Model Training (`pages/3_Model_Training.py`).
  - [x] Page 4: Model Evaluation (`pages/4_Model_Evaluation.py`).
  - [x] Page 5: Research Experiments (`pages/5_Research_Experiments.py`).
  - [x] Page 6: About & Documentation (`pages/6_About.py`).

- [x] **Module 9: Prediction History**
  - [x] Timestamped history log with confidence and latency.
  - [x] Clear history control.

- [x] **Module 10: Gesture Sequence and Sentence Builder**
  - [x] Stable gesture accumulation into sentence string.
  - [x] Cooldown lock preventing duplicate insertion during sustained gesture.
  - [x] Undo and Clear controls.
  - [x] CSV export of recognized tokens.

- [x] **Research Experiments**
  - [x] Exp A: Cross-User Generalization (Signer-Independent vs Signer-Dependent).
  - [x] Exp D: Architecture Comparison (LSTM vs GRU vs Random Forest).
  - [x] Exp E: Sequence Length Ablation (15, 30, 45 frames).
  - [x] Exp F: Confidence Threshold Calibration.
  - [x] Actual CSV and plot generation in `artifacts/`.

- [x] **Unit & Integration Tests**
  - [x] 16 tests passing across `tests/`.

- [x] **Documentation & Deliverables**
  - [x] `README.md`, `docs/architecture.md`, `docs/dataset_guide.md`, `docs/training_guide.md`, `docs/research_methodology.md`, `docs/troubleshooting.md`, `docs/research_report.md`.
