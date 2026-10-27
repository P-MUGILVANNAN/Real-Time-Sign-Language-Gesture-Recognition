# SignVision AI — System Architecture Specification

## 1. Overview
SignVision AI is an end-to-end, research-grade computer vision and deep learning application designed to recognize dynamic sign language gestures in real time using a standard webcam. The system utilizes **MediaPipe HandLandmarker** to extract 3D skeletal landmark coordinates and a **2-Layer Long Short-Term Memory (LSTM)** neural network to classify the temporal evolution of hand gestures over a rolling 30-frame window.

---

## 2. Architectural Dataflow

```
[Webcam Video Stream (30 FPS)]
              │
              ▼
   [MediaPipe HandLandmarker] ──> (21 3D Landmarks: x, y, z)
              │
              ▼
[Invariant Preprocessing]
  ├── Wrist Centering: P_i' = P_i - P_0
  └── Scale Normalization: P_i'' = P_i' / ||P_9 - P_0||_2
              │
              ▼ (63-dimensional feature vector per frame)
  [Rolling Buffer: Deque(maxlen=30)] ──> Tensor Shape: (1, 30, 63)
              │
              ▼
    [2-Layer LSTM Classifier]
  ├── LSTM 1: 128 units (return_sequences=True) + Dropout(0.3)
  ├── LSTM 2: 64 units (return_sequences=False) + Dropout(0.3)
  ├── Dense: 64 units (ReLU)
  └── Dense: K units (Softmax)
              │
              ▼ (Softmax Probability Vector)
[Prediction Smoother & Gating Engine]
  ├── Rolling Probability Averaging (window=5)
  ├── Confidence Thresholding (threshold >= 0.70)
  └── Consecutive Stability Filter (>= 4 frames)
              │
              ▼
     [Recognized Gesture Event]
              │
              ▼
  [Sentence Builder & UI Dashboard]
```

---

## 3. Module Breakdown

### 3.1. Landmark Extraction (`src/landmark_extractor.py`)
- Employs MediaPipe Tasks HandLandmarker (`hand_landmarker.task`).
- Detects up to 2 hands per frame, extracting 21 anatomical landmarks per hand.
- Each landmark consists of normalized image coordinates $(x, y)$ and relative depth $(z)$.
- Supports both single-hand (63 features) and two-hand (126 features) modes with consistent zero-padding for absent hands.

### 3.2. Normalization & Preprocessing (`src/preprocessing.py`)
- **Wrist Origin Centering**: Shifts the origin $(0, 0, 0)$ to the wrist joint (landmark 0).
- **Scale Invariance**: Normalizes all coordinates by dividing by the distance between the wrist ($P_0$) and the middle finger MCP joint ($P_9$).
- **Temporal Resampling**: Interpolates variable-length gesture sequences into a uniform 30-frame sequence using 1D linear spline interpolation.
- **Sanitization**: Identifies and removes non-finite values (NaN, Inf).

### 3.3. Deep Learning Architecture (`src/model_lstm.py`)
- **Input Dimension**: $(B, 30, 63)$ where $B$ is the batch size, 30 is the number of time frames, and 63 is the number of landmark features.
- **Layer 1**: LSTM with 128 units and `return_sequences=True`.
- **Dropout 1**: 30% dropout probability to prevent co-adaptation.
- **Layer 2**: LSTM with 64 units capturing the temporal sequence summary.
- **Dropout 2**: 30% dropout probability.
- **Dense Projection**: 64 units with ReLU non-linearity.
- **Output Layer**: $K$ neurons with Softmax activation.
- **Loss**: Categorical Cross-Entropy; **Optimizer**: Adam ($\alpha=0.001$).

### 3.4. Temporal Debouncing & Smoothing (`src/prediction_smoothing.py`)
- Standard argmax classification on individual frames suffers from high-frequency prediction flickering.
- SignVision AI solves this via:
  1. A rolling probability window of 5 frames.
  2. An abstention threshold (default 0.70).
  3. A stability counter requiring candidate predictions to remain consistent across 4 consecutive frames before updating the displayed state.

### 3.5. Live Inference Engine (`src/inference.py`)
- Encapsulates the video capture loop, landmark extraction, rolling buffer, model inference, and sentence building into a thread-safe, modular Python class.
- Operates on edge CPU hardware with typical latency under 15 ms per frame (>60 FPS throughput).
