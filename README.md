# SignVision AI: Real-Time Sign Language Gesture Recognition

[![Python](https://img.shields.io/badge/Python-3.11%20|%203.14-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-00C7B7?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

An end-to-end, research-oriented real-time sign language gesture recognition application combining **OpenCV**, **MediaPipe HandLandmarker**, **2-Layer LSTM Deep Learning**, and an interactive **Streamlit** dashboard.

---

## 🌟 Key Features

- **Real-Time Webcam Recognition**: Classifies dynamic gestures from a rolling 30-frame temporal buffer at >50 FPS on consumer CPU hardware.
- **Invariant Landmark Normalization**: Mathematical transformation rendering coordinates translation-invariant (wrist origin) and scale-invariant (wrist-to-MCP reference).
- **Temporal Sequence Modeling**: 2-Layer LSTM network ($152,392$ parameters) with categorical cross-entropy, early stopping, and learning rate scheduling.
- **Double-Tier Prediction Gating**: Rolling probability averaging (5 frames) and consecutive stability gating (4 frames) to eliminate false triggers and flickering.
- **Gesture Sentence Builder**: Accumulates recognized signs into text sequences with cooldown deduplication, Undo, Clear, and CSV export.
- **Signer-Independent Splitting**: Enforces strict `GroupShuffleSplit` on `signer_id` with mathematical leakage verification ($0\%$ group overlap).
- **Interactive Multi-Page Streamlit UI**:
  - `Home`: System status, metrics overview, technology stack.
  - `Live Recognition`: Webcam feed with skeleton overlay, probability bars, sentence accumulator.
  - `Dataset Management`: Custom webcam recorder, class distribution charts, WLASL dataset guide.
  - `Model Training`: Hyperparameter configuration, convergence curves, checkpointing.
  - `Model Evaluation`: Normalized confusion matrices, per-class F1-scores, CPU latency benchmarking.
  - `Research Experiments`: Cross-user generalization, speed variations, sequence length ablations, and LSTM vs GRU vs Random Forest baselines.
  - `About & Docs`: Mathematical formulations, architecture diagrams, and scientific citations.

---

## 📁 Repository Structure

```
Real-Time Sign Language Gesture Recognition/
├── app.py                      # Home Dashboard Streamlit entrypoint
├── requirements.txt            # Pinned dependencies
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation
├── config/
│   ├── config.yaml             # Pipeline, model, camera, and experiment settings
│   └── classes.json            # Supported gesture vocabulary & definitions
├── data/
│   ├── raw/                    # Raw video clips (WLASL)
│   ├── processed/              # Processed feature arrays (X_processed.npy, y_processed.npy)
│   ├── custom/                 # Custom webcam recorded sequences (.npy)
│   └── metadata/
│       └── dataset_manifest.csv# Dataset metadata & signer manifest
├── models/
│   ├── hand_landmarker.task    # MediaPipe HandLandmarker task bundle
│   ├── signvision_lstm_best.keras # Trained best LSTM model weights
│   ├── model_metadata.json     # Model architecture & evaluation metrics
│   └── checkpoints/            # Model training checkpoints
├── src/
│   ├── __init__.py
│   ├── utils.py                # Logging, config loaders, timer helpers
│   ├── landmark_extractor.py   # MediaPipe HandLandmarker & invariant normalization
│   ├── preprocessing.py        # Sequence resampling (interp1d), padding, label encoder
│   ├── video_processor.py      # Video file batch landmark extraction
│   ├── data_collection.py      # Custom webcam dataset collector
│   ├── dataset_loader.py       # Dataset loaders (custom, WLASL, synthetic benchmarks)
│   ├── dataset_split.py        # Signer-independent group splitting & leakage verification
│   ├── model_lstm.py           # 2-Layer LSTM network architecture
│   ├── model_gru.py            # Comparative 2-Layer GRU architecture
│   ├── model_baseline.py       # Random Forest statistical landmark baseline
│   ├── train.py                # Training engine with callbacks & class weighting
│   ├── evaluate.py             # Evaluation engine & latency benchmarking
│   ├── metrics.py              # Precision, Recall, F1, confusion matrix, curves
│   ├── prediction_smoothing.py # Rolling probability averager & debouncer
│   ├── sentence_builder.py     # Gesture sequence accumulator & token exporter
│   ├── inference.py            # Real-time inference engine
│   └── experiment_runner.py    # Runner for Experiments A through F
├── pages/
│   ├── 1_Live_Recognition.py
│   ├── 2_Dataset_Management.py
│   ├── 3_Model_Training.py
│   ├── 4_Model_Evaluation.py
│   ├── 5_Research_Experiments.py
│   └── 6_About.py
├── scripts/
│   ├── download_instructions.py# WLASL access & compliance guide
│   ├── extract_landmarks.py    # CLI tool for batch video landmark extraction
│   ├── prepare_dataset.py      # CLI tool for dataset compilation
│   ├── train_model.py          # CLI tool for model training
│   ├── evaluate_model.py       # CLI tool for model evaluation
│   └── run_experiments.py      # CLI tool for scientific research experiments
├── tests/
│   ├── test_preprocessing.py
│   ├── test_landmark_extraction.py
│   ├── test_dataset_splitting.py
│   ├── test_model.py
│   └── test_inference.py
├── artifacts/
│   ├── reports/                # Evaluation markdown reports
│   ├── plots/                  # Confusion matrices, curves, experiment charts
│   └── experiments/            # Experiment CSV results
└── docs/
    ├── architecture.md
    ├── dataset_guide.md
    ├── training_guide.md
    ├── research_methodology.md
    ├── troubleshooting.md
    ├── research_report.md
    └── final_submission_checklist.md
```

---

## 🚀 Quickstart & Execution Guide

### 1. Environment Setup
Create and activate a virtual environment:

**Windows (PowerShell)**:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS**:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset
To compile existing recordings or generate synthetic benchmark sequences:
```bash
python scripts/prepare_dataset.py --generate_benchmark
```

### 3. Train the Model
Train the 2-Layer LSTM network:
```bash
python scripts/train_model.py
```

### 4. Evaluate the Model
Evaluate test accuracy, confusion matrix, and latency:
```bash
python scripts/evaluate_model.py
```

### 5. Run Scientific Experiments
Execute the experimental research suite (Experiments A, D, E, F):
```bash
python scripts/run_experiments.py
```

### 6. Run Unit & Integration Tests
Execute the pytest suite:
```bash
python -m pytest tests/ -v
```

### 7. Launch the Streamlit Web Application
Start the interactive dashboard:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🔬 Mathematical Formulations

### 1. Invariant Coordinate Normalization
For raw landmarks $P_i = (x_i, y_i, z_i), i \in \{0, \dots, 20\}$:
- **Wrist Origin Centering**:
  $$P_i' = P_i - P_0$$
- **Scale Factor** (Euclidean distance between wrist $P_0$ and middle MCP joint $P_9$):
  $$d_{\text{ref}} = \sqrt{(x_9 - x_0)^2 + (y_9 - y_0)^2 + (z_9 - z_0)^2}$$
- **Scale Normalization**:
  $$P_i'' = \frac{P_i'}{d_{\text{ref}}}$$

### 2. Temporal Resampling
Variable-length sequences of length $T_{\text{in}}$ are resampled to $T=30$ frames using linear interpolation:
$$X(t) = (1 - \alpha) X_k + \alpha X_{k+1}, \quad \alpha = \frac{t - t_k}{t_{k+1} - t_k}$$

---

## 📜 Scientific References
1. **Li, D., et al.** (2020). *Word-level Deep Sign Language Recognition from Video*. IEEE WACV 2020.
2. **Lugaresi, C., et al.** (2019). *MediaPipe: A Framework for Building Perception Pipelines*. arXiv:1906.08172.
3. **Zhang, F., et al.** (2020). *MediaPipe Hands: On-device Real-time Hand Tracking*. CVPR 2020 Workshop.
4. **Hochreiter, S., & Schmidhuber, J.** (1997). *Long Short-Term Memory*. Neural Computation.
