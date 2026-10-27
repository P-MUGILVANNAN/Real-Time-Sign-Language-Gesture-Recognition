# SignVision AI — Scientific Research Methodology & Experiments

## 1. Research Objectives
The primary research question addressed by SignVision AI is:
> *How effectively do normalized MediaPipe hand landmark sequences paired with LSTM architectures generalize across unseen users, diverse background environments, and performance speeds without incurring data leakage?*

---

## 2. Methodology & Leakage Prevention

### 2.1. The Critical Issue of Signer Data Leakage
In computer vision gesture recognition literature, a prevalent flaw is random frame-level or sample-level splitting across recordings from the same individual. This causes **biometric and spatial data leakage**, where models memorize idiosyncratic movement styles rather than learning genuine semantic sign trajectories.

### 2.2. Disjoint Group Splitting Strategy
SignVision AI formally enforces **signer-independent group partitioning** (`src/dataset_split.py`):
- All sequences produced by a given `signer_id` are placed strictly into either the training, validation, or test partition.
- `verify_split_leakage` mathematically validates that:
  $$\text{Signers}_{\text{train}} \cap \text{Signers}_{\text{val}} = \emptyset$$
  $$\text{Signers}_{\text{train}} \cap \text{Signers}_{\text{test}} = \emptyset$$
  $$\text{Signers}_{\text{val}} \cap \text{Signers}_{\text{test}} = \emptyset$$

---

## 3. Experimental Suite Overview

### 3.1. Experiment A: Cross-User Generalization
- **Setup**: Evaluates model performance on signers that were completely withheld during training compared to a signer-dependent baseline.
- **Metric**: Macro-F1 drop ($\Delta \text{F1}$).
- **Output**: `artifacts/experiments/experiment_a_cross_user.csv`, `artifacts/plots/experiment_a_cross_user.png`.

### 3.2. Experiment B: Background Robustness
- **Setup**: Compares recognition consistency across varying environmental backgrounds (plain wall, office desk, living room, dim lighting).
- **Hypothesis**: Skeletal landmarks abstract away texture and background appearance, achieving higher invariance than raw pixel CNNs.

### 3.3. Experiment C: Gesture-Speed Robustness
- **Setup**: Evaluates gestures executed at slow, normal, and fast cadences.
- **Hypothesis**: Linear temporal resampling (`resample_sequence`) normalizes duration variations into a consistent 30-frame temporal tensor.

### 3.4. Experiment D: Architecture Comparison
- **Setup**: Rigorous benchmarking of:
  1. Proposed 2-Layer LSTM (152,392 parameters).
  2. Comparative 2-Layer GRU (116,040 parameters).
  3. Conventional Random Forest classifier trained on 315 temporal statistical landmark features (mean, std, min, max, velocity delta).
- **Output**: `artifacts/experiments/experiment_d_model_comparison.csv`, `artifacts/plots/experiment_d_model_comparison.png`.

### 3.5. Experiment E: Sequence Length Ablation
- **Setup**: Tests sequence window lengths of 15, 30, and 45 frames.
- **Evaluation**: Assesses the trade-off between temporal contextual capacity (Macro-F1) and inference latency (ms).
- **Output**: `artifacts/experiments/experiment_e_sequence_length.csv`, `artifacts/plots/experiment_e_sequence_length.png`.

### 3.6. Experiment F: Confidence Threshold Calibration
- **Setup**: Evaluates acceptance thresholds ranging from $0.50$ to $0.90$.
- **Evaluation**: Measures the trade-off between coverage (% accepted predictions) and precision on accepted gestures.
- **Output**: `artifacts/experiments/experiment_f_confidence_threshold.csv`, `artifacts/plots/experiment_f_confidence_threshold.png`.
