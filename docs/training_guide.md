# SignVision AI — Model Training & Optimization Guide

## 1. Overview
This guide details the procedures for training the 2-Layer LSTM network, configuring training callbacks, and exporting model artifacts.

---

## 2. Training Workflow via CLI

Execute the end-to-end training pipeline using the default configuration:
```bash
python scripts/train_model.py
```

Optional arguments:
- `--config`: Custom configuration file path (default: `config/config.yaml`).
- `--classes`: Custom classes definition path (default: `config/classes.json`).

---

## 3. Training Architecture & Hyperparameters

| Hyperparameter | Default Value | Description |
| :--- | :---: | :--- |
| `sequence_length` | 30 | Temporal window frames (1.0 sec @ 30 FPS) |
| `feature_dim` | 63 | 21 landmarks × 3 coordinates |
| `lstm_units_1` | 128 | First recurrent layer hidden units |
| `lstm_units_2` | 64 | Second recurrent layer hidden units |
| `dropout_rate` | 0.30 | Regularization dropout rate |
| `dense_units` | 64 | Projection layer units before softmax |
| `learning_rate` | 0.001 | Initial Adam optimizer learning rate |
| `batch_size` | 16 | Mini-batch sample size |
| `epochs` | 60 | Maximum training iterations |
| `early_stopping_patience` | 12 | Epochs to wait before early stopping |
| `reduce_lr_patience` | 6 | Epochs to wait before decaying learning rate by 0.5 |

---

## 4. Training Callbacks & Checkpointing

The training script automatically registers three production-grade Keras callbacks:
1. **ModelCheckpoint**:
   - Monitors `val_categorical_accuracy`.
   - Saves the optimal weights to `models/checkpoints/lstm_checkpoint.keras`.
2. **EarlyStopping**:
   - Monitors `val_loss`.
   - Halts training if no progress is observed for 12 epochs and restores the best model weights.
3. **ReduceLROnPlateau**:
   - Monitors `val_loss`.
   - Multiplies the learning rate by $0.5$ when validation loss stagnates for 6 epochs, with a minimum floor of $10^{-5}$.

---

## 5. Generated Artifacts

Upon completion of training, the following artifacts are generated:
- `models/signvision_lstm_best.keras`: The final optimized Keras model.
- `models/model_metadata.json`: Machine-readable metadata (classes, dimensions, parameters, test accuracy, timestamp).
- `artifacts/plots/training_curves.png`: Side-by-side plots of loss and accuracy.
- `artifacts/plots/confusion_matrix.png`: Normalized confusion matrix evaluated on the held-out test split.
