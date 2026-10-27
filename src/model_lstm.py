"""
LSTM Deep Learning Model for temporal sign gesture classification.
Implements the research-grade 2-layer LSTM architecture specified in Module 5.
"""

from typing import Optional, Tuple
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from src.utils import setup_logger

logger = setup_logger("ModelLSTM")


def build_lstm_model(
    sequence_length: int = 30,
    feature_dim: int = 63,
    num_classes: int = 8,
    lstm_units_1: int = 128,
    lstm_units_2: int = 64,
    dropout_rate: float = 0.3,
    dense_units: int = 64,
    learning_rate: float = 0.001
) -> models.Sequential:
    """
    Constructs and compiles the 2-layer LSTM network for dynamic gesture sequence recognition.

    Architecture:
    1. Input layer: (sequence_length, feature_dim) = (30, 63)
    2. LSTM 1: 128 units, return_sequences=True
    3. Dropout: 0.3
    4. LSTM 2: 64 units, return_sequences=False
    5. Dropout: 0.3
    6. Dense: 64 units, ReLU activation
    7. Dense: num_classes units, Softmax activation

    Loss: categorical_crossentropy
    Optimizer: Adam
    """
    model = models.Sequential(name="SignVision_LSTM")

    # Layer 1: First LSTM layer returning sequences
    model.add(layers.Input(shape=(sequence_length, feature_dim), name="landmark_sequence_input"))
    model.add(layers.LSTM(
        units=lstm_units_1,
        return_sequences=True,
        name="lstm_layer_1"
    ))
    model.add(layers.Dropout(rate=dropout_rate, name="dropout_1"))

    # Layer 2: Second LSTM layer summarizing temporal features
    model.add(layers.LSTM(
        units=lstm_units_2,
        return_sequences=False,
        name="lstm_layer_2"
    ))
    model.add(layers.Dropout(rate=dropout_rate, name="dropout_2"))

    # Layer 3: Fully Connected feature projection
    model.add(layers.Dense(units=dense_units, activation="relu", name="dense_projection"))

    # Output Layer
    model.add(layers.Dense(units=num_classes, activation="softmax", name="gesture_probabilities"))

    # Compile
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"]
    )

    logger.info(
        f"Built LSTM model: input=({sequence_length}, {feature_dim}), "
        f"classes={num_classes}, total params={model.count_params():,}"
    )
    return model


def save_trained_model(model: models.Model, filepath: str):
    """
    Saves the Keras model in native format.
    """
    model.save(filepath)
    logger.info(f"Model saved successfully to {filepath}")


def load_trained_model(filepath: str) -> models.Model:
    """
    Loads a saved Keras model.
    """
    model = models.load_model(filepath)
    logger.info(f"Loaded trained model from {filepath}")
    return model
