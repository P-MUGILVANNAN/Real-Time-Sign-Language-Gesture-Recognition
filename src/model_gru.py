"""
GRU (Gated Recurrent Unit) Model for comparative sequence classification.
Used for model benchmarking against the proposed LSTM architecture (Experiment D).
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from src.utils import setup_logger

logger = setup_logger("ModelGRU")


def build_gru_model(
    sequence_length: int = 30,
    feature_dim: int = 63,
    num_classes: int = 8,
    gru_units_1: int = 128,
    gru_units_2: int = 64,
    dropout_rate: float = 0.3,
    dense_units: int = 64,
    learning_rate: float = 0.001
) -> models.Sequential:
    """
    Constructs and compiles a 2-layer GRU network with identical dimensional capacity to the LSTM model.
    """
    model = models.Sequential(name="SignVision_GRU")

    model.add(layers.Input(shape=(sequence_length, feature_dim), name="landmark_sequence_input"))
    model.add(layers.GRU(
        units=gru_units_1,
        return_sequences=True,
        name="gru_layer_1"
    ))
    model.add(layers.Dropout(rate=dropout_rate, name="dropout_1"))

    model.add(layers.GRU(
        units=gru_units_2,
        return_sequences=False,
        name="gru_layer_2"
    ))
    model.add(layers.Dropout(rate=dropout_rate, name="dropout_2"))

    model.add(layers.Dense(units=dense_units, activation="relu", name="dense_projection"))
    model.add(layers.Dense(units=num_classes, activation="softmax", name="gesture_probabilities"))

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"]
    )

    logger.info(
        f"Built GRU model: input=({sequence_length}, {feature_dim}), "
        f"classes={num_classes}, total params={model.count_params():,}"
    )
    return model
