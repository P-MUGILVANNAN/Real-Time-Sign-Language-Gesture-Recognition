"""
Unit tests for real-time inference, prediction smoothing, debouncing, and sentence builder.
"""

import numpy as np
import pytest

from src.prediction_smoothing import PredictionSmoother
from src.preprocessing import GestureLabelEncoder
from src.sentence_builder import GestureSentenceBuilder


def test_prediction_smoothing_and_debouncing():
    """Verify that transient noise is ignored and stable gestures trigger cleanly."""
    classes = ["hello", "thank_you", "yes", "no"]
    encoder = GestureLabelEncoder(classes)

    smoother = PredictionSmoother(
        encoder=encoder,
        smoothing_window_size=3,
        confidence_threshold=0.70,
        stable_frames_required=3,
        unknown_label="Neutral"
    )

    # Initial state
    assert smoother.active_stable_class == "Neutral"

    # Single transient noise frame should not change stable state
    noise_prob = np.array([0.9, 0.05, 0.02, 0.03])  # hello
    cls, conf, new_trig, _ = smoother.update(noise_prob)
    assert cls == "Neutral"
    assert new_trig is False

    # Second frame of hello
    cls, conf, new_trig, _ = smoother.update(noise_prob)
    assert cls == "Neutral"
    assert new_trig is False

    # Third frame of hello (satisfies stable_frames_required=3)
    cls, conf, new_trig, _ = smoother.update(noise_prob)
    assert cls == "hello"
    assert new_trig is True  # New trigger fired!

    # Fourth frame of hello (persists stable state, but not a new trigger)
    cls, conf, new_trig, _ = smoother.update(noise_prob)
    assert cls == "hello"
    assert new_trig is False


def test_sentence_builder_sequence_and_cooldown():
    """Verify sentence accumulator, token cooldown, undo, and clear controls."""
    builder = GestureSentenceBuilder(neutral_label="Neutral", cooldown_frames=3)

    assert builder.get_sentence() == ""

    # Add first gesture
    builder.process_gesture_event("hello", is_new_trigger=True)
    assert builder.get_sentence() == "Hello"

    # Continuous trigger of same gesture without neutral should not duplicate
    builder.process_gesture_event("hello", is_new_trigger=True)
    assert builder.get_sentence() == "Hello"

    # User enters neutral pause
    builder.process_gesture_event("Neutral", is_new_trigger=False)

    # User performs second gesture: thank_you
    builder.process_gesture_event("thank_you", is_new_trigger=True)
    assert builder.get_sentence() == "Hello Thank you"

    # Test Undo
    removed = builder.undo_last_token()
    assert removed == "thank_you"
    assert builder.get_sentence() == "Hello"

    # Test Clear
    builder.clear()
    assert builder.get_sentence() == ""
