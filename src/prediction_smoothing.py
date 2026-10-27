"""
Prediction smoothing, temporal debouncing, and confidence thresholding engine.
Prevents rapid visual flicker and false positives in real-time gesture streams.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.preprocessing import GestureLabelEncoder
from src.utils import setup_logger

logger = setup_logger("PredictionSmoothing")


class PredictionSmoother:
    """
    Applies probability averaging and consecutive stability gating to raw model outputs.
    """
    def __init__(
        self,
        encoder: GestureLabelEncoder,
        smoothing_window_size: int = 5,
        confidence_threshold: float = 0.70,
        stable_frames_required: int = 4,
        unknown_label: str = "No confident gesture detected"
    ):
        self.encoder = encoder
        self.window_size = smoothing_window_size
        self.confidence_threshold = confidence_threshold
        self.stable_frames_required = stable_frames_required
        self.unknown_label = unknown_label

        # Rolling deque for recent softmax probability vectors
        self.prob_history: deque = deque(maxlen=smoothing_window_size)

        # State tracking for debouncing
        self.candidate_class: Optional[str] = None
        self.candidate_counter: int = 0
        self.active_stable_class: str = unknown_label
        self.active_confidence: float = 0.0

    def update(self, raw_probabilities: np.ndarray) -> Tuple[str, float, bool, np.ndarray]:
        """
        Updates the smoother with a new frame's probability distribution.

        Returns:
            stable_class (str): The current stable prediction or unknown state.
            confidence (float): The smoothed confidence score [0.0, 1.0].
            is_new_trigger (bool): True on the exact frame the stable prediction changes.
            smoothed_probs (np.ndarray): The averaged probability distribution.
        """
        raw_probs = np.asarray(raw_probabilities, dtype=np.float32).flatten()
        self.prob_history.append(raw_probs)

        # 1. Compute rolling mean probability across window
        smoothed_probs = np.mean(self.prob_history, axis=0)

        best_idx = int(np.argmax(smoothed_probs))
        best_conf = float(smoothed_probs[best_idx])

        # 2. Confidence Thresholding
        if best_conf < self.confidence_threshold:
            instant_class = self.unknown_label
        else:
            instant_class = self.encoder.inverse_transform_idx(best_idx)

        # 3. Temporal Stability Gating (Debouncing)
        is_new_trigger = False
        if instant_class == self.candidate_class:
            self.candidate_counter += 1
        else:
            self.candidate_class = instant_class
            self.candidate_counter = 1

        # Check if candidate has persisted for enough consecutive frames
        if self.candidate_counter >= self.stable_frames_required:
            if self.candidate_class != self.active_stable_class:
                self.active_stable_class = self.candidate_class
                if self.active_stable_class != self.unknown_label:
                    is_new_trigger = True
            self.active_confidence = best_conf
        else:
            # If not yet stabilized on a new gesture, retain current active class or default to candidate conf
            self.active_confidence = best_conf

        return self.active_stable_class, self.active_confidence, is_new_trigger, smoothed_probs

    def reset(self):
        """Resets smoother state when camera restarts or hand leaves frame."""
        self.prob_history.clear()
        self.candidate_class = None
        self.candidate_counter = 0
        self.active_stable_class = self.unknown_label
        self.active_confidence = 0.0
