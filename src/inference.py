"""
Real-time inference engine for SignVision AI.
Maintains a rolling sequence buffer, handles live video frames, performs LSTM forward pass,
and invokes prediction smoothing and sentence building.
"""

from collections import deque
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import tensorflow as tf

from src.landmark_extractor import LandmarkExtractor
from src.model_lstm import load_trained_model
from src.prediction_smoothing import PredictionSmoother
from src.preprocessing import GestureLabelEncoder
from src.sentence_builder import GestureSentenceBuilder
from src.utils import load_classes, load_config, setup_logger

logger = setup_logger("InferenceEngine")


class RealTimeInferenceEngine:
    """
    Core runtime engine linking camera frame input to smoothed gesture predictions.
    """
    def __init__(
        self,
        model_path: str = "models/signvision_lstm_best.keras",
        config_path: str = "config/config.yaml",
        classes_path: str = "config/classes.json",
        extractor: Optional[LandmarkExtractor] = None
    ):
        self.config = load_config(config_path)
        self.classes = load_classes(classes_path)
        self.encoder = GestureLabelEncoder(self.classes)

        self.seq_len = self.config["features"]["sequence_length"]
        self.feat_dim = self.config["features"]["single_hand_features"]
        self.conf_thresh = self.config["inference"]["confidence_threshold"]
        self.smoothing_win = self.config["inference"]["smoothing_window_size"]
        self.stable_frames = self.config["inference"]["stable_frames_required"]

        # Landmark Extractor
        self.extractor = extractor or LandmarkExtractor(
            mode=self.config["features"]["mode"],
            num_hands=2 if self.config["features"]["mode"] == "two_hand" else 1
        )

        # Rolling Buffer: Holds the last `seq_len` landmark vectors (each of length `feat_dim`)
        self.sequence_buffer: deque = deque(maxlen=self.seq_len)

        # Prediction Smoother
        self.smoother = PredictionSmoother(
            encoder=self.encoder,
            smoothing_window_size=self.smoothing_win,
            confidence_threshold=self.conf_thresh,
            stable_frames_required=self.stable_frames,
            unknown_label=self.config["inference"]["unknown_class_label"]
        )

        # Sentence Builder
        self.sentence_builder = GestureSentenceBuilder(
            neutral_label=self.config["inference"]["unknown_class_label"]
        )

        # Load Deep Learning Model
        self.model_path = model_path
        self.model: Optional[tf.keras.Model] = None
        self._load_model_safe()

    def _load_model_safe(self):
        """Attempts to load the trained model; gracefully alerts if missing."""
        if os.path.isfile(self.model_path):
            try:
                self.model = load_trained_model(self.model_path)
                logger.info(f"Inference engine successfully loaded model from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed loading model from {self.model_path}: {e}")
                self.model = None
        else:
            logger.warning(
                f"Model file not found at {self.model_path}. "
                "Inference engine will operate in landmark-visualization mode until trained."
            )

    def is_model_loaded(self) -> bool:
        return self.model is not None

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        draw_landmarks: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Processes a single live camera frame:
        1. Extracts normalized landmarks.
        2. Renders skeleton overlay (optional).
        3. Appends landmarks to rolling buffer.
        4. Runs LSTM classification when buffer is primed.
        5. Returns annotated frame and inference payload.
        """
        t_start = time.perf_counter()

        # Step 1: Landmark Extraction
        features, hand_detected, info = self.extractor.extract_from_frame(frame_bgr)

        # Step 2: Overlay
        annotated_frame = frame_bgr
        if draw_landmarks and hand_detected:
            annotated_frame = self.extractor.draw_landmarks_on_image(frame_bgr, info)

        # Step 3: Sequence Buffer Management
        self.sequence_buffer.append(features)
        buffer_len = len(self.sequence_buffer)
        buffer_fill_pct = float(buffer_len / self.seq_len)

        result_payload = {
            "hand_detected": hand_detected,
            "buffer_fill_pct": buffer_fill_pct,
            "buffer_frames": buffer_len,
            "gesture_class": self.config["inference"]["unknown_class_label"],
            "confidence": 0.0,
            "latency_ms": 0.0,
            "probabilities": {},
            "sentence": self.sentence_builder.get_sentence(),
            "model_ready": self.is_model_loaded()
        }

        # If model is not loaded or buffer not yet full, return baseline status
        if not self.is_model_loaded() or buffer_len < self.seq_len:
            result_payload["latency_ms"] = (time.perf_counter() - t_start) * 1000.0
            return annotated_frame, result_payload

        # Step 4: Model Inference
        # Input tensor shape: (1, seq_len, feat_dim)
        input_seq = np.expand_dims(np.array(self.sequence_buffer, dtype=np.float32), axis=0)

        # If hand not detected in recent frame, dampen probabilities
        if not hand_detected:
            self.smoother.reset()
            result_payload["latency_ms"] = (time.perf_counter() - t_start) * 1000.0
            return annotated_frame, result_payload

        t_infer_start = time.perf_counter()
        raw_probs = self.model(input_seq, training=False).numpy()[0]
        infer_latency_ms = (time.perf_counter() - t_infer_start) * 1000.0

        # Step 5: Smoothing and Debouncing
        stable_class, conf, is_new_trigger, smoothed_probs = self.smoother.update(raw_probs)

        # Step 6: Sentence building
        self.sentence_builder.process_gesture_event(stable_class, is_new_trigger)

        # Format probability distribution
        prob_dict = {
            cls: float(smoothed_probs[i])
            for i, cls in enumerate(self.classes)
            if i < len(smoothed_probs)
        }

        result_payload.update({
            "gesture_class": stable_class,
            "confidence": round(float(conf), 4),
            "latency_ms": round(infer_latency_ms, 2),
            "total_frame_ms": round((time.perf_counter() - t_start) * 1000.0, 2),
            "probabilities": prob_dict,
            "sentence": self.sentence_builder.get_sentence(),
            "is_new_trigger": is_new_trigger
        })

        return annotated_frame, result_payload

    def reset_buffer(self):
        """Clears rolling history and resets smoothing."""
        self.sequence_buffer.clear()
        self.smoother.reset()
