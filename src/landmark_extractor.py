"""
Hand landmark extraction module using MediaPipe Tasks HandLandmarker API.
Extracts 21 landmarks per hand (x, y, z), handles handedness, and applies
wrist-relative translation and scale-invariant normalization.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from src.utils import setup_logger

logger = setup_logger("LandmarkExtractor")

# Standard MediaPipe Hand Connections (Pairs of landmark indices)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]


def normalize_hand_landmarks(landmarks_21x3: np.ndarray) -> np.ndarray:
    """
    Normalizes 21 hand landmarks (shape: [21, 3]) for scale and translation invariance.

    Mathematical steps:
    1. Wrist centering: P_i' = P_i - P_0 (wrist landmark 0 becomes origin [0, 0, 0]).
    2. Reference scale calculation: Euclidean distance from wrist (0) to middle finger MCP (landmark 9).
       If reference distance is too small (< 1e-5), fallback to max distance from wrist.
    3. Scale normalization: P_i'' = P_i' / scale.

    Returns:
        np.ndarray: Normalized 1D feature vector of length 63.
    """
    if landmarks_21x3.shape != (21, 3):
        raise ValueError(f"Expected landmarks array of shape (21, 3), got {landmarks_21x3.shape}")

    # Check for NaN / Inf
    if not np.isfinite(landmarks_21x3).all():
        return np.zeros(63, dtype=np.float32)

    # If all zeros (no hand detected), return zeros
    if np.all(landmarks_21x3 == 0):
        return np.zeros(63, dtype=np.float32)

    # 1. Translation: Wrist (index 0) as origin
    wrist = landmarks_21x3[0, :].copy()
    translated = landmarks_21x3 - wrist

    # 2. Scale factor: Distance from wrist to middle finger MCP (index 9)
    middle_mcp = translated[9, :]
    scale = float(np.linalg.norm(middle_mcp))

    if scale < 1e-5:
        # Fallback to maximum landmark distance from wrist
        distances = np.linalg.norm(translated, axis=1)
        scale = float(np.max(distances))

    if scale < 1e-5:
        scale = 1.0

    # 3. Scale normalization
    normalized = translated / scale

    return normalized.flatten().astype(np.float32)


class LandmarkExtractor:
    """
    High-performance MediaPipe HandLandmarker wrapper for real-time video and image processing.
    """
    def __init__(
        self,
        model_path: str = "models/hand_landmarker.task",
        num_hands: int = 2,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        mode: str = "single_hand"  # "single_hand" (63 features) or "two_hand" (126 features)
    ):
        self.mode = mode
        self.num_hands = num_hands
        self.expected_features = 63 if mode == "single_hand" else 126
        self.model_path = model_path
        self._detector: Optional[vision.HandLandmarker] = None

        self._initialize_detector(
            min_hand_detection_confidence,
            min_hand_presence_confidence,
            min_tracking_confidence
        )

    def _initialize_detector(
        self,
        min_det_conf: float,
        min_pres_conf: float,
        min_track_conf: float
    ):
        if not os.path.exists(self.model_path):
            logger.warning(f"MediaPipe model file not found at {self.model_path}. Attempting download...")
            import urllib.request
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, self.model_path)
            logger.info(f"Model downloaded successfully to {self.model_path}")

        base_options = mp_python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=self.num_hands,
            min_hand_detection_confidence=min_det_conf,
            min_hand_presence_confidence=min_pres_conf,
            min_tracking_confidence=min_track_conf
        )
        self._detector = vision.HandLandmarker.create_from_options(options)
        logger.info(f"MediaPipe HandLandmarker initialized in {self.mode} mode.")

    def extract_from_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, bool, Dict]:
        """
        Processes a single BGR OpenCV frame and extracts normalized landmark features.

        Returns:
            features (np.ndarray): 1D feature array of shape (63,) or (126,).
            hand_detected (bool): True if at least one hand was detected.
            info (dict): Auxiliary metadata (raw landmarks, handedness, bounding boxes).
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return np.zeros(self.expected_features, dtype=np.float32), False, {}

        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        detection_result = self._detector.detect(mp_image)
        has_hands = bool(detection_result.hand_landmarks)

        info = {
            "num_hands": len(detection_result.hand_landmarks) if has_hands else 0,
            "handedness": [],
            "raw_landmarks": []
        }

        if not has_hands:
            return np.zeros(self.expected_features, dtype=np.float32), False, info

        # Parse handedness
        if detection_result.handedness:
            for h_list in detection_result.handedness:
                if h_list:
                    info["handedness"].append(h_list[0].category_name)

        # Parse raw coordinates
        raw_hands = []
        for hand in detection_result.hand_landmarks:
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
            raw_hands.append(coords)
        info["raw_landmarks"] = raw_hands

        if self.mode == "single_hand":
            # Select dominant hand (first detected hand)
            primary_hand = raw_hands[0]
            normalized = normalize_hand_landmarks(primary_hand)
            return normalized, True, info
        else:
            # Two-hand schema: Left hand (63) + Right hand (63)
            left_feats = np.zeros(63, dtype=np.float32)
            right_feats = np.zeros(63, dtype=np.float32)

            for i, hand_coords in enumerate(raw_hands):
                hand_type = info["handedness"][i] if i < len(info["handedness"]) else ("Left" if i == 0 else "Right")
                norm_hand = normalize_hand_landmarks(hand_coords)
                if hand_type == "Left":
                    left_feats = norm_hand
                else:
                    right_feats = norm_hand

            combined = np.concatenate([left_feats, right_feats]).astype(np.float32)
            return combined, True, info

    def draw_landmarks_on_image(
        self,
        frame_bgr: np.ndarray,
        detection_info: Dict,
        landmark_color: Tuple[int, int, int] = (0, 215, 255),      # Gold
        connection_color: Tuple[int, int, int] = (255, 128, 0)     # Neon Cyan / Electric Blue
    ) -> np.ndarray:
        """
        Renders aesthetic hand skeleton overlays onto the given OpenCV frame.
        """
        output_frame = frame_bgr.copy()
        raw_landmarks = detection_info.get("raw_landmarks", [])
        if not raw_landmarks:
            return output_frame

        h, w, _ = output_frame.shape

        for hand_coords in raw_landmarks:
            pixel_pts = []
            for pt in hand_coords:
                px = int(pt[0] * w)
                py = int(pt[1] * h)
                pixel_pts.append((px, py))

            # Draw bones / connections
            for p1_idx, p2_idx in HAND_CONNECTIONS:
                p1 = pixel_pts[p1_idx]
                p2 = pixel_pts[p2_idx]
                cv2.line(output_frame, p1, p2, connection_color, 2, cv2.LINE_AA)

            # Draw joints / landmark dots
            for idx, (px, py) in enumerate(pixel_pts):
                radius = 5 if idx in (0, 4, 8, 12, 16, 20) else 3
                cv2.circle(output_frame, (px, py), radius, landmark_color, -1, cv2.LINE_AA)
                cv2.circle(output_frame, (px, py), radius + 1, (20, 20, 20), 1, cv2.LINE_AA)

        return output_frame

    def close(self):
        """Releases detector resources cleanly."""
        if self._detector:
            try:
                self._detector.close()
            except Exception:
                pass
            self._detector = None
