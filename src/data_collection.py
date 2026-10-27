"""
Custom webcam dataset recording and management module.
Captures gesture samples, extracts landmarks frame-by-frame, validates sequences,
and records comprehensive metadata (signer, background, speed, session).
"""

import datetime
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd

from src.landmark_extractor import LandmarkExtractor
from src.preprocessing import resample_sequence
from src.utils import ensure_dir, setup_logger

logger = setup_logger("DataCollection")


class DataCollector:
    """
    Manages recording of custom sign language landmark sequences from webcam or frames.
    """
    def __init__(
        self,
        base_dir: str = "data/custom",
        manifest_path: str = "data/metadata/dataset_manifest.csv",
        sequence_length: int = 30,
        feature_dim: int = 63,
        extractor: Optional[LandmarkExtractor] = None
    ):
        self.base_dir = ensure_dir(base_dir)
        self.manifest_path = Path(manifest_path)
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.extractor = extractor or LandmarkExtractor()
        self._ensure_manifest()

    def _ensure_manifest(self):
        """Ensures the dataset manifest CSV exists with required columns."""
        ensure_dir(str(self.manifest_path.parent))
        if not self.manifest_path.is_file():
            df = pd.DataFrame(columns=[
                "sample_id",
                "gesture_class",
                "signer_id",
                "session_id",
                "background",
                "speed",
                "timestamp",
                "num_frames",
                "detection_rate",
                "filepath"
            ])
            df.to_csv(self.manifest_path, index=False)
            logger.info(f"Initialized empty dataset manifest at {self.manifest_path}")

    def save_sequence_sample(
        self,
        sequence_frames: List[np.ndarray],
        gesture_class: str,
        signer_id: str = "signer_1",
        session_id: str = "session_1",
        background: str = "neutral",
        speed: str = "normal",
        detection_flags: Optional[List[bool]] = None
    ) -> Tuple[bool, str, Dict]:
        """
        Validates, resamples, and persists a recorded landmark sequence sample.

        Returns:
            success (bool): Whether the sample passed validation and was saved.
            message (str): Human-readable status message.
            metadata (dict): The recorded sample metadata.
        """
        if not sequence_frames or len(sequence_frames) < 5:
            return False, f"Sequence too short ({len(sequence_frames)} frames). At least 5 frames required.", {}

        # Convert to numpy array (T, D)
        seq_array = np.array(sequence_frames, dtype=np.float32)

        # Calculate detection rate
        if detection_flags:
            det_rate = sum(detection_flags) / len(detection_flags)
        else:
            # Check non-zero rows
            non_zeros = np.sum(np.any(seq_array != 0, axis=1))
            det_rate = non_zeros / len(seq_array)

        if det_rate < 0.20:
            return False, f"Hand detection rate too low ({det_rate:.1%}). Ensure hands are clearly visible.", {}

        # Resample to fixed target length
        resampled_seq = resample_sequence(seq_array, target_length=self.sequence_length)

        # Unique ID and path
        sample_id = f"{gesture_class}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        class_folder = ensure_dir(str(self.base_dir / gesture_class))
        save_file = class_folder / f"{sample_id}.npy"

        np.save(save_file, resampled_seq)

        metadata = {
            "sample_id": sample_id,
            "gesture_class": gesture_class,
            "signer_id": signer_id,
            "session_id": session_id,
            "background": background,
            "speed": speed,
            "timestamp": datetime.datetime.now().isoformat(),
            "num_frames": int(len(sequence_frames)),
            "detection_rate": round(float(det_rate), 4),
            "filepath": str(save_file)
        }

        # Append to manifest CSV
        df = pd.read_csv(self.manifest_path)
        new_row = pd.DataFrame([metadata])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(self.manifest_path, index=False)

        logger.info(f"Saved gesture '{gesture_class}' sample '{sample_id}' (Signer: {signer_id}).")
        return True, f"Successfully recorded sample '{sample_id}'", metadata

    def get_dataset_statistics(self) -> Dict:
        """
        Returns class distribution and dataset breakdown from the manifest.
        """
        self._ensure_manifest()
        df = pd.read_csv(self.manifest_path)
        if df.empty:
            return {
                "total_samples": 0,
                "class_counts": {},
                "signer_counts": {},
                "background_counts": {},
                "speed_counts": {}
            }

        return {
            "total_samples": len(df),
            "class_counts": df["gesture_class"].value_counts().to_dict(),
            "signer_counts": df["signer_id"].value_counts().to_dict(),
            "background_counts": df["background"].value_counts().to_dict(),
            "speed_counts": df["speed"].value_counts().to_dict()
        }

    def delete_sample(self, sample_id: str) -> Tuple[bool, str]:
        """
        Removes a sample from disk and updates the manifest.
        """
        self._ensure_manifest()
        df = pd.read_csv(self.manifest_path)
        target = df[df["sample_id"] == sample_id]

        if target.empty:
            return False, f"Sample '{sample_id}' not found in manifest."

        file_path = target.iloc[0]["filepath"]
        if os.path.exists(file_path):
            os.remove(file_path)

        df = df[df["sample_id"] != sample_id]
        df.to_csv(self.manifest_path, index=False)
        logger.info(f"Deleted sample {sample_id} from disk and manifest.")
        return True, f"Sample '{sample_id}' successfully deleted."
