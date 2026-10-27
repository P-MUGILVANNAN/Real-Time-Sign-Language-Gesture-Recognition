"""
Dataset loader module supporting:
1. Custom collected sign dataset via manifest CSV and .npy sequences.
2. Public dataset integration (WLASL index parser and converter).
3. Synthetic benchmark sequence generator strictly for unit/integration testing.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.preprocessing import PreprocessingPipeline
from src.utils import ensure_dir, setup_logger

logger = setup_logger("DatasetLoader")


class DatasetLoader:
    """
    Unified dataset loading and management for SignVision AI.
    """
    def __init__(
        self,
        custom_data_dir: str = "data/custom",
        manifest_path: str = "data/metadata/dataset_manifest.csv",
        classes: Optional[List[str]] = None,
        sequence_length: int = 30,
        feature_dim: int = 63
    ):
        self.custom_data_dir = Path(custom_data_dir)
        self.manifest_path = Path(manifest_path)
        self.classes = classes or []
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.pipeline = PreprocessingPipeline(
            sequence_length=sequence_length,
            feature_dim=feature_dim,
            classes=self.classes
        )

    def load_custom_dataset(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Loads all validated gesture sequences registered in the manifest CSV.

        Returns:
            X (np.ndarray): Shape (N, sequence_length, feature_dim).
            y (np.ndarray): One-hot encoded labels of shape (N, num_classes).
            metadata_df (pd.DataFrame): Matching metadata (signer_id, background, speed, etc.).
        """
        if not self.manifest_path.is_file():
            logger.warning(f"Manifest not found at {self.manifest_path}. Searching folders...")
            return self._load_from_folders()

        df = pd.read_csv(self.manifest_path)
        if df.empty:
            logger.warning("Manifest CSV is empty.")
            return np.empty((0, self.sequence_length, self.feature_dim)), np.empty((0, len(self.classes))), df

        # Filter by configured classes if specified
        if self.classes:
            df = df[df["gesture_class"].isin(self.classes)].copy()

        sequences = []
        valid_indices = []

        for idx, row in df.iterrows():
            fp = Path(row["filepath"])
            if not fp.is_file():
                # Try relative to custom data dir
                fp = self.custom_data_dir / row["gesture_class"] / f"{row['sample_id']}.npy"

            if fp.is_file():
                try:
                    arr = np.load(fp)
                    # Resample if needed
                    arr = self.pipeline.process_sequence(arr)
                    sequences.append(arr)
                    valid_indices.append(idx)
                except Exception as e:
                    logger.warning(f"Failed to load {fp}: {e}")
            else:
                logger.warning(f"Sample file not found: {fp}")

        if not sequences:
            logger.warning("No valid sequence files could be loaded.")
            return np.empty((0, self.sequence_length, self.feature_dim)), np.empty((0, len(self.classes))), pd.DataFrame()

        matched_df = df.loc[valid_indices].reset_index(drop=True)
        X = np.array(sequences, dtype=np.float32)
        y = self.pipeline.encoder.to_one_hot(matched_df["gesture_class"].tolist())

        logger.info(f"Loaded {len(X)} samples across {len(np.unique(matched_df['gesture_class']))} classes.")
        return X, y, matched_df

    def _load_from_folders(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Fallback loader scanning directories when manifest is missing."""
        if not self.custom_data_dir.is_dir():
            return np.empty((0, self.sequence_length, self.feature_dim)), np.empty((0, len(self.classes))), pd.DataFrame()

        sequences = []
        labels = []
        records = []

        for class_dir in self.custom_data_dir.iterdir():
            if class_dir.is_dir():
                c_name = class_dir.name
                if self.classes and c_name not in self.classes:
                    continue
                for npy_file in class_dir.glob("*.npy"):
                    try:
                        arr = np.load(npy_file)
                        proc = self.pipeline.process_sequence(arr)
                        sequences.append(proc)
                        labels.append(c_name)
                        records.append({
                            "sample_id": npy_file.stem,
                            "gesture_class": c_name,
                            "signer_id": "signer_unknown",
                            "session_id": "session_unknown",
                            "background": "unknown",
                            "speed": "normal",
                            "filepath": str(npy_file)
                        })
                    except Exception as e:
                        logger.warning(f"Error reading {npy_file}: {e}")

        if not sequences:
            return np.empty((0, self.sequence_length, self.feature_dim)), np.empty((0, len(self.classes))), pd.DataFrame()

        X = np.array(sequences, dtype=np.float32)
        y = self.pipeline.encoder.to_one_hot(labels)
        df = pd.DataFrame(records)
        return X, y, df

    @staticmethod
    def parse_wlasl_index(json_path: str, target_glosses: Optional[List[str]] = None) -> List[Dict]:
        """
        Parses the official WLASL_v0.3.json index file.
        Returns a structured list of video metadata entries.
        """
        p = Path(json_path)
        if not p.is_file():
            raise FileNotFoundError(f"WLASL index file not found at: {json_path}")

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = []
        target_set = set(g.lower() for g in target_glosses) if target_glosses else None

        for item in data:
            gloss = item.get("gloss", "").lower()
            if target_set and gloss not in target_set:
                continue

            for instance in item.get("instances", []):
                records.append({
                    "gloss": gloss,
                    "video_id": instance.get("video_id"),
                    "url": instance.get("url"),
                    "signer_id": instance.get("signer_id", "wlasl_signer"),
                    "source": instance.get("source"),
                    "split": instance.get("split"),
                    "frame_start": instance.get("frame_start"),
                    "frame_end": instance.get("frame_end")
                })

        logger.info(f"Parsed {len(records)} WLASL instances for target glosses.")
        return records

    @staticmethod
    def generate_synthetic_benchmark_dataset(
        num_samples_per_class: int = 15,
        classes: Optional[List[str]] = None,
        sequence_length: int = 30,
        feature_dim: int = 63,
        num_signers: int = 3,
        seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Generates realistic synthetic landmark sequences for unit testing and CI pipeline validation.
        Each gesture class has a distinct temporal trajectory (sinusoidal, linear, helical motion).
        Signer variations add individual scale and translation offsets.

        NOTE: This is strictly for software testing and pipeline verification.
        """
        rng = np.random.default_rng(seed)
        classes = classes or ["hello", "thank_you", "yes", "no", "please", "help", "sorry", "iloveyou"]

        sequences = []
        labels = []
        meta_rows = []

        signers = [f"signer_{i+1}" for i in range(num_signers)]
        backgrounds = ["office", "living_room", "plain_white"]
        speeds = ["slow", "normal", "fast"]

        time_steps = np.linspace(0, np.pi, sequence_length)

        for c_idx, class_name in enumerate(classes):
            for s_i in range(num_samples_per_class):
                signer = signers[s_i % num_signers]
                bg = backgrounds[s_i % len(backgrounds)]
                sp = speeds[s_i % len(speeds)]

                # Base gesture pattern: each class has unique harmonic signature
                freq = 1.0 + (c_idx * 0.5)
                phase = (c_idx * np.pi) / len(classes)

                base_trajectory = np.sin(freq * time_steps + phase)[:, None]  # (T, 1)

                # 21 landmarks x 3 coords = 63
                # Random landmark base positions with class temporal oscillation
                landmark_base = rng.normal(loc=0.0, scale=0.3, size=(1, feature_dim)).astype(np.float32)
                # Ensure landmark 0 (wrist) starts at 0
                landmark_base[0, :3] = 0.0

                temporal_modulation = base_trajectory * rng.uniform(0.2, 0.6, size=(1, feature_dim))
                noise = rng.normal(0.0, 0.02, size=(sequence_length, feature_dim))

                # Signer-specific variation (biometric variance)
                signer_bias = (hash(signer) % 100) / 500.0
                seq = landmark_base + temporal_modulation + noise + signer_bias

                # Normalize landmark 0 to 0 (wrist origin)
                for t in range(sequence_length):
                    wrist = seq[t, :3].copy()
                    for lm in range(21):
                        seq[t, lm*3:(lm+1)*3] -= wrist

                sequences.append(seq.astype(np.float32))
                labels.append(class_name)

                sample_id = f"synth_{class_name}_{s_i:03d}"
                meta_rows.append({
                    "sample_id": sample_id,
                    "gesture_class": class_name,
                    "signer_id": signer,
                    "session_id": f"sess_{s_i % 2}",
                    "background": bg,
                    "speed": sp,
                    "timestamp": "2026-01-01T00:00:00",
                    "num_frames": sequence_length,
                    "detection_rate": 1.0,
                    "filepath": f"synthetic/{sample_id}.npy"
                })

        X = np.array(sequences, dtype=np.float32)
        pipeline = PreprocessingPipeline(sequence_length=sequence_length, feature_dim=feature_dim, classes=classes)
        y = pipeline.encoder.to_one_hot(labels)
        df = pd.DataFrame(meta_rows)

        logger.info(f"Generated synthetic benchmark dataset: {len(X)} samples, {len(classes)} classes.")
        return X, y, df
