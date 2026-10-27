"""
Video processing module for offline and batch video landmark extraction.
Extracts landmark sequences from video clips (e.g., MP4/AVI), handles missing frames,
and resamples sequences to standard length.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from tqdm import tqdm

from src.landmark_extractor import LandmarkExtractor
from src.preprocessing import resample_sequence
from src.utils import setup_logger

logger = setup_logger("VideoProcessor")


class VideoProcessor:
    """
    Extracts temporal hand landmark sequences from video files using LandmarkExtractor.
    """
    def __init__(
        self,
        extractor: Optional[LandmarkExtractor] = None,
        target_sequence_length: int = 30,
        feature_dim: int = 63
    ):
        self.extractor = extractor or LandmarkExtractor()
        self.target_length = target_sequence_length
        self.feature_dim = feature_dim

    def process_video_file(
        self,
        video_path: str,
        resample: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reads a video file, extracts landmarks for every frame, and returns a sequence array.

        Returns:
            sequence (np.ndarray): (target_length, feature_dim) if resample=True else (num_frames, feature_dim).
            metadata (dict): Extraction statistics (fps, total_frames, hands_detected_count, detection_rate).
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        raw_sequence = []
        detected_frames = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            features, detected, _ = self.extractor.extract_from_frame(frame)
            raw_sequence.append(features)
            if detected:
                detected_frames += 1
            frame_idx += 1

        cap.release()

        actual_frames = len(raw_sequence)
        detection_rate = (detected_frames / actual_frames) if actual_frames > 0 else 0.0

        stats = {
            "video_path": str(video_path),
            "video_fps": float(fps),
            "total_frames": int(total_frames),
            "extracted_frames": int(actual_frames),
            "detected_frames": int(detected_frames),
            "detection_rate": float(detection_rate)
        }

        if actual_frames == 0:
            logger.warning(f"No frames could be extracted from {video_path}")
            empty_seq = np.zeros((self.target_length if resample else 0, self.feature_dim), dtype=np.float32)
            return empty_seq, stats

        seq_arr = np.array(raw_sequence, dtype=np.float32)

        if resample:
            seq_arr = resample_sequence(seq_arr, target_length=self.target_length)

        return seq_arr, stats

    def batch_process_directory(
        self,
        input_dir: str,
        output_dir: str,
        label: str,
        video_extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")
    ) -> List[Dict]:
        """
        Processes all video files in a directory and saves .npy landmark sequences.
        """
        in_path = Path(input_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        video_files = [f for f in in_path.iterdir() if f.suffix.lower() in video_extensions]
        logger.info(f"Found {len(video_files)} videos in {input_dir} for class '{label}'.")

        manifest = []
        for v_file in tqdm(video_files, desc=f"Extracting {label}"):
            try:
                seq, stats = self.process_video_file(str(v_file), resample=True)
                save_name = f"{v_file.stem}_landmarks.npy"
                save_dest = out_path / save_name
                np.save(save_dest, seq)

                stats["npy_path"] = str(save_dest)
                stats["label"] = label
                manifest.append(stats)
            except Exception as e:
                logger.error(f"Error processing video {v_file.name}: {e}")

        return manifest
