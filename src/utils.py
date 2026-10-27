"""
Utility functions for SignVision AI.
Provides logging configuration, configuration loaders, directory checkers, and timing helpers.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


def setup_logger(name: str = "SignVisionAI", log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Configures and returns a standardized logger with console and optional file handlers.
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Loads YAML configuration with fallback defaults.
    """
    p = Path(config_path)
    if not p.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(p, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_classes(classes_path: str = "config/classes.json") -> List[str]:
    """
    Loads the list of supported gesture classes.
    """
    p = Path(classes_path)
    if not p.is_file():
        raise FileNotFoundError(f"Classes configuration file not found at: {classes_path}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("classes", [])


def ensure_dir(dir_path: str) -> Path:
    """
    Ensures that a directory exists, creating it if necessary.
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


class FPSCounter:
    """
    Helper class to calculate smooth frames-per-second (FPS) over a rolling window.
    """
    def __init__(self, window_size: int = 15):
        self.window_size = window_size
        self.timestamps: List[float] = []

    def update(self) -> float:
        now = time.perf_counter()
        self.timestamps.append(now)
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)

        if len(self.timestamps) < 2:
            return 0.0

        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.timestamps) - 1) / elapsed


class LatencyTimer:
    """
    Context manager to measure execution time in milliseconds.
    """
    def __init__(self):
        self.elapsed_ms: float = 0.0
        self._start_time: float = 0.0

    def __enter__(self):
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.perf_counter() - self._start_time) * 1000.0
