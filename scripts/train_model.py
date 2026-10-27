"""
CLI tool for training the SignVision LSTM deep learning model.
Usage:
    python scripts/train_model.py [--epochs 50] [--batch_size 16]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train import run_training_pipeline
from src.utils import setup_logger

logger = setup_logger("TrainModelCLI")


def main():
    parser = argparse.ArgumentParser(description="Train SignVision LSTM Gesture Recognition Model.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to YAML config.")
    parser.add_argument("--classes", type=str, default="config/classes.json", help="Path to classes JSON.")
    args = parser.parse_args()

    logger.info("Starting model training pipeline...")
    results = run_training_pipeline(
        config_path=args.config,
        classes_path=args.classes,
        use_synthetic_if_empty=True
    )
    logger.info("Training script executed successfully.")


if __name__ == "__main__":
    main()
