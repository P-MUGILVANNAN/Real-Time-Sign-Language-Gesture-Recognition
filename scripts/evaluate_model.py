"""
CLI tool for evaluating the trained SignVision model on test partitions.
Usage:
    python scripts/evaluate_model.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import evaluate_saved_model
from src.utils import setup_logger

logger = setup_logger("EvaluateModelCLI")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SignVision Model.")
    parser.add_argument("--model", type=str, default="models/signvision_lstm_best.keras", help="Path to saved Keras model.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to YAML config.")
    parser.add_argument("--classes", type=str, default="config/classes.json", help="Path to classes JSON.")
    args = parser.parse_args()

    logger.info("Executing model evaluation...")
    results = evaluate_saved_model(
        model_path=args.model,
        config_path=args.config,
        classes_path=args.classes
    )
    logger.info("Evaluation completed. Check artifacts/reports/evaluation_report.md for results.")


if __name__ == "__main__":
    main()
