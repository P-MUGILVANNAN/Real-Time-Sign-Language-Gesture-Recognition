"""
CLI tool for running the scientific research experiments (Experiments A, D, E, F).
Usage:
    python scripts/run_experiments.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_runner import ExperimentRunner
from src.utils import setup_logger

logger = setup_logger("RunExperimentsCLI")


def main():
    logger.info("Starting Research Experiments execution...")
    runner = ExperimentRunner()
    results = runner.run_all_experiments()
    logger.info("All experiments concluded. Output CSVs and charts stored in artifacts/.")


if __name__ == "__main__":
    main()
