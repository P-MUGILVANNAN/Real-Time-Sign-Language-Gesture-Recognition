"""
CLI tool for preparing and validating the dataset manifest and sequence arrays.
Usage:
    python scripts/prepare_dataset.py [--generate_benchmark]
"""

import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_loader import DatasetLoader
from src.utils import ensure_dir, load_classes, load_config, setup_logger

logger = setup_logger("PrepareDatasetCLI")


def main():
    parser = argparse.ArgumentParser(description="Prepares, validates, and reports on the dataset.")
    parser.add_argument("--generate_benchmark", action="store_true", help="Generate synthetic benchmark sequences if custom data is empty.")
    parser.add_argument("--samples_per_class", type=int, default=25, help="Number of benchmark samples per class.")
    args = parser.parse_args()

    config = load_config("config/config.yaml")
    classes = load_classes("config/classes.json")

    loader = DatasetLoader(
        custom_data_dir=config["data"]["custom_dir"],
        manifest_path=config["data"]["metadata_csv"],
        classes=classes,
        sequence_length=config["features"]["sequence_length"],
        feature_dim=config["features"]["single_hand_features"]
    )

    X, y, df = loader.load_custom_dataset()

    if len(X) == 0:
        if args.generate_benchmark:
            logger.info("Generating synthetic benchmark sequences for all configured classes...")
            X, y, df = DatasetLoader.generate_synthetic_benchmark_dataset(
                num_samples_per_class=args.samples_per_class,
                classes=classes,
                num_signers=4
            )
            # Save to custom dataset directory so it persists
            custom_dir = Path(config["data"]["custom_dir"])
            for idx, row in df.iterrows():
                class_folder = ensure_dir(str(custom_dir / row["gesture_class"]))
                np.save(class_folder / f"{row['sample_id']}.npy", X[idx])
                row["filepath"] = str(class_folder / f"{row['sample_id']}.npy")

            ensure_dir(str(Path(config["data"]["metadata_csv"]).parent))
            df.to_csv(config["data"]["metadata_csv"], index=False)
            logger.info(f"Persisted {len(X)} benchmark samples and updated {config['data']['metadata_csv']}")
        else:
            logger.warning("No data found. Use --generate_benchmark or record custom gestures using the web app.")
            return

    # Export processed arrays
    processed_dir = ensure_dir(config["data"]["processed_dir"])
    np.save(processed_dir / "X_processed.npy", X)
    np.save(processed_dir / "y_processed.npy", y)
    logger.info(f"Saved processed feature tensor X: {X.shape} and label matrix y: {y.shape} to {processed_dir}")


if __name__ == "__main__":
    main()
