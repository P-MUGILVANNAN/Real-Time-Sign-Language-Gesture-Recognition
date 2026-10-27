"""
CLI tool for batch extracting hand landmark sequences from video directories.
Usage:
    python scripts/extract_landmarks.py --input_dir data/raw/videos/hello --output_dir data/custom/hello --label hello
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.video_processor import VideoProcessor
from src.utils import setup_logger

logger = setup_logger("ExtractLandmarksCLI")


def main():
    parser = argparse.ArgumentParser(description="Extract hand landmarks from videos into .npy sequences.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing video files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save extracted .npy sequences.")
    parser.add_argument("--label", type=str, required=True, help="Gesture class label for the videos.")
    parser.add_argument("--target_length", type=int, default=30, help="Resampled sequence length (default 30).")

    args = parser.parse_args()

    processor = VideoProcessor(target_sequence_length=args.target_length)
    logger.info(f"Extracting landmarks from {args.input_dir} for gesture '{args.label}'...")
    manifest = processor.batch_process_directory(args.input_dir, args.output_dir, args.label)
    logger.info(f"Extraction completed. {len(manifest)} sequences saved to {args.output_dir}.")


if __name__ == "__main__":
    main()
