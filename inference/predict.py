"""Command-line entry point for local single-case inference."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from inference.ensemble import initialize_predictors, load_calibration, predict_case


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the final four-family, five-fold ISLES'26 ensemble on one "
            "T1 MRI volume."
        )
    )
    parser.add_argument("--input-image", required=True, help="Input .nii, .nii.gz, or .mha T1 image")
    parser.add_argument(
        "--results-root",
        default=os.environ.get("nnUNet_results"),
        help="nnUNet_results directory (defaults to the nnUNet_results environment variable)",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for both output volumes")
    parser.add_argument(
        "--config",
        default=str(REPOSITORY_ROOT / "configs" / "final_output_calibration.json"),
        help="Frozen final calibration JSON",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="checkpoint_final.pth",
        help="Checkpoint filename within each fold directory",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or an explicit device such as cuda:1",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Optional directory for temporary nnU-Net probability files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.results_root:
        raise SystemExit("Set nnUNet_results or pass --results-root")
    calibration = load_calibration(args.config)
    predictors, device = initialize_predictors(
        args.results_root,
        calibration,
        checkpoint_name=args.checkpoint_name,
        device=args.device,
    )
    predict_case(
        args.input_image,
        args.output_dir,
        predictors,
        calibration,
        device,
        work_dir=args.work_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
