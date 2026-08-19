"""Validate that every packaged fold initializes with nnU-Net 2.8.0."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


RELATIVE_MODEL_FOLDERS = {
    "resencm": (
        "nnUNet_results/Dataset004_ATLAS3_RAW/"
        "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres"
    ),
    "dtk10": (
        "nnUNet_results/Dataset004_ATLAS3_RAW/"
        "nnUNetTrainerDiceTopK10Loss__nnUNetPlans__3d_fullres"
    ),
    "msl": (
        "nnUNet_results/Dataset005_ATLAS3_RAW_MSL/"
        "nnUNetTrainer__nnUNetPlans__3d_fullres"
    ),
    "ici": (
        "nnUNet_results/Dataset004_ATLAS3_RAW/"
        "nnUNetTrainerICILoss__nnUNetPlans__3d_fullres"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path(os.environ.get("ISLES26_MODEL_ROOT", "model")),
    )
    return parser.parse_args()


def main() -> None:
    model_root = parse_args().model_root.resolve()
    config = json.loads((model_root / "ensemble_config.json").read_text(encoding="utf-8"))
    for name, relative_folder in RELATIVE_MODEL_FOLDERS.items():
        folder = model_root / relative_folder
        predictor = nnUNetPredictor(
            perform_everything_on_device=False,
            device=torch.device("cpu"),
            verbose=False,
            allow_tqdm=False,
        )
        folds = tuple(config["folds_by_model"][name])
        predictor.initialize_from_trained_model_folder(
            str(folder),
            use_folds=folds,
            checkpoint_name=config["checkpoint_name"],
        )
        parameter_sets = len(predictor.list_of_parameters)
        parameter_count = sum(
            value.numel() for value in predictor.list_of_parameters[0].values()
        )
        print(
            f"OK {name}: trainer={predictor.trainer_name}, folds={parameter_sets}, "
            f"parameters_per_fold={parameter_count}",
            flush=True,
        )
        if parameter_sets != len(folds):
            raise AssertionError(f"{name} did not load all configured folds")
        del predictor
        gc.collect()


if __name__ == "__main__":
    main()
