"""Create the Grand Challenge model resource from four five-fold nnU-Net models.

The script writes checkpoints outside the Git repository. It keeps only the
checkpoint fields required by nnU-Net inference and records SHA-256 digests.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch


MODEL_SPECS = {
    "resencm": (
        "Dataset004_ATLAS3_RAW/"
        "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres",
        (0, 1, 2, 3, 4),
    ),
    "dtk10": (
        "Dataset004_ATLAS3_RAW/"
        "nnUNetTrainerDiceTopK10Loss__nnUNetPlans__3d_fullres",
        (0, 1, 2, 3, 4),
    ),
    "msl": (
        "Dataset005_ATLAS3_RAW_MSL/"
        "nnUNetTrainer__nnUNetPlans__3d_fullres",
        (0, 1, 2, 3, 4),
    ),
    "ici": (
        "Dataset004_ATLAS3_RAW/"
        "nnUNetTrainerICILoss__nnUNetPlans__3d_fullres",
        (0, 1, 2, 3, 4),
    ),
}

CHECKPOINT_KEYS = (
    "network_weights",
    "trainer_name",
    "init_args",
    "inference_allowed_mirroring_axes",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(os.environ.get("nnUNet_results", "nnUNet_results")),
        help="Directory containing the trained nnUNet_results tree.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("ISLES26_MODEL_OUTPUT", "model")),
        help="New model-resource directory to create (default: ./model).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_results = args.results_root.resolve()
    output_root = args.output_root.resolve()
    results_output = output_root / "nnUNet_results"
    if results_output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing packaged models: {results_output}"
        )
    results_output.mkdir(parents=True)

    manifest: dict[str, object] = {
        "files": [],
        "checkpoint_keys": list(CHECKPOINT_KEYS),
        "inference_trainer_overrides": {"ici": "nnUNetTrainer"},
    }
    try:
        for model_name, (relative_root, folds) in MODEL_SPECS.items():
            source_root = source_results / relative_root
            target_root = results_output / relative_root
            print(f"Packaging {model_name} from {source_root}", flush=True)
            target_root.mkdir(parents=True)
            for config_name in ("dataset.json", "plans.json"):
                shutil.copy2(source_root / config_name, target_root / config_name)

            for fold in folds:
                source = source_root / f"fold_{fold}/checkpoint_final.pth"
                target_dir = target_root / f"fold_{fold}"
                target_dir.mkdir()
                target = target_dir / "checkpoint_final.pth"
                temporary = target.with_suffix(".pth.tmp")

                checkpoint = torch.load(source, map_location="cpu", weights_only=False)
                missing = [key for key in CHECKPOINT_KEYS if key not in checkpoint]
                if missing:
                    raise KeyError(f"{source} is missing keys: {missing}")
                slim_checkpoint = {key: checkpoint[key] for key in CHECKPOINT_KEYS}
                # The custom ICI loss is training-only. The network architecture is
                # standard nnU-Net, so the stock trainer is sufficient at inference.
                if model_name == "ici":
                    slim_checkpoint["trainer_name"] = "nnUNetTrainer"
                torch.save(slim_checkpoint, temporary)
                del checkpoint, slim_checkpoint
                gc.collect()
                os.replace(temporary, target)

                item = {
                    "path": target.relative_to(output_root).as_posix(),
                    "bytes": target.stat().st_size,
                    "sha256": sha256(target),
                }
                manifest["files"].append(item)
                print(
                    f"  fold {fold}: {item['bytes']} bytes {item['sha256']}",
                    flush=True,
                )

        # This is the identity/configuration embedded in the uploaded model
        # resource. Final c09 output calibration lives in the container.
        config = {
            "model_order": ["resencm", "dtk10", "msl", "ici"],
            "weights": {"resencm": 0.30, "dtk10": 0.30, "msl": 0.30, "ici": 0.10},
            "folds_by_model": {name: list(folds) for name, (_, folds) in MODEL_SPECS.items()},
            "checkpoint_name": "checkpoint_final.pth",
            "threshold": 0.425,
            "postprocessing": {
                "connectivity": 26,
                "minimum_volume_mm3": 200.0,
                "peak_probability": 0.65,
                "probability_map_is_unfiltered": True,
            },
        }
        (output_root / "ensemble_config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        (output_root / "model_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Packaged model resource at {output_root}", flush=True)
    except Exception:
        shutil.rmtree(results_output, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
