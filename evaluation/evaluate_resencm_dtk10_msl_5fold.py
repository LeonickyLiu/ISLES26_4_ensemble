#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from itertools import permutations
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch


BASE = Path(
    os.environ.get("ISLES26_ROOT", Path(__file__).resolve().parents[1])
).resolve()
GT_DIR = BASE / "nnUNet_raw/Dataset004_ATLAS3_RAW/labelsTr"
MODEL_ROOTS = {
    "resencm": BASE
    / "nnUNet_results/Dataset004_ATLAS3_RAW/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres",
    "dtk10": BASE
    / "nnUNet_results/Dataset004_ATLAS3_RAW/nnUNetTrainerDiceTopK10Loss__nnUNetPlans__3d_fullres",
    "msl": BASE
    / "nnUNet_results/Dataset005_ATLAS3_RAW_MSL/nnUNetTrainer__nnUNetPlans__3d_fullres",
}
OUT_DIR = BASE / "ensemble_results/resencm_dtk10_msl_5fold"
CHECKPOINT_PATH = OUT_DIR / "progress.npz"
THRESHOLDS = np.round(np.arange(0.30, 0.601, 0.025), 3).astype(np.float32)
PRIMARY_THRESHOLD = 0.35
SIZE_BINS = (
    ("tiny_lt100", 0, 100),
    ("small_100_999", 100, 1000),
    ("medium_1000_9999", 1000, 10000),
    ("large_ge10000", 10000, None),
)


def case_id(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    return path.stem


def files_by_case(folder: Path, pattern: str) -> dict[str, Path]:
    return {case_id(path): path for path in folder.glob(pattern)}


def load_gt(path: Path) -> np.ndarray:
    return np.asarray(sitk.GetArrayFromImage(sitk.ReadImage(str(path)))) > 0


def align_probabilities(
    probabilities: np.ndarray, reference_shape: tuple[int, ...], path: Path
) -> np.ndarray:
    if probabilities.ndim != 4:
        raise ValueError(f"Expected [C, spatial] in {path}, got {probabilities.shape}")
    spatial_shape = tuple(int(value) for value in probabilities.shape[1:])
    reference_shape = tuple(int(value) for value in reference_shape)
    if spatial_shape == reference_shape:
        return probabilities

    matches = [
        perm
        for perm in permutations(range(3))
        if tuple(spatial_shape[index] for index in perm) == reference_shape
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Cannot uniquely align {probabilities.shape} to {reference_shape} in {path}"
        )
    return np.transpose(
        probabilities, (0,) + tuple(index + 1 for index in matches[0])
    )


def load_foreground_probability(
    path: Path, reference_shape: tuple[int, ...]
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "probabilities" in data.files:
            probabilities = data["probabilities"]
        elif "softmax" in data.files:
            probabilities = data["softmax"]
        else:
            probabilities = data[data.files[0]]

    probabilities = align_probabilities(
        np.asarray(probabilities, dtype=np.float32), reference_shape, path
    )
    if probabilities.shape[0] == 1:
        foreground = probabilities[0]
    else:
        foreground = probabilities[1:].sum(axis=0)
    return np.ascontiguousarray(np.clip(foreground, 0.0, 1.0), dtype=np.float32)


def size_group(voxels: int) -> str:
    for name, lower, upper in SIZE_BINS:
        if voxels >= lower and (upper is None or voxels < upper):
            return name
    raise ValueError(voxels)


def collect_cases() -> list[dict]:
    gt_files = files_by_case(GT_DIR, "case_*.nii.gz")
    cases: list[dict] = []
    seen: set[str] = set()
    for fold in range(5):
        model_files = {
            name: files_by_case(root / f"fold_{fold}" / "validation", "case_*.npz")
            for name, root in MODEL_ROOTS.items()
        }
        keys = set(gt_files)
        for files in model_files.values():
            keys &= set(files)
        fold_keys = sorted(keys)
        if not fold_keys:
            raise RuntimeError(f"No common probability files for fold {fold}")
        duplicates = seen.intersection(fold_keys)
        if duplicates:
            raise RuntimeError(
                f"Cases occur in multiple validation folds: {sorted(duplicates)[:5]}"
            )
        seen.update(fold_keys)
        for key in fold_keys:
            cases.append(
                {
                    "fold": fold,
                    "case": key,
                    "gt": gt_files[key],
                    **{name: files[key] for name, files in model_files.items()},
                }
            )

    if seen != set(gt_files):
        missing = sorted(set(gt_files) - seen)
        extra = sorted(seen - set(gt_files))
        raise RuntimeError(
            f"OOF coverage mismatch: covered={len(seen)} gt={len(gt_files)} "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    return cases


def dice_from_counts(intersection: int, predicted: int, reference: int) -> float:
    denominator = predicted + reference
    return 1.0 if denominator == 0 else 2.0 * intersection / denominator


def evaluate_case_cpu(
    blend: np.ndarray, gt: np.ndarray, thresholds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.empty(len(thresholds), dtype=np.float32)
    missed = np.empty(len(thresholds), dtype=bool)
    gt_sum = int(gt.sum())
    for index, threshold in enumerate(thresholds):
        prediction = blend >= threshold
        intersection = int(np.logical_and(prediction, gt).sum())
        values[index] = dice_from_counts(
            intersection, int(prediction.sum()), gt_sum
        )
        missed[index] = gt_sum > 0 and intersection == 0
    return values, missed


def evaluate_case_gpu(
    blend: np.ndarray,
    gt: np.ndarray,
    thresholds: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    blend_tensor = torch.from_numpy(blend.reshape(-1)).to(device)
    gt_tensor = torch.from_numpy(np.ascontiguousarray(gt.reshape(-1))).to(device)
    gt_sum = int(gt_tensor.sum().item())
    values = np.empty(len(thresholds), dtype=np.float32)
    missed = np.empty(len(thresholds), dtype=bool)
    with torch.no_grad():
        for index, threshold in enumerate(thresholds):
            prediction = blend_tensor >= float(threshold)
            predicted = int(prediction.sum().item())
            intersection = int(
                torch.logical_and(prediction, gt_tensor).sum().item()
            )
            values[index] = dice_from_counts(intersection, predicted, gt_sum)
            missed[index] = gt_sum > 0 and intersection == 0
    del blend_tensor, gt_tensor
    torch.cuda.empty_cache()
    return values, missed


def summarize(values: np.ndarray, missed: np.ndarray) -> dict:
    return {
        "n": int(values.size),
        "mean_dice": float(values.mean()),
        "median_dice": float(np.median(values)),
        "std_dice": float(values.std()),
        "min_dice": float(values.min()),
        "max_dice": float(values.max()),
        "missed_cases": int(missed.sum()),
    }


def metrics_at_threshold(
    dices: np.ndarray,
    missed: np.ndarray,
    thresholds: np.ndarray,
    threshold: float,
    folds: np.ndarray,
    groups: np.ndarray,
) -> dict:
    threshold_index = int(np.argmin(np.abs(thresholds - threshold)))
    values = dices[:, threshold_index]
    missed_values = missed[:, threshold_index]
    result = {
        "threshold": float(thresholds[threshold_index]),
        "overall": summarize(values, missed_values),
        "by_fold": {},
        "by_total_gt_size": {},
    }
    for fold in range(5):
        mask = folds == fold
        result["by_fold"][str(fold)] = summarize(
            values[mask], missed_values[mask]
        )
    for group, _, _ in SIZE_BINS:
        mask = groups == group
        result["by_total_gt_size"][group] = summarize(
            values[mask], missed_values[mask]
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    cases = collect_cases()
    if args.check_only:
        first = cases[0]
        gt = load_gt(first["gt"])
        print(
            "OOF cases:",
            len(cases),
            "fold counts:",
            {
                fold: sum(item["fold"] == fold for item in cases)
                for fold in range(5)
            },
        )
        print("first case:", first["case"], "GT shape:", gt.shape)
        for model_name in MODEL_ROOTS:
            foreground = load_foreground_probability(
                first[model_name], gt.shape
            )
            print(
                model_name,
                "shape:",
                foreground.shape,
                "range:",
                (float(foreground.min()), float(foreground.max())),
            )
        return

    keys = [item["case"] for item in cases]
    folds = np.asarray([item["fold"] for item in cases], dtype=np.int8)
    thresholds = THRESHOLDS.copy()
    n_cases = len(cases)
    n_thresholds = len(thresholds)
    dices = np.full((n_cases, n_thresholds), np.nan, dtype=np.float32)
    missed = np.zeros((n_cases, n_thresholds), dtype=bool)
    single_dices = np.full((n_cases, len(MODEL_ROOTS)), np.nan, dtype=np.float32)
    single_missed = np.zeros((n_cases, len(MODEL_ROOTS)), dtype=bool)
    gt_sizes = np.full(n_cases, -1, dtype=np.int64)
    groups = np.full(n_cases, "", dtype="<U24")
    start_case = 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if CHECKPOINT_PATH.is_file():
        with np.load(CHECKPOINT_PATH, allow_pickle=False) as checkpoint:
            if checkpoint["keys"].tolist() != keys:
                raise RuntimeError("Checkpoint cases do not match current OOF inputs")
            if not np.allclose(checkpoint["thresholds"], thresholds):
                raise RuntimeError("Checkpoint thresholds do not match")
            start_case = int(checkpoint["next_case"])
            dices[:start_case] = checkpoint["dices"]
            missed[:start_case] = checkpoint["missed"]
            single_dices[:start_case] = checkpoint["single_dices"]
            single_missed[:start_case] = checkpoint["single_missed"]
            gt_sizes[:start_case] = checkpoint["gt_sizes"]
            groups[:start_case] = checkpoint["groups"]

    def save_checkpoint(next_case: int) -> None:
        temporary = CHECKPOINT_PATH.with_name("progress.tmp.npz")
        np.savez_compressed(
            temporary,
            next_case=np.asarray(next_case, dtype=np.int64),
            keys=np.asarray(keys),
            folds=folds,
            thresholds=thresholds,
            dices=dices[:next_case],
            missed=missed[:next_case],
            single_dices=single_dices[:next_case],
            single_missed=single_missed[:next_case],
            gt_sizes=gt_sizes[:next_case],
            groups=groups[:next_case],
        )
        temporary.replace(CHECKPOINT_PATH)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    model_names = list(MODEL_ROOTS)
    print("OOF cases:", n_cases, flush=True)
    print(
        "fold counts:",
        {fold: int(np.count_nonzero(folds == fold)) for fold in range(5)},
        flush=True,
    )
    print("models:", model_names, flush=True)
    print("thresholds:", thresholds.tolist(), flush=True)
    print("device:", device, "resuming:", start_case, "/", n_cases, flush=True)

    for case_index in range(start_case, n_cases):
        item = cases[case_index]
        gt = load_gt(item["gt"])
        gt_sum = int(gt.sum())
        gt_sizes[case_index] = gt_sum
        groups[case_index] = size_group(gt_sum)

        foregrounds = [
            load_foreground_probability(item[name], gt.shape)
            for name in model_names
        ]
        for model_index, foreground in enumerate(foregrounds):
            prediction = foreground >= 0.5
            intersection = int(np.logical_and(prediction, gt).sum())
            single_dices[case_index, model_index] = dice_from_counts(
                intersection, int(prediction.sum()), gt_sum
            )
            single_missed[case_index, model_index] = (
                gt_sum > 0 and intersection == 0
            )

        blend = np.ascontiguousarray(
            sum(foregrounds) / float(len(foregrounds)), dtype=np.float32
        )
        del foregrounds
        try:
            if device.type == "cuda":
                case_dices, case_missed = evaluate_case_gpu(
                    blend, gt, thresholds, device
                )
            else:
                case_dices, case_missed = evaluate_case_cpu(
                    blend, gt, thresholds
                )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(
                f"GPU OOM for {item['case']}; evaluating this case on CPU",
                flush=True,
            )
            case_dices, case_missed = evaluate_case_cpu(
                blend, gt, thresholds
            )

        dices[case_index] = case_dices
        missed[case_index] = case_missed
        if (case_index + 1) % 10 == 0 or case_index + 1 == n_cases:
            save_checkpoint(case_index + 1)
        if (case_index + 1) % 25 == 0:
            print(f"processed {case_index + 1}/{n_cases}", flush=True)

    groups_array = np.asarray(groups)
    threshold_rows = []
    for threshold_index, threshold in enumerate(thresholds):
        row = {
            "threshold": float(threshold),
            **summarize(dices[:, threshold_index], missed[:, threshold_index]),
        }
        threshold_rows.append(row)
    threshold_rows.sort(key=lambda row: row["mean_dice"], reverse=True)
    global_best_threshold = float(threshold_rows[0]["threshold"])

    lofo_thresholds: dict[str, float] = {}
    lofo_values = np.empty(n_cases, dtype=np.float32)
    lofo_missed = np.empty(n_cases, dtype=bool)
    for held_out_fold in range(5):
        train_mask = folds != held_out_fold
        validation_mask = folds == held_out_fold
        train_means = dices[train_mask].mean(axis=0)
        best_index = int(np.argmax(train_means))
        lofo_thresholds[str(held_out_fold)] = float(thresholds[best_index])
        lofo_values[validation_mask] = dices[validation_mask, best_index]
        lofo_missed[validation_mask] = missed[validation_mask, best_index]

    single_model_summary = {}
    for model_index, model_name in enumerate(model_names):
        single_model_summary[model_name] = {
            "threshold": 0.5,
            "overall": summarize(
                single_dices[:, model_index], single_missed[:, model_index]
            ),
            "by_fold": {
                str(fold): summarize(
                    single_dices[folds == fold, model_index],
                    single_missed[folds == fold, model_index],
                )
                for fold in range(5)
            },
        }

    primary = metrics_at_threshold(
        dices,
        missed,
        thresholds,
        PRIMARY_THRESHOLD,
        folds,
        groups_array,
    )
    global_best = metrics_at_threshold(
        dices,
        missed,
        thresholds,
        global_best_threshold,
        folds,
        groups_array,
    )
    standard = metrics_at_threshold(
        dices, missed, thresholds, 0.5, folds, groups_array
    )
    summary = {
        "note": (
            "All predictions are out-of-fold. Equal model weights are fixed. "
            "Threshold 0.35 is the fold-0 development choice; the global-best "
            "threshold is exploratory because it is selected on all OOF cases."
        ),
        "n_cases": n_cases,
        "fold_counts": {
            str(fold): int(np.count_nonzero(folds == fold))
            for fold in range(5)
        },
        "model_order": model_names,
        "weights": {name: 1.0 / len(model_names) for name in model_names},
        "single_models_at_0_5": single_model_summary,
        "fixed_primary": primary,
        "standard_0_5": standard,
        "global_best_exploratory": global_best,
        "lofo_calibrated": {
            "thresholds_by_held_out_fold": lofo_thresholds,
            "overall": summarize(lofo_values, lofo_missed),
            "by_fold": {
                str(fold): summarize(
                    lofo_values[folds == fold], lofo_missed[folds == fold]
                )
                for fold in range(5)
            },
            "by_total_gt_size": {
                group: summarize(
                    lofo_values[groups_array == group],
                    lofo_missed[groups_array == group],
                )
                for group, _, _ in SIZE_BINS
            },
        },
        "threshold_grid": sorted(
            threshold_rows, key=lambda row: row["threshold"]
        ),
    }

    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    with (OUT_DIR / "threshold_grid.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "threshold",
                "n",
                "mean_dice",
                "median_dice",
                "std_dice",
                "min_dice",
                "max_dice",
                "missed_cases",
            ],
        )
        writer.writeheader()
        writer.writerows(sorted(threshold_rows, key=lambda row: row["threshold"]))

    primary_index = int(np.argmin(np.abs(thresholds - PRIMARY_THRESHOLD)))
    with (OUT_DIR / "per_case_fixed_0_35.csv").open("w", newline="") as handle:
        fieldnames = [
            "case",
            "fold",
            "gt_voxels",
            "size_group",
            "dice",
            "missed",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, item in enumerate(cases):
            writer.writerow(
                {
                    "case": item["case"],
                    "fold": int(folds[index]),
                    "gt_voxels": int(gt_sizes[index]),
                    "size_group": groups_array[index],
                    "dice": float(dices[index, primary_index]),
                    "missed": int(missed[index, primary_index]),
                }
            )

    print(
        json.dumps(
            {
                "fixed_primary": primary,
                "standard_0_5": standard,
                "global_best_exploratory": global_best,
                "lofo_calibrated": summary["lofo_calibrated"],
                "single_models_at_0_5": single_model_summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    print("saved:", OUT_DIR / "summary.json", flush=True)


if __name__ == "__main__":
    main()
