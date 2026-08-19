#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from itertools import permutations
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.stats import rankdata


BASE = Path(
    os.environ.get("ISLES26_ROOT", Path(__file__).resolve().parents[1])
).resolve()
GT_DIR = BASE / "nnUNet_raw/Dataset004_ATLAS3_RAW/labelsTr"
MODEL_ROOTS = {
    "resencm": BASE
    / "nnUNet_results/Dataset004_ATLAS3_RAW/"
    "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres",
    "dtk10": BASE
    / "nnUNet_results/Dataset004_ATLAS3_RAW/"
    "nnUNetTrainerDiceTopK10Loss__nnUNetPlans__3d_fullres",
    "msl": BASE
    / "nnUNet_results/Dataset005_ATLAS3_RAW_MSL/"
    "nnUNetTrainer__nnUNetPlans__3d_fullres",
}
OUT_DIR = BASE / "ensemble_results/resencm_dtk10_msl_native_cvpp"
CHECKPOINT_PATH = OUT_DIR / "progress.npz"

BASE_THRESHOLDS = (0.30, 0.325, 0.35)
SMALL_LIMITS_MM3 = (25.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0)
MAX_CONFIDENCES = tuple(np.round(np.arange(0.35, 0.851, 0.05), 3))
SUPPORT_RULES = ((0.0, 0),)
SIZE_BINS_MM3 = (
    ("tiny_lt100_mm3", 0.0, 100.0),
    ("small_100_999_mm3", 100.0, 1000.0),
    ("medium_1000_9999_mm3", 1000.0, 10000.0),
    ("large_ge10000_mm3", 10000.0, None),
)
METRIC_NAMES = (
    "dice",
    "lesion_f1",
    "lesion_count_difference",
    "volume_difference_voxels",
    "volume_difference_mm3",
)


def case_id(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    return path.stem


def files_by_case(folder: Path, pattern: str) -> dict[str, Path]:
    return {case_id(path): path for path in folder.glob(pattern)}


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
            f"Cannot uniquely align {probabilities.shape} to "
            f"{reference_shape} in {path}"
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


def physical_size_group(volume_mm3: float) -> str:
    for name, lower, upper in SIZE_BINS_MM3:
        if volume_mm3 >= lower and (upper is None or volume_mm3 < upper):
            return name
    raise ValueError(volume_mm3)


def build_configurations() -> list[dict]:
    configurations: list[dict] = []
    for base_threshold in BASE_THRESHOLDS:
        configurations.append(
            {
                "name": f"raw_t{base_threshold:.3f}",
                "base_threshold": base_threshold,
                "small_limit_mm3": 0.0,
                "max_confidence": 0.0,
                "support_threshold": 0.0,
                "min_support_models": 0,
                "is_raw": True,
            }
        )
        for small_limit in SMALL_LIMITS_MM3:
            for max_confidence in MAX_CONFIDENCES:
                if max_confidence <= base_threshold + 1e-6:
                    continue
                for support_threshold, min_support_models in SUPPORT_RULES:
                    configurations.append(
                        {
                            "name": (
                                f"t{base_threshold:.3f}"
                                f"_size{small_limit:g}"
                                f"_conf{max_confidence:.3f}"
                                f"_sup{support_threshold:.3f}"
                                f"_n{min_support_models}"
                            ),
                            "base_threshold": base_threshold,
                            "small_limit_mm3": small_limit,
                            "max_confidence": max_confidence,
                            "support_threshold": support_threshold,
                            "min_support_models": min_support_models,
                            "is_raw": False,
                        }
                    )
    return configurations


def configuration_arrays(configurations: list[dict]) -> dict[str, np.ndarray]:
    return {
        "base_threshold": np.asarray(
            [item["base_threshold"] for item in configurations], dtype=np.float32
        ),
        "small_limit_mm3": np.asarray(
            [item["small_limit_mm3"] for item in configurations], dtype=np.float32
        ),
        "max_confidence": np.asarray(
            [item["max_confidence"] for item in configurations], dtype=np.float32
        ),
        "support_threshold": np.asarray(
            [item["support_threshold"] for item in configurations], dtype=np.float32
        ),
        "min_support_models": np.asarray(
            [item["min_support_models"] for item in configurations], dtype=np.int8
        ),
        "is_raw": np.asarray(
            [item["is_raw"] for item in configurations], dtype=bool
        ),
    }


def truth_to_prediction_components(
    prediction_labels: np.ndarray,
    truth_labels: np.ndarray,
    number_truth: int,
) -> list[np.ndarray]:
    result: list[list[int]] = [[] for _ in range(number_truth)]
    overlap = (prediction_labels > 0) & (truth_labels > 0)
    if not np.any(overlap):
        return [np.empty(0, dtype=np.int64) for _ in range(number_truth)]

    codes = np.unique(
        prediction_labels[overlap].astype(np.int64) * (number_truth + 1)
        + truth_labels[overlap].astype(np.int64)
    )
    for code in codes:
        prediction_id = int(code // (number_truth + 1))
        truth_id = int(code % (number_truth + 1))
        result[truth_id - 1].append(prediction_id - 1)
    return [np.asarray(values, dtype=np.int64) for values in result]


def evaluate_component_rules(
    blend: np.ndarray,
    truth: np.ndarray,
    truth_labels: np.ndarray,
    number_truth: int,
    voxel_volume_mm3: float,
    config_indices: np.ndarray,
    config_values: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    number_configs = len(config_indices)
    base_threshold = float(config_values["base_threshold"][config_indices[0]])
    prediction_mask = blend >= base_threshold
    prediction_labels, number_prediction = ndimage.label(prediction_mask)
    truth_voxels = int(truth.sum())

    if number_prediction == 0:
        zeros = np.zeros(number_configs, dtype=np.float32)
        ones = np.ones(number_configs, dtype=np.float32)
        f1 = ones if number_truth == 0 else zeros
        return {
            "dice": zeros if truth_voxels else ones,
            "lesion_f1": f1,
            "lesion_count_difference": np.full(
                number_configs, number_truth, dtype=np.float32
            ),
            "volume_difference_voxels": np.full(
                number_configs, truth_voxels, dtype=np.float32
            ),
            "volume_difference_mm3": np.full(
                number_configs, truth_voxels * voxel_volume_mm3, dtype=np.float32
            ),
            "missed": np.full(number_configs, truth_voxels > 0, dtype=bool),
        }

    foreground_labels = prediction_labels[prediction_mask]
    component_voxels = np.bincount(
        foreground_labels, minlength=number_prediction + 1
    )[1:].astype(np.int64)
    component_intersections = np.bincount(
        prediction_labels[truth].ravel(), minlength=number_prediction + 1
    )[1:].astype(np.int64)
    component_volume_mm3 = component_voxels.astype(np.float64) * voxel_volume_mm3

    small_limits = config_values["small_limit_mm3"][config_indices, None]
    max_confidences = config_values["max_confidence"][config_indices, None]
    confidence_reached = np.ones(
        (number_configs, number_prediction), dtype=bool
    )
    for confidence in np.unique(max_confidences):
        if confidence <= base_threshold + 1e-6:
            continue
        component_ids = np.unique(prediction_labels[blend >= confidence])
        component_ids = component_ids[component_ids > 0]
        reached = np.zeros(number_prediction, dtype=bool)
        reached[component_ids - 1] = True
        confidence_reached[
            np.isclose(max_confidences[:, 0], confidence)
        ] = reached
    selected = (
        component_volume_mm3[None, :] >= small_limits
    ) | confidence_reached
    selected_i64 = selected.astype(np.int64, copy=False)

    predicted_voxels = selected_i64 @ component_voxels
    intersections = selected_i64 @ component_intersections
    number_selected = selected_i64.sum(axis=1)
    false_positives = selected_i64 @ (component_intersections == 0).astype(np.int64)

    true_positives = np.zeros(number_configs, dtype=np.int64)
    for overlapping_components in truth_to_prediction_components(
        prediction_labels, truth_labels, number_truth
    ):
        if overlapping_components.size:
            true_positives += np.any(
                selected[:, overlapping_components], axis=1
            ).astype(np.int64)
    false_negatives = number_truth - true_positives

    dice_denominator = predicted_voxels + truth_voxels
    dice = np.ones(number_configs, dtype=np.float64)
    nonempty = dice_denominator > 0
    dice[nonempty] = 2.0 * intersections[nonempty] / dice_denominator[nonempty]

    f1_denominator = true_positives + (false_positives + false_negatives) / 2.0
    lesion_f1 = np.ones(number_configs, dtype=np.float64)
    nonempty_f1 = f1_denominator > 0
    lesion_f1[nonempty_f1] = (
        true_positives[nonempty_f1] / f1_denominator[nonempty_f1]
    )

    volume_difference_voxels = np.abs(predicted_voxels - truth_voxels)
    return {
        "dice": dice.astype(np.float32),
        "lesion_f1": lesion_f1.astype(np.float32),
        "lesion_count_difference": np.abs(
            number_selected - number_truth
        ).astype(np.float32),
        "volume_difference_voxels": volume_difference_voxels.astype(np.float32),
        "volume_difference_mm3": (
            volume_difference_voxels * voxel_volume_mm3
        ).astype(np.float32),
        "missed": (intersections == 0) & (truth_voxels > 0),
    }


def direct_metrics(
    truth: np.ndarray, prediction: np.ndarray, voxel_volume_mm3: float
) -> dict[str, float]:
    truth = truth.astype(bool)
    prediction = prediction.astype(bool)
    truth_labels, number_truth = ndimage.label(truth)
    prediction_labels, number_prediction = ndimage.label(prediction)
    truth_voxels = int(truth.sum())
    prediction_voxels = int(prediction.sum())
    intersection = int(np.logical_and(truth, prediction).sum())
    denominator = truth_voxels + prediction_voxels
    dice = 1.0 if denominator == 0 else 2.0 * intersection / denominator

    true_positives = 0
    for truth_id in range(1, number_truth + 1):
        if np.any(prediction[truth_labels == truth_id]):
            true_positives += 1
    false_negatives = number_truth - true_positives
    false_positives = 0
    for prediction_id in range(1, number_prediction + 1):
        if not np.any(truth[prediction_labels == prediction_id]):
            false_positives += 1
    f1_denominator = true_positives + (false_positives + false_negatives) / 2.0
    lesion_f1 = (
        1.0 if f1_denominator == 0 else true_positives / f1_denominator
    )
    volume_difference_voxels = abs(prediction_voxels - truth_voxels)
    return {
        "dice": float(dice),
        "lesion_f1": float(lesion_f1),
        "lesion_count_difference": float(
            abs(number_prediction - number_truth)
        ),
        "volume_difference_voxels": float(volume_difference_voxels),
        "volume_difference_mm3": float(
            volume_difference_voxels * voxel_volume_mm3
        ),
    }


def self_test() -> None:
    truth = np.zeros((8, 8, 8), dtype=bool)
    truth[1:3, 1:3, 1:3] = True
    truth[5:7, 5:7, 5:7] = True
    prediction = np.zeros_like(truth)
    prediction[1:3, 1:3, 1:3] = True
    prediction[0, 7, 0] = True
    metrics = direct_metrics(truth, prediction, 2.5)
    assert np.isclose(metrics["dice"], 16.0 / 25.0)
    assert np.isclose(metrics["lesion_f1"], 0.5)
    assert metrics["lesion_count_difference"] == 0.0
    assert metrics["volume_difference_voxels"] == 7.0
    assert metrics["volume_difference_mm3"] == 17.5

    foreground = prediction.astype(np.float32) * 0.9
    truth_labels, number_truth = ndimage.label(truth)
    configurations = build_configurations()
    config_values = configuration_arrays(configurations)
    raw_index = int(
        np.flatnonzero(
            config_values["is_raw"]
            & np.isclose(config_values["base_threshold"], 0.325)
        )[0]
    )
    vectorized = evaluate_component_rules(
        blend=foreground,
        truth=truth,
        truth_labels=truth_labels,
        number_truth=number_truth,
        voxel_volume_mm3=2.5,
        config_indices=np.asarray([raw_index], dtype=np.int64),
        config_values=config_values,
    )
    for name in METRIC_NAMES:
        assert np.isclose(vectorized[name][0], metrics[name]), (
            name,
            vectorized[name][0],
            metrics[name],
        )
    print("Self-test passed:", json.dumps(metrics, sort_keys=True))


def summarize_metric_vectors(values: dict[str, np.ndarray]) -> dict:
    return {
        "n": int(values["dice"].size),
        "mean_dice": float(np.mean(values["dice"])),
        "median_dice": float(np.median(values["dice"])),
        "std_dice": float(np.std(values["dice"])),
        "mean_lesion_f1": float(np.mean(values["lesion_f1"])),
        "median_lesion_f1": float(np.median(values["lesion_f1"])),
        "mean_lesion_count_difference": float(
            np.mean(values["lesion_count_difference"])
        ),
        "median_lesion_count_difference": float(
            np.median(values["lesion_count_difference"])
        ),
        "mean_volume_difference_voxels": float(
            np.mean(values["volume_difference_voxels"])
        ),
        "median_volume_difference_voxels": float(
            np.median(values["volume_difference_voxels"])
        ),
        "mean_volume_difference_mm3": float(
            np.mean(values["volume_difference_mm3"])
        ),
        "median_volume_difference_mm3": float(
            np.median(values["volume_difference_mm3"])
        ),
        "missed_cases": int(np.count_nonzero(values["missed"])),
    }


def vectors_for_config(
    metrics: dict[str, np.ndarray],
    missed: np.ndarray,
    case_indices: np.ndarray,
    config_index: int,
) -> dict[str, np.ndarray]:
    return {
        **{
            name: metrics[name][case_indices, config_index]
            for name in METRIC_NAMES
        },
        "missed": missed[case_indices, config_index],
    }


def select_configuration(
    metrics: dict[str, np.ndarray],
    case_indices: np.ndarray,
    objective: str,
) -> tuple[int, float]:
    if objective == "dice":
        values = np.mean(metrics["dice"][case_indices], axis=0)
        best = int(np.argmax(values))
        return best, float(values[best])
    if objective == "lesion_f1":
        values = np.mean(metrics["lesion_f1"][case_indices], axis=0)
        best = int(np.argmax(values))
        return best, float(values[best])
    if objective != "balanced_rank":
        raise ValueError(objective)

    number_configs = metrics["dice"].shape[1]
    rank_sum = np.zeros(number_configs, dtype=np.float64)
    for case_index in case_indices:
        rank_sum += rankdata(-metrics["dice"][case_index], method="average")
        rank_sum += rankdata(-metrics["lesion_f1"][case_index], method="average")
        rank_sum += rankdata(
            metrics["lesion_count_difference"][case_index], method="average"
        )
        rank_sum += rankdata(
            metrics["volume_difference_voxels"][case_index], method="average"
        )
    average_rank = rank_sum / (4.0 * len(case_indices))
    best = int(np.argmin(average_rank))
    return best, float(average_rank[best])


def cross_fitted_vectors(
    metrics: dict[str, np.ndarray],
    missed: np.ndarray,
    folds: np.ndarray,
    configurations: list[dict],
    objective: str,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    number_cases = len(folds)
    selected_by_case = np.full(number_cases, -1, dtype=np.int32)
    selections: list[dict] = []
    for fold in range(5):
        train_indices = np.flatnonzero(folds != fold)
        test_indices = np.flatnonzero(folds == fold)
        selected, training_objective = select_configuration(
            metrics, train_indices, objective
        )
        selected_by_case[test_indices] = selected
        selections.append(
            {
                "held_out_fold": fold,
                "training_cases": int(train_indices.size),
                "test_cases": int(test_indices.size),
                "config_index": selected,
                "training_objective": training_objective,
                "configuration": configurations[selected],
            }
        )

    if np.any(selected_by_case < 0):
        raise RuntimeError("Cross-fitted selection did not cover every case")
    row_indices = np.arange(number_cases)
    values = {
        name: metrics[name][row_indices, selected_by_case]
        for name in METRIC_NAMES
    }
    values["missed"] = missed[row_indices, selected_by_case]
    values["selected_config_index"] = selected_by_case
    return values, selections


def summarize_by_size(
    values: dict[str, np.ndarray], groups: np.ndarray
) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for name, _, _ in SIZE_BINS_MM3:
        indices = np.flatnonzero(groups == name)
        subset = {
            metric: values[metric][indices]
            for metric in METRIC_NAMES
        }
        subset["missed"] = values["missed"][indices]
        result[name] = summarize_metric_vectors(subset)
    return result


def paired_comparison(
    candidate: dict[str, np.ndarray], baseline: dict[str, np.ndarray]
) -> dict:
    result = {}
    directions = {
        "dice": 1,
        "lesion_f1": 1,
        "lesion_count_difference": -1,
        "volume_difference_voxels": -1,
        "volume_difference_mm3": -1,
    }
    for name, direction in directions.items():
        raw_delta = candidate[name] - baseline[name]
        favorable_delta = raw_delta * direction
        result[name] = {
            "mean_raw_delta": float(np.mean(raw_delta)),
            "median_raw_delta": float(np.median(raw_delta)),
            "better_cases": int(np.count_nonzero(favorable_delta > 1e-8)),
            "worse_cases": int(np.count_nonzero(favorable_delta < -1e-8)),
            "ties": int(np.count_nonzero(np.abs(favorable_delta) <= 1e-8)),
        }
    result["missed_case_delta"] = int(
        np.count_nonzero(candidate["missed"])
        - np.count_nonzero(baseline["missed"])
    )
    return result


def write_configurations(
    path: Path, configurations: list[dict], metrics: dict[str, np.ndarray]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "config_index",
            *configurations[0].keys(),
            "mean_dice",
            "mean_lesion_f1",
            "mean_lesion_count_difference",
            "mean_volume_difference_voxels",
            "mean_volume_difference_mm3",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, configuration in enumerate(configurations):
            writer.writerow(
                {
                    "config_index": index,
                    **configuration,
                    **{
                        f"mean_{name}": float(np.mean(metrics[name][:, index]))
                        for name in METRIC_NAMES
                    },
                }
            )


def write_per_case(
    path: Path,
    keys: list[str],
    folds: np.ndarray,
    gt_voxels: np.ndarray,
    gt_volume_mm3: np.ndarray,
    number_gt_lesions: np.ndarray,
    groups: np.ndarray,
    named_vectors: dict[str, dict[str, np.ndarray]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "case",
            "fold",
            "gt_voxels",
            "gt_volume_mm3",
            "gt_lesions",
            "physical_size_group",
        ]
        for prefix in named_vectors:
            fields.extend(
                [
                    f"{prefix}_dice",
                    f"{prefix}_lesion_f1",
                    f"{prefix}_lesion_count_difference",
                    f"{prefix}_volume_difference_voxels",
                    f"{prefix}_volume_difference_mm3",
                    f"{prefix}_missed",
                ]
            )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, key in enumerate(keys):
            row = {
                "case": key,
                "fold": int(folds[index]),
                "gt_voxels": int(gt_voxels[index]),
                "gt_volume_mm3": float(gt_volume_mm3[index]),
                "gt_lesions": int(number_gt_lesions[index]),
                "physical_size_group": groups[index],
            }
            for prefix, values in named_vectors.items():
                for metric in METRIC_NAMES:
                    row[f"{prefix}_{metric}"] = float(values[metric][index])
                row[f"{prefix}_missed"] = int(values["missed"][index])
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    cases = collect_cases()
    configurations = build_configurations()
    config_values = configuration_arrays(configurations)
    keys = [item["case"] for item in cases]
    folds = np.asarray([item["fold"] for item in cases], dtype=np.int8)
    number_cases = len(cases)
    number_configs = len(configurations)
    indices_by_base = {
        threshold: np.flatnonzero(
            np.isclose(config_values["base_threshold"], threshold)
        )
        for threshold in BASE_THRESHOLDS
    }

    print("OOF cases:", number_cases)
    print(
        "Fold counts:",
        {fold: int(np.count_nonzero(folds == fold)) for fold in range(5)},
    )
    print("Configurations:", number_configs)
    print(
        "Configurations by base threshold:",
        {
            str(threshold): int(len(indices))
            for threshold, indices in indices_by_base.items()
        },
    )
    if args.check_only:
        first = cases[0]
        image = sitk.ReadImage(str(first["gt"]))
        truth = sitk.GetArrayFromImage(image) > 0
        print("First case:", first["case"])
        print("Shape:", truth.shape)
        print("Spacing xyz:", image.GetSpacing())
        print("Voxel volume mm3:", float(np.prod(image.GetSpacing())))
        for model_name in MODEL_ROOTS:
            probability = load_foreground_probability(
                first[model_name], truth.shape
            )
            print(
                model_name,
                probability.shape,
                float(probability.min()),
                float(probability.max()),
            )
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        name: np.full((number_cases, number_configs), np.nan, dtype=np.float32)
        for name in METRIC_NAMES
    }
    missed = np.zeros((number_cases, number_configs), dtype=bool)
    voxel_volume_mm3 = np.full(number_cases, np.nan, dtype=np.float32)
    gt_voxels = np.full(number_cases, -1, dtype=np.int64)
    gt_volume_mm3 = np.full(number_cases, np.nan, dtype=np.float32)
    number_gt_lesions = np.full(number_cases, -1, dtype=np.int32)
    groups = np.full(number_cases, "", dtype="<U32")
    start_case = 0

    if CHECKPOINT_PATH.is_file():
        with np.load(CHECKPOINT_PATH, allow_pickle=False) as checkpoint:
            if checkpoint["keys"].tolist() != keys:
                raise RuntimeError("Checkpoint cases do not match current OOF inputs")
            if checkpoint["config_names"].tolist() != [
                item["name"] for item in configurations
            ]:
                raise RuntimeError("Checkpoint configurations do not match")
            start_case = int(checkpoint["next_case"])
            for name in METRIC_NAMES:
                metrics[name][:start_case] = checkpoint[name]
            missed[:start_case] = checkpoint["missed"]
            voxel_volume_mm3[:start_case] = checkpoint["voxel_volume_mm3"]
            gt_voxels[:start_case] = checkpoint["gt_voxels"]
            gt_volume_mm3[:start_case] = checkpoint["gt_volume_mm3"]
            number_gt_lesions[:start_case] = checkpoint["number_gt_lesions"]
            groups[:start_case] = checkpoint["groups"]

    def save_checkpoint(next_case: int) -> None:
        temporary = CHECKPOINT_PATH.with_name("progress.tmp.npz")
        np.savez_compressed(
            temporary,
            next_case=np.asarray(next_case, dtype=np.int64),
            keys=np.asarray(keys),
            config_names=np.asarray([item["name"] for item in configurations]),
            folds=folds,
            **{
                name: metrics[name][:next_case]
                for name in METRIC_NAMES
            },
            missed=missed[:next_case],
            voxel_volume_mm3=voxel_volume_mm3[:next_case],
            gt_voxels=gt_voxels[:next_case],
            gt_volume_mm3=gt_volume_mm3[:next_case],
            number_gt_lesions=number_gt_lesions[:next_case],
            groups=groups[:next_case],
        )
        temporary.replace(CHECKPOINT_PATH)

    print("Resuming:", start_case, "/", number_cases, flush=True)
    started = time.time()
    for case_index in range(start_case, number_cases):
        item = cases[case_index]
        image = sitk.ReadImage(str(item["gt"]))
        truth = np.asarray(sitk.GetArrayFromImage(image)) > 0
        truth_labels, number_truth = ndimage.label(truth)
        current_voxel_volume = float(np.prod(image.GetSpacing()))
        current_gt_voxels = int(truth.sum())
        current_gt_volume = current_gt_voxels * current_voxel_volume

        voxel_volume_mm3[case_index] = current_voxel_volume
        gt_voxels[case_index] = current_gt_voxels
        gt_volume_mm3[case_index] = current_gt_volume
        number_gt_lesions[case_index] = number_truth
        groups[case_index] = physical_size_group(current_gt_volume)

        foregrounds = [
            load_foreground_probability(item[name], truth.shape)
            for name in MODEL_ROOTS
        ]
        blend = np.ascontiguousarray(
            sum(foregrounds) / float(len(foregrounds)), dtype=np.float32
        )
        for threshold, config_indices in indices_by_base.items():
            result = evaluate_component_rules(
                blend=blend,
                truth=truth,
                truth_labels=truth_labels,
                number_truth=number_truth,
                voxel_volume_mm3=current_voxel_volume,
                config_indices=config_indices,
                config_values=config_values,
            )
            for name in METRIC_NAMES:
                metrics[name][case_index, config_indices] = result[name]
            missed[case_index, config_indices] = result["missed"]

        del foregrounds, blend, truth, truth_labels
        next_case = case_index + 1
        if (
            next_case % args.checkpoint_every == 0
            or next_case == number_cases
        ):
            save_checkpoint(next_case)
        if next_case % 10 == 0 or next_case == number_cases:
            elapsed = time.time() - started
            processed = next_case - start_case
            rate = elapsed / max(processed, 1)
            remaining = rate * (number_cases - next_case)
            print(
                f"{next_case}/{number_cases} "
                f"elapsed={elapsed / 60:.1f}m "
                f"eta={remaining / 60:.1f}m",
                flush=True,
            )

    for name in METRIC_NAMES:
        if np.isnan(metrics[name]).any():
            raise RuntimeError(f"Metric array contains NaN: {name}")

    all_indices = np.arange(number_cases)
    raw_indices = {
        threshold: int(
            np.flatnonzero(
                config_values["is_raw"]
                & np.isclose(config_values["base_threshold"], threshold)
            )[0]
        )
        for threshold in BASE_THRESHOLDS
    }
    raw_vectors = {
        threshold: vectors_for_config(
            metrics, missed, all_indices, config_index
        )
        for threshold, config_index in raw_indices.items()
    }

    objectives = ("balanced_rank", "dice", "lesion_f1")
    cross_fitted: dict[str, dict[str, np.ndarray]] = {}
    cross_fitted_selections: dict[str, list[dict]] = {}
    final_selections: dict[str, dict] = {}
    for objective in objectives:
        values, selections = cross_fitted_vectors(
            metrics, missed, folds, configurations, objective
        )
        cross_fitted[objective] = values
        cross_fitted_selections[objective] = selections
        selected, objective_value = select_configuration(
            metrics, all_indices, objective
        )
        final_selections[objective] = {
            "config_index": selected,
            "objective_value": objective_value,
            "configuration": configurations[selected],
            "all_oof_summary": summarize_metric_vectors(
                vectors_for_config(
                    metrics, missed, all_indices, selected
                )
            ),
        }

    summary = {
        "metric_protocol": {
            "reference": (
                "ATLAS challenge definitions: Dice, lesion-wise F1, "
                "simple lesion count difference, absolute volume difference"
            ),
            "connectivity": (
                "scipy.ndimage.label default 3D connectivity "
                "(6-connected foreground)"
            ),
            "lesion_detection": (
                "A ground-truth component is detected when at least one voxel "
                "overlaps the prediction"
            ),
            "native_space": (
                "Component size thresholds use physical mm3 from the NIfTI "
                "spacing. Volume difference is reported in both voxels and mm3."
            ),
            "cross_fitting": (
                "For each held-out fold, the post-processing configuration is "
                "selected using only the other four folds."
            ),
        },
        "n_cases": number_cases,
        "fold_counts": {
            str(fold): int(np.count_nonzero(folds == fold))
            for fold in range(5)
        },
        "n_configurations": number_configs,
        "voxel_volume_mm3": {
            "min": float(np.min(voxel_volume_mm3)),
            "median": float(np.median(voxel_volume_mm3)),
            "max": float(np.max(voxel_volume_mm3)),
        },
        "physical_size_counts": {
            name: int(np.count_nonzero(groups == name))
            for name, _, _ in SIZE_BINS_MM3
        },
        "raw_thresholds": {
            str(threshold): {
                "configuration": configurations[raw_indices[threshold]],
                "overall": summarize_metric_vectors(raw_vectors[threshold]),
                "by_physical_gt_size": summarize_by_size(
                    raw_vectors[threshold], groups
                ),
            }
            for threshold in BASE_THRESHOLDS
        },
        "cross_fitted": {
            objective: {
                "overall": summarize_metric_vectors(cross_fitted[objective]),
                "by_physical_gt_size": summarize_by_size(
                    cross_fitted[objective], groups
                ),
                "selected_by_fold": cross_fitted_selections[objective],
                "paired_vs_raw_0.325": paired_comparison(
                    cross_fitted[objective], raw_vectors[0.325]
                ),
                "paired_vs_raw_0.35": paired_comparison(
                    cross_fitted[objective], raw_vectors[0.35]
                ),
            }
            for objective in objectives
        },
        "deployment_configuration_selected_on_all_oof": final_selections,
    }

    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    write_configurations(
        OUT_DIR / "configuration_grid.csv", configurations, metrics
    )
    write_per_case(
        OUT_DIR / "per_case_metrics.csv",
        keys,
        folds,
        gt_voxels,
        gt_volume_mm3,
        number_gt_lesions,
        groups,
        {
            "raw_0_30": raw_vectors[0.30],
            "raw_0_325": raw_vectors[0.325],
            "raw_0_35": raw_vectors[0.35],
            "cvpp_balanced": cross_fitted["balanced_rank"],
            "cvpp_dice": cross_fitted["dice"],
            "cvpp_f1": cross_fitted["lesion_f1"],
        },
    )
    print(json.dumps(summary["raw_thresholds"], indent=2), flush=True)
    print(json.dumps(summary["cross_fitted"], indent=2), flush=True)
    print("Outputs:", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
