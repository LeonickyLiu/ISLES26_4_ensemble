#!/usr/bin/env python3
"""Full-volume OOF evaluation of shortlisted four-model fusion settings.

The shortlist is produced by ``screen_three_model_weights_samples.py``.  This
second stage loads each native-resolution validation volume, computes the
current ISLES'26 binary metrics (including IoU>=0.25 one-to-one lesion
matching), and compares raw and component-filtered predictions.  Connected
components are computed once per weight/threshold pair and reused by all
post-processing modes.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
from itertools import permutations
from pathlib import Path

import cc3d
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.stats import rankdata


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get("ISLES26_ROOT", REPO_ROOT)).resolve()
OFFICIAL_METRICS_ROOT = Path(
    os.environ.get("ISLES26_OFFICIAL_METRICS", BASE / "isles26_official_metrics")
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
    "ici": BASE
    / "nnUNet_results/Dataset004_ATLAS3_RAW/"
    "nnUNetTrainerICILoss__nnUNetPlans__3d_fullres",
}
DEFAULT_SELECTED_JSON = REPO_ROOT / "configs/core4_basis_candidates_preregistered_20260818.json"
DEFAULT_OUT_DIR = BASE / "ensemble_results/core4_weight_threshold_search"
MODEL_ORDER = ("resencm", "dtk10", "msl", "ici")
METRICS = (
    "dice",
    "lesion_f1",
    "lesion_count_difference",
    "absolute_volume_difference_ml",
)
DIRECTIONS = {
    "dice": 1.0,
    "lesion_f1": 1.0,
    "lesion_count_difference": -1.0,
    "absolute_volume_difference_ml": -1.0,
}
SIZE_BINS_MM3 = (
    ("tiny_lt100_mm3", 0.0, 100.0),
    ("small_100_999_mm3", 100.0, 1000.0),
    ("medium_1000_9999_mm3", 1000.0, 10000.0),
    ("large_ge10000_mm3", 10000.0, None),
)
MODES = tuple(
    {"name": f"pp_s{small}_c{int(confidence * 100):03d}",
     "small_limit_mm3": float(small), "max_confidence": confidence}
    for confidence in (0.64, 0.65, 0.66)
    for small in (100, 150, 200, 250, 300)
)

WORKER_CANDIDATES: list[dict] = []


def case_id(path: Path) -> str:
    return path.name[:-7] if path.name.endswith(".nii.gz") else path.stem


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
            f"Cannot uniquely align {probabilities.shape} to {reference_shape} in {path}"
        )
    return np.transpose(probabilities, (0,) + tuple(index + 1 for index in matches[0]))


def load_foreground_probability(path: Path, reference_shape: tuple[int, ...]) -> np.ndarray:
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
    foreground = probabilities[0] if probabilities.shape[0] == 1 else probabilities[1:].sum(axis=0)
    return np.ascontiguousarray(np.clip(foreground, 0.0, 1.0), dtype=np.float32)


def collect_cases(folds: set[int]) -> list[dict]:
    gt_files = files_by_case(GT_DIR, "case_*.nii.gz")
    cases: list[dict] = []
    seen: set[str] = set()
    for fold in sorted(folds):
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
            raise RuntimeError(f"Duplicate OOF cases: {sorted(duplicates)[:5]}")
        seen.update(fold_keys)
        cases.extend(
            {
                "fold": fold,
                "case": key,
                "gt": str(gt_files[key]),
                **{name: str(files[key]) for name, files in model_files.items()},
            }
            for key in fold_keys
        )
    if folds == set(range(5)) and seen != set(gt_files):
        raise RuntimeError(
            f"OOF coverage mismatch: covered={len(seen)} gt={len(gt_files)}"
        )
    return cases


def evenly_sample_cases_per_fold(cases: list[dict], limit: int) -> list[dict]:
    """Choose deterministic, range-spanning cases from every requested fold."""
    if limit <= 0:
        return cases
    selected: list[dict] = []
    for fold in sorted({int(item["fold"]) for item in cases}):
        fold_cases = [item for item in cases if int(item["fold"]) == fold]
        if len(fold_cases) <= limit:
            selected.extend(fold_cases)
            continue
        # Midpoints of equal-width bins span the fold without always selecting
        # pathologically large first/last case IDs.
        indices = np.floor(
            (np.arange(limit, dtype=np.float64) + 0.5)
            * len(fold_cases)
            / limit
        ).astype(np.int64)
        selected.extend(fold_cases[int(index)] for index in indices)
    return selected


def candidate_key(weights: dict[str, float], threshold: float) -> tuple[float, ...]:
    return tuple(round(float(weights[name]), 7) for name in MODEL_ORDER) + (
        round(float(threshold), 7),
    )


def load_candidates(path: Path, max_candidates: int) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    source = payload.get("selected_configs", payload)
    if not isinstance(source, list):
        raise ValueError(f"Expected a candidate list in {path}")
    candidates: list[dict] = []
    seen: set[tuple[float, ...]] = set()
    default_thresholds = payload.get("thresholds", [0.35])

    def add(weights: dict[str, float], threshold: float, reason: str) -> None:
        normalized = {name: float(weights[name]) for name in MODEL_ORDER}
        total = sum(normalized.values())
        if not np.isclose(total, 1.0, atol=1e-5):
            raise ValueError((normalized, total))
        key = candidate_key(normalized, threshold)
        if key in seen:
            return
        index = len(candidates)
        candidates.append(
            {
                "index": index,
                "weights": normalized,
                "threshold": float(threshold),
                "reason": reason,
                "base_name": (
                    f"c{index:02d}_r{normalized['resencm']:.4f}"
                    f"_d{normalized['dtk10']:.4f}_m{normalized['msl']:.4f}"
                    f"_i{normalized['ici']:.4f}"
                    f"_t{float(threshold):.3f}"
                ),
            }
        )
        seen.add(key)

    add(
        {"resencm": 0.375, "dtk10": 0.375, "msl": 0.250, "ici": 0.0},
        0.35,
        "mandatory_current3_baseline",
    )
    for item in source:
        thresholds = item.get("thresholds", default_thresholds)
        if "threshold" in item:
            thresholds = [item["threshold"]]
        for threshold in thresholds:
            add(item["weights"], threshold, item.get("screen_reason", "shortlist"))
            if max_candidates > 0 and len(candidates) >= max_candidates:
                return candidates
    return candidates


def init_worker(candidates: list[dict]) -> None:
    global WORKER_CANDIDATES
    WORKER_CANDIDATES = candidates


def overlap_context(
    truth: np.ndarray,
    truth_labels: np.ndarray,
    number_truth: int,
    prediction_labels: np.ndarray,
    number_prediction: int,
) -> dict:
    truth_areas = np.bincount(
        truth_labels.ravel(), minlength=number_truth + 1
    ).astype(np.int64, copy=False)
    prediction_areas = np.bincount(
        prediction_labels.ravel(), minlength=number_prediction + 1
    ).astype(np.int64, copy=False)
    overlap = truth & (prediction_labels > 0)
    prediction_truth_overlap = np.bincount(
        prediction_labels[overlap], minlength=number_prediction + 1
    ).astype(np.int64, copy=False)
    if np.any(overlap):
        modulus = number_truth + 1
        encoded = (
            truth_labels[overlap].astype(np.int64)
            + prediction_labels[overlap].astype(np.int64) * modulus
        )
        values, intersections = np.unique(encoded, return_counts=True)
        pair_truth = values % modulus
        pair_prediction = values // modulus
        pair_iou = intersections.astype(np.float64) / (
            truth_areas[pair_truth]
            + prediction_areas[pair_prediction]
            - intersections
        )
        order = np.argsort(-pair_iou, kind="stable")
        pair_truth = pair_truth[order]
        pair_prediction = pair_prediction[order]
        pair_iou = pair_iou[order]
    else:
        pair_truth = np.empty(0, dtype=np.int64)
        pair_prediction = np.empty(0, dtype=np.int64)
        pair_iou = np.empty(0, dtype=np.float64)
    return {
        "truth_areas": truth_areas,
        "prediction_areas": prediction_areas,
        "prediction_truth_overlap": prediction_truth_overlap,
        "pair_truth": pair_truth,
        "pair_prediction": pair_prediction,
        "pair_iou": pair_iou,
    }


def metrics_for_kept_components(
    context: dict,
    keep: np.ndarray,
    number_truth: int,
    voxel_volume_mm3: float,
    truth_size_bin_by_id: np.ndarray,
) -> dict:
    selected_ids = np.flatnonzero(keep)
    selected_ids = selected_ids[selected_ids > 0]
    prediction_voxels = int(context["prediction_areas"][selected_ids].sum())
    truth_voxels = int(context["truth_areas"][1:].sum())
    intersection = int(context["prediction_truth_overlap"][selected_ids].sum())
    denominator = prediction_voxels + truth_voxels
    dice = 1.0 if denominator == 0 else 2.0 * intersection / denominator

    matched_truth: set[int] = set()
    matched_prediction: set[int] = set()
    for truth_id, prediction_id, iou in zip(
        context["pair_truth"], context["pair_prediction"], context["pair_iou"]
    ):
        if iou < 0.25:
            break
        truth_id = int(truth_id)
        prediction_id = int(prediction_id)
        if not keep[prediction_id]:
            continue
        if truth_id in matched_truth or prediction_id in matched_prediction:
            continue
        matched_truth.add(truth_id)
        matched_prediction.add(prediction_id)

    true_positives = len(matched_truth)
    number_prediction = int(selected_ids.size)
    denominator_f1 = number_truth + number_prediction
    lesion_f1 = 1.0 if denominator_f1 == 0 else 2.0 * true_positives / denominator_f1
    detected_truth = {
        int(truth_id)
        for truth_id, prediction_id in zip(
            context["pair_truth"], context["pair_prediction"]
        )
        if keep[int(prediction_id)]
    }
    size_detection = {}
    for bin_index, (name, _, _) in enumerate(SIZE_BINS_MM3):
        truth_ids = np.flatnonzero(truth_size_bin_by_id == bin_index)
        detected = sum(int(truth_id) in detected_truth for truth_id in truth_ids)
        size_detection[name] = [int(detected), int(truth_ids.size)]
    return {
        "dice": float(dice),
        "lesion_f1": float(lesion_f1),
        "lesion_count_difference": float(abs(number_prediction - number_truth)),
        "absolute_volume_difference_ml": float(
            abs(prediction_voxels - truth_voxels) * voxel_volume_mm3 / 1000.0
        ),
        "missed_case": int(truth_voxels > 0 and intersection == 0),
        "lesion_size_detection": size_detection,
    }


def evaluate_candidate(
    blend: np.ndarray,
    threshold: float,
    truth: np.ndarray,
    truth_labels: np.ndarray,
    number_truth: int,
    voxel_volume_mm3: float,
    truth_size_bin_by_id: np.ndarray,
) -> dict[str, dict]:
    prediction = blend >= threshold
    prediction_labels, number_prediction = cc3d.connected_components(
        prediction, return_N=True
    )
    context = overlap_context(
        truth, truth_labels, number_truth, prediction_labels, number_prediction
    )
    keep_all = np.ones(number_prediction + 1, dtype=bool)
    keep_all[0] = False
    if number_prediction:
        component_ids = np.arange(1, number_prediction + 1)
        peaks = np.asarray(
            ndimage.maximum(blend, labels=prediction_labels, index=component_ids),
            dtype=np.float32,
        )
    else:
        peaks = np.empty(0, dtype=np.float32)
    output = {}
    for mode in MODES:
        if mode["name"] == "raw":
            keep = keep_all
        else:
            volumes = context["prediction_areas"][1:] * voxel_volume_mm3
            selected = (volumes >= mode["small_limit_mm3"]) | (
                peaks >= mode["max_confidence"]
            )
            keep = np.zeros(number_prediction + 1, dtype=bool)
            keep[1:] = selected
        output[mode["name"]] = metrics_for_kept_components(
            context,
            keep,
            number_truth,
            voxel_volume_mm3,
            truth_size_bin_by_id,
        )
    return output


def process_case(item: dict) -> dict:
    image = sitk.ReadImage(item["gt"])
    truth = np.asarray(sitk.GetArrayFromImage(image)) > 0
    voxel_volume_mm3 = float(np.prod(image.GetSpacing()))
    truth_labels, number_truth = cc3d.connected_components(truth, return_N=True)
    truth_areas = np.bincount(
        truth_labels.ravel(), minlength=number_truth + 1
    )
    truth_volumes = truth_areas[1:].astype(np.float64) * voxel_volume_mm3
    truth_size_bin_by_id = np.full(number_truth + 1, -1, dtype=np.int8)
    for bin_index, (_, lower, upper) in enumerate(SIZE_BINS_MM3):
        selected = truth_volumes >= lower
        if upper is not None:
            selected &= truth_volumes < upper
        truth_size_bin_by_id[np.flatnonzero(selected) + 1] = bin_index

    probabilities = {
        name: load_foreground_probability(Path(item[name]), truth.shape)
        for name in MODEL_ORDER
    }
    configurations = {}
    candidate_groups: dict[tuple[float, ...], list[dict]] = {}
    for candidate in WORKER_CANDIDATES:
        key = tuple(candidate["weights"][name] for name in MODEL_ORDER)
        candidate_groups.setdefault(key, []).append(candidate)
    for group in candidate_groups.values():
        weights = group[0]["weights"]
        blend = np.ascontiguousarray(
            sum(weights[name] * probabilities[name] for name in MODEL_ORDER),
            dtype=np.float32,
        )
        for candidate in group:
            modes = evaluate_candidate(
                blend,
                candidate["threshold"],
                truth,
                truth_labels,
                number_truth,
                voxel_volume_mm3,
                truth_size_bin_by_id,
            )
            for mode_name, metrics in modes.items():
                configurations[f"{candidate['base_name']}__{mode_name}"] = metrics
    return {
        "case": item["case"],
        "fold": int(item["fold"]),
        "gt_lesions": int(number_truth),
        "gt_volume_ml": float(truth.sum() * voxel_volume_mm3 / 1000.0),
        "configurations": configurations,
    }


def self_test_official() -> None:
    sys.path.insert(0, str(OFFICIAL_METRICS_ROOT / ".metric_deps"))
    sys.path.insert(0, str(OFFICIAL_METRICS_ROOT))
    from utils.eval_utils import (  # pylint: disable=import-outside-toplevel
        compute_absolute_volume_difference,
        compute_dice_f1_instance_difference,
    )

    truth = np.zeros((32, 32, 32), dtype=bool)
    truth[2:7, 2:7, 2:7] = True
    truth[20:24, 20:24, 20:24] = True
    prediction = np.zeros_like(truth)
    prediction[2:7, 2:7, 2:7] = True
    prediction[20:24, 20:24, 22:26] = True
    prediction[10:12, 24:26, 4:6] = True
    truth_labels, number_truth = cc3d.connected_components(truth, return_N=True)
    prediction_labels, number_prediction = cc3d.connected_components(
        prediction, return_N=True
    )
    context = overlap_context(
        truth, truth_labels, number_truth, prediction_labels, number_prediction
    )
    keep = np.ones(number_prediction + 1, dtype=bool)
    keep[0] = False
    bins = np.full(number_truth + 1, 0, dtype=np.int8)
    fast = metrics_for_kept_components(context, keep, number_truth, 1.0, bins)
    official_f1, official_lcd, official_dice = compute_dice_f1_instance_difference(
        truth, prediction
    )
    official_avd = compute_absolute_volume_difference(
        truth, prediction, np.asarray(0.001)
    )
    expected = {
        "dice": float(official_dice),
        "lesion_f1": float(official_f1),
        "lesion_count_difference": float(official_lcd),
        "absolute_volume_difference_ml": float(official_avd),
    }
    for metric in METRICS:
        if not np.isclose(fast[metric], expected[metric]):
            raise AssertionError((metric, fast[metric], expected[metric]))
    print("Official metric self-test passed:", json.dumps(expected), flush=True)


def nondominated_indices(objectives: np.ndarray) -> list[int]:
    keep = []
    for index, value in enumerate(objectives):
        dominated = np.any(
            np.all(objectives <= value + 1e-12, axis=1)
            & np.any(objectives < value - 1e-12, axis=1)
        )
        if not dominated:
            keep.append(index)
    return keep


def config_metadata(candidates: list[dict]) -> dict[str, dict]:
    output = {}
    for candidate in candidates:
        for mode in MODES:
            name = f"{candidate['base_name']}__{mode['name']}"
            output[name] = {
                "candidate_index": candidate["index"],
                "weights": candidate["weights"],
                "threshold": candidate["threshold"],
                "screen_reason": candidate["reason"],
                "mode": mode,
            }
    return output


def summarize(results: list[dict], candidates: list[dict]) -> dict:
    results = sorted(results, key=lambda item: item["case"])
    names = sorted(results[0]["configurations"])
    folds = np.asarray([item["fold"] for item in results], dtype=np.int8)
    arrays = {
        metric: np.asarray(
            [
                [item["configurations"][name][metric] for name in names]
                for item in results
            ],
            dtype=np.float64,
        )
        for metric in METRICS
    }
    missed = np.asarray(
        [
            [item["configurations"][name]["missed_case"] for name in names]
            for item in results
        ],
        dtype=np.int16,
    )
    metadata = config_metadata(candidates)
    baseline_by_mode = {}
    baseline_candidate = next(
        candidate
        for candidate in candidates
        if candidate["reason"] == "mandatory_current3_baseline"
    )
    for mode in MODES:
        baseline_by_mode[mode["name"]] = (
            f"{baseline_candidate['base_name']}__{mode['name']}"
        )

    summary_configs = {}
    for column, name in enumerate(names):
        size_detection = {}
        for size_name, _, _ in SIZE_BINS_MM3:
            detected = sum(
                item["configurations"][name]["lesion_size_detection"][size_name][0]
                for item in results
            )
            total = sum(
                item["configurations"][name]["lesion_size_detection"][size_name][1]
                for item in results
            )
            size_detection[size_name] = {
                "detected": int(detected),
                "total": int(total),
                "recall": float(detected / total) if total else 1.0,
            }
        fold_means = {}
        for fold in sorted(set(folds.tolist())):
            selected = folds == fold
            fold_means[str(fold)] = {
                f"mean_{metric}": float(arrays[metric][selected, column].mean())
                for metric in METRICS
            }
        summary_configs[name] = {
            "metadata": metadata[name],
            **{
                f"mean_{metric}": float(arrays[metric][:, column].mean())
                for metric in METRICS
            },
            "missed_cases": int(missed[:, column].sum()),
            "lesion_size_detection": size_detection,
            "fold_means": fold_means,
        }

    objective = np.column_stack(
        (
            -arrays["dice"].mean(axis=0),
            -arrays["lesion_f1"].mean(axis=0),
            arrays["lesion_count_difference"].mean(axis=0),
            arrays["absolute_volume_difference_ml"].mean(axis=0),
        )
    )
    pareto_columns = nondominated_indices(objective)

    rank_sum = np.zeros(len(names), dtype=np.float64)
    for row in range(len(results)):
        for metric in METRICS:
            rank_sum += rankdata(
                -arrays[metric][row] * DIRECTIONS[metric], method="average"
            )
    average_rank = rank_sum / (len(results) * len(METRICS))
    balanced_column = int(np.argmin(average_rank))

    reference_baseline_name = baseline_by_mode["pp_s300_c065"]
    reference_baseline_column = names.index(reference_baseline_name)
    strict_columns = []
    for column in range(len(names)):
        favorable = [
            (arrays[metric][:, column].mean() - arrays[metric][:, reference_baseline_column].mean())
            * DIRECTIONS[metric]
            for metric in METRICS
        ]
        if min(favorable) > 1e-12 and missed[:, column].sum() <= missed[:, reference_baseline_column].sum():
            strict_columns.append(column)

    paired_vs_reference = {}
    for column, name in enumerate(names):
        comparison = {}
        for metric in METRICS:
            raw_delta = arrays[metric][:, column] - arrays[metric][:, reference_baseline_column]
            favorable = raw_delta * DIRECTIONS[metric]
            comparison[metric] = {
                "mean_raw_delta": float(raw_delta.mean()),
                "better_cases": int(np.count_nonzero(favorable > 1e-8)),
                "worse_cases": int(np.count_nonzero(favorable < -1e-8)),
                "ties": int(np.count_nonzero(np.abs(favorable) <= 1e-8)),
            }
        comparison["missed_case_delta"] = int(
            missed[:, column].sum() - missed[:, reference_baseline_column].sum()
        )
        paired_vs_reference[name] = comparison

    cross_fitted = []
    selected_by_case = np.full(len(results), -1, dtype=np.int32)
    unique_folds = sorted(set(folds.tolist()))
    if len(unique_folds) >= 2:
        for fold in unique_folds:
            train = np.flatnonzero(folds != fold)
            test = np.flatnonzero(folds == fold)
            train_rank = np.zeros(len(names), dtype=np.float64)
            for row in train:
                for metric in METRICS:
                    train_rank += rankdata(
                        -arrays[metric][row] * DIRECTIONS[metric], method="average"
                    )
            selected_column = int(np.argmin(train_rank))
            selected_by_case[test] = selected_column
            cross_fitted.append(
                {
                    "held_out_fold": int(fold),
                    "selected_configuration": names[selected_column],
                    "training_average_rank": float(
                        train_rank[selected_column] / (len(train) * len(METRICS))
                    ),
                    "held_out_means": {
                        f"mean_{metric}": float(arrays[metric][test, selected_column].mean())
                        for metric in METRICS
                    },
                    "held_out_missed_cases": int(missed[test, selected_column].sum()),
                }
            )
    if np.any(selected_by_case < 0):
        cross_fitted_summary = None
    else:
        rows = np.arange(len(results))
        cross_fitted_summary = {
            **{
                f"mean_{metric}": float(arrays[metric][rows, selected_by_case].mean())
                for metric in METRICS
            },
            "missed_cases": int(missed[rows, selected_by_case].sum()),
        }

    return {
        "protocol": {
            "n_cases": len(results),
            "folds": sorted(set(folds.tolist())),
            "metrics": list(METRICS),
            "lesion_matching": "current ISLES'26 one-to-one IoU >= 0.25",
            "selection": "per-case rank then mean over four binary metrics",
            "pr_auc_note": "PR-AUC is threshold-independent and evaluated separately for final soft-map weights",
        },
        "baseline_by_mode": baseline_by_mode,
        "configurations": summary_configs,
        "selection": {
            "balanced_rank": {
                "configuration": names[balanced_column],
                "average_rank": float(average_rank[balanced_column]),
            },
            "pareto_front": [names[column] for column in pareto_columns],
            "strict_improvements_vs_current3_docker": [
                names[column] for column in strict_columns
            ],
            "cross_fitted_by_fold": cross_fitted,
            "cross_fitted_overall": cross_fitted_summary,
        },
        "paired_vs_current3_docker": paired_vs_reference,
    }


def write_outputs(out_dir: Path, results: list[dict], summary: dict) -> None:
    with (out_dir / "per_case.jsonl").open("w", encoding="utf-8") as handle:
        for item in sorted(results, key=lambda value: value["case"]):
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "configuration",
            "resencm_weight",
            "dtk10_weight",
            "msl_weight",
            "threshold",
            "mode",
            "mean_dice",
            "mean_lesion_f1",
            "mean_lesion_count_difference",
            "mean_absolute_volume_difference_ml",
            "missed_cases",
            "tiny_recall",
            "small_recall",
            "medium_recall",
            "large_recall",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, values in sorted(summary["configurations"].items()):
            metadata = values["metadata"]
            size = values["lesion_size_detection"]
            writer.writerow(
                {
                    "configuration": name,
                    "resencm_weight": metadata["weights"]["resencm"],
                    "dtk10_weight": metadata["weights"]["dtk10"],
                    "msl_weight": metadata["weights"]["msl"],
                    "threshold": metadata["threshold"],
                    "mode": metadata["mode"]["name"],
                    "mean_dice": values["mean_dice"],
                    "mean_lesion_f1": values["mean_lesion_f1"],
                    "mean_lesion_count_difference": values["mean_lesion_count_difference"],
                    "mean_absolute_volume_difference_ml": values["mean_absolute_volume_difference_ml"],
                    "missed_cases": values["missed_cases"],
                    "tiny_recall": size["tiny_lt100_mm3"]["recall"],
                    "small_recall": size["small_100_999_mm3"]["recall"],
                    "medium_recall": size["medium_1000_9999_mm3"]["recall"],
                    "large_recall": size["large_ge10000_mm3"]["recall"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-json", type=Path, default=DEFAULT_SELECTED_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--max-cases-per-fold", type=int, default=0)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    folds = {int(value) for value in args.folds.split(",") if value.strip()}
    if not folds or not folds.issubset(set(range(5))):
        raise ValueError(f"Invalid folds: {folds}")
    candidates = load_candidates(args.selected_json, args.max_candidates)
    cases = collect_cases(folds)
    cases = evenly_sample_cases_per_fold(cases, args.max_cases_per_fold)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    print("Cases:", len(cases), "folds:", sorted(folds), flush=True)
    print("Candidates:", len(candidates), "configurations:", len(candidates) * len(MODES), flush=True)
    self_test_official()
    if args.check_only:
        first = cases[0]
        image = sitk.ReadImage(first["gt"])
        truth = sitk.GetArrayFromImage(image) > 0
        print("First case:", first["case"], truth.shape, image.GetSpacing(), flush=True)
        for name in MODEL_ORDER:
            probability = load_foreground_probability(Path(first[name]), truth.shape)
            print(name, probability.shape, float(probability.min()), float(probability.max()), flush=True)
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "cases": [[item["case"], item["fold"]] for item in cases],
        "candidates": candidates,
        "modes": list(MODES),
    }
    manifest_path = args.out_dir / "manifest.json"
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as handle:
            if json.load(handle) != manifest:
                raise RuntimeError(f"Existing manifest differs in {args.out_dir}")
    else:
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

    progress_path = args.out_dir / "progress.jsonl"
    completed = {}
    if progress_path.is_file():
        with progress_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                completed[item["case"]] = item
    pending = [item for item in cases if item["case"] not in completed]
    results = list(completed.values())
    print("Resuming:", len(results), "completed,", len(pending), "pending", flush=True)

    if args.workers == 1:
        init_worker(candidates)
        iterator = map(process_case, pending)
        pool = None
    else:
        pool = mp.Pool(
            processes=args.workers,
            initializer=init_worker,
            initargs=(candidates,),
            maxtasksperchild=4,
        )
        iterator = pool.imap_unordered(process_case, pending, chunksize=1)
    try:
        with progress_path.open("a", encoding="utf-8") as progress:
            for item in iterator:
                results.append(item)
                progress.write(json.dumps(item, sort_keys=True) + "\n")
                progress.flush()
                print(f"Completed {len(results)}/{len(cases)}: {item['case']}", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    if len(results) != len(cases):
        print(f"Partial run: {len(results)}/{len(cases)}", flush=True)
        return
    summary = summarize(results, candidates)
    write_outputs(args.out_dir, results, summary)
    print(json.dumps(summary["selection"], indent=2), flush=True)
    print("Outputs:", args.out_dir, flush=True)


if __name__ == "__main__":
    main()
