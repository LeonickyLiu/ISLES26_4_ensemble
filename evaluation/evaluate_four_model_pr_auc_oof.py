#!/usr/bin/env python3
"""Evaluate the unthresholded final four-model fusion with official PR-AUC."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from evaluate_core4_weight_threshold_oof_v2_server import (
    MODEL_ORDER,
    collect_cases,
    load_foreground_probability,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get("ISLES26_ROOT", REPO_ROOT)).resolve()
OFFICIAL_METRICS_ROOT = Path(
    os.environ.get("ISLES26_OFFICIAL_METRICS", BASE / "isles26_official_metrics")
).resolve()
sys.path.insert(0, str(OFFICIAL_METRICS_ROOT / ".metric_deps"))
sys.path.insert(0, str(OFFICIAL_METRICS_ROOT))
from utils.eval_utils import compute_pr_auc  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs/final_output_calibration.json"
DEFAULT_OUT_DIR = BASE / "ensemble_results/final_oof_four_model_pr_auc"
WORKER_WEIGHTS: dict[str, float] = {}


def load_weights(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    weights = {
        name: float(payload["segmentation"]["weights"][name])
        for name in MODEL_ORDER
    }
    if any(value < 0.0 for value in weights.values()):
        raise ValueError("Fusion weights must be non-negative")
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("Fusion weights must sum to one")
    return weights


def init_worker(weights: dict[str, float]) -> None:
    global WORKER_WEIGHTS
    WORKER_WEIGHTS = weights


def process_case(item: dict) -> dict:
    truth = np.asarray(sitk.GetArrayFromImage(sitk.ReadImage(item["gt"]))) > 0
    blend = np.zeros(truth.shape, dtype=np.float32)
    for name in MODEL_ORDER:
        probability = load_foreground_probability(Path(item[name]), truth.shape)
        blend += np.float32(WORKER_WEIGHTS[name]) * probability
    np.clip(blend, 0.0, 1.0, out=blend)
    return {
        "case": item["case"],
        "fold": int(item["fold"]),
        "pr_auc": float(compute_pr_auc(truth, np.ascontiguousarray(blend))),
    }


def summarize(results: list[dict], weights: dict[str, float]) -> dict:
    values = np.asarray([item["pr_auc"] for item in results], dtype=np.float64)
    folds = np.asarray([item["fold"] for item in results], dtype=np.int8)
    return {
        "metric": "official voxel-wise PR-AUC via utils.eval_utils.compute_pr_auc",
        "n_cases": len(results),
        "weights": weights,
        "mean_pr_auc": float(np.mean(values)),
        "median_pr_auc": float(np.median(values)),
        "standard_deviation": float(np.std(values, ddof=1)),
        "fold_means": {
            str(fold): float(np.mean(values[folds == fold]))
            for fold in sorted(set(folds.tolist()))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()

    folds = {int(value) for value in args.folds.split(",") if value.strip()}
    weights = load_weights(args.config)
    cases = collect_cases(folds)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "cases": [[item["case"], item["fold"]] for item in cases],
        "weights": weights,
    }
    manifest_path = args.out_dir / "manifest.json"
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as handle:
            if json.load(handle) != manifest:
                raise RuntimeError("Existing PR-AUC manifest differs")
    else:
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

    progress_path = args.out_dir / "progress.jsonl"
    completed: dict[str, dict] = {}
    if progress_path.is_file():
        with progress_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                completed[item["case"]] = item
    pending = [item for item in cases if item["case"] not in completed]
    print(
        f"Cases={len(cases)} completed={len(completed)} pending={len(pending)}",
        flush=True,
    )

    pool = None
    if args.workers == 1:
        init_worker(weights)
        iterator = map(process_case, pending)
    else:
        pool = mp.Pool(
            args.workers,
            initializer=init_worker,
            initargs=(weights,),
            maxtasksperchild=2,
        )
        iterator = pool.imap_unordered(process_case, pending, chunksize=1)
    try:
        with progress_path.open("a", encoding="utf-8") as progress:
            for item in iterator:
                completed[item["case"]] = item
                progress.write(json.dumps(item, sort_keys=True) + "\n")
                progress.flush()
                print(f"Completed {len(completed)}/{len(cases)}: {item['case']}", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    if len(completed) != len(cases):
        return
    results = sorted(completed.values(), key=lambda item: item["case"])
    summary = summarize(results, weights)
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
