#!/usr/bin/env python3
"""Exact current-official PR-AUC evaluation for final three-model weights."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from evaluate_three_model_weight_candidates_oof import (
    BASE,
    MODEL_ORDER,
    collect_cases,
    load_foreground_probability,
)

OFFICIAL_METRICS_ROOT = Path(
    os.environ.get("ISLES26_OFFICIAL_METRICS", BASE / "isles26_official_metrics")
).resolve()
sys.path.insert(0, str(OFFICIAL_METRICS_ROOT / ".metric_deps"))
sys.path.insert(0, str(OFFICIAL_METRICS_ROOT))
from utils.eval_utils import compute_pr_auc


DEFAULT_CANDIDATES = BASE / "final_weight_candidates.json"
DEFAULT_OUT_DIR = BASE / "ensemble_results/three_model_weight_search_cv/exact_pr_auc_oof"
WORKER_CANDIDATES: list[dict] = []


def load_soft_map_candidates(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    source = payload.get("selected_configs", payload)
    output = []
    seen = set()
    for item in source:
        weights = {name: float(item["weights"][name]) for name in MODEL_ORDER}
        key = tuple(round(weights[name], 8) for name in MODEL_ORDER)
        if key in seen:
            continue
        seen.add(key)
        output.append(
            {
                "name": (
                    f"r{weights['resencm']:.4f}_d{weights['dtk10']:.4f}"
                    f"_m{weights['msl']:.4f}"
                ),
                "weights": weights,
                "reason": item.get("screen_reason", "finalist"),
            }
        )
    return output


def init_worker(candidates: list[dict]) -> None:
    global WORKER_CANDIDATES
    WORKER_CANDIDATES = candidates


def process_case(item: dict) -> dict:
    truth = np.asarray(
        sitk.GetArrayFromImage(sitk.ReadImage(item["gt"]))
    ) > 0
    probabilities = {
        name: load_foreground_probability(Path(item[name]), truth.shape)
        for name in MODEL_ORDER
    }
    values = {}
    for candidate in WORKER_CANDIDATES:
        weights = candidate["weights"]
        blend = np.ascontiguousarray(
            weights["resencm"] * probabilities["resencm"]
            + weights["dtk10"] * probabilities["dtk10"]
            + weights["msl"] * probabilities["msl"],
            dtype=np.float32,
        )
        values[candidate["name"]] = float(compute_pr_auc(truth, blend))
    return {"case": item["case"], "fold": int(item["fold"]), "pr_auc": values}


def summarize(results: list[dict], candidates: list[dict]) -> dict:
    names = [item["name"] for item in candidates]
    folds = np.asarray([item["fold"] for item in results], dtype=np.int8)
    arrays = {
        name: np.asarray([item["pr_auc"][name] for item in results], dtype=np.float64)
        for name in names
    }
    baseline = names[0]
    output = {
        "protocol": {
            "status": "exact_full_volume_current_official_compute_pr_auc",
            "n_cases": len(results),
            "baseline": baseline,
        },
        "candidates": {},
    }
    for candidate in candidates:
        name = candidate["name"]
        values = arrays[name]
        delta = values - arrays[baseline]
        output["candidates"][name] = {
            "weights": candidate["weights"],
            "reason": candidate["reason"],
            "mean_pr_auc": float(np.nanmean(values)),
            "median_pr_auc": float(np.nanmedian(values)),
            "mean_delta_vs_baseline": float(np.nanmean(delta)),
            "better_cases": int(np.count_nonzero(delta > 1e-8)),
            "worse_cases": int(np.count_nonzero(delta < -1e-8)),
            "ties": int(np.count_nonzero(np.abs(delta) <= 1e-8)),
            "fold_means": {
                str(fold): float(np.nanmean(values[folds == fold]))
                for fold in sorted(set(folds.tolist()))
            },
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()

    folds = {int(value) for value in args.folds.split(",") if value.strip()}
    candidates = load_soft_map_candidates(args.candidates)
    cases = collect_cases(folds)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "cases": [[item["case"], item["fold"]] for item in cases],
        "candidates": candidates,
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
    completed = {}
    if progress_path.is_file():
        with progress_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                completed[item["case"]] = item
    pending = [item for item in cases if item["case"] not in completed]
    results = list(completed.values())
    print(
        f"Cases={len(cases)} candidates={len(candidates)} "
        f"completed={len(results)} pending={len(pending)}",
        flush=True,
    )
    if args.workers == 1:
        init_worker(candidates)
        iterator = map(process_case, pending)
        pool = None
    else:
        pool = mp.Pool(
            args.workers,
            initializer=init_worker,
            initargs=(candidates,),
            maxtasksperchild=2,
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
        return
    results.sort(key=lambda item: item["case"])
    summary = summarize(results, candidates)
    with (args.out_dir / "per_case.jsonl").open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
