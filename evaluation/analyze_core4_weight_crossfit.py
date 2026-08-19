#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np


BASE = Path(__file__).resolve().parent
INPUT = BASE / "core4_weight_finalists_per_case.jsonl"
OUTPUT = BASE / "core4_weight_crossfit_summary.json"
METRICS = (
    "dice",
    "lesion_f1",
    "lesion_count_difference",
    "absolute_volume_difference_ml",
)
BASELINE = "core4_r0.250_d0.250_m0.250_i0.250_pp_fixed"


def rank_cost(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    return ranks


def choose_candidate(train_values: np.ndarray, mode: str) -> int:
    means = train_values.mean(axis=0)
    costs = np.column_stack((-means[:, 0], -means[:, 1], means[:, 2], means[:, 3]))
    if mode == "official_binary3":
        columns = (1, 2, 3)
    elif mode == "balanced4":
        columns = (0, 1, 2, 3)
    else:
        raise ValueError(mode)
    score = sum(rank_cost(costs[:, column]) for column in columns)
    best = np.flatnonzero(score == score.min())
    if best.size == 1:
        return int(best[0])
    return int(best[np.argmax(means[best, 0])])


def pareto_front(means: np.ndarray) -> list[int]:
    costs = np.column_stack((-means[:, 0], -means[:, 1], means[:, 2], means[:, 3]))
    keep = []
    for index in range(costs.shape[0]):
        dominated = any(
            other != index
            and np.all(costs[other] <= costs[index])
            and np.any(costs[other] < costs[index])
            for other in range(costs.shape[0])
        )
        if not dominated:
            keep.append(index)
    return keep


def main() -> None:
    with INPUT.open("r", encoding="utf-8") as handle:
        cases = [json.loads(line) for line in handle if line.strip()]
    cases.sort(key=lambda item: item["case"])
    names = sorted(cases[0]["binary"])
    values = np.asarray(
        [
            [
                [case["binary"][name][metric] for metric in METRICS]
                for name in names
            ]
            for case in cases
        ],
        dtype=np.float64,
    )
    baseline_index = names.index(BASELINE)
    baseline_mean = values[:, baseline_index].mean(axis=0)
    full_means = values.mean(axis=0)

    output = {
        "n_cases": len(cases),
        "n_candidates": len(names),
        "metrics": METRICS,
        "baseline": BASELINE,
        "baseline_mean": baseline_mean.tolist(),
        "full_data_best": {
            metric: names[int(np.argmax(full_means[:, index]))]
            if index < 2
            else names[int(np.argmin(full_means[:, index]))]
            for index, metric in enumerate(METRICS)
        },
        "pareto_front": [names[index] for index in pareto_front(full_means)],
        "repeated_crossfit": {},
    }

    for mode in ("official_binary3", "balanced4"):
        selection_counts: Counter[str] = Counter()
        repeat_means = []
        repeat_deltas = []
        for repeat in range(100):
            rng = np.random.default_rng(20260802 + repeat)
            shuffled = rng.permutation(len(cases))
            folds = np.array_split(shuffled, 5)
            selected_values = np.empty((len(cases), len(METRICS)), dtype=np.float64)
            for fold in folds:
                train_mask = np.ones(len(cases), dtype=bool)
                train_mask[fold] = False
                selected = choose_candidate(values[train_mask], mode)
                selection_counts[names[selected]] += 1
                selected_values[fold] = values[fold, selected]
            mean = selected_values.mean(axis=0)
            repeat_means.append(mean)
            repeat_deltas.append(mean - baseline_mean)
        repeat_means = np.asarray(repeat_means)
        repeat_deltas = np.asarray(repeat_deltas)
        output["repeated_crossfit"][mode] = {
            "selection_counts": dict(selection_counts.most_common()),
            "mean_oof_metrics": repeat_means.mean(axis=0).tolist(),
            "mean_delta_vs_equal": repeat_deltas.mean(axis=0).tolist(),
            "delta_95_interval_across_repeats": np.percentile(
                repeat_deltas, [2.5, 97.5], axis=0
            ).T.tolist(),
        }

    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
