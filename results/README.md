# Aggregate out-of-fold results

Only aggregate search summaries are committed here. Per-case predictions,
probability arrays, ground-truth labels, and checkpoints are excluded.

- `core4_basis_preregistered_20260818_summary.csv` contains the frozen
  four-model weight/threshold candidate comparison.
- `core4_postprocess_refine_c09_20260819_summary.csv` contains the conservative
  component-filter refinement grid after c09 was fixed.
- `three_model_pr_auc_oof_summary.json` contains exact full-volume PR-AUC for
  the four shortlisted probability-map weight sets.
- `four_model_single_branch_pr_auc_oof_summary.json` contains exact full-volume
  PR-AUC for the unthresholded final four-model segmentation fusion.

For the final binary configuration (ResEncM/DTK10/MSL/ICI weights
`0.31875/0.31875/0.2125/0.15`, threshold `0.425`, component rule
`300 mm3 OR peak >= 0.65`), the 1,453-case five-fold OOF aggregate was:

| Metric | Value |
|---|---:|
| Mean Dice | 0.666484 |
| Mean lesion F1 | 0.616320 |
| Mean lesion-count difference | 1.793531 |
| Mean absolute-volume difference (mL) | 5.089944 |

For the final continuous probability map (ResEncM/DTK10/MSL weights
`0.375/0.400/0.225`), mean PR-AUC was `0.761344`, an absolute increase of
`0.001936` over equal three-model weights on the same OOF cases.

If a single fusion is preferred for all outputs, the unthresholded four-model
segmentation fusion (ResEncM/DTK10/MSL/ICI weights
`0.31875/0.31875/0.2125/0.15`) can also be saved directly as the continuous
probability map. Its mean PR-AUC was `0.762594`. In the paired comparison, the
four-model map was better on 582 cases, the submitted three-model map was
better on 866 cases (`59.6%`), and 5 cases tied. Thus the four-model output has
a slightly higher mean PR-AUC and is a valid single-branch alternative, while
the submitted dual-output mode retains the three-model probability map because
its per-case PR-AUC was higher more often. This comparison alone does not
determine the official rank.

These are internal out-of-fold validation results, not hidden-test leaderboard
scores. The computations used the organizer metric repository at commit
`e589d022953f797bdc6acc1ce9701f793dab295a`.
