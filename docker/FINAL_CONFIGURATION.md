# ISLES'26 Final c09 Hybrid-Output Configuration

## Binary segmentation

- Model families: ResEncM / DTK10 / MSL / ICI, five folds each.
- Family weights: `0.31875 / 0.31875 / 0.2125 / 0.15`.
- Threshold: `0.425`.
- Components: 26-connectivity.
- Keep a component when its volume is at least `300 mm3` **or** its peak
  c09-fused probability is at least `0.65`.

## Lesion probability map

- Model families: ResEncM / DTK10 / MSL, five folds each.
- Family weights: `0.375 / 0.400 / 0.225`.
- Continuous float32 output clipped to `[0, 1]`.
- No threshold or connected-component postprocessing.

All four families are inferred once per case. The ICI prediction contributes
only to the binary-mask fusion; no model forward is duplicated.

The container reuses the validated 20-checkpoint `model.tar.gz` from the full
ICI10 package. The output calibration above is frozen in `inference.py`.
