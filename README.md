# ISLES'26 Four-Model Ensemble

Code used for the final ISLES'26 submission from team FORSIAT. The system is a
20-checkpoint ensemble: four nnU-Net model families, each trained with five
folds. Dataset files, trained weights, validation predictions, and Docker image
archives are intentionally excluded from this repository.

## Final inference configuration

| Output | Models and family weights | Decision rule |
|---|---|---|
| Stroke lesion segmentation | ResEncM `0.31875`, DTK10 `0.31875`, MSL `0.2125`, ICI `0.15` | threshold `0.425`; keep each 26-connected component when volume is at least `300 mm3` **or** peak fused probability is at least `0.65` |
| Lesion probability map | ResEncM `0.375`, DTK10 `0.400`, MSL `0.225` | continuous float32 map clipped to `[0, 1]`; no thresholding or component filtering |

All four model families are inferred once. The ICI output contributes only to
the binary branch. The exact implementation is in `docker/inference.py`, and a
machine-readable copy of the calibration is in
`configs/final_output_calibration.json`.

## Repository layout

```text
configs/                  frozen search and final calibration configurations
docker/                   Grand Challenge invoke API, inference, build/test tools
evaluation/               five-fold OOF fusion and post-processing evaluation
reproducibility/          frozen nnU-Net plans, fingerprint and five-fold split
results/                  aggregate OOF search summaries (no per-case data)
training/data/            dataset conversion and MSL target construction
training/jobs/            fold training/resume scripts
training/nnunet_extensions/
                          custom DTK10 and ICI nnU-Net training components
requirements-training.txt exact training environment used for the final models
```

## Model families

- **ResEncM**: `nnUNetTrainer` with `nnUNetResEncUNetMPlans` on Dataset004.
- **DTK10**: `nnUNetTrainerDiceTopK10Loss` with `nnUNetPlans` on Dataset004.
- **MSL**: standard trainer on Dataset005, whose targets split connected lesions
  into four size classes; inference sums all foreground classes.
- **ICI**: `nnUNetTrainerICILoss` with `nnUNetPlans` on Dataset004.

The training setup used nnU-Net v2.8.0. Install the extension files under the
matching paths in an nnU-Net checkout/environment before training:

```text
training/nnunet_extensions/nnUNetTrainerTopkLoss.py
  -> nnunetv2/training/nnUNetTrainer/variants/loss/
training/nnunet_extensions/nnUNetTrainerICILoss.py
  -> nnunetv2/training/nnUNetTrainer/variants/loss/
training/nnunet_extensions/ici_official/
  -> nnunetv2/training/loss/ici_official/
```

Set `ISLES26_ROOT` to a work directory containing `nnUNet_raw`,
`nnUNet_preprocessed`, and `nnUNet_results`. The job scripts also accept
`NNUNET_TRAIN_BIN` (and, where relevant, `NNUNET_PREDICT_BIN`) so no user-specific
paths are required.

For a clean reproduction from the organizer-provided corrected training data,
including environment creation, exact preprocessing metadata and all 20 training
runs, follow [`training/README.md`](training/README.md). The repository freezes
the exact five-fold split and plans used for the submitted models.

## Evaluation protocol

Fusion weights and post-processing were selected only with five-fold
out-of-fold validation predictions. The final search was preregistered in two
stages:

1. Freeze and compare a small set of four-model weight/threshold candidates.
2. Freeze c09, then compare a conservative `5 x 3` component-filter grid.

Selection used repeated/five-fold cross-fitted per-case ranks across Dice,
lesion F1, lesion-count difference, and absolute-volume difference. The sanity
check cases were not used for selection. The probability-map weights were
evaluated separately with the official PR-AUC implementation.

The evaluation scripts expect the official metric code in
`$ISLES26_OFFICIAL_METRICS` (or `$ISLES26_ROOT/isles26_official_metrics`). This
dependency is not vendored here. Evaluation used organizer repository commit
`e589d022953f797bdc6acc1ce9701f793dab295a`. Aggregate OOF results and their
scope are documented in [`results/README.md`](results/README.md).

## Package the model resource

The model resource contains 20 slimmed `checkpoint_final.pth` files plus the
nnU-Net plans and dataset metadata. Create it outside the repository:

```bash
python docker/prepare_model.py \
  --results-root /path/to/nnUNet_results \
  --output-root /path/to/model

python docker/validate_model.py --model-root /path/to/model
tar -czvf algorithmmodel.tar.gz -C /path/to/model .
```

The submitted model resource retained the earlier `0.30/0.30/0.30/0.10`
identity and `200 mm3` legacy post-processing metadata in
`ensemble_config.json`. The final c09 binary calibration (`300 mm3`) and the
separate probability-map calibration are deliberately frozen in the container;
`docker/inference.py` validates the compatible model identity before applying
them.

## Build and test the container

Docker with NVIDIA Container Toolkit is required for GPU inference.

```bash
cd docker
./do_build.sh

MODEL_DIR=/path/to/model \
TEST_INPUT_DIR=/path/to/test/input \
TEST_OUTPUT_DIR=/path/to/test/output \
./do_test_run.sh

./do_save.sh
```

The container implements the Grand Challenge `invoke` API and writes both
`stroke-lesion-segmentation` and `lesion-probability-map` outputs. 

## License

Project code is released under Apache-2.0. The ICI training-loss source retains
its bundled Apache-2.0 license. nnU-Net and other dependencies remain subject to
their respective licenses.
