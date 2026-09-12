# ISLES'26 Four-Model Ensemble

This repository reproduces the lesion-segmentation algorithm developed by team
**FORSIAT** for ISLES'26. The final system combines four complementary 3D
nnU-Net families over five folds (20 checkpoints), then produces a binary
lesion mask and a separately calibrated continuous probability map.

The repository contains the complete data-conversion, training, inference, and
out-of-fold evaluation code. Challenge data, trained checkpoints, and generated
predictions are not redistributed.

## Method

### Four complementary model families

All models take one skull-stripped T1-weighted MRI volume as input and use the
same frozen five-fold split of 1,453 training cases.

| Family | Dataset / plan | Training objective or target |
|---|---|---|
| **ResEncM** | Dataset004, `nnUNetResEncUNetMPlans` | Standard nnU-Net loss with a residual-encoder M configuration |
| **DTK10** | Dataset004, `nnUNetPlans` | Dice + Top-K cross-entropy; the cross-entropy term concentrates on the hardest 10% of voxels |
| **MSL** | Dataset005, `nnUNetPlans` | Multi-size lesion targets: each 26-connected lesion is assigned to `<100`, `100-999`, `1000-9999`, or `>=10000` voxels |
| **ICI** | Dataset004, `nnUNetPlans` | Compound global, instance, and center supervision with weights `0.25/0.50/0.25` |

Dataset005 uses the same images and folds as Dataset004. Only its targets are
changed. At inference, the four MSL foreground-class probabilities are summed
back into one lesion probability.

The frozen 3D plans use 1 mm isotropic spacing, a `128 x 128 x 128` patch,
batch size 2, and the original nnU-Net schedule of 1,000 epochs with 250
iterations per epoch.

### Five-fold prediction and dual-output fusion

For family $m$, its probability is the average of the five fold models:

$$
p_m(x) = \frac{1}{5}\sum_{f=0}^{4} p_{m,f}(x).
$$

The binary-mask branch fuses all four families:

$$
p_{\mathrm{seg}} = 0.31875p_{\mathrm{ResEncM}} + 0.31875p_{\mathrm{DTK10}} + 0.2125p_{\mathrm{MSL}} + 0.15p_{\mathrm{ICI}}.
$$

It first thresholds $p_{\mathrm{seg}}$ at `0.425`. For every 26-connected
component $C$, the component is retained when either

$$
\operatorname{volume}(C)\geq300\;\mathrm{mm}^3
\quad\text{or}\quad
\max_{x\in C}p_{\mathrm{seg}}(x)\geq0.65.
$$

This removes small low-confidence false positives without discarding small
high-confidence lesions.

The continuous probability-map branch excludes ICI and uses:

$$
p_{\mathrm{prob}} = 0.375p_{\mathrm{ResEncM}} + 0.400p_{\mathrm{DTK10}} + 0.225p_{\mathrm{MSL}}.
$$

`p_prob` is clipped to `[0, 1]` and saved as `float32`; no threshold or
connected-component filtering is applied. Each family is inferred only once,
and the same family probability is accumulated into the applicable branches.
The exact calibration is frozen in
[`configs/final_output_calibration.json`](configs/final_output_calibration.json).

## Installation

The reproduction commands target a Linux CUDA host. Clone the repository and
create the pinned Conda environment named `isles`:

```bash
git clone https://github.com/LeonickyLiu/ISLES26_4_ensemble.git
cd ISLES26_4_ensemble

conda env create --file environment.yml
conda activate isles
python -m pip install --upgrade pip
python -m pip install -r requirements-training.txt
./training/install_nnunet_extensions.sh
```

If the environment already exists, activate it and rerun the two `pip`
commands. The original environment used Python 3.10.20, PyTorch 2.6.0 with
CUDA 11.8, and nnU-Net v2.8.0.

## Prepare the data

Use the organizer-provided training data including the 2026-07-28 corrections.
Choose a work directory with enough space for the raw data, preprocessing,
checkpoints, and validation probabilities:

```bash
export ISLES26_ROOT=/path/to/isles26-work
./training/prepare_data.sh /path/to/official/corrected/training-data
```

The script performs four reproducible steps:

1. Convert the released BIDS-like T1 images and lesion masks to Dataset004.
2. Align a mask to its image grid with nearest-neighbor resampling when needed.
3. Construct the four-class MSL targets as Dataset005.
4. Install the frozen fingerprint, plans, and exact five-fold split, then run
   nnU-Net preprocessing.

The split SHA256 is
`5b89628f3e4ecc196967578c6b3813fb9099063ffc6c4d4c3a8af348fc1fc2c7`.

## Train the final ensemble

Train all four families and all five folds serially on one visible GPU:

```bash
CUDA_VISIBLE_DEVICES=0 ./training/train_all_folds.sh
```

The final trained algorithm is the collection of 20
`checkpoint_final.pth` files under `$ISLES26_ROOT/nnUNet_results`; no additional
weight merging is needed. Independent jobs can be distributed across GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 FAMILIES="resencm dtk10" FOLDS="0 1" \
  ./training/train_all_folds.sh

CUDA_VISIBLE_DEVICES=1 FAMILIES="msl ici" FOLDS="2 3 4" \
  ./training/train_all_folds.sh
```

Use `RESUME=1` to resume existing nnU-Net runs. The driver keeps validation
softmax arrays with `--npz`, because they are needed for out-of-fold ensemble
evaluation. Exact family commands and expected output paths are documented in
[`training/README.md`](training/README.md).

## Run inference

After all 20 checkpoints are available, run one T1 volume directly from the
nnU-Net results tree:

```bash
conda activate isles

python -m inference.predict \
  --input-image /path/to/case.nii.gz \
  --results-root "$ISLES26_ROOT/nnUNet_results" \
  --output-dir /path/to/prediction \
  --device cuda:0
```

The command writes:

- `stroke_lesion_segmentation.nii.gz`: final post-processed binary mask.
- `lesion_probability_map.nii.gz`: final continuous probability map.

Both outputs preserve the input image geometry. The implementation is split
between [`inference/ensemble.py`](inference/ensemble.py), which contains model
loading, fusion and post-processing, and
[`inference/predict.py`](inference/predict.py), which provides the command-line
interface.

## Reproduce the five-fold evaluation

If the checkpoints exist but `fold_N/validation/case_*.npz` was removed,
regenerate out-of-fold probabilities without retraining:

```bash
CUDA_VISIBLE_DEVICES=0 ./training/generate_oof_predictions.sh
```

Install the evaluation dependency and check out the exact organizer metric
implementation used by the experiments:

```bash
python -m pip install -r requirements-evaluation.txt

git clone https://github.com/ezequieldlrosa/isles26.git \
  "$ISLES26_ROOT/isles26_official_metrics"
git -C "$ISLES26_ROOT/isles26_official_metrics" checkout \
  e589d022953f797bdc6acc1ce9701f793dab295a
export ISLES26_OFFICIAL_METRICS="$ISLES26_ROOT/isles26_official_metrics"
```

First validate that all ground-truth masks and four-family OOF probabilities
are aligned, then compute the final binary metrics:

```bash
python evaluation/evaluate_core4_postprocess_refine_server.py \
  --selected-json configs/core4_postprocess_refine_c09_20260819.json \
  --out-dir "$ISLES26_ROOT/ensemble_results/final_oof_binary" \
  --folds 0,1,2,3,4 --workers 4 --check-only

python evaluation/evaluate_core4_postprocess_refine_server.py \
  --selected-json configs/core4_postprocess_refine_c09_20260819.json \
  --out-dir "$ISLES26_ROOT/ensemble_results/final_oof_binary" \
  --folds 0,1,2,3,4 --workers 4
```

Evaluate the continuous probability-map branch separately:

```bash
python evaluation/evaluate_three_model_pr_auc_oof.py \
  --candidates configs/final_probability_map_candidates.json \
  --out-dir "$ISLES26_ROOT/ensemble_results/final_oof_pr_auc" \
  --folds 0,1,2,3,4 --workers 4
```

The binary evaluation writes `summary.csv` and `summary.json`; the PR-AUC
evaluation writes `summary.json`. Completed binary cases are recorded in
`progress.jsonl`, so that run can safely resume after interruption.

## Results

Fusion weights and post-processing were selected only from five-fold
out-of-fold predictions. A small preregistered four-model candidate set was
evaluated first; after c09 was fixed, a conservative `5 x 3` component-filter
grid was evaluated. The sanity-check cases were not used for model selection.

Final aggregate results over all 1,453 OOF cases are:

| Output | Metric | Mean |
|---|---|---:|
| Binary mask | Dice | **0.666484** |
| Binary mask | Lesion F1 | **0.616320** |
| Binary mask | Lesion-count difference | **1.793531** |
| Binary mask | Absolute-volume difference (mL) | **5.089944** |
| Probability map | PR-AUC | **0.761344** |

The final probability weights improved mean PR-AUC by `0.001936` over equal
three-model weights on the same OOF cases. These are internal OOF validation
results, not scores from a hidden test set. Frozen aggregate search summaries
and scope notes are available in [`results/`](results/README.md).

## Implementation map

```text
configs/                  frozen search and final calibration configurations
inference/                model loading, five-fold fusion and local prediction
evaluation/               OOF weight, threshold and post-processing evaluation
training/data/            official-data conversion and MSL target construction
training/jobs/            individual fold training/resume scripts
training/nnunet_extensions/
                          DTK10 and ICI objectives used by nnU-Net
reproducibility/          frozen plans, fingerprint and exact five-fold split
results/                  aggregate OOF experiment summaries
environment.yml           Conda environment definition
requirements-*.txt        pinned training and evaluation dependencies
```

## License

Project code is released under Apache-2.0. The bundled ICI loss source retains
its Apache-2.0 license. nnU-Net and other dependencies remain subject to their
respective licenses.
