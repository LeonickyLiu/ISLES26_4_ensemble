# Training reproduction

The submitted method contains four five-fold nnU-Net families (20 checkpoints
in total). Raw challenge data and trained checkpoints are not redistributed.

## 1. Environment

The original training environment used Python 3.10.20, PyTorch 2.6.0 with
CUDA 11.8, and nnU-Net v2.8.0. On a Linux CUDA host:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-training.txt
./training/install_nnunet_extensions.sh
```

## 2. Data

Use the organizer-provided training data including the corrections released on
2026-07-28. The converter expects the released BIDS-like filenames ending in
`_space-orig_desc-brain_T1w.nii.gz` and the corresponding lesion masks.

Choose a work directory and prepare Dataset004 plus its MSL relabeling:

```bash
export ISLES26_ROOT=/path/to/isles26-work
./training/prepare_data.sh /path/to/official/corrected/training-data
```

This command validates the converted data and installs the frozen fingerprint,
plans, and five-fold split from `reproducibility/nnunet/` before preprocessing.
The split contains 1,453 case identifiers and has SHA256
`5b89628f3e4ecc196967578c6b3813fb9099063ffc6c4d4c3a8af348fc1fc2c7`.

## 3. Training

Run all four families and all five folds serially on the visible GPU:

```bash
CUDA_VISIBLE_DEVICES=0 ./training/train_all_folds.sh
```

The full run is intentionally serial and takes substantial time. Independent
families or folds can be distributed across GPUs or cluster jobs, for example:

```bash
CUDA_VISIBLE_DEVICES=0 FAMILIES="resencm dtk10" FOLDS="0 1" \
  ./training/train_all_folds.sh
CUDA_VISIBLE_DEVICES=1 FAMILIES="msl ici" FOLDS="2 3 4" \
  ./training/train_all_folds.sh
```

Set `RESUME=1` to pass nnU-Net's `--c` flag when resuming existing runs.
Validation softmax arrays are retained with `--npz` because they are required
for the out-of-fold ensemble evaluation.

The four commands used by the driver are:

```text
nnUNetv2_train 4 3d_fullres FOLD -p nnUNetResEncUNetMPlans --npz
nnUNetv2_train 4 3d_fullres FOLD -tr nnUNetTrainerDiceTopK10Loss --npz
nnUNetv2_train 5 3d_fullres FOLD --npz
nnUNetv2_train 4 3d_fullres FOLD -tr nnUNetTrainerICILoss --npz
```

The frozen 3D plans use 1 mm isotropic spacing, a `128 x 128 x 128` patch and
batch size 2. The original nnU-Net trainer schedule used 1,000 epochs and 250
iterations per epoch.

## 4. Official evaluation code

The evaluation scripts were run against commit
`e589d022953f797bdc6acc1ce9701f793dab295a` of the organizer metric repository:

```bash
git clone https://github.com/ezequieldlrosa/isles26.git \
  "$ISLES26_ROOT/isles26_official_metrics"
git -C "$ISLES26_ROOT/isles26_official_metrics" checkout \
  e589d022953f797bdc6acc1ce9701f793dab295a
export ISLES26_OFFICIAL_METRICS="$ISLES26_ROOT/isles26_official_metrics"
```
