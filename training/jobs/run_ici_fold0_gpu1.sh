#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE="${ISLES26_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
LOG_DIR="$BASE/logs"
TRAIN="${NNUNET_TRAIN_BIN:-nnUNetv2_train}"
RESULT_DIR="$BASE/nnUNet_results/Dataset004_ATLAS3_RAW/nnUNetTrainerICILoss__nnUNetPlans__3d_fullres/fold_0"
LOG="$LOG_DIR/train_raw_ici_fold0_20260728.log"

export nnUNet_raw="$BASE/nnUNet_raw"
export nnUNet_preprocessed="$BASE/nnUNet_preprocessed"
export nnUNet_results="$BASE/nnUNet_results"
export nnUNet_compile=false
export PYTHONPATH="$BASE/nnUNet${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$LOG_DIR"

command=(
    "$TRAIN"
    4
    3d_fullres
    0
    -tr
    nnUNetTrainerICILoss
    --npz
)
if [[ -f "$RESULT_DIR/checkpoint_latest.pth" ]]; then
    command+=(--c)
fi

CUDA_VISIBLE_DEVICES=1 "${command[@]}" >"$LOG" 2>&1
