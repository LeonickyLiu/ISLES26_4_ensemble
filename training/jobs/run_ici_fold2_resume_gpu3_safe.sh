#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE="${ISLES26_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
GPU_ID=3
TRAIN="${NNUNET_TRAIN_BIN:-nnUNetv2_train}"
RESULT_DIR="$BASE/nnUNet_results/Dataset004_ATLAS3_RAW/nnUNetTrainerICILoss__nnUNetPlans__3d_fullres/fold_2"
LOG="$BASE/logs/train_raw_ici_fold2_resume_gpu3_safe_20260811.log"

export nnUNet_raw="$BASE/nnUNet_raw"
export nnUNet_preprocessed="$BASE/nnUNet_preprocessed"
export nnUNet_results="$BASE/nnUNet_results"
export nnUNet_compile=false
export nnUNet_n_proc_DA=4
export nnUNet_def_n_proc=4
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$BASE/nnUNet${PYTHONPATH:+:$PYTHONPATH}"

if [[ -f "$RESULT_DIR/checkpoint_final.pth" ]]; then
    printf '[%s] fold 2 already complete; nothing to resume.\n' \
        "$(date '+%F %T')" >>"$LOG"
    exit 0
fi

command=(
    "$TRAIN"
    4
    3d_fullres
    2
    -tr
    nnUNetTrainerICILoss
    --npz
)
if [[ -f "$RESULT_DIR/checkpoint_latest.pth" ]]; then
    command+=(--c)
fi

printf '[%s] starting fold 2 on physical GPU %s; DA workers=4, default workers=4\n' \
    "$(date '+%F %T')" "$GPU_ID" >>"$LOG"
CUDA_VISIBLE_DEVICES="$GPU_ID" "${command[@]}" >>"$LOG" 2>&1
status=$?
printf '[%s] fold 2 exited with status %s\n' \
    "$(date '+%F %T')" "$status" >>"$LOG"
exit "$status"
