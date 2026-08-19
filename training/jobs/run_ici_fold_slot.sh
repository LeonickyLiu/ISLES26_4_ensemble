#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 GPU_ID PRIMARY_FOLD" >&2
    exit 2
fi

GPU_ID="$1"
PRIMARY_FOLD="$2"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE="${ISLES26_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
LOG_DIR="$BASE/logs"
TRAIN="${NNUNET_TRAIN_BIN:-nnUNetv2_train}"
RESULT_ROOT="$BASE/nnUNet_results/Dataset004_ATLAS3_RAW/nnUNetTrainerICILoss__nnUNetPlans__3d_fullres"
FOLD4_CLAIM="$LOG_DIR/ici_fold4_claim.lock"

export nnUNet_raw="$BASE/nnUNet_raw"
export nnUNet_preprocessed="$BASE/nnUNet_preprocessed"
export nnUNet_results="$BASE/nnUNet_results"
export nnUNet_compile=false
export nnUNet_n_proc_DA=4
export nnUNet_def_n_proc=4
export PYTHONPATH="$BASE/nnUNet${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$LOG_DIR"

run_fold() {
    local fold="$1"
    local result_dir="$RESULT_ROOT/fold_${fold}"
    local log="$LOG_DIR/train_raw_ici_fold${fold}_gpu${GPU_ID}.log"
    local -a command=(
        "$TRAIN"
        4
        3d_fullres
        "$fold"
        -tr
        nnUNetTrainerICILoss
        --npz
    )

    if [[ -f "$result_dir/checkpoint_final.pth" ]]; then
        printf '[%s] fold %s already has checkpoint_final.pth; skipping training.\n' \
            "$(date '+%F %T')" "$fold" >>"$log"
        return 0
    fi

    if [[ -f "$result_dir/checkpoint_latest.pth" ]]; then
        command+=(--c)
    fi

    printf '[%s] starting ICI fold %s on physical GPU %s\n' \
        "$(date '+%F %T')" "$fold" "$GPU_ID" >>"$log"
    CUDA_VISIBLE_DEVICES="$GPU_ID" "${command[@]}" >>"$log" 2>&1
    local status=$?
    printf '[%s] ICI fold %s exited with status %s\n' \
        "$(date '+%F %T')" "$fold" "$status" >>"$log"
    return "$status"
}

if run_fold "$PRIMARY_FOLD"; then
    if mkdir "$FOLD4_CLAIM" 2>/dev/null; then
        printf 'claimed_by_gpu=%s primary_fold=%s claimed_at=%s\n' \
            "$GPU_ID" "$PRIMARY_FOLD" "$(date '+%F %T')" \
            >"$FOLD4_CLAIM/owner.txt"
        run_fold 4
    fi
fi
