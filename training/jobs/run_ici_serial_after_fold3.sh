#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE="${ISLES26_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
GPU_ID=3
LOG_DIR="$BASE/logs"
TRAIN="${NNUNET_TRAIN_BIN:-nnUNetv2_train}"
RESULT_ROOT="$BASE/nnUNet_results/Dataset004_ATLAS3_RAW/nnUNetTrainerICILoss__nnUNetPlans__3d_fullres"

export nnUNet_raw="$BASE/nnUNet_raw"
export nnUNet_preprocessed="$BASE/nnUNet_preprocessed"
export nnUNet_results="$BASE/nnUNet_results"
export nnUNet_compile=false
export nnUNet_n_proc_DA=12
export nnUNet_def_n_proc=8
export PYTHONUNBUFFERED=1
export PYTHONPATH="$BASE/nnUNet${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$LOG_DIR"

run_fold() {
    local fold="$1"
    local result_dir="$RESULT_ROOT/fold_${fold}"
    local log="$LOG_DIR/train_raw_ici_fold${fold}_serial_gpu${GPU_ID}.log"
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
        printf '[%s] fold %s already complete; skipping.\n' \
            "$(date '+%F %T')" "$fold" >>"$log"
        return 0
    fi

    if [[ -f "$result_dir/checkpoint_latest.pth" ]]; then
        command+=(--c)
    fi

    printf '[%s] starting serial ICI fold %s on physical GPU %s\n' \
        "$(date '+%F %T')" "$fold" "$GPU_ID" >>"$log"
    CUDA_VISIBLE_DEVICES="$GPU_ID" "${command[@]}" >>"$log" 2>&1
    local status=$?
    printf '[%s] serial ICI fold %s exited with status %s\n' \
        "$(date '+%F %T')" "$fold" "$status" >>"$log"
    return "$status"
}

while tmux has-session -t ici_fold3_gpu3 2>/dev/null; do
    sleep 60
done

if [[ ! -f "$RESULT_ROOT/fold_3/checkpoint_final.pth" ]]; then
    printf '[%s] fold 3 session ended without checkpoint_final.pth; serial queue stopped.\n' \
        "$(date '+%F %T')" >>"$LOG_DIR/train_raw_ici_serial_queue.log"
    exit 1
fi

for fold in 1 2 4; do
    if ! run_fold "$fold"; then
        printf '[%s] fold %s failed; serial queue stopped.\n' \
            "$(date '+%F %T')" "$fold" >>"$LOG_DIR/train_raw_ici_serial_queue.log"
        exit 1
    fi
done
