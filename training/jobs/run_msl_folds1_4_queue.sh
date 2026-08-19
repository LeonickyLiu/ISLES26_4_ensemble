#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE="${ISLES26_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
LOG_DIR="$BASE/logs"
RESULT_ROOT="$BASE/nnUNet_results/Dataset005_ATLAS3_RAW_MSL/nnUNetTrainer__nnUNetPlans__3d_fullres"
TRAIN="${NNUNET_TRAIN_BIN:-nnUNetv2_train}"
MANAGER_LOG="$LOG_DIR/train_raw_msl_folds1_4_queue_20260725.log"
LOCK_FILE="$LOG_DIR/train_raw_msl_folds1_4_queue.lock"

export nnUNet_raw="$BASE/nnUNet_raw"
export nnUNet_preprocessed="$BASE/nnUNet_preprocessed"
export nnUNet_results="$BASE/nnUNet_results"

mkdir -p "$LOG_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "An MSL folds 1-4 queue is already running."
    exit 1
fi

log() {
    printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$MANAGER_LOG"
}

MIN_FREE_MIB=22000
MAX_UTILIZATION=20
POLL_SECONDS=60
MAX_ATTEMPTS=2

gpu_ready() {
    local gpu=$1
    local values free_mib utilization
    values=$(nvidia-smi -i "$gpu" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null) || return 1
    IFS=',' read -r free_mib utilization <<< "$values"
    free_mib=${free_mib//[[:space:]]/}
    utilization=${utilization//[[:space:]]/}
    [[ "$free_mib" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]] || return 1
    (( free_mib >= MIN_FREE_MIB && utilization <= MAX_UTILIZATION ))
}

wait_for_gpu() {
    local -a ready_count=(0 0 0 0)
    local gpu
    while true; do
        for gpu in 0 1 2 3; do
            if gpu_ready "$gpu"; then
                ready_count[$gpu]=$(( ready_count[$gpu] + 1 ))
            else
                ready_count[$gpu]=0
            fi
            if (( ready_count[$gpu] >= 2 )); then
                printf '%s\n' "$gpu"
                return 0
            fi
        done
        sleep "$POLL_SECONDS"
    done
}

log "serial queue started; only one MSL fold will run at a time"
log "waiting for a GPU with at least ${MIN_FREE_MIB} MiB free"

for fold in 1 2 3 4; do
    fold_dir="$RESULT_ROOT/fold_$fold"
    fold_log="$LOG_DIR/train_raw_msl_fold${fold}_20260725.log"
    if [[ -f "$fold_dir/checkpoint_final.pth" ]]; then
        log "fold $fold already complete; skipping"
        continue
    fi

    attempt=0
    while (( attempt < MAX_ATTEMPTS )); do
        gpu=$(wait_for_gpu)
        attempt=$(( attempt + 1 ))
        command=("$TRAIN" 5 3d_fullres "$fold" --npz)
        if [[ -f "$fold_dir/checkpoint_latest.pth" ]]; then
            command+=(--c)
            log "continuing fold $fold on GPU $gpu (attempt $attempt)"
        else
            log "starting fold $fold on GPU $gpu (attempt $attempt)"
        fi

        CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}" > "$fold_log" 2>&1
        status=$?
        if (( status == 0 )) && [[ -f "$fold_dir/checkpoint_final.pth" ]]; then
            log "fold $fold completed successfully on GPU $gpu"
            break
        fi
        log "fold $fold exited with status $status; waiting to resume"
    done

    if [[ ! -f "$fold_dir/checkpoint_final.pth" ]]; then
        log "fold $fold failed after $MAX_ATTEMPTS attempts; serial queue stopped"
        exit 1
    fi
done

log "all requested MSL folds completed"
