#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE="${ISLES26_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PREDICT="${NNUNET_PREDICT_BIN:-nnUNetv2_predict}"
WORK="$BASE/data/isles26_corrections_20260728/case0578_oof"
INPUT="$WORK/input"
BACKUP="$BASE/data/backup_before_isles26_corrections_20260728/oof_case0578"

export nnUNet_raw="$BASE/nnUNet_raw"
export nnUNet_preprocessed="$BASE/nnUNet_preprocessed"
export nnUNet_results="$BASE/nnUNet_results"
export nnUNet_compile=false
export PYTHONPATH="$BASE/nnUNet${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$INPUT" "$BACKUP"
ln -sfn \
    "$BASE/nnUNet_raw/Dataset004_ATLAS3_RAW/imagesTr/case_0578_0000.nii.gz" \
    "$INPUT/case_0578_0000.nii.gz"

replace_prediction() {
    local output=$1
    local validation=$2
    local tag=$3
    local backup_dir="$BACKUP/$tag"
    mkdir -p "$backup_dir"

    for suffix in .nii.gz .npz .pkl; do
        local source="$output/case_0578$suffix"
        local target="$validation/case_0578$suffix"
        test -s "$source"
        if [[ ! -e "$backup_dir/case_0578$suffix" ]]; then
            cp -a "$target" "$backup_dir/"
        fi
        cp -p "$source" "$target.corrected"
        mv -f "$target.corrected" "$target"
    done
}

run_model() {
    local tag=$1
    local dataset=$2
    local plans=$3
    local trainer=$4
    local validation=$5
    local output="$WORK/$tag"

    mkdir -p "$output"
    CUDA_VISIBLE_DEVICES=3 "$PREDICT" \
        -i "$INPUT" \
        -o "$output" \
        -d "$dataset" \
        -p "$plans" \
        -tr "$trainer" \
        -c 3d_fullres \
        -f 2 \
        -npp 1 \
        -nps 1 \
        --save_probabilities \
        --disable_progress_bar
    replace_prediction "$output" "$validation" "$tag"
}

run_model \
    resencm \
    4 \
    nnUNetResEncUNetMPlans \
    nnUNetTrainer \
    "$BASE/nnUNet_results/Dataset004_ATLAS3_RAW/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_2/validation"

run_model \
    dtk10 \
    4 \
    nnUNetPlans \
    nnUNetTrainerDiceTopK10Loss \
    "$BASE/nnUNet_results/Dataset004_ATLAS3_RAW/nnUNetTrainerDiceTopK10Loss__nnUNetPlans__3d_fullres/fold_2/validation"

run_model \
    msl \
    5 \
    nnUNetPlans \
    nnUNetTrainer \
    "$BASE/nnUNet_results/Dataset005_ATLAS3_RAW_MSL/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2/validation"

printf '%s\n' "case_0578 OOF predictions replaced successfully"
