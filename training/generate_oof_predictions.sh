#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
BASE="${ISLES26_ROOT:-$REPO_ROOT}"
TRAIN="${NNUNET_TRAIN_BIN:-nnUNetv2_train}"

export nnUNet_raw="$BASE/nnUNet_raw"
export nnUNet_preprocessed="$BASE/nnUNet_preprocessed"
export nnUNet_results="$BASE/nnUNet_results"
export nnUNet_compile=false

read -r -a FAMILIES_ARRAY <<< "${FAMILIES:-resencm dtk10 msl ici}"
read -r -a FOLDS_ARRAY <<< "${FOLDS:-0 1 2 3 4}"

validate_one() {
    local family="$1"
    local fold="$2"
    case "$family" in
        resencm)
            "$TRAIN" 4 3d_fullres "$fold" \
                -p nnUNetResEncUNetMPlans --val --npz
            ;;
        dtk10)
            "$TRAIN" 4 3d_fullres "$fold" \
                -tr nnUNetTrainerDiceTopK10Loss --val --npz
            ;;
        msl)
            "$TRAIN" 5 3d_fullres "$fold" --val --npz
            ;;
        ici)
            "$TRAIN" 4 3d_fullres "$fold" \
                -tr nnUNetTrainerICILoss --val --npz
            ;;
        *)
            echo "Unknown family: $family" >&2
            return 2
            ;;
    esac
}

for family in "${FAMILIES_ARRAY[@]}"; do
    for fold in "${FOLDS_ARRAY[@]}"; do
        echo "Generating OOF probabilities for family=$family fold=$fold"
        validate_one "$family" "$fold"
    done
done
