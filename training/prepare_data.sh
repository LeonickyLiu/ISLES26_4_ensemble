#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /path/to/official/corrected/training-data" >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
BASE="${ISLES26_ROOT:-$REPO_ROOT}"
SOURCE_DATA="$1"
PYTHON_BIN="${PYTHON_BIN:-python}"
FINGERPRINT_BIN="${NNUNET_FINGERPRINT_BIN:-nnUNetv2_extract_fingerprint}"
PREPROCESS_BIN="${NNUNET_PREPROCESS_BIN:-nnUNetv2_preprocess}"
REPRO_DIR="$REPO_ROOT/reproducibility/nnunet"

export nnUNet_raw="$BASE/nnUNet_raw"
export nnUNet_preprocessed="$BASE/nnUNet_preprocessed"
export nnUNet_results="$BASE/nnUNet_results"

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

"$PYTHON_BIN" "$SCRIPT_DIR/data/convert_raw_to_nnUNet.py" \
    --raw-root "$SOURCE_DATA" \
    --out-root "$nnUNet_raw/Dataset004_ATLAS3_RAW"

"$PYTHON_BIN" "$SCRIPT_DIR/data/create_msl_dataset_raw.py"

# Validate the converted datasets once. The frozen fingerprint, plans and split
# below then make the preprocessing and folds match the submitted experiment.
"$FINGERPRINT_BIN" -d 4 5 --verify_dataset_integrity

for dataset in Dataset004_ATLAS3_RAW Dataset005_ATLAS3_RAW_MSL; do
    mkdir -p "$nnUNet_preprocessed/$dataset"
    cp "$REPRO_DIR/$dataset/dataset.json" "$nnUNet_preprocessed/$dataset/"
    cp "$REPRO_DIR/$dataset/dataset_fingerprint.json" "$nnUNet_preprocessed/$dataset/"
    cp "$REPRO_DIR/$dataset/nnUNetPlans.json" "$nnUNet_preprocessed/$dataset/"
    cp "$REPRO_DIR/$dataset/splits_final.json" "$nnUNet_preprocessed/$dataset/"
done
cp "$REPRO_DIR/Dataset004_ATLAS3_RAW/nnUNetResEncUNetMPlans.json" \
    "$nnUNet_preprocessed/Dataset004_ATLAS3_RAW/"

"$PREPROCESS_BIN" -d 4 -plans_name nnUNetPlans -c 3d_fullres -np 4
"$PREPROCESS_BIN" -d 5 -plans_name nnUNetPlans -c 3d_fullres -np 4

echo "Prepared Dataset004 and Dataset005 under: $BASE"
