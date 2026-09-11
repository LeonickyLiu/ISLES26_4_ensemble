#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PYTHON_BIN="${PYTHON_BIN:-python}"

NNUNET_PACKAGE_DIR="${NNUNET_PACKAGE_DIR:-$("$PYTHON_BIN" -c \
  'from pathlib import Path; import nnunetv2; print(Path(nnunetv2.__file__).resolve().parent)')}"
TRAINER_DIR="$NNUNET_PACKAGE_DIR/training/nnUNetTrainer/variants/loss"
ICI_DIR="$NNUNET_PACKAGE_DIR/training/loss/ici_official"

mkdir -p "$TRAINER_DIR" "$ICI_DIR"
cp "$SCRIPT_DIR/nnunet_extensions/nnUNetTrainerTopkLoss.py" "$TRAINER_DIR/"
cp "$SCRIPT_DIR/nnunet_extensions/nnUNetTrainerICILoss.py" "$TRAINER_DIR/"
cp "$SCRIPT_DIR/nnunet_extensions/ici_official/ICI_loss.py" "$ICI_DIR/"
cp "$SCRIPT_DIR/nnunet_extensions/ici_official/__init__.py" "$ICI_DIR/"
cp "$SCRIPT_DIR/nnunet_extensions/ici_official/tools.py" "$ICI_DIR/"
cp "$SCRIPT_DIR/nnunet_extensions/ici_official/LICENSE" "$ICI_DIR/"

echo "Installed nnU-Net extensions into: $NNUNET_PACKAGE_DIR"
"$PYTHON_BIN" -c \
  'from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerTopkLoss import nnUNetTrainerDiceTopK10Loss; from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerICILoss import nnUNetTrainerICILoss; print("Extension import check passed")'
