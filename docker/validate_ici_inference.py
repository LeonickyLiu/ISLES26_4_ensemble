"""Exercise the packaged five-fold ICI model on one local synthetic case."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from inference import _find_single_input_image, _predict_family


MODEL_FOLDER = (
    Path("/opt/ml/model")
    / "nnUNet_results/Dataset004_ATLAS3_RAW/"
    "nnUNetTrainerICILoss__nnUNetPlans__3d_fullres"
)
INPUT_FOLDER = Path("/input/images/t1-brain-mri")


def main() -> None:
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    input_image = _find_single_input_image(INPUT_FOLDER)
    reference = sitk.ReadImage(str(input_image))
    reference_shape = tuple(
        int(value) for value in sitk.GetArrayViewFromImage(reference).shape
    )

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=False,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(MODEL_FOLDER),
        use_folds=(0, 1, 2, 3, 4),
        checkpoint_name="checkpoint_final.pth",
    )

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="ici5_validate_", dir="/tmp") as temporary:
        probability = _predict_family(
            predictor,
            input_image,
            Path(temporary) / "prediction",
            reference_shape,
        )
    elapsed = time.monotonic() - started

    if probability.shape != reference_shape:
        raise AssertionError("ICI probability shape mismatch")
    if probability.dtype != np.float32:
        raise AssertionError(f"ICI probability dtype is {probability.dtype}")
    if not np.all(np.isfinite(probability)):
        raise AssertionError("ICI probability contains NaN or infinity")
    if probability.min() < 0.0 or probability.max() > 1.0:
        raise AssertionError("ICI probability is outside [0, 1]")
    print(
        "ICI5 FORWARD PASSED: "
        f"device={device}, elapsed={elapsed:.2f}s, "
        f"range=({float(probability.min()):.6f}, {float(probability.max()):.6f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
