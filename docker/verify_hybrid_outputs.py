"""Verify the two calibrated output branches without running model forwards."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

import inference


def main() -> None:
    output_root = Path("/tmp/hybrid_output_test")
    shutil.rmtree(output_root, ignore_errors=True)
    inference.OUTPUT_PATH = output_root

    input_path = inference._find_single_input_image(Path("/input/images/t1-brain-mri"))
    reference = sitk.ReadImage(str(input_path))
    shape = tuple(int(value) for value in sitk.GetArrayViewFromImage(reference).shape)
    family_values = {
        "resencm": 0.8,
        "dtk10": 0.4,
        "msl": 0.2,
        "ici": 1.0,
    }

    def fake_predictor(_predictor, _input_image, output_dir, reference_shape):
        assert tuple(reference_shape) == shape
        return np.full(shape, family_values[output_dir.name], dtype=np.float32)

    inference._predict_family = fake_predictor
    model = {
        "config": {
            "model_order": list(inference.EXPECTED_MODEL_ORDER),
            "folds_by_model": {name: [0, 1, 2, 3, 4] for name in inference.EXPECTED_MODEL_ORDER},
        },
        "device": torch.device("cpu"),
        "predictors": {name: object() for name in inference.EXPECTED_MODEL_ORDER},
    }
    inference.interf0_handler(model)

    binary_image = sitk.ReadImage(
        str(output_root / "images/stroke-lesion-segmentation/output.mha")
    )
    probability_image = sitk.ReadImage(
        str(output_root / "images/lesion-probability-map/output.mha")
    )
    binary = sitk.GetArrayFromImage(binary_image)
    probability = sitk.GetArrayFromImage(probability_image)

    # c09 mask fusion = 0.575, so every voxel exceeds threshold 0.425.
    assert np.all(binary == 1)
    # Three-model map fusion = 0.505; ICI=1.0 must not alter this output.
    assert probability.dtype == np.float32
    assert np.allclose(probability, 0.505, rtol=0, atol=1e-6)
    for image in (binary_image, probability_image):
        assert image.GetSize() == reference.GetSize()
        assert image.GetSpacing() == reference.GetSpacing()
        assert image.GetOrigin() == reference.GetOrigin()
        assert image.GetDirection() == reference.GetDirection()

    print("HYBRID TWO-OUTPUT TEST PASSED: mask_fusion=0.575 probability_map=0.505")


if __name__ == "__main__":
    main()
