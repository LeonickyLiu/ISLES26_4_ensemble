"""Core local inference implementation for the final ISLES'26 method.

The module intentionally contains no challenge-platform or container code. It
loads the four trained nnU-Net families directly from ``nnUNet_results``, runs
their five-fold ensembles once, and constructs the two calibrated outputs.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from itertools import permutations
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import SimpleITK as sitk
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from scipy import ndimage


MODEL_ORDER = ("resencm", "dtk10", "msl", "ici")

MODEL_LAYOUTS = {
    "resencm": (
        "Dataset004_ATLAS3_RAW",
        "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres",
    ),
    "dtk10": (
        "Dataset004_ATLAS3_RAW",
        "nnUNetTrainerDiceTopK10Loss__nnUNetPlans__3d_fullres",
    ),
    "msl": (
        "Dataset005_ATLAS3_RAW_MSL",
        "nnUNetTrainer__nnUNetPlans__3d_fullres",
    ),
    "ici": (
        "Dataset004_ATLAS3_RAW",
        "nnUNetTrainerICILoss__nnUNetPlans__3d_fullres",
    ),
}


def load_calibration(path: str | Path) -> dict[str, Any]:
    """Load and validate the frozen final fusion configuration."""
    path = Path(path)
    calibration = json.loads(path.read_text(encoding="utf-8"))

    folds_by_model = calibration["folds_by_model"]
    if set(folds_by_model) != set(MODEL_ORDER):
        raise ValueError("Calibration contains unexpected model families")
    for name in MODEL_ORDER:
        if tuple(folds_by_model[name]) != (0, 1, 2, 3, 4):
            raise ValueError(f"Expected folds 0-4 for {name}")

    segmentation = calibration["segmentation"]
    probability_map = calibration["probability_map"]
    _validate_weights(segmentation["weights"], "segmentation")
    _validate_weights(probability_map["weights"], "probability map")

    if not 0.0 <= float(segmentation["threshold"]) <= 1.0:
        raise ValueError("Segmentation threshold must be in [0, 1]")
    postprocessing = segmentation["postprocessing"]
    if int(postprocessing["connectivity"]) != 26:
        raise ValueError("Only the frozen 26-connected post-processing is supported")
    if postprocessing["keep_rule"] != "volume_gte_minimum_or_peak_gte_threshold":
        raise ValueError("Unexpected connected-component keep rule")
    return calibration


def _validate_weights(weights: Mapping[str, float], label: str) -> None:
    if set(weights) != set(MODEL_ORDER):
        raise ValueError(f"{label} weights contain unexpected model families")
    if any(float(value) < 0.0 for value in weights.values()):
        raise ValueError(f"{label} weights must be non-negative")
    if not np.isclose(sum(float(value) for value in weights.values()), 1.0):
        raise ValueError(f"{label} weights must sum to one")


def model_folders(results_root: str | Path) -> dict[str, Path]:
    """Return the expected nnU-Net result folder for every family."""
    root = Path(results_root)
    return {
        name: root / dataset / trainer
        for name, (dataset, trainer) in MODEL_LAYOUTS.items()
    }


def validate_checkpoints(
    results_root: str | Path,
    calibration: Mapping[str, Any],
    checkpoint_name: str = "checkpoint_final.pth",
) -> dict[str, Path]:
    """Validate plans, metadata, and all checkpoints before allocating a GPU."""
    folders = model_folders(results_root)
    for name, folder in folders.items():
        for required in ("dataset.json", "plans.json"):
            required_path = folder / required
            if not required_path.is_file():
                raise FileNotFoundError(required_path)
        for fold in calibration["folds_by_model"][name]:
            checkpoint = folder / f"fold_{fold}" / checkpoint_name
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
    return folders


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve ``auto``, ``cpu``, ``cuda`` or an explicit CUDA device."""
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device")
    return device


def initialize_predictors(
    results_root: str | Path,
    calibration: Mapping[str, Any],
    checkpoint_name: str = "checkpoint_final.pth",
    device: str = "auto",
) -> tuple[dict[str, nnUNetPredictor], torch.device]:
    """Load all five folds for the four model families."""
    folders = validate_checkpoints(results_root, calibration, checkpoint_name)
    torch_device = resolve_device(device)
    print(f"PyTorch {torch.__version__}; inference device: {torch_device}")
    if torch_device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(torch_device)}")

    predictors: dict[str, nnUNetPredictor] = {}
    for name in MODEL_ORDER:
        folder = folders[name]
        folds = tuple(int(fold) for fold in calibration["folds_by_model"][name])
        print(f"Loading {name}: {folder} (folds={folds})")
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=False,
            device=torch_device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.initialize_from_trained_model_folder(
            str(folder),
            use_folds=folds,
            checkpoint_name=checkpoint_name,
        )
        predictors[name] = predictor
    return predictors, torch_device


def _align_spatial_array(array: np.ndarray, reference_shape: tuple[int, ...]) -> np.ndarray:
    if tuple(array.shape) == tuple(reference_shape):
        return array
    matches = [
        order
        for order in permutations(range(3))
        if tuple(array.shape[index] for index in order) == tuple(reference_shape)
    ]
    if len(matches) != 1:
        raise ValueError(f"Cannot align prediction shape {array.shape} to {reference_shape}")
    return np.transpose(array, matches[0])


def _load_foreground_probability(
    path: Path,
    reference_shape: tuple[int, ...],
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        probabilities = np.asarray(data["probabilities"], dtype=np.float32)
    if probabilities.ndim != 4 or probabilities.shape[0] < 2:
        raise ValueError(f"Unexpected probability array shape in {path}: {probabilities.shape}")

    # ResEncM, DTK10, and ICI have one foreground channel. MSL has four
    # mutually exclusive lesion-size foreground classes, which are summed back
    # into a single lesion probability for the challenge task.
    foreground = probabilities[1:].sum(axis=0, dtype=np.float32)
    foreground = _align_spatial_array(foreground, reference_shape)
    return np.ascontiguousarray(np.clip(foreground, 0.0, 1.0), dtype=np.float32)


def _predict_family(
    predictor: nnUNetPredictor,
    input_image: Path,
    output_dir: Path,
    reference_shape: tuple[int, ...],
) -> np.ndarray:
    output_dir.mkdir(parents=True, exist_ok=False)
    output_stem = output_dir / "prediction"
    predictor.predict_from_files(
        [[str(input_image)]],
        [str(output_stem)],
        save_probabilities=True,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
    )
    probability_path = output_stem.with_suffix(".npz")
    if not probability_path.is_file():
        raise FileNotFoundError(probability_path)
    return _load_foreground_probability(probability_path, reference_shape)


def component_postprocess(
    probability: np.ndarray,
    spacing: tuple[float, ...],
    threshold: float,
    minimum_volume_mm3: float,
    peak_probability: float,
) -> np.ndarray:
    """Apply the frozen 26-connected volume-or-confidence keep rule."""
    base_mask = probability >= threshold
    labels, number_components = ndimage.label(
        base_mask,
        structure=np.ones((3, 3, 3), dtype=np.uint8),
    )
    if number_components == 0:
        return base_mask.astype(np.uint8)

    component_ids = np.arange(1, number_components + 1)
    voxel_counts = np.bincount(labels.ravel(), minlength=number_components + 1)[1:]
    peaks = ndimage.maximum(probability, labels=labels, index=component_ids)
    volumes_mm3 = voxel_counts.astype(np.float64) * float(np.prod(spacing))
    keep = (volumes_mm3 >= minimum_volume_mm3) | (peaks >= peak_probability)
    lookup = np.zeros(number_components + 1, dtype=bool)
    lookup[component_ids] = keep
    return lookup[labels].astype(np.uint8)


def _write_image(path: Path, array: np.ndarray, reference: sitk.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference)
    sitk.WriteImage(image, str(path), useCompression=True)


def predict_case(
    input_image: str | Path,
    output_dir: str | Path,
    predictors: Mapping[str, nnUNetPredictor],
    calibration: Mapping[str, Any],
    device: torch.device,
    work_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Run one T1 image and write the final mask and continuous probability map."""
    input_image = Path(input_image).resolve()
    output_dir = Path(output_dir).resolve()
    if not input_image.is_file():
        raise FileNotFoundError(input_image)
    if set(predictors) != set(MODEL_ORDER):
        raise ValueError("Predictors must contain all four model families")

    reference = sitk.ReadImage(str(input_image))
    reference_shape = tuple(int(value) for value in sitk.GetArrayViewFromImage(reference).shape)
    segmentation_weights = calibration["segmentation"]["weights"]
    probability_weights = calibration["probability_map"]["weights"]

    temporary_parent = None if work_dir is None else str(Path(work_dir).resolve())
    if temporary_parent is not None:
        Path(temporary_parent).mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="isles26_ensemble_", dir=temporary_parent))
    try:
        segmentation_probability = np.zeros(reference_shape, dtype=np.float32)
        probability_map = np.zeros(reference_shape, dtype=np.float32)
        for name in MODEL_ORDER:
            print(f"Predicting {name}")
            foreground = _predict_family(
                predictors[name],
                input_image,
                run_dir / name,
                reference_shape,
            )
            segmentation_probability += np.float32(segmentation_weights[name]) * foreground
            probability_map += np.float32(probability_weights[name]) * foreground
            del foreground
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if not np.all(np.isfinite(segmentation_probability)):
            raise ValueError("Segmentation fusion contains NaN or infinity")
        if not np.all(np.isfinite(probability_map)):
            raise ValueError("Probability-map fusion contains NaN or infinity")
        np.clip(segmentation_probability, 0.0, 1.0, out=segmentation_probability)
        np.clip(probability_map, 0.0, 1.0, out=probability_map)

        postprocessing = calibration["segmentation"]["postprocessing"]
        binary = component_postprocess(
            segmentation_probability,
            reference.GetSpacing(),
            threshold=float(calibration["segmentation"]["threshold"]),
            minimum_volume_mm3=float(postprocessing["minimum_volume_mm3"]),
            peak_probability=float(postprocessing["peak_probability"]),
        )

        segmentation_path = output_dir / "stroke_lesion_segmentation.nii.gz"
        probability_path = output_dir / "lesion_probability_map.nii.gz"
        _write_image(segmentation_path, binary, reference)
        _write_image(
            probability_path,
            probability_map.astype(np.float32, copy=False),
            reference,
        )
        print(f"Segmentation: {segmentation_path}")
        print(f"Probability map: {probability_path}")
        print(f"Foreground voxels: {int(binary.sum())}")
        return segmentation_path, probability_path
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
