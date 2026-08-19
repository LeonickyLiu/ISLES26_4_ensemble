"""ISLES'26 final five-fold hybrid-output inference implementation.

All four model families use folds 0-4. The binary segmentation is produced
from the c09 fusion (0.31875/0.31875/0.2125/0.15), while the submitted
probability map uses the three-model fusion (0.375/0.400/0.225). Each family
is inferred only once and its probability map is accumulated into the
applicable outputs.
"""

from __future__ import annotations

import glob
import json
import shutil
import tempfile
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from scipy import ndimage


INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_PATH = Path("/opt/ml/model")
CONFIG_PATH = MODEL_PATH / "ensemble_config.json"

MODEL_FOLDERS = {
    "resencm": MODEL_PATH
    / "nnUNet_results/Dataset004_ATLAS3_RAW/"
    "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres",
    "dtk10": MODEL_PATH
    / "nnUNet_results/Dataset004_ATLAS3_RAW/"
    "nnUNetTrainerDiceTopK10Loss__nnUNetPlans__3d_fullres",
    "msl": MODEL_PATH
    / "nnUNet_results/Dataset005_ATLAS3_RAW_MSL/"
    "nnUNetTrainer__nnUNetPlans__3d_fullres",
    "ici": MODEL_PATH
    / "nnUNet_results/Dataset004_ATLAS3_RAW/"
    "nnUNetTrainerICILoss__nnUNetPlans__3d_fullres",
}

EXPECTED_MODEL_ORDER = ("resencm", "dtk10", "msl", "ici")

# Final output calibration is deliberately owned by this container rather
# than ensemble_config.json. This lets the already validated 20-checkpoint
# model resource be reused without repacking or re-uploading several GB.
# The model archive was already validated and uploaded with the earlier ICI10
# calibration. Keep validating its identity, but apply the final calibration
# below in the container so that the model resource itself stays unchanged.
EXPECTED_PACKAGED_WEIGHTS = {
    "resencm": 0.3,
    "dtk10": 0.3,
    "msl": 0.3,
    "ici": 0.1,
}
SEGMENTATION_WEIGHTS = {
    "resencm": 0.31875,
    "dtk10": 0.31875,
    "msl": 0.2125,
    "ici": 0.15,
}
PROBABILITY_MAP_WEIGHTS = {
    "resencm": 0.375,
    "dtk10": 0.4,
    "msl": 0.225,
    "ici": 0.0,
}
SEGMENTATION_THRESHOLD = 0.425
MINIMUM_COMPONENT_VOLUME_MM3 = 300.0
COMPONENT_PEAK_PROBABILITY = 0.65


def _show_runtime_info(device: torch.device) -> None:
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Inference device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")


def _load_and_validate_config() -> dict[str, Any]:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Missing model configuration: {CONFIG_PATH}")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    expected_models = set(EXPECTED_MODEL_ORDER)
    if set(config["weights"]) != expected_models:
        raise ValueError("ensemble_config.json has unexpected model names")
    if tuple(config["model_order"]) != EXPECTED_MODEL_ORDER:
        raise ValueError("ensemble_config.json has unexpected model order")
    if not np.isclose(sum(config["weights"].values()), 1.0):
        raise ValueError("Ensemble weights must sum to one")
    for name, expected_weight in EXPECTED_PACKAGED_WEIGHTS.items():
        if not np.isclose(config["weights"][name], expected_weight):
            raise ValueError(f"Unexpected packaged ICI10 weight for {name}")
    if not np.isclose(float(config["threshold"]), SEGMENTATION_THRESHOLD):
        raise ValueError("Unexpected packaged ICI10 threshold")
    if not np.isclose(sum(SEGMENTATION_WEIGHTS.values()), 1.0):
        raise ValueError("Segmentation weights must sum to one")
    if not np.isclose(sum(PROBABILITY_MAP_WEIGHTS.values()), 1.0):
        raise ValueError("Probability-map weights must sum to one")
    if set(config["folds_by_model"]) != expected_models:
        raise ValueError("ensemble_config.json has unexpected fold definitions")
    expected_folds = {
        "resencm": (0, 1, 2, 3, 4),
        "dtk10": (0, 1, 2, 3, 4),
        "msl": (0, 1, 2, 3, 4),
        "ici": (0, 1, 2, 3, 4),
    }
    for name, folds in expected_folds.items():
        if tuple(config["folds_by_model"][name]) != folds:
            raise ValueError(f"Unexpected folds for {name}")

    for name, folder in MODEL_FOLDERS.items():
        for required in ("dataset.json", "plans.json"):
            if not (folder / required).is_file():
                raise FileNotFoundError(folder / required)
        for fold in config["folds_by_model"][name]:
            checkpoint = folder / f"fold_{fold}" / config["checkpoint_name"]
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
        print(f"Validated model resource: {name} -> {folder}")
    print(
        "Frozen output calibration: "
        f"mask_weights={SEGMENTATION_WEIGHTS}, "
        f"threshold={SEGMENTATION_THRESHOLD}, "
        f"minimum_volume_mm3={MINIMUM_COMPONENT_VOLUME_MM3}, "
        f"peak_probability={COMPONENT_PEAK_PROBABILITY}; "
        f"probability_map_weights={PROBABILITY_MAP_WEIGHTS}"
    )
    return config


def init_model() -> dict[str, Any]:
    """Load the configured fold parameter sets for each model family."""
    config = _load_and_validate_config()
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    _show_runtime_info(device)

    predictors: dict[str, nnUNetPredictor] = {}
    for name, folder in MODEL_FOLDERS.items():
        print(f"Initializing {name} from {folder}")
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
            str(folder),
            use_folds=tuple(config["folds_by_model"][name]),
            checkpoint_name=config["checkpoint_name"],
        )
        predictors[name] = predictor

    print("All four model families initialized")
    return {"config": config, "device": device, "predictors": predictors}


def run(model: dict[str, Any]) -> int:
    interface_key = get_interface_key()
    handler = {("stroke-metadata", "t1-brain-mri"): interf0_handler}[interface_key]
    return handler(model)


def _find_single_input_image(location: Path) -> Path:
    matches: list[str] = []
    for pattern in ("*.mha", "*.nii.gz", "*.nii"):
        matches.extend(glob.glob(str(location / pattern)))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one T1 image in {location}, found {len(matches)}")
    return Path(matches[0])


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


def _load_foreground_probability(path: Path, reference_shape: tuple[int, ...]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        probabilities = np.asarray(data["probabilities"], dtype=np.float32)
    if probabilities.ndim != 4 or probabilities.shape[0] < 2:
        raise ValueError(f"Unexpected probability array shape in {path}: {probabilities.shape}")

    # Binary ResEncM/DTK10 have one foreground channel. MSL has four lesion-size
    # channels, all of which represent foreground for the challenge output.
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
    output_truncated = output_dir / "prediction"
    predictor.predict_from_files(
        [[str(input_image)]],
        [str(output_truncated)],
        save_probabilities=True,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
    )
    probability_path = output_truncated.with_suffix(".npz")
    if not probability_path.is_file():
        raise FileNotFoundError(probability_path)
    return _load_foreground_probability(probability_path, reference_shape)


def _component_postprocess(
    probability: np.ndarray,
    spacing: tuple[float, ...],
    threshold: float,
    minimum_volume_mm3: float,
    peak_probability: float,
) -> np.ndarray:
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


def interf0_handler(model: dict[str, Any]) -> int:
    input_image_path = _find_single_input_image(INPUT_PATH / "images/t1-brain-mri")
    reference_image = sitk.ReadImage(str(input_image_path))
    reference_shape = tuple(int(value) for value in sitk.GetArrayViewFromImage(reference_image).shape)
    print(f"Input image: {input_image_path.name}, shape={reference_shape}")

    metadata_path = INPUT_PATH / "stroke-metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        print(f"Metadata fields available: {sorted(metadata)}")

    run_dir = Path(tempfile.mkdtemp(prefix="isles26_hybrid_final_", dir="/tmp"))
    try:
        segmentation_probability = np.zeros(reference_shape, dtype=np.float32)
        probability_map = np.zeros(reference_shape, dtype=np.float32)
        for name in model["config"]["model_order"]:
            folds = tuple(model["config"]["folds_by_model"][name])
            print(f"Running {name} fold ensemble: {folds}")
            foreground = _predict_family(
                model["predictors"][name],
                input_image_path,
                run_dir / name,
                reference_shape,
            )
            segmentation_probability += np.float32(SEGMENTATION_WEIGHTS[name]) * foreground
            probability_map += np.float32(PROBABILITY_MAP_WEIGHTS[name]) * foreground
            del foreground
            if model["device"].type == "cuda":
                torch.cuda.empty_cache()

        if not np.all(np.isfinite(segmentation_probability)):
            raise ValueError("Segmentation fusion contains NaN or infinity")
        if not np.all(np.isfinite(probability_map)):
            raise ValueError("Probability-map fusion contains NaN or infinity")
        np.clip(segmentation_probability, 0.0, 1.0, out=segmentation_probability)
        np.clip(probability_map, 0.0, 1.0, out=probability_map)

        binary = _component_postprocess(
            segmentation_probability,
            reference_image.GetSpacing(),
            threshold=SEGMENTATION_THRESHOLD,
            minimum_volume_mm3=MINIMUM_COMPONENT_VOLUME_MM3,
            peak_probability=COMPONENT_PEAK_PROBABILITY,
        )

        write_array_as_image_file(
            OUTPUT_PATH / "images/stroke-lesion-segmentation",
            binary,
            reference_image,
        )
        write_array_as_image_file(
            OUTPUT_PATH / "images/lesion-probability-map",
            probability_map.astype(np.float32, copy=False),
            reference_image,
        )
        print(
            f"Outputs written. Foreground voxels={int(binary.sum())}, "
            f"segmentation-fusion range=({float(segmentation_probability.min()):.6f}, "
            f"{float(segmentation_probability.max()):.6f}), "
            f"probability-map range=({float(probability_map.min()):.6f}, "
            f"{float(probability_map.max()):.6f})"
        )
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    return 0


def get_interface_key() -> tuple[str, ...]:
    inputs = json.loads((INPUT_PATH / "inputs.json").read_text(encoding="utf-8"))
    return tuple(sorted(item["socket"]["slug"] for item in inputs))


def write_array_as_image_file(
    location: Path,
    array: np.ndarray,
    reference_image: sitk.Image,
) -> None:
    location.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference_image)
    sitk.WriteImage(image, str(location / "output.mha"), useCompression=True)


if __name__ == "__main__":
    raise SystemExit(run(init_model()))
