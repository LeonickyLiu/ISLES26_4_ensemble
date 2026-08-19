#!/usr/bin/env python3
"""Convert ATLAS3 raw BIDS-like data to an nnU-Net v2 training dataset."""

import argparse
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


def find_case_files(raw_root: Path):
    cases = []
    for image_path in sorted(raw_root.rglob("*_space-orig_desc-brain_T1w.nii.gz")):
        anat_dir = image_path.parent
        mask_candidates = sorted(anat_dir.glob("*_space-orig_label-lesion_desc-T1lesion_mask.nii.gz"))
        if len(mask_candidates) != 1:
            raise RuntimeError(f"Expected one lesion mask for {image_path}, found {len(mask_candidates)}")
        cases.append((image_path, mask_candidates[0]))
    if not cases:
        raise RuntimeError(f"No raw ATLAS3 cases found under {raw_root}")
    return cases


def make_label_on_image_grid(image_path: Path, mask_path: Path, output_path: Path) -> bool:
    image = nib.load(str(image_path))
    mask = nib.load(str(mask_path))

    if len(image.shape) != 3 or len(mask.shape) != 3:
        raise RuntimeError(f"Only 3D images and masks are supported: {image_path}, {mask_path}")

    needs_resample = image.shape != mask.shape or not np.allclose(
        image.affine, mask.affine, rtol=0.0, atol=1e-3
    )
    if needs_resample:
        aligned = resample_from_to(mask, (image.shape, image.affine), order=0)
        mask_data = np.asarray(aligned.dataobj)
        affine = image.affine
        header = aligned.header.copy()
    else:
        mask_data = np.asarray(mask.dataobj)
        affine = mask.affine
        header = mask.header.copy()

    mask_data = (mask_data > 0).astype(np.uint8)
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(mask_data, affine, header=header), str(output_path))
    return needs_resample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    out_root = args.out_root.resolve()
    images_tr = out_root / "imagesTr"
    labels_tr = out_root / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    cases = find_case_files(raw_root)
    resampled_count = 0
    empty_count = 0

    for case_index, (image_path, mask_path) in enumerate(cases):
        case_name = f"case_{case_index:04d}"
        image_out = images_tr / f"{case_name}_0000.nii.gz"
        label_out = labels_tr / f"{case_name}.nii.gz"

        image = nib.load(str(image_path))
        mask = nib.load(str(mask_path))
        if len(mask.shape) != 3:
            raise RuntimeError(f"Invalid mask shape for {mask_path}: {mask.shape}")
        if not np.any(np.asarray(mask.dataobj) > 0):
            empty_count += 1

        shutil.copy2(image_path, image_out)
        if make_label_on_image_grid(image_path, mask_path, label_out):
            resampled_count += 1

    dataset_json = {
        "channel_names": {"0": "T1w"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
        "name": "ATLAS3_RAW",
        "description": "ATLAS3 raw space brain T1w images with lesion masks",
    }
    (out_root / "dataset.json").write_text(
        json.dumps(dataset_json, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Converted cases: {len(cases)}")
    print(f"Empty masks retained: {empty_count}")
    print(f"Masks resampled to image grid: {resampled_count}")
    print(f"Output: {out_root}")


if __name__ == "__main__":
    main()
