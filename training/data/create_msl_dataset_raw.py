#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

BASE = Path(
    os.environ.get("ISLES26_ROOT", Path(__file__).resolve().parents[2])
).resolve()
SRC_RAW = BASE / 'nnUNet_raw/Dataset004_ATLAS3_RAW'
SRC_PREP = BASE / 'nnUNet_preprocessed/Dataset004_ATLAS3_RAW'
DST_RAW = BASE / 'nnUNet_raw/Dataset005_ATLAS3_RAW_MSL'
DST_PREP = BASE / 'nnUNet_preprocessed/Dataset005_ATLAS3_RAW_MSL'

# StrokeLesSeg MSL thresholds: tiny < 100, small < 1000, medium < 10000, large >= 10000 voxels.
THRESHOLDS = (100, 1000, 10000)
LABELS = {
    'background': 0,
    'tiny_lesion_lt100': 1,
    'small_lesion_100_999': 2,
    'medium_lesion_1000_9999': 3,
    'large_lesion_ge10000': 4,
}
STRUCT = np.ones((3, 3, 3), dtype=bool)


def component_label(size: int) -> int:
    if size < THRESHOLDS[0]:
        return 1
    if size < THRESHOLDS[1]:
        return 2
    if size < THRESHOLDS[2]:
        return 3
    return 4


def convert_label(src_label: Path, dst_label: Path) -> dict:
    image = sitk.ReadImage(str(src_label))
    arr = sitk.GetArrayFromImage(image)
    mask = arr > 0
    labeled, n_comp = ndi.label(mask, structure=STRUCT)
    out = np.zeros(arr.shape, dtype=np.uint8)
    counts = Counter()
    sizes = np.bincount(labeled.ravel()) if n_comp > 0 else np.array([0])
    for comp_idx in range(1, n_comp + 1):
        size = int(sizes[comp_idx])
        cls = component_label(size)
        out[labeled == comp_idx] = cls
        counts[cls] += 1
    out_img = sitk.GetImageFromArray(out)
    out_img.CopyInformation(image)
    sitk.WriteImage(out_img, str(dst_label))
    return {
        'case': src_label.name[:-7],
        'n_components': int(n_comp),
        'component_class_counts': {str(k): int(v) for k, v in sorted(counts.items())},
        'lesion_voxels': int(mask.sum()),
    }


def main() -> None:
    if not SRC_RAW.is_dir():
        raise FileNotFoundError(SRC_RAW)
    images_src = SRC_RAW / 'imagesTr'
    labels_src = SRC_RAW / 'labelsTr'
    images_dst = DST_RAW / 'imagesTr'
    labels_dst = DST_RAW / 'labelsTr'
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)
    DST_PREP.mkdir(parents=True, exist_ok=True)

    image_files = sorted(images_src.glob('*_0000.nii.gz'))
    label_files = sorted(labels_src.glob('*.nii.gz'))
    if not image_files or not label_files:
        raise RuntimeError('No images/labels found')

    for image_file in image_files:
        dst = images_dst / image_file.name
        if not dst.exists():
            try:
                dst.symlink_to(image_file)
            except OSError:
                shutil.copy2(image_file, dst)

    summaries = []
    total_component_counts = Counter()
    for idx, label_file in enumerate(label_files, start=1):
        summary = convert_label(label_file, labels_dst / label_file.name)
        summaries.append(summary)
        for cls, count in summary['component_class_counts'].items():
            total_component_counts[int(cls)] += int(count)
        if idx % 100 == 0:
            print(f'converted {idx}/{len(label_files)}')

    source_dataset_json = SRC_RAW / 'dataset.json'
    if not source_dataset_json.is_file():
        raise FileNotFoundError(source_dataset_json)
    dataset_json = json.loads(source_dataset_json.read_text())
    dataset_json['labels'] = LABELS
    dataset_json['numTraining'] = len(image_files)
    (DST_RAW / 'dataset.json').write_text(json.dumps(dataset_json, indent=2, ensure_ascii=False))

    src_split = SRC_PREP / 'splits_final.json'
    if src_split.exists():
        shutil.copy2(src_split, DST_PREP / 'splits_final.json')

    stats = {
        'source_dataset': str(SRC_RAW),
        'target_dataset': str(DST_RAW),
        'thresholds_voxels': list(THRESHOLDS),
        'label_mapping': LABELS,
        'num_cases': len(label_files),
        'total_component_class_counts': {str(k): int(v) for k, v in sorted(total_component_counts.items())},
        'case_summaries': summaries,
    }
    (DST_RAW / 'msl_stats.json').write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print('done')
    print(json.dumps({k: stats[k] for k in ['num_cases', 'total_component_class_counts']}, indent=2))


if __name__ == '__main__':
    main()
