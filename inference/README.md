# Local inference

This directory contains the platform-independent implementation of the final
ensemble. It reads the 20 trained checkpoints directly from `nnUNet_results`;
no container or separately packaged model archive is required.

After preparing the environment and training all folds, run one case from the
repository root:

```bash
conda activate isles

python -m inference.predict \
  --input-image /path/to/case.nii.gz \
  --results-root "$ISLES26_ROOT/nnUNet_results" \
  --output-dir /path/to/prediction
```

The command writes:

- `stroke_lesion_segmentation.nii.gz`: post-processed binary lesion mask.
- `lesion_probability_map.nii.gz`: continuous three-family probability map.

Both outputs copy the input image geometry. The input must follow the same
single-channel, skull-stripped T1 definition as the official ISLES'26 data used
for training; nnU-Net performs its configured preprocessing internally.
