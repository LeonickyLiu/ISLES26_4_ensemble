from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
from monai.losses import DiceLoss
from scipy import ndimage
from torch import nn

from nnunetv2.training.loss.ici_official import ICI_loss as ici_module
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def _fast_connected_components(
    image: torch.Tensor,
    num_iterations: int = 75,
    spatial_dims: int = 3,
) -> torch.Tensor:
    del num_iterations
    binary = image.detach().float().cpu().numpy() == 1
    labels = np.zeros(binary.shape, dtype=np.int32)
    structure = ndimage.generate_binary_structure(spatial_dims, spatial_dims)
    for index in np.ndindex(binary.shape[:2]):
        labels[index], _ = ndimage.label(binary[index], structure=structure)
    return torch.as_tensor(labels, device=image.device, dtype=image.dtype)


def _fast_connected_components_with_gradients(
    image: torch.Tensor,
    num_iterations: int = 75,
    threshold: float = 0.5,
    spatial_dims: int = 3,
) -> torch.Tensor:
    del num_iterations
    binary = image.detach().float().cpu().numpy() >= threshold
    labels = np.zeros(binary.shape, dtype=np.int32)
    structure = ndimage.generate_binary_structure(spatial_dims, spatial_dims)
    for index in np.ndindex(binary.shape[:2]):
        labels[index], _ = ndimage.label(binary[index], structure=structure)
    return torch.as_tensor(labels, device=image.device, dtype=image.dtype)


class _ICICompoundLoss(nn.Module):
    """Default nnU-Net loss plus ICI supervision at the highest resolution."""

    def __init__(
        self,
        base_loss: nn.Module,
        global_weight: float = 0.25,
        instance_weight: float = 0.50,
        center_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.global_weight = global_weight
        self.instance_weight = instance_weight
        self.center_weight = center_weight

        dice = DiceLoss(
            include_background=True,
            to_onehot_y=False,
            sigmoid=False,
            softmax=False,
            smooth_nr=1e-5,
            smooth_dr=1e-5,
        )
        # ICI uses component IDs only as masks; exact scipy CCA avoids hundreds
        # of full-volume max-pooling passes without changing loss gradients.
        ici_module.connected_components = _fast_connected_components
        ici_module.connected_components_with_gradients = (
            _fast_connected_components_with_gradients
        )
        self.ici = ici_module.ICILoss(
            loss_function_pixel=dice,
            loss_function_instance=dice,
            loss_function_center=dice,
            activation="none",
            num_out_chn=1,
            object_chn=1,
            spatial_dims=3,
            reduce_segmentation="mean",
            instance_wise_reduce="instance",
            num_iterations=128,
            segmentation_threshold=0.5,
            max_cc_out=50,
            mul_too_many=50,
            min_instance_size=0,
            centroid_offset=3,
            instance_wise_loss_no_tp=True,
        )

    def forward(self, outputs, targets):
        base = self.base_loss(outputs, targets)
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        target = targets[0] if isinstance(targets, (tuple, list)) else targets

        # Component labels must remain exactly representable, so ICI runs in fp32.
        context = (
            torch.autocast(device_type=logits.device.type, enabled=False)
            if logits.device.type in {"cuda", "cpu"}
            else nullcontext()
        )
        with context:
            foreground = torch.softmax(logits.float(), dim=1)[:, 1:].sum(
                dim=1, keepdim=True
            )
            binary_target = (target > 0).float()
            _, instance, center, _, _, _ = self.ici(
                foreground, binary_target
            )
            combined = (
                self.global_weight * base.float()
                + self.instance_weight * instance
                + self.center_weight * center
            )
        return combined


class nnUNetTrainerICILoss(nnUNetTrainer):
    """ATLAS fold experiment using the paper's 1:2:1 ICI weighting."""

    def _do_i_compile(self):
        return False

    def _build_loss(self):
        return _ICICompoundLoss(super()._build_loss())
