"""EOT photometric augmentation for the patch tensor.

Identical to the one used for YOLO training (src/yolo_chroma_attack/eot.py).
Kept here as a copy so this package has no internal dependency on the YOLO one,
making it easier to ship or fork independently.
"""
from __future__ import annotations

import torch


def _rand(lo: float, hi: float, device: torch.device) -> torch.Tensor:
    return torch.empty((), device=device).uniform_(lo, hi)


def eot_apply(
    patch: torch.Tensor,
    brightness: tuple[float, float] = (0.7, 1.3),
    contrast: tuple[float, float] = (0.8, 1.2),
    color_jitter: float = 0.10,
    noise_std: float = 0.02,
) -> torch.Tensor:
    device = patch.device
    out = patch

    b = _rand(brightness[0], brightness[1], device)
    out = out * b

    c = _rand(contrast[0], contrast[1], device)
    mean = out.mean(dim=(-1, -2), keepdim=True)
    out = (out - mean) * c + mean

    if color_jitter > 0:
        offsets = torch.empty(3, 1, 1, device=device).uniform_(-color_jitter, color_jitter)
        out = out + offsets

    if noise_std > 0:
        out = out + torch.randn_like(out) * noise_std

    return out.clamp(0.0, 1.0)
