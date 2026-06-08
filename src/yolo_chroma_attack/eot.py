"""EOT (Expectation over Transformation) augmentation for physical robustness.

Applied to the patch tensor BEFORE it is warped onto the chroma-key quad. The
training-time variability captured by EOT (brightness, contrast, color jitter,
slight blur) is meant to mimic the conditions the printed patch will face in
the physical world (different lighting, camera ISO, cheap print colors).

We deliberately do NOT apply geometric augmentations to the patch here because
the chroma-key quad already gives a rich distribution of poses (angle,
distance, lateral offset) thanks to the CARLA capture pipeline. Geometric
robustness is therefore implicit in the dataset; EOT only needs to cover
photometric robustness.

All ops are differentiable.
"""
from __future__ import annotations

import torch


def _rand(lo: float, hi: float, device: torch.device) -> torch.Tensor:
    return torch.empty((), device=device).uniform_(lo, hi)


def eot_apply(
    patch: torch.Tensor,            # (3, Ph, Pw) in [0, 1]
    brightness: tuple[float, float] = (0.7, 1.3),
    contrast: tuple[float, float] = (0.8, 1.2),
    color_jitter: float = 0.10,
    noise_std: float = 0.02,
) -> torch.Tensor:
    """Sample one transformation and apply it to the patch.

    Returns a new tensor of the same shape, still in [0, 1].
    """
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
