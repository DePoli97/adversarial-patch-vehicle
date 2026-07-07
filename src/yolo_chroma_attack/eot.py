"""EOT (Expectation over Transformation) augmentation for physical robustness.

Applied to the patch tensor BEFORE it is warped onto the chroma-key quad. The
training-time variability captured by EOT (brightness, contrast, color jitter,
slight blur) is meant to mimic the conditions the printed patch will face in
the physical world (different lighting, camera ISO, cheap print colors).

Geometric EOT (rotation/scale/translation of the patch) is OPTIONAL (`geom`).
Historically we skipped it because a large, varied capture (many angles /
distances) already provided geometric variety implicitly. That assumption
breaks on a single-road Fase 1 dataset where the vehicle is always seen from
the same rear angle: there the patch overfits ~90 near-identical views into
high-frequency noise, and geometric EOT is needed to recover structure.

All ops are differentiable.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _rand(lo: float, hi: float, device: torch.device) -> torch.Tensor:
    return torch.empty((), device=device).uniform_(lo, hi)


def _apply_affine(patch: torch.Tensor, rot_deg: float, scale: float,
                  tx: float, ty: float) -> torch.Tensor:
    """Differentiable affine warp of the patch (rotation, scale, translation).

    tx/ty are fractions of width/height. Uses reflection padding via
    grid_sample so the patch stays fully populated (no black borders that would
    leak into the composite)."""
    C, H, W = patch.shape
    theta_rad = torch.tensor(rot_deg * 3.14159265 / 180.0, device=patch.device)
    cos, sin = torch.cos(theta_rad), torch.sin(theta_rad)
    # 2x3 affine matrix for grid_sample (inverse mapping, so divide by scale)
    s = 1.0 / max(scale, 1e-3)
    mat = torch.stack([
        torch.stack([cos * s, -sin * s, torch.tensor(tx, device=patch.device)]),
        torch.stack([sin * s, cos * s, torch.tensor(ty, device=patch.device)]),
    ]).unsqueeze(0)
    grid = F.affine_grid(mat, (1, C, H, W), align_corners=False)
    warped = F.grid_sample(patch.unsqueeze(0), grid, mode="bilinear",
                           padding_mode="reflection", align_corners=False)
    return warped.squeeze(0)


def eot_apply(
    patch: torch.Tensor,            # (3, Ph, Pw) in [0, 1]
    brightness: tuple[float, float] = (0.7, 1.3),
    contrast: tuple[float, float] = (0.8, 1.2),
    color_jitter: float = 0.10,
    noise_std: float = 0.02,
    geom: bool = False,
    rot_deg: float = 12.0,
    scale_range: tuple[float, float] = (0.85, 1.15),
    translate: float = 0.06,
) -> torch.Tensor:
    """Sample one transformation and apply it to the patch.

    Photometric ops (brightness/contrast/color/noise) are always applied.
    When `geom=True`, a random affine warp (rotation, scale, translation) is
    applied FIRST. Geometric EOT is essential when the underlying dataset has
    little pose variety (e.g. a single straight road where the vehicle is
    always seen from the same rear angle): without it the patch overfits to a
    handful of near-identical views and degenerates into high-frequency noise
    instead of a robust structured pattern.

    Returns a new tensor of the same shape, still in [0, 1].
    """
    device = patch.device
    out = patch

    if geom:
        rot = _rand(-rot_deg, rot_deg, device).item()
        scl = _rand(scale_range[0], scale_range[1], device).item()
        tx = _rand(-translate, translate, device).item()
        ty = _rand(-translate, translate, device).item()
        out = _apply_affine(out, rot, scl, tx, ty)

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


def total_variation(patch: torch.Tensor) -> torch.Tensor:
    """Total-variation of the patch: mean absolute difference between adjacent
    pixels (horizontal + vertical). Adding this to the loss with a weight
    penalizes high-frequency noise and pushes the patch toward smooth,
    contiguous colour regions — i.e. structured, printable patterns rather than
    per-pixel static. Standard in adversarial-patch work (Thys et al. 2019)."""
    dh = (patch[:, 1:, :] - patch[:, :-1, :]).abs().mean()
    dw = (patch[:, :, 1:] - patch[:, :, :-1]).abs().mean()
    return dh + dw
