"""Crop geometry + differentiable cropping for the crop-based CLIP attack.

The v1/v2 CLIP attack scored the GLOBAL [CLS] embedding of the whole frame,
which let the optimizer hijack a single pooled 512-d vector with high-frequency
noise instead of changing what the *truck* looks like (see
`docs/clip_attack_survey.md`, Option A). Everything in this module exists to
make the objective LOCAL: cut the truck region out of the composited frame,
resize it to CLIP's native input, and score only that crop.

Two properties matter:
  1. every op is differentiable w.r.t. the input image, so gradients reach the
     patch through the crop (`F.grid_sample`, no integer indexing);
  2. the crop boxes are jittered in scale and offset, so the patch cannot
     overfit one exact crop rectangle.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# Geometry of the CarlaCola rear as seen from the follower: the chroma-key
# marker sits on the upper-centre of the rear panel, so the truck extends
# sideways by ~1.5x the marker width and mostly DOWNWARD (cabin roof is just
# above the marker, wheels are far below it). Measured on Town04 dist10m
# frames: marker 190x105 px, truck 285x260 px with the marker top 0.24*h below
# the roof and the wheels 1.24*h under the marker bottom. The defaults below
# add a bit of road context around that box, which is what CLIP needs to tell
# "truck on a road" from "empty road".
DEFAULT_EXPAND_X = 2.0
DEFAULT_MARGIN_TOP = 0.5
DEFAULT_MARGIN_BOTTOM = 1.8

MIN_SIDE_PX = 8.0


def truck_box_from_quad(
    corners: torch.Tensor,              # (B, 4, 2) marker quad, image pixels
    image_hw: tuple[int, int],
    expand_x: float = DEFAULT_EXPAND_X,
    margin_top: float = DEFAULT_MARGIN_TOP,
    margin_bottom: float = DEFAULT_MARGIN_BOTTOM,
    square: bool = True,
    clamp: bool = True,
) -> torch.Tensor:
    """Derive the truck crop box (x1, y1, x2, y2) from the marker quad.

    `expand_x` multiplies the marker width around its centre; `margin_top` and
    `margin_bottom` are expressed in units of the marker height and are applied
    asymmetrically (the truck body hangs below the marker).

    If `square` the box is grown to its longer side (CLIP resizes to a square
    input, so a square crop avoids aspect distortion). `clamp` intersects the
    result with the image; at short range the truck fills the frame and the
    clamped box degenerates to the full frame, which is the correct behaviour.

    Returns (B, 4) float32.
    """
    H, W = image_hw
    x = corners[..., 0]
    y = corners[..., 1]
    x1q, x2q = x.min(dim=1).values, x.max(dim=1).values
    y1q, y2q = y.min(dim=1).values, y.max(dim=1).values
    w = (x2q - x1q).clamp(min=1.0)
    h = (y2q - y1q).clamp(min=1.0)
    cx = 0.5 * (x1q + x2q)

    half_w = 0.5 * w * expand_x
    x1 = cx - half_w
    x2 = cx + half_w
    y1 = y1q - h * margin_top
    y2 = y2q + h * margin_bottom

    if square:
        side = torch.maximum(x2 - x1, y2 - y1)
        cx_b = 0.5 * (x1 + x2)
        cy_b = 0.5 * (y1 + y2)
        x1, x2 = cx_b - 0.5 * side, cx_b + 0.5 * side
        y1, y2 = cy_b - 0.5 * side, cy_b + 0.5 * side

    if clamp:
        x1 = x1.clamp(0.0, W - 1.0 - MIN_SIDE_PX)
        y1 = y1.clamp(0.0, H - 1.0 - MIN_SIDE_PX)
        x2 = torch.maximum(x2.clamp(0.0, W - 1.0), x1 + MIN_SIDE_PX)
        y2 = torch.maximum(y2.clamp(0.0, H - 1.0), y1 + MIN_SIDE_PX)

    return torch.stack([x1, y1, x2, y2], dim=-1)


def jitter_boxes(
    boxes: torch.Tensor,                # (B, 4)
    image_hw: tuple[int, int],
    n_crops: int = 1,
    scale_range: tuple[float, float] = (1.0, 1.0),
    shift_frac: float = 0.0,
    generator: torch.Generator | None = None,
    clamp: bool = True,
) -> torch.Tensor:
    """Replicate each box `n_crops` times with random scale / offset.

    `scale_range` multiplies the box side; `shift_frac` shifts the centre by up
    to that fraction of the box side in each axis. With the defaults this is
    the identity (used at eval time so the reported metric is deterministic).

    Returns (B, n_crops, 4).
    """
    H, W = image_hw
    B = boxes.shape[0]
    dev = boxes.device
    out = boxes.unsqueeze(1).expand(B, n_crops, 4).clone()

    if scale_range != (1.0, 1.0) or shift_frac > 0.0:
        cx = 0.5 * (out[..., 0] + out[..., 2])
        cy = 0.5 * (out[..., 1] + out[..., 3])
        bw = out[..., 2] - out[..., 0]
        bh = out[..., 3] - out[..., 1]
        s = torch.empty((B, n_crops), device=dev).uniform_(
            scale_range[0], scale_range[1], generator=generator)
        if shift_frac > 0.0:
            dx = torch.empty((B, n_crops), device=dev).uniform_(
                -shift_frac, shift_frac, generator=generator) * bw
            dy = torch.empty((B, n_crops), device=dev).uniform_(
                -shift_frac, shift_frac, generator=generator) * bh
        else:
            dx = torch.zeros((B, n_crops), device=dev)
            dy = torch.zeros((B, n_crops), device=dev)
        cx, cy = cx + dx, cy + dy
        bw, bh = bw * s, bh * s
        out = torch.stack([cx - 0.5 * bw, cy - 0.5 * bh,
                           cx + 0.5 * bw, cy + 0.5 * bh], dim=-1)

    if clamp:
        x1 = out[..., 0].clamp(0.0, W - 1.0 - MIN_SIDE_PX)
        y1 = out[..., 1].clamp(0.0, H - 1.0 - MIN_SIDE_PX)
        x2 = torch.maximum(out[..., 2].clamp(0.0, W - 1.0), x1 + MIN_SIDE_PX)
        y2 = torch.maximum(out[..., 3].clamp(0.0, H - 1.0), y1 + MIN_SIDE_PX)
        out = torch.stack([x1, y1, x2, y2], dim=-1)
    return out


def crop_resize(
    images: torch.Tensor,               # (B, C, H, W)
    boxes: torch.Tensor,                # (B, n, 4) in pixels
    out_size: int = 224,
    padding_mode: str = "border",
) -> torch.Tensor:
    """Differentiable crop + bilinear resize. Returns (B, n, C, S, S).

    Implemented with an affine `grid_sample` rather than `roi_align` so the
    gradient path is the plain PyTorch one and boxes may legally fall partly
    outside the frame (handled by `padding_mode`). The image tensor is never
    replicated: we loop over the (small) crop axis and reuse the same input.
    """
    B, C, H, W = images.shape
    n = boxes.shape[1]
    crops = []
    for j in range(n):
        b = boxes[:, j]                                   # (B, 4)
        cx = 0.5 * (b[:, 0] + b[:, 2])
        cy = 0.5 * (b[:, 1] + b[:, 3])
        bw = (b[:, 2] - b[:, 0]).clamp(min=MIN_SIDE_PX)
        bh = (b[:, 3] - b[:, 1]).clamp(min=MIN_SIDE_PX)
        # Affine that maps the output square onto the box, in [-1, 1] coords.
        theta = torch.zeros((B, 2, 3), device=images.device, dtype=images.dtype)
        theta[:, 0, 0] = bw / W
        theta[:, 1, 1] = bh / H
        theta[:, 0, 2] = 2.0 * cx / W - 1.0
        theta[:, 1, 2] = 2.0 * cy / H - 1.0
        grid = F.affine_grid(theta, (B, C, out_size, out_size),
                             align_corners=False)
        crops.append(F.grid_sample(images, grid, mode="bilinear",
                                   padding_mode=padding_mode,
                                   align_corners=False))
    return torch.stack(crops, dim=1)
