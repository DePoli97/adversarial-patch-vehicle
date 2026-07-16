"""Differentiable patch rendering: warp a learnable patch onto a chroma-key quad.

Given an image (B, 3, H, W) and per-image quad corners (B, 4, 2) in pixel
coordinates, paste a learnable patch tensor `patch (3, Ph, Pw)` onto the quad
via perspective transform. The whole pipeline is differentiable, so gradients
flow back into `patch`.

Strategy:
  1. Build per-image 3x3 homography from the patch's canonical corners
     [[0,0], [Pw-1,0], [Pw-1,Ph-1], [0,Ph-1]] to the destination quad corners.
  2. kornia.geometry.warp_perspective the patch to (H, W). Same for a
     constant-1 mask, which we use to soft-blend onto the image:
        out = image * (1 - mask_warped) + patch_warped * mask_warped
  3. Output is (B, 3, H, W), still in [0, 1] if patch is.

This mirrors `extract_quad.warp_patch_onto_quad` (which uses cv2 and is
non-differentiable) but in a torch.autograd-compatible form.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from kornia.geometry.transform import warp_perspective, get_perspective_transform


def patch_canonical_corners(patch_h: int, patch_w: int,
                             device: torch.device,
                             batch: int) -> torch.Tensor:
    """Canonical TL/TR/BR/BL corners of the patch tensor, shape (B, 4, 2)."""
    corners = torch.tensor(
        [[0.0, 0.0],
         [patch_w - 1.0, 0.0],
         [patch_w - 1.0, patch_h - 1.0],
         [0.0, patch_h - 1.0]],
        dtype=torch.float32, device=device,
    )
    return corners.unsqueeze(0).expand(batch, -1, -1).contiguous()


def render_patch_on_image(
    image: torch.Tensor,            # (B, 3, H, W) in [0, 1]
    patch: torch.Tensor,            # (3, Ph, Pw) in [0, 1], learnable
    corners_dst: torch.Tensor,      # (B, 4, 2) destination quad in image pixels
    min_det: float = 1e-3,
    illum: torch.Tensor | None = None,  # (B, 1, Ph, Pw) per-frame illumination map
) -> torch.Tensor:
    """Render the patch onto each image's quad. Returns (B, 3, H, W).

    Samples with a near-singular perspective transform (|det(H)| < min_det)
    are skipped — their output is the original image. This protects training
    from rare degenerate quads (e.g. captured at extreme glancing angles)
    that would otherwise crash `torch.linalg.inv`.

    `illum`, if given, is a per-frame illumination map at patch resolution: the
    patch (an albedo) is multiplied by it before warping so it appears lit by
    the scene — at night the map is small, darkening the patch to match, closing
    the train↔deploy lighting gap (see illumination.py). Differentiable.
    """
    B, _, H, W = image.shape
    Ph, Pw = patch.shape[-2], patch.shape[-1]
    device = image.device

    # Warp sample-by-sample so a singular homography on one frame doesn't
    # poison the whole batch. The per-sample call is cheap compared to YOLO
    # forward, and lets us skip degenerate quads gracefully.
    src_one = patch_canonical_corners(Ph, Pw, device, 1)
    mask_one = torch.ones((1, 1, Ph, Pw), device=device, dtype=patch.dtype)
    out_patched = torch.zeros((B, 3, H, W), device=device, dtype=image.dtype)
    out_mask = torch.zeros((B, 1, H, W), device=device, dtype=image.dtype)
    for b in range(B):
        dst_b = corners_dst[b:b+1].to(device)
        # Light the patch (albedo) by this frame's illumination map, if given.
        patch_b = patch if illum is None else patch * illum[b]
        patch_one = patch_b.unsqueeze(0)  # (1, 3, Ph, Pw)
        try:
            M_b = get_perspective_transform(src_one, dst_b)            # (1, 3, 3)
            pw = warp_perspective(patch_one, M_b, dsize=(H, W),
                                  mode="bilinear", padding_mode="zeros")
            mw = warp_perspective(mask_one, M_b, dsize=(H, W),
                                  mode="bilinear", padding_mode="zeros")
        except Exception:
            continue  # singular, skip — image_b stays clean
        out_patched[b] = pw[0]
        out_mask[b] = mw[0]
    return image * (1.0 - out_mask) + out_patched * out_mask


def init_patch(shape: tuple[int, int, int] = (3, 256, 256),
               device: torch.device | str = "cuda",
               init: str = "uniform") -> torch.Tensor:
    """Initialize a learnable patch tensor in [0, 1]. Returns a leaf with grad."""
    if init == "uniform":
        t = torch.rand(shape, device=device)
    elif init == "gray":
        t = torch.full(shape, 0.5, device=device)
    else:
        raise ValueError(f"Unknown init: {init}")
    t.requires_grad_(True)
    return t
