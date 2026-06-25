"""Differentiable patch rendering: warp a learnable patch onto a chroma-key quad.

Copy of src/yolo_chroma_attack/patch_render.py so this package is self-contained.
See that file for the full design notes.
"""
from __future__ import annotations

import torch
from kornia.geometry.transform import warp_perspective, get_perspective_transform


def patch_canonical_corners(patch_h: int, patch_w: int,
                             device: torch.device,
                             batch: int) -> torch.Tensor:
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
    patch: torch.Tensor,            # (3, Ph, Pw) in [0, 1]
    corners_dst: torch.Tensor,      # (B, 4, 2) destination quad in image pixels
    min_det: float = 1e-3,
) -> torch.Tensor:
    B, _, H, W = image.shape
    Ph, Pw = patch.shape[-2], patch.shape[-1]
    device = image.device

    src_one = patch_canonical_corners(Ph, Pw, device, 1)
    mask_one = torch.ones((1, 1, Ph, Pw), device=device, dtype=patch.dtype)
    patch_one = patch.unsqueeze(0)
    out_patched = torch.zeros((B, 3, H, W), device=device, dtype=image.dtype)
    out_mask = torch.zeros((B, 1, H, W), device=device, dtype=image.dtype)
    for b in range(B):
        dst_b = corners_dst[b:b+1].to(device)
        try:
            M_b = get_perspective_transform(src_one, dst_b)
            pw = warp_perspective(patch_one, M_b, dsize=(H, W),
                                  mode="bilinear", padding_mode="zeros")
            mw = warp_perspective(mask_one, M_b, dsize=(H, W),
                                  mode="bilinear", padding_mode="zeros")
        except Exception:
            continue
        out_patched[b] = pw[0]
        out_mask[b] = mw[0]
    return image * (1.0 - out_mask) + out_patched * out_mask


def init_patch(shape: tuple[int, int, int] = (3, 256, 512),
               device: torch.device | str = "cuda",
               init: str = "uniform") -> torch.Tensor:
    if init == "uniform":
        t = torch.rand(shape, device=device)
    elif init == "gray":
        t = torch.full(shape, 0.5, device=device)
    else:
        raise ValueError(f"Unknown init: {init}")
    t.requires_grad_(True)
    return t
