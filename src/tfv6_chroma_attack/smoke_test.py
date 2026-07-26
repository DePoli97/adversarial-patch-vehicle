"""Smoke + overfit test for `Tfv6HideLoss`. Run this before any real training.

Two checks, in order of importance:

1. SMOKE — a random (B, 3, 384, 2304) image: the loss is finite, `backward()`
   populates the patch gradient, and the info dict reports sane physical
   numbers (expected speed in m/s inside the bin range, P(bin 0) in [0, 1]).
2. OVERFIT — one fixed image, ~100 Adam steps on the patch alone. The expected
   speed must RISE and P(bin 0) must FALL. This is the single most important
   signal that the attack is wired correctly: if these do not move, the
   gradient is not reaching the pixels the model actually looks at.

Both checks run on a synthetic random image by default (no dataset required),
with the patch pasted into the FRONT camera slice, [768:1152], where the leader
truck appears. Point `--run-dir` at a real capture to use its first frame and
real quad instead — do that as soon as a capture exists, because on random noise
the model's prior is "stop" and the numbers are an upper bound, not a
prediction of what a realistically sized patch achieves.

Usage on Vortex:
    conda activate PCLA15
    PYTHONPATH=.:/home/vortex/PCLA:/home/vortex/PCLA/pcla_agents/transfuserv6 \\
    python src/tfv6_chroma_attack/smoke_test.py --steps 100
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.tfv6_chroma_attack.dataset import (
    CAM_SIZE, FRONT_CAM_INDEX, TFV6_IMAGE_HW, Tfv6ChromaDataset,
)
from src.tfv6_chroma_attack.tfv6_loss import DEFAULT_CKPT_DIR, Tfv6HideLoss
from src.yolo_chroma_attack.patch_render import init_patch, render_patch_on_image

PIXEL_SCALE = 255.0


def front_camera_quad(patch_h: int, patch_w: int, scale: float = 0.55):
    """A centred axis-aligned quad inside the front camera slice, TL/TR/BR/BL."""
    x0 = FRONT_CAM_INDEX * CAM_SIZE
    h = CAM_SIZE * scale
    w = h * (patch_w / patch_h)
    cx, cy = x0 + CAM_SIZE / 2.0, CAM_SIZE / 2.0
    return torch.tensor([[[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
                          [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]]])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, default=None,
                   help="Real capture folder. Uses its first frame + quad "
                        "instead of a synthetic image.")
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--patch-h", type=int, default=256)
    p.add_argument("--patch-w", type=int, default=512)
    p.add_argument("--single-ckpt", action="store_true")
    p.add_argument("--lambda-wp", type=float, default=0.1)
    p.add_argument("--lambda-detect", type=float, default=0.1)
    p.add_argument("--detect-mode", default="shrink",
                   choices=["shrink", "suppress", "farther"])
    p.add_argument("--leader-distance", type=float, default=15.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    H, W = TFV6_IMAGE_HW

    loss_fn = Tfv6HideLoss(
        ckpt_dir=args.ckpt_dir, device=args.device, ensemble=not args.single_ckpt,
        lambda_wp=args.lambda_wp, lambda_detect=args.lambda_detect,
        detect_mode=args.detect_mode, leader_distance_m=args.leader_distance,
    )
    print(f"victim: {len(loss_fn.nets)} ckpt {loss_fn.ckpt_files}")
    print(f"BEV: cells_per_meter={loss_fn.cells_per_meter} "
          f"min_x={loss_fn.min_x_m} min_y={loss_fn.min_y_m}")

    leader_xy = None
    if args.run_dir is not None:
        ds = Tfv6ChromaDataset(args.run_dir, split="all")
        item = ds[0]
        image = item["image"].unsqueeze(0).to(device)
        corners = item["corners"].unsqueeze(0).to(device)
        if "leader_xy" in item:
            leader_xy = item["leader_xy"].unsqueeze(0)
        print(f"frame: {item['stem']}  leader_xy={leader_xy}")
    else:
        image = torch.rand(1, 3, H, W, device=device)
        corners = front_camera_quad(args.patch_h, args.patch_w).to(device)
        print("frame: SYNTHETIC random noise (upper-bound numbers, not a "
              "prediction — rerun with --run-dir once a capture exists)")

    # --- 1. smoke -----------------------------------------------------
    print("\n=== SMOKE ===")
    patch = init_patch((3, args.patch_h, args.patch_w), device=device, init="gray")
    comp = render_patch_on_image(image, patch, corners) * PIXEL_SCALE
    loss, info = loss_fn(comp, None, leader_xy=leader_xy)
    loss.backward()

    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    assert patch.grad is not None, "backward() did not populate patch.grad"
    gnorm = float(patch.grad.abs().sum())
    assert gnorm > 0.0, "patch gradient is all zeros"
    lo, hi = float(loss_fn.target_speeds.min()), float(loss_fn.target_speeds.max())
    assert lo <= info["expected_speed_ms"] <= hi, info["expected_speed_ms"]
    assert 0.0 <= info["p0"] <= 1.0, info["p0"]
    print(f"loss={info['loss']:.4f}  |grad|_1={gnorm:.4f}")
    for k in sorted(info):
        print(f"  {k} = {info[k]}")

    # --- 2. overfit ---------------------------------------------------
    print(f"\n=== OVERFIT ({args.steps} steps, fixed image) ===")
    patch = init_patch((3, args.patch_h, args.patch_w), device=device, init="gray")
    opt = torch.optim.Adam([patch], lr=args.lr)
    first = last = None
    for step in range(args.steps):
        comp = render_patch_on_image(image, patch, corners) * PIXEL_SCALE
        loss, info = loss_fn(comp, None, leader_xy=leader_xy)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            patch.clamp_(0.0, 1.0)
        if first is None:
            first = info
        last = info
        if step % 10 == 0 or step == args.steps - 1:
            print(f"  step {step:4d}  loss={info['loss']:8.4f}  "
                  f"speed={info['expected_speed_ms']:7.4f} m/s  "
                  f"p0={info['p0']:.6f}  bin={info['top_speed_bin']}")

    d_speed = last["expected_speed_ms"] - first["expected_speed_ms"]
    d_p0 = last["p0"] - first["p0"]
    print(f"\nspeed {first['expected_speed_ms']:.4f} -> "
          f"{last['expected_speed_ms']:.4f} m/s  (delta {d_speed:+.4f})")
    print(f"p0    {first['p0']:.6f} -> {last['p0']:.6f}  (delta {d_p0:+.6f})")
    ok = d_speed > 0 and d_p0 <= 0
    print("RESULT:", "PASS — attack is wired correctly" if ok else "FAIL — investigate")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
