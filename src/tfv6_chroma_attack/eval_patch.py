"""Score a trained patch against tfv6, resolved by leader gap and ego speed.

Why not just read the training log
----------------------------------
The aggregate validation number the trainer prints averages over the whole gap
sweep, and that average is close to meaningless here. The baseline diagnostic
showed that tfv6's target-speed head has a sharp, speed-dependent decision
boundary: at 8 m/s ego speed it commands a full stop for every gap up to 17.5 m
and cruises from 20 m out. Frames on the far side of that boundary are already
"attacked" before the patch exists, and frames deep inside it are saturated at
P(stop) = 1.0. Averaging the two hides the only thing that matters.

What matters is where the boundary MOVES. This script therefore reports, per
(ego speed, gap), the clean and patched values of the two quantities that drive
the controller:

    E[speed] = sum(softmax(logits) * target_speeds)   -> the PID setpoint
    P(stop)  = softmax(logits)[0]                     -> arms the hard-brake override

and summarises them as a **boundary shift in metres**: the closest gap at which
the model still commands motion, clean versus patched. A patch that moves the
boundary from 20 m down to 10 m has halved the distance at which the ego reacts
to a truck, which is a physical claim a closed-loop run can then confirm or deny.

The patch is composited through exactly the same code path the trainer uses
(`render_patch_on_image` over the chroma-key quads), so an improvement measured
here is an improvement on the thing that was optimised, not on a re-implementation
of it.

Usage on Vortex:
    conda activate PCLA15
    cd /home/vortex/adversarial-patch-vehicle
    PYTHONPATH=.:/home/vortex/PCLA:/home/vortex/PCLA/pcla_agents/transfuserv6 \\
    python src/tfv6_chroma_attack/eval_patch.py \\
        --run-dir data/chroma_key_dataset/tfv6/Town04_spawn273_day/clean \\
        --patch experiments/tfv6_attack/run01_t04day/patch_final.pt \\
        --out experiments/tfv6_attack/run01_t04day/eval.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.tfv6_chroma_attack.dataset import Tfv6ChromaDataset, collate
from src.tfv6_chroma_attack.tfv6_loss import (
    DEFAULT_CKPT_DIR, DEFAULT_PCLA_ROOT, Tfv6HideLoss,
)
from src.yolo_chroma_attack.patch_render import render_patch_on_image

PIXEL_SCALE = 255.0

# A gap counts as "the model is still willing to move" above this setpoint. Well
# below the lowest non-zero bin (4 m/s) so it triggers on genuine motion commands
# rather than on softmax leakage.
MOTION_THRESHOLD_MS = 1.0


def boundary_m(rows: list[dict], key: str) -> float:
    """Closest gap (m) at which the mean setpoint still commands motion.

    Walks inward from the far end and stops at the first gap that is braking, so
    an isolated non-braking frame deep inside the braking zone cannot fake a big
    shift. Returns inf when the model brakes at every captured gap.
    """
    by_gap = defaultdict(list)
    for r in rows:
        by_gap[r["gap_m"]].append(r[key])
    best = float("inf")
    for gap in sorted(by_gap, reverse=True):
        if float(np.mean(by_gap[gap])) > MOTION_THRESHOLD_MS:
            best = gap
        else:
            break
    return best


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--patch", type=Path, required=True,
                   help="patch_final.pt (or any patch_ep*.pt) from a training run")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--split", default="all", choices=["all", "train", "val"])
    p.add_argument("--index-name", default="quads_index.json")
    p.add_argument("--leader-source", default="meta")
    p.add_argument("--ego-speeds", type=float, nargs="+", default=[4.0, 8.0],
                   help="tfv6 conditions on ego speed, so the boundary is only "
                        "defined at a given speed")
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    p.add_argument("--pcla-root", default=DEFAULT_PCLA_ROOT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--single-ckpt", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device)
    patch = torch.load(args.patch, map_location=device, weights_only=True)
    if patch.dim() == 4:
        patch = patch[0]
    patch = patch.to(device).clamp(0.0, 1.0)
    print(f"patch {tuple(patch.shape)} from {args.patch}")

    ds = Tfv6ChromaDataset(args.run_dir, split=args.split, seed=args.seed,
                           index_name=args.index_name,
                           leader_source=args.leader_source)
    print(f"{len(ds)} frames ({args.split} split of {args.run_dir})")

    loss = Tfv6HideLoss(ckpt_dir=args.ckpt_dir, device=args.device,
                        pcla_root=args.pcla_root, ensemble=not args.single_ckpt)
    speeds_t = loss.target_speeds

    # Gaps come from the capture sidecars, not from the quad index.
    gaps = {}
    for meta_path in Path(args.run_dir).glob("*.json"):
        if meta_path.name == args.index_name:
            continue
        try:
            gaps[meta_path.stem] = float(
                json.loads(meta_path.read_text()).get("leader_gap_m", -1.0))
        except (ValueError, KeyError):
            continue

    rows = []
    for i in range(len(ds)):
        item = ds[i]
        batch = collate([item])
        img = batch["image"].to(device)
        corners = batch["corners"].to(device)
        with torch.no_grad():
            clean_in = img * PIXEL_SCALE
            patched_in = render_patch_on_image(img, patch, corners) * PIXEL_SCALE
            for ego_v in args.ego_speeds:
                ego_t = torch.tensor([float(ego_v)], device=device)
                out = {}
                for tag, tensor in (("clean", clean_in), ("patched", patched_in)):
                    fw = loss._forward_ensemble(tensor, ego_t)
                    probs = torch.softmax(fw["speed_logits"], dim=-1)[0]
                    out[f"{tag}_speed"] = float((probs * speeds_t).sum())
                    out[f"{tag}_p0"] = float(probs[0])
                rows.append({
                    "stem": item["stem"], "gap_m": gaps.get(item["stem"], -1.0),
                    "ego_speed_ms": ego_v, **out,
                    "d_speed": out["patched_speed"] - out["clean_speed"],
                    "d_p0": out["patched_p0"] - out["clean_p0"],
                })

    for ego_v in args.ego_speeds:
        sub = [r for r in rows if r["ego_speed_ms"] == ego_v]
        print(f"\n=== ego speed {ego_v:.1f} m/s ===")
        print(f"{'gap m':>7} {'n':>3} | {'clean E':>8} {'patch E':>8} {'dE':>8} "
              f"| {'clean P0':>9} {'patch P0':>9} {'dP0':>9}")
        by_gap = defaultdict(list)
        for r in sub:
            by_gap[r["gap_m"]].append(r)
        for gap in sorted(by_gap):
            g = by_gap[gap]
            m = lambda k: float(np.mean([x[k] for x in g]))  # noqa: E731
            print(f"{gap:7.1f} {len(g):3d} | {m('clean_speed'):8.3f} "
                  f"{m('patched_speed'):8.3f} {m('d_speed'):+8.3f} | "
                  f"{m('clean_p0'):9.4f} {m('patched_p0'):9.4f} {m('d_p0'):+9.4f}")

        b_clean = boundary_m(sub, "clean_speed")
        b_patch = boundary_m(sub, "patched_speed")
        shift = (b_clean - b_patch) if np.isfinite(b_clean) and np.isfinite(b_patch) \
            else float("nan")
        print(f"  reaction boundary: clean {b_clean} m -> patched {b_patch} m "
              f"(shift {shift:+.1f} m)")
        # Frames the patch flipped from braking to moving are the ones that would
        # actually change a closed-loop trajectory.
        flipped = [r for r in sub
                   if r["clean_speed"] <= MOTION_THRESHOLD_MS
                   and r["patched_speed"] > MOTION_THRESHOLD_MS]
        print(f"  frames flipped brake -> move: {len(flipped)}/{len(sub)}")
        if flipped:
            fg = sorted({r['gap_m'] for r in flipped})
            print(f"  at gaps: {fg}")

    if args.out and rows:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
