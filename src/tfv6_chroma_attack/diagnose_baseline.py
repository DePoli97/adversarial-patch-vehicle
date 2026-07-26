"""Measure how tfv6 reacts to a leader truck across the captured distance sweep.

Why this exists
---------------
The first training smoke run reported a baseline of `p0 = 0.9999` (probability of
the 0 m/s bin) on the validation frames. That number alone is ambiguous, and the
ambiguity is expensive:

  * If the model says "stop" only when the truck is genuinely close and relaxes as
    the gap opens, the harness is faithful and the attack simply has to work in the
    mid-range where a decision is actually being made.
  * If it says "stop" at every gap, including 25 m of empty road, then the offline
    input dict differs from the one the agent builds in the closed loop, the
    baseline is a harness artefact, and any patch trained on it would be optimising
    against that artefact rather than against the truck. That is exactly how the
    YOLO-trained campaign failed, wearing a different costume.

So: sweep the captured gaps, print the response curve, and let the numbers decide.

The same pass also answers a second question for free. Running it on the *patched*
capture folder measures what the existing YOLO-trained patches do to tfv6 — the
expected answer is "nothing", which is the thesis' central diagnostic finding and
deserves a number rather than an assertion.

Model loading and input-dict construction are delegated to `Tfv6HideLoss`, so the
frames are treated bit-identically to how they are treated during training.

Usage on Vortex:
    conda activate PCLA15
    cd /home/vortex/adversarial-patch-vehicle
    PYTHONPATH=.:/home/vortex/PCLA:/home/vortex/PCLA/pcla_agents/transfuserv6 \\
    python src/tfv6_chroma_attack/diagnose_baseline.py \\
        --dirs data/chroma_key_dataset/tfv6/Town04_spawn273_day/clean \\
               data/chroma_key_dataset/tfv6/Town04_spawn273_day/patched \\
        --out experiments/tfv6_attack/baseline_Town04_day.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from src.tfv6_chroma_attack.tfv6_loss import (
    DEFAULT_CKPT_DIR, DEFAULT_PCLA_ROOT, Tfv6HideLoss,
)


def load_composite(path: Path, device: torch.device) -> torch.Tensor:
    """PNG -> (1, 3, H, W) float tensor in [0, 255], RGB.

    Same convention as `Tfv6ChromaDataset`: cv2 reads BGR, the model wants RGB,
    and tfv6 consumes raw 0-255 pixels (ImageNet normalisation lives inside the
    backbone).
    """
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dirs", nargs="+", required=True,
                   help="capture folders; each is labelled by its own name")
    p.add_argument("--out", type=Path, default=None, help="CSV to write")
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    p.add_argument("--pcla-root", default=DEFAULT_PCLA_ROOT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--single-ckpt", action="store_true",
                   help="one checkpoint instead of the 3-model ensemble that "
                        "real inference uses")
    p.add_argument("--ego-speed", type=float, default=8.0,
                   help="ego speed (m/s) reported to the model; the captures are "
                        "static so there is no measured value to use")
    p.add_argument("--ego-speed-sweep", type=float, nargs="*", default=None,
                   help="if given, repeat the whole sweep at each of these ego "
                        "speeds — tfv6 conditions on speed, so a saturated stop "
                        "prediction may simply mean 'you are already stopped'")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    loss = Tfv6HideLoss(ckpt_dir=args.ckpt_dir, device=args.device,
                        pcla_root=args.pcla_root, ensemble=not args.single_ckpt,
                        ego_speed_ms=args.ego_speed)
    device = torch.device(args.device)
    bins = loss.target_speeds.tolist()
    print(f"victim: {len(loss.nets)} checkpoint(s); bins = {bins}")

    speeds_to_try = args.ego_speed_sweep if args.ego_speed_sweep else [args.ego_speed]
    rows = []

    for d in args.dirs:
        dpath = Path(d)
        label = dpath.name
        stems = sorted(q.stem for q in dpath.glob("*.png"))
        if args.limit:
            stems = stems[: args.limit]
        print(f"\n=== {dpath} ({len(stems)} frames) ===")

        for ego_v in speeds_to_try:
            ego_t = torch.tensor([float(ego_v)], device=device)
            per_gap = defaultdict(list)
            for stem in stems:
                img = load_composite(dpath / f"{stem}.png", device)
                meta_path = dpath / f"{stem}.json"
                meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
                gap = float(meta.get("leader_gap_m", -1.0))

                with torch.no_grad():
                    fw = loss._forward_ensemble(img, ego_t)
                probs = torch.softmax(fw["speed_logits"], dim=-1)[0]
                expected = float((probs * loss.target_speeds).sum())
                p0 = float(probs[0])
                argmax_bin = int(probs.argmax())

                per_gap[gap].append((expected, p0))
                rows.append({
                    "dir": label, "stem": stem, "gap_m": gap, "ego_speed_ms": ego_v,
                    "expected_speed_ms": expected, "p0": p0,
                    "argmax_bin_ms": bins[argmax_bin],
                    "walk_m": meta.get("walk_offset_m", ""),
                })

            print(f"\n  ego_speed = {ego_v:.1f} m/s")
            print(f"  {'gap m':>7} {'n':>3} {'E[speed] m/s':>13} {'P(stop)':>9}")
            for gap in sorted(per_gap):
                vals = per_gap[gap]
                me = float(np.mean([v[0] for v in vals]))
                mp = float(np.mean([v[1] for v in vals]))
                print(f"  {gap:7.1f} {len(vals):3d} {me:13.3f} {mp:9.4f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows -> {args.out}")

    # A response curve that is flat in the gap means the model is not reacting to
    # the truck at all, which would invalidate the offline baseline.
    for ego_v in speeds_to_try:
        for label in {r["dir"] for r in rows}:
            sub = [r for r in rows if r["dir"] == label and r["ego_speed_ms"] == ego_v]
            if len(sub) < 2:
                continue
            gaps = sorted({r["gap_m"] for r in sub})
            near = np.mean([r["expected_speed_ms"] for r in sub
                            if r["gap_m"] == gaps[0]])
            far = np.mean([r["expected_speed_ms"] for r in sub
                           if r["gap_m"] == gaps[-1]])
            verdict = "RESPONSIVE" if far - near > 0.5 else "FLAT (suspicious)"
            print(f"[{label} @ ego {ego_v:.1f}] E[speed] {gaps[0]:.1f} m -> "
                  f"{gaps[-1]:.1f} m : {near:.3f} -> {far:.3f}   {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
