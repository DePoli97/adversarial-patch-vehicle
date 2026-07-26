"""Sign assertion for `ClipCropLoss` on real CARLA frames.

The crop loss minimizes `relu(cos(crop, VEHICLE) - cos(crop, ROAD) + margin)`.
That is only an attack if the quantity is genuinely *higher* when a truck is in
the crop than when the road is empty — otherwise we would be optimizing the
wrong direction and any "improvement" during training would be meaningless
(exactly the trap flagged in docs/ISSUES_AND_IDEAS.md for the v1/v2 runs).

This test takes the chroma-key triplet capture, which gives three versions of
the same pose:
    _clean     : the CarlaCola is there, no marker      -> "truck" ground truth
    _marker    : the CarlaCola with the yellow marker   -> attack starting point
    _noleader  : the leader removed, empty road         -> "road" ground truth
crops the SAME truck box (derived from the marker quad) out of all three, and
checks that the loss ranks them truck > road.

Run (Vortex):
    conda activate PCLA310
    cd /home/vortex/adversarial-patch-vehicle
    PYTHONPATH=. python -u src/clip_chroma_attack/test_crop_sign.py \
        --capture data/chroma_key_dataset/capture_20260609_014138 --n-frames 64
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from src.clip_chroma_attack.clip_loss import ClipCropLoss
from src.clip_chroma_attack.crop_utils import crop_resize

DEFAULT_CAPTURE = Path("data/chroma_key_dataset/capture_20260609_014138")


def _load_batch(paths: list[Path], device: torch.device) -> torch.Tensor:
    imgs = []
    for p in paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            raise RuntimeError(f"Could not load image: {p}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        imgs.append(torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0)
    return torch.stack(imgs).to(device)


def run_sign_test(
    capture: Path = DEFAULT_CAPTURE,
    n_frames: int = 64,
    device: str = "cuda",
    batch_size: int = 16,
    min_area: float = 400.0,
    **loss_kwargs,
) -> dict:
    """Score clean/marker/noleader crops. Returns per-condition means + rates."""
    marker_dir = capture.parent / f"{capture.name}_marker"
    clean_dir = capture.parent / f"{capture.name}_clean"
    noleader_dir = capture.parent / f"{capture.name}_noleader"
    for d in (marker_dir, clean_dir, noleader_dir):
        if not d.exists():
            raise FileNotFoundError(f"Missing triplet folder: {d}")

    with open(marker_dir / "quads_index.json") as f:
        index = json.load(f)
    stems = [
        s for s in sorted(index)
        if cv2.contourArea(np.asarray(index[s]["corners"], dtype=np.float32)) >= min_area
        and (clean_dir / f"{s}.png").exists()
        and (noleader_dir / f"{s}.png").exists()
    ]
    if not stems:
        raise RuntimeError(f"No usable frames in {marker_dir}")
    # Evenly spaced over the capture so we cover the whole approach, not just
    # the frames where the truck is closest.
    sel = [stems[i] for i in np.linspace(0, len(stems) - 1, min(n_frames, len(stems))).astype(int)]

    dev = torch.device(device)
    loss_fn = ClipCropLoss(device=device, **loss_kwargs)

    acc: dict[str, list[np.ndarray]] = {c: [] for c in ("clean", "marker", "noleader")}
    for i in range(0, len(sel), batch_size):
        chunk = sel[i:i + batch_size]
        corners = torch.tensor(
            np.stack([np.asarray(index[s]["corners"], dtype=np.float32) for s in chunk]),
            device=dev)
        imgs = {
            "clean":    _load_batch([clean_dir / f"{s}.png" for s in chunk], dev),
            "marker":   _load_batch([marker_dir / f"{s}.png" for s in chunk], dev),
            "noleader": _load_batch([noleader_dir / f"{s}.png" for s in chunk], dev),
        }
        hw = imgs["clean"].shape[-2:]
        boxes = loss_fn.boxes_for(hw, corners=corners, jitter=False)
        with torch.no_grad():
            for cond, im in imgs.items():
                crops = crop_resize(im, boxes, out_size=loss_fn.image_size)
                sc = loss_fn.score_crops(crops.flatten(0, 1))
                acc[cond].append(np.stack([
                    sc["cos_veh"].cpu().numpy(), sc["cos_road"].cpu().numpy(),
                    sc["p_vehicle"].cpu().numpy(), sc["text_term"].cpu().numpy(),
                ], axis=1))

    out = {"n_frames": len(sel)}
    arr = {c: np.concatenate(v, axis=0) for c, v in acc.items()}
    for cond, a in arr.items():
        out[cond] = {"cos_veh": float(a[:, 0].mean()), "cos_road": float(a[:, 1].mean()),
                     "p_vehicle": float(a[:, 2].mean()), "text_term": float(a[:, 3].mean())}
    # Per-frame rates: how often is the truck crop ranked above the empty one?
    out["win_rate_p_vehicle"] = float((arr["clean"][:, 2] > arr["noleader"][:, 2]).mean())
    out["win_rate_text_term"] = float((arr["clean"][:, 3] >= arr["noleader"][:, 3]).mean())
    out["road_rate_noleader"] = float((arr["noleader"][:, 1] > arr["noleader"][:, 0]).mean())
    out["road_rate_clean"] = float((arr["clean"][:, 1] > arr["clean"][:, 0]).mean())
    return out


def assert_sign(res: dict) -> None:
    """Fail loudly if the loss does not rank truck crops above empty-road crops."""
    clean, noleader, marker = res["clean"], res["noleader"], res["marker"]
    assert clean["p_vehicle"] > noleader["p_vehicle"], (
        "SIGN ERROR: CLIP does not find the truck crop more vehicle-like than the "
        f"empty-road crop (p_vehicle {clean['p_vehicle']:.4f} vs "
        f"{noleader['p_vehicle']:.4f}). Fix the prompts before training.")
    assert clean["text_term"] > noleader["text_term"], (
        "SIGN ERROR: the minimized term is not higher on a truck than on empty "
        f"road ({clean['text_term']:.4f} vs {noleader['text_term']:.4f}). "
        "Minimizing it would not remove the truck.")
    assert res["win_rate_p_vehicle"] > 0.5, (
        f"SIGN ERROR: truck crop wins on only {res['win_rate_p_vehicle']:.1%} of "
        "frames; the ranking is not reliable enough to train on.")
    # Not a sign error, but the attack is pointless if the marker alone already
    # hides the truck — the run would start from a solved problem.
    assert marker["text_term"] > 0.0, (
        "The marker frame already scores as empty road (text_term = 0): there is "
        "no gradient left to train on. Tighten the crop or the prompts.")


def test_crop_loss_sign():
    """pytest entry point (needs the triplet capture + a GPU/CPU torch)."""
    res = run_sign_test(device="cuda" if torch.cuda.is_available() else "cpu")
    assert_sign(res)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE,
                   help="capture_<ts> stem; _marker/_clean/_noleader are appended")
    p.add_argument("--n-frames", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--clip-model", default="ViT-B-32")
    p.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    p.add_argument("--crop-expand-x", type=float, default=2.0)
    p.add_argument("--crop-margin-top", type=float, default=0.5)
    p.add_argument("--crop-margin-bottom", type=float, default=1.8)
    p.add_argument("--text-margin", type=float, default=0.10)
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    res = run_sign_test(
        capture=args.capture, n_frames=args.n_frames, device=args.device,
        batch_size=args.batch_size,
        model_name=args.clip_model, pretrained=args.clip_pretrained,
        expand_x=args.crop_expand_x, margin_top=args.crop_margin_top,
        margin_bottom=args.crop_margin_bottom, text_margin=args.text_margin,
    )

    print(f"\ncapture={args.capture}  frames={res['n_frames']}")
    print(f"{'condition':<26}{'cos_veh':>9}{'cos_road':>10}{'p_vehicle':>11}{'text_term':>11}")
    labels = {"clean": "clean (truck, no marker)", "marker": "marker (yellow quad)",
              "noleader": "noleader (empty road)"}
    for cond in ("clean", "marker", "noleader"):
        r = res[cond]
        print(f"{labels[cond]:<26}{r['cos_veh']:>9.4f}{r['cos_road']:>10.4f}"
              f"{r['p_vehicle']:>11.4f}{r['text_term']:>11.4f}")
    print(f"\nwin_rate p_vehicle (clean > noleader): {res['win_rate_p_vehicle']:.1%}")
    print(f"win_rate text_term (clean >= noleader): {res['win_rate_text_term']:.1%}")
    print(f"crops read as road — clean: {res['road_rate_clean']:.1%}  "
          f"noleader: {res['road_rate_noleader']:.1%}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(res, f, indent=2)

    assert_sign(res)
    print("\nSIGN TEST PASSED: the minimized term is higher on truck crops than "
          "on empty-road crops.")


if __name__ == "__main__":
    main()
