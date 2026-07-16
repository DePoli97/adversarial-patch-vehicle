"""Evaluate an adversarial patch against YOLO on the held-out split.

Metric mirrors Yang 2020 Table 1 / Muller 2022 success rate:
  detection_rate = fraction of frames where YOLO predicts at least one
                   vehicle (car/bus/truck) whose center falls inside the
                   target_bbox, with conf >= threshold.

We report three numbers:
  - clean (no patch)         → upper bound (~1.0 expected)
  - random patch             → baseline (texture noise inside the quad)
  - trained patch            → attack performance

Usage:
    python src/yolo_chroma_attack/evaluate.py \\
        --run-dir data/chroma_key_dataset/capture_20260602_211812 \\
        --patch experiments/yolo_attack/run01/patch_final.pt \\
        --out experiments/yolo_attack/run01/eval.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from ultralytics import YOLO

from src.yolo_chroma_attack.dataset import ChromaKeyDataset, collate
from src.yolo_chroma_attack.patch_render import init_patch, render_patch_on_image


VEHICLE_CLASSES = (2, 5, 7)


def has_vehicle_in_bbox(image_bchw01: torch.Tensor, target_bbox_b4: torch.Tensor,
                       yolo, conf_threshold: float = 0.25) -> list[bool]:
    """Return list of bool — True if YOLO detects a vehicle inside the target bbox."""
    B = image_bchw01.shape[0]
    # YOLO predict expects HWC uint8 or float; ultralytics accepts a list of
    # numpy/tensor in (B, 3, H, W) float [0,1] -> we pass as a numpy list.
    imgs_np = (image_bchw01.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).clip(0, 255).astype("uint8")
    results = yolo.predict(list(imgs_np), verbose=False, conf=conf_threshold, imgsz=image_bchw01.shape[-1])
    out = []
    for b, r in enumerate(results):
        x1, y1, x2, y2 = target_bbox_b4[b].tolist()
        found = False
        if r.boxes is not None and len(r.boxes) > 0:
            for box, cls in zip(r.boxes.xywh.cpu().numpy(), r.boxes.cls.cpu().int().tolist()):
                if cls not in VEHICLE_CLASSES:
                    continue
                cx, cy = float(box[0]), float(box[1])
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    found = True
                    break
        out.append(found)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Marker dataset (with yellow chroma-key). Used for "
                        "random/trained eval — patch is rendered onto the quad.")
    p.add_argument("--clean-run-dir", type=Path, default=None,
                   help="Optional paired CLEAN dataset (no marker). If set, the "
                        "'clean' baseline reads frames from here instead of the "
                        "marker frames without patch. Frame N must correspond.")
    p.add_argument("--patch", type=Path, required=True,
                   help="Trained patch .pt (3, Ph, Pw) in [0, 1].")
    p.add_argument("--out", type=Path, required=True, help="Output JSON path.")
    p.add_argument("--yolo-weights", default="yolov8n.pt")
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--conf-threshold", type=float, default=0.25)
    p.add_argument("--target-expand-x", type=float, default=3.5)
    p.add_argument("--target-expand-y", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-batches", type=int, default=None,
                   help="If set, limit eval to first N batches (debug).")
    p.add_argument("--save-previews", type=int, default=8,
                   help="N composite previews to save per condition.")
    p.add_argument("--illum-fix", action="store_true",
                   help="Light the patch by each frame's marker luminance "
                        "(deployment-realistic; must match how the patch was trained).")
    p.add_argument("--illum-yellow-ref", type=float, default=0.65)
    p.add_argument("--patch-h", type=int, default=256)
    p.add_argument("--patch-w", type=int, default=512)
    args = p.parse_args()

    device = torch.device(args.device)
    target_expand = (args.target_expand_x, args.target_expand_y)
    image_size = (args.image_size, args.image_size)

    illum_hw = (args.patch_h, args.patch_w) if args.illum_fix else None
    val_ds = ChromaKeyDataset(args.run_dir, split="val", seed=args.seed,
                              image_size=image_size, target_expand=target_expand,
                              illum_patch_hw=illum_hw,
                              illum_yellow_ref=args.illum_yellow_ref)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, collate_fn=collate)
    print(f"val (marker): {len(val_ds)} frames")
    # Optional paired clean dataset (true baseline: red CarlaCola, no marker).
    # Frames share filenames with marker, so we just swap the directory.
    clean_dir = args.clean_run_dir if args.clean_run_dir else None
    if clean_dir is not None:
        print(f"clean baseline dataset: {clean_dir}")

    # Load trained patch
    patch = torch.load(args.patch, map_location=device)
    if patch.dim() == 4:
        patch = patch.squeeze(0)
    patch = patch.to(device).clamp(0, 1)
    print(f"loaded patch: {tuple(patch.shape)} from {args.patch}")

    # Random patch baseline
    random_patch = init_patch(patch.shape, device=device).detach().clamp(0, 1)

    yolo = YOLO(args.yolo_weights)
    yolo.model.to(device).eval()

    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    counters = {
        "clean": {"detected": 0, "total": 0},
        "random": {"detected": 0, "total": 0},
        "trained": {"detected": 0, "total": 0},
    }
    previews_saved = 0

    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            if args.max_batches is not None and bi >= args.max_batches:
                break
            img = batch["image"].to(device)
            corners = batch["corners"].to(device)
            tgt = batch["target_bbox"]
            illum = batch["illum"].to(device) if "illum" in batch else None

            # 1. clean — either marker frame without patch (default), or the
            #    paired truly-clean frame from --clean-run-dir if provided.
            clean_img = img
            if clean_dir is not None:
                frame_ids = batch["stems"]
                clean_imgs = []
                for fid in frame_ids:
                    cp = clean_dir / f"{fid}.png"
                    cbgr = cv2.imread(str(cp))
                    if cbgr is None:
                        clean_imgs.append(img[len(clean_imgs)].cpu())
                        continue
                    crgb = cv2.cvtColor(cbgr, cv2.COLOR_BGR2RGB)
                    crgb = cv2.resize(crgb, (image_size[1], image_size[0]))
                    clean_imgs.append(torch.from_numpy(crgb).permute(2, 0, 1).float() / 255.0)
                clean_img = torch.stack(clean_imgs).to(device)
            for det in has_vehicle_in_bbox(clean_img, tgt, yolo, args.conf_threshold):
                counters["clean"]["total"] += 1
                if det:
                    counters["clean"]["detected"] += 1

            # 2. random
            out_random = render_patch_on_image(img, random_patch, corners, illum=illum)
            for det in has_vehicle_in_bbox(out_random, tgt, yolo, args.conf_threshold):
                counters["random"]["total"] += 1
                if det:
                    counters["random"]["detected"] += 1

            # 3. trained
            out_trained = render_patch_on_image(img, patch, corners, illum=illum)
            for det in has_vehicle_in_bbox(out_trained, tgt, yolo, args.conf_threshold):
                counters["trained"]["total"] += 1
                if det:
                    counters["trained"]["detected"] += 1

            # save a few previews from the first batch
            if previews_saved < args.save_previews and bi == 0:
                vutils.save_image(img[:args.save_previews].cpu(),
                                   out_dir / "preview_clean.png", nrow=4)
                vutils.save_image(out_random[:args.save_previews].cpu(),
                                   out_dir / "preview_random.png", nrow=4)
                vutils.save_image(out_trained[:args.save_previews].cpu(),
                                   out_dir / "preview_trained.png", nrow=4)
                previews_saved = args.save_previews

            if bi % 10 == 0:
                print(f"  batch {bi}/{len(val_loader)} "
                      f"clean={counters['clean']['detected']}/{counters['clean']['total']} "
                      f"random={counters['random']['detected']}/{counters['random']['total']} "
                      f"trained={counters['trained']['detected']}/{counters['trained']['total']}",
                      flush=True)

    summary = {}
    for k, c in counters.items():
        rate = c["detected"] / max(1, c["total"])
        summary[k] = {"detected": c["detected"], "total": c["total"], "detection_rate": rate}
        print(f"{k:8s}: {c['detected']}/{c['total']} = {rate*100:.1f}%")

    summary["args"] = vars(args)
    summary["args"] = {k: str(v) for k, v in summary["args"].items()}
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
