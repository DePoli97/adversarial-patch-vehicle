"""Train an adversarial patch against the OpenCLIP ViT-B/32 image encoder.

Pipeline:
  1. Load ClipChromaDataset (marker frame with quad index, plus the matching
     noleader frame when the loss needs it).
  2. Initialize a learnable 256x512 patch in [0, 1].
  3. Each step:
       - EOT augment the patch (photometric only)
       - Differentiably warp the patch onto the marker quad of every frame
       - Score the composite with the selected frozen-CLIP loss (--loss)
       - loss_tv  = TV(patch)                                (smoothness reg, optional)
       - total   = loss_adv + lambda_tv * loss_tv
       - backprop, Adam step, clamp patch to [0, 1]
  4. Periodically eval on val split, save patch_ep<N>.pt + preview composites
     (and, for --loss crop, the exact crops CLIP is scored on).

Losses (--loss):
  targeted : v1/v2. GLOBAL [CLS] cosine toward the no-leader frame.
             1 - cos(e_patched, e_target). Reproduces run01/run02; known to
             mode-collapse into high-frequency noise (global-embedding shortcut).
  crop     : v3, Option A of docs/clip_attack_survey.md. Crop the truck region
             out of the composite, resize to CLIP input, and score only that
             crop — against text prompts (vehicle vs empty road) and/or against
             the same crop taken from the no-leader frame (--crop-mode).

Loss versions tracked in the thesis log:
  v1 (run01_20260625_001129): targeted, lambda_tv = 0     -> high-frequency noise
  v2 (run02_20260625_131845): targeted, lambda_tv = 5e-2  -> smooth but still no structure
  v3 (this addition):         crop, text prompts on jittered truck crops

Usage on Vortex:
    conda activate PCLA310
    cd /home/vortex/adversarial-patch-vehicle

    # v2 (legacy global loss)
    PYTHONPATH=. python -u src/clip_chroma_attack/train.py --loss targeted \\
        --marker-dir data/chroma_key_dataset/capture_20260609_014138_marker \\
        --out-dir   experiments/clip_attack/run02 \\
        --epochs 100 --batch-size 16 --lr 0.02 --lambda-tv 5e-2 --device cuda

    # v3 (crop-based, no no-leader pass needed -> can use the fase1 pool)
    PYTHONPATH=. python -u src/clip_chroma_attack/train.py --loss crop \\
        --marker-dir data/chroma_key_dataset/fase1/_pooled_all \\
        --out-dir   experiments/clip_attack/run03 \\
        --crop-mode text --n-crops 3 --epochs 20 --batch-size 8 --lr 0.02
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from src.clip_chroma_attack.dataset import ClipChromaDataset, collate
from src.clip_chroma_attack.patch_render import init_patch, render_patch_on_image
from src.clip_chroma_attack.clip_loss import ClipCropLoss, ClipTargetedLoss, tv_loss
from src.clip_chroma_attack.crop_utils import crop_resize
from src.clip_chroma_attack.eot import eot_apply


def forward_loss(loss_fn, args, image, batch, device, jitter: bool = True):
    """Run the configured victim loss on an already-composited batch."""
    if args.loss == "targeted":
        noleader = batch["noleader_image"].to(device, non_blocking=True)
        return loss_fn(image, noleader)
    ref = None
    if args.crop_mode in ("image", "both"):
        ref = batch["noleader_image"].to(device, non_blocking=True)
    corners = batch["corners"].to(device, non_blocking=True)
    return loss_fn(image, corners=corners, ref_image=ref, jitter=jitter)


def evaluate(loader, patch, loss_fn, args, device, max_batches: int | None = None):
    """Mean of every scalar the loss reports, over (part of) a loader.

    Crops are NOT jittered here so the val numbers are comparable across epochs.
    """
    sums: dict[str, float] = {}
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            marker = batch["marker_image"].to(device)
            corners = batch["corners"].to(device)
            out = render_patch_on_image(marker, patch, corners)
            _, info = forward_loss(loss_fn, args, out, batch, device, jitter=False)
            for k, v in info.items():
                if isinstance(v, (int, float)):
                    sums[k] = sums.get(k, 0.0) + float(v)
            n += 1
    return {k: v / max(1, n) for k, v in sums.items()}


def fmt(d: dict) -> str:
    return "  ".join(f"{k}={v:.4f}" for k, v in d.items())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--marker-dir", type=Path, required=True,
                   help="capture_<ts>_marker/ folder with quads_index.json "
                        "(or a fase1 _pooled_* folder for --loss crop)")
    p.add_argument("--noleader-dir", type=Path, default=None,
                   help="Sibling capture_<ts>_noleader/ folder. Auto-resolved if not set.")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Where to dump patch.pt + previews + log")
    p.add_argument("--loss", choices=["targeted", "crop"], default="targeted",
                   help="targeted = v1/v2 global [CLS] cosine (mode-collapses to "
                        "noise); crop = v3 localized truck-crop loss.")
    p.add_argument("--clip-model", default="ViT-B-32",
                   help="OpenCLIP model name. ViT-B-32 is fastest (good for v1).")
    p.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    p.add_argument("--clip-image-size", type=int, default=224,
                   help="Native CLIP input size. ViT-B/32 and ViT-B/16 = 224.")
    # --- crop loss (v3) ---------------------------------------------------
    p.add_argument("--crop-mode", choices=["text", "image", "both"], default="text",
                   help="text: vehicle-vs-road prompts on the crop (no noleader "
                        "pass needed). image: pull the crop toward the same crop "
                        "of the noleader frame. both: weighted sum.")
    p.add_argument("--n-crops", type=int, default=3,
                   help="Jittered crops sampled per frame per step.")
    p.add_argument("--crop-scale-min", type=float, default=0.85)
    p.add_argument("--crop-scale-max", type=float, default=1.25)
    p.add_argument("--crop-shift", type=float, default=0.08,
                   help="Centre jitter as a fraction of the box side.")
    p.add_argument("--crop-expand-x", type=float, default=2.0,
                   help="Truck box width = marker width * this.")
    p.add_argument("--crop-margin-top", type=float, default=0.5,
                   help="Above the marker, in marker heights.")
    p.add_argument("--crop-margin-bottom", type=float, default=1.8,
                   help="Below the marker, in marker heights (truck body).")
    p.add_argument("--no-square-crop", action="store_true",
                   help="Do not grow the crop to a square before resizing.")
    p.add_argument("--text-objective", choices=["margin", "prob"], default="margin")
    p.add_argument("--text-margin", type=float, default=0.10)
    p.add_argument("--w-text", type=float, default=1.0)
    p.add_argument("--w-image", type=float, default=1.0)
    # --- rendering / optimization ----------------------------------------
    p.add_argument("--render-h", type=int, default=0,
                   help="Frame height fed to the warp. 0 = auto (native 720 for "
                        "--loss crop, clip_image_size for --loss targeted).")
    p.add_argument("--render-w", type=int, default=0, help="Frame width. 0 = auto.")
    p.add_argument("--patch-h", type=int, default=256)
    p.add_argument("--patch-w", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=0,
                   help="Stop after this many optimizer steps (0 = no limit). "
                        "Handy for short sanity runs.")
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--eot-noise", type=float, default=0.02)
    p.add_argument("--lambda-tv", type=float, default=0.0,
                   help="Weight of the Total Variation regularization term. "
                        "v1 baseline used 0 (got high-freq noise); "
                        "v2 uses 5e-2 (smooth low-freq patch).")
    p.add_argument("--cosine-lr", action="store_true",
                   help="Cosine LR schedule from --lr down to --lr-min")
    p.add_argument("--lr-min", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--patch-init", choices=["uniform", "gray"], default="uniform")
    p.add_argument("--index-name", default="quads_index.json",
                   help="JSON index inside --marker-dir; use 'quads_index_visible.json' "
                        "to train only on frames where the marker is visible.")
    p.add_argument("--eval-every", type=int, default=2, help="epochs")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if args.render_h > 0 and args.render_w > 0:
        image_size = (args.render_h, args.render_w)
    elif args.loss == "crop":
        # Native capture resolution: the far-distance crops are only ~200 px
        # wide in the original frame, downscaling the frame first would blur
        # exactly the region CLIP is asked to judge.
        image_size = (720, 1280)
    else:
        image_size = (args.clip_image_size, args.clip_image_size)

    # The text-only crop loss needs no no-leader pass, so it can train on the
    # fase1 pooled captures (3 towns x day/night x 3 distances).
    need_noleader = (args.loss == "targeted") or (args.crop_mode in ("image", "both"))

    ds_kwargs = dict(
        marker_dir=args.marker_dir, noleader_dir=args.noleader_dir,
        seed=args.seed, image_size=image_size, index_name=args.index_name,
        require_noleader=need_noleader,
    )
    train_ds = ClipChromaDataset(split="train", **ds_kwargs)
    val_ds = ClipChromaDataset(split="val", **ds_kwargs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate)

    print(f"train: {len(train_ds)} samples / val: {len(val_ds)} samples")
    print(f"image_size={image_size}, patch=({args.patch_h}, {args.patch_w})")

    patch = init_patch((3, args.patch_h, args.patch_w),
                       device=device, init=args.patch_init)
    optimizer = torch.optim.Adam([patch], lr=args.lr)

    if args.loss == "targeted":
        clip_loss = ClipTargetedLoss(
            model_name=args.clip_model, pretrained=args.clip_pretrained,
            device=args.device, image_size=args.clip_image_size,
        )
    else:
        clip_loss = ClipCropLoss(
            model_name=args.clip_model, pretrained=args.clip_pretrained,
            device=args.device, image_size=args.clip_image_size,
            mode=args.crop_mode, n_crops=args.n_crops,
            crop_scale=(args.crop_scale_min, args.crop_scale_max),
            crop_shift=args.crop_shift,
            expand_x=args.crop_expand_x,
            margin_top=args.crop_margin_top,
            margin_bottom=args.crop_margin_bottom,
            square_crop=not args.no_square_crop,
            text_objective=args.text_objective, text_margin=args.text_margin,
            w_text=args.w_text, w_image=args.w_image,
        )

    scheduler = None
    if args.cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr_min)

    log_path = args.out_dir / "train.log"
    log_file = open(log_path, "w")

    def log(msg: str):
        print(msg, flush=True)
        print(msg, file=log_file, flush=True)

    log(f"# clip_chroma_attack — train run ({args.loss})")
    log(f"args: {vars(args)}")

    def save_crop_preview(name: str):
        """Dump the exact crops CLIP scores, so the box geometry is auditable."""
        if args.loss != "crop":
            return
        with torch.no_grad():
            vbatch = next(iter(val_loader))
            vmarker = vbatch["marker_image"].to(device)
            vcorners = vbatch["corners"].to(device)
            comp = render_patch_on_image(vmarker, patch.detach(), vcorners)
            boxes = clip_loss.boxes_for(comp.shape[-2:], corners=vcorners, jitter=False)
            crops = crop_resize(comp, boxes, out_size=args.clip_image_size)
            vutils.save_image(crops.flatten(0, 1).cpu(), args.out_dir / name, nrow=4)

    val0 = evaluate(val_loader, patch.detach(), clip_loss, args, device, max_batches=10)
    log(f"[init] val (10 batches): {fmt(val0)}")
    save_crop_preview("crop_init.png")

    step = 0
    stop = False
    t0 = time.time()
    for epoch in range(args.epochs):
        epoch_losses = []
        for batch in train_loader:
            marker  = batch["marker_image"].to(device, non_blocking=True)
            corners = batch["corners"].to(device, non_blocking=True)

            patch_aug = eot_apply(patch, noise_std=args.eot_noise)
            out = render_patch_on_image(marker, patch_aug, corners)
            loss_adv, info = forward_loss(clip_loss, args, out, batch, device)
            if args.lambda_tv > 0:
                loss_tv_val = tv_loss(patch)
                loss = loss_adv + args.lambda_tv * loss_tv_val
                info["loss_tv"] = float(loss_tv_val.detach().cpu())
            else:
                loss = loss_adv
                info["loss_tv"] = 0.0
            info["loss_total"] = float(loss.detach().cpu())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                patch.clamp_(0.0, 1.0)

            epoch_losses.append(info["loss_total"])
            if step % 20 == 0:
                log(f"  step {step:5d}  ep {epoch:2d}  {fmt(info)}")
            step += 1
            if args.max_steps and step >= args.max_steps:
                stop = True
                break

        if scheduler is not None:
            scheduler.step()
        mean_train = sum(epoch_losses) / max(1, len(epoch_losses))
        if stop or (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            val = evaluate(val_loader, patch.detach(), clip_loss, args, device,
                           max_batches=20)
            log(f"[ep {epoch+1:2d}/{args.epochs}] train_loss={mean_train:.4f}  "
                f"val: {fmt(val)}  elapsed={(time.time()-t0)/60:.1f} min")

            torch.save(patch.detach().cpu(), args.out_dir / f"patch_ep{epoch+1:03d}.pt")
            vutils.save_image(patch.detach().cpu(),
                              args.out_dir / f"patch_ep{epoch+1:03d}.png")
            with torch.no_grad():
                vbatch = next(iter(val_loader))
                vmarker  = vbatch["marker_image"].to(device)
                vcorners = vbatch["corners"].to(device)
                preview = render_patch_on_image(vmarker, patch.detach(), vcorners)
                vutils.save_image(preview.cpu(),
                                  args.out_dir / f"preview_ep{epoch+1:03d}.png",
                                  nrow=4)
            save_crop_preview(f"crop_ep{epoch+1:03d}.png")
        if stop:
            log(f"stopping at --max-steps={args.max_steps}")
            break

    torch.save(patch.detach().cpu(), args.out_dir / "patch_final.pt")
    vutils.save_image(patch.detach().cpu(), args.out_dir / "patch_final.png")
    with open(args.out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, default=str, indent=2)
    log(f"\nDONE. patch_final.pt -> {args.out_dir}")
    log_file.close()


if __name__ == "__main__":
    main()
