"""Train an adversarial patch against OpenCLIP ViT-B/32 image encoder.

Pipeline:
  1. Load ClipChromaDataset (marker frame with quad index + matching noleader frame).
  2. Initialize a learnable 256x512 patch in [0, 1].
  3. Each step:
       - EOT augment the patch (photometric only)
       - Differentiably warp the patch onto the marker quad of every frame
       - Forward composite through frozen CLIP -> e_patched
       - Forward noleader frame through frozen CLIP (no_grad) -> e_target
       - loss_adv = 1 - cos(e_patched, e_target)             (targeted attack)
       - loss_tv  = TV(patch)                                (smoothness reg, optional)
       - total   = loss_adv + lambda_tv * loss_tv
       - backprop, Adam step, clamp patch to [0, 1]
  4. Periodically eval on val split, save patch_ep<N>.pt + preview composites.

Loss versions tracked in the thesis log:
  v1 (run01_20260625_001129): lambda_tv = 0       -> patch is high-frequency noise
  v2 (this default):          lambda_tv = 2.5e-3  -> smooth low-frequency patch

Usage on Vortex:
    conda activate PCLA310
    cd /home/vortex/adversarial-patch-vehicle
    PYTHONPATH=. python -u src/clip_chroma_attack/train.py \\
        --marker-dir data/chroma_key_dataset/capture_20260609_014138_marker \\
        --out-dir   experiments/clip_attack/run02 \\
        --epochs 100 --batch-size 16 --lr 0.02 --lambda-tv 2.5e-3 --device cuda
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
from src.clip_chroma_attack.clip_loss import ClipTargetedLoss, tv_loss
from src.clip_chroma_attack.eot import eot_apply


def evaluate(loader, patch, clip_loss, device, max_batches: int | None = None):
    losses, cos_means = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            marker = batch["marker_image"].to(device)
            noleader = batch["noleader_image"].to(device)
            corners = batch["corners"].to(device)
            out = render_patch_on_image(marker, patch, corners)
            _, info = clip_loss(out, noleader)
            losses.append(info["loss"])
            cos_means.append(info["cos_mean"])
    return (sum(losses) / max(1, len(losses)),
            sum(cos_means) / max(1, len(cos_means)))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--marker-dir", type=Path, required=True,
                   help="capture_<ts>_marker/ folder with quads_index.json")
    p.add_argument("--noleader-dir", type=Path, default=None,
                   help="Sibling capture_<ts>_noleader/ folder. Auto-resolved if not set.")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Where to dump patch.pt + previews + log")
    p.add_argument("--clip-model", default="ViT-B-32",
                   help="OpenCLIP model name. ViT-B-32 is fastest (good for v1).")
    p.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    p.add_argument("--clip-image-size", type=int, default=224,
                   help="Native CLIP input size. ViT-B/32 and ViT-B/16 = 224.")
    p.add_argument("--patch-h", type=int, default=256)
    p.add_argument("--patch-w", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--eot-noise", type=float, default=0.02)
    p.add_argument("--lambda-tv", type=float, default=0.0,
                   help="Weight of the Total Variation regularization term. "
                        "v1 baseline used 0 (got high-freq noise); "
                        "v2 uses 2.5e-3 (smooth low-freq patch).")
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
    image_size = (args.clip_image_size, args.clip_image_size)

    train_ds = ClipChromaDataset(
        marker_dir=args.marker_dir, noleader_dir=args.noleader_dir,
        split="train", seed=args.seed,
        image_size=image_size, index_name=args.index_name,
    )
    val_ds = ClipChromaDataset(
        marker_dir=args.marker_dir, noleader_dir=args.noleader_dir,
        split="val", seed=args.seed,
        image_size=image_size, index_name=args.index_name,
    )
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

    clip_loss = ClipTargetedLoss(
        model_name=args.clip_model, pretrained=args.clip_pretrained,
        device=args.device, image_size=args.clip_image_size,
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

    log(f"# clip_chroma_attack — train run")
    log(f"args: {vars(args)}")

    val_loss0, val_cos0 = evaluate(val_loader, patch.detach(), clip_loss,
                                    device, max_batches=10)
    log(f"[init] val (10 batches): loss={val_loss0:.4f}  cos_mean={val_cos0:.4f}")

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        epoch_losses = []
        for batch in train_loader:
            marker   = batch["marker_image"].to(device, non_blocking=True)
            noleader = batch["noleader_image"].to(device, non_blocking=True)
            corners  = batch["corners"].to(device, non_blocking=True)

            patch_aug = eot_apply(patch, noise_std=args.eot_noise)
            out = render_patch_on_image(marker, patch_aug, corners)
            loss_adv, info = clip_loss(out, noleader)
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
                log(f"  step {step:5d}  ep {epoch:2d}  total={info['loss_total']:.4f}  "
                    f"adv={info['loss']:.4f}  tv={info['loss_tv']:.4f}  "
                    f"cos={info['cos_mean']:.4f}")
            step += 1

        if scheduler is not None:
            scheduler.step()
        mean_train = sum(epoch_losses) / max(1, len(epoch_losses))
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            val_loss, val_cos = evaluate(val_loader, patch.detach(), clip_loss,
                                          device, max_batches=20)
            log(f"[ep {epoch+1:2d}/{args.epochs}] "
                f"train_loss={mean_train:.4f}  val_loss={val_loss:.4f}  "
                f"val_cos={val_cos:.4f}  "
                f"elapsed={(time.time()-t0)/60:.1f} min")

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

    torch.save(patch.detach().cpu(), args.out_dir / "patch_final.pt")
    vutils.save_image(patch.detach().cpu(), args.out_dir / "patch_final.png")
    with open(args.out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, default=str, indent=2)
    log(f"\nDONE. patch_final.pt -> {args.out_dir}")
    log_file.close()


if __name__ == "__main__":
    main()
