"""Train an adversarial patch that hides the leader from YOLOv8.

Pipeline:
  1. Load ChromaKeyDataset (1693 valid CARLA frames with quad index).
  2. Initialize a learnable patch tensor.
  3. Each step:
       - EOT augment the patch (photometric only)
       - Render onto the chroma-key quad in each image (differentiable warp)
       - Feed composite to frozen YOLOv8, compute hide loss (topk veh score)
       - Backprop, PGD-style update, clamp to [0, 1]
  4. Periodically eval on val split and dump checkpoints.

Usage on Vortex:
    conda activate PCLA310
    cd /home/vortex/adversarial-patch-vehicle
    python src/yolo_chroma_attack/train.py \\
        --run-dir data/chroma_key_dataset/capture_20260602_211812 \\
        --out-dir experiments/yolo_attack/run01 \\
        --epochs 20 --batch-size 8 --lr 0.02
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from src.yolo_chroma_attack.dataset import ChromaKeyDataset, collate
from src.yolo_chroma_attack.patch_render import init_patch, render_patch_on_image
from src.yolo_chroma_attack.yolo_loss import YoloHideLoss
from src.yolo_chroma_attack.eot import eot_apply, total_variation

try:
    import wandb
except ImportError:
    wandb = None


def evaluate(loader, patch, hide_loss, device, max_batches: int | None = None):
    """Compute mean hide-loss on a loader (no grad). Returns (loss, mean_in_box_score)."""
    losses, in_box_scores = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            img = batch["image"].to(device)
            corners = batch["corners"].to(device)
            tgt = batch["target_bbox"].to(device)
            out = render_patch_on_image(img, patch, corners)
            loss, info = hide_loss(out, tgt)
            losses.append(info["loss"])
            in_box_scores.append(info["veh_max_in_box_mean"])
    return sum(losses) / max(1, len(losses)), sum(in_box_scores) / max(1, len(in_box_scores))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="capture_<ts>/ folder with quads_index.json")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Where to dump patch.pt + previews + log")
    p.add_argument("--yolo-weights", default="yolov8n.pt")
    p.add_argument("--image-size", type=int, default=640)
    # Patch tensor dimensions. Default matches the yellow marker (512x256, AR=2.0)
    # so the patch learns texture in its native aspect ratio — the differentiable
    # warp only does perspective distortion, no anisotropic stretching, and the
    # exported TGA drops in 1:1 over yellow_marker.TGA.
    p.add_argument("--patch-h", type=int, default=256)
    p.add_argument("--patch-w", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--target-expand-x", type=float, default=3.5)
    p.add_argument("--target-expand-y", type=float, default=3.5)
    p.add_argument("--eot-noise", type=float, default=0.02)
    p.add_argument("--geom-eot", action="store_true",
                   help="Enable geometric EOT (random rotation/scale/translation "
                        "of the patch). Needed on low-pose-variety datasets to "
                        "avoid high-frequency-noise overfitting.")
    p.add_argument("--tv-weight", type=float, default=0.0,
                   help="Total-variation regularization weight. >0 penalizes "
                        "high-frequency noise, pushing toward smooth structured "
                        "patterns. Typical: 0.5-5.")
    p.add_argument("--topk", type=int, default=10,
                   help="Top-k anchor aggregation in hide loss")
    p.add_argument("--margin-tau", type=float, default=0.0,
                   help="Margin/hinge threshold: penalize only confidence above "
                        "tau (C&W-style). 0 disables (plain absolute loss). "
                        "~0.2 = detection threshold, stop once hidden.")
    p.add_argument("--cosine-lr", action="store_true",
                   help="Cosine LR schedule from --lr down to --lr-min")
    p.add_argument("--lr-min", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--patch-init", choices=["uniform", "gray"], default="uniform")
    p.add_argument("--index-name", default="quads_index.json",
                   help="JSON index file inside --run-dir. Use "
                        "'quads_index_visible.json' to train only on frames "
                        "where YOLO actually detects the leader in clean.")
    p.add_argument("--eval-every", type=int, default=1, help="epochs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-project", default=None,
                   help="If set, log this run to Weights & Biases under this project.")
    p.add_argument("--wandb-name", default=None, help="W&B run name (e.g. Town04_day)")
    p.add_argument("--wandb-group", default=None,
                   help="W&B group, to cluster runs from the same sweep")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    use_wandb = args.wandb_project is not None
    if use_wandb:
        if wandb is None:
            raise SystemExit("--wandb-project set but wandb is not installed")
        wandb.init(project=args.wandb_project, name=args.wandb_name,
                   group=args.wandb_group, config=vars(args))

    device = torch.device(args.device)
    image_size = (args.image_size, args.image_size)
    target_expand = (args.target_expand_x, args.target_expand_y)

    train_ds = ChromaKeyDataset(args.run_dir, split="train", seed=args.seed,
                                image_size=image_size, target_expand=target_expand,
                                index_name=args.index_name)
    val_ds = ChromaKeyDataset(args.run_dir, split="val", seed=args.seed,
                              image_size=image_size, target_expand=target_expand,
                              index_name=args.index_name)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate)

    print(f"train: {len(train_ds)} samples / val: {len(val_ds)} samples")
    print(f"image_size={image_size}, patch=({args.patch_h}, {args.patch_w}), "
          f"target_expand={target_expand}")

    patch = init_patch((3, args.patch_h, args.patch_w),
                       device=device, init=args.patch_init)
    optimizer = torch.optim.Adam([patch], lr=args.lr)

    hide_loss = YoloHideLoss(weights=args.yolo_weights, device=args.device,
                             conf_aggregation="topk", topk=args.topk,
                             margin_tau=args.margin_tau)
    scheduler = None
    if args.cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr_min)

    log_path = args.out_dir / "train.log"
    log_file = open(log_path, "w")

    def log(msg: str):
        print(msg, flush=True)
        print(msg, file=log_file, flush=True)

    log(f"# yolo_chroma_attack — train run")
    log(f"args: {vars(args)}")

    val_loss0, val_score0 = evaluate(val_loader, patch.detach(), hide_loss,
                                      device, max_batches=10)
    log(f"[init] val (10 batches): loss={val_loss0:.4f}  "
        f"mean_veh_score_in_box={val_score0:.4f}")

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        epoch_losses = []
        for batch in train_loader:
            img = batch["image"].to(device, non_blocking=True)
            corners = batch["corners"].to(device, non_blocking=True)
            tgt = batch["target_bbox"].to(device, non_blocking=True)

            patch_aug = eot_apply(patch, noise_std=args.eot_noise,
                                  geom=args.geom_eot)
            out = render_patch_on_image(img, patch_aug, corners)
            hide, info = hide_loss(out, tgt)
            # Total-variation regularization on the raw (un-augmented) patch:
            # smooths high-frequency noise into structured regions.
            tv = total_variation(patch) if args.tv_weight > 0 else torch.zeros((), device=device)
            loss = hide + args.tv_weight * tv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                patch.clamp_(0.0, 1.0)

            epoch_losses.append(info["loss"])
            if step % 20 == 0:
                log(f"  step {step:5d}  ep {epoch:2d}  hide={info['loss']:.4f}  "
                    f"tv={float(tv):.4f}  veh_in_box={info['veh_max_in_box_mean']:.4f}")
                if use_wandb:
                    wandb.log({"step": step, "train/hide_step": info["loss"],
                               "train/tv_step": float(tv),
                               "train/veh_in_box_step": info["veh_max_in_box_mean"]})
            step += 1

        if scheduler is not None:
            scheduler.step()
        mean_train = sum(epoch_losses) / max(1, len(epoch_losses))
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            val_loss, val_score = evaluate(val_loader, patch.detach(), hide_loss,
                                            device, max_batches=20)
            log(f"[ep {epoch+1:2d}/{args.epochs}] "
                f"train_loss={mean_train:.4f}  val_loss={val_loss:.4f}  "
                f"val_veh_score={val_score:.4f}  "
                f"elapsed={(time.time()-t0)/60:.1f} min")

            patch_path = args.out_dir / f"patch_ep{epoch+1:03d}.png"
            preview_path = args.out_dir / f"preview_ep{epoch+1:03d}.png"
            torch.save(patch.detach().cpu(), args.out_dir / f"patch_ep{epoch+1:03d}.pt")
            vutils.save_image(patch.detach().cpu(), patch_path)
            with torch.no_grad():
                # preview composite on the first val batch
                vbatch = next(iter(val_loader))
                vimg = vbatch["image"].to(device)
                vcorners = vbatch["corners"].to(device)
                preview = render_patch_on_image(vimg, patch.detach(), vcorners)
                vutils.save_image(preview.cpu(), preview_path, nrow=4)

            if use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train/loss_epoch": mean_train,
                    "val/loss": val_loss,
                    "val/veh_score": val_score,
                    "patch": wandb.Image(str(patch_path)),
                    "preview": wandb.Image(str(preview_path)),
                })

    # Final dump
    torch.save(patch.detach().cpu(), args.out_dir / "patch_final.pt")
    vutils.save_image(patch.detach().cpu(), args.out_dir / "patch_final.png")
    with open(args.out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, default=str, indent=2)
    if use_wandb:
        wandb.log({"final/val_veh_score": val_score, "final/val_loss": val_loss})
        wandb.finish()
    log(f"\nDONE. patch_final.pt -> {args.out_dir}")
    log_file.close()


if __name__ == "__main__":
    main()
