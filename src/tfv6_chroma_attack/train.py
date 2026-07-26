"""Train an adversarial patch against TransFuser v6's own driving heads.

Adapted from `yolo_chroma_attack/train.py`: same Adam-over-pixels loop, the same
EoT and total-variation regularisers, the same checkpoint/preview cadence. The
only real change is the victim — `Tfv6HideLoss` (white-box, the policy's own
target-speed / waypoint / BEV-detection heads) instead of `YoloHideLoss` (an
external detector that did not transfer). Rendering, EoT and TV are imported
from the YOLO package rather than copied.

Two mechanical differences forced by the victim:
  * images are the 6-camera panorama (3, 384, 2304), so batches are small;
  * tfv6 wants raw [0, 255] pixels, so the composite is scaled by 255 after
    rendering (rendering itself stays in [0, 1], as does the patch).

Success signal: `expected_speed_ms` should RISE and `p0` (the stop-bin
probability that arms the hard-brake override at 0.9) should FALL.

Usage on Vortex:
    conda activate PCLA15
    cd /home/vortex/adversarial-patch-vehicle
    PYTHONPATH=.:/home/vortex/PCLA:/home/vortex/PCLA/pcla_agents/transfuserv6 \\
    python src/tfv6_chroma_attack/train.py \\
        --run-dir data/tfv6_chroma/capture_20260726 \\
        --out-dir experiments/tfv6_attack/run01 \\
        --epochs 20 --batch-size 2 --lr 0.02
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from src.tfv6_chroma_attack.dataset import Tfv6ChromaDataset, collate
from src.tfv6_chroma_attack.tfv6_loss import (
    DEFAULT_CKPT_DIR, DEFAULT_PCLA_ROOT, Tfv6HideLoss,
)
from src.yolo_chroma_attack.patch_render import init_patch, render_patch_on_image
from src.yolo_chroma_attack.eot import eot_apply, total_variation

try:
    import wandb
except ImportError:
    wandb = None

# tfv6 consumes raw [0, 255] pixels; ImageNet normalisation lives inside the
# backbone and is differentiable, so we scale here and optimise in [0, 1].
PIXEL_SCALE = 255.0


def composite(batch, patch, device, illum=None):
    """Render the patch onto a batch and hand back tfv6-scale pixels."""
    img = batch["image"].to(device, non_blocking=True)
    corners = batch["corners"].to(device, non_blocking=True)
    out = render_patch_on_image(img, patch, corners, illum=illum)
    return out * PIXEL_SCALE


def evaluate(loader, patch, hide_loss, device, max_batches: int | None = None):
    """Mean loss / expected speed / P(bin 0) on a loader (no grad)."""
    losses, speeds, p0s = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            illum = batch["illum"].to(device) if "illum" in batch else None
            leader_xy = batch.get("leader_xy")
            out = composite(batch, patch, device, illum=illum)
            _, info = hide_loss(out, None, leader_xy=leader_xy,
                                ego_speed=batch.get("speed"))
            losses.append(info["loss"])
            speeds.append(info["expected_speed_ms"])
            p0s.append(info["p0"])
    n = max(1, len(losses))
    return sum(losses) / n, sum(speeds) / n, sum(p0s) / n


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="capture folder with the tfv6 frames + quads index")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Where to dump patch.pt + previews + log")
    # --- dataset layout ---
    p.add_argument("--index-name", default="quads_index.json")
    p.add_argument("--image-layout", choices=["stitched", "cameras"],
                   default="stitched",
                   help="'stitched': one 2304x384 png per frame. 'cameras': six "
                        "{stem}_cam{i}.png concatenated along width.")
    p.add_argument("--corners-frame", choices=["stitched", "front"],
                   default="stitched",
                   help="Coordinate frame the stored quad lives in. 'front' "
                        "means the 384x384 front camera; x is offset into the "
                        "panorama. Never auto-detected — state it.")
    p.add_argument("--front-cam-index", type=int, default=2,
                   help="Rig index of the forward camera (yaw 0).")
    p.add_argument("--leader-source", choices=["auto", "index", "meta", "none"],
                   default="auto",
                   help="Where the per-frame leader BEV position comes from. "
                        "'meta' derives it from the {stem}.json sidecar written "
                        "by capture_tfv6.py.")
    p.add_argument("--target-expand-x", type=float, default=3.5)
    p.add_argument("--target-expand-y", type=float, default=3.5)
    # --- patch / optimisation ---
    p.add_argument("--patch-h", type=int, default=256)
    p.add_argument("--patch-w", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2,
                   help="Small by default: each sample is a 3x384x2304 panorama.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--eot-noise", type=float, default=0.02)
    p.add_argument("--geom-eot", action="store_true",
                   help="Enable geometric EOT (random rotation/scale/translation "
                        "of the patch). Needed on low-pose-variety datasets to "
                        "avoid high-frequency-noise overfitting.")
    p.add_argument("--tv-weight", type=float, default=0.0,
                   help="Total-variation regularization weight. >0 penalizes "
                        "high-frequency noise, pushing toward smooth structured "
                        "patterns. Typical: 0.5-5.")
    p.add_argument("--cosine-lr", action="store_true")
    p.add_argument("--lr-min", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--patch-init", choices=["uniform", "gray"], default="uniform")
    p.add_argument("--illum-fix", action="store_true",
                   help="Light the patch by each frame's marker luminance "
                        "(closes the train-deploy night lighting gap).")
    p.add_argument("--illum-yellow-ref", type=float, default=0.65)
    # --- victim ---
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    p.add_argument("--pcla-root", default=DEFAULT_PCLA_ROOT)
    p.add_argument("--single-ckpt", action="store_true",
                   help="Use one checkpoint instead of the 3-model ensemble. "
                        "~3x faster; for development only — real inference "
                        "averages all three.")
    # --- loss weights ---
    p.add_argument("--w-expected-speed", type=float, default=1.0)
    p.add_argument("--w-p0", type=float, default=1.0)
    p.add_argument("--p0-form", choices=["logbarrier", "linear"],
                   default="logbarrier")
    p.add_argument("--speed-margin", type=float, default=0.0,
                   help="C&W hinge on expected speed (m/s): stop pushing once "
                        "E exceeds it. 0 disables.")
    p.add_argument("--p0-margin", type=float, default=0.0,
                   help="C&W hinge on P(bin 0). 0 disables.")
    p.add_argument("--lambda-wp", type=float, default=0.0)
    p.add_argument("--lambda-detect", type=float, default=0.0)
    p.add_argument("--detect-mode", choices=["shrink", "suppress", "farther"],
                   default="shrink")
    p.add_argument("--bev-locate", choices=["analytic", "auto"], default="analytic")
    p.add_argument("--leader-distance", type=float, default=None,
                   help="Fallback leader distance ahead of ego (m), used when a "
                        "frame carries no leader_xy.")
    p.add_argument("--leader-lateral", type=float, default=0.0)
    p.add_argument("--ego-speed", type=float, default=8.0,
                   help="Ego speed (m/s) fed to the model when a frame carries "
                        "no per-frame speed.")
    # --- bookkeeping ---
    p.add_argument("--eval-every", type=int, default=1, help="epochs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-group", default=None)
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
    illum_hw = (args.patch_h, args.patch_w) if args.illum_fix else None
    ds_kwargs = dict(
        seed=args.seed, index_name=args.index_name,
        image_layout=args.image_layout, corners_frame=args.corners_frame,
        front_cam_index=args.front_cam_index, leader_source=args.leader_source,
        target_expand=(args.target_expand_x, args.target_expand_y),
        illum_patch_hw=illum_hw, illum_yellow_ref=args.illum_yellow_ref,
    )
    train_ds = Tfv6ChromaDataset(args.run_dir, split="train", **ds_kwargs)
    val_ds = Tfv6ChromaDataset(args.run_dir, split="val", **ds_kwargs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate)

    print(f"train: {len(train_ds)} samples / val: {len(val_ds)} samples")
    print(f"patch=({args.patch_h}, {args.patch_w})")

    patch = init_patch((3, args.patch_h, args.patch_w),
                       device=device, init=args.patch_init)
    optimizer = torch.optim.Adam([patch], lr=args.lr)

    hide_loss = Tfv6HideLoss(
        ckpt_dir=args.ckpt_dir, device=args.device, pcla_root=args.pcla_root,
        ensemble=not args.single_ckpt,
        w_expected_speed=args.w_expected_speed, w_p0=args.w_p0,
        p0_form=args.p0_form,
        speed_margin_ms=args.speed_margin or None, p0_margin=args.p0_margin,
        lambda_wp=args.lambda_wp,
        lambda_detect=args.lambda_detect, detect_mode=args.detect_mode,
        bev_locate=args.bev_locate,
        leader_distance_m=args.leader_distance,
        leader_lateral_m=args.leader_lateral,
        ego_speed_ms=args.ego_speed,
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

    log("# tfv6_chroma_attack — train run")
    log(f"args: {vars(args)}")
    log(f"victim: {len(hide_loss.nets)} checkpoint(s) {hide_loss.ckpt_files}")

    val_loss0, val_speed0, val_p00 = evaluate(val_loader, patch.detach(), hide_loss,
                                              device, max_batches=10)
    log(f"[init] val (10 batches): loss={val_loss0:.4f}  "
        f"speed={val_speed0:.3f} m/s  p0={val_p00:.4f}")

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        epoch_losses = []
        for batch in train_loader:
            illum = batch["illum"].to(device, non_blocking=True) if "illum" in batch else None
            patch_aug = eot_apply(patch, noise_std=args.eot_noise, geom=args.geom_eot)
            out = composite(batch, patch_aug, device, illum=illum)
            hide, info = hide_loss(out, None, leader_xy=batch.get("leader_xy"),
                                   ego_speed=batch.get("speed"))
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
                log(f"  step {step:5d}  ep {epoch:2d}  loss={info['loss']:.4f}  "
                    f"tv={float(tv):.4f}  speed={info['expected_speed_ms']:.3f} m/s  "
                    f"p0={info['p0']:.4f}")
                if use_wandb:
                    wandb.log({"step": step, "train/loss_step": info["loss"],
                               "train/tv_step": float(tv),
                               "train/speed_step": info["expected_speed_ms"],
                               "train/p0_step": info["p0"]})
            step += 1

        if scheduler is not None:
            scheduler.step()
        mean_train = sum(epoch_losses) / max(1, len(epoch_losses))
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            val_loss, val_speed, val_p0 = evaluate(val_loader, patch.detach(),
                                                   hide_loss, device, max_batches=20)
            log(f"[ep {epoch+1:2d}/{args.epochs}] "
                f"train_loss={mean_train:.4f}  val_loss={val_loss:.4f}  "
                f"val_speed={val_speed:.3f} m/s  val_p0={val_p0:.4f}  "
                f"elapsed={(time.time()-t0)/60:.1f} min")

            patch_path = args.out_dir / f"patch_ep{epoch+1:03d}.png"
            preview_path = args.out_dir / f"preview_ep{epoch+1:03d}.png"
            torch.save(patch.detach().cpu(), args.out_dir / f"patch_ep{epoch+1:03d}.pt")
            vutils.save_image(patch.detach().cpu(), patch_path)
            with torch.no_grad():
                vbatch = next(iter(val_loader))
                preview = composite(vbatch, patch.detach(), device) / PIXEL_SCALE
                vutils.save_image(preview.cpu(), preview_path, nrow=1)

            if use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train/loss_epoch": mean_train,
                    "val/loss": val_loss,
                    "val/speed": val_speed,
                    "val/p0": val_p0,
                    # Same two curves under one prefix so W&B plots them on a
                    # single chart -> overfitting is visible at a glance.
                    "overfit/train_loss": mean_train,
                    "overfit/val_loss": val_loss,
                    "patch": wandb.Image(str(patch_path)),
                    "preview": wandb.Image(str(preview_path)),
                })

    torch.save(patch.detach().cpu(), args.out_dir / "patch_final.pt")
    vutils.save_image(patch.detach().cpu(), args.out_dir / "patch_final.png")
    with open(args.out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, default=str, indent=2)
    if use_wandb:
        wandb.log({"final/val_speed": val_speed, "final/val_p0": val_p0,
                   "final/val_loss": val_loss})
        wandb.finish()
    log(f"\nDONE. patch_final.pt -> {args.out_dir}")
    log(f"init speed={val_speed0:.3f} -> final {val_speed:.3f} m/s   "
        f"init p0={val_p00:.4f} -> final {val_p0:.4f}")
    log_file.close()


if __name__ == "__main__":
    main()
