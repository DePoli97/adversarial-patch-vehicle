"""Train an adversarial patch that makes SimLingo accelerate into a braking leader.

Same Adam-over-pixels skeleton as `yolo_chroma_attack/train.py`; only the victim
changes. Per step:
    EOT augment the patch -> differentiably warp it onto the chroma-key quad ->
    feed the composite to the frozen SimLingo -> read its waypoints ->
    maximise the desired speed `control_pid` derives from them -> backprop into
    the patch pixels -> clamp to [0, 1].

Measured on vortex (RTX 4000 Ada, 20 GiB), 1024x512 frames:
    batch  1 ->  3.6 GiB, ~0.13 s/step
    batch  2 ->  5.3 GiB
    batch  4 ->  8.7 GiB, ~0.51 s/step   <- default
    batch  8 -> 15.7 GiB, ~1.06 s/step   <- practical ceiling
    batch 12 -> OOM

Usage on Vortex:
    conda activate PCLA15
    cd /home/vortex/adversarial-patch-vehicle
    PYTHONPATH=.:/home/vortex/PCLA:/home/vortex/PCLA/pcla_agents/simlingo \\
    python src/simlingo_chroma_attack/train.py \\
        --run-dir data/chroma_key_dataset/capture_20260609_014138_marker \\
        --out-dir experiments/simlingo_attack/run01 \\
        --epochs 10 --batch-size 4 --lr 0.02 --geom-eot --tv-weight 1.0
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# The tokenizer runs before the DataLoader forks its workers, which makes
# HuggingFace spam a fork warning on every batch. Must be set before
# `transformers` is imported (SimlingoWrapper imports it lazily).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from src.simlingo_chroma_attack.simlingo_loss import SimlingoSpeedUpLoss
from src.simlingo_chroma_attack.simlingo_model import CAM_H, CAM_W
from src.yolo_chroma_attack.dataset import ChromaKeyDataset, collate
from src.yolo_chroma_attack.eot import eot_apply, total_variation
from src.yolo_chroma_attack.patch_render import init_patch, render_patch_on_image

try:
    import wandb
except ImportError:
    wandb = None


def evaluate(loader, patch, speed_loss, device, max_batches: int | None = None,
             clean: bool = False):
    """Mean loss / desired speed / brake fraction over a loader.

    With `clean=True` the patch is not rendered — that is the baseline the
    attack has to beat.

    Speed jitter is disabled for the duration: the clean and the patched pass
    must be conditioned on the SAME ego speed or the delta between them is
    confounded by the sampling.
    """
    losses, speeds, brakes = [], [], []
    jitter = speed_loss.speed_jitter
    speed_loss.speed_jitter = None
    try:
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if max_batches is not None and i >= max_batches:
                    break
                img = batch["image"].to(device)
                if clean:
                    out = img
                else:
                    corners = batch["corners"].to(device)
                    illum = batch["illum"].to(device) if "illum" in batch else None
                    out = render_patch_on_image(img, patch, corners, illum=illum)
                _loss, info = speed_loss(out, batch["target_bbox"].to(device))
                losses.append(info["loss"])
                speeds.append(info["desired_speed_kmh"])
                brakes.append(info["brake_frac"])
    finally:
        speed_loss.speed_jitter = jitter
    n = max(1, len(losses))
    return sum(losses) / n, sum(speeds) / n, sum(brakes) / n


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="capture_<ts>/ folder with quads_index.json")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Where to dump patch.pt + previews + log")
    # --- victim ---
    p.add_argument("--pcla-root", default=None,
                   help="PCLA checkout (default: $PCLA_ROOT or /home/vortex/PCLA)")
    p.add_argument("--ckpt", default=None,
                   help="SimLingo pytorch_model.pt (default: shipped epoch=013)")
    # SimLingo's own camera is 1024x512; anything else is geometrically wrong.
    p.add_argument("--image-h", type=int, default=CAM_H)
    p.add_argument("--image-w", type=int, default=CAM_W)
    # --- patch ---
    p.add_argument("--patch-h", type=int, default=256)
    p.add_argument("--patch-w", type=int, default=512)
    p.add_argument("--patch-init", choices=["uniform", "gray"], default="uniform")
    # --- optimisation ---
    p.add_argument("--batch-size", type=int, default=4,
                   help="8 is the 20 GiB ceiling (15.7 GiB peak)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--cosine-lr", action="store_true")
    p.add_argument("--lr-min", type=float, default=1e-3)
    # --- loss ---
    p.add_argument("--speed-ms", type=float, default=8.0,
                   help="Ego speed the model is conditioned on (prompt + "
                        "vehicle_speed) and the brake predicate compares to")
    p.add_argument("--speed-jitter", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="Resample the ego speed each step from [LO, HI] m/s so "
                        "the patch does not overfit one conditioning")
    p.add_argument("--speed-cap", type=float, default=None,
                   help="Hinge: stop pushing a frame once its implied speed "
                        "exceeds this (m/s). None = plain maximisation.")
    p.add_argument("--w-pid", type=float, default=1.0,
                   help="Weight on ||wp[0]-wp[2]||*2, the scalar control_pid uses")
    p.add_argument("--w-long", type=float, default=0.5,
                   help="Weight on the mean implied speed over all 10 waypoints")
    p.add_argument("--w-fd", type=float, default=0.0,
                   help="Weight on the finite-difference speed profile")
    p.add_argument("--target-point", type=float, nargs=2, default=(20.0, 0.0))
    # --- EoT / regularisation ---
    p.add_argument("--eot-noise", type=float, default=0.02)
    p.add_argument("--geom-eot", action="store_true",
                   help="Random rotation/scale/translation of the patch. Needed "
                        "on low-pose-variety captures.")
    p.add_argument("--tv-weight", type=float, default=0.0,
                   help="Total-variation weight; >0 pushes toward printable, "
                        "smooth patterns. Typical 0.5-5.")
    p.add_argument("--target-expand-x", type=float, default=3.5)
    p.add_argument("--target-expand-y", type=float, default=3.5)
    p.add_argument("--illum-fix", action="store_true")
    p.add_argument("--illum-yellow-ref", type=float, default=0.65)
    # --- plumbing ---
    p.add_argument("--index-name", default="quads_index.json")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval-every", type=int, default=1, help="epochs")
    p.add_argument("--eval-batches", type=int, default=20)
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
    image_size = (args.image_h, args.image_w)
    target_expand = (args.target_expand_x, args.target_expand_y)
    illum_hw = (args.patch_h, args.patch_w) if args.illum_fix else None

    ds_kw = dict(seed=args.seed, image_size=image_size, target_expand=target_expand,
                 index_name=args.index_name, illum_patch_hw=illum_hw,
                 illum_yellow_ref=args.illum_yellow_ref)
    train_ds = ChromaKeyDataset(args.run_dir, split="train", **ds_kw)
    val_ds = ChromaKeyDataset(args.run_dir, split="val", **ds_kw)
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

    speed_loss = SimlingoSpeedUpLoss(
        pcla_root=args.pcla_root, ckpt=args.ckpt, device=args.device,
        speed_ms=args.speed_ms,
        speed_jitter=tuple(args.speed_jitter) if args.speed_jitter else None,
        target_point=tuple(args.target_point),
        next_target_point=(args.target_point[0] * 2.0, args.target_point[1] * 2.0),
        w_pid=args.w_pid, w_long=args.w_long, w_fd=args.w_fd,
        speed_cap_ms=args.speed_cap,
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

    log("# simlingo_chroma_attack — train run")
    log(f"args: {vars(args)}")

    # Two baselines: the untouched scene, and the scene with the random-init
    # patch. The attack has to beat the CLEAN one — that is the number that
    # says whether the patch changed SimLingo's mind.
    nb = args.eval_batches
    c_loss, c_speed, c_brake = evaluate(val_loader, None, speed_loss, device,
                                        max_batches=nb, clean=True)
    i_loss, i_speed, i_brake = evaluate(val_loader, patch.detach(), speed_loss,
                                        device, max_batches=nb)
    log(f"[clean] val desired_speed={c_speed:.2f} km/h  brake_frac={c_brake:.3f}")
    log(f"[init ] val desired_speed={i_speed:.2f} km/h  brake_frac={i_brake:.3f}  "
        f"loss={i_loss:.4f}")

    step = 0
    t0 = time.time()
    val_speed, val_brake, val_loss = i_speed, i_brake, i_loss
    for epoch in range(args.epochs):
        ep_loss, ep_speed, ep_brake = [], [], []
        for batch in train_loader:
            img = batch["image"].to(device, non_blocking=True)
            corners = batch["corners"].to(device, non_blocking=True)
            tgt = batch["target_bbox"].to(device, non_blocking=True)
            illum = batch["illum"].to(device, non_blocking=True) if "illum" in batch else None

            patch_aug = eot_apply(patch, noise_std=args.eot_noise, geom=args.geom_eot)
            out = render_patch_on_image(img, patch_aug, corners, illum=illum)
            speed_term, info = speed_loss(out, tgt)
            tv = total_variation(patch) if args.tv_weight > 0 else torch.zeros((), device=device)
            loss = speed_term + args.tv_weight * tv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                patch.clamp_(0.0, 1.0)

            ep_loss.append(info["loss"])
            ep_speed.append(info["desired_speed_kmh"])
            ep_brake.append(info["brake_frac"])
            if step % 20 == 0:
                log(f"  step {step:5d}  ep {epoch:2d}  loss={info['loss']:+.4f}  "
                    f"tv={float(tv):.4f}  desired={info['desired_speed_kmh']:6.2f} km/h  "
                    f"brake={info['brake_frac']:.2f}  ego={info['ego_speed_ms']:.1f} m/s")
                if use_wandb:
                    wandb.log({"step": step, "train/loss_step": info["loss"],
                               "train/tv_step": float(tv),
                               "train/desired_speed_kmh_step": info["desired_speed_kmh"],
                               "train/brake_frac_step": info["brake_frac"]})
            step += 1

        if scheduler is not None:
            scheduler.step()
        m_loss = sum(ep_loss) / max(1, len(ep_loss))
        m_speed = sum(ep_speed) / max(1, len(ep_speed))
        m_brake = sum(ep_brake) / max(1, len(ep_brake))

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            val_loss, val_speed, val_brake = evaluate(
                val_loader, patch.detach(), speed_loss, device, max_batches=nb)
            log(f"[ep {epoch+1:2d}/{args.epochs}] "
                f"train desired={m_speed:6.2f} km/h brake={m_brake:.3f} | "
                f"val desired={val_speed:6.2f} km/h brake={val_brake:.3f} "
                f"(clean {c_speed:.2f} / {c_brake:.3f}) | "
                f"loss t={m_loss:+.4f} v={val_loss:+.4f} | "
                f"{(time.time()-t0)/60:.1f} min")

            patch_path = args.out_dir / f"patch_ep{epoch+1:03d}.png"
            preview_path = args.out_dir / f"preview_ep{epoch+1:03d}.png"
            torch.save(patch.detach().cpu(), args.out_dir / f"patch_ep{epoch+1:03d}.pt")
            vutils.save_image(patch.detach().cpu(), patch_path)
            with torch.no_grad():
                vbatch = next(iter(val_loader))
                preview = render_patch_on_image(vbatch["image"].to(device),
                                                patch.detach(),
                                                vbatch["corners"].to(device))
                vutils.save_image(preview.cpu(), preview_path, nrow=2)

            if use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train/loss_epoch": m_loss,
                    "train/desired_speed_kmh": m_speed,
                    "train/brake_frac": m_brake,
                    "val/loss": val_loss,
                    "val/desired_speed_kmh": val_speed,
                    "val/brake_frac": val_brake,
                    "clean/desired_speed_kmh": c_speed,
                    "clean/brake_frac": c_brake,
                    "overfit/train_loss": m_loss,
                    "overfit/val_loss": val_loss,
                    "patch": wandb.Image(str(patch_path)),
                    "preview": wandb.Image(str(preview_path)),
                })

    torch.save(patch.detach().cpu(), args.out_dir / "patch_final.pt")
    vutils.save_image(patch.detach().cpu(), args.out_dir / "patch_final.png")
    summary = {
        "clean_desired_speed_kmh": c_speed, "clean_brake_frac": c_brake,
        "init_desired_speed_kmh": i_speed, "init_brake_frac": i_brake,
        "final_desired_speed_kmh": val_speed, "final_brake_frac": val_brake,
        "delta_desired_speed_kmh": val_speed - c_speed,
    }
    with open(args.out_dir / "args.json", "w") as f:
        json.dump({"args": vars(args), "summary": summary}, f, default=str, indent=2)
    log(f"\nsummary: {json.dumps(summary, indent=2)}")
    if use_wandb:
        wandb.log({f"final/{k}": v for k, v in summary.items()})
        wandb.finish()
    log(f"DONE. patch_final.pt -> {args.out_dir}")
    log_file.close()


if __name__ == "__main__":
    main()
