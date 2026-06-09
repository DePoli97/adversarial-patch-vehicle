"""Standalone analysis of a trained adversarial patch on the paired clean/marker
val set. Produces a self-contained HTML report with metrics tables, class
confusion, side-by-side samples, and confidence histograms.

Excludes frames where YOLO already misses the leader on clean — there is no
meaningful delta on a frame where confidence was already 0.

Usage:
    python -m src.yolo_chroma_attack.analyze_patch \\
        --patch experiments/yolo_attack/run02_20260609_093207/patch_final.pt \\
        --marker-dir data/chroma_key_dataset/capture_20260609_014138_marker \\
        --clean-dir  data/chroma_key_dataset/capture_20260609_014138_clean \\
        --out-dir   experiments/yolo_attack/run02_20260609_093207/analysis
"""
import argparse
import base64
import io
import json
from collections import Counter
from pathlib import Path
from random import Random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}
CONF_THRESHOLD = 0.25


def order_corners(c):
    c = np.asarray(c, np.float32)
    s = c.sum(1)
    d = np.diff(c, axis=1).reshape(-1)
    return np.stack([c[np.argmin(s)], c[np.argmin(d)], c[np.argmax(s)], c[np.argmax(d)]])


def target_bbox(corners, img_shape, expand=2.5):
    H, W = img_shape[:2]
    cx, cy = corners[:, 0].mean(), corners[:, 1].mean()
    hw = (corners[:, 0].max() - corners[:, 0].min()) * 0.5 * expand
    hh = (corners[:, 1].max() - corners[:, 1].min()) * 0.5 * expand
    return np.array([max(0, cx - hw), max(0, cy - hh),
                     min(W - 1, cx + hw), min(H - 1, cy + hh)], np.float32)


def warp_patch_onto(img_bgr, patch_chw, corners):
    Ph, Pw = patch_chw.shape[1], patch_chw.shape[2]
    patch_bgr = (patch_chw.permute(1, 2, 0).numpy()[..., ::-1] * 255).clip(0, 255).astype("uint8")
    src = np.array([[0, 0], [Pw - 1, 0], [Pw - 1, Ph - 1], [0, Ph - 1]], np.float32)
    dst = order_corners(corners)
    M = cv2.getPerspectiveTransform(src, dst)
    Hf, Wf = img_bgr.shape[:2]
    warped = cv2.warpPerspective(patch_bgr, M, (Wf, Hf))
    mask = np.ones((Ph, Pw), np.uint8) * 255
    warped_mask = cv2.warpPerspective(mask, M, (Wf, Hf))
    out = img_bgr.copy()
    out[warped_mask > 0] = warped[warped_mask > 0]
    return out


def yolo_leader(bgr, tbox, yolo):
    res = yolo.predict(bgr, verbose=False, conf=CONF_THRESHOLD)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None, 0.0, None
    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clss = res.boxes.cls.cpu().int().tolist()
    cands = []
    for box, conf, cls in zip(boxes, confs, clss):
        if cls not in VEHICLE_CLASSES:
            continue
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        if tbox[0] <= cx <= tbox[2] and tbox[1] <= cy <= tbox[3]:
            cands.append((conf, box, cls))
    if not cands:
        return None, 0.0, None
    cands.sort(reverse=True, key=lambda t: t[0])
    return cands[0][1], float(cands[0][0]), VEHICLE_CLASSES[cands[0][2]]


def draw_box(bgr, box, conf, cls, color):
    img = bgr.copy()
    if box is None:
        return img
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    cv2.putText(img, f"{cls} {conf:.2f}", (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--patch", type=Path, required=True)
    p.add_argument("--marker-dir", type=Path, required=True)
    p.add_argument("--clean-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--yolo-weights", default="src/vehicle_counting_model/yolov8n.pt")
    p.add_argument("--expand", type=float, default=2.5)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-examples", type=int, default=8)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    QUADS = json.loads((args.marker_dir / "quads_index.json").read_text())
    PATCH = torch.load(args.patch, map_location="cpu").clamp(0, 1)
    if PATCH.dim() == 4:
        PATCH = PATCH.squeeze(0)
    print(f"patch shape: {tuple(PATCH.shape)}")
    yolo = YOLO(args.yolo_weights)

    rng = Random(args.seed)
    all_stems = sorted(QUADS.keys())
    rng.shuffle(all_stems)
    n_val = int(args.val_fraction * len(all_stems))
    val_stems = sorted(all_stems[:n_val])
    print(f"val set: {len(val_stems)} frames")

    rows = []
    for i, stem in enumerate(val_stems):
        if i % 50 == 0:
            print(f"  {i}/{len(val_stems)}", flush=True)
        cp = args.clean_dir / f"{stem}.png"
        mp = args.marker_dir / f"{stem}.png"
        if not cp.exists() or not mp.exists():
            continue
        clean_bgr = cv2.imread(str(cp))
        marker_bgr = cv2.imread(str(mp))
        corners = np.asarray(QUADS[stem]["corners"], np.float32)
        tbox = target_bbox(corners, clean_bgr.shape, expand=args.expand)
        patched_bgr = warp_patch_onto(marker_bgr, PATCH, corners)
        _, cc, kc = yolo_leader(clean_bgr, tbox, yolo)
        _, cp_, kp = yolo_leader(patched_bgr, tbox, yolo)
        rows.append({
            "stem": stem, "conf_clean": cc, "cls_clean": kc,
            "conf_patched": cp_, "cls_patched": kp,
            "leader_visible_clean": cc > 0, "leader_visible_patched": cp_ > 0,
        })
    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "per_frame.csv", index=False)

    n_total = len(df)
    n_seen = int(df["leader_visible_clean"].sum())
    n_blind = n_total - n_seen
    df_seen = df[df["leader_visible_clean"]].copy()
    n_hidden = int((~df_seen["leader_visible_patched"]).sum())
    mc = df_seen["conf_clean"].mean()
    mp_ = df_seen["conf_patched"].mean()
    delta = mp_ - mc
    delta_pct = 100 * delta / mc if mc > 0 else 0.0

    summary = {
        "n_total": n_total, "n_visible_in_clean": n_seen, "n_blind_in_clean": n_blind,
        "mean_conf_clean": float(mc), "mean_conf_patched": float(mp_),
        "delta_absolute": float(delta), "delta_relative_pct": float(delta_pct),
        "n_hidden_by_patch": n_hidden,
        "det_rate_clean_all": float(df["leader_visible_clean"].mean()),
        "det_rate_patched_all": float(df["leader_visible_patched"].mean()),
        "det_rate_patched_on_visible": float(df_seen["leader_visible_patched"].mean()),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # Class confusion
    cc_counts = Counter(df_seen["cls_clean"])
    cp_counts = Counter(df_seen["cls_patched"].fillna("<HIDDEN>"))
    all_cls = sorted(set(list(cc_counts) + list(cp_counts)))
    class_df = pd.DataFrame({
        "class": all_cls,
        "clean": [cc_counts.get(k, 0) for k in all_cls],
        "patched": [cp_counts.get(k, 0) for k in all_cls],
    })

    # Histograms
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].hist(df_seen["conf_clean"], bins=30, alpha=0.6, label="clean", color="green")
    axes[0].hist(df_seen["conf_patched"], bins=30, alpha=0.6, label="patched", color="red")
    axes[0].set_xlabel("YOLO confidence on leader")
    axes[0].set_ylabel("frames")
    axes[0].set_title(f"Conf distribution (excl. {n_blind} clean-blind)")
    axes[0].legend()
    axes[1].scatter(df_seen["conf_clean"], df_seen["conf_patched"], s=8, alpha=0.5)
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[1].set_xlabel("conf clean")
    axes[1].set_ylabel("conf patched")
    axes[1].set_title("per-frame paired confidence")
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(args.out_dir / "histograms.png", bbox_inches="tight", dpi=100)
    plt.close(fig)

    # Side-by-side examples
    df_sorted = df_seen.copy()
    df_sorted["drop"] = df_sorted["conf_clean"] - df_sorted["conf_patched"]
    df_sorted = df_sorted.sort_values("drop", ascending=False)
    examples = list(df_sorted.head(args.n_examples - 2)["stem"]) + list(df_sorted.tail(2)["stem"])
    fig, axes = plt.subplots(len(examples), 2, figsize=(16, 4 * len(examples)))
    for row, stem in enumerate(examples):
        clean_bgr = cv2.imread(str(args.clean_dir / f"{stem}.png"))
        marker_bgr = cv2.imread(str(args.marker_dir / f"{stem}.png"))
        corners = np.asarray(QUADS[stem]["corners"], np.float32)
        tbox = target_bbox(corners, clean_bgr.shape, expand=args.expand)
        patched_bgr = warp_patch_onto(marker_bgr, PATCH, corners)
        bc, cc_, kc_ = yolo_leader(clean_bgr, tbox, yolo)
        bp, cp_, kp_ = yolo_leader(patched_bgr, tbox, yolo)
        ic = draw_box(clean_bgr, bc, cc_, kc_ or "none", (0, 220, 0))
        ip = draw_box(patched_bgr, bp, cp_, kp_ or "HIDDEN", (0, 0, 255))
        axes[row, 0].imshow(cv2.cvtColor(ic, cv2.COLOR_BGR2RGB))
        axes[row, 0].set_title(f"{stem} CLEAN conf={cc_:.3f}", fontsize=10)
        axes[row, 0].axis("off")
        axes[row, 1].imshow(cv2.cvtColor(ip, cv2.COLOR_BGR2RGB))
        title = f"{stem} PATCH conf={cp_:.3f}" if cp_ > 0 else f"{stem} PATCH HIDDEN"
        axes[row, 1].set_title(title, fontsize=10)
        axes[row, 1].axis("off")
    plt.tight_layout()
    fig.savefig(args.out_dir / "side_by_side.png", bbox_inches="tight", dpi=100)
    plt.close(fig)

    # HTML report
    metrics_html = f"""
    <table border="1" cellpadding="6" cellspacing="0">
      <tr><td>val frames total</td><td>{n_total}</td></tr>
      <tr><td>leader visible in clean</td><td>{n_seen} ({100*n_seen/n_total:.1f}%)</td></tr>
      <tr><td>already blind in clean (excluded)</td><td>{n_blind} ({100*n_blind/n_total:.1f}%)</td></tr>
      <tr><th colspan=2>— on visible-in-clean only —</th></tr>
      <tr><td>mean conf clean</td><td>{mc:.4f}</td></tr>
      <tr><td>mean conf with patch</td><td>{mp_:.4f}</td></tr>
      <tr><td>absolute delta</td><td>{delta:+.4f}</td></tr>
      <tr><td>relative delta</td><td>{delta_pct:+.2f}%</td></tr>
      <tr><td>detections lost (visible→hidden)</td><td>{n_hidden}/{n_seen} ({100*n_hidden/max(1,n_seen):.1f}%)</td></tr>
      <tr><th colspan=2>— including blind frames —</th></tr>
      <tr><td>det rate clean (all)</td><td>{100*df['leader_visible_clean'].mean():.1f}%</td></tr>
      <tr><td>det rate patched (all)</td><td>{100*df['leader_visible_patched'].mean():.1f}%</td></tr>
    </table>
    """
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
    <title>Patch analysis: {args.patch.name}</title>
    <style>body{{font-family:sans-serif;max-width:1400px;margin:20px auto;padding:0 20px}}
    table{{border-collapse:collapse;margin-top:10px}}th,td{{padding:6px 12px}}
    img{{max-width:100%;height:auto;display:block;margin:10px 0}}</style>
    </head><body>
    <h1>YOLO adversarial patch — analysis</h1>
    <p><b>Patch:</b> {args.patch}<br>
       <b>Marker dataset:</b> {args.marker_dir}<br>
       <b>Clean dataset (true baseline):</b> {args.clean_dir}<br>
       <b>YOLO weights:</b> {args.yolo_weights}<br>
       <b>Target bbox expand:</b> {args.expand}</p>
    <h2>Metrics</h2>{metrics_html}
    <h2>Class confusion (visible-in-clean only)</h2>{class_df.to_html(index=False)}
    <h2>Histograms</h2><img src='histograms.png'>
    <h2>Side-by-side examples (top {args.n_examples-2} drops + 2 failures)</h2><img src='side_by_side.png'>
    </body></html>"""
    (args.out_dir / "report.html").write_text(html)
    print(f"\nSaved report -> {args.out_dir}/report.html")


if __name__ == "__main__":
    main()
