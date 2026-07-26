"""Derive the patch quad per frame by differencing a clean capture against a
patched one, and emit a `quads_index.json` that `ChromaKeyDataset` can read.

The Fase 1 pipeline painted a saturated yellow marker on the truck and found it
with an HSV threshold (`chroma_key_dataset_generator/extract_quad.py`). That
needs a dedicated marker texture and a colour that nothing else in the scene
wears — fragile, and it burns one whole capture pass on a texture that is never
deployed. Here the region is recovered instead from the only thing that changed
between two otherwise bit-identical captures: the truck's own texture. Whatever
texels the deployed patch covers light up in the absolute difference, so the
quad is by construction exactly where the real patch will appear.

This only works if `capture_tfv6.py` did its job: same seed, same poses, same
weather, physics disabled, and nothing but the CarlaCola .ubulk swapped between
the two runs. Any drift shows up as diff along every silhouette edge in the
scene, which the connected-component + shape filters will mostly reject — and
the reported statistics will look obviously wrong (huge areas, wandering
camera indices).

Pipeline per frame:
  1. abs-diff clean vs patched, max over channels
  2. median blur, threshold, morphological open + close
  3. connected components; keep the largest
  4. clip the mask to the 384-wide camera slice holding that component, so the
     quad lives inside ONE pinhole view and a single homography is valid
     (a quad straddling two cameras of the surround rig is not a perspective
     projection of a plane and would warp the patch incorrectly)
  5. fit a 4-corner polygon (approxPolyDP eps sweep, minAreaRect fallback)
  6. order TL/TR/BR/BL with extract_quad.order_corners — the same convention
     patch_render.patch_canonical_corners expects
  7. reject degenerate quads: too small, non-convex, ribbon-thin, or poorly
     filled by the diff mask

Output schema (identical to what `extract_quad.py --batch-index` writes and
what `yolo_chroma_attack/dataset.py` reads):

    {"<stem>": {"corners": [[x, y] x4], "shape": [H, W, 3], "area": float, ...}}

`corners` are in FULL COMPOSITE pixel coordinates, so the trainer warps the
patch straight onto the (384, 2304) image. The extra keys (`camera_idx`,
`fill_ratio`, `diff_area`, `other_components`) are ignored by ChromaKeyDataset
and are there for diagnostics.

By default the index is written next to the CLEAN frames: the patch should be
rendered over an untouched truck, not over a truck that already wears a
deployment texture whose edges would bleed around the warped quad.

Usage:
    python src/tfv6_chroma_attack/build_quads.py \\
        --clean-dir   data/chroma_key_dataset/tfv6/Town04_spawn273_day/clean \\
        --patched-dir data/chroma_key_dataset/tfv6/Town04_spawn273_day/patched \\
        --debug-dir   experiments/tfv6_attack/quad_debug/Town04_day
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.chroma_key_dataset_generator.extract_quad import (  # noqa: E402
        _is_convex_quad, order_corners,
    )
except ImportError:  # running with a different cwd / no namespace package
    sys.path.insert(0, str(ROOT / "src" / "chroma_key_dataset_generator"))
    from extract_quad import _is_convex_quad, order_corners  # noqa: E402


def diff_mask(clean: np.ndarray, patched: np.ndarray, thresh: int,
              blur: int, open_k: int, close_k: int) -> np.ndarray:
    """Binary mask of pixels the texture swap changed.

    Max over the three channels rather than a grayscale difference: a patch can
    be a pure hue shift at constant luminance, which grayscale would erase.
    """
    d = cv2.absdiff(clean, patched).max(axis=2)
    if blur >= 3 and blur % 2 == 1:
        d = cv2.medianBlur(d, blur)
    mask = (d >= thresh).astype(np.uint8) * 255
    if open_k > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((open_k, open_k), np.uint8))
    if close_k > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((close_k, close_k), np.uint8))
    return mask


def fit_quad(contour: np.ndarray) -> np.ndarray:
    """4-corner polygon for `contour`, TL/TR/BR/BL ordered.

    approxPolyDP with a widening epsilon first (it lands on the true corners of
    a rasterised quad to within a pixel); minAreaRect only as a fallback, which
    is axis-aligned-in-rotation and therefore loses perspective foreshortening.
    """
    for eps in (0.01, 0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(contour, eps * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            return order_corners(approx.reshape(4, 2).astype(np.float32))
    rect = cv2.minAreaRect(contour)
    return order_corners(cv2.boxPoints(rect).astype(np.float32))


def quad_side_ratio(quad: np.ndarray) -> float:
    """short/long of the oriented bounding box — 0 for a ribbon, 1 for a square."""
    (_, _), (w, h), _ = cv2.minAreaRect(quad.astype(np.float32))
    long_ = max(w, h)
    if long_ < 1e-6:
        return 0.0
    return float(min(w, h) / long_)


def camera_of(x: float, slices: dict[int, list[int]]) -> int | None:
    for idx, (x0, x1) in slices.items():
        if x0 <= x < x1:
            return idx
    return None


def load_slices(clean_dir: Path, num_cameras_fallback: int,
                width: int) -> tuple[dict[int, list[int]], int | None]:
    """Camera slices from the capture's calibration dump, or an even split."""
    calib_path = clean_dir / "camera_calibration.json"
    if calib_path.exists():
        with open(calib_path) as f:
            c = json.load(f)
        slices = {int(k): v for k, v in c["camera_slices"].items()}
        return slices, c.get("front_camera_index")
    n = num_cameras_fallback
    w = width // n
    return {i + 1: [i * w, (i + 1) * w] for i in range(n)}, None


def process_pair(clean_path: Path, patched_path: Path, args,
                 slices: dict[int, list[int]], debug_dir: Path | None):
    """Return (entry_dict, reason_or_None)."""
    clean = cv2.imread(str(clean_path))
    patched = cv2.imread(str(patched_path))
    if clean is None or patched is None:
        return None, "unreadable"
    if clean.shape != patched.shape:
        return None, f"shape mismatch {clean.shape} vs {patched.shape}"

    mask = diff_mask(clean, patched, args.diff_thresh, args.blur,
                     args.open_kernel, args.close_kernel)
    n_lab, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if n_lab <= 1:
        return None, "no difference"

    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(areas)[::-1]
    best = int(order[0]) + 1
    best_area = int(stats[best, cv2.CC_STAT_AREA])
    if best_area < args.min_area:
        return None, f"largest component too small ({best_area} px)"

    cam_idx = camera_of(float(centroids[best][0]), slices)
    if cam_idx is None:
        return None, "component centroid outside every camera slice"
    x0, x1 = slices[cam_idx]

    # Restrict to the owning camera: a quad spanning a seam between two views of
    # the surround rig is not a single planar perspective and would warp wrong.
    comp = ((labels == best).astype(np.uint8)) * 255
    clipped = np.zeros_like(comp)
    clipped[:, x0:x1] = comp[:, x0:x1]
    clipped_area = int((clipped > 0).sum())
    if clipped_area < args.min_area:
        return None, f"component too small inside its camera ({clipped_area} px)"

    contours, _ = cv2.findContours(clipped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "no contour"
    contour = max(contours, key=cv2.contourArea)

    quad = fit_quad(contour)
    quad_area = float(cv2.contourArea(quad))
    if quad_area < args.min_area:
        return None, f"quad too small ({quad_area:.0f} px)"
    if not _is_convex_quad(quad):
        return None, "quad not convex"
    ratio = quad_side_ratio(quad)
    if ratio < args.min_side_ratio:
        return None, f"quad too thin (side ratio {ratio:.3f})"
    fill = clipped_area / quad_area
    if fill < args.min_fill:
        return None, f"quad poorly filled by the diff (fill {fill:.2f})"

    # Everything else that changed: a second camera catching the same patch at a
    # glancing angle is expected; anything large elsewhere means the two passes
    # were not identical and the capture should be redone.
    others = []
    for lab in order[1:]:
        lab = int(lab) + 1
        a = int(stats[lab, cv2.CC_STAT_AREA])
        if a < args.min_area:
            break
        others.append({
            "area": a,
            "camera_idx": camera_of(float(centroids[lab][0]), slices),
            "centroid": [float(centroids[lab][0]), float(centroids[lab][1])],
        })

    entry = {
        "corners": quad.tolist(),
        "shape": list(clean.shape),
        "area": quad_area,
        "camera_idx": cam_idx,
        "camera_slice": [x0, x1],
        "diff_area": clipped_area,
        "fill_ratio": float(fill),
        "side_ratio": float(ratio),
        "other_components": others,
        "source": "build_quads.py (clean/patched differencing)",
    }

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        stem = clean_path.stem
        cv2.imwrite(str(debug_dir / f"{stem}_mask.png"), mask)
        dbg = clean[:, x0:x1].copy()
        shifted = (quad - np.array([[x0, 0.0]], dtype=np.float32)).astype(np.int32)
        cv2.polylines(dbg, [shifted], True, (0, 0, 255), 2)
        for i, (px, py) in enumerate(shifted):
            cv2.circle(dbg, (int(px), int(py)), 4, (0, 255, 0), -1)
            cv2.putText(dbg, str(i), (int(px) + 6, int(py) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(str(debug_dir / f"{stem}_quad_cam{cam_idx}.png"), dbg)

    return entry, None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clean-dir", type=Path, required=True)
    p.add_argument("--patched-dir", type=Path, required=True)
    p.add_argument("--reuse-index", type=Path, default=None,
                   help="Adopt this quads_index.json instead of differencing. "
                        "Only legal when the two captures share a pose list; "
                        "that is verified against the sidecars and the run aborts "
                        "on any mismatch. Intended for night cells, where the "
                        "diff is too faint but the geometry is identical.")
    p.add_argument("--out", type=Path, default=None,
                   help="index path; defaults to <clean-dir>/quads_index.json")
    p.add_argument("--debug-dir", type=Path, default=None,
                   help="write per-frame mask + quad overlay here")

    p.add_argument("--diff-thresh", type=int, default=25,
                   help="per-channel abs-diff level counted as 'changed'")
    p.add_argument("--blur", type=int, default=3,
                   help="median blur kernel on the diff (odd, 0 disables)")
    p.add_argument("--open-kernel", type=int, default=3)
    p.add_argument("--close-kernel", type=int, default=7)

    p.add_argument("--min-area", type=float, default=400.0)
    p.add_argument("--min-side-ratio", type=float, default=0.15,
                   help="reject ribbon-thin quads (matches ChromaKeyDataset's own filter)")
    p.add_argument("--min-fill", type=float, default=0.5,
                   help="diff pixels / quad area; rejects scattered noise fitted "
                        "by a big empty quad")
    p.add_argument("--num-cameras", type=int, default=6,
                   help="only used if the capture's camera_calibration.json is missing")
    p.add_argument("--enrich-sidecars", action="store_true",
                   help="also write the quad into each clean/<stem>.json, "
                        "preserving the capture metadata (mirrors extract_quad.py)")
    return p.parse_args()


def reuse_index(clean_dir: Path, source_index: Path, out_path: Path,
                tol_m: float = 1e-6) -> None:
    """Adopt another capture's quads after proving the two share the same poses.

    A night capture of the same cell is photometrically much weaker: the truck's
    rear panel is lit only by a twilight sky, so the clean-vs-patched difference
    barely clears the threshold and whole cells come back with almost no quads
    (Town11 night: 0 of 54). The geometry, however, is not in question. The pose
    list is precomputed from `--seed` before CARLA is touched, and physics is
    disabled, so the day and night runs of a cell place the actors at bitwise
    identical transforms — measured max |difference| across all 54 frames of all
    three towns: 0.000000000.

    The quad is a function of that geometry alone, so the day index is exact for
    the night frames. This is a reuse of a measurement, not an approximation —
    but it is only valid while the poses match, so verify rather than assume.
    """
    src = json.loads(source_index.read_text())
    src_dir = source_index.parent
    checked = 0
    for stem in src:
        a_path, b_path = src_dir / f"{stem}.json", clean_dir / f"{stem}.json"
        if not (a_path.exists() and b_path.exists()):
            raise SystemExit(
                f"cannot verify pose identity for {stem}: missing sidecar. "
                "Refusing to reuse an index that might not describe these frames."
            )
        a, b = json.loads(a_path.read_text()), json.loads(b_path.read_text())
        for key in ("ego_loc", "leader_loc"):
            for u, v in zip(a[key], b[key]):
                if abs(u - v) > tol_m:
                    raise SystemExit(
                        f"{stem}: {key} differs by {abs(u - v):.6g} m between "
                        f"{src_dir} and {clean_dir}. The captures do not share a "
                        "pose list, so their quads are not interchangeable."
                    )
        if abs(a["ego_yaw_deg"] - b["ego_yaw_deg"]) > 1e-4:
            raise SystemExit(f"{stem}: ego yaw differs between the two captures")
        checked += 1

    kept = {s: e for s, e in src.items() if (clean_dir / f"{s}.png").exists()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(kept))
    print(f"[INFO] pose identity verified on {checked} frames "
          f"(max tolerance {tol_m} m)")
    print(f"[INFO] reused {len(kept)} quads from {source_index} -> {out_path}")


def main() -> None:
    args = parse_args()
    clean_dir: Path = args.clean_dir
    patched_dir: Path = args.patched_dir
    out_path: Path = args.out or (clean_dir / "quads_index.json")

    if args.reuse_index is not None:
        reuse_index(clean_dir, args.reuse_index, out_path)
        return

    clean_frames = sorted(p for p in clean_dir.glob("*.png"))
    if not clean_frames:
        raise SystemExit(f"no PNG frames in {clean_dir}")

    probe = cv2.imread(str(clean_frames[0]))
    if probe is None:
        raise SystemExit(f"cannot read {clean_frames[0]}")
    slices, front_idx = load_slices(clean_dir, args.num_cameras, probe.shape[1])
    print(f"[INFO] {len(clean_frames)} clean frames, composite {probe.shape}")
    print(f"[INFO] camera slices: {slices}"
          + (f"  (front = {front_idx})" if front_idx else ""))

    index: dict[str, dict] = {}
    rejected: dict[str, str] = {}
    missing = 0
    for i, cpath in enumerate(clean_frames):
        ppath = patched_dir / cpath.name
        if not ppath.exists():
            missing += 1
            continue
        entry, reason = process_pair(cpath, ppath, args, slices, args.debug_dir)
        if entry is None:
            rejected[cpath.stem] = reason
            continue
        index[cpath.stem] = entry

        if args.enrich_sidecars:
            sidecar = cpath.with_suffix(".json")
            meta = {}
            if sidecar.exists():
                try:
                    with open(sidecar) as f:
                        meta = json.load(f)
                except Exception:
                    meta = {}
            meta["detected_corners"] = entry
            with open(sidecar, "w") as f:
                json.dump(meta, f, indent=2)

        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{len(clean_frames)}  ok={len(index)} "
                  f"rejected={len(rejected)}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(index, f)

    print(f"\n[INFO] ok={len(index)}  rejected={len(rejected)}  "
          f"missing_patched={missing}  -> {out_path}")

    if index:
        areas = np.array([e["area"] for e in index.values()])
        fills = np.array([e["fill_ratio"] for e in index.values()])
        cams = [e["camera_idx"] for e in index.values()]
        n_multi = sum(1 for e in index.values() if e["other_components"])
        print(f"[INFO] quad area   min/median/max = "
              f"{areas.min():.0f} / {np.median(areas):.0f} / {areas.max():.0f} px")
        print(f"[INFO] fill ratio  min/median     = {fills.min():.2f} / "
              f"{np.median(fills):.2f}")
        print(f"[INFO] camera histogram: "
              f"{ {c: cams.count(c) for c in sorted(set(cams))} }")
        if n_multi:
            print(f"[WARN] {n_multi} frames have extra changed components — a few "
                  f"are the patch seen by a neighbouring camera, many mean the two "
                  f"passes were not identical (check poses/weather/physics)")
    if rejected:
        by_reason: dict[str, int] = {}
        for r in rejected.values():
            key = r.split("(")[0].strip()
            by_reason[key] = by_reason.get(key, 0) + 1
        print(f"[INFO] rejection reasons: {by_reason}")


if __name__ == "__main__":
    main()
