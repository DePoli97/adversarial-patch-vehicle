"""Extract the yellow chroma-key quadrilateral from a CARLA frame.

Given a frame where the leader vehicle wears a solid yellow marker (see
`build_yellow_tga.py`), this script:
  1. Masks the yellow pixels in HSV space.
  2. Finds the largest contour and approximates a 4-corner polygon.
  3. Optionally overlays an arbitrary RGB image (e.g. the trained patch)
     onto the quad via perspective warp.

Outputs per input frame:
    <name>_mask.png       binary yellow mask (debug)
    <name>_quad.json      {"corners": [[x,y], x4]}
    <name>_quad.png       original frame with detected quad drawn in red
    <name>_patched.png    (if --patch given) frame with patch warped onto quad

Usage:
    python src/chroma_key_dataset_generator/extract_quad.py \\
        --image <path-to-frame.png> \\
        [--patch assets/carla_rear_window/rear_window_raw.TGA] \\
        [--out-dir experiments/chroma_key_demo/]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# HSV bounds for "saturated yellow" — tuned to RGB(255, 220, 0). Generous on V
# (brightness) to survive shading/distance; tight on H/S to reject white headlights.
HSV_LOW = np.array([20, 120, 100], dtype=np.uint8)
HSV_HIGH = np.array([35, 255, 255], dtype=np.uint8)


def find_yellow_quad(bgr: np.ndarray):
    """Return (corners[4,2] in pixel coords) or None if no quad found."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)
    # Clean up speckle
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 200:  # too small, probably noise
        return None, mask

    # Approximate to a polygon; relax epsilon until we get ~4 corners
    for eps in (0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(largest, eps * cv2.arcLength(largest, True), True)
        if len(approx) == 4:
            return approx.reshape(4, 2), mask
    # Fallback: use minAreaRect (oriented bbox), gives 4 ordered corners
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect).astype(np.int32)
    return box, mask


def order_corners(corners: np.ndarray) -> np.ndarray:
    """Order 4 corners as [top-left, top-right, bottom-right, bottom-left]."""
    c = corners.astype(np.float32)
    s = c.sum(axis=1)
    d = np.diff(c, axis=1).reshape(-1)
    tl = c[np.argmin(s)]
    br = c[np.argmax(s)]
    tr = c[np.argmin(d)]
    bl = c[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def warp_patch_onto_quad(frame: np.ndarray, patch_bgr: np.ndarray,
                         corners: np.ndarray) -> np.ndarray:
    """Perspective-warp the patch onto the quad in the frame, return overlaid copy."""
    H, W = patch_bgr.shape[:2]
    src = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    dst = order_corners(corners)
    M = cv2.getPerspectiveTransform(src, dst)
    Hf, Wf = frame.shape[:2]
    warped = cv2.warpPerspective(patch_bgr, M, (Wf, Hf))
    # Build mask of warped patch region
    patch_mask = np.ones((H, W), dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(patch_mask, M, (Wf, Hf))
    out = frame.copy()
    out[warped_mask > 0] = warped[warped_mask > 0]
    return out


def load_patch_rgba_as_bgr(path: Path) -> np.ndarray:
    """Load TGA/PNG patch as BGR, dropping alpha if present."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"Could not load patch: {path}")
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def process(image_path: Path, patch_path: Path | None, out_dir: Path):
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise SystemExit(f"Could not load image: {image_path}")
    corners, mask = find_yellow_quad(bgr)

    stem = image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{stem}_mask.png"), mask)

    if corners is None:
        print(f"[!] {image_path.name}: no yellow quad detected")
        return

    # Save corners JSON
    ordered = order_corners(corners)
    with open(out_dir / f"{stem}_quad.json", "w") as f:
        json.dump({"corners": ordered.tolist(),
                   "image": str(image_path),
                   "shape": list(bgr.shape)}, f, indent=2)

    # Debug overlay
    dbg = bgr.copy()
    pts = ordered.astype(np.int32)
    cv2.polylines(dbg, [pts], isClosed=True, color=(0, 0, 255), thickness=3)
    for i, (x, y) in enumerate(ordered.astype(int)):
        cv2.circle(dbg, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(dbg, str(i), (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(str(out_dir / f"{stem}_quad.png"), dbg)

    print(f"[OK] {image_path.name}: corners {ordered.astype(int).tolist()}")

    if patch_path is not None:
        patch_bgr = load_patch_rgba_as_bgr(patch_path)
        patched = warp_patch_onto_quad(bgr, patch_bgr, ordered)
        cv2.imwrite(str(out_dir / f"{stem}_patched.png"), patched)
        print(f"     patched -> {stem}_patched.png")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", type=Path, required=True,
                   help="Input frame (jpg/png). Can be a single file or a glob pattern.")
    p.add_argument("--patch", type=Path, default=None,
                   help="Optional patch image (TGA/PNG) to warp onto the detected quad.")
    p.add_argument("--out-dir", type=Path, default=Path("experiments/chroma_key_demo"),
                   help="Where to write debug outputs.")
    args = p.parse_args()

    # Support glob via shell, so --image accepts either a file or whatever the shell expands
    images = [args.image] if args.image.is_file() else sorted(args.image.parent.glob(args.image.name))
    if not images:
        raise SystemExit(f"No images found at {args.image}")

    print(f"Processing {len(images)} image(s) -> {args.out_dir}\n")
    for img in images:
        process(img, args.patch, args.out_dir)


if __name__ == "__main__":
    main()
