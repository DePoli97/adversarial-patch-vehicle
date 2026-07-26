"""Author CarlaCola patch textures directly, without an Unreal Engine cook.

Background
----------
Deploying a trained patch used to require importing the texture into the Unreal Editor
(as UserInterface2D RGBA, sRGB off) and running ``make package`` to cook a new ``.ubulk``
-- roughly 25 minutes per patch, and awkward to drive headless.

That step turns out to be unnecessary. The CarlaCola bodywork ``.ubulk`` is a *raw
uncompressed RGBA mip chain* with no headers: 4096^2, 2048^2, 1024^2, 512^2, 256^2,
128^2 concatenated largest-first, which is exactly 89,456,640 bytes. The adversarial
patch occupies a fixed rectangle at ``(2885, 2400)`` of size ``409x204`` in the level-0
mip (the 2:1 footprint of the 512x256 patch scaled by 0.8), and every cooked patch file
is byte-identical to every other one outside that rectangle.

So a new patch texture can be authored by copying an existing cooked file and
overwriting just that rectangle at each mip level. Verified by round-trip: rebuilding
``123_carlacola_pooled.ubulk`` from ``123_carlacola_generalist.ubulk`` plus the pooled
rectangle reproduces the target file byte-for-byte (100.00000% identical).

Choose the base carefully
-------------------------
Use an existing *cooked patch* as the base rather than ``123_CarlaCola_clean.ubulk.ORIG2``.
The clean backup was cooked separately and its mips 2/4/5 differ from the patch files by
up to 11/255 in regions unrelated to the patch. Harmless visually, but using a cooked
patch as the base keeps the result exact.

Mip fidelity
------------
Level 0 is written exactly. Lower mips are produced by box-downsampling the full patched
level-0 image and taking the corresponding rectangle, so edge pixels correctly blend with
the surrounding bodywork. Unreal's own mip filter appears to apply additional sharpening
(reproducing its output exactly was not possible with box/bilinear/Lanczos, mean error
~12/255), but only level 0 matters at the ranges this experiment measures, and the lower
mips remain smooth and artefact-free.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image

# Raw RGBA mip chain, largest first. Sums to exactly 89,456,640 bytes.
MIP_SIZES = (4096, 2048, 1024, 512, 256, 128)
UBULK_BYTES = sum(s * s * 4 for s in MIP_SIZES)

# Patch rectangle in the level-0 mip: x, y, width, height.
# Recovered by diffing clean against cooked patch textures; matches the documented
# overlay offset (x=2885, y=2400) at scale 0.8 of a 512x256 patch.
PATCH_RECT_L0 = (2885, 2400, 409, 204)


def rect_for_level(level: int) -> tuple[int, int, int, int]:
    """Patch rectangle at the given mip level.

    The origin halves by floor division and the size by ceiling division, which is what
    reproduces the rectangles observed in the cooked files (409 -> 205, not 204).
    """
    x, y, w, h = PATCH_RECT_L0
    d = 2**level
    return x // d, y // d, max(1, -(-w // d)), max(1, -(-h // d))


def load_mips(path: str) -> list[np.ndarray]:
    """Parse a .ubulk into its list of (size, size, 4) uint8 mip levels."""
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size != UBULK_BYTES:
        raise ValueError(
            f"{path}: expected {UBULK_BYTES} bytes for a 4096^2 RGBA mip chain, "
            f"got {raw.size}"
        )
    mips, offset = [], 0
    for size in MIP_SIZES:
        n = size * size * 4
        mips.append(raw[offset : offset + n].reshape(size, size, 4).copy())
        offset += n
    return mips


def save_mips(mips: list[np.ndarray], path: str) -> None:
    """Write mip levels back out as a .ubulk, checking the size is unchanged."""
    blob = b"".join(m.tobytes() for m in mips)
    if len(blob) != UBULK_BYTES:
        raise ValueError(f"refusing to write {len(blob)} bytes, expected {UBULK_BYTES}")
    with open(path, "wb") as fh:
        fh.write(blob)


def _box_halve(img: np.ndarray) -> np.ndarray:
    """Downsample by 2 with a box filter, in float to avoid rounding drift."""
    h, w = img.shape[0] // 2, img.shape[1] // 2
    f = img.astype(np.float32).reshape(h, 2, w, 2, img.shape[2]).mean(axis=(1, 3))
    return np.clip(np.rint(f), 0, 255).astype(np.uint8)


def load_patch_rgba(path: str) -> np.ndarray:
    """Load a patch image (PNG/TGA/...) as (H, W, 4) uint8."""
    im = Image.open(path).convert("RGBA")
    return np.asarray(im)


def author(base_ubulk: str, patch: np.ndarray, out_ubulk: str) -> dict:
    """Write a new .ubulk: `base_ubulk` with `patch` placed in the patch rectangle.

    `patch` is (H, W, 3 or 4) uint8; it is resized to the rectangle at each mip level.
    The alpha channel of the base is preserved -- the bodywork material relies on it and
    the patch itself is fully opaque.
    """
    mips = load_mips(base_ubulk)
    if patch.ndim != 3 or patch.shape[2] not in (3, 4):
        raise ValueError(f"patch must be HxWx3 or HxWx4, got {patch.shape}")

    x, y, w, h = rect_for_level(0)
    resized = np.asarray(
        Image.fromarray(patch[:, :, :3]).resize((w, h), Image.LANCZOS)
    )
    mips[0][y : y + h, x : x + w, :3] = resized

    # Lower mips: downsample the whole patched level-0 image so that pixels on the
    # rectangle's edge blend with the surrounding bodywork exactly as a real mip would,
    # then transplant only the rectangle.
    full = mips[0]
    for level in range(1, len(MIP_SIZES)):
        full = _box_halve(full)
        lx, ly, lw, lh = rect_for_level(level)
        mips[level][ly : ly + lh, lx : lx + lw, :3] = full[
            ly : ly + lh, lx : lx + lw, :3
        ]

    save_mips(mips, out_ubulk)
    return {
        "base": base_ubulk,
        "out": out_ubulk,
        "bytes": UBULK_BYTES,
        "rect_l0": (x, y, w, h),
        "patch_src_shape": tuple(patch.shape),
    }


def extract(ubulk: str, out_png: str, level: int = 0) -> np.ndarray:
    """Dump the patch rectangle of a .ubulk to a PNG, for visual inspection."""
    mips = load_mips(ubulk)
    x, y, w, h = rect_for_level(level)
    crop = mips[level][y : y + h, x : x + w]
    Image.fromarray(crop[:, :, :3]).save(out_png)
    return crop


def self_test(grid_dir: str) -> bool:
    """Prove the authoring model is exact.

    Rebuild one cooked patch from another by transplanting only the patch rectangle at
    every mip level, and require a byte-for-byte match with the real file.
    """
    a = os.path.join(grid_dir, "123_carlacola_generalist.ubulk")
    b = os.path.join(grid_dir, "123_carlacola_pooled.ubulk")
    base, target = load_mips(a), load_mips(b)
    rebuilt = [m.copy() for m in base]
    for level in range(len(MIP_SIZES)):
        x, y, w, h = rect_for_level(level)
        rebuilt[level][y : y + h, x : x + w] = target[level][y : y + h, x : x + w]
    ok = all(np.array_equal(r, t) for r, t in zip(rebuilt, target))
    print(f"self-test rebuild(generalist + pooled_rect) == pooled : "
          f"{'PASS (byte-identical)' if ok else 'FAIL'}")
    for level in range(len(MIP_SIZES)):
        diff = int((rebuilt[level] != target[level]).sum())
        print(f"  mip{level} ({MIP_SIZES[level]:4d}^2): differing bytes = {diff}")
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("author", help="write a new .ubulk from a patch image")
    pa.add_argument("--base", required=True, help="existing cooked patch .ubulk")
    pa.add_argument("--patch", required=True, help="patch image (PNG/TGA)")
    pa.add_argument("--out", required=True)

    pe = sub.add_parser("extract", help="dump a .ubulk's patch rectangle to PNG")
    pe.add_argument("--ubulk", required=True)
    pe.add_argument("--out", required=True)
    pe.add_argument("--level", type=int, default=0)

    ps = sub.add_parser("self-test", help="prove the authoring model is byte-exact")
    ps.add_argument("--grid-dir", required=True)

    args = p.parse_args()
    if args.cmd == "author":
        info = author(args.base, load_patch_rgba(args.patch), args.out)
        print(f"wrote {info['out']} ({info['bytes']} bytes), rect {info['rect_l0']}")
    elif args.cmd == "extract":
        crop = extract(args.ubulk, args.out, args.level)
        print(f"wrote {args.out} shape={crop.shape}")
    else:
        return 0 if self_test(args.grid_dir) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
