"""Overlay one image on top of another (TGA/PNG) at given pixel coordinates.

Lossless: output size, bit depth, and channels match the base image. Alpha is
respected — fully transparent pixels of the overlay don't affect the base.

Remembers the last coordinates used in a `.overlay_state.json` file next to
the script, so you can re-run without retyping them when iterating.

Usage:
    # First run: provide everything
    python overlay_on_tga.py \\
        --base /path/to/truck_texture.TGA \\
        --overlay /path/to/yellow_marker.PNG \\
        --x 512 --y 1024 \\
        --output /path/to/truck_texture_with_yellow.TGA

    # Re-run with last-used coords (x, y come from state file):
    python overlay_on_tga.py --base ... --overlay ... --output ...

    # Override only one coordinate:
    python overlay_on_tga.py --base ... --overlay ... --output ... --x 300

Optional:
    --scale 0.5         scale overlay before pasting (e.g. 0.5 = half)
    --rotate 90         rotate overlay by N degrees (counterclockwise)
    --no-state          ignore state file (don't load or save)
    --anchor center     interpret (x, y) as the overlay's center instead of top-left
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
STATE_FILE = SCRIPT_DIR / ".overlay_state.json"


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def overlay(base_path: Path, overlay_path: Path, out_path: Path,
            x: int, y: int, anchor: str,
            scale: float, rotate: float) -> tuple[int, int]:
    base = Image.open(base_path)
    over = Image.open(overlay_path)

    # Preserve base mode for lossless write later
    base_mode = base.mode
    if base_mode != "RGBA":
        base = base.convert("RGBA")
    if over.mode != "RGBA":
        over = over.convert("RGBA")

    if scale != 1.0:
        new_size = (max(1, int(over.width * scale)), max(1, int(over.height * scale)))
        over = over.resize(new_size, Image.Resampling.LANCZOS)
    if rotate != 0:
        over = over.rotate(rotate, expand=True, resample=Image.Resampling.BICUBIC)

    if anchor == "center":
        paste_x = x - over.width // 2
        paste_y = y - over.height // 2
    else:  # top-left
        paste_x = x
        paste_y = y

    result = base.copy()
    # 3-arg form uses the overlay's alpha channel as mask — proper alpha compositing
    result.alpha_composite(over, dest=(paste_x, paste_y))

    # Restore the base's original mode (so the saved TGA stays compatible)
    if base_mode != "RGBA":
        result = result.convert(base_mode)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Use the original file's format inferred by extension
    result.save(out_path)
    return paste_x, paste_y


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--base", type=Path, required=True,
                   help="Path to base image (TGA/PNG).")
    p.add_argument("--overlay", type=Path, required=True,
                   help="Path to overlay image (TGA/PNG with transparency).")
    p.add_argument("--output", type=Path, required=True,
                   help="Path to write the composited image.")
    p.add_argument("--x", type=int, default=None,
                   help="X coordinate (pixels). If omitted, uses last value from state.")
    p.add_argument("--y", type=int, default=None,
                   help="Y coordinate (pixels). If omitted, uses last value from state.")
    p.add_argument("--anchor", choices=["top-left", "center"], default=None,
                   help="How (x, y) is interpreted. Default top-left, or last used.")
    p.add_argument("--scale", type=float, default=None,
                   help="Scale factor for the overlay (default 1.0 or last used).")
    p.add_argument("--rotate", type=float, default=None,
                   help="Rotate overlay by N degrees CCW (default 0 or last used).")
    p.add_argument("--no-state", action="store_true",
                   help="Don't read or write the state file.")
    args = p.parse_args()

    state = {} if args.no_state else load_state()

    # Merge args with state — args win, then state, then sensible defaults
    x = args.x if args.x is not None else state.get("x", 0)
    y = args.y if args.y is not None else state.get("y", 0)
    anchor = args.anchor or state.get("anchor", "top-left")
    scale = args.scale if args.scale is not None else state.get("scale", 1.0)
    rotate = args.rotate if args.rotate is not None else state.get("rotate", 0.0)

    if not args.base.exists():
        sys.exit(f"Base not found: {args.base}")
    if not args.overlay.exists():
        sys.exit(f"Overlay not found: {args.overlay}")
    # Refuse to overwrite the source images.
    out_resolved = args.output.resolve()
    if out_resolved == args.base.resolve():
        sys.exit(f"Refusing to overwrite the base image. Pick a different --output.")
    if out_resolved == args.overlay.resolve():
        sys.exit(f"Refusing to overwrite the overlay image. Pick a different --output.")

    print(f"Base    : {args.base}")
    print(f"Overlay : {args.overlay}")
    print(f"Output  : {args.output}")
    print(f"  x={x}, y={y}, anchor={anchor}, scale={scale}, rotate={rotate}")

    paste_x, paste_y = overlay(args.base, args.overlay, args.output,
                                x, y, anchor, scale, rotate)

    print(f"OK. Pasted at top-left ({paste_x}, {paste_y}).")

    if not args.no_state:
        save_state({
            "base": str(args.base),
            "overlay": str(args.overlay),
            "x": x, "y": y, "anchor": anchor,
            "scale": scale, "rotate": rotate,
        })
        print(f"State saved to {STATE_FILE}")


if __name__ == "__main__":
    main()
