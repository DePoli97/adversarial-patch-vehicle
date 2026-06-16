"""Build side-by-side clean|patch MP4s from a multi_agent_<ts> scenario sweep.

For each <agent>/<town>/ we pair the clean and patch run directories by
**alphabetical order** (NOT by timestamp — clean and patch were captured
sequentially, so timestamps don't match). Output goes under
    <agent>/<town>/videos/
        run_01.mp4 ... run_NN.mp4      # per-run, vertical split clean|patch
        overview.mp4                   # grid summary, max 4 columns, 16:9

Per-run video frame layout:
    +----------------+----------------+
    | CLEAN  seed=X  | PATCH  seed=Y  |
    |   (image)      |   (image)      |
    +----------------+----------------+

Overview frame layout: max 4 columns, as many rows as needed; each cell is one
mini side-by-side; trailing cells stay black. Total canvas is 1920x1080 (16:9).

Frame rate is set so the whole video is ~15 s regardless of how many ticks
were saved (typical: 30 images / run -> 2 fps).

Run on Vortex (raw images live there):
    python src/carla_scenario/make_comparison_videos.py \
        --root experiments/carla_scenarios/multi_agent_20260613_154355 \
        [--agents simlingo_simlingo tfv4_aim_0 tfv6_visiononly] \
        [--no-overview] [--no-per-run]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


PANEL_W, PANEL_H = 960, 1080          # each half of a per-run video frame
FRAME_W, FRAME_H = PANEL_W * 2, PANEL_H  # 1920x1080
OVERVIEW_W, OVERVIEW_H = 1920, 1080
TARGET_DURATION_S = 15.0
DEFAULT_FPS_FALLBACK = 2


def list_run_dirs(d: Path) -> list[Path]:
    if not d.exists():
        return []
    runs = sorted(p for p in d.iterdir() if p.is_dir() and (p / "images").exists())
    return runs


def list_image_files(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "images").glob("tick_*.jpg"))


def fit_into_panel(img: np.ndarray, panel_w: int, panel_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(panel_w / w, panel_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    y0 = (panel_h - new_h) // 2
    x0 = (panel_w - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def draw_label(panel: np.ndarray, text: str, color=(255, 255, 255)) -> np.ndarray:
    out = panel.copy()
    cv2.rectangle(out, (0, 0), (panel.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(out, text, (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)
    return out


def write_video_with_ffmpeg(frames_dir: Path, out_path: Path, fps: float, width: int, height: int):
    """Use ffmpeg to encode frames/*.png into an h264 mp4."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", f"{fps:.3f}",
        "-i", str(frames_dir / "%06d.png"),
        "-vf", f"scale={width}:{height}:flags=lanczos",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def make_per_run_video(clean_dir: Path, patch_dir: Path, out_path: Path, label_idx: int) -> bool:
    clean_imgs = list_image_files(clean_dir)
    patch_imgs = list_image_files(patch_dir)
    if not clean_imgs or not patch_imgs:
        print(f"  [skip] missing images: {clean_dir.name}  /  {patch_dir.name}")
        return False
    n = min(len(clean_imgs), len(patch_imgs))
    if n == 0:
        return False
    fps = max(1.0, n / TARGET_DURATION_S)
    seed_clean = clean_dir.name.split("_seed")[-1].split("_")[0]
    seed_patch = patch_dir.name.split("_seed")[-1].split("_")[0]

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for i in range(n):
            c = cv2.imread(str(clean_imgs[i]))
            p = cv2.imread(str(patch_imgs[i]))
            if c is None or p is None:
                continue
            c_panel = draw_label(fit_into_panel(c, PANEL_W, PANEL_H),
                                  f"CLEAN  seed={seed_clean}  tick={i+1:02d}/{n}")
            p_panel = draw_label(fit_into_panel(p, PANEL_W, PANEL_H),
                                  f"PATCH  seed={seed_patch}  tick={i+1:02d}/{n}",
                                  color=(80, 80, 255))
            frame = np.hstack([c_panel, p_panel])
            cv2.imwrite(str(td_path / f"{i:06d}.png"), frame)
        write_video_with_ffmpeg(td_path, out_path, fps, FRAME_W, FRAME_H)
    print(f"  [ok] {out_path.relative_to(out_path.parents[3])}  ({n} ticks @ {fps:.1f} fps)")
    return True


def make_overview_video(pairs: list[tuple[Path, Path]], out_path: Path, max_cols: int = 4) -> bool:
    """Compose a grid of pair side-by-sides into one overview video.
    Each cell shows one per-run side-by-side scaled to fit.
    All runs play in sync (same tick simultaneously).
    """
    # Decide grid: at most max_cols columns, enough rows for all pairs
    n_pairs = len(pairs)
    if n_pairs == 0:
        print(f"  [skip] no pairs for overview")
        return False
    cols = min(max_cols, n_pairs)
    rows = (n_pairs + cols - 1) // cols
    cell_w = OVERVIEW_W // cols
    cell_h = OVERVIEW_H // rows
    half_w = cell_w // 2

    # Compute tick count = min over all pairs
    tick_counts = []
    for c_dir, p_dir in pairs:
        nc = len(list_image_files(c_dir))
        np_ = len(list_image_files(p_dir))
        tick_counts.append(min(nc, np_))
    n_ticks = min(tick_counts) if tick_counts else 0
    if n_ticks == 0:
        print(f"  [skip] no ticks for overview")
        return False
    fps = max(1.0, n_ticks / TARGET_DURATION_S)

    # Pre-list image paths
    pair_imgs = []
    for c_dir, p_dir in pairs:
        pair_imgs.append((list_image_files(c_dir), list_image_files(p_dir)))

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for t in range(n_ticks):
            canvas = np.zeros((OVERVIEW_H, OVERVIEW_W, 3), dtype=np.uint8)
            for k, (c_files, p_files) in enumerate(pair_imgs):
                r, col = divmod(k, cols)
                y0, x0 = r * cell_h, col * cell_w
                if t >= len(c_files) or t >= len(p_files):
                    continue
                c = cv2.imread(str(c_files[t]))
                p = cv2.imread(str(p_files[t]))
                if c is None or p is None:
                    continue
                # Each half panel of the cell
                left = fit_into_panel(c, half_w, cell_h)
                right = fit_into_panel(p, half_w, cell_h)
                # Tag with run number
                cv2.rectangle(left, (0, 0), (half_w, 30), (0, 0, 0), -1)
                cv2.putText(left, f"run {k+1:02d}  CLEAN", (4, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.rectangle(right, (0, 0), (half_w, 30), (0, 0, 0), -1)
                cv2.putText(right, f"run {k+1:02d}  PATCH", (4, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 255), 1, cv2.LINE_AA)
                cell = np.hstack([left, right])
                canvas[y0:y0 + cell_h, x0:x0 + cell_w] = cell
            cv2.imwrite(str(td_path / f"{t:06d}.png"), canvas)
        write_video_with_ffmpeg(td_path, out_path, fps, OVERVIEW_W, OVERVIEW_H)
    print(f"  [ok] {out_path.relative_to(out_path.parents[3])}  "
          f"({n_pairs} pairs, {rows}x{cols} grid, {n_ticks} ticks @ {fps:.1f} fps)")
    return True


def process_agent_town(root: Path, agent: str, town: str,
                       make_per_run: bool, make_overview: bool):
    clean_root = root / agent / "clean" / town
    patch_root = root / agent / "patch" / town
    clean_runs = list_run_dirs(clean_root)
    patch_runs = list_run_dirs(patch_root)
    if not clean_runs or not patch_runs:
        print(f"[skip] {agent}/{town}: no runs (clean={len(clean_runs)}, patch={len(patch_runs)})")
        return
    n = min(len(clean_runs), len(patch_runs))
    pairs = list(zip(clean_runs[:n], patch_runs[:n]))
    out_dir = root / agent / town / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{agent}/{town}] {n} pairs -> {out_dir}")

    if make_per_run:
        for i, (c, p) in enumerate(pairs, start=1):
            out_path = out_dir / f"run_{i:02d}.mp4"
            if out_path.exists():
                print(f"  [skip exists] run_{i:02d}.mp4")
                continue
            make_per_run_video(c, p, out_path, i)

    if make_overview:
        out_path = out_dir / "overview.mp4"
        if out_path.exists():
            print(f"  [skip exists] overview.mp4")
        else:
            make_overview_video(pairs, out_path)


def discover_agent_towns(root: Path) -> list[tuple[str, str]]:
    """Find every (agent, town) pair that has both clean and patch sub-folders."""
    found = []
    for agent_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        clean_dir = agent_dir / "clean"
        patch_dir = agent_dir / "patch"
        if not clean_dir.exists() or not patch_dir.exists():
            continue
        towns = set(p.name for p in clean_dir.iterdir() if p.is_dir())
        towns &= set(p.name for p in patch_dir.iterdir() if p.is_dir())
        for t in sorted(towns):
            found.append((agent_dir.name, t))
    return found


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True,
                   help="multi_agent_<ts> folder under experiments/carla_scenarios/")
    p.add_argument("--agents", nargs="*", default=None,
                   help="Restrict to these agent folder names (default: all that have clean+patch).")
    p.add_argument("--towns", nargs="*", default=None,
                   help="Restrict to these towns (default: all).")
    p.add_argument("--no-per-run", action="store_true",
                   help="Skip individual run_NN.mp4 generation (overview only).")
    p.add_argument("--no-overview", action="store_true",
                   help="Skip overview.mp4 generation.")
    args = p.parse_args()

    if not args.root.exists():
        raise SystemExit(f"root not found: {args.root}")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not on PATH")

    all_pairs = discover_agent_towns(args.root)
    if args.agents:
        all_pairs = [(a, t) for a, t in all_pairs if a in args.agents]
    if args.towns:
        all_pairs = [(a, t) for a, t in all_pairs if t in args.towns]
    print(f"Will process {len(all_pairs)} (agent, town) cells under {args.root.name}")

    for agent, town in all_pairs:
        process_agent_town(args.root, agent, town,
                           make_per_run=not args.no_per_run,
                           make_overview=not args.no_overview)


if __name__ == "__main__":
    main()
