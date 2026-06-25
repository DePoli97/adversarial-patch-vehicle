"""PyTorch Dataset for CLIP-targeted adversarial patch training.

Each sample pairs:
  - `marker_image`  : the CARLA frame with the yellow chroma-key marker on the
                      leader's rear window — corners of the quad come from the
                      same `quads_index.json` we already produce for YOLO training
  - `noleader_image`: the matching frame captured at the same vehicle pose with
                      the leader truck removed. This is the **target** the
                      patched embedding should be driven toward.

Both images are resized to CLIP's expected input size (224x224 for ViT-B/*).
Corner coordinates are rescaled accordingly so the patch warps correctly onto
the marker quad after resize.

The marker / clean / noleader triplet folders live side by side, e.g.:
    data/chroma_key_dataset/
        capture_20260609_014138_marker/    <- has quads_index.json, used for warp
        capture_20260609_014138_clean/     <- truck visible, no marker (unused here)
        capture_20260609_014138_noleader/  <- truck removed, target for the attack
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _resolve_noleader_dir(marker_dir: Path) -> Path:
    """Given .../capture_<ts>_marker, return .../capture_<ts>_noleader."""
    name = marker_dir.name
    if not name.endswith("_marker"):
        raise ValueError(
            f"Expected marker_dir to end with '_marker' (got '{name}'). "
            f"Pass --marker-dir explicitly if your naming differs."
        )
    base = name[: -len("_marker")]
    noleader = marker_dir.parent / f"{base}_noleader"
    if not noleader.exists():
        raise FileNotFoundError(
            f"Sibling noleader folder not found: {noleader}. "
            "The triplet capture must include the no-leader pass."
        )
    return noleader


class ClipChromaDataset(Dataset):
    """Loads (marker_image, noleader_image, quad_corners) tuples.

    Args:
        marker_dir   : path to capture_<ts>_marker/ (must contain quads_index.json).
        noleader_dir : path to capture_<ts>_noleader/. If None, resolved from marker_dir.
        split        : "train" / "val" / "all" — deterministic split on sorted stems.
        seed         : RNG seed for the split.
        image_size   : (H, W) target after resize. Defaults to (224, 224) for CLIP ViT.
        val_fraction : fraction of frames held out for val.
        min_area     : skip quads smaller than this many pixels in the original frame.
        min_side_ratio: skip ribbon-like quads (failed marker detection).
        index_name   : JSON file inside marker_dir produced by extract_quad.
    """

    def __init__(
        self,
        marker_dir: str | Path,
        noleader_dir: str | Path | None = None,
        split: str = "train",
        seed: int = 0,
        image_size: tuple[int, int] = (224, 224),
        val_fraction: float = 0.2,
        min_area: float = 400.0,
        min_side_ratio: float = 0.15,
        index_name: str = "quads_index.json",
    ):
        self.marker_dir = Path(marker_dir)
        self.noleader_dir = Path(noleader_dir) if noleader_dir else _resolve_noleader_dir(self.marker_dir)
        self.image_size = image_size

        index_path = self.marker_dir / index_name
        if not index_path.exists():
            raise FileNotFoundError(
                f"Missing {index_path}. Run extract_quad.py --batch-index first."
            )
        with open(index_path) as f:
            raw_index = json.load(f)

        # Filter degenerate quads (same rule as the YOLO dataset).
        def _ok(entry):
            corners = np.asarray(entry["corners"], dtype=np.float32)
            if cv2.contourArea(corners) < min_area:
                return False
            (_, _), (w, h), _ = cv2.minAreaRect(corners)
            short = min(w, h)
            long_ = max(w, h)
            if long_ < 1 or short / long_ < min_side_ratio:
                return False
            return True

        self.index = {k: v for k, v in raw_index.items() if _ok(v)}

        # Need the matching noleader image to exist for every retained stem.
        self.index = {
            k: v for k, v in self.index.items()
            if (self.noleader_dir / f"{k}.png").exists()
        }

        stems = sorted(self.index.keys())
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(stems))
        n_val = int(round(len(stems) * val_fraction))
        val_idx = set(order[:n_val].tolist())
        if split == "train":
            self.stems = [s for i, s in enumerate(stems) if i not in val_idx]
        elif split == "val":
            self.stems = [s for i, s in enumerate(stems) if i in val_idx]
        elif split == "all":
            self.stems = stems
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self) -> int:
        return len(self.stems)

    def _load_resized(self, path: Path, orig_shape) -> tuple[np.ndarray, float, float]:
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise RuntimeError(f"Could not load image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        new_h, new_w = self.image_size
        old_h, old_w = orig_shape[0], orig_shape[1]
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        sx = new_w / old_w
        sy = new_h / old_h
        return rgb, sx, sy

    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        entry = self.index[stem]
        corners = np.asarray(entry["corners"], dtype=np.float32)
        orig_shape = entry["shape"]

        marker_rgb, sx, sy = self._load_resized(self.marker_dir / f"{stem}.png", orig_shape)
        noleader_rgb, _, _ = self._load_resized(self.noleader_dir / f"{stem}.png", orig_shape)
        corners = corners * np.array([sx, sy], dtype=np.float32)

        marker_t = torch.from_numpy(marker_rgb).float().permute(2, 0, 1) / 255.0
        noleader_t = torch.from_numpy(noleader_rgb).float().permute(2, 0, 1) / 255.0

        return {
            "marker_image": marker_t,
            "noleader_image": noleader_t,
            "corners": torch.from_numpy(corners),
            "stem": stem,
        }


def collate(batch):
    marker = torch.stack([b["marker_image"] for b in batch], dim=0)
    noleader = torch.stack([b["noleader_image"] for b in batch], dim=0)
    corners = torch.stack([b["corners"] for b in batch], dim=0)
    stems = [b["stem"] for b in batch]
    return {
        "marker_image": marker,
        "noleader_image": noleader,
        "corners": corners,
        "stems": stems,
    }
