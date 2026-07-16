"""PyTorch Dataset for chroma-key adversarial patch training.

Reads frames captured by `chroma_key_dataset_generator/capture_frames.py` and
the matching quad index produced by `extract_quad.py --batch-index ...`.

Each item is a tuple (image, corners, target_bbox, frame_id):
  - image: float32 tensor (3, H, W) in [0, 1], RGB
  - corners: float32 (4, 2) in pixel coords, ordered TL/TR/BR/BL
  - target_bbox: float32 (4,) = (x1, y1, x2, y2) — leader region in pixels,
                 derived by expanding the quad's bounding box by a fixed factor.
                 Used to filter YOLO predictions during the hide loss.
  - frame_id: str, e.g. "000123"

The expansion factor approximates the carlacola truck around the rear-window
chroma-key marker; tune via the `--target-expand-x` / `--target-expand-y`
training args.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.yolo_chroma_attack.illumination import illumination_map_ref


def expand_bbox(corners: np.ndarray, img_shape: tuple,
                expand_x: float = 3.5, expand_y: float = 3.5) -> np.ndarray:
    """Expand the bounding box of `corners` around its centroid.

    Returns (x1, y1, x2, y2) clamped to image bounds. Asymmetric expand
    (x vs y) lets us bias toward the truck's actual aspect ratio: the
    chroma-key is on a roughly horizontal panel, the truck body extends
    mostly downward and sideways.
    """
    H, W = img_shape[:2]
    x = corners[:, 0]
    y = corners[:, 1]
    cx = float(x.mean())
    cy = float(y.mean())
    half_w = (x.max() - x.min()) * 0.5 * expand_x
    half_h = (y.max() - y.min()) * 0.5 * expand_y
    x1 = max(0.0, cx - half_w)
    y1 = max(0.0, cy - half_h)
    x2 = min(W - 1.0, cx + half_w)
    y2 = min(H - 1.0, cy + half_h)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


class ChromaKeyDataset(Dataset):
    """Loads CARLA chroma-key frames + their detected quads.

    Args:
        run_dir: path to a `capture_<ts>/` folder containing N.png + the
                 `quads_index.json` produced by extract_quad.
        split: "train" / "val" / "all" — deterministic 80/20 split on
               sorted frame stems (seedable).
        seed: RNG seed for the split.
        image_size: optional resize (h, w). Patch warping needs the actual
                    image size so corners are rescaled accordingly.
        target_expand: (ex, ey) factor used to derive target_bbox from quad.
    """

    def __init__(
        self,
        run_dir: str | Path,
        split: str = "train",
        seed: int = 0,
        image_size: tuple[int, int] | None = None,
        target_expand: tuple[float, float] = (3.5, 3.5),
        val_fraction: float = 0.2,
        min_area: float = 400.0,
        min_side_ratio: float = 0.15,
        index_name: str = "quads_index.json",
        illum_patch_hw: tuple[int, int] | None = None,
        illum_yellow_ref: float = 0.65,
    ):
        self.run_dir = Path(run_dir)
        self.image_size = image_size
        self.target_expand = target_expand
        # If illum_patch_hw is set, each item carries a per-frame illumination
        # map (marker luminance / yellow_ref) at patch resolution, computed once
        # and cached. Multiplying the patch by it lights the patch like the
        # scene (dark at night) — closes the train↔deploy lighting gap.
        self.illum_patch_hw = illum_patch_hw
        self.illum_yellow_ref = illum_yellow_ref
        self._illum_cache: dict[str, torch.Tensor] = {}

        index_path = self.run_dir / index_name
        if not index_path.exists():
            raise FileNotFoundError(
                f"Missing {index_path}. Run extract_quad.py --batch-index first."
            )
        with open(index_path) as f:
            self.index = json.load(f)

        # Filter degenerate quads: too small or too thin (near-collinear).
        # near-collinear breaks the perspective transform inversion.
        def _ok(entry):
            corners = np.asarray(entry["corners"], dtype=np.float32)
            if cv2.contourArea(corners) < min_area:
                return False
            # Aspect ratio of the oriented bbox: rejects ribbon-like quads.
            (_, _), (w, h), _ = cv2.minAreaRect(corners)
            short = min(w, h)
            long_ = max(w, h)
            if long_ < 1 or short / long_ < min_side_ratio:
                return False
            return True

        self.index = {k: v for k, v in self.index.items() if _ok(v)}

        # Deterministic train/val split on sorted stems.
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

    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        entry = self.index[stem]
        corners = np.asarray(entry["corners"], dtype=np.float32)  # (4, 2)
        orig_shape = entry["shape"]                                # (H, W, 3)

        img_path = self.run_dir / f"{stem}.png"
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise RuntimeError(f"Could not load image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if self.image_size is not None:
            new_h, new_w = self.image_size
            old_h, old_w = orig_shape[0], orig_shape[1]
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            sx = new_w / old_w
            sy = new_h / old_h
            corners = corners * np.array([sx, sy], dtype=np.float32)
            shape_for_bbox = (new_h, new_w, 3)
        else:
            shape_for_bbox = tuple(orig_shape)

        target_bbox = expand_bbox(
            corners, shape_for_bbox,
            expand_x=self.target_expand[0],
            expand_y=self.target_expand[1],
        )

        image = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0  # (3, H, W)
        item = {
            "image": image,
            "corners": torch.from_numpy(corners),         # (4, 2)
            "target_bbox": torch.from_numpy(target_bbox), # (4,)
            "stem": stem,
        }
        if self.illum_patch_hw is not None:
            if stem not in self._illum_cache:
                ph, pw = self.illum_patch_hw
                # rgb here is already at the training image_size (corners rescaled
                # to match), so the rectified marker is consistent with rendering.
                bgr_now = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                m = illumination_map_ref(bgr_now, corners, ph, pw,
                                         yellow_ref=self.illum_yellow_ref)
                self._illum_cache[stem] = torch.from_numpy(m).unsqueeze(0)  # (1,ph,pw)
            item["illum"] = self._illum_cache[stem]
        return item


def collate(batch):
    """Collate that stacks images/corners/bboxes; keeps stems as a list."""
    images = torch.stack([b["image"] for b in batch], dim=0)
    corners = torch.stack([b["corners"] for b in batch], dim=0)
    bboxes = torch.stack([b["target_bbox"] for b in batch], dim=0)
    stems = [b["stem"] for b in batch]
    out = {"image": images, "corners": corners, "target_bbox": bboxes, "stems": stems}
    if "illum" in batch[0]:
        out["illum"] = torch.stack([b["illum"] for b in batch], dim=0)  # (B,1,ph,pw)
    return out
