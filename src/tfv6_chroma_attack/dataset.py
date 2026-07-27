"""Dataset for tfv6 white-box patch training.

Same idea as `yolo_chroma_attack.dataset.ChromaKeyDataset` — CARLA frames plus
the chroma-key quad the patch is warped onto — but shaped for tfv6's input:
the 6 surround cameras (384x384 each) concatenated along width into a single
(3, 384, 2304) tensor.

Expected `run_dir` layout
-------------------------
This is exactly what `capture_tfv6.py` + `build_quads.py` produce:

    <run_dir>/
        000001.png              (384, 2304, 3) BGR composite
        000001.json             per-frame metadata sidecar
        per_camera/000001/cam1..cam6.png    (optional, --save-per-camera)
        quads_index.json        {"<stem>": {"corners": [[x,y] x4],
                                            "shape": [384, 2304, 3], ...}}

`build_quads.py` stores the quad in COMPOSITE coordinates, so the defaults
(`image_layout="stitched"`, `corners_frame="stitched"`) read that output
directly.

Images come from one of two layouts, selected by `image_layout`:

  - ``"stitched"``: one `{stem}.png` already 2304x384 (the 6 views side by side)
  - ``"cameras"``:  `per_camera/{stem}/cam{i}.png` (i = 1..6, in rig order),
    which this class concatenates along width

`corners_frame` says which coordinate frame the stored quad lives in:

  - ``"stitched"``: already in the 2304-wide composite
  - ``"front"``:    in the FRONT camera's own 384x384 frame; x is shifted by
    `front_cam_index * 384` to land in the composite. Camera index 2 (yaw 0) is
    the front view that sees the leader truck.

There is deliberately NO auto-detection between the two: guessing the frame
would silently mis-place the patch, so the layout must be stated.

Leader position
---------------
`leader_xy` is what places the BEV region of interest for `Tfv6HideLoss`'s
detection term: (x forward, y lateral, +y = right) in ego-relative metres.
`leader_source` picks where it comes from:

  - ``"index"``: a `leader_xy` field in the quads-index entry;
  - ``"meta"``: derived from the `{stem}.json` sidecar by projecting
    `leader_loc - ego_loc` onto the ego's forward/right axes using
    `ego_yaw_deg`. NB the sidecar's own `leader_distance_m` is a 3D Euclidean
    range between actor origins, NOT a longitudinal gap, so it is not used;
  - ``"auto"`` (default): index if present, else sidecar, else nothing;
  - ``"none"``: never load one.

Frames with no leader position fall back to the trainer's `--leader-distance`,
and the loss reports `bev_locate="auto"` if no position is available at all.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.yolo_chroma_attack.dataset import expand_bbox
from src.yolo_chroma_attack.illumination import illumination_map_ref

# tfv6 rig: 6 cameras of 384x384, concatenated along width.
CAM_SIZE = 384
NUM_CAMS = 6
FRONT_CAM_INDEX = 2  # ONE-BASED rig index (cam1..cam6); yaw 0 -> slice [384:768]
TFV6_IMAGE_HW = (CAM_SIZE, CAM_SIZE * NUM_CAMS)  # (384, 2304)


def leader_xy_from_meta(meta: dict) -> list[float] | None:
    """Ego-relative leader position (x forward, y right) in metres.

    CARLA yaw is measured about +z with the forward unit vector
    (cos yaw, sin yaw) and the right unit vector (-sin yaw, cos yaw), so the
    world-frame delta projects onto the ego axes directly. +y = right matches
    TransFuser's vehicle system (`min_y_meter` is the LEFT boundary).
    """
    if "ego_loc" not in meta or "leader_loc" not in meta or "ego_yaw_deg" not in meta:
        return None
    dx = meta["leader_loc"][0] - meta["ego_loc"][0]
    dy = meta["leader_loc"][1] - meta["ego_loc"][1]
    yaw = math.radians(meta["ego_yaw_deg"])
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    return [dx * cos_y + dy * sin_y, -dx * sin_y + dy * cos_y]


class Tfv6ChromaDataset(Dataset):
    """CARLA 6-camera frames + chroma-key quads, in tfv6 input format.

    Args:
        run_dir: capture folder containing the images and the JSON index.
        split: "train" / "val" / "all" — deterministic split on sorted stems.
        seed: RNG seed for the split.
        val_fraction: fraction of stems held out for validation.
        index_name: JSON index file name inside `run_dir`.
        image_layout: "stitched" or "cameras" (see module docstring).
        corners_frame: "stitched" or "front" (see module docstring).
        front_cam_index: which camera the quad was measured in, when
            `corners_frame="front"`.
        leader_source: "auto" | "index" | "meta" | "none" (see module docstring).
        target_expand: (ex, ey) factor used to derive `target_bbox` from the
            quad. Kept only so previews and the `YoloHideLoss` contract stay
            usable; `Tfv6HideLoss` ignores `target_bbox`.
        min_area / min_side_ratio: reject degenerate quads whose homography
            would be near-singular.
        illum_patch_hw: if set, attach a per-frame illumination map at patch
            resolution (see `yolo_chroma_attack.illumination`).
        illum_yellow_ref: day-marker luminance reference for that map.
        gap_range: (min, max) leader gap in metres, read from each frame's
            capture sidecar. Use it to train only on the band where the model
            is actually deciding whether to brake.
    """

    def __init__(
        self,
        run_dir: str | Path,
        split: str = "train",
        seed: int = 0,
        val_fraction: float = 0.2,
        index_name: str = "quads_index.json",
        image_layout: str = "stitched",
        corners_frame: str = "stitched",
        front_cam_index: int = FRONT_CAM_INDEX,
        leader_source: str = "auto",
        target_expand: tuple[float, float] = (3.5, 3.5),
        min_area: float = 100.0,
        min_side_ratio: float = 0.15,
        illum_patch_hw: tuple[int, int] | None = None,
        illum_yellow_ref: float = 0.65,
        gap_range: tuple[float, float] | None = None,
    ):
        if image_layout not in ("stitched", "cameras"):
            raise ValueError(f"Unknown image_layout: {image_layout}")
        if corners_frame not in ("stitched", "front"):
            raise ValueError(f"Unknown corners_frame: {corners_frame}")
        if leader_source not in ("auto", "index", "meta", "none"):
            raise ValueError(f"Unknown leader_source: {leader_source}")
        if not 1 <= front_cam_index <= NUM_CAMS:
            raise ValueError(
                f"front_cam_index is the 1-based rig index (1..{NUM_CAMS}), "
                f"got {front_cam_index}"
            )
        if illum_patch_hw is not None:
            # `illumination_map_ref` divides the measured luminance by the albedo
            # of the yellow chroma marker. There is no marker in the tfv6 capture
            # — the quad sits on the CarlaCola's own livery, whose albedo swings
            # from dark lettering to white bodywork — so the resulting "map" is
            # the truck's logo, not the lighting. Training would then optimise
            # patch x livery-ghost, and deployment overwrites exactly those
            # texels, so the ghost disappears and the deployed pattern is not the
            # one that was optimised. Use illumination_map_twin (day/night ratio
            # of the same surface, which cancels the albedo) if this is needed.
            raise ValueError(
                "illum_patch_hw is not supported for the tfv6 capture: the "
                "reference-albedo illumination map assumes a yellow marker of "
                "known albedo, and these quads sit on the truck's own livery."
            )

        self.run_dir = Path(run_dir)
        self.image_layout = image_layout
        self.corners_frame = corners_frame
        self.front_cam_index = front_cam_index
        self.leader_source = leader_source
        self.target_expand = target_expand
        self.illum_patch_hw = illum_patch_hw
        self.illum_yellow_ref = illum_yellow_ref
        self._illum_cache: dict[str, torch.Tensor] = {}

        index_path = self.run_dir / index_name
        if not index_path.exists():
            raise FileNotFoundError(
                f"Missing {index_path}. Run the tfv6 capture + quad extraction first."
            )
        with open(index_path) as f:
            self.index = json.load(f)

        def _ok(entry) -> bool:
            corners = np.asarray(entry["corners"], dtype=np.float32)
            if cv2.contourArea(corners) < min_area:
                return False
            (_, _), (w, h), _ = cv2.minAreaRect(corners)
            long_ = max(w, h)
            return long_ >= 1 and min(w, h) / long_ >= min_side_ratio

        self.index = {k: v for k, v in self.index.items() if _ok(v)}

        if gap_range is not None:
            # Restrict to a band of leader distances. Measured against the real
            # closed loop, tfv6 decides whether to brake at a centre-to-centre
            # distance of about 13-15 m; frames much closer than that are
            # saturated at P(stop)=1 and frames much further are already
            # cruising, so training on the full sweep spends most of its capacity
            # where no decision is being made. The bound is read from the capture
            # sidecar's `leader_gap_m` (along-lane, centre to centre).
            lo, hi = gap_range
            kept = {}
            for stem, entry in self.index.items():
                meta_path = self.run_dir / f"{stem}.json"
                if not meta_path.exists():
                    continue
                with open(meta_path) as f:
                    gap = float(json.load(f).get("leader_gap_m", float("nan")))
                if lo <= gap <= hi:
                    kept[stem] = entry
            if not kept:
                raise ValueError(
                    f"gap_range={gap_range} keeps no frames in {self.run_dir}; "
                    "the capture's leader_gap_m values are outside that band."
                )
            self.index = kept

        stems = sorted(self.index.keys())
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(stems))
        val_idx = set(order[: int(round(len(stems) * val_fraction))].tolist())
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

    def _load_rgb(self, stem: str) -> np.ndarray:
        """Return the (384, 2304, 3) RGB composite for this frame."""
        if self.image_layout == "stitched":
            path = self.run_dir / f"{stem}.png"
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise RuntimeError(f"Could not load image: {path}")
        else:
            views = []
            for i in range(1, NUM_CAMS + 1):  # capture_tfv6.py names cam1..cam6
                path = self.run_dir / "per_camera" / stem / f"cam{i}.png"
                view = cv2.imread(str(path))
                if view is None:
                    raise RuntimeError(f"Could not load camera view: {path}")
                views.append(view)
            bgr = cv2.hconcat(views)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _leader_xy(self, stem: str, entry: dict) -> list[float] | None:
        """Resolve the ego-relative leader position for this frame, or None."""
        if self.leader_source in ("index", "auto") and "leader_xy" in entry:
            return list(entry["leader_xy"])
        if self.leader_source in ("meta", "auto"):
            meta_path = self.run_dir / f"{stem}.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    return leader_xy_from_meta(json.load(f))
        return None

    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        entry = self.index[stem]
        corners = np.asarray(entry["corners"], dtype=np.float32)  # (4, 2)

        rgb = self._load_rgb(stem)
        H, W = rgb.shape[:2]

        # Corners were measured in `entry["shape"]`; rescale them if the loaded
        # composite is a different size, then move them into composite space.
        src_h, src_w = entry.get("shape", [H, W, 3])[:2]
        if self.corners_frame == "front":
            if src_w >= CAM_SIZE * 2:
                # The entry was measured in the full composite, so shifting by a
                # camera origin would put the patch in the wrong place. Refuse
                # rather than train on a silently mis-placed quad.
                raise ValueError(
                    f"{stem}: corners_frame='front' but the entry's shape is "
                    f"{src_w}px wide (a full composite). Use "
                    "corners_frame='stitched' — that is what build_quads.py emits."
                )
            # Quad lives in one 384x384 view: rescale within that view, then
            # offset by the camera's slice origin in the composite.
            corners = corners * np.array([CAM_SIZE / src_w, CAM_SIZE / src_h],
                                         dtype=np.float32)
            # front_cam_index is the ONE-BASED rig index, matching the cam1..cam6
            # filenames and the `front_camera_index` key capture_tfv6.py writes
            # into camera_calibration.json. Camera 2 (yaw 0) starts at 384, not
            # 768 — treating it as 0-based would land every quad in the +57.5 deg
            # side view, which never sees the leader truck.
            corners[:, 0] += (self.front_cam_index - 1) * CAM_SIZE
        elif (src_h, src_w) != (H, W):
            corners = corners * np.array([W / src_w, H / src_h], dtype=np.float32)

        if (H, W) != TFV6_IMAGE_HW:
            raise ValueError(
                f"{stem}: composite is {(H, W)}, tfv6 needs {TFV6_IMAGE_HW}. "
                "Check --image-layout and the capture resolution."
            )

        target_bbox = expand_bbox(corners, (H, W, 3),
                                  expand_x=self.target_expand[0],
                                  expand_y=self.target_expand[1])

        item = {
            "image": torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0,
            "corners": torch.from_numpy(corners),
            "target_bbox": torch.from_numpy(target_bbox),
            "stem": stem,
        }
        leader_xy = self._leader_xy(stem, entry) if self.leader_source != "none" else None
        if leader_xy is not None:
            item["leader_xy"] = torch.tensor(leader_xy, dtype=torch.float32)
        if "speed" in entry:
            item["speed"] = torch.tensor(float(entry["speed"]), dtype=torch.float32)
        if self.illum_patch_hw is not None:
            if stem not in self._illum_cache:
                ph, pw = self.illum_patch_hw
                bgr_now = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                m = illumination_map_ref(bgr_now, corners, ph, pw,
                                         yellow_ref=self.illum_yellow_ref)
                self._illum_cache[stem] = torch.from_numpy(m).unsqueeze(0)
            item["illum"] = self._illum_cache[stem]
        return item


def collate(batch):
    """Stack tensors; keep stems as a list. Optional keys pass through."""
    out = {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "corners": torch.stack([b["corners"] for b in batch], dim=0),
        "target_bbox": torch.stack([b["target_bbox"] for b in batch], dim=0),
        "stems": [b["stem"] for b in batch],
    }
    for key in ("leader_xy", "speed", "illum"):
        if key in batch[0]:
            out[key] = torch.stack([b[key] for b in batch], dim=0)
    return out
