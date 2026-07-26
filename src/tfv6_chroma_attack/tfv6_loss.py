"""White-box adversarial loss against TransFuser v6 (vision-only, ResNet-34).

Unlike `yolo_chroma_attack.yolo_loss`, which attacks an *external* detector and
therefore only transfers by luck, this loss attacks the driving policy's OWN
differentiable heads. The victim is the exact network PCLA runs as
`tfv6_visiononly`, loaded offline from its checkpoint directory.

Victim wiring
-------------
The model is instantiated with the proven offline recipe (see README): build a
`TrainingConfig` from the checkpoint's `config.json`, `create_model(cfg)`, load
the state dict, freeze all parameters, then call `net(data)` DIRECTLY. The
`OpenLoopInference` / `ClosedLoopInference` wrappers are decorated with
`@torch.inference_mode()` and would silently destroy the gradient, so they are
never used here.

Input is `rgb` of shape (B, 3, 384, 2304): the 6 surround cameras (each
384x384, FOV 60) concatenated along width. ImageNet normalisation happens
*inside* the backbone and is differentiable, so the patch is optimised directly
in raw [0, 255] pixel space.

What we attack
--------------
    L = L_speed + lambda_wp * L_wp + lambda_detect * L_detect

**L_speed (the workhorse).** `pred_target_speed_distribution` is a set of 8
LOGITS over the speed bins `cfg.target_speeds`
(0, 4, 8, 10, 13.89, 16, 17.78, 20 m/s). Closed-loop inference turns them into
the PID setpoint with `decode_two_hot` — the expected value under the softmax —
and additionally slams the setpoint to 0 when `P(bin 0) > brake_threshold`
(0.9, `config_open_loop.py:24`). Throttle AND brake both come from this head by
default (`config_closed_loop.py:33-35`: `throttle_modality =
brake_modality = "target_speed"`), which is precisely why it is the workhorse.
So we push on both quantities:

  (a) maximise the expected speed  E = sum(softmax(logits) * target_speeds),
      contributing  -E / max(target_speeds)  (normalised, so the weights below
      are interpretable and comparable across models);
  (b) suppress P(bin 0), the "stop" bin that arms the hard-brake override.

For (b) the default form is a log barrier, `-log(1 - P0)`, not plain `P0`. Its
derivative w.r.t. P0 is `1 / (1 - P0)`: it explodes exactly in the regime we
must escape (P0 near 1, override firing) and decays to ~1 once P0 is small.
Plain `P0` has a constant derivative and therefore keeps spending optimiser
capacity on frames that are already safe at P0 = 0.001, while under-serving the
frames that actually brake. Set `p0_form="linear"` to recover the plain term.

Both terms accept an optional C&W-style hinge (`speed_margin_ms`, `p0_margin`),
mirroring `margin_tau` in `YoloHideLoss`: once a frame has clearly succeeded it
stops producing gradient, so the batch mean is not dominated by the flat tail
of already-broken frames.

**L_wp (secondary).** `pred_future_waypoints` (B, 8, 2) is the ego trajectory in
ego-relative BEV metres; we push it forward along +x. Note this is genuinely
secondary for tfv6: with the default modalities, waypoints drive NOTHING
(steer comes from `pred_route`, throttle/brake from the speed head). It is kept
as a cheap consistency pressure and because `throttle_modality="waypoint"` is a
supported configuration where waypoint spacing sets the desired speed
(`closed_loop_inference.py:146-148`). Default weight is 0.

**L_detect (secondary).** `pred_bounding_box` is a CenterNet head on an 80x96
BEV grid. Two modes are required and a third is offered:

  - ``"suppress"``: push down the vehicle-class objectness so the truck
    disappears. This is the classic disappearance attack.
  - ``"shrink"`` (DEFAULT): push down `wh_pred` so the predicted box is
    smaller. Preferred because the ACC paper this work builds on found pure
    disappearance unstable and flickering — the box blinks in and out and the
    downstream planner recovers — whereas consistently mis-reporting the target
    made the attack reliable.
  - ``"farther"``: shift the vehicle-class heatmap mass to cells farther ahead,
    inflating the perceived longitudinal distance.

    CAVEAT A REVIEWER MUST CHECK: in this head the BEV *position* already
    encodes distance, so shrinking `wh` does NOT literally move the truck away
    the way shrinking a 2D image box does in a monocular detector — `wh` here is
    the object's metric footprint. ``"shrink"`` therefore makes the truck look
    like a small object rather than a distant one, and ``"farther"`` is the mode
    that actually inflates distance in BEV. ``"shrink"`` is kept as the default
    per the ACC-paper rationale above, but if the goal is specifically distance
    inflation, use ``"farther"``. Also note `pred_bounding_box` is a
    *perception* output: PCLA's tfv6 control path does not consume it, so
    L_detect is an auxiliary/interpretability term, not a control lever.

BEV grid mapping (DERIVED, not guessed)
---------------------------------------
Read off `center_net_decoder.py:400-440` + `carla_dataset_utils.py:1149-1169` +
`config_training.py:86-134`, and confirmed live against the checkpoint's config:

    carla_leaderboard_mode = True
    min_x_meter = -32, max_x_meter = 64   (x = FORWARD)
    min_y_meter = -40, max_y_meter =  40  (y = LATERAL, +y = right)
    pixels_per_meter = 4.0, bev_down_sample_factor = 4

`lidar_width_pixel  = (max_x - min_x) * ppm = 384` -> grid W = 384/4 = 96 (x)
`lidar_height_pixel = (max_y - min_y) * ppm = 320` -> grid H = 320/4 = 80 (y)

matching the observed heatmap shape (B, 8, 80, 96). `get_topk_from_heatmap`
returns `topk_ys = idx // W` (the ROW) and `topk_xs = idx % W` (the COLUMN), and
`bb_image_to_vehicle_system` then yields

    x_forward_m = col / cells_per_meter + min_x_meter
    y_lateral_m = row / cells_per_meter + min_y_meter

with `cells_per_meter = pixels_per_meter / bev_down_sample_factor = 1.0`, i.e.
ONE GRID CELL IS ONE METRE and the ego sits at (row 40, col 32). The mapping is
therefore unambiguous and `bev_locate="analytic"` is the default: we place the
region of interest from the known leader distance.

The `"auto"` fallback (select the cells with the strongest vehicle-class
response in the forward half of the grid) is implemented and is used
automatically when no leader position is supplied. When it is active, the info
dict carries ``bev_locate="auto"`` — this is a heuristic, not the derived
mapping, and results obtained under it must be reported as such.

One upstream quirk worth recording: the decode multiplies by `pixels_per_meter`
and then divides by it again, which is a no-op that only lands on metres because
`pixels_per_meter == bev_down_sample_factor` for this config. This module uses
the general `cells_per_meter` form instead and warns if the two ever diverge.

Ensembling
----------
Real inference averages a 3-checkpoint ensemble. `open_loop_inference.py:93-116`
averages the target-speed LOGITS across members and only then applies softmax /
`decode_two_hot`, so that is exactly what `ensemble=True` does here; waypoints
and routes are likewise averaged post-hoc. Bounding boxes are ensembled
upstream only AFTER non-differentiable top-k decoding, so for the detection term
this module averages the raw heads (heatmap logits and `wh_pred`) instead — a
deliberate differentiable surrogate, not a reproduction of the upstream merge.
`ensemble=False` loads a single checkpoint and is ~3x faster for development.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

# `TransfuserBoundingBoxClass.VEHICLE` (lead/common/constants.py:79). The
# CarlaCola leader is a vehicle, so this is the class we attack.
VEHICLE_CLASS = 0

# `command_to_one_hot` (carla_dataset_utils.py:1583) maps CARLA command 4
# (LANEFOLLOW) to one-hot index 3. Following a leader down a straight road is
# LANEFOLLOW, so this is the right default context.
LANEFOLLOW_ONEHOT_INDEX = 3
NUM_COMMAND_CLASSES = 6

DEFAULT_CKPT_DIR = "/home/vortex/PCLA/pcla_agents/transfuserv6_pretrained/visiononly_resnet34"
DEFAULT_PCLA_ROOT = "/home/vortex/PCLA"
DEFAULT_CKPT_FILES = ("model_0030_0.pth", "model_0030_1.pth", "model_0030_2.pth")

# Radar is fed as zeros: the branch runs but, in the vision-only variant, its
# output never reaches the planner (verified by probe).
RADAR_POINTS = 300


def add_tfv6_to_syspath(pcla_root: str | Path = DEFAULT_PCLA_ROOT) -> None:
    """Make `lead.*` importable. Idempotent."""
    pcla_root = Path(pcla_root)
    for p in (pcla_root, pcla_root / "pcla_agents" / "transfuserv6"):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


class Tfv6HideLoss:
    """White-box adversarial loss against tfv6 vision-only.

    Matches the `YoloHideLoss` contract: ``__call__(image, target_bbox)``
    returns ``(loss, info_dict)``.

    Args:
        ckpt_dir: checkpoint directory holding `config.json` + `model_0030_*.pth`.
        device: 'cuda' or 'cpu' (cpu is impractically slow, dev only).
        ensemble: average all 3 checkpoints (matches real inference). False
            loads only `ckpt_files[0]` — ~3x faster, for development.
        ckpt_files: checkpoint file names, in ensemble order.
        pcla_root: PCLA install root, prepended to `sys.path`.

        w_expected_speed: weight of the -E/vmax term.
        w_p0: weight of the P(bin 0) suppression term.
        p0_form: 'logbarrier' (default, `-log(1 - P0)`) or 'linear' (`P0`).
        speed_margin_ms: C&W hinge on E. When set, the term becomes
            `relu(margin - E)/vmax`: identical gradient below the margin, zero
            above it. None/0 disables (plain maximisation).
        p0_margin: C&W hinge on P0. P0 below this contributes nothing; the
            remainder is rescaled to [0, 1]. 0 disables.

        lambda_wp: weight of the waypoint term (default 0 = off).
        wp_forward_index: which component of `pred_future_waypoints[..., i]` is
            the ego forward axis. 0 per the BEV convention.
        wp_norm_m: normaliser for the waypoint term, in metres.

        lambda_detect: weight of the detection term (default 0 = off).
        detect_mode: 'shrink' (default) | 'suppress' | 'farther'.
        bev_locate: 'analytic' (default) | 'auto'. 'analytic' needs a leader
            position, from `leader_distance_m` / `leader_lateral_m` or the
            per-sample `leader_xy` argument; without one it degrades to 'auto'.
        leader_distance_m: default leader distance ahead of ego, in metres.
        leader_lateral_m: default leader lateral offset (+ = right), in metres.
        bev_sigma_m: Gaussian radius of the BEV region of interest, in metres.
        bev_auto_topk: number of cells kept by the 'auto' fallback.
        heat_margin: hinge for 'suppress'; `bb_confidence_threshold` is 0.3, so
            0.3 means "stop once the box would be filtered out anyway".
        wh_floor_m: hinge for 'shrink'; `wh_pred` is an unbounded conv output,
            so this floor stops the optimiser driving it to -inf.
        wh_ref_m: normaliser for 'shrink', in metres (CarlaCola BEV footprint).
        farther_target_m: hinge for 'farther'; stop once the expected
            longitudinal position of the vehicle mass exceeds this. None = keep
            pushing.
        corridor_half_width_m: lateral half-width of the 'farther' corridor.

        ego_speed_ms: ego speed fed to the model, m/s.
        target_point_m: (x, y) next navigation target, ego-relative metres.
        town: town string fed to the model (affects nothing measurable but the
            forward pass requires it).
    """

    def __init__(
        self,
        ckpt_dir: str | Path = DEFAULT_CKPT_DIR,
        device: str = "cuda",
        ensemble: bool = True,
        ckpt_files: tuple[str, ...] = DEFAULT_CKPT_FILES,
        pcla_root: str | Path = DEFAULT_PCLA_ROOT,
        # --- L_speed ---
        w_expected_speed: float = 1.0,
        w_p0: float = 1.0,
        p0_form: str = "logbarrier",
        speed_margin_ms: float | None = None,
        p0_margin: float = 0.0,
        # --- L_wp ---
        lambda_wp: float = 0.0,
        wp_forward_index: int = 0,
        wp_norm_m: float = 20.0,
        # --- L_detect ---
        lambda_detect: float = 0.0,
        detect_mode: str = "shrink",
        bev_locate: str = "analytic",
        leader_distance_m: float | None = None,
        leader_lateral_m: float = 0.0,
        bev_sigma_m: float = 3.0,
        bev_auto_topk: int = 8,
        heat_margin: float = 0.3,
        wh_floor_m: float = 1.0,
        wh_ref_m: float = 7.0,
        farther_target_m: float | None = None,
        corridor_half_width_m: float = 4.0,
        # --- driving context ---
        ego_speed_ms: float = 8.0,
        target_point_m: tuple[float, float] = (10.0, 0.0),
        town: str = "Town04",
    ):
        if p0_form not in ("logbarrier", "linear"):
            raise ValueError(f"Unknown p0_form: {p0_form}")
        if detect_mode not in ("shrink", "suppress", "farther"):
            raise ValueError(f"Unknown detect_mode: {detect_mode}")
        if bev_locate not in ("analytic", "auto"):
            raise ValueError(f"Unknown bev_locate: {bev_locate}")

        add_tfv6_to_syspath(pcla_root)
        from lead.training.config_training import TrainingConfig
        from lead.training.training_utils import create_model

        self.device = torch.device(device)
        self.ckpt_dir = Path(ckpt_dir)
        with open(self.ckpt_dir / "config.json") as f:
            cfg_json = json.load(f)
        # NB: never assign cfg.device — it is a read-only property.
        self.cfg = TrainingConfig(cfg_json)

        files = list(ckpt_files) if ensemble else [ckpt_files[0]]
        self.nets = []
        for name in files:
            net = create_model(self.cfg).to(self.device).eval()
            state = torch.load(self.ckpt_dir / name, map_location=self.device,
                               weights_only=True)
            net.load_state_dict(state, strict=False)
            for p in net.parameters():
                p.requires_grad_(False)
            self.nets.append(net)
        self.ckpt_files = files

        # --- speed bins ---
        self.target_speeds = torch.tensor(self.cfg.target_speeds,
                                          dtype=torch.float32, device=self.device)
        self.vmax = float(self.target_speeds.max())

        # --- BEV geometry, derived from the config (see module docstring) ---
        self.cells_per_meter = self.cfg.pixels_per_meter / self.cfg.bev_down_sample_factor
        self.min_x_m = float(self.cfg.min_x_meter)
        self.min_y_m = float(self.cfg.min_y_meter)
        if abs(self.cfg.pixels_per_meter - self.cfg.bev_down_sample_factor) > 1e-6:
            # Upstream's decode multiplies then divides by pixels_per_meter,
            # which only lands on metres when these two are equal. If a future
            # config breaks that, our mapping and theirs disagree — say so
            # loudly rather than silently reporting wrong distances.
            warnings.warn(
                "pixels_per_meter != bev_down_sample_factor: this module's BEV "
                "metric mapping no longer matches center_net_decoder's decode. "
                "Re-derive before trusting L_detect diagnostics.",
                RuntimeWarning, stacklevel=2,
            )

        self.w_expected_speed = w_expected_speed
        self.w_p0 = w_p0
        self.p0_form = p0_form
        self.speed_margin_ms = speed_margin_ms
        self.p0_margin = p0_margin

        self.lambda_wp = lambda_wp
        self.wp_forward_index = wp_forward_index
        self.wp_norm_m = wp_norm_m

        self.lambda_detect = lambda_detect
        self.detect_mode = detect_mode
        self.bev_locate = bev_locate
        self.leader_distance_m = leader_distance_m
        self.leader_lateral_m = leader_lateral_m
        self.bev_sigma_m = bev_sigma_m
        self.bev_auto_topk = bev_auto_topk
        self.heat_margin = heat_margin
        self.wh_floor_m = wh_floor_m
        self.wh_ref_m = wh_ref_m
        self.farther_target_m = farther_target_m
        self.corridor_half_width_m = corridor_half_width_m

        self.ego_speed_ms = ego_speed_ms
        self.target_point_m = target_point_m
        self.town = town

        self._grid_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def _build_data(self, image: torch.Tensor,
                    ego_speed: torch.Tensor | None) -> dict:
        """Assemble the model input dict for a batch of composited images."""
        B = image.shape[0]
        dev = self.device
        tp = torch.tensor(self.target_point_m, dtype=torch.float32, device=dev)
        if ego_speed is None:
            speed = torch.full((B,), self.ego_speed_ms, dtype=torch.float32, device=dev)
        else:
            speed = ego_speed.to(dev, dtype=torch.float32).reshape(B)
        cmd = F.one_hot(torch.full((B,), LANEFOLLOW_ONEHOT_INDEX, device=dev),
                        NUM_COMMAND_CLASSES).float()
        return {
            "rgb": image,
            "radar": torch.zeros(B, RADAR_POINTS, 5, device=dev),
            "target_point_previous": torch.zeros(B, 2, device=dev),
            "target_point": tp.unsqueeze(0).expand(B, -1).contiguous(),
            "target_point_next": (tp * 2.0).unsqueeze(0).expand(B, -1).contiguous(),
            "speed": speed,
            "command": cmd,
            "next_command": cmd.clone(),
            "town": [self.town] * B,
        }

    def _forward_ensemble(self, image: torch.Tensor,
                          ego_speed: torch.Tensor | None) -> dict:
        """Run every checkpoint and average the heads we attack.

        Speed logits / waypoints / route are averaged exactly as
        `open_loop_inference.ensemble_planning_decoder` does (mean over members
        BEFORE softmax). Heatmap logits and `wh_pred` are averaged as a
        differentiable stand-in for the upstream post-decode box merge.
        """
        data = self._build_data(image, ego_speed)
        speed_logits, waypoints, heat_logits, wh = [], [], [], []
        for net in self.nets:
            # Deliberately NOT wrapped in torch.enable_grad(): that would
            # override a caller's torch.no_grad() and make evaluation build a
            # full backward graph for every ensemble member. Calling net()
            # directly is what preserves gradients — the thing that destroys
            # them is OpenLoopInference/ClosedLoopInference.forward, which are
            # decorated @torch.inference_mode().
            with torch.amp.autocast(
                device_type="cuda", dtype=self.cfg.torch_float_type,
                enabled=self.cfg.use_mixed_precision_training,
            ):
                out = net(data)
            speed_logits.append(out.pred_target_speed_distribution.float())
            if out.pred_future_waypoints is not None:
                waypoints.append(out.pred_future_waypoints.float())
            bb = out.pred_bounding_box
            if bb is not None:
                heat_logits.append(bb.center_heatmap_logit_pred.float())
                wh.append(bb.wh_pred.float())

        def _mean(xs):
            return torch.stack(xs, dim=0).mean(dim=0) if xs else None

        return {
            "speed_logits": _mean(speed_logits),
            "waypoints": _mean(waypoints),
            "heat_logits": _mean(heat_logits),
            "wh": _mean(wh),
        }

    # ------------------------------------------------------------------
    # loss terms
    # ------------------------------------------------------------------

    def _speed_terms(self, logits: torch.Tensor):
        """(a) maximise expected speed, (b) suppress the stop bin."""
        probs = logits.softmax(dim=-1)                       # (B, 8)
        expected = (probs * self.target_speeds).sum(dim=-1)  # (B,)
        p0 = probs[:, 0]                                     # (B,)

        if self.speed_margin_ms:
            # Hinge: same gradient as -E while E < margin, none once past it.
            l_expected = torch.relu(self.speed_margin_ms - expected) / self.vmax
        else:
            l_expected = -expected / self.vmax

        if self.p0_margin > 0.0:
            p0_eff = torch.relu(p0 - self.p0_margin) / (1.0 - self.p0_margin)
        else:
            p0_eff = p0
        if self.p0_form == "logbarrier":
            # -log(1 - P0): gradient 1/(1-P0) blows up exactly where the
            # hard-brake override fires, and decays once the frame is safe.
            l_p0 = -torch.log1p(-p0_eff.clamp(max=1.0 - 1e-6))
        else:
            l_p0 = p0_eff

        l_speed = (self.w_expected_speed * l_expected + self.w_p0 * l_p0).mean()
        return l_speed, l_expected.mean(), l_p0.mean(), expected, p0

    def _wp_term(self, waypoints: torch.Tensor):
        """Push the predicted ego trajectory forward along +x."""
        fwd = waypoints[..., self.wp_forward_index]  # (B, n_wp)
        return -fwd.mean() / self.wp_norm_m, fwd.mean()

    # --- BEV helpers ---------------------------------------------------

    def _bev_axes(self, H: int, W: int):
        """Metric coordinate of every BEV row (y) and column (x). Cached."""
        key = (H, W)
        if key not in self._grid_cache:
            rows = torch.arange(H, dtype=torch.float32, device=self.device)
            cols = torch.arange(W, dtype=torch.float32, device=self.device)
            y_m = rows / self.cells_per_meter + self.min_y_m
            x_m = cols / self.cells_per_meter + self.min_x_m
            self._grid_cache[key] = (y_m, x_m)
        return self._grid_cache[key]

    def _roi_analytic(self, leader_xy: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Gaussian BEV region of interest around the known leader position.

        `leader_xy` is (B, 2) in ego-relative metres, (x forward, y lateral).
        Returns (B, H, W) weights that sum to 1 per sample. Constant w.r.t. the
        patch, so it carries no gradient of its own — it only selects cells.
        """
        y_m, x_m = self._bev_axes(H, W)
        dy = y_m.view(1, H, 1) - leader_xy[:, 1].view(-1, 1, 1)
        dx = x_m.view(1, 1, W) - leader_xy[:, 0].view(-1, 1, 1)
        w = torch.exp(-(dx ** 2 + dy ** 2) / (2.0 * self.bev_sigma_m ** 2))
        return w / w.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)

    def _roi_auto(self, heat_v: torch.Tensor) -> torch.Tensor:
        """Fallback ROI: the top-k strongest vehicle cells in the FORWARD half.

        HEURISTIC, not the derived mapping — used only when no leader position
        is supplied. `heat_v` is (B, H, W) vehicle-class objectness; it is
        detached so the ROI itself never receives gradient.
        """
        B, H, W = heat_v.shape
        _, x_m = self._bev_axes(H, W)
        forward = (x_m > 0.0).view(1, 1, W).expand(B, H, W)
        scores = heat_v.detach().masked_fill(~forward, float("-inf")).reshape(B, -1)
        k = min(self.bev_auto_topk, scores.shape[1])
        _, idx = scores.topk(k, dim=1)
        w = torch.zeros(B, H * W, device=heat_v.device)
        w.scatter_(1, idx, 1.0)
        w = w.reshape(B, H, W)
        return w / w.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)

    def _detect_term(self, heat_logits: torch.Tensor, wh: torch.Tensor,
                     leader_xy: torch.Tensor | None):
        """Detection term over the truck's BEV cells. Returns (loss, info)."""
        heat = heat_logits.sigmoid()               # (B, C, H, W)
        heat_v = heat[:, VEHICLE_CLASS]            # (B, H, W)
        B, H, W = heat_v.shape

        if self.bev_locate == "analytic" and leader_xy is not None:
            roi = self._roi_analytic(leader_xy, H, W)
            locate = "analytic"
        else:
            roi = self._roi_auto(heat_v)
            locate = "auto"

        info = {
            "bev_locate": locate,
            "detect_heat_roi": float((roi * heat_v.detach()).sum(dim=(1, 2)).mean()),
            "detect_wh_roi_m": float(
                (roi.unsqueeze(1) * wh.detach()).sum(dim=(2, 3)).mean()),
        }

        if self.detect_mode == "suppress":
            # Classic disappearance: drive vehicle objectness below the
            # confidence threshold, then stop (hinge).
            l = (roi * torch.relu(heat_v - self.heat_margin)).sum(dim=(1, 2)).mean()
            l = l / max(1.0 - self.heat_margin, 1e-6)
        elif self.detect_mode == "shrink":
            # `wh_pred` is in vehicle-system metres (see module docstring). The
            # floor keeps the unbounded conv output from running to -inf.
            shrink = torch.relu(wh - self.wh_floor_m).mean(dim=1)  # (B, H, W)
            l = (roi * shrink).sum(dim=(1, 2)).mean() / self.wh_ref_m
        else:  # "farther"
            # Move vehicle-class mass down the forward corridor: normalise the
            # heatmap over the corridor into a distribution and maximise its
            # expected longitudinal position.
            y_m, x_m = self._bev_axes(H, W)
            if leader_xy is not None:
                lat = leader_xy[:, 1].view(-1, 1)
            else:
                lat = torch.zeros(B, 1, device=heat_v.device)
            in_lane = (y_m.view(1, H) - lat).abs() <= self.corridor_half_width_m
            corridor = in_lane.view(B, H, 1) & (x_m > 0.0).view(1, 1, W)
            mass = heat_v * corridor.float()
            p = mass / mass.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            exp_x = (p * x_m.view(1, 1, W)).sum(dim=(1, 2))  # (B,)
            if self.farther_target_m is not None:
                l = (torch.relu(self.farther_target_m - exp_x)
                     / self.farther_target_m).mean()
            else:
                l = (-exp_x / float(self.cfg.max_x_meter)).mean()
            info["detect_expected_x_m"] = float(exp_x.detach().mean())

        return l, info

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        image: torch.Tensor,                    # (B, 3, 384, 2304) in [0, 255]
        target_bbox: torch.Tensor | None = None,
        leader_xy: torch.Tensor | None = None,  # (B, 2) ego-relative metres
        ego_speed: torch.Tensor | None = None,  # (B,) m/s
    ) -> tuple[torch.Tensor, dict]:
        """Compute the white-box loss. Returns (loss, info_dict).

        `target_bbox` is accepted only to match the `YoloHideLoss` contract and
        is ignored: tfv6 is attacked through its own heads, not through an
        image-space detection region. Pass `leader_xy` instead when the
        detection term is enabled — it is what places the BEV region of
        interest. When it is None and `bev_locate="analytic"`, the class falls
        back to the `leader_distance_m` / `leader_lateral_m` defaults, and if
        those are unset too it degrades to the `"auto"` heuristic and says so in
        `info["bev_locate"]`.
        """
        del target_bbox  # API compatibility only

        if leader_xy is None and self.leader_distance_m is not None:
            leader_xy = torch.tensor(
                [[self.leader_distance_m, self.leader_lateral_m]],
                dtype=torch.float32, device=self.device,
            ).expand(image.shape[0], -1)
        elif leader_xy is not None:
            leader_xy = leader_xy.to(self.device, dtype=torch.float32)

        heads = self._forward_ensemble(image, ego_speed)

        l_speed, l_expected, l_p0, expected, p0 = self._speed_terms(
            heads["speed_logits"])
        loss = l_speed

        info = {
            "l_speed": float(l_speed.detach()),
            "l_expected": float(l_expected.detach()),
            "l_p0": float(l_p0.detach()),
            "expected_speed_ms": float(expected.detach().mean()),
            "p0": float(p0.detach().mean()),
            "top_speed_bin": int(heads["speed_logits"].detach().argmax(dim=-1)[0]),
            "n_ckpt": len(self.nets),
        }

        if self.lambda_wp != 0.0 and heads["waypoints"] is not None:
            l_wp, fwd_mean = self._wp_term(heads["waypoints"])
            loss = loss + self.lambda_wp * l_wp
            info["l_wp"] = float(l_wp.detach())
            info["wp_forward_mean_m"] = float(fwd_mean.detach())

        if self.lambda_detect != 0.0 and heads["heat_logits"] is not None:
            l_detect, det_info = self._detect_term(
                heads["heat_logits"], heads["wh"], leader_xy)
            loss = loss + self.lambda_detect * l_detect
            info["l_detect"] = float(l_detect.detach())
            info.update(det_info)

        info["loss"] = float(loss.detach())
        return loss, info
