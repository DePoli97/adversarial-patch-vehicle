"""Adversarial 'speed-up' loss against SimLingo's waypoint heads.

Same contract as `YoloHideLoss`: `__call__(image, target_bbox) -> (loss, info)`,
so it drops straight into the shared training skeleton.

Why 'speed up' and not 'hide'
-----------------------------
SimLingo has no objectness head and no distance head, so there is nothing to
suppress. What it does have is a waypoint head whose output is read as a speed
command. The scenario is a leader truck that brakes hard: SimLingo reacts by
predicting waypoints bunched close to the ego, which `control_pid` reads as a
low desired speed and converts into brake. Pushing those waypoints forward is
therefore *exactly* the same thing as telling the car not to brake — one lever,
not two.

The three signals
-----------------
All are expressed in m/s so their weights are directly comparable.

  v_pid   `||wp[0] - wp[2]|| * 2.0` — the literal scalar
          `agent_simlingo.control_pid` turns into throttle and compares against
          the brake thresholds. Maximising it is the objective that matters,
          but it touches only 2 of the 10 waypoints.
  v_long  `mean_k( x_k / ((k+1) * 0.25s) )` — the mean speed implied by EVERY
          waypoint. A denser, better-conditioned version of the same push
          (all ten heads get gradient), used as an auxiliary term.
  v_fd    `mean( diff(x) / 0.25s )` — the finite-difference speed profile from
          driving.py:540-542. Nearly collinear with `v_long`; off by default,
          exposed because it is the quantity the SimLingo authors report.

Each is folded through

    f(v) = -v / norm                      (plain maximisation, cap=None)
    f(v) =  relu(cap - v) / norm          (hinge: stop once v exceeds cap)

The hinge is the analogue of `YoloHideLoss.margin_tau`: once a frame's implied
speed is comfortably above anything the scenario needs, it stops consuming
patch capacity and the optimiser spends it on the frames still braking.

`target_bbox` is accepted and ignored — SimLingo produces no spatially indexed
output to filter, so there is no per-region head to restrict the loss to. It is
in the signature purely to keep the swap-in contract.
"""
from __future__ import annotations

import torch

from src.simlingo_chroma_attack.simlingo_model import (
    BRAKE_RATIO, BRAKE_SPEED, SimlingoWrapper, desired_speed_from_wps,
    finite_difference_speeds, mean_implied_speed, would_brake,
)

MS_TO_KMH = 3.6


class SimlingoSpeedUpLoss:
    """Push SimLingo's waypoints forward so it accelerates instead of braking.

    Args:
        wrapper: an already-built `SimlingoWrapper`. If None, one is
                 constructed from `pcla_root` / `ckpt` / `device`.
        pcla_root, ckpt, device: forwarded to `SimlingoWrapper` when it has to
                 build one.
        speed_ms: ego speed used to condition the model — it goes into the
                 prompt ("Current speed: X m/s") and into `vehicle_speed`, and
                 it is the reference the brake predicate compares against.
                 8.0 m/s (~29 km/h) is the urban cruise the PCLA agents were
                 trained around.
        speed_jitter: (lo, hi) m/s. When set, `speed_ms` is resampled once per
                 call, so the patch does not overfit a single conditioning.
                 Sampling per CALL and not per SAMPLE keeps every prompt in the
                 batch the same length, which keeps the batch a single forward.
        target_point / next_target_point: ego-relative route hint, straight
                 ahead by default (the braking-leader scenario is a straight
                 road).
        w_pid / w_long / w_fd: weights on the three signals above.
        speed_cap_ms: hinge threshold. None = plain maximisation.
        norm_speed_ms: divisor that makes the loss dimensionless.
    """

    def __init__(
        self,
        wrapper: SimlingoWrapper | None = None,
        pcla_root: str | None = None,
        ckpt: str | None = None,
        device: str = "cuda",
        speed_ms: float = 8.0,
        speed_jitter: tuple[float, float] | None = None,
        target_point: tuple[float, float] = (20.0, 0.0),
        next_target_point: tuple[float, float] = (40.0, 0.0),
        w_pid: float = 1.0,
        w_long: float = 0.5,
        w_fd: float = 0.0,
        speed_cap_ms: float | None = None,
        norm_speed_ms: float = 10.0,
    ):
        if wrapper is None:
            kwargs = {"device": device}
            if pcla_root is not None:
                kwargs["pcla_root"] = pcla_root
            if ckpt is not None:
                kwargs["ckpt"] = ckpt
            wrapper = SimlingoWrapper(**kwargs)
        self.wrapper = wrapper
        self.device = wrapper.device
        self.speed_ms = speed_ms
        self.speed_jitter = speed_jitter
        self.target_point = target_point
        self.next_target_point = next_target_point
        self.w_pid = w_pid
        self.w_long = w_long
        self.w_fd = w_fd
        self.speed_cap_ms = speed_cap_ms
        self.norm_speed_ms = norm_speed_ms
        if max(w_pid, w_long, w_fd) <= 0:
            raise ValueError("at least one of w_pid / w_long / w_fd must be > 0")

    # ------------------------------------------------------------------ util
    def _sample_speed(self) -> float:
        if self.speed_jitter is None:
            return self.speed_ms
        lo, hi = self.speed_jitter
        return float(torch.empty(()).uniform_(lo, hi).item())

    def _term(self, v: torch.Tensor) -> torch.Tensor:
        """Fold a per-frame speed (B,) into a scalar to MINIMISE."""
        if self.speed_cap_ms is None:
            return (-v).mean() / self.norm_speed_ms
        return torch.relu(self.speed_cap_ms - v).mean() / self.norm_speed_ms

    # ------------------------------------------------------------------ call
    def __call__(
        self,
        image: torch.Tensor,                     # (B, 3, H, W) in [0, 1]
        target_bbox: torch.Tensor | None = None,  # unused, see module docstring
    ) -> tuple[torch.Tensor, dict]:
        ego_speed = self._sample_speed()
        pred = self.wrapper.predict(
            image, speed_ms=ego_speed,
            target_point=self.target_point,
            next_target_point=self.next_target_point,
        )
        wps = pred["speed_wps"].float()          # (B, 10, 2), ego-relative m

        v_pid = desired_speed_from_wps(wps)      # (B,) m/s — drives the throttle
        v_long = mean_implied_speed(wps)         # (B,) m/s — dense auxiliary
        v_fd = finite_difference_speeds(wps).mean(dim=-1)   # (B,) m/s

        loss = torch.zeros((), device=image.device)
        if self.w_pid:
            loss = loss + self.w_pid * self._term(v_pid)
        if self.w_long:
            loss = loss + self.w_long * self._term(v_long)
        if self.w_fd:
            loss = loss + self.w_fd * self._term(v_fd)

        with torch.no_grad():
            brake = would_brake(v_pid, ego_speed)
            info = {
                "loss": float(loss.detach().item()),
                # The headline, human-readable number: the speed SimLingo is
                # being told to drive at by its own waypoints.
                "desired_speed_kmh": float(v_pid.mean().item() * MS_TO_KMH),
                "desired_speed_kmh_min": float(v_pid.min().item() * MS_TO_KMH),
                "desired_speed_kmh_max": float(v_pid.max().item() * MS_TO_KMH),
                "implied_speed_kmh": float(v_long.mean().item() * MS_TO_KMH),
                "fd_speed_kmh": float(v_fd.mean().item() * MS_TO_KMH),
                # The money metric: on what fraction of frames would
                # control_pid still hit the brake? The attack succeeds when
                # this goes to 0 on frames where the clean model brakes.
                "brake_frac": float(brake.float().mean().item()),
                "ego_speed_ms": ego_speed,
                # Longitudinal reach of the last waypoint (2.5 s horizon).
                "wp_final_x_m": float(wps[:, -1, 0].mean().item()),
                "lateral_absmax_m": float(wps[..., 1].abs().max().item()),
            }
        return loss, info

    # ------------------------------------------------------------- reporting
    @torch.no_grad()
    def report(self, image: torch.Tensor, speed_ms: float | None = None) -> dict:
        """Metrics only, no graph — for clean-vs-patched baselines in eval."""
        ego_speed = self.speed_ms if speed_ms is None else speed_ms
        wps = self.wrapper.predict(
            image, speed_ms=ego_speed, target_point=self.target_point,
            next_target_point=self.next_target_point)["speed_wps"].float()
        v_pid = desired_speed_from_wps(wps)
        return {
            "desired_speed_kmh": float(v_pid.mean().item() * MS_TO_KMH),
            "brake_frac": float(would_brake(v_pid, ego_speed).float().mean().item()),
            "implied_speed_kmh": float(mean_implied_speed(wps).mean().item() * MS_TO_KMH),
            "wp_final_x_m": float(wps[:, -1, 0].mean().item()),
        }


__all__ = ["SimlingoSpeedUpLoss", "BRAKE_SPEED", "BRAKE_RATIO"]
