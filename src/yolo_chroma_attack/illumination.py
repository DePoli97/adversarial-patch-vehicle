"""Illumination-aware compositing: bridge the train↔deploy lighting gap.

During training the patch is composited at its own RGB values, ignoring scene
light. In deployment it is baked into the CarlaCola BaseColor (albedo) and lit by
the scene: `pixel ≈ albedo × illumination`. At night the real patch is therefore
far darker than a raw 2D composite assumes (measured 9.5× too bright, see
FASE1_RESULTS.md).

Fix: estimate the local scene illumination from the truck body *around* the
marker (a ring just outside the quad), and scale the patch brightness to match
before YOLO sees it. Day frames (already bright) get a scale ≈ 1 → near no-op;
night frames get a small scale → the patch is darkened, matching deployment.

The scale is a per-frame constant (detached), so the optimiser must find a
pattern that fools YOLO even when darkened, instead of exploiting absolute
brightness. This is a first-order model: a single luminance scalar, ignoring
per-pixel lighting/tonemapping — but it removes most of the gap.
"""
from __future__ import annotations

import cv2
import numpy as np

# Rec.601 luma weights (BGR order for cv2 images).
_LUMA_BGR = np.array([0.114, 0.587, 0.299], dtype=np.float32)


def ring_luminance(bgr: np.ndarray, corners: np.ndarray,
                   margin_frac: float = 0.35) -> float:
    """Median luminance [0,1] of a ring just outside the marker quad — i.e. the
    truck body immediately around the marker, which is lit by the same scene
    light the marker would receive. Robust (median) to background leakage.

    corners: (4,2) float, ordered TL/TR/BR/BL in pixel coords.
    margin_frac: ring thickness as a fraction of the quad's mean side length.
    """
    h, w = bgr.shape[:2]
    quad = corners.astype(np.int32)
    side = float(np.linalg.norm(corners[0] - corners[1]) +
                 np.linalg.norm(corners[1] - corners[2])) / 2.0
    d = max(3, int(side * margin_frac))

    inner = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(inner, quad, 255)
    outer = cv2.dilate(inner, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))
    ring = cv2.subtract(outer, inner)

    ys, xs = np.where(ring > 0)
    if len(xs) < 20:
        # fall back to whole-frame luminance
        y = (bgr.astype(np.float32) / 255.0) @ _LUMA_BGR
        return float(np.median(y))
    px = bgr[ys, xs].astype(np.float32) / 255.0     # (N,3) BGR
    lum = px @ _LUMA_BGR
    return float(np.median(lum))


def rectify_quad(bgr: np.ndarray, corners: np.ndarray, ph: int, pw: int) -> np.ndarray:
    """Warp the marker quad region to the patch canonical grid (ph, pw), so it is
    pixel-aligned with the patch. corners ordered TL/TR/BR/BL."""
    canonical = np.array([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]],
                         dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), canonical)
    return cv2.warpPerspective(bgr, M, (pw, ph))  # (ph, pw, 3) BGR


def illumination_map_twin(night_bgr: np.ndarray, day_bgr: np.ndarray,
                          corners: np.ndarray, ph: int, pw: int,
                          s_min: float = 0.03, s_max: float = 1.15,
                          blur: bool = True) -> np.ndarray:
    """PER-PIXEL illumination map for the patch region, from the day/night twin.

    The yellow marker occupies exactly the patch pixels. `marker_night /
    marker_day` per pixel is the local illumination: same surface, same albedo,
    only the light differs, so the ratio cancels the (yellow) albedo and captures
    spatial lighting variation (e.g. a streetlight grazing one corner). Returned
    as an (ph, pw) scalar map in [s_min, s_max]; multiply the patch by it.
    """
    n = rectify_quad(night_bgr, corners, ph, pw).astype(np.float32) / 255.0
    d = rectify_quad(day_bgr, corners, ph, pw).astype(np.float32) / 255.0
    Ln = n @ _LUMA_BGR
    Ld = d @ _LUMA_BGR
    m = np.clip(Ln / np.maximum(Ld, 1e-3), s_min, s_max).astype(np.float32)
    if blur:
        k = max(3, (min(ph, pw) // 16) | 1)  # odd kernel
        m = cv2.GaussianBlur(m, (k, k), 0)
    return m


def illumination_map_ref(marker_bgr: np.ndarray, corners: np.ndarray,
                         ph: int, pw: int, yellow_ref: float = 0.80,
                         s_min: float = 0.03, s_max: float = 1.15,
                         blur: bool = True) -> np.ndarray:
    """PER-PIXEL illumination map for captures WITHOUT a twin (e.g. old _014138):
    the yellow marker's luminance divided by the known bright-yellow albedo
    reference. Daylight frames → ≈1; darker frames/regions → dimmer."""
    r = rectify_quad(marker_bgr, corners, ph, pw).astype(np.float32) / 255.0
    L = r @ _LUMA_BGR
    m = np.clip(L / max(yellow_ref, 1e-3), s_min, s_max).astype(np.float32)
    if blur:
        k = max(3, (min(ph, pw) // 16) | 1)
        m = cv2.GaussianBlur(m, (k, k), 0)
    return m


def illumination_scale_twin(night_bgr: np.ndarray, day_bgr: np.ndarray,
                            corners: np.ndarray,
                            s_min: float = 0.04, s_max: float = 1.15,
                            margin_frac: float = 0.35) -> float:
    """Illumination scale from the deterministic day/night twin: same pose, same
    albedo, only the lighting differs, so the ring-luminance RATIO is the pure
    illumination drop with NO albedo confound. Day frame vs itself → 1.0.

    This is the principled estimate for Fase 1 (deterministic capture). corners
    are shared between the twins (night corners were transferred from day).
    """
    L_night = ring_luminance(night_bgr, corners, margin_frac)
    L_day = ring_luminance(day_bgr, corners, margin_frac)
    return float(np.clip(L_night / max(L_day, 1e-6), s_min, s_max))


def illumination_scale(bgr: np.ndarray, corners: np.ndarray,
                       ref: float, s_min: float = 0.04, s_max: float = 1.15,
                       margin_frac: float = 0.35) -> float:
    """Fallback for captures WITHOUT a day/night twin (e.g. the old _014138):
    scale = ring_luminance / ref, clamped. `ref` should be a robust daylight
    reference (e.g. the 90th percentile of ring luminance over that capture), so
    bright frames give ≈ 1.0 and any darker frames are dimmed.
    """
    L = ring_luminance(bgr, corners, margin_frac)
    return float(np.clip(L / max(ref, 1e-6), s_min, s_max))
