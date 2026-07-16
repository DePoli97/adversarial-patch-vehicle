# Fase 1 — patch-vs-YOLO results (2026-07-11)

Durable record of the Fase 1 patch sweep and the open-loop evaluation matrix.
This is the tracked thesis record: the `.pt` weights live on Vortex under
`experiments/yolo_attack/` (gitignored); this file + `MANIFEST.md` are the index.

## What was trained (5 patches)

All patches: surrogate `yolov8m.pt`, topk=3, margin_tau=0.05, geom-EOT on, tv=0,
30 epochs, batch 24, lr 0.06. Same CarlaCola van in every dataset.

| patch | training dataset | frames | run dir on Vortex |
|-------|------------------|--------|-------------------|
| Town04 | fase1 Town04 (day+night × 6/10/20 m) | 666 | `fase1_20260711_153141/Town04_spawn273/` |
| Town07 | fase1 Town07 (day+night × dist) | 744 | `fase1_20260711_153141/Town07_spawn38/` |
| Town11 | fase1 Town11 (day+night × dist) | 906 | `fase1_20260711_153141/Town11_spawn1713/` |
| pooled | fase1 all 3 roads | 2316 | `fase1_20260711_153141/pooled/` |
| generalist | `_014138` (multi-city, random NPC/spots) + fase1 all | 4125 | `generalist_full_20260711_164353/generalist_full/` |

`_014138` corners were re-detected with the NEW diff-based algorithm
(`build_flat_index.py`, marker−noleader box + HSV), 97.9% recall (1809/1848).

## Open-loop evaluation matrix (PROXY, not the real test)

Metric = `detection_rate` on each road's held-out val split = fraction of frames
where **YOLOv8m** still sees a vehicle in the target box (lower = better attack).
**This is NOT the system-level evaluation.** It is a fast proxy to compare
patches. The real evaluation is the CARLA closed-loop crash count (pending), and
it uses a PCLA *driving* agent, not YOLO — so it also tests YOLO→agent transfer.

| patch \ road | Town04 | Town07 | Town11 |
|--------------|--------|--------|--------|
| Town04 (own) | 24.8%  | 8.8%   | 0.0%   |
| Town07 (own) | 32.3%  | **7.5%** | 0.0% |
| Town11 (own) | 43.6%  | 13.6%  | **0.0%** |
| pooled       | **12.8%** | **6.8%** | 0.0% |
| generalist   | 20.3%  | 6.8%   | 0.0%   |
| *clean (no patch)* | *58.6%* | *27.9%* | *5.0%* |
| *random patch*     | *52.6%* | *26.5%* | *3.3%* |

## Reading (with caveats — do NOT overclaim)

1. **Town11 → 0% with every patch.** Genuine, reportable result: in a
   low-clutter scene (empty fields, no background) the attack fully hides the
   vehicle. This matches Tonella's actual ask — *find a scenario where such an
   attack is feasible*. Caveat: clean is already 5% on Town11, i.e. YOLO barely
   detects the CarlaCola there even without a patch, so the scene does part of
   the work; the patch finishes it (5%→0%).

2. **Per-road does NOT win on its own road.** On Town04 and Town07 (the roads
   with real margin), the road-specific patch is beaten by pooled. So Tonella's
   "train on the specific road" hypothesis is NOT supported by this proxy.

3. **But it is NOT simply "more data wins" either.** The full generalist (most
   data) does not beat pooled (only the 3 roads); pooled is best on Town04
   (12.8% vs 20.3%) and tied on Town07. The likely explanation is not "variety
   hurts" but **in-distribution advantage**: pooled is trained on exactly the 3
   roads it is tested on, while the generalist spends capacity on `_014138`'s
   other towns that aren't in the test set. This is a confound of the proxy, not
   a clean finding — a fair test needs the CARLA closed-loop and/or evaluation on
   a road held out of training entirely.

## FINDING for the thesis: train↔deploy lighting gap (2026-07-11)

During training the patch is composited onto the frame at its **own RGB values**,
independent of scene illumination. In deployment it is baked into the CarlaCola
**BaseColor (albedo)** texture, so the renderer lights it: `pixel ≈ albedo × light`.
At night the scene light is low → the real patch is far darker than training assumed.

Measured on Town04 dist10m, frame 000030 (Town04 patch composited):
- **day**:   scene_mean=0.652, patch_mean=0.540 → patch is **0.83×** the scene (matched, realistic)
- **night**: scene_mean=0.057, patch_mean=0.540 → patch is **9.48×** the scene (impossible once lit as albedo)

Figure: `docs/lighting_gap/{day,night}_000030.png`. Reproduce with
`src/yolo_chroma_attack/render_composite_doc.py`.

Consequences:
- Night-trained (or night-included) patches are optimised against an unrealistically
  bright patch → their attack may not transfer to the real night simulation. This is
  also why pooling day+night "fights itself": the optimiser can't reconcile the two.
- Day patches have a small gap and are trustworthy.

Planned fix: **illumination-aware compositing** — estimate local scene luminance
around the marker and scale the patch brightness to match BEFORE the YOLO forward
pass, so the optimiser must find a pattern that fools YOLO even when darkened. This
is a first-order illumination model (scalar luminance match; ignores per-pixel
lighting/tonemapping) but bridges most of the gap. The winning loss/EOT config
(topk3, margin 0.05, geom-EOT) stays FROZEN — this is one added transform, NOT a new
sweep axis.

## Next step (decided)

CARLA closed-loop, 4 packages:
- Town04 road ← Town04 patch, Town07 road ← Town07 patch, Town11 road ← Town11 patch
- **generalist** as the baseline on all 3 roads (NOT pooled: pooled = "3 random
  roads" has no attacker threat-model; a real attacker trains on all data they
  have → generalist).
- Plus a **clean (no-patch)** run per road to measure the patch's actual effect
  on collisions.
Metric = collisions with the leader (spatially-gated detector, see
`src/carla_scenario/scenario_two_vehicles.py`). This also tests transfer from the
YOLO surrogate to the PCLA driving agent.
