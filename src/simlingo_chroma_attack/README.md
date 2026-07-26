# simlingo_chroma_attack — white-box patch against SimLingo

Attacks `simlingo_simlingo` (Mini-InternVL2-1B: InternViT-300M + LoRA-tuned
Qwen2-0.5B) through **its own waypoint head**, not through an external detector.

## Why this loss and not a "hide" loss

SimLingo emits no objectness and no distance, so there is nothing to suppress —
which is exactly why the Fase-1 patches (trained against YOLOv8) transferred to
nothing. What it does emit is waypoints, and the agent turns two of them into
throttle (`agent_simlingo.py:858-863`):

```
one_second    = carla_fps // (wp_dilation * data_save_freq) = 20 // 5 = 4
half_second   = 2
desired_speed = ||wp[0] - wp[2]|| * 2.0                   # m/s
brake         = desired_speed < 0.4  or  ego_speed / desired_speed > 1.1
```

One lever. Pushing the waypoints forward *is* telling the car it may accelerate,
and it is the same quantity that decides whether it brakes at all.
`SimlingoSpeedUpLoss` maximises it.

## Probe results (vortex, RTX 4000 Ada 20 GiB, `conda activate PCLA15`)

| | |
|---|---|
| model load | 957.2 M params, **1.83 GiB** weights, ~8 s |
| gradient reaches the raw image | **yes** — L2 `1.99e+00`, absmax `2.34e-01`, 70 % of pixels non-zero, all finite |
| peak memory, batch 1 | **3.55 GiB** |
| peak memory, batch 4 | 8.72 GiB (~0.51 s/step) |
| peak memory, batch 8 | 15.65 GiB (~1.06 s/step) — practical ceiling |
| batch 12 | OOM |

Memory was never the problem: a 1 B-parameter VLM with a ~600-token sequence is
small next to a ViT backbone at 448².

Single-frame sanity (60 Adam steps, 4 frames, patch 256x512):

| | desired speed | brake fraction |
|---|---|---|
| clean | 31.4 km/h | 0.25 |
| patched (differentiable surrogate) | **48.7 km/h** | **0.00** |
| patched, **deployed autoregressive path** | 47.9 km/h (clean 35.3) | — |

The commentary the VLM emits changes too:

* clean — *"Accelerate to follow the black car that is to the front in 32.0 meters."*
* patched — *"Accelerate to drive through the junction."*

i.e. the leader stops being mentioned at all.

## Two upstream facts worth knowing

1. **`DrivingModel.forward(predict_language=False)` is broken upstream.**
   driving.py:179 does `features = self.forward_model(...)` but `forward_model`
   returns the tuple `(adaptor_features, adaptor_logits)`, so the next line
   indexes a tuple with a tuple and raises `TypeError`. `SimlingoWrapper.predict`
   reimplements the branch correctly (and drops the redundant second
   `replace_placeholder_tokens`, which made `forward` run the ViT twice).
2. **The differentiable path is a surrogate.** Deployment uses
   `predict_language=True`, which greedily decodes ~100 commentary tokens and
   conditions the driving queries on them — discrete, no gradient. On a random
   frame the two agreed to within 1.88 m per waypoint (34.99 vs 37.46 km/h).
   Close enough to optimise against, and the overfit test above confirms the
   attack carries over — but every patch must still be checked with
   `SimlingoWrapper.predict_deployed` before closed loop.

## Dataset caveat — READ THIS BEFORE TRUSTING A NUMBER

SimLingo drives off **one** camera: 1024x512, FOV 110, mounted at
`(x=-1.5, y=0, z=2.0)` (`config_simlingo.py:53-61`). The Fase-1 chroma-key
capture is **1280x720, FOV 90, at `(x=1.2, y=0, z=1.5)`**
(`chroma_key_dataset_generator/capture_fase1.py:51-53,240`) — a different
resolution, a different field of view and a mount 2.7 m further back and 0.5 m
higher.

`train.py` will happily resize those frames to 1024x512 and run, and that is
enough to develop and validate the loss (everything above was measured that
way). It is **not** enough to claim a deployable patch: the marker quad lands
at the wrong scale and the wrong place in the frame relative to what SimLingo
actually sees. A capture through SimLingo's own rig is required for the real
run — the same conclusion `tfv6_chroma_attack/capture_tfv6.py` reached for the
6-camera rig.

## Files

| file | role |
|---|---|
| `simlingo_model.py` | all PCLA coupling: standalone checkpoint load, differentiable re-implementation of the agent's PIL preprocessing, prompt construction, the gradient-preserving forward |
| `simlingo_loss.py` | `SimlingoSpeedUpLoss`, contract `__call__(image, target_bbox) -> (loss, info)`, identical to `YoloHideLoss` so it drops into the shared skeleton |
| `train.py` | Adam-over-pixels loop, reusing `yolo_chroma_attack`'s dataset / renderer / EoT / TV |

Patch export is unchanged — reuse `src.yolo_chroma_attack.export_tga`.

## The three loss signals

All in m/s, so the weights are directly comparable.

* `v_pid` = `||wp[0]-wp[2]||*2` — the literal scalar `control_pid` uses.
  The objective that matters, but it touches only 2 of the 10 waypoints.
  Weight `--w-pid`, default 1.0.
* `v_long` = `mean_k( x_k / ((k+1)*0.25s) )` — the mean speed implied by every
  waypoint. Denser and better conditioned; all ten heads get gradient.
  Weight `--w-long`, default 0.5.
* `v_fd` = `mean(diff(x)/0.25s)` — driving.py:540-542, the profile the SimLingo
  authors report. Nearly collinear with `v_long`; off by default (`--w-fd 0`).

`--speed-cap` turns each term into a hinge `relu(cap - v)`, the analogue of
`YoloHideLoss.margin_tau`: frames already driven above `cap` m/s stop consuming
patch capacity. Default off.

`target_bbox` is accepted and ignored — there is no spatially indexed output to
filter. It is in the signature only to keep the swap-in contract.

## Running a training job

```bash
ssh vortex
conda activate PCLA15
cd /home/vortex/adversarial-patch-vehicle

PYTHONPATH=/home/vortex/adversarial-patch-vehicle \
python src/simlingo_chroma_attack/train.py \
    --run-dir data/chroma_key_dataset/capture_20260609_014138_marker \
    --out-dir experiments/simlingo_attack/run01 \
    --epochs 10 --batch-size 4 --lr 0.02 \
    --geom-eot --tv-weight 1.0 \
    --speed-jitter 5 11
```

`PYTHONPATH` needs only the repo root — `SimlingoWrapper` puts `/home/vortex/PCLA`
and `/home/vortex/PCLA/pcla_agents/simlingo` on `sys.path` itself (override the
PCLA location with `--pcla-root` or `$PCLA_ROOT`).

Every eval prints the patched **and** the clean baseline, because the only
meaningful claim is the delta:

```
[ep  4/10] train desired= 41.20 km/h brake=0.02 | val desired= 39.80 km/h brake=0.05 (clean 28.66 / 0.250) | ...
```

`--speed-jitter LO HI` resamples the ego speed once per step (not per sample, so
all prompts in a batch keep the same token length and the batch stays a single
forward), which stops the patch overfitting one conditioning.
