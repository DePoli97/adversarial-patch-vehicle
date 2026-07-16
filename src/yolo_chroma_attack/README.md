# yolo_chroma_attack — adversarial patch vs YOLO

Train an adversarial patch that **minimises YOLOv8's confidence** on the
CarlaCola truck's rear marker. The patch is warped onto the yellow chroma-key
marker (via the detected quad corners) and optimised against a *frozen* YOLO —
we train the patch, never the detector.

## Pipeline (in order)

1. **Capture** the dataset — done by `../chroma_key_dataset_generator/`
   (`capture_fase1.py`, `capture_distances.sh`). Produces, per
   `<town>/<light>/dist<N>m/{marker,noleader,clean}/`, deterministic frames of
   the truck following the ego at a fixed gap.
2. **Index the marker corners** — `../chroma_key_dataset_generator/build_fase1_indexes.py`.
   Detects the quad on the lit *day* frame and transfers the same corners to the
   deterministic *night* twin, writing `quads_index.json` next to the frames.
3. **Train** — `train.py` (or the sweep wrapper `sweep_fase1.sh`). Renders the
   patch on each frame's quad, runs YOLO, and descends the hide loss.
4. **Evaluate / pick** — `evaluate.py`, `pick_best.py`, `analyze_patch.py`.
5. **Export to CARLA** — `export_tga.py` bakes the patch into the marker TGA for
   the closed-loop system-level test (`../carla_scenario/`).

## Files

| File | Role |
|------|------|
| `train.py` | main training loop (patch vs frozen YOLO); writes `patch_final.pt` + **`args.json`** |
| `yolo_loss.py` | hide loss: margin/hinge on the vehicle-class score inside the target box |
| `eot.py` | Expectation-over-Transformation: photometric noise + **geometric** warp (`--geom-eot`) + `total_variation` |
| `patch_render.py` | warp the patch onto the marker quad and composite |
| `dataset.py` | loads frames + their `quads_index.json` corners |
| `sweep_fase1.sh` | trains **1 patch per road** (day+night+all distances pooled) + 1 generalist |
| `evaluate.py` / `pick_best.py` | score a patch / choose the best across a sweep |
| `export_tga.py` | bake a `.pt` patch into a marker TGA for CARLA |
| `build_run_manifest.py` | rebuild the "which patch came from which dataset" index (see below) |
| `_diag_*.py` | one-off diagnostic scripts (kept for reference; the `_` prefix marks them scratch) |

## Winning config (from the 2026-07-07 Town04_day ablation)

- surrogate **`yolov8m.pt`** — `n/s/x` don't even detect the clean CarlaCola
- **`--topk 3`** — attack the strongest anchors, not a diluted top-20 mean
- **`--margin-tau 0.05`** — hinge just above the noise floor
- **`--geom-eot`** — the decisive lever: forces a spatially-coherent pattern
  instead of high-frequency noise (99% confidence drop on its own)
- **TV loss off** — smooths but blurs the pattern with no perf gain

See `../../docs/ablation_geom_tv/README.md` for the ablation table.

## Anti-chaos rule: every patch is traceable via its `args.json`

`train.py` writes `args.json` next to `patch_final.pt` recording the **exact
dataset and every hyperparameter**. The run-folder names (timestamps, `run01`,
`pooled`, …) are *not* authoritative — the `args.json` is. To rebuild the full
index across all runs:

```bash
python src/yolo_chroma_attack/build_run_manifest.py \
    --root experiments/yolo_attack \
    --out  experiments/yolo_attack/MANIFEST.md
```

The runs themselves (weights, large tensors) live under
`experiments/yolo_attack/` and are gitignored — only `*.json/.csv/.md/.log` and
`analysis/*.png` are tracked. A copy of the manifest is kept at
`../../docs/RUN_MANIFEST.md`.
