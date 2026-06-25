# CLIP-targeted adversarial patch — thesis history

Training of an adversarial patch against the **OpenCLIP ViT-B/32** image encoder
(LAION-2B pretrained). The patch is composited (via differentiable perspective
warp) onto the chroma-key marker on the leader truck's rear window, and we drive
the embedding of the patched scene toward the embedding of the *no-leader*
version of the same scene.

The package mirrors `src/yolo_chroma_attack/` (same dataset, same warp, same
EOT) so the difference is just the loss target.

---

## Loss versions

Each training run is identified by a loss formulation. The patch tensor, log,
and exported TGA for each version is kept under `experiments/clip_attack/runNN_*`
**without overwriting** so the thesis can document the progression.

### v1 — baseline (no regularization)

```
L_v1 = (1 - cos(e_patched, e_target)).mean()
```

- `e_patched` : CLIP CLS embedding of the marker frame after warping the patch.
- `e_target`  : CLIP CLS embedding of the same scene with the leader removed
                (the `_noleader` triplet folder).
- No regularizer.

**Run:** `experiments/clip_attack/run01_20260625_001129/`
**Asset:** `assets/chroma_key/adv_patch_clip_v1.TGA`
**Outcome:** val_loss 0.190 → 0.117 (-38 %), val_cos 0.810 → 0.883 (+9 pp) over
100 epochs in ~23 min. **Visual inspection failed**: the patch is pure
high-frequency noise. The optimizer found the statistical shortcut — perturbing
global [CLS] statistics without learning any semantic pattern — exactly the
mode-collapse described in Brown et al. 2017 when TV regularization is absent.

### v2 — with Total Variation regularization

```
L_v2     = (1 - cos(e_patched, e_target)).mean()  +  lambda_tv * TV(patch)
TV(p)    = mean(|p[:, 1:, :] - p[:, :-1, :]|) + mean(|p[:, :, 1:] - p[:, :, :-1]|)
lambda_tv = 5e-2     # was 2.5e-3 in a first attempt — too weak, TV did not decrease.
```

- Anisotropic L1 Total Variation on the patch tensor (3, 256, 512).
- Forces adjacent pixels to be similar → smooth, low-frequency patterns
  instead of high-frequency noise.
- `lambda_tv` is set so that the TV gradient effectively competes with the adv
  gradient under Adam's per-parameter normalization. The contribution to the
  loss value is a misleading proxy because Adam normalizes magnitudes — what
  matters is that `|lambda_tv * dTV/dpatch| ≈ |dL_adv/dpatch|`. With 2.5e-3
  the TV value stayed pinned at the random-init value (~0.67); 5e-2 actually
  drives TV down.

**Run:** `experiments/clip_attack/run02_<ts>/` (this version)
**Asset:** `assets/chroma_key/adv_patch_clip_v2.TGA`

### v3 — DPT-guided per-token loss (planned, post-presentation 2026-06-26)

```
L_v3 = sum_{k in T_truck} (1 - cos(f_patched[k], f_noleader[k]))
       + alpha * (1 - cos(CLS_patched, CLS_target))
       + lambda_tv * TV(patch)
```

`T_truck` is the set of ViT patch tokens that "see" the truck, identified via
a pretrained DPT segmentation head on the clean frame. Token-level loss
constrains the optimizer to act on truck-relevant features only, rather than
on the global CLS embedding. Likely requires switching to ViT-B/16 (14x14
token grid) for spatial fidelity.

---

## CLI flags

| Flag | Default | v1 used | v2 uses |
|---|---|---|---|
| `--lambda-tv` | 0.0 | 0.0 (omitted) | 2.5e-3 |
| `--clip-model` | ViT-B-32 | ViT-B-32 | ViT-B-32 |
| `--clip-pretrained` | laion2b_s34b_b79k | same | same |
| `--epochs` | 100 | 100 | 100 |
| `--batch-size` | 16 | 16 | 16 |
| `--lr` | 0.02 | 0.02 | 0.02 |
| `--cosine-lr` | off | on | on |
| `--lr-min` | 1e-3 | 1e-3 | 1e-3 |
| `--eot-noise` | 0.02 | 0.02 | 0.02 |
| `--patch-h` | 256 | 256 | 256 |
| `--patch-w` | 512 | 512 | 512 |

---

## How to run on Vortex

```bash
conda activate PCLA310
cd /home/vortex/adversarial-patch-vehicle

# v2 (this default)
PYTHONPATH=. python -u src/clip_chroma_attack/train.py \
    --marker-dir data/chroma_key_dataset/capture_20260609_014138_marker \
    --out-dir   experiments/clip_attack/run02_$(date +%Y%m%d_%H%M%S) \
    --epochs 100 --batch-size 16 --lr 0.02 --cosine-lr --lambda-tv 2.5e-3
```

The exported TGA is dropped in `assets/chroma_key/` and is swapped over
`yellow_marker.TGA` inside the CARLA package to deploy in the simulator.
