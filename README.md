# Adversarial Patches on Autonomous Driving Agents

Master's thesis (USI, defense September 2026) — **Paolo Deidda**.
Advisor: Prof. Paolo Tonella · Co-advisor: Masoud J. Tehrani.

Adversarial patches applied to a leader vehicle in CARLA, tested in
closed-loop against pretrained CARLA Leaderboard agents (PCLA framework).
Attack targets range from a CNN detector (YOLOv8) to a Vision Transformer
image encoder (OpenCLIP, as a surrogate for SimLingo's InternVL2).

Not intended as a standalone tool — this is a research repo tied to the
thesis timeline.

## Directory layout

```
adversarial-patch-vehicle/
├── src/                             # code
│   ├── patch_on_surface/            # YOLO patch training (CCPD + synthetic)
│   ├── chroma_key_dataset_generator/# CARLA synthetic dataset (chroma key marker)
│   ├── yolo_chroma_attack/          # YOLO patch trained on synthetic CARLA data
│   ├── clip_chroma_attack/          # OpenCLIP-targeted patch (v1/v2 done, v3 pending)
│   ├── carla_scenario/              # closed-loop leader+follower scenario
│   └── vehicle_counting_model/      # YOLOv8n weights (gitignored)
├── assets/                          # textures, TGAs, marker files
├── data/                            # datasets (CCPD, chroma_key_dataset)
├── experiments/                     # run outputs (patches, telemetry, videos)
├── docs/                            # thesis docs, presentations, figures
│   ├── .STATUS.md .LOG.md           # agent-facing internal state (Italian)
│   ├── .INIT_AGENT.md               # bootstrap instructions for coding agents
│   ├── THESIS_OVERVIEW.md           # long-form project overview
│   ├── MEETING_NOTES.md             # meeting minutes (Italian)
│   ├── presentation_tonella_*/      # slide decks per meeting
│   └── figures/                     # reusable plots
├── report/                          # thesis PDF (drafted August 2026)
├── pretrained/                      # pretrained model checkpoints
└── gittalo.sh                       # local sync helper (pull+add+commit+push)
```

## Setup

- Python 3.14 in `.venv/` (repo root, gitignored)
- CARLA 0.9.15 on Vortex machine (source build in `/home/vortex/carla/`)
- PCLA framework at `/home/vortex/PCLA/` (36 agents, all validated)
- GPU: Vortex RTX 4000 20 GB — the machine we run everything on
- Model weights and large data are gitignored; regenerate via the scripts in `src/`

## Current experimental pipeline

Two attack tracks, both trained on synthetic CARLA frames:

1. **YOLO track** — `src/yolo_chroma_attack/`. Loss = drop the confidence of
   the vehicle bbox that contains the patch centroid. Training on synthetic
   Town04+05 gives −18 % relative confidence drop, 18 % lost detections.
2. **CLIP track** — `src/clip_chroma_attack/`. Loss = pull the CLIP `[CLS]`
   embedding of the patched scene toward the `[CLS]` of the same scene with
   the leader truck removed. v1 (raw) and v2 (with Total Variation reg) both
   produce visually-noise patches that satisfy the loss but don't form
   semantic structure. v3 (planned July 2026) will use a DPT segmentation
   head to enforce pixel-level locality.

Closed-loop evaluation on 6 PCLA agents (1080 simulations, manual review of
180 valid runs): only SimLingo on Town04 shows a soft signal (+13 % net patch
attribution rate); all other agents are dominated by scenario-mismatch noise.

## Timeline

- **July 2026** — one-way-road dataset + velocity-based collision detection
  + CLIP attack v3 (crop-based or DPT pixel-level) + re-run closed-loop
- **August 2026** — thesis writing + defense preparation
- **September 2026** — defense at USI

## Language / Git conventions

- All code, comments, docstrings, notebooks, commit messages: **English only**.
- Internal agent docs (`.INIT_AGENT.md`, `.STATUS.md`, `.LOG.md`,
  `MEETING_NOTES.md`) stay in Italian.
- Sole author is Paolo Deidda — no co-author trailers.
- Push is manual (Paolo only).

## For coding agents

Start from `docs/.INIT_AGENT.md`, then read `docs/.STATUS.md` for current
state and `docs/.LOG.md` for the working history. `docs/THESIS_OVERVIEW.md`
gives the long-form context.
