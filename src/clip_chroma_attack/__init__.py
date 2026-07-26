"""Adversarial patch trained against the OpenCLIP ViT-B/32 image encoder.

Two attacks share the same warp/EOT/dataset machinery and differ only in the
loss (`--loss` in train.py):

targeted (v1/v2) : GLOBAL objective — drive the [CLS] embedding of the patched
    scene toward the embedding of the same scene with the leader removed.
    Mode-collapses into high-frequency noise: a pooled vector is too easy to
    hijack.
crop (v3)        : LOCAL objective — crop the truck out of the composite,
    resize to CLIP's input, and score only that crop against vehicle/road text
    prompts (and optionally against the same crop of the no-leader frame).
    See docs/clip_attack_survey.md, Option A.

CLIP weights are frozen — the only trainable tensor is the patch.
Reuses the differentiable warp + EOT from yolo_chroma_attack.
"""
