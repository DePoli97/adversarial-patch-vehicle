"""Adversarial patch trained against OpenCLIP ViT-B/32 image encoder.

Targeted attack: drive the embedding of the patched scene toward the embedding
of the same scene with the leader removed (no-leader version of the triplet).
CLIP weights are frozen — the only trainable tensor is the patch.

Reuses the differentiable warp + EOT from yolo_chroma_attack.
"""
