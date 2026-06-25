"""Targeted cosine-distance loss against a frozen OpenCLIP image encoder.

Given an image with the trainable patch composited in, we compute its CLIP
image embedding (the [CLS] token after the projection head) and compare it
against the embedding of the *target* image — the same scene with the leader
truck removed. Minimizing `1 - cos(e_patched, e_target)` drives the patched
embedding toward the no-leader embedding, i.e. fools CLIP into "not seeing"
the truck even when the patch is composited onto its rear window.

The model is created via `open_clip.create_model_and_transforms(...)` and kept
in eval / no_grad except for the gradient path through the input image. Only
the image tower is used (no text encoding is involved in this attack).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# Default CLIP normalization (OpenAI/OpenCLIP convention).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


def tv_loss(patch: torch.Tensor) -> torch.Tensor:
    """Anisotropic L1 Total Variation regularization on the patch tensor.

    `patch` shape is (3, H, W) in [0, 1]. Penalizes adjacent-pixel differences
    along H and W axes — forces the patch toward smooth, low-frequency content
    instead of high-frequency adversarial noise.

    This is the classic regularizer used in Brown et al. 2017 ("Adversarial
    Patch") and downstream physical-adversarial-patch papers. Without it the
    cosine-distance loss alone tends to find the "noise shortcut": a high-
    frequency perturbation that shifts CLIP's [CLS] statistics without learning
    any semantic pattern.

    Returns a scalar tensor.
    """
    diff_h = (patch[..., 1:, :] - patch[..., :-1, :]).abs().mean()
    diff_w = (patch[..., :, 1:] - patch[..., :, :-1]).abs().mean()
    return diff_h + diff_w


class ClipTargetedLoss:
    """Wraps a frozen OpenCLIP image encoder + cosine-distance loss.

    Args:
        model_name : OpenCLIP model name (e.g. 'ViT-B-32', 'ViT-B-16').
        pretrained : pretrained tag (e.g. 'laion2b_s34b_b79k').
        device     : torch device.
        image_size : CLIP input size; must match the encoder. Defaults to 224.

    Call:
        loss, info = clip_loss(patched_image, target_image)
            patched_image : (B, 3, H, W) in [0, 1], REQUIRES_GRAD via patch
            target_image  : (B, 3, H, W) in [0, 1], frozen — no grad path
        Returns:
            loss   : scalar tensor (mean over batch)
            info   : {'loss', 'cos_mean', 'cos_min', 'cos_max'}
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | torch.device = "cuda",
        image_size: int = 224,
    ):
        import open_clip
        self.device = torch.device(device)
        self.image_size = image_size
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=str(self.device)
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        std  = torch.tensor(CLIP_STD,  device=self.device).view(1, 3, 1, 1)
        self.register_normalization(mean, std)

    def register_normalization(self, mean: torch.Tensor, std: torch.Tensor):
        self._mean = mean
        self._std  = std

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CLIP normalization. Input in [0, 1], output ~zero-mean/unit-std."""
        return (x - self._mean) / self._std

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run the image tower and L2-normalize the resulting [CLS] embedding.

        Input shape (B, 3, H, W) in [0, 1]. We resize-if-needed to CLIP's native
        input size with bilinear interpolation. L2 normalization makes the
        cosine similarity equivalent to a dot product, which keeps the loss
        well-conditioned regardless of embedding magnitude.
        """
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                              mode="bilinear", align_corners=False, antialias=True)
        x = self._normalize(x)
        feat = self.model.encode_image(x)
        return F.normalize(feat, dim=-1)

    def __call__(
        self,
        patched_image: torch.Tensor,
        target_image:  torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        e_patched = self._encode(patched_image)
        with torch.no_grad():
            e_target = self._encode(target_image)
        cos = (e_patched * e_target).sum(dim=-1)
        loss = (1.0 - cos).mean()
        info = {
            "loss":     float(loss.detach().cpu()),
            "cos_mean": float(cos.detach().mean().cpu()),
            "cos_min":  float(cos.detach().min().cpu()),
            "cos_max":  float(cos.detach().max().cpu()),
        }
        return loss, info
