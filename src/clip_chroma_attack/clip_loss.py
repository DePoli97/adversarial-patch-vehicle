"""Losses against a frozen OpenCLIP encoder.

Two formulations live here:

`ClipTargetedLoss` (v1/v2, runs 01-02) — GLOBAL objective. Given an image with
the trainable patch composited in, compute its CLIP image embedding (the [CLS]
token after the projection head) and compare it against the embedding of the
*target* image — the same scene with the leader truck removed. Minimizing
`1 - cos(e_patched, e_target)` was meant to fool CLIP into "not seeing" the
truck. It does move the cosine (0.81 -> 0.88) but the resulting patch is pure
high-frequency noise: a 512-d pooled vector is an easy shortcut, the optimizer
hijacks it instead of changing what the truck looks like. Kept unchanged so the
old runs stay reproducible.

`ClipCropLoss` (v3) — LOCAL objective, Option A of docs/clip_attack_survey.md.
The truck region is cropped out of the composited frame (box derived from the
marker quad), resized to CLIP's input and scored on its own. Two signals, both
usable at once:
  * text: push the crop embedding away from vehicle prompts and toward
    empty-road prompts (zero-shot vehicle-vs-road classification of the crop);
  * image: pull the crop embedding toward the SAME crop taken from the
    no-leader frame (the localized version of the v1/v2 target).
Multiple jittered crops per frame prevent overfitting one exact rectangle.

Models are created via `open_clip.create_model_and_transforms(...)` and kept in
eval with `requires_grad_(False)`; the only gradient path is through the input
image.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.clip_chroma_attack.crop_utils import (
    DEFAULT_EXPAND_X, DEFAULT_MARGIN_BOTTOM, DEFAULT_MARGIN_TOP,
    crop_resize, jitter_boxes, truck_box_from_quad,
)


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


# Zero-shot prompt sets for the crop classifier. The leader is the CARLA
# "carlacola" — a red delivery truck seen from behind — so the vehicle set
# mixes truck/van wordings; the negative set describes the road that should be
# there instead. Prompt ensembling (mean of the L2-normalized text embeddings,
# renormalized) is the standard CLIP zero-shot recipe.
VEHICLE_PROMPTS = (
    "a photo of a truck",
    "a photo of a delivery truck on the road",
    "a photo of the back of a truck",
    "a photo of a van on the road",
)
ROAD_PROMPTS = (
    "a photo of an empty road",
    "a photo of an empty asphalt road",
    "a photo of a road with no vehicles",
    "a photo of an empty street",
)


class ClipCropLoss:
    """Crop-localized CLIP loss (survey Option A).

    The frame is cropped around the truck, resized to CLIP's input and scored
    there, so the patch can no longer win by perturbing a global pooled vector:
    inside the crop it occupies a large, fixed fraction of the pixels and has
    to actually change what the region looks like.

    Sign convention (the whole point of this class — do not flip it silently):
        text term  = relu(cos(crop, VEHICLE) - cos(crop, ROAD) + margin)
    We MINIMIZE it, i.e. we push the crop to look *less* like a vehicle than
    like an empty road. It follows that a real truck crop must score HIGHER on
    this term than an empty-road crop; `test_crop_sign.py` asserts exactly that
    on real frames before any training is trusted.

    Args:
        model_name / pretrained / device / image_size : frozen OpenCLIP tower.
        mode        : 'text' (prompts only, no no-leader frame needed),
                      'image' (pull toward the no-leader crop),
                      'both'  (weighted sum).
        n_crops     : jittered crops sampled per frame per step.
        crop_scale  : (lo, hi) multiplicative jitter on the box side.
        crop_shift  : centre jitter, as a fraction of the box side.
        expand_x / margin_top / margin_bottom / square_crop :
                      marker-quad -> truck-box geometry, see crop_utils.
        text_objective : 'margin' (hinge on raw cosines, well conditioned) or
                      'prob' (softmax vehicle probability; saturates because
                      CLIP's logit_scale is ~100, kept for diagnostics).
        text_margin : hinge margin on the cosine difference.
        w_text / w_image : weight of each term (both applied whenever the
                      corresponding term is active, not only in mode='both').

    Call:
        loss, info = clip_loss(image, corners=..., ref_image=...)
            image     : (B, 3, H, W) in [0, 1], REQUIRES_GRAD via the patch
            corners   : (B, 4, 2) marker quad in pixels (or pass target_bbox)
            ref_image : (B, 3, H, W) no-leader frame, needed for 'image'/'both'
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | torch.device = "cuda",
        image_size: int = 224,
        mode: str = "text",
        n_crops: int = 3,
        crop_scale: tuple[float, float] = (0.85, 1.25),
        crop_shift: float = 0.08,
        expand_x: float = DEFAULT_EXPAND_X,
        margin_top: float = DEFAULT_MARGIN_TOP,
        margin_bottom: float = DEFAULT_MARGIN_BOTTOM,
        square_crop: bool = True,
        vehicle_prompts: tuple[str, ...] = VEHICLE_PROMPTS,
        road_prompts: tuple[str, ...] = ROAD_PROMPTS,
        text_objective: str = "margin",
        text_margin: float = 0.10,
        w_text: float = 1.0,
        w_image: float = 1.0,
        padding_mode: str = "border",
    ):
        import open_clip
        if mode not in ("text", "image", "both"):
            raise ValueError(f"Unknown mode: {mode}")
        if text_objective not in ("margin", "prob"):
            raise ValueError(f"Unknown text_objective: {text_objective}")

        self.device = torch.device(device)
        self.image_size = image_size
        self.mode = mode
        self.n_crops = n_crops
        self.crop_scale = tuple(crop_scale)
        self.crop_shift = crop_shift
        self.expand_x = expand_x
        self.margin_top = margin_top
        self.margin_bottom = margin_bottom
        self.square_crop = square_crop
        self.text_objective = text_objective
        self.text_margin = text_margin
        self.w_text = w_text
        self.w_image = w_image
        self.padding_mode = padding_mode

        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=str(self.device)
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self._mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std  = torch.tensor(CLIP_STD,  device=self.device).view(1, 3, 1, 1)

        tokenizer = open_clip.get_tokenizer(model_name)
        self.vehicle_prompts = tuple(vehicle_prompts)
        self.road_prompts = tuple(road_prompts)
        self.t_vehicle = self._encode_prompts(tokenizer, self.vehicle_prompts)
        self.t_road    = self._encode_prompts(tokenizer, self.road_prompts)
        self.logit_scale = float(self.model.logit_scale.detach().exp().cpu())

    # ---------------------------------------------------------------- encoders
    def _encode_prompts(self, tokenizer, prompts: tuple[str, ...]) -> torch.Tensor:
        """Prompt-ensemble a group of texts into one L2-normalized (D,) vector."""
        with torch.no_grad():
            tok = tokenizer(list(prompts)).to(self.device)
            emb = F.normalize(self.model.encode_text(tok).float(), dim=-1)
            return F.normalize(emb.mean(dim=0), dim=-1)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CLIP normalization. Input in [0, 1]."""
        return (x - self._mean) / self._std

    def encode_crops(self, crops: torch.Tensor) -> torch.Tensor:
        """(N, 3, S, S) in [0, 1] -> (N, D) L2-normalized image embeddings."""
        if crops.shape[-1] != self.image_size or crops.shape[-2] != self.image_size:
            crops = F.interpolate(crops, size=(self.image_size, self.image_size),
                                  mode="bilinear", align_corners=False, antialias=True)
        feat = self.model.encode_image(self._normalize(crops)).float()
        return F.normalize(feat, dim=-1)

    # ------------------------------------------------------------------ scoring
    def score_embeddings(self, emb: torch.Tensor) -> dict:
        """Zero-shot vehicle-vs-road scores for (N, D) normalized embeddings.

        Returns tensors (all (N,)): `cos_veh`, `cos_road`, `p_vehicle`
        (softmax over the two cosines at CLIP's own logit scale) and
        `text_term`, the per-crop quantity the attack minimizes.
        """
        cos_veh  = emb @ self.t_vehicle
        cos_road = emb @ self.t_road
        logits = torch.stack([cos_veh, cos_road], dim=-1) * self.logit_scale
        p_vehicle = logits.softmax(dim=-1)[..., 0]
        if self.text_objective == "margin":
            text_term = F.relu(cos_veh - cos_road + self.text_margin)
        else:
            text_term = p_vehicle
        return {"cos_veh": cos_veh, "cos_road": cos_road,
                "p_vehicle": p_vehicle, "text_term": text_term}

    def score_crops(self, crops: torch.Tensor) -> dict:
        """Convenience: encode (N, 3, S, S) crops and score them."""
        return self.score_embeddings(self.encode_crops(crops))

    # ------------------------------------------------------------------ boxes
    def boxes_for(
        self,
        image_hw: tuple[int, int],
        corners: torch.Tensor | None = None,
        target_bbox: torch.Tensor | None = None,
        jitter: bool = True,
    ) -> torch.Tensor:
        """Truck boxes for a batch, (B, n_crops, 4). `jitter=False` -> (B, 1, 4)."""
        if target_bbox is None:
            if corners is None:
                raise ValueError("Pass either corners (B,4,2) or target_bbox (B,4).")
            target_bbox = truck_box_from_quad(
                corners, image_hw, expand_x=self.expand_x,
                margin_top=self.margin_top, margin_bottom=self.margin_bottom,
                square=self.square_crop,
            )
        if not jitter:
            return jitter_boxes(target_bbox, image_hw, n_crops=1)
        return jitter_boxes(target_bbox, image_hw, n_crops=self.n_crops,
                            scale_range=self.crop_scale, shift_frac=self.crop_shift)

    # ------------------------------------------------------------------- call
    def __call__(
        self,
        image: torch.Tensor,
        corners: torch.Tensor | None = None,
        target_bbox: torch.Tensor | None = None,
        ref_image: torch.Tensor | None = None,
        jitter: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        H, W = image.shape[-2], image.shape[-1]
        boxes = self.boxes_for((H, W), corners=corners,
                               target_bbox=target_bbox, jitter=jitter)
        B, n = boxes.shape[0], boxes.shape[1]

        crops = crop_resize(image, boxes, out_size=self.image_size,
                            padding_mode=self.padding_mode)      # (B, n, 3, S, S)
        emb = self.encode_crops(crops.flatten(0, 1))             # (B*n, D)
        sc = self.score_embeddings(emb)

        info = {
            "cos_veh":  float(sc["cos_veh"].detach().mean().cpu()),
            "cos_road": float(sc["cos_road"].detach().mean().cpu()),
            "p_veh":    float(sc["p_vehicle"].detach().mean().cpu()),
            # fraction of crops that CLIP already reads as road, not vehicle
            "road_rate": float((sc["cos_road"] > sc["cos_veh"])
                               .float().detach().mean().cpu()),
            "n_crops": B * n,
        }

        loss = torch.zeros((), device=image.device)
        if self.mode in ("text", "both"):
            loss_text = sc["text_term"].mean()
            loss = loss + self.w_text * loss_text
            info["loss_text"] = float(loss_text.detach().cpu())
        if self.mode in ("image", "both"):
            if ref_image is None:
                raise ValueError(f"mode='{self.mode}' needs ref_image (no-leader frame).")
            with torch.no_grad():
                ref_crops = crop_resize(ref_image, boxes, out_size=self.image_size,
                                        padding_mode=self.padding_mode)
                emb_ref = self.encode_crops(ref_crops.flatten(0, 1))
            cos_ref = (emb * emb_ref).sum(dim=-1)
            loss_img = (1.0 - cos_ref).mean()
            loss = loss + self.w_image * loss_img
            info["loss_image"] = float(loss_img.detach().cpu())
            info["cos_ref"] = float(cos_ref.detach().mean().cpu())

        info["loss"] = float(loss.detach().cpu())
        return loss, info
