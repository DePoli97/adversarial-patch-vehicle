"""Adversarial 'hide' loss against YOLOv8 (Ultralytics), filtered by target bbox.

We attack the leader vehicle (carlacola) wearing the chroma-key patch. The
target_bbox passed in from the Dataset is the expanded chroma-key region — an
approximation of the carlacola's bounding box in the image.

Loss design
-----------
YOLOv8 returns predictions of shape (B, 4+K, A) where K=80 (COCO classes) and
A is the number of anchors. Channels:
  [0..3]: cx, cy, w, h in pixels (after model post-processing, in input resolution)
  [4..]:  per-class probabilities (already sigmoid'd in inference mode)

For "hide", we:
  1. Filter anchors whose center (cx, cy) falls inside target_bbox
  2. Pick vehicle classes (car=2, truck=7, bus=5)
  3. Take max class score across vehicle classes per anchor
  4. Loss = mean of those max scores → minimizing pushes them to 0 = "no vehicle here"

If no anchors fall inside the target_bbox for a given image we contribute 0
(rare; happens only if the patch warp deformed the target completely).
"""
from __future__ import annotations

import torch
from ultralytics import YOLO


# COCO class indices for vehicles. Carlacola → "truck" (7) primarily, but
# detector can also call it car (2) depending on view; include both + bus (5).
VEHICLE_CLASSES = (2, 5, 7)


class YoloHideLoss:
    """Adversarial 'hide' loss against a pretrained YOLOv8 model.

    Args:
        weights: path to a YOLOv8 .pt file. If 'yolov8n.pt' / 'yolov8s.pt', etc.,
                 Ultralytics will auto-download.
        device: 'cuda' or 'cpu'.
        vehicle_classes: COCO class indices considered "leader" candidates.
        conf_aggregation: 'mean' (mean over matching anchors), 'max' (peak),
                          or 'topk' (mean of top-K).
        topk: K when conf_aggregation='topk'.
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        device: str = "cuda",
        vehicle_classes: tuple[int, ...] = VEHICLE_CLASSES,
        conf_aggregation: str = "topk",
        topk: int = 10,
        margin_tau: float = 0.0,
    ):
        self.device = torch.device(device)
        # Load via Ultralytics wrapper, keep underlying nn.Module for forward.
        self.yolo = YOLO(weights)
        self.model = self.yolo.model.to(self.device).eval()
        # Freeze YOLO; gradients flow through inputs only.
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.vehicle_classes = vehicle_classes
        self.conf_aggregation = conf_aggregation
        self.topk = topk
        # Margin/hinge threshold (C&W-style). When > 0, the loss becomes
        # relu(conf - tau): once a frame's confidence is pushed below the
        # detection threshold tau (~0.2), it is already "hidden" and stops
        # contributing gradient. This avoids wasting capacity driving an
        # already-invisible target to exactly 0, and prevents strong-detection
        # frames from being dominated by the flat tail of easy frames.
        self.margin_tau = margin_tau

    def __call__(
        self,
        image: torch.Tensor,        # (B, 3, H, W) in [0, 1]
        target_bbox: torch.Tensor,  # (B, 4) (x1, y1, x2, y2) in image pixels
    ) -> tuple[torch.Tensor, dict]:
        """Compute hide loss. Returns (loss, info_dict)."""
        # YOLO expects 0-1 float (Ultralytics handles normalization internally
        # when fed via its predictor; using the raw nn.Module skips that, so
        # passing [0, 1] directly is what the network was trained on).
        preds = self.model(image)  # tuple, first element is the detection head out
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        # preds: (B, 4+K, A) where A varies with input size (8400 at 640)
        B = preds.shape[0]
        nc = preds.shape[1] - 4
        if nc < max(self.vehicle_classes) + 1:
            raise ValueError(f"YOLO has {nc} classes, can't index {self.vehicle_classes}")

        # cx, cy: (B, A)
        cx = preds[:, 0, :]
        cy = preds[:, 1, :]
        # class scores: (B, K, A) for vehicle classes only
        cls_idx = torch.tensor(self.vehicle_classes, device=image.device)
        veh_scores = preds[:, 4:, :].index_select(dim=1, index=cls_idx)  # (B, V, A)
        # Per anchor, take max over vehicle classes
        veh_max, _ = veh_scores.max(dim=1)  # (B, A)

        # Per-image: mask anchors whose center is inside the target bbox.
        x1 = target_bbox[:, 0].unsqueeze(1)  # (B, 1)
        y1 = target_bbox[:, 1].unsqueeze(1)
        x2 = target_bbox[:, 2].unsqueeze(1)
        y2 = target_bbox[:, 3].unsqueeze(1)
        in_box = (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)  # (B, A) bool

        losses = []
        n_matched = []
        for b in range(B):
            mask_b = in_box[b]
            if not mask_b.any():
                losses.append(torch.zeros((), device=image.device))
                n_matched.append(0)
                continue
            scores_b = veh_max[b][mask_b]  # (M_b,)
            # Margin/hinge: only penalize confidence above tau. relu keeps the
            # gradient alive while conf > tau and zeroes it once hidden.
            if self.margin_tau > 0.0:
                scores_b = torch.relu(scores_b - self.margin_tau)
            n_matched.append(int(mask_b.sum().item()))
            if self.conf_aggregation == "max":
                losses.append(scores_b.max())
            elif self.conf_aggregation == "topk":
                k = min(self.topk, scores_b.numel())
                top, _ = scores_b.topk(k)
                losses.append(top.mean())
            else:  # "mean"
                losses.append(scores_b.mean())

        loss = torch.stack(losses).mean()
        info = {
            "loss": float(loss.detach().item()),
            "n_matched_anchors": n_matched,
            "veh_max_in_box_mean": float(veh_max[in_box].mean().item())
                if in_box.any() else 0.0,
        }
        return loss, info
