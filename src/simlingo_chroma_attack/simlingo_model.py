"""Standalone, differentiable wrapper around PCLA's SimLingo `DrivingModel`.

Everything that couples this package to the checkpoint on disk lives here, so
`simlingo_loss.py` stays a pure loss.

Why a wrapper is needed at all
------------------------------
1. `agent_simlingo.py` builds its model inside the CARLA leaderboard agent
   lifecycle and calls it under `@torch.no_grad()`. We need the same weights
   without CARLA and with autograd enabled.
2. The agent's image preprocessing goes through PIL
   (`internvl2_utils.build_transform` + `dynamic_preprocess`), which is not
   differentiable. `preprocess()` below is a torch re-implementation of the
   exact same chain.
3. `DrivingModel.forward` has two branches and BOTH are unusable as-is:
     * `predict_language=True` (the deployed one) runs
       `LLM.greedy_sample(..., max_new_tokens=100)` — discrete argmax sampling,
       no gradient path.
     * `predict_language=False` is differentiable but *crashes upstream*:
       driving.py:179 does `features = self.forward_model(...)` while
       `forward_model` returns the tuple `(adaptor_features, adaptor_logits)`,
       so the next line indexes a tuple with a tuple. Verified live on vortex.
   `predict()` therefore reimplements that branch correctly in three lines
   (adaptors -> forward_model -> get_predictions). It is also cheaper than
   `DrivingModel.forward`, which calls `replace_placeholder_tokens` twice and
   so runs the ViT twice.

Surrogate fidelity
------------------
`predict_language=False` is a SURROGATE for the deployed path: the deployed
path first greedily decodes a commentary string and conditions the driving
queries on those decoded tokens. Measured on a random frame, surrogate vs
deployed gave desired_speed 34.99 vs 37.46 km/h (max per-waypoint difference
1.88 m). Close, and the only differentiable option, but patches trained here
must still be validated against the deployed path (and ultimately closed loop).

Geometry
--------
SimLingo drives off ONE camera (`config_simlingo.py:53` -> `num_cameras = [0]`):
1024x512, FOV 110, mounted at x=-1.5, y=0, z=2.0. The preprocessing chain is

    1024x512 raw
      -> drop the bottom 4.8/16 of the rows          -> 1024x359
      -> resize to 896x448 (PIL bicubic)
      -> split along width into two 448x448 tiles    (dynamic_preprocess,
                                                      max_num=2, no thumbnail
                                                      because the checkpoint's
                                                      hydra config has
                                                      use_global_img: false)
      -> ToTensor + ImageNet normalise
    => camera_images (B, T=1, NP=2, 3, 448, 448)

That is a different rig from the Fase-1 chroma-key capture (1280x720, FOV 90).
Feeding those frames in works numerically but is geometrically wrong; see
README.md.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------- constants
# Vortex paths. Override with the PCLA_ROOT env var or the constructor arg.
DEFAULT_PCLA_ROOT = Path(os.environ.get("PCLA_ROOT", "/home/vortex/PCLA"))
DEFAULT_CKPT_REL = Path("pcla_agents/simlingo_pretrained/checkpoints/epoch=013.ckpt/pytorch_model.pt")
DEFAULT_HYDRA_REL = Path("pcla_agents/simlingo_pretrained/.hydra/config.yaml")

# SimLingo's native camera (config_simlingo.py:58-60).
CAM_H, CAM_W = 512, 1024
# Fraction of rows dropped from the bottom, from the agent's tick():
#   rgb[:int(H - (H * 4.8) // 16)]
BOTTOM_CROP_NUM, BOTTOM_CROP_DEN = 4.8, 16
TILE = 448
N_TILES = 2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Waypoint spacing: wp_freq 5 / carla_fps 20 (driving.py:431-432).
WP_DT = 0.25
# Indices control_pid() differences (agent_simlingo.py:858-861):
#   one_second = carla_fps // (wp_dilation * data_save_freq) = 20 // (1*5) = 4
#   half_second = one_second // 2 = 2
#   desired_speed = ||wp[half_second-2] - wp[one_second-2]|| * 2.0
PID_IDX_A, PID_IDX_B = 0, 2
PID_GAIN = 2.0
# agent_simlingo.control_pid brake rule (config_simlingo.py:21-23).
BRAKE_SPEED = 0.4
BRAKE_RATIO = 1.1

TARGET_POINT_TOKEN = "<TARGET_POINT>"
EXTRA_SPECIAL_TOKENS = [
    "<WAYPOINTS>", "<WAYPOINTS_DIFF>", "<ORG_WAYPOINTS_DIFF>", "<ORG_WAYPOINTS>",
    "<WAYPOINT_LAST>", "<ROUTE>", "<ROUTE_DIFF>", TARGET_POINT_TOKEN,
]


def desired_speed_from_wps(speed_wps: torch.Tensor) -> torch.Tensor:
    """The scalar `agent_simlingo.control_pid` turns into throttle, in m/s.

    `speed_wps` is (B, 10, 2) ego-relative metres, x forward / y left.
    Returns (B,).
    """
    w = speed_wps.float()
    return (w[:, PID_IDX_A] - w[:, PID_IDX_B]).norm(dim=-1) * PID_GAIN


def finite_difference_speeds(speed_wps: torch.Tensor) -> torch.Tensor:
    """Per-step longitudinal speed implied by the waypoints, (B, 9) in m/s.

    This is driving.py:540-542 (`diff(pred_wps_1d[:, 0]) / (wp_freq/carla_fps)`).
    """
    return speed_wps.float()[..., 0].diff(dim=-1) / WP_DT


def mean_implied_speed(speed_wps: torch.Tensor) -> torch.Tensor:
    """Average speed implied by EVERY waypoint, (B,) in m/s.

    Waypoint k lies (k+1)*WP_DT seconds ahead, so `x_k / ((k+1) * WP_DT)` is the
    mean speed needed to reach it. Averaging over all 10 gives a dense signal
    (all ten heads receive gradient) in the same units as `desired_speed`,
    unlike `desired_speed` which only touches waypoints 0 and 2.
    """
    w = speed_wps.float()
    t = torch.arange(1, w.shape[1] + 1, device=w.device, dtype=w.dtype) * WP_DT
    return (w[..., 0] / t).mean(dim=-1)


def would_brake(desired_speed: torch.Tensor, ego_speed_ms: float) -> torch.Tensor:
    """Replicates control_pid's brake predicate. Returns a (B,) bool tensor."""
    safe = desired_speed.clamp_min(1e-6)
    return (desired_speed < BRAKE_SPEED) | ((ego_speed_ms / safe) > BRAKE_RATIO)


class SimlingoWrapper:
    """Loads SimLingo once and exposes a differentiable image -> waypoints call.

    Args:
        pcla_root: PCLA checkout (must contain `pcla_agents/simlingo` and the
                   `pretrained/<variant>` folder with `conversation.py`).
        ckpt: path to `pytorch_model.pt`. Defaults to the shipped epoch=013.
        hydra_cfg: path to the training `.hydra/config.yaml`. Defaults to the
                   one three levels above the checkpoint, mirroring
                   agent_simlingo.py:159.
        device: 'cuda' or 'cpu' (cpu is impractically slow, kept for debugging).
        dtype: model dtype. bfloat16 is what the agent uses.
    """

    def __init__(
        self,
        pcla_root: str | Path = DEFAULT_PCLA_ROOT,
        ckpt: str | Path | None = None,
        hydra_cfg: str | Path | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        verbose: bool = True,
    ):
        self.pcla_root = Path(pcla_root)
        self.device = torch.device(device)
        self.dtype = dtype
        ckpt = Path(ckpt) if ckpt is not None else self.pcla_root / DEFAULT_CKPT_REL
        hydra_cfg = Path(hydra_cfg) if hydra_cfg is not None else self.pcla_root / DEFAULT_HYDRA_REL
        if not ckpt.exists():
            raise FileNotFoundError(f"SimLingo checkpoint not found: {ckpt}")
        if not hydra_cfg.exists():
            raise FileNotFoundError(f"SimLingo hydra config not found: {hydra_cfg}")

        self._add_import_paths()
        import hydra
        from omegaconf import OmegaConf
        from transformers import AutoConfig, AutoProcessor

        cfg = OmegaConf.load(hydra_cfg)
        # agent_simlingo.py:163 — the encoder needs the data module's flag.
        cfg.model.vision_model.use_global_img = cfg.data_module.use_global_img
        if cfg.data_module.use_global_img:
            raise NotImplementedError(
                "This checkpoint sets use_global_img=True, which appends a "
                "thumbnail tile. preprocess() assumes the 2-tile layout of the "
                "shipped checkpoint (use_global_img: false).")
        self.cfg = cfg
        self.variant = cfg.model.vision_model.variant

        processor = AutoProcessor.from_pretrained(self.variant, trust_remote_code=True)
        tokenizer = processor.tokenizer if "tokenizer" in processor.__dict__ else processor
        tokenizer.add_special_tokens({"additional_special_tokens": EXTRA_SPECIAL_TOKENS})
        tokenizer.padding_side = "left"
        self.processor, self.tokenizer = processor, tokenizer

        cache_dir = f"pretrained/{self.variant.split('/')[1]}"
        # The model is built under a bfloat16 default dtype, exactly like the
        # agent (agent_simlingo.py:170-180); restoring the previous default
        # afterwards matters because it is process-global.
        prev_default = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            self.model = hydra.utils.instantiate(
                cfg.model, cfg_data_module=cfg.data_module, processor=processor,
                cache_dir=cache_dir, _recursive_=False).to(self.device)
        finally:
            torch.set_default_dtype(prev_default)
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        # Force the differentiable branch. `predict()` bypasses forward()
        # anyway, but this keeps the flag honest if anyone calls the model.
        self.model.predict_language = False

        # Conversation template, loaded from the local snapshot the agent uses.
        conv_path = self.pcla_root / cache_dir / "conversation.py"
        if not conv_path.exists():
            raise FileNotFoundError(
                f"{conv_path} missing — the agent downloads it via "
                f"snapshot_download(repo_id='{self.variant}')")
        spec = importlib.util.spec_from_file_location("get_conv_template", conv_path)
        self._conv = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("get_conv_template", self._conv)
        spec.loader.exec_module(self._conv)

        hf_cfg = AutoConfig.from_pretrained(self.variant, trust_remote_code=True)
        img_sz = hf_cfg.force_image_size or hf_cfg.vision_config.image_size
        self.num_image_token = int((img_sz // hf_cfg.vision_config.patch_size) ** 2
                                   * (hf_cfg.downsample_ratio ** 2))

        self._mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 1, 3, 1, 1)
        self._prompt_cache: dict[tuple, torch.Tensor] = {}
        self._warned_size = False

        if verbose:
            n = sum(p.numel() for p in self.model.parameters())
            mem = (f"  {torch.cuda.memory_allocated()/2**30:.2f} GiB on device"
                   if self.device.type == "cuda" else "")
            print(f"[SimlingoWrapper] {self.variant}  {n/1e6:.1f}M params{mem}")

    # ------------------------------------------------------------------ setup
    def _add_import_paths(self):
        """`simlingo_training.*` resolves from PCLA root, `custom_types` from the
        agent folder — agent_simlingo.py relies on both being importable."""
        for p in (self.pcla_root, self.pcla_root / "pcla_agents" / "simlingo"):
            sp = str(p)
            if sp not in sys.path:
                sys.path.insert(0, sp)

    # ------------------------------------------------------------ preprocess
    def preprocess(self, rgb01: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) in [0, 1] -> (B, 1, 2, 3, 448, 448), differentiable.

        Torch re-implementation of the agent's PIL chain. `antialias=True`
        matches PIL's bicubic behaviour on the downscaling axis; it is the
        closest differentiable equivalent, not a bit-exact one.

        Not modelled here (deliberately — these belong in EoT, not in the
        deterministic preprocessing): the JPEG quality round-trip the agent
        applies in tick() to mimic its training data.
        """
        if rgb01.dim() != 4 or rgb01.shape[1] != 3:
            raise ValueError(f"expected (B, 3, H, W), got {tuple(rgb01.shape)}")
        if not self._warned_size and rgb01.shape[-2:] != (CAM_H, CAM_W):
            self._warned_size = True
            print(f"[SimlingoWrapper] WARNING: input is {tuple(rgb01.shape[-2:])}, "
                  f"SimLingo's camera is {(CAM_H, CAM_W)}. The chain still runs "
                  f"but the geometry no longer matches the deployed rig.")
        h = rgb01.shape[-2]
        keep = int(h - (h * BOTTOM_CROP_NUM) // BOTTOM_CROP_DEN)
        x = rgb01[..., :keep, :]
        x = F.interpolate(x, size=(TILE, TILE * N_TILES), mode="bicubic",
                          align_corners=False, antialias=True).clamp(0.0, 1.0)
        tiles = torch.stack([x[..., i * TILE:(i + 1) * TILE] for i in range(N_TILES)], dim=1)
        tiles = (tiles - self._mean) / self._std
        return tiles.unsqueeze(1).to(self.dtype)          # (B, T=1, NP, C, H, W)

    # ---------------------------------------------------------------- prompt
    def _prompt_ids(self, batch: int, speed_ms: float) -> torch.Tensor:
        """Tokenised prompt, cached. Mirrors agent tick() with the shipped
        settings (`eval_route_as='target_point'`, `use_cot=True`, no user flag,
        no custom prompt)."""
        speed_r = round(float(speed_ms), 1)
        key = (batch, speed_r)
        if key in self._prompt_cache:
            return self._prompt_cache[key]

        text = (f"Current speed: {speed_r} m/s. "
                f"Target waypoint: {TARGET_POINT_TOKEN}{TARGET_POINT_TOKEN}. "
                f"What should the ego do next?")
        tpl = self._conv.get_conv_template("internlm2-chat")
        tpl.append_message(tpl.roles[0], "<image>\n" + text)
        tpl.append_message(tpl.roles[1], None)     # assistant turn left open
        query = tpl.get_prompt()
        system = tpl.system_template.replace("{system_message}", tpl.system_message) + tpl.sep
        query = query.replace(system, "")
        img_tokens = ("<img>" + "<IMG_CONTEXT>" * self.num_image_token * N_TILES + "</img>")
        query = query.replace("<image>", img_tokens, 1)

        tok = self.tokenizer([query] * batch, padding=True, return_tensors="pt",
                             return_offsets_mapping=True, add_special_tokens=False)
        ids = tok["input_ids"].to(self.device)
        self._prompt_cache[key] = ids
        return ids

    def _language_label(self, batch: int, speed_ms: float,
                        target_point, next_target_point):
        from custom_types import LanguageLabel
        ids = self._prompt_ids(batch, speed_ms)
        valid = ids != self.tokenizer.pad_token_id
        tp = np.asarray([list(target_point), list(next_target_point)], dtype="float32")
        tok_id = self.tokenizer.convert_tokens_to_ids(TARGET_POINT_TOKEN)
        placeholders = [{tok_id: tp} for _ in range(batch)]
        return LanguageLabel(phrase_ids=ids, phrase_valid=valid, phrase_mask=valid,
                             placeholder_values=placeholders,
                             language_string=[""] * batch, loss_masking=None)

    def driving_input(self, rgb01: torch.Tensor, speed_ms: float = 8.0,
                      target_point=(20.0, 0.0), next_target_point=(40.0, 0.0)):
        """Assemble the `DrivingInput` namedtuple the model consumes.

        `speed_ms` goes into BOTH the prompt text and `vehicle_speed`; the
        prompt copy is what actually conditions the LLM.
        """
        from custom_types import DrivingInput
        b = rgb01.shape[0]
        ll = self._language_label(b, speed_ms, target_point, next_target_point)
        eye3 = torch.eye(3, device=self.device).view(1, 3, 3).expand(b, 3, 3)
        eye4 = torch.eye(4, device=self.device).view(1, 4, 4).expand(b, 4, 4)
        return DrivingInput(
            camera_images=self.preprocess(rgb01),
            image_sizes=None,                       # the agent also passes None
            camera_intrinsics=eye3,                 # unused by the driving heads
            camera_extrinsics=eye4,                 # unused by the driving heads
            vehicle_speed=torch.full((b, 1), float(speed_ms), device=self.device),
            target_point=torch.tensor([list(target_point)], device=self.device,
                                      dtype=torch.float32).expand(b, 2),
            prompt=ll, prompt_inference=ll,
        )

    # --------------------------------------------------------------- forward
    def predict(self, rgb01: torch.Tensor, speed_ms: float = 8.0,
                target_point=(20.0, 0.0), next_target_point=(40.0, 0.0)) -> dict:
        """image -> {'speed_wps': (B,10,2), 'route': (B,20,2)}, differentiable.

        This is `DrivingModel.forward`'s `predict_language=False` branch, fixed
        (see module docstring) and with the redundant second ViT pass removed.
        """
        di = self.driving_input(rgb01, speed_ms, target_point, next_target_point)
        adaptor_dict = self.model.adaptors(di, inference=True)
        features, _logits = self.model.forward_model(di, adaptor_dict)
        by_adaptor = self.model.adaptors.split_outputs_by_adaptor(adaptor_dict, features)
        return self.model.adaptors.driving.get_predictions(by_adaptor["driving"])

    @torch.no_grad()
    def predict_deployed(self, rgb01: torch.Tensor, speed_ms: float = 8.0,
                         target_point=(20.0, 0.0), next_target_point=(40.0, 0.0)):
        """The real inference path (`predict_language=True`), for validation only.

        Autoregressive and therefore NOT differentiable. Returns
        (speed_wps, route, language). Use it to check that a patch trained on
        the surrogate still moves the deployed output.
        """
        di = self.driving_input(rgb01, speed_ms, target_point, next_target_point)
        self.model.predict_language = True
        try:
            return self.model(di)
        finally:
            self.model.predict_language = False
