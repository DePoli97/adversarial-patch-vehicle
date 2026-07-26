"""Record the exact inputs tfv6 receives during a real closed-loop run, and replay them.

Why this exists
---------------
Offline captures of a parked scene make tfv6 predict the 0 m/s bin with probability ~0.97
even on an empty road with no leader in sight, while the same checkpoint demonstrably
cruises at ~30 km/h in the closed loop. Something in a hand-built input dict differs from
the real one -- the known candidates being the JPEG-90 re-encode the agent applies before
inference, the route-planner target points (we hard-code them), the measured ego speed,
and vehicles held at their spawn offset with physics disabled.

That gap is not a cosmetic problem. If the baseline is already saturated at "stop" for a
reason unrelated to the truck, a patch trained to raise the predicted speed can succeed by
undoing *that* artefact, and will then do nothing once deployed -- which is exactly how the
YOLO-trained campaign failed, wearing a different costume.

The fix is to stop guessing. `record` hooks the real agent mid-run and dumps every tensor
the model actually consumed; `verify` re-runs our offline harness on those dumps and checks
it reproduces the recorded prediction. Once they agree, the recorded frames are also the
ideal training set, because their domain gap is zero by construction.

Usage
-----
    # 1. record, by running the normal closed-loop scenario under the hook
    python -m src.tfv6_chroma_attack.record_real_inputs record \
        --out data/tfv6_real_inputs/Town04_day \
        --every 5 \
        -- --agent tfv6_visiononly --town Town04 --light day --seed 0

    # 2. check our offline forward reproduces the recorded distribution
    python -m src.tfv6_chroma_attack.record_real_inputs verify \
        --dump data/tfv6_real_inputs/Town04_day
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SCENARIO = REPO / "src" / "carla_scenario" / "scenario_two_vehicles.py"
DEFAULT_PCLA_ROOT = Path("/home/vortex/PCLA")
CKPT_DIR = DEFAULT_PCLA_ROOT / "pcla_agents/transfuserv6_pretrained/visiononly_resnet34"

# Keys of the input dict that are small enough to store verbatim as JSON.
SCALAR_KEYS = (
    "speed",
    "command",
    "next_command",
    "target_point",
    "target_point_previous",
    "target_point_next",
    "town",
)


def _to_jsonable(v):
    import torch

    if isinstance(v, torch.Tensor):
        return {"__tensor__": v.detach().float().cpu().numpy().tolist(),
                "dtype": str(v.dtype), "shape": list(v.shape)}
    if isinstance(v, np.ndarray):
        return {"__array__": v.tolist(), "shape": list(v.shape)}
    return v


def install_hook(out_dir: Path, every: int) -> None:
    """Patch TFv6.forward so each call writes its inputs and key outputs to disk.

    The image is stored losslessly as PNG in the model's own channel order, so a replay
    can feed back byte-identical pixels. Everything else goes to a per-tick JSON.
    """
    import cv2
    import torch
    from lead.tfv6 import tfv6 as tfv6_mod

    out_dir.mkdir(parents=True, exist_ok=True)
    original = tfv6_mod.TFv6.forward
    state = {"calls": 0, "saved": 0}

    def hooked(self, data):
        out = original(self, data)
        n = state["calls"]
        state["calls"] = n + 1
        if n % every != 0:
            return out

        try:
            rgb = data["rgb"]
            # (B, 3, H, W) float in [0, 255] -> HWC uint8, batch element 0.
            img = rgb[0].detach().float().clamp(0, 255).byte().cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            stem = f"{n:06d}"
            # cv2 writes the array as-is; the replay reads it back with the same
            # convention, so no channel juggling is introduced here.
            cv2.imwrite(str(out_dir / f"{stem}.png"), img)

            rec = {"call_index": n, "rgb_shape": list(rgb.shape)}
            for k in SCALAR_KEYS:
                if k in data:
                    rec[k] = _to_jsonable(data[k])
            ts = getattr(out, "pred_target_speed_distribution", None)
            if ts is not None:
                logits = ts[0].detach().float().cpu()
                probs = torch.softmax(logits, dim=-1)
                rec["pred_target_speed_logits"] = logits.tolist()
                rec["pred_target_speed_probs"] = probs.tolist()
            wp = getattr(out, "pred_future_waypoints", None)
            if wp is not None:
                rec["pred_future_waypoints"] = wp[0].detach().float().cpu().tolist()
            bb = getattr(out, "pred_bounding_box", None)
            if bb is not None and getattr(bb, "center_heatmap_pred", None) is not None:
                rec["heat_max"] = float(bb.center_heatmap_pred.max())
            (out_dir / f"{stem}.json").write_text(json.dumps(rec, indent=1))
            state["saved"] += 1
        except Exception as exc:  # never let recording break the run
            print(f"[record_real_inputs] tick {n} dump failed: {exc}", file=sys.stderr)
        return out

    tfv6_mod.TFv6.forward = hooked
    print(f"[record_real_inputs] hook installed -> {out_dir} (every {every} calls)",
          flush=True)


def cmd_record(args, passthrough: list[str]) -> int:
    """Install the hook, then run the ordinary scenario script unchanged."""
    out_dir = Path(args.out)
    install_hook(out_dir, args.every)
    if not SCENARIO.exists():
        print(f"scenario not found: {SCENARIO}", file=sys.stderr)
        return 2
    sys.argv = [str(SCENARIO)] + passthrough
    print(f"[record_real_inputs] running {SCENARIO.name} {' '.join(passthrough)}",
          flush=True)
    runpy.run_path(str(SCENARIO), run_name="__main__")
    return 0


def cmd_verify(args) -> int:
    """Replay the dumps through our own model instance and compare predictions.

    A close match means the offline harness is faithful to the closed loop, and these
    frames can be used for training with no domain gap. A mismatch localises the problem
    to the input dict rather than the model.
    """
    import cv2
    import torch
    import torch.nn.functional as F
    from lead.training.config_training import TrainingConfig
    from lead.training.training_utils import create_model

    dump = Path(args.dump)
    metas = sorted(dump.glob("*.json"))
    if not metas:
        print(f"no dumps in {dump}", file=sys.stderr)
        return 2

    cfg = TrainingConfig(json.loads((Path(args.ckpt) / "config.json").read_text()))
    dev = torch.device(args.device)
    nets = []
    for i in range(args.members):
        net = create_model(cfg).to(dev).eval()
        net.load_state_dict(
            torch.load(Path(args.ckpt) / f"model_0030_{i}.pth", map_location=dev,
                       weights_only=True),
            strict=False,
        )
        for p in net.parameters():
            p.requires_grad_(False)
        nets.append(net)
    speeds = torch.tensor(cfg.target_speeds, device=dev)

    def tensor_from(rec, key, default):
        if key not in rec:
            return default
        v = rec[key]
        if isinstance(v, dict) and "__tensor__" in v:
            return torch.tensor(v["__tensor__"], device=dev, dtype=torch.float32)
        return default

    print(f"{'frame':>10} {'recorded E':>11} {'replay E':>10} {'dE':>8} "
          f"{'rec P0':>8} {'rep P0':>8}")
    dE, dP = [], []
    for meta in metas[: args.limit]:
        rec = json.loads(meta.read_text())
        img = cv2.imread(str(meta.with_suffix(".png")), cv2.IMREAD_COLOR)
        if img is None:
            continue
        rgb = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        rgb = rgb.to(dev, dtype=torch.float32)[None]

        data = {
            "rgb": rgb,
            "radar": torch.zeros(1, 300, 5, device=dev),
            "speed": tensor_from(rec, "speed", torch.zeros(1, device=dev)).reshape(-1),
            "command": tensor_from(
                rec, "command", F.one_hot(torch.tensor([3]), 6).float().to(dev)
            ).reshape(1, -1),
            "next_command": tensor_from(
                rec, "next_command", F.one_hot(torch.tensor([3]), 6).float().to(dev)
            ).reshape(1, -1),
            "target_point": tensor_from(
                rec, "target_point", torch.zeros(1, 2, device=dev)).reshape(1, 2),
            "target_point_previous": tensor_from(
                rec, "target_point_previous", torch.zeros(1, 2, device=dev)).reshape(1, 2),
            "target_point_next": tensor_from(
                rec, "target_point_next", torch.zeros(1, 2, device=dev)).reshape(1, 2),
            "town": rec.get("town", ["Town04"]),
        }
        if isinstance(data["town"], str):
            data["town"] = [data["town"]]

        logits = []
        with torch.no_grad(), torch.amp.autocast(
            device_type="cuda", dtype=cfg.torch_float_type,
            enabled=cfg.use_mixed_precision_training,
        ):
            for net in nets:
                logits.append(net(data).pred_target_speed_distribution.float())
        probs = torch.softmax(torch.stack(logits).mean(0), dim=-1)[0]
        rep_e = float((probs * speeds).sum())
        rep_p0 = float(probs[0])

        rec_probs = rec.get("pred_target_speed_probs")
        if rec_probs is None:
            continue
        rec_probs_t = torch.tensor(rec_probs, device=dev)
        rec_e = float((rec_probs_t * speeds).sum())
        rec_p0 = float(rec_probs_t[0])
        dE.append(abs(rep_e - rec_e))
        dP.append(abs(rep_p0 - rec_p0))
        print(f"{meta.stem:>10} {rec_e:11.3f} {rep_e:10.3f} {rep_e-rec_e:8.3f} "
              f"{rec_p0:8.4f} {rep_p0:8.4f}")

    if dE:
        print(f"\nframes compared: {len(dE)}")
        print(f"mean |dE| = {np.mean(dE):.4f} m/s   max |dE| = {np.max(dE):.4f}")
        print(f"mean |dP0| = {np.mean(dP):.5f}      max |dP0| = {np.max(dP):.5f}")
        faithful = np.max(dE) < 0.5 and np.max(dP) < 0.05
        print("REPLAY_FAITHFUL" if faithful else
              "REPLAY_MISMATCH (the offline input dict differs from the real one)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("record", help="run the scenario under a dumping hook")
    pr.add_argument("--out", required=True)
    pr.add_argument("--every", type=int, default=5,
                    help="dump one in every N model calls")

    pv = sub.add_parser("verify", help="replay dumps and compare against the recording")
    pv.add_argument("--dump", required=True)
    pv.add_argument("--ckpt", default=str(CKPT_DIR))
    pv.add_argument("--device", default="cuda:0")
    pv.add_argument("--members", type=int, default=3)
    pv.add_argument("--limit", type=int, default=20)

    argv = sys.argv[1:]
    passthrough: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, passthrough = argv[:i], argv[i + 1:]
    args = p.parse_args(argv)

    if args.cmd == "record":
        return cmd_record(args, passthrough)
    return cmd_verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
