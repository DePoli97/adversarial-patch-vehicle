"""
Detection probe for PCLA agents.

Hooks the internal detection head of TransFuser-family agents (tfv4, tfv6)
to log per-tick the bounding box + confidence score of the vehicle ahead.

This gives a measurable signal -- internal to the agent -- of whether the
adversarial patch reduces the agent's perception of the leader vehicle,
analogous to the YOLO confidence analysis but inside the driving model.

Output format (TSV):
    step  score  x  y  w  h  class

Where (x, y) is bbox center in vehicle frame (meters, +x forward, +y left),
(w, h) bbox size, class is detected object class id.
A row with score=0 means no vehicle was detected ahead this tick.

Usage from scenario script:
    from detection_probe import setup_detection_probe
    setup_detection_probe(pcla, agent_name="tfv6_visiononly", out_path="...")
"""
from __future__ import annotations

import os
import math


# Bbox tuple layout in TransFuser vehicle frame (from center_net_decoder.py):
#   (x, y, w, h, yaw, velocity, brake, class, score)
IDX_X, IDX_Y, IDX_W, IDX_H = 0, 1, 2, 3
IDX_CLASS, IDX_SCORE = 7, 8

# Acceptance window for "vehicle ahead": positive x (forward), small lateral y.
FRONT_X_MIN = 0.5      # meters -- ignore objects behind / very close
FRONT_X_MAX = 60.0     # meters -- ignore far away
FRONT_Y_ABS = 4.0      # meters -- lateral tolerance (~one lane width)


def _find_front_vehicle(bbs):
    """Pick the closest vehicle directly ahead, if any.

    bbs: array-like shape (N, 9). Can be empty / None.
    Returns (score, x, y, w, h, class) or None.
    """
    if bbs is None:
        return None
    try:
        n = len(bbs)
    except TypeError:
        return None
    if n == 0:
        return None

    best = None
    best_x = math.inf
    for row in bbs:
        x = float(row[IDX_X])
        y = float(row[IDX_Y])
        if x < FRONT_X_MIN or x > FRONT_X_MAX:
            continue
        if abs(y) > FRONT_Y_ABS:
            continue
        if x < best_x:
            best_x = x
            best = (
                float(row[IDX_SCORE]),
                x, y,
                float(row[IDX_W]), float(row[IDX_H]),
                int(row[IDX_CLASS]),
            )
    return best


def _write_row(path: str, step: int, front):
    with open(path, "a") as f:
        if front is None:
            f.write(f"{step}\t0.0000\t\t\t\t\t\n")
        else:
            score, x, y, w, h, cls = front
            f.write(f"{step}\t{score:.4f}\t{x:.2f}\t{y:.2f}\t{w:.2f}\t{h:.2f}\t{cls}\n")


def _init_log(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("step\tscore\tx\ty\tw\th\tclass\n")


def setup_detection_probe(pcla, agent_name: str, out_path: str) -> bool:
    """Install a monkey-patch that logs detection of the front vehicle per tick.

    Returns True if a probe was installed, False if the agent type is not
    supported (e.g. SimLingo VLM has no standard detection head).
    """
    agent_lower = agent_name.lower()
    agent = pcla.agent_instance

    if agent_lower.startswith("tfv6"):
        _init_log(out_path)
        orig_run_step = agent.run_step

        def wrapped_run_step(*args, **kwargs):
            ctrl = orig_run_step(*args, **kwargs)
            try:
                buf = getattr(agent, "bb_buffer", None)
                latest = buf[-1] if buf and len(buf) > 0 else None
                front = _find_front_vehicle(latest)
                _write_row(out_path, getattr(agent, "step", -1), front)
            except Exception as e:
                print(f"[detection_probe:tfv6] log failed: {e}")
            return ctrl

        agent.run_step = wrapped_run_step
        print(f"[detection_probe] tfv6 hook installed -> {out_path}")
        return True

    if agent_lower.startswith("tfv4") or agent_lower.startswith("tfv3") or agent_lower.startswith("tfv5"):
        # tfv3/4/5 share the same per-net convert_features_to_bb_metric API.
        nets = getattr(agent, "nets", None)
        if not nets:
            print(f"[detection_probe] tfv4/5: no .nets found on agent")
            return False
        _init_log(out_path)
        net = nets[0]
        orig_conv = net.convert_features_to_bb_metric

        def wrapped_conv(*args, **kwargs):
            bbs = orig_conv(*args, **kwargs)
            try:
                front = _find_front_vehicle(bbs)
                _write_row(out_path, getattr(agent, "step", -1), front)
            except Exception as e:
                print(f"[detection_probe:tfv4] log failed: {e}")
            return bbs

        net.convert_features_to_bb_metric = wrapped_conv
        print(f"[detection_probe] tfv4-family hook installed -> {out_path}")
        return True

    # SimLingo / LMDrive / others: VLM-based, no standard detection head.
    print(f"[detection_probe] no probe available for agent '{agent_name}'")
    return False
