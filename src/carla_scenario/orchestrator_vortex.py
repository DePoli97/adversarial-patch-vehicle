#!/usr/bin/env python3
"""Closed-loop experiment orchestrator — ALL LOCAL ON VORTEX.

CARLA server (patched package) and the driving agent both run on Vortex.
Patch deployed by swapping the cooked .ubulk of the "clean" texture slot,
then restarting CARLA fresh per (town, light, condition) block.
Paired seeds: the SAME seed set is reused for every combination.

Run: python orchestrator_vortex.py --agents tfv6_visiononly neat_aim2dsem --seeds 30
"""
import argparse
import json
import os
import subprocess
import time
from datetime import datetime

HOST = "localhost"
REPO = "/home/vortex/adversarial-patch-vehicle"
CARLA_DIR = "/home/vortex/carla/Dist/CARLA_Shipping_0.9.15.2-dirty/LinuxNoEditor"
GRID = f"{CARLA_DIR}/CarlaUE4/Content/Carla/Static/Truck/CarlaCola/grid_experiment"
CONDA = "source ~/miniconda3/etc/profile.d/conda.sh && conda activate PCLA15"

TOWNS = ["Town04", "Town07", "Town11"]
LIGHTS = ["day", "night"]
# "wb_*" are the white-box patches trained against tfv6's own target-speed head.
# The four legacy conditions were trained against YOLOv8 and are kept only as the
# published null baseline.
CONDITIONS = ["clean", "specialist", "generalist", "pooled"]
LOCK_PATH = "/tmp/carla_orchestrator.lock"


def ubulk_for(condition, town, light):
    if condition == "clean":
        return "123_CarlaCola_clean.ubulk.ORIG2"
    if condition == "specialist":
        return f"123_carlacola_{town}_{light}.ubulk"
    if condition == "generalist":
        return "123_carlacola_generalist.ubulk"
    if condition == "pooled":
        return "123_carlacola_pooled.ubulk"
    if condition == "wb_pooled":
        return "123_carlacola_wb_pooled.ubulk"
    if condition == "wb_specialist":
        return f"123_carlacola_wb_{town}_{light}.ubulk"
    raise ValueError(condition)


def acquire_single_instance_lock():
    """Refuse to start if another orchestrator is already driving this CARLA.

    Two orchestrators sharing one simulator destroy each other's actors, which
    surfaces as an unexplained "destroyed actor" mid-run and silently corrupts a
    whole night of results. This has happened once (2026-07-18) and cost a full
    campaign. The lock is held for the process lifetime and released by the OS
    even on a hard kill, so a crashed run never leaves a stale lock behind.
    """
    import fcntl
    fh = open(LOCK_PATH, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(
            f"another orchestrator holds {LOCK_PATH}. Two orchestrators on one "
            "CARLA destroy each other's actors — refusing to start. Check with "
            "'pgrep -af orchestrator_vortex'."
        )
    fh.write(f"{os.getpid()}\n")
    fh.flush()
    return fh  # keep the handle alive: closing it releases the lock


def swap_ubulk(condition, town, light):
    src = ubulk_for(condition, town, light)
    subprocess.run(["cp", f"{GRID}/{src}", f"{GRID}/123_CarlaCola_clean.ubulk"], check=True)


def carla_running():
    r = subprocess.run("ss -tlnp 2>/dev/null | grep -q :2000", shell=True)
    return r.returncode == 0


def kill_carla():
    # The bracket stops the pattern from matching this very shell: `pkill -f`
    # tests full command lines, and the /bin/sh spawned here contains the literal
    # string, so the unbracketed form makes the killer kill itself before the
    # sleep runs.
    subprocess.run('pkill -f "[C]arlaUE4-Linux" 2>/dev/null; sleep 3', shell=True)


def carla_ready(town):
    import carla
    try:
        c = carla.Client(HOST, 2000)
        c.set_timeout(60.0)
        c.load_world(town)
        _ = c.get_world().get_map().get_spawn_points()
        return True
    except Exception:
        return False


def start_carla(town):
    kill_carla()
    subprocess.Popen(
        f"cd {CARLA_DIR} && ./CarlaUE4.sh -RenderOffScreen -nosound "
        f"-carla-rpc-port=2000 > /tmp/carla_orch_vortex.log 2>&1",
        shell=True, preexec_fn=os.setsid,
    )
    for _ in range(60):
        if carla_running():
            break
        time.sleep(3)
    else:
        return False
    for _ in range(24):   # up to 2 min for real readiness (Town11 is heavy)
        time.sleep(5)
        if carla_ready(town):
            return True
    return False


def run_one(agent, town, light, condition, seed, out_root):
    sub = f"{out_root}/{agent}/{town}_{light}/{condition}"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    cmd = [
        "bash", "-lc",
        f"{CONDA} && cd {REPO} && python src/carla_scenario/scenario_two_vehicles.py "
        f"--host {HOST} --port 2000 --agent {agent} --town {town} "
        f"--light {light} --seed {seed} --out_subdir {sub}",
    ]
    subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    base = f"{REPO}/experiments/carla_scenarios/{sub}"
    if not os.path.isdir(base):
        return None
    # Match this seed's own directory. Taking sorted(...)[-1] returned the last
    # ALPHABETICAL name, and "seed9" sorts after "seed10", so every seed from 10
    # on read seed9's summary and a genuine failure was never retried.
    runs = [d for d in os.listdir(base) if f"_seed{seed}_" in d]
    if not runs:
        return None
    newest = max(runs, key=lambda d: os.path.getmtime(os.path.join(base, d)))
    spath = os.path.join(base, newest, "summary.json")
    return json.load(open(spath)) if os.path.exists(spath) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--towns", nargs="+", default=TOWNS)
    ap.add_argument("--lights", nargs="+", default=LIGHTS)
    ap.add_argument("--out_root", default=None)
    args = ap.parse_args()

    lock = acquire_single_instance_lock()  # noqa: F841 — held for the process life

    for condition in args.conditions:
        for town in args.towns:
            for light in args.lights:
                src = f"{GRID}/{ubulk_for(condition, town, light)}"
                if not os.path.exists(src):
                    raise SystemExit(
                        f"missing texture for condition '{condition}': {src}. "
                        "Author it before starting the matrix — discovering this "
                        "hours in wastes the whole block."
                    )

    out_root = args.out_root or f"matrix_{datetime.now():%Y%m%d_%H%M%S}"
    seeds = list(range(args.seeds))
    log_path = f"{REPO}/experiments/carla_scenarios/{out_root}_progress.jsonl"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    total = (len(args.agents) * len(args.towns) * len(args.lights)
             * len(args.conditions) * len(seeds))
    print(f"[ORCH] matrix: {total} runs -> {out_root}", flush=True)
    done = 0
    t0 = time.time()

    for agent in args.agents:
        for town in args.towns:
            for light in args.lights:
                for condition in args.conditions:
                    swap_ubulk(condition, town, light)
                    if not start_carla(town):
                        print(f"[ORCH][ERR] CARLA not ready: {agent}/{town}_{light}/{condition}", flush=True)
                        # still log the seeds as failed so the matrix is complete
                        for seed in seeds:
                            done += 1
                            with open(log_path, "a") as f:
                                f.write(json.dumps({"agent": agent, "town": town, "light": light,
                                    "condition": condition, "seed": seed, "ok": False,
                                    "error": "carla_not_ready"}) + "\n")
                        continue
                    for seed in seeds:
                        summ = None
                        for attempt in range(2):
                            try:
                                summ = run_one(agent, town, light, condition, seed, out_root)
                            except Exception as e:
                                summ = {"error": str(e)}
                            if summ is not None and "error" not in summ:
                                break
                            print(f"[ORCH][retry] {agent} {town}_{light} {condition} seed{seed}", flush=True)
                            if not start_carla(town):
                                break
                        done += 1
                        rec = {"agent": agent, "town": town, "light": light,
                               "condition": condition, "seed": seed,
                               "crashed": (summ or {}).get("crashed"),
                               "crash_time_since_brake_s": (summ or {}).get("crash_time_since_brake_s"),
                               "min_gap_m": (summ or {}).get("min_gap_m"),
                               "min_ttc_s": (summ or {}).get("min_ttc_s"),
                               "brake_delay_s": (summ or {}).get("brake_delay_s"),
                               "ok": summ is not None and "error" not in (summ or {})}
                        with open(log_path, "a") as f:
                            f.write(json.dumps(rec) + "\n")
                        elapsed = time.time() - t0
                        eta = elapsed / done * (total - done)
                        print(f"[ORCH] {done}/{total}  {agent} {town}_{light} {condition} "
                              f"seed{seed}  crashed={rec['crashed']} ok={rec['ok']}  "
                              f"ETA {eta/3600:.1f}h", flush=True)
    kill_carla()
    print(f"[ORCH] DONE {done}/{total} in {(time.time()-t0)/3600:.1f}h", flush=True)


if __name__ == "__main__":
    main()
