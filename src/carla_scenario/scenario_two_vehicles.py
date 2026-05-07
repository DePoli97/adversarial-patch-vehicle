"""Phase A (post meeting #4 redirect): two-vehicle CARLA scenario, single PCLA agent in control.

Setup (on Vortex):
  Terminal 1: cd /home/vortex/carla && conda activate carla && make launch
              → press Play in the Unreal Editor GUI
              → (before each condition) Reimport the adversarial TGA on the
                plate material slot (assets/carla_plates/T_LicensePlate_d_*.TGA)
  Terminal 2: conda activate PCLA310 && cd /home/vortex/adversarial-patch-vehicle
              python src/carla_scenario/scenario_two_vehicles.py \
                  --condition none --agent tfv6_visiononly

Setup summary:
  - Leader (NPC): straight highway cruise at --leader_speed km/h until t=10s,
    then emergency brake (throttle=0, brake=0.8). Carries the adversarial plate.
    No Traffic Manager involved.
  - Follower (Ego): single PCLA agent driving the vehicle — get_action() is
    applied every tick. Has to detect the leader and brake in time.
    This is the unit under test.

Timeline (15 s = 300 ticks at SIM_DELTA=0.05):
   0-10 s   both cruise at --leader_speed km/h
  10-15 s   leader hard brake -> follower-agent must react

Run the same scenario 3 x N times: 3 conditions (none / raw / camouflaged) x
N agents. Between conditions, swap the plate texture in UE via Reimport on
the material slot (workflow documented in docs/.STATUS.md).

Output: experiments/carla_scenarios/<condition>_<agent>_<timestamp>/
  telemetry.csv     per-tick ground-truth state (positions, speeds, TTC)
  agent.csv         per-tick PCLA agent control (throttle, steer, brake)
  images/           follower front-camera frames (every --save_interval ticks)
  summary.json      run metadata
"""

import os

# Silence OpenBLAS/OpenMP nested-parallelism warning that floods the log when
# PCLA agents run scipy/numpy internals. Must be set before numpy/scipy/torch
# are imported anywhere in this process.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import json
import math
import sys
import traceback
from datetime import datetime

import carla
import numpy as np

from common import (
    IMAGE_FOV,
    IMAGE_H,
    IMAGE_W,
    PCLA_DIR,
    REPO_ROOT,
    SIM_DELTA,
    compute_ttc,
    cruise_control,
    euclidean_distance,
    get_speed_kmh,
)
from spawn_utils import (
    give_initial_velocity,
    load_spawn_cache,
    move_to_rightmost_driving_lane,
    spawn_follower_behind_leader,
)

if PCLA_DIR not in sys.path:
    sys.path.insert(0, PCLA_DIR)

from PCLA import PCLA, location_to_waypoint, route_maker  # noqa: E402

# Detection probe (in this same dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detection_probe import setup_detection_probe  # noqa: E402


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_AGENT = "tfv6_visiononly"
DEFAULT_TOWN = "Town06"
FOLLOWER_BLUEPRINT = "vehicle.tesla.model3"   # ego (PCLA agent), kept fixed
LEADER_BLUEPRINT = "vehicle.carlamotors.european_hgv"  # NPC ahead — carries the patch
FOLLOWER_GAP_M = 10.0
LEADER_SPEED_KMH = 30
INITIAL_SPEED_KMH = 20
SAVE_INTERVAL_TICKS = 10
MAX_TICKS = 300  # 15 s at SIM_DELTA=0.05
DEFAULT_SEED = 0

# Leader emergency brake: fixed event that forces the follower-agent to react.
# Comparing none vs raw vs camouflaged shows how the plate patch affects the
# agent's ability to brake in time.
BRAKE_START_TICK = 200  # t = 10 s
BRAKE_STRENGTH = 0.8


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Two-vehicle CARLA scenario with a single PCLA agent in control."
    )
    p.add_argument(
        "--condition",
        choices=["none", "raw", "camouflaged"],
        default="none",
        help="Plate patch condition (output label; the TGA is swapped via UE Reimport).",
    )
    p.add_argument(
        "--agent",
        default=DEFAULT_AGENT,
        help="PCLA agent name to drive the follower (one per run).",
    )
    p.add_argument("--town", default=DEFAULT_TOWN)
    p.add_argument("--num_ticks", type=int, default=MAX_TICKS)
    p.add_argument("--save_interval", type=int, default=SAVE_INTERVAL_TICKS)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--leader_speed", type=float, default=LEADER_SPEED_KMH)
    p.add_argument("--gap_m", type=float, default=FOLLOWER_GAP_M)
    p.add_argument("--initial_speed", type=float, default=INITIAL_SPEED_KMH)
    p.add_argument(
        "--out_subdir",
        default="",
        help="Optional subfolder under experiments/carla_scenarios/ (groups multiple runs of one batch).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="RNG seed for python/numpy/torch. Same seed → reproducible run when "
             "the simulator and agent are deterministic. Used to measure variance.",
    )
    return p.parse_args()


def set_global_seed(seed: int):
    """Seed every RNG that could affect the agent's output.

    The CARLA simulator itself is deterministic in synchronous mode (no TM,
    no pedestrians, fixed delta). Variance across runs comes from agent
    internals: torch CPU/CUDA RNG, numpy, python random. Seeding all of them
    eliminates most of it; CUDA non-determinism in cudnn ops may still leak
    a tiny bit unless `torch.use_deterministic_algorithms(True)` is set,
    which slows things down — left off by default.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Sensors
# ---------------------------------------------------------------------------
def setup_debug_camera(world: carla.World, vehicle: carla.Actor) -> carla.Actor:
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(IMAGE_W))
    bp.set_attribute("image_size_y", str(IMAGE_H))
    bp.set_attribute("fov", str(IMAGE_FOV))
    transform = carla.Transform(carla.Location(x=1.6, z=1.7))
    return world.spawn_actor(bp, transform, attach_to=vehicle)


class CameraListener:
    def __init__(self):
        self.latest_frame = None
        self.tick_idx = 0

    def listen_callback(self, image: carla.Image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        self.latest_frame = arr.reshape((image.height, image.width, 4))[:, :, :3]

    def save_if_due(self, out_dir: str, tick: int, interval: int):
        if tick % interval == 0 and self.latest_frame is not None:
            import cv2

            path = os.path.join(out_dir, "images", f"tick_{tick:06d}.jpg")
            cv2.imwrite(path, self.latest_frame)
            self.tick_idx += 1


# ---------------------------------------------------------------------------
# Route / I/O helpers
# ---------------------------------------------------------------------------
def generate_route(
    client: carla.Client,
    start_loc: carla.Location,
    end_loc: carla.Location,
    output_path: str,
) -> str:
    waypoints = location_to_waypoint(client, start_loc, end_loc)
    route_maker(waypoints, output_path)
    print(f"[INFO] Route saved to {output_path} ({len(waypoints)} waypoints)")
    return output_path


def build_output_dir(condition: str, agent: str, seed: int, out_subdir: str = "") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [REPO_ROOT, "experiments", "carla_scenarios"]
    if out_subdir:
        parts.append(out_subdir)
    parts.append(f"{condition}_{agent}_seed{seed}_{ts}")
    out_dir = os.path.join(*parts)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    return out_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_global_seed(args.seed)
    out_dir = build_output_dir(args.condition, args.agent, args.seed, args.out_subdir)

    print(f"\n{'=' * 60}")
    print(f"  Condition : {args.condition}")
    print(f"  Agent     : {args.agent}  (IN CONTROL)")
    print(f"  Town      : {args.town}")
    print(
        f"  Ticks     : {args.num_ticks}  ({args.num_ticks * SIM_DELTA:.1f}s sim time)"
    )
    print(f"  Leader BP : {LEADER_BLUEPRINT}  (NPC, carries patch)")
    print(f"  Follower  : {FOLLOWER_BLUEPRINT}  (ego, PCLA agent)")
    print(f"  Seed      : {args.seed}")
    print(f"  Output    : {out_dir}")
    print(f"{'=' * 60}\n")

    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    print(f"[INFO] Connecting to CARLA at {args.host}:{args.port} ...")
    client.load_world(args.town)
    print(f"[INFO] World '{args.town}' loaded.")

    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = SIM_DELTA
    world.apply_settings(settings)

    leader = None
    follower = None
    debug_cam = None
    col_sensor = None
    pcla = None

    try:
        bplib = world.get_blueprint_library()
        leader_bp = bplib.filter(LEADER_BLUEPRINT)[0]
        follower_bp = bplib.filter(FOLLOWER_BLUEPRINT)[0]
        spawn_points = world.get_map().get_spawn_points()

        leader_idx = load_spawn_cache(args.town)
        if leader_idx is None:
            raise RuntimeError(
                f"No cached spawn index for '{args.town}'. "
                f"Run: python src/carla_scenario/tools/scan_spawn.py --town {args.town}"
            )
        leader_sp = spawn_points[leader_idx]
        carla_map = world.get_map()
        leader_wp = carla_map.get_waypoint(
            leader_sp.location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        rightmost_wp = move_to_rightmost_driving_lane(leader_wp)
        leader_transform = rightmost_wp.transform
        leader_transform.location.z += 0.5
        if rightmost_wp.lane_id != leader_wp.lane_id:
            print(
                f"[INFO] Shifted leader from lane {leader_wp.lane_id} "
                f"-> rightmost driving lane {rightmost_wp.lane_id}"
            )
        leader = world.try_spawn_actor(leader_bp, leader_transform)
        if leader is None:
            raise RuntimeError(f"Failed to spawn leader near spawn {leader_idx}")
        print(
            f"[INFO] Leader spawned at "
            f"({leader_transform.location.x:.1f}, {leader_transform.location.y:.1f}) "
            f"lane {rightmost_wp.lane_id}"
        )

        follower = spawn_follower_behind_leader(
            world, follower_bp, leader_transform, args.gap_m
        )
        world.tick()
        fpos = follower.get_location()
        print(f"[INFO] Follower spawned at ({fpos.x:.1f}, {fpos.y:.1f}, {fpos.z:.1f})")

        give_initial_velocity(leader, args.initial_speed)
        give_initial_velocity(follower, args.initial_speed)
        world.tick()
        print(f"[INFO] Initial velocity set to {args.initial_speed} km/h")

        # Follower needs a route for PCLA (straight ~600 m along the current lane)
        follower_start_wp = carla_map.get_waypoint(
            follower.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        end_wp = follower_start_wp
        walked = 0.0
        while walked < 600.0:
            nexts = end_wp.next(10.0)
            if not nexts:
                break
            end_wp = nexts[0]
            walked += 10.0
        route_path = os.path.join(out_dir, "follower_route.xml")
        generate_route(
            client,
            follower_start_wp.transform.location,
            end_wp.transform.location,
            route_path,
        )

        # Single PCLA agent in CONTROL on the follower
        print(f"[INFO] Attaching PCLA agent '{args.agent}' in control mode ...")
        pcla = PCLA(args.agent, follower, route_path, client)
        print(f"[INFO] Agent attached.")

        # For SimLingo (VLM), wrap the underlying InternVL model so every
        # forward pass also logs the generated language output to file.
        # Zero changes to PCLA source.
        simlingo_log_path = None
        if args.agent.startswith("simlingo"):
            simlingo_log_path = os.path.join(out_dir, "simlingo_language.tsv")
            with open(simlingo_log_path, "w") as _f:
                _f.write("step\tlanguage\n")

            _orig_model = pcla.agent_instance.model
            _agent_ref = pcla.agent_instance

            class _LingoLogger:
                def __init__(self, orig, agent, path):
                    self._orig = orig
                    self._agent = agent
                    self._path = path

                def __call__(self, *a, **kw):
                    out = self._orig(*a, **kw)
                    try:
                        if isinstance(out, tuple) and len(out) >= 3 and out[2] is not None:
                            lang = out[2]
                            if len(lang) > 0:
                                with open(self._path, "a") as f:
                                    f.write(f"{self._agent.step}\t{lang[0]}\n")
                    except Exception:
                        pass
                    return out

                def __getattr__(self, name):
                    return getattr(self._orig, name)

            pcla.agent_instance.model = _LingoLogger(
                _orig_model, _agent_ref, simlingo_log_path
            )
            print(f"[INFO] SimLingo language logging enabled -> {simlingo_log_path}")

        # Detection probe: for TransFuser-family agents (tfv4, tfv6),
        # log front-vehicle bbox + confidence per tick from the agent's own
        # detection head. Gives a measurable signal on whether the patch
        # weakens the agent's perception of the leader.
        detection_log_path = os.path.join(out_dir, "detection_probe.tsv")
        probe_active = setup_detection_probe(
            pcla, agent_name=args.agent, out_path=detection_log_path
        )
        if not probe_active:
            detection_log_path = None

        debug_cam = setup_debug_camera(world, follower)
        cam_listener = CameraListener()
        debug_cam.listen(cam_listener.listen_callback)

        col_bp = bplib.find("sensor.other.collision")
        col_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=follower)
        collision_events: list = []
        col_sensor.listen(lambda e: collision_events.append(e))

        spectator = world.get_spectator()

        telemetry_path = os.path.join(out_dir, "telemetry.csv")
        telemetry_fields = [
            "tick",
            "sim_time_s",
            "leader_x",
            "leader_y",
            "leader_speed_kmh",
            "follower_x",
            "follower_y",
            "follower_speed_kmh",
            "distance_m",
            "ttc_s",
            "collision_detected",
        ]

        agent_path = os.path.join(out_dir, "agent.csv")
        agent_fields = ["tick", "sim_time_s", "throttle", "steer", "brake"]

        collision_count = 0
        distances: list[float] = []
        agent_ok = 0
        agent_none = 0
        agent_error = 0
        first_agent_error_logged = False

        world.tick()

        print(f"\n[INFO] Starting simulation loop ({args.num_ticks} ticks)...\n")

        with (
            open(telemetry_path, "w", newline="") as tf,
            open(agent_path, "w", newline="") as af,
        ):
            telemetry_writer = csv.DictWriter(tf, fieldnames=telemetry_fields)
            telemetry_writer.writeheader()
            agent_writer = csv.DictWriter(af, fieldnames=agent_fields)
            agent_writer.writeheader()

            for tick in range(args.num_ticks):
                sim_time = tick * SIM_DELTA

                if tick == BRAKE_START_TICK:
                    print(f"\n[EVENT] t={sim_time:.1f}s  ->  leader emergency brake")

                # Leader: cruise straight at target speed until t=10s, then
                # hard-brake. This is the event the follower-agent has to
                # react to; the plate patch should degrade that reaction.
                if tick < BRAKE_START_TICK:
                    leader.apply_control(cruise_control(leader, args.leader_speed))
                else:
                    leader.apply_control(
                        carla.VehicleControl(throttle=0.0, brake=BRAKE_STRENGTH)
                    )

                # Follower: PCLA agent in CONTROL
                ctrl = None
                try:
                    ctrl = pcla.get_action()
                except Exception:
                    agent_error += 1
                    if not first_agent_error_logged:
                        print(f"[WARN] agent.get_action() raised at tick {tick}:")
                        print(traceback.format_exc())
                        first_agent_error_logged = True

                if ctrl is None:
                    agent_none += 1
                    # Hold previous control implicitly (do nothing this tick)
                else:
                    agent_ok += 1
                    follower.apply_control(ctrl)
                    agent_writer.writerow(
                        {
                            "tick": tick,
                            "sim_time_s": round(sim_time, 3),
                            "throttle": round(ctrl.throttle, 4),
                            "steer": round(ctrl.steer, 4),
                            "brake": round(ctrl.brake, 4),
                        }
                    )

                # Spectator: third-person behind the follower
                ft = follower.get_transform()
                cam_loc = carla.Location(
                    x=ft.location.x - 10 * math.cos(math.radians(ft.rotation.yaw)),
                    y=ft.location.y - 10 * math.sin(math.radians(ft.rotation.yaw)),
                    z=ft.location.z + 6,
                )
                spectator.set_transform(
                    carla.Transform(
                        cam_loc, carla.Rotation(pitch=-20, yaw=ft.rotation.yaw)
                    )
                )

                ###############
                ###############
                world.tick()###
                ###############
                ###############

                # Measurements
                lloc = leader.get_location()
                floc = follower.get_location()
                dist_m = euclidean_distance(lloc, floc)
                leader_spd = get_speed_kmh(leader)
                follower_spd = get_speed_kmh(follower)
                ttc = compute_ttc(dist_m, follower_spd / 3.6, leader_spd / 3.6)

                has_collision = len(collision_events) > 0
                if has_collision:
                    collision_count += len(collision_events)
                    collision_events.clear()

                distances.append(dist_m)

                telemetry_writer.writerow(
                    {
                        "tick": tick,
                        "sim_time_s": round(sim_time, 3),
                        "leader_x": round(lloc.x, 3),
                        "leader_y": round(lloc.y, 3),
                        "leader_speed_kmh": round(leader_spd, 2),
                        "follower_x": round(floc.x, 3),
                        "follower_y": round(floc.y, 3),
                        "follower_speed_kmh": round(follower_spd, 2),
                        "distance_m": round(dist_m, 3),
                        "ttc_s": round(ttc, 3) if ttc != float("inf") else -1,
                        "collision_detected": int(has_collision),
                    }
                )

                cam_listener.save_if_due(out_dir, tick, args.save_interval)

                if tick % 50 == 0:
                    print(
                        f"  tick={tick:4d} | dist={dist_m:5.1f}m | "
                        f"TTC={ttc:5.1f}s | "
                        f"leader={leader_spd:.1f}km/h | "
                        f"follower={follower_spd:.1f}km/h | "
                        f"collisions={collision_count}"
                    )

        summary = {
            "condition": args.condition,
            "agent": args.agent,
            "leader_blueprint": LEADER_BLUEPRINT,
            "follower_blueprint": FOLLOWER_BLUEPRINT,
            "seed": args.seed,
            "town": args.town,
            "num_ticks": args.num_ticks,
            "sim_duration_s": args.num_ticks * SIM_DELTA,
            "leader_speed_kmh": args.leader_speed,
            "initial_gap_m": args.gap_m,
            "brake_start_tick": BRAKE_START_TICK,
            "brake_start_s": BRAKE_START_TICK * SIM_DELTA,
            "brake_strength": BRAKE_STRENGTH,
            "agent_ticks_ok": agent_ok,
            "agent_ticks_none": agent_none,
            "agent_ticks_error": agent_error,
            "total_collisions": collision_count,
            "mean_distance_m": round(float(np.mean(distances)), 3)
            if distances
            else None,
            "min_distance_m": round(float(np.min(distances)), 3) if distances else None,
            "max_distance_m": round(float(np.max(distances)), 3) if distances else None,
            "images_saved": cam_listener.tick_idx,
            "simlingo_language_log": os.path.basename(simlingo_log_path)
            if simlingo_log_path is not None
            else None,
            "detection_probe_log": os.path.basename(detection_log_path)
            if detection_log_path is not None
            else None,
            "output_dir": out_dir,
        }
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'=' * 60}")
        print(f"  Run complete.")
        print(
            f"  Agent      : ok={agent_ok} none={agent_none} error={agent_error}"
        )
        print(f"  Collisions : {collision_count}")
        print(f"  Mean dist  : {summary['mean_distance_m']} m")
        print(f"  Min dist   : {summary['min_distance_m']} m")
        print(f"  Images     : {cam_listener.tick_idx}")
        print(f"  Output     : {out_dir}")
        print(f"{'=' * 60}\n")

    finally:
        print("[INFO] Cleaning up...")
        if debug_cam is not None and debug_cam.is_alive:
            debug_cam.stop()
            debug_cam.destroy()
        if col_sensor is not None and col_sensor.is_alive:
            col_sensor.stop()
            col_sensor.destroy()
        if pcla is not None:
            try:
                pcla.cleanup()
            except Exception:
                print("[WARN] pcla.cleanup() raised:")
                print(traceback.format_exc())
        if follower is not None and follower.is_alive:
            follower.destroy()
        if leader is not None and leader.is_alive:
            leader.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("[INFO] Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
