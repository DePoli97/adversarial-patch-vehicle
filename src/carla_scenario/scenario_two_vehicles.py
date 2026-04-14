"""Phase A: two-vehicle CARLA scenario with shadow-mode PCLA agents.

Setup (on Vortex):
  Terminal 1: cd /home/vortex/carla && conda activate carla && make launch
              → press ▶ Play in the Unreal Editor GUI
  Terminal 2: conda activate PCLA310 && cd /home/vortex/adversarial-patch-vehicle
              python src/carla_scenario/scenario_two_vehicles.py --condition none

Scenario timeline (30s, 600 ticks at 0.05s):
   0–10s  cruise        — both vehicles at --leader_speed, gap fixed via TM
  10–20s  patch visible — plate texture swap (if --condition != none)
  20–30s  leader brake  — TM detached from leader, manual brake applied

Both vehicles are driven by Traffic Manager (leader: fixed speed, follower:
keeps distance). PCLA agents listed in --agents are attached to the follower
in SHADOW mode — their VehicleControl is logged but not applied. One CSV per
agent is produced next to telemetry.csv.

Output: experiments/carla_scenarios/<condition>_<timestamp>/
  telemetry.csv        per-tick ground-truth state (positions, speeds, TTC)
  agent_<name>.csv     per-tick shadow actions for each PCLA agent
  images/              follower front-camera frames (every --save_interval ticks)
  summary.json         run metadata
"""

import argparse
import csv
import json
import os
import sys
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
    euclidean_distance,
    get_speed_kmh,
)
from shadow_agents import ShadowAgentSet
from spawn_utils import (
    give_initial_velocity,
    load_spawn_cache,
    move_to_rightmost_driving_lane,
    spawn_follower_behind_leader,
)

if PCLA_DIR not in sys.path:
    sys.path.insert(0, PCLA_DIR)

from PCLA import location_to_waypoint, route_maker  # noqa: E402


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_AGENTS = ["tfv4_l6_0", "tfv6_visiononly", "simlingo_simlingo"]
DEFAULT_TOWN = "Town06"
FOLLOWER_GAP_M = 10.0
LEADER_SPEED_KMH = 40
INITIAL_SPEED_KMH = 20
SAVE_INTERVAL_TICKS = 10
MAX_TICKS = 600  # 30 s at SIM_DELTA=0.05

CRUISE_END_TICK = 200  # t = 10 s
BRAKE_START_TICK = 400  # t = 20 s
BRAKE_STRENGTH = 0.8


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Two-vehicle CARLA scenario.")
    p.add_argument(
        "--condition",
        choices=["none", "raw", "camouflaged"],
        default="none",
        help="Plate patch condition (appears at t=10s).",
    )
    p.add_argument(
        "--agents",
        nargs="+",
        default=DEFAULT_AGENTS,
        help="PCLA agent names to attach in SHADOW mode on the follower.",
    )
    p.add_argument("--town", default=DEFAULT_TOWN)
    p.add_argument("--num_ticks", type=int, default=MAX_TICKS)
    p.add_argument("--save_interval", type=int, default=SAVE_INTERVAL_TICKS)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--leader_speed", type=float, default=LEADER_SPEED_KMH)
    p.add_argument("--gap_m", type=float, default=FOLLOWER_GAP_M)
    p.add_argument("--initial_speed", type=float, default=INITIAL_SPEED_KMH)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Plate texture swap  —  TODO: fill in once the CARLA plate object/material
# path is known on Vortex. Called at CRUISE_END_TICK if condition != 'none'.
# ---------------------------------------------------------------------------
def swap_plate_texture(world: carla.World, condition: str):
    """Apply the adversarial plate texture for the given condition.

    TODO(paolo): implement once we know where the plate material lives in the
    CARLA content and how to swap it at runtime.
    """
    print(f"[TODO] plate swap requested for condition='{condition}' — not implemented yet")


# ---------------------------------------------------------------------------
# Sensors
# ---------------------------------------------------------------------------
def setup_debug_camera(world: carla.World, vehicle: carla.Actor) -> carla.Actor:
    """Front-facing RGB camera for visualisation/recording (separate from agent sensors)."""
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


def build_output_dir(condition: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        REPO_ROOT, "experiments", "carla_scenarios", f"{condition}_{ts}"
    )
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    return out_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    out_dir = build_output_dir(args.condition)

    print(f"\n{'=' * 60}")
    print(f"  Condition : {args.condition}")
    print(f"  Agents    : {args.agents}")
    print(f"  Town      : {args.town}")
    print(f"  Ticks     : {args.num_ticks}  ({args.num_ticks * SIM_DELTA:.1f}s sim time)")
    print(f"  Schedule  : cruise→{CRUISE_END_TICK}, brake from {BRAKE_START_TICK}")
    print(f"  Output    : {out_dir}")
    print(f"{'=' * 60}\n")

    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    print(f"[INFO] Connecting to CARLA at {args.host}:{args.port} ...")
    client.load_world(args.town)
    print(f"[INFO] World '{args.town}' loaded.")

    world = client.get_world()
    traffic_manager = client.get_trafficmanager(8000)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = SIM_DELTA
    world.apply_settings(settings)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_random_device_seed(42)  # reproducibility across runs

    leader = None
    follower = None
    debug_cam = None
    col_sensor = None
    shadow_set: ShadowAgentSet | None = None
    plate_applied = False

    try:
        bplib = world.get_blueprint_library()
        vehicle_bp = bplib.filter("model3")[0]
        spawn_points = world.get_map().get_spawn_points()

        # Leader spawn comes from the cached scan — run tools/scan_spawn.py once
        # per town to (re)populate experiments/carla_scenarios/spawn_cache.json.
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
                f"→ rightmost driving lane {rightmost_wp.lane_id}"
            )
        leader = world.try_spawn_actor(vehicle_bp, leader_transform)
        if leader is None:
            raise RuntimeError(f"Failed to spawn leader near spawn {leader_idx}")
        print(
            f"[INFO] Leader spawned at "
            f"({leader_transform.location.x:.1f}, {leader_transform.location.y:.1f}) "
            f"lane {rightmost_wp.lane_id}"
        )

        follower = spawn_follower_behind_leader(
            world, vehicle_bp, leader_transform, args.gap_m
        )
        world.tick()
        fpos = follower.get_location()
        print(f"[INFO] Follower spawned at ({fpos.x:.1f}, {fpos.y:.1f}, {fpos.z:.1f})")

        give_initial_velocity(leader, args.initial_speed)
        give_initial_velocity(follower, args.initial_speed)
        world.tick()
        print(f"[INFO] Initial velocity set to {args.initial_speed} km/h")

        # Both vehicles on Traffic Manager — leader cruises, follower keeps distance
        leader.set_autopilot(True, 8000)
        follower.set_autopilot(True, 8000)
        traffic_manager.set_desired_speed(leader, args.leader_speed)
        traffic_manager.set_desired_speed(follower, args.leader_speed)
        traffic_manager.distance_to_leading_vehicle(follower, args.gap_m)
        for v in (leader, follower):
            traffic_manager.ignore_lights_percentage(v, 100.0)
            traffic_manager.ignore_signs_percentage(v, 100.0)
            traffic_manager.auto_lane_change(v, False)
        print(
            f"[INFO] Both vehicles on TM autopilot "
            f"({args.leader_speed} km/h, {args.gap_m} m gap)"
        )

        # Route generation for the follower (required by PCLA agents)
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

        # Attach all shadow agents (each brings its own sensor suite)
        print(f"[INFO] Attaching {len(args.agents)} shadow agents to follower...")
        shadow_set = ShadowAgentSet(
            args.agents, follower, route_path, client, out_dir
        )

        # Our own debug camera for visualisation (saved every N ticks)
        debug_cam = setup_debug_camera(world, follower)
        cam_listener = CameraListener()
        debug_cam.listen(cam_listener.listen_callback)

        # Collision sensor on follower (ground-truth metric)
        col_bp = bplib.find("sensor.other.collision")
        col_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=follower)
        collision_events: list = []
        col_sensor.listen(lambda e: collision_events.append(e))

        spectator = world.get_spectator()

        # Telemetry CSV (ground truth)
        csv_path = os.path.join(out_dir, "telemetry.csv")
        csv_fields = [
            "tick",
            "sim_time_s",
            "phase",
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
        collision_count = 0
        distances: list[float] = []

        world.tick()

        print(f"\n[INFO] Starting simulation loop ({args.num_ticks} ticks)...\n")

        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
            writer.writeheader()

            for tick in range(args.num_ticks):
                sim_time = tick * SIM_DELTA

                # Phase transitions
                if tick == CRUISE_END_TICK and args.condition != "none" and not plate_applied:
                    print(f"\n[EVENT] t={sim_time:.1f}s  →  plate swap ({args.condition})")
                    swap_plate_texture(world, args.condition)
                    plate_applied = True

                if tick == BRAKE_START_TICK:
                    print(f"\n[EVENT] t={sim_time:.1f}s  →  leader emergency brake")
                    leader.set_autopilot(False)

                if tick >= BRAKE_START_TICK:
                    leader.apply_control(
                        carla.VehicleControl(throttle=0.0, brake=BRAKE_STRENGTH)
                    )

                phase = (
                    "cruise"
                    if tick < CRUISE_END_TICK
                    else ("patch" if tick < BRAKE_START_TICK else "brake")
                )

                # Shadow agents: compute + log, not applied
                shadow_set.tick(tick, sim_time)

                # Spectator cam: third-person follow behind the follower
                import math as _math

                ft = follower.get_transform()
                cam_loc = carla.Location(
                    x=ft.location.x - 10 * _math.cos(_math.radians(ft.rotation.yaw)),
                    y=ft.location.y - 10 * _math.sin(_math.radians(ft.rotation.yaw)),
                    z=ft.location.z + 6,
                )
                spectator.set_transform(
                    carla.Transform(cam_loc, carla.Rotation(pitch=-20, yaw=ft.rotation.yaw))
                )

                world.tick()

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

                writer.writerow(
                    {
                        "tick": tick,
                        "sim_time_s": round(sim_time, 3),
                        "phase": phase,
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
                        f"  tick={tick:4d} [{phase:6s}] | dist={dist_m:5.1f}m | "
                        f"TTC={ttc:5.1f}s | "
                        f"leader={leader_spd:.1f}km/h | "
                        f"follower={follower_spd:.1f}km/h | "
                        f"collisions={collision_count}"
                    )

        summary = {
            "condition": args.condition,
            "agents": args.agents,
            "town": args.town,
            "num_ticks": args.num_ticks,
            "sim_duration_s": args.num_ticks * SIM_DELTA,
            "cruise_end_tick": CRUISE_END_TICK,
            "brake_start_tick": BRAKE_START_TICK,
            "leader_speed_kmh": args.leader_speed,
            "initial_gap_m": args.gap_m,
            "plate_applied": plate_applied,
            "total_collisions": collision_count,
            "mean_distance_m": round(float(np.mean(distances)), 3) if distances else None,
            "min_distance_m": round(float(np.min(distances)), 3) if distances else None,
            "max_distance_m": round(float(np.max(distances)), 3) if distances else None,
            "images_saved": cam_listener.tick_idx,
            "output_dir": out_dir,
        }
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'=' * 60}")
        print(f"  Run complete.")
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
        if shadow_set is not None:
            shadow_set.cleanup()
        if follower is not None and follower.is_alive:
            follower.destroy()
        if leader is not None and leader.is_alive:
            try:
                leader.set_autopilot(False)
            except Exception:
                pass
            leader.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("[INFO] Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
