"""
Phase A: Two-vehicle following scenario in CARLA with adversarial license plate.

Setup (on Vortex):
  Terminal 1: cd /home/vortex/carla && conda activate carla && make launch
  Terminal 2: conda activate PCLA310 && cd /home/vortex/adversarial-patch-vehicle
              python src/carla_scenario/scenario_two_vehicles.py --condition none

License plate conditions:
  none         — original CARLA plate texture (baseline)
  raw          — adversarial raw patch (place TGA before launching CARLA, see README below)
  camouflaged  — adversarial camouflaged patch

NOTE: CARLA loads textures at startup. To switch plate conditions you must:
  1. Copy the appropriate TGA to the CARLA content path (see PLATE_TEXTURE_SRC in args)
  2. Restart the CARLA server
  3. Run this script with the new --condition flag

Output: experiments/carla_scenarios/<condition>_<timestamp>/
  - telemetry.csv      : per-tick telemetry (speed, distance, TTC, control)
  - images/            : front camera frames (every --save_interval ticks)
  - summary.json       : run metadata and aggregate stats
"""

import os
import sys
import csv
import json
import time
import math
import argparse
import numpy as np
from datetime import datetime

import carla

# Allow importing PCLA from adjacent repo on Vortex
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
PCLA_DIR = os.path.abspath(os.path.join(REPO_ROOT, '..', 'PCLA'))
if PCLA_DIR not in sys.path:
    sys.path.insert(0, PCLA_DIR)

from PCLA import PCLA, route_maker, location_to_waypoint


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
DEFAULT_AGENT       = 'tfv4_l6_0'   # top Leaderboard 1 performer, camera+LiDAR
DEFAULT_TOWN        = 'Town06'      # long straight highways
LEADER_SPAWN_IDX    = -1            # -1 = auto-detect longest straight segment
FOLLOWER_GAP_M      = 10.0          # initial gap behind leader (metres)
LEADER_SPEED_KMH    = 40            # target speed for leader — keep low so PCLA agent can follow
INITIAL_SPEED_KMH   = 20            # both vehicles start already moving
SIM_DELTA           = 0.05          # fixed simulation step (seconds)
IMAGE_W, IMAGE_H    = 1280, 720
IMAGE_FOV           = 90
SAVE_INTERVAL_TICKS = 10            # save camera frame every N ticks
MAX_TICKS           = 1500          # ~75 seconds of simulated time


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Two-vehicle CARLA scenario.')
    p.add_argument('--condition', choices=['none', 'raw', 'camouflaged'], default='none',
                   help='Plate patch condition.')
    p.add_argument('--agent', default=DEFAULT_AGENT,
                   help='PCLA agent name for the follower vehicle.')
    p.add_argument('--town', default=DEFAULT_TOWN)
    p.add_argument('--num_ticks', type=int, default=MAX_TICKS)
    p.add_argument('--save_interval', type=int, default=SAVE_INTERVAL_TICKS,
                   help='Save camera image every N ticks.')
    p.add_argument('--host', default='localhost')
    p.add_argument('--port', type=int, default=2000)
    p.add_argument('--leader_speed', type=float, default=LEADER_SPEED_KMH,
                   help='Leader target speed (km/h) via Traffic Manager.')
    p.add_argument('--leader_spawn', type=int, default=LEADER_SPAWN_IDX,
                   help='Spawn point index for the leader.')
    p.add_argument('--gap_m', type=float, default=FOLLOWER_GAP_M,
                   help='Initial gap (m) between leader and follower.')
    p.add_argument('--initial_speed', type=float, default=INITIAL_SPEED_KMH,
                   help='Initial forward velocity (km/h) for both vehicles.')
    p.add_argument('--list_spawn_points', action='store_true',
                   help='List all spawn points for the given --town and exit.')
    p.add_argument('--rescan', action='store_true',
                   help='Force re-scan for best spawn point, ignoring cache.')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def euclidean_distance(loc_a: carla.Location, loc_b: carla.Location) -> float:
    return math.sqrt((loc_a.x - loc_b.x)**2 + (loc_a.y - loc_b.y)**2 + (loc_a.z - loc_b.z)**2)


def get_speed_kmh(vehicle: carla.Actor) -> float:
    v = vehicle.get_velocity()
    return 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)


def compute_ttc(dist_m: float, follower_speed_ms: float, leader_speed_ms: float) -> float:
    """Time-to-collision (seconds). Returns inf if closing speed <= 0."""
    closing_speed = follower_speed_ms - leader_speed_ms
    if closing_speed <= 0:
        return float('inf')
    return dist_m / closing_speed


def spawn_follower_behind_leader(
    world: carla.World,
    blueprint: carla.ActorBlueprint,
    leader_transform: carla.Transform,
    gap_m: float
) -> carla.Actor:
    """Spawn follower gap_m metres behind leader on the same lane.

    Uses CARLA's road network (waypoint.previous) to walk backward along the
    same lane — avoids colliding with walls/sidewalks that happen when you
    compute the position with naive trigonometry.
    """
    carla_map = world.get_map()
    leader_wp = carla_map.get_waypoint(
        leader_transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if leader_wp is None:
        raise RuntimeError('Leader is not on a driving lane; cannot place follower.')

    prev_wps = leader_wp.previous(gap_m)
    if not prev_wps:
        raise RuntimeError(f'No previous waypoint {gap_m}m behind leader.')
    follower_wp = prev_wps[0]
    follower_transform = follower_wp.transform
    # Lift slightly to avoid ground clipping at spawn
    follower_transform.location.z += 0.5

    vehicle = world.try_spawn_actor(blueprint, follower_transform)
    if vehicle is None:
        # Try shorter gaps progressively
        for alt_gap in (gap_m - 3, gap_m - 6, gap_m + 3):
            if alt_gap <= 0:
                continue
            alt_wps = leader_wp.previous(alt_gap)
            if not alt_wps:
                continue
            alt_t = alt_wps[0].transform
            alt_t.location.z += 0.5
            vehicle = world.try_spawn_actor(blueprint, alt_t)
            if vehicle is not None:
                print(f'[WARN] Fallback follower gap={alt_gap}m (requested {gap_m}m)')
                break
    if vehicle is None:
        raise RuntimeError('Failed to spawn follower vehicle on the lane.')
    return vehicle


def give_initial_velocity(vehicle: carla.Actor, speed_kmh: float):
    """Set an instantaneous forward velocity on the vehicle."""
    yaw = math.radians(vehicle.get_transform().rotation.yaw)
    speed_ms = speed_kmh / 3.6
    vehicle.set_target_velocity(carla.Vector3D(
        x=speed_ms * math.cos(yaw),
        y=speed_ms * math.sin(yaw),
        z=0.0,
    ))


SPAWN_CACHE_PATH = os.path.join(REPO_ROOT, 'experiments', 'carla_scenarios', 'spawn_cache.json')


def load_spawn_cache(town: str) -> int | None:
    """Return cached best spawn index for this town, or None if not cached."""
    if not os.path.exists(SPAWN_CACHE_PATH):
        return None
    with open(SPAWN_CACHE_PATH) as f:
        cache = json.load(f)
    idx = cache.get(town)
    if idx is not None:
        print(f'[INFO] Using cached spawn index [{idx}] for {town} '
              f'(pass --rescan to override)')
    return idx


def save_spawn_cache(town: str, idx: int):
    os.makedirs(os.path.dirname(SPAWN_CACHE_PATH), exist_ok=True)
    cache = {}
    if os.path.exists(SPAWN_CACHE_PATH):
        with open(SPAWN_CACHE_PATH) as f:
            cache = json.load(f)
    cache[town] = idx
    with open(SPAWN_CACHE_PATH, 'w') as f:
        json.dump(cache, f, indent=2)
    print(f'[INFO] Spawn cache saved: {town} → [{idx}]')


def find_best_highway_spawn(world: carla.World, scan_distance_m: float = 300.0,
                             step_m: float = 10.0, yaw_tolerance_deg: float = 10.0) -> int:
    """Scan all spawn points; return the index with the longest straight segment ahead.

    For each spawn point, walk forward along the road network; the segment stays
    'straight' while the lane yaw remains within yaw_tolerance_deg of the start.
    Return the spawn index with the longest such distance — this is almost
    always a highway in CARLA's topology.
    """
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    best_idx = 0
    best_distance = 0.0
    print(f'[INFO] Scanning {len(spawn_points)} spawn points for longest straight segment...')
    for i, sp in enumerate(spawn_points):
        wp = carla_map.get_waypoint(sp.location, project_to_road=True,
                                    lane_type=carla.LaneType.Driving)
        if wp is None:
            continue
        initial_yaw = wp.transform.rotation.yaw
        current = wp
        dist = 0.0
        while dist < scan_distance_m:
            next_wps = current.next(step_m)
            if not next_wps:
                break
            current = next_wps[0]
            yaw_delta = abs(((current.transform.rotation.yaw - initial_yaw) + 180) % 360 - 180)
            if yaw_delta > yaw_tolerance_deg:
                break
            dist += step_m
        if dist > best_distance:
            best_distance = dist
            best_idx = i
    print(f'[INFO] Best spawn point: [{best_idx}]  straight for {best_distance:.0f}m ahead '
          f'at ({spawn_points[best_idx].location.x:.1f}, {spawn_points[best_idx].location.y:.1f})')
    return best_idx


def setup_rgb_camera(world: carla.World, vehicle: carla.Actor) -> carla.Actor:
    """Attach a front-facing RGB camera to vehicle, return sensor actor."""
    bp = world.get_blueprint_library().find('sensor.camera.rgb')
    bp.set_attribute('image_size_x', str(IMAGE_W))
    bp.set_attribute('image_size_y', str(IMAGE_H))
    bp.set_attribute('fov', str(IMAGE_FOV))
    # Mount on hood, pointing forward
    transform = carla.Transform(carla.Location(x=1.6, z=1.7))
    camera = world.spawn_actor(bp, transform, attach_to=vehicle)
    return camera


def generate_route(client: carla.Client, start_loc: carla.Location, end_loc: carla.Location,
                   output_path: str) -> str:
    """Generate PCLA-compatible XML route between two locations."""
    waypoints = location_to_waypoint(client, start_loc, end_loc)
    route_maker(waypoints, output_path)
    print(f'[INFO] Route saved to {output_path} ({len(waypoints)} waypoints)')
    return output_path


def build_output_dir(condition: str) -> str:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(
        REPO_ROOT, 'experiments', 'carla_scenarios',
        f'{condition}_{timestamp}'
    )
    os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)
    return out_dir


# ---------------------------------------------------------------------------
# Image listener (writes raw BGRA → numpy, stored in shared list)
# ---------------------------------------------------------------------------
class CameraListener:
    def __init__(self):
        self.latest_frame = None
        self.tick_idx = 0

    def listen_callback(self, image: carla.Image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        self.latest_frame = arr.reshape((image.height, image.width, 4))[:, :, :3]  # BGR

    def save_if_due(self, out_dir: str, tick: int, interval: int):
        if tick % interval == 0 and self.latest_frame is not None:
            import cv2
            path = os.path.join(out_dir, 'images', f'tick_{tick:06d}.jpg')
            cv2.imwrite(path, self.latest_frame)
            self.tick_idx += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    out_dir = build_output_dir(args.condition)
    print(f'\n{"="*60}')
    print(f'  Condition : {args.condition}')
    print(f'  Agent     : {args.agent}')
    print(f'  Town      : {args.town}')
    print(f'  Ticks     : {args.num_ticks}  ({args.num_ticks * SIM_DELTA:.1f}s sim time)')
    print(f'  Output    : {out_dir}')
    print(f'{"="*60}\n')

    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)  # load_world can take >30s on first call
    print(f'[INFO] Connecting to CARLA at {args.host}:{args.port} ...')
    client.load_world(args.town)
    print(f'[INFO] World "{args.town}" loaded.')

    world = client.get_world()
    traffic_manager = client.get_trafficmanager(8000)

    # --list_spawn_points: print then exit (debug helper)
    if args.list_spawn_points:
        sps = world.get_map().get_spawn_points()
        print(f'\n[{args.town}] {len(sps)} spawn points:')
        for i, sp in enumerate(sps):
            print(f'  [{i:3d}]  x={sp.location.x:8.2f}  y={sp.location.y:8.2f}  '
                  f'z={sp.location.z:5.2f}  yaw={sp.rotation.yaw:6.1f}')
        return

    # Synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = SIM_DELTA
    world.apply_settings(settings)
    traffic_manager.set_synchronous_mode(True)

    leader = None
    follower = None
    camera = None
    pcla = None

    try:
        bplib = world.get_blueprint_library()
        vehicle_bp = bplib.filter('model3')[0]
        spawn_points = world.get_map().get_spawn_points()

        # --- Pick leader spawn: auto-detect longest straight if < 0 ---
        if args.leader_spawn < 0:
            cached = None if args.rescan else load_spawn_cache(args.town)
            if cached is not None:
                leader_idx = cached
            else:
                leader_idx = find_best_highway_spawn(world)
                save_spawn_cache(args.town, leader_idx)
        else:
            leader_idx = args.leader_spawn
        leader_sp = spawn_points[leader_idx]
        leader = world.try_spawn_actor(vehicle_bp, leader_sp)
        if leader is None:
            raise RuntimeError(f'Failed to spawn leader at spawn point {leader_idx}')
        print(f'[INFO] Leader spawned at spawn_point[{leader_idx}] '
              f'= ({leader_sp.location.x:.1f}, {leader_sp.location.y:.1f})')

        # --- Spawn follower behind leader on the same lane ---
        follower = spawn_follower_behind_leader(world, vehicle_bp, leader_sp, args.gap_m)
        # Tick once so get_location() returns the actual spawned transform
        world.tick()
        fpos = follower.get_location()
        print(f'[INFO] Follower spawned at ({fpos.x:.1f}, {fpos.y:.1f}, {fpos.z:.1f})')

        # --- Give both vehicles initial forward velocity ---
        give_initial_velocity(leader, args.initial_speed)
        give_initial_velocity(follower, args.initial_speed)
        world.tick()
        print(f'[INFO] Initial velocity set to {args.initial_speed} km/h for both vehicles')

        world.tick()

        # --- Leader: Traffic Manager autopilot ---
        leader.set_autopilot(True, 8000)
        # set_desired_speed takes km/h and is absolute (no speed-limit dependency)
        traffic_manager.set_desired_speed(leader, args.leader_speed)
        traffic_manager.distance_to_leading_vehicle(leader, 2.0)
        traffic_manager.ignore_lights_percentage(leader, 100.0)
        traffic_manager.ignore_signs_percentage(leader, 100.0)
        traffic_manager.auto_lane_change(leader, False)
        print(f'[INFO] Leader autopilot enabled ({args.leader_speed} km/h, '
              f'ignoring lights/signs, no lane change)')

        # --- Generate route for follower ---
        # Walk 600m along the road network from the follower's waypoint to find
        # the route end — this guarantees the destination is on the road, not off it.
        carla_map = world.get_map()
        follower_start_wp = carla_map.get_waypoint(
            follower.get_location(), project_to_road=True,
            lane_type=carla.LaneType.Driving)
        end_wp = follower_start_wp
        walked = 0.0
        while walked < 600.0:
            nexts = end_wp.next(10.0)
            if not nexts:
                break
            end_wp = nexts[0]
            walked += 10.0
        start_loc = follower_start_wp.transform.location
        end_loc = end_wp.transform.location
        route_path = os.path.join(out_dir, 'follower_route.xml')
        generate_route(client, start_loc, end_loc, route_path)

        # --- PCLA follower agent ---
        pcla = PCLA(args.agent, follower, route_path, client)
        print(f'[INFO] PCLA agent "{args.agent}" initialized for follower')

        # --- Front camera on follower ---
        camera = setup_rgb_camera(world, follower)
        cam_listener = CameraListener()
        camera.listen(cam_listener.listen_callback)
        print(f'[INFO] Camera attached to follower')

        # Spectator: overhead view to observe the scene
        spectator = world.get_spectator()

        # --- Telemetry CSV ---
        csv_path = os.path.join(out_dir, 'telemetry.csv')
        csv_fields = [
            'tick', 'sim_time_s',
            'leader_x', 'leader_y', 'leader_speed_kmh',
            'follower_x', 'follower_y', 'follower_speed_kmh',
            'distance_m', 'ttc_s',
            'follower_throttle', 'follower_steer', 'follower_brake',
            'collision_detected',
        ]
        collision_count = 0
        distances = []

        # Collision sensor on follower
        col_bp = bplib.find('sensor.other.collision')
        col_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=follower)
        collision_events = []
        col_sensor.listen(lambda e: collision_events.append(e))

        world.tick()

        print(f'\n[INFO] Starting simulation loop ({args.num_ticks} ticks)...\n')

        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
            writer.writeheader()

            for tick in range(args.num_ticks):
                # Follower action
                ego_action = pcla.get_action()
                follower.apply_control(ego_action)

                # Update spectator to follow follower from above-behind
                follower_t = follower.get_transform()
                yaw = follower_t.rotation.yaw
                cam_loc = carla.Location(
                    x=follower_t.location.x - 10 * math.cos(math.radians(yaw)),
                    y=follower_t.location.y - 10 * math.sin(math.radians(yaw)),
                    z=follower_t.location.z + 6
                )
                spectator.set_transform(carla.Transform(cam_loc, carla.Rotation(pitch=-20, yaw=yaw)))

                world.tick()

                # --- Measurements ---
                leader_loc   = leader.get_location()
                follower_loc_now = follower.get_location()
                dist_m       = euclidean_distance(leader_loc, follower_loc_now)
                leader_spd   = get_speed_kmh(leader)
                follower_spd = get_speed_kmh(follower)
                ctrl         = follower.get_control()
                sim_time     = tick * SIM_DELTA

                closing_ms   = (follower_spd - leader_spd) / 3.6
                ttc          = compute_ttc(dist_m, follower_spd / 3.6, leader_spd / 3.6)

                has_collision = len(collision_events) > 0
                if has_collision:
                    collision_count += len(collision_events)
                    collision_events.clear()

                distances.append(dist_m)

                writer.writerow({
                    'tick': tick,
                    'sim_time_s': round(sim_time, 3),
                    'leader_x': round(leader_loc.x, 3),
                    'leader_y': round(leader_loc.y, 3),
                    'leader_speed_kmh': round(leader_spd, 2),
                    'follower_x': round(follower_loc_now.x, 3),
                    'follower_y': round(follower_loc_now.y, 3),
                    'follower_speed_kmh': round(follower_spd, 2),
                    'distance_m': round(dist_m, 3),
                    'ttc_s': round(ttc, 3) if ttc != float('inf') else -1,
                    'follower_throttle': round(ctrl.throttle, 3),
                    'follower_steer': round(ctrl.steer, 3),
                    'follower_brake': round(ctrl.brake, 3),
                    'collision_detected': int(has_collision),
                })

                # Save camera image
                cam_listener.save_if_due(out_dir, tick, args.save_interval)

                # Console progress
                if tick % 50 == 0:
                    print(f'  tick={tick:4d} | dist={dist_m:5.1f}m | '
                          f'TTC={ttc:5.1f}s | '
                          f'leader={leader_spd:.1f}km/h | '
                          f'follower={follower_spd:.1f}km/h | '
                          f'collisions={collision_count}')

        # --- Summary ---
        summary = {
            'condition': args.condition,
            'agent': args.agent,
            'town': args.town,
            'num_ticks': args.num_ticks,
            'sim_duration_s': args.num_ticks * SIM_DELTA,
            'total_collisions': collision_count,
            'mean_distance_m': round(float(np.mean(distances)), 3) if distances else None,
            'min_distance_m': round(float(np.min(distances)), 3) if distances else None,
            'max_distance_m': round(float(np.max(distances)), 3) if distances else None,
            'images_saved': cam_listener.tick_idx,
            'output_dir': out_dir,
        }
        with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print(f'\n{"="*60}')
        print(f'  Run complete.')
        print(f'  Collisions : {collision_count}')
        print(f'  Mean dist  : {summary["mean_distance_m"]} m')
        print(f'  Min dist   : {summary["min_distance_m"]} m')
        print(f'  Images     : {cam_listener.tick_idx}')
        print(f'  Output     : {out_dir}')
        print(f'{"="*60}\n')

    finally:
        print('[INFO] Cleaning up...')
        # Stop sensors first
        if camera is not None and camera.is_alive:
            camera.stop()
            camera.destroy()
        if 'col_sensor' in dir() and col_sensor.is_alive:
            col_sensor.stop()
            col_sensor.destroy()

        if pcla is not None:
            pcla.cleanup()
        elif follower is not None and follower.is_alive:
            follower.destroy()

        if leader is not None and leader.is_alive:
            leader.set_autopilot(False)
            leader.destroy()

        # Restore async mode
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print('[INFO] Done.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n[INFO] Interrupted by user.')
