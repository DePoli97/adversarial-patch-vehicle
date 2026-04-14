"""Standalone utility: scan a town for the spawn with the longest straight ahead.

Writes the result to experiments/carla_scenarios/spawn_cache.json so the main
scenario can load it without rescanning.

Usage (on Vortex, with CARLA server already running):
    python src/carla_scenario/tools/scan_spawn.py --town Town06
"""

import argparse
import os
import sys

import carla

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)

from spawn_utils import save_spawn_cache  # noqa: E402


def find_best_highway_spawn(
    world: carla.World,
    scan_distance_m: float = 300.0,
    step_m: float = 10.0,
    yaw_tolerance_deg: float = 10.0,
) -> int:
    """Return spawn index with the longest straight road segment ahead."""
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    best_idx = 0
    best_distance = 0.0
    print(f"[INFO] Scanning {len(spawn_points)} spawn points...")
    for i, sp in enumerate(spawn_points):
        wp = carla_map.get_waypoint(
            sp.location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
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
    sp = spawn_points[best_idx]
    print(
        f"[INFO] Best spawn: [{best_idx}] straight for {best_distance:.0f}m "
        f"at ({sp.location.x:.1f}, {sp.location.y:.1f})"
    )
    return best_idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--town", default="Town06")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    args = p.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    client.load_world(args.town)
    world = client.get_world()

    idx = find_best_highway_spawn(world)
    save_spawn_cache(args.town, idx)


if __name__ == "__main__":
    main()
