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

from spawn_utils import (  # noqa: E402
    SPAWN_CACHE_PATH,
    save_spawn_cache,
)
import json  # noqa: E402


def save_spawn_pool(town: str, ranked: list[tuple[int, float]], top_k: int = 10):
    """Save the top-K ranked spawn indices for the town under
    'pools' in spawn_cache.json. Keeps the legacy single-best entry untouched."""
    os.makedirs(os.path.dirname(SPAWN_CACHE_PATH), exist_ok=True)
    cache: dict = {}
    if os.path.exists(SPAWN_CACHE_PATH):
        with open(SPAWN_CACHE_PATH) as f:
            cache = json.load(f)
    pool = [idx for idx, _ in ranked[:top_k]]
    cache.setdefault("pools", {})[town] = pool
    with open(SPAWN_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"[INFO] Spawn pool saved: {town} top-{top_k} -> {pool}")


def rank_highway_spawns(
    world: carla.World,
    scan_distance_m: float = 300.0,
    step_m: float = 10.0,
    yaw_tolerance_deg: float = 10.0,
    min_straight_m: float = 150.0,
) -> list[tuple[int, float]]:
    """Return spawn indices sorted by descending 'straight ahead' length.

    Each entry is `(spawn_index, straight_length_m)`. Only spawns with at least
    `min_straight_m` of straight road ahead are kept. Used to build a pool of
    distinct, varied starting points for randomized scenario seeds.
    """
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    scored: list[tuple[int, float]] = []
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
        if dist >= min_straight_m:
            scored.append((i, dist))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored


def find_best_highway_spawn(world: carla.World, **kwargs) -> int:
    """Back-compat single-best lookup (legacy callers)."""
    scored = rank_highway_spawns(world, **kwargs)
    if not scored:
        raise RuntimeError("No straight highway spawn found in this town")
    best_idx, best_distance = scored[0]
    sp = world.get_map().get_spawn_points()[best_idx]
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
    p.add_argument(
        "--top-k", type=int, default=10,
        help="Save the top-K straight spawn indices for this town as the pool.",
    )
    p.add_argument(
        "--min-straight-m", type=float, default=150.0,
        help="Reject spawns whose straight-ahead distance is shorter than this.",
    )
    args = p.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    client.load_world(args.town)
    world = client.get_world()

    ranked = rank_highway_spawns(world, min_straight_m=args.min_straight_m)
    if not ranked:
        raise SystemExit(
            f"No spawn point with >= {args.min_straight_m:.0f} m of straight road "
            f"in {args.town}."
        )
    # Keep legacy "best" entry compatible with old scenario calls.
    save_spawn_cache(args.town, ranked[0][0])
    # New: pool of top-K so randomized seeds can pick distinct starting points.
    save_spawn_pool(args.town, ranked, top_k=args.top_k)


if __name__ == "__main__":
    main()
