"""Spawn/lane helpers for the two-vehicle highway scenario."""

import json
import math
import os

import carla

from common import REPO_ROOT


SPAWN_CACHE_PATH = os.path.join(
    REPO_ROOT, "experiments", "carla_scenarios", "spawn_cache.json"
)


def move_to_rightmost_driving_lane(wp: carla.Waypoint) -> carla.Waypoint:
    """Walk right across lanes until we hit the outermost driving lane.

    PCLA agents are trained on right-hand-traffic with road edge on the right.
    Starting in the leftmost lane of a multi-lane highway with empty lanes to
    the right produces out-of-distribution inputs.
    """
    current = wp
    while True:
        right = current.get_right_lane()
        if right is None or right.lane_type != carla.LaneType.Driving:
            break
        if right.lane_id * current.lane_id < 0:
            break
        current = right
    return current


def spawn_follower_behind_leader(
    world: carla.World,
    blueprint: carla.ActorBlueprint,
    leader_transform: carla.Transform,
    gap_m: float,
) -> carla.Actor:
    """Spawn follower gap_m metres behind leader on the same lane."""
    carla_map = world.get_map()
    leader_wp = carla_map.get_waypoint(
        leader_transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if leader_wp is None:
        raise RuntimeError("Leader is not on a driving lane; cannot place follower.")

    prev_wps = leader_wp.previous(gap_m)
    if not prev_wps:
        raise RuntimeError(f"No previous waypoint {gap_m}m behind leader.")
    follower_transform = prev_wps[0].transform
    follower_transform.location.z += 0.5

    vehicle = world.try_spawn_actor(blueprint, follower_transform)
    if vehicle is None:
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
                print(f"[WARN] Fallback follower gap={alt_gap}m (requested {gap_m}m)")
                break
    if vehicle is None:
        raise RuntimeError("Failed to spawn follower vehicle on the lane.")
    return vehicle


def give_initial_velocity(vehicle: carla.Actor, speed_kmh: float):
    """Set an instantaneous forward velocity on the vehicle."""
    yaw = math.radians(vehicle.get_transform().rotation.yaw)
    speed_ms = speed_kmh / 3.6
    vehicle.set_target_velocity(
        carla.Vector3D(
            x=speed_ms * math.cos(yaw),
            y=speed_ms * math.sin(yaw),
            z=0.0,
        )
    )


def load_spawn_cache(town: str) -> int | None:
    if not os.path.exists(SPAWN_CACHE_PATH):
        return None
    with open(SPAWN_CACHE_PATH) as f:
        cache = json.load(f)
    idx = cache.get(town)
    if idx is not None:
        print(
            f"[INFO] Using cached spawn index [{idx}] for {town} "
            f"(pass --rescan to override)"
        )
    return idx


def save_spawn_cache(town: str, idx: int):
    os.makedirs(os.path.dirname(SPAWN_CACHE_PATH), exist_ok=True)
    cache = {}
    if os.path.exists(SPAWN_CACHE_PATH):
        with open(SPAWN_CACHE_PATH) as f:
            cache = json.load(f)
    cache[town] = idx
    with open(SPAWN_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"[INFO] Spawn cache saved: {town} → [{idx}]")


def load_spawn_pool(town: str) -> list[int] | None:
    """Return the pool of pre-ranked straight spawn indices for the town, or None.

    Populated by `tools/scan_spawn.py --top-k N`. Used by the scenario runner
    to pick distinct starting points per seed instead of always reusing the
    single legacy `best` index.
    """
    if not os.path.exists(SPAWN_CACHE_PATH):
        return None
    with open(SPAWN_CACHE_PATH) as f:
        cache = json.load(f)
    pool = cache.get("pools", {}).get(town)
    return list(pool) if pool else None


