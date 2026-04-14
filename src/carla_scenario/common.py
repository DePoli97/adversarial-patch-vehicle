"""Shared constants and small math helpers for the CARLA scenario."""

import math
import os

import carla


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
PCLA_DIR = os.path.abspath(os.path.join(REPO_ROOT, "..", "PCLA"))

SIM_DELTA = 0.05
IMAGE_W, IMAGE_H = 1280, 720
IMAGE_FOV = 90


def euclidean_distance(loc_a: carla.Location, loc_b: carla.Location) -> float:
    return math.sqrt(
        (loc_a.x - loc_b.x) ** 2 + (loc_a.y - loc_b.y) ** 2 + (loc_a.z - loc_b.z) ** 2
    )


def get_speed_kmh(vehicle: carla.Actor) -> float:
    v = vehicle.get_velocity()
    return 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)


def compute_ttc(dist_m: float, follower_speed_ms: float, leader_speed_ms: float) -> float:
    """Time-to-collision (seconds). Returns inf if closing speed <= 0."""
    closing = follower_speed_ms - leader_speed_ms
    if closing <= 0:
        return float("inf")
    return dist_m / closing


def cruise_control(
    vehicle: carla.Actor, target_kmh: float, kp: float = 0.15
) -> carla.VehicleControl:
    """Straight-line P-controller that holds `target_kmh`. Steer = 0."""
    error = target_kmh - get_speed_kmh(vehicle)
    throttle = max(0.0, min(1.0, kp * error))
    return carla.VehicleControl(throttle=throttle, steer=0.0, brake=0.0)
