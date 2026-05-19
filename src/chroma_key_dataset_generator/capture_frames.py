"""Capture still frames of a leader vehicle wearing the yellow chroma-key marker.

For each (town, weather, sun, distance) combination, spawns leader + follower
on a straight road, ticks the world once so physics & rendering settle, grabs
a frame from the follower's front camera, and saves it with a sidecar JSON of
the metadata. No PCLA agent, no driving — just photo sessions.

Output (default):
    data/chroma_key_dataset/<run_id>/
        0001.png           BGR frame from follower camera
        0001.json          {town, weather, sun_altitude, distance_m, ...}
        ...

Usage (on Vortex, with CARLA server running on :2000):
    python src/chroma_key_dataset_generator/capture_frames.py \\
        --towns Town06 Town04 \\
        --weather ClearNoon WetCloudyNoon HardRainNoon \\
        --sun-altitudes 60 20 -10 \\
        --distances 8 15 25 \\
        --leader vehicle.carlamotors.european_hgv

Defaults give 2 x 3 x 3 x 3 = 54 frames.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import carla
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "data" / "chroma_key_dataset"

IMAGE_W = 1280
IMAGE_H = 720
IMAGE_FOV = 90

# Built-in CARLA weather presets. Use the names that appear on carla.WeatherParameters
# (case-sensitive). We resolve them via getattr at runtime; unknown names error early.
DEFAULT_WEATHER = ["ClearNoon", "WetCloudyNoon", "HardRainNoon"]


def make_camera_bp(world):
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(IMAGE_W))
    bp.set_attribute("image_size_y", str(IMAGE_H))
    bp.set_attribute("fov", str(IMAGE_FOV))
    return bp


def save_frame(image: carla.Image, path: Path):
    """Save a CARLA Image (BGRA) as PNG."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    bgr = arr[..., :3].copy()
    import cv2
    cv2.imwrite(str(path), bgr)


def resolve_weather(name: str) -> carla.WeatherParameters:
    if not hasattr(carla.WeatherParameters, name):
        raise SystemExit(f"Unknown weather preset '{name}'. "
                         f"Try one of CARLA's WeatherParameters class attributes "
                         f"(ClearNoon, CloudyNoon, WetNoon, HardRainNoon, ClearSunset, ...).")
    return getattr(carla.WeatherParameters, name)


def find_straight_spawn(world, min_lane_length=80.0):
    """Pick a spawn point on a long enough driving lane (so the leader fits ahead)."""
    spawns = world.get_map().get_spawn_points()
    carla_map = world.get_map()
    for sp in spawns:
        wp = carla_map.get_waypoint(sp.location, project_to_road=True,
                                    lane_type=carla.LaneType.Driving)
        # Walk forward to verify we have room
        walked = 0.0
        cur = wp
        while walked < min_lane_length:
            nxt = cur.next(2.0)
            if not nxt:
                break
            cur = nxt[0]
            walked += 2.0
        if walked >= min_lane_length:
            return wp
    raise RuntimeError("No driving lane with sufficient length found.")


def walk_along_lane(wp: carla.Waypoint, distance_m: float) -> carla.Waypoint:
    """Return a waypoint 'distance_m' meters ahead along the same lane."""
    walked = 0.0
    cur = wp
    while walked < distance_m:
        nxt = cur.next(1.0)
        if not nxt:
            break
        cur = nxt[0]
        walked += 1.0
    return cur


def capture_one(world, leader_bp_name: str, follower_bp_name: str,
                distance_m: float, out_dir: Path, frame_id: str,
                meta: dict):
    bplib = world.get_blueprint_library()
    leader_bp = bplib.filter(leader_bp_name)[0]
    follower_bp = bplib.filter(follower_bp_name)[0]

    follower_wp = find_straight_spawn(world)
    leader_wp = walk_along_lane(follower_wp, distance_m)

    follower_tf = follower_wp.transform
    follower_tf.location.z += 0.5
    leader_tf = leader_wp.transform
    leader_tf.location.z += 0.5

    follower = world.spawn_actor(follower_bp, follower_tf)
    leader = world.spawn_actor(leader_bp, leader_tf)
    cam = None

    saved = {"received": False}

    try:
        cam_bp = make_camera_bp(world)
        cam_tf = carla.Transform(carla.Location(x=1.6, z=1.7))  # roughly windshield
        cam = world.spawn_actor(cam_bp, cam_tf, attach_to=follower)

        out_dir.mkdir(parents=True, exist_ok=True)
        img_path = out_dir / f"{frame_id}.png"

        def on_image(image):
            if saved["received"]:
                return
            save_frame(image, img_path)
            saved["received"] = True

        cam.listen(on_image)

        # Let physics / rendering settle. A few ticks for the world to update
        # weather and for the camera to deliver a frame.
        for _ in range(6):
            world.tick()
            time.sleep(0.02)

        # In case the listener was slow:
        deadline = time.time() + 2.0
        while not saved["received"] and time.time() < deadline:
            world.tick()
            time.sleep(0.02)

        if not saved["received"]:
            print(f"  [WARN] no image arrived for {frame_id}")
            return

        with open(out_dir / f"{frame_id}.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  [OK] {frame_id}.png  ({meta['town']}, {meta['weather']}, "
              f"sun={meta['sun_altitude']}°, dist={meta['distance_m']}m)")

    finally:
        if cam is not None:
            cam.stop()
            cam.destroy()
        leader.destroy()
        follower.destroy()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--towns", nargs="+", default=["Town06"])
    p.add_argument("--weather", nargs="+", default=DEFAULT_WEATHER)
    p.add_argument("--sun-altitudes", type=float, nargs="+", default=[60.0, 20.0, -10.0])
    p.add_argument("--distances", type=float, nargs="+", default=[8.0, 15.0, 25.0])
    p.add_argument("--leader", default="vehicle.carlamotors.european_hgv",
                   help="Leader vehicle blueprint id (must wear the yellow TGA in UE).")
    p.add_argument("--follower", default="vehicle.tesla.model3",
                   help="Follower vehicle blueprint id (camera-carrying ego).")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / f"capture_{ts}"

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    print(f"Connecting to CARLA at {args.host}:{args.port} ...")

    # Validate weather names up front so we fail fast.
    weather_presets = [(name, resolve_weather(name)) for name in args.weather]

    total = (len(args.towns) * len(weather_presets)
             * len(args.sun_altitudes) * len(args.distances))
    print(f"Will capture {total} frame(s) -> {run_dir}\n")

    counter = 0
    for town in args.towns:
        print(f"\n--- Loading {town} ---")
        world = client.load_world(town)
        # Sync mode so ticks are deterministic
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        for weather_name, weather_obj in weather_presets:
            for sun_alt in args.sun_altitudes:
                # Apply weather + override sun altitude
                w = carla.WeatherParameters(
                    cloudiness=weather_obj.cloudiness,
                    precipitation=weather_obj.precipitation,
                    precipitation_deposits=weather_obj.precipitation_deposits,
                    wind_intensity=weather_obj.wind_intensity,
                    sun_azimuth_angle=weather_obj.sun_azimuth_angle,
                    sun_altitude_angle=float(sun_alt),
                    fog_density=weather_obj.fog_density,
                    fog_distance=weather_obj.fog_distance,
                    wetness=weather_obj.wetness,
                )
                world.set_weather(w)

                for dist in args.distances:
                    counter += 1
                    frame_id = f"{counter:04d}"
                    meta = {
                        "town": town,
                        "weather": weather_name,
                        "sun_altitude": float(sun_alt),
                        "distance_m": float(dist),
                        "leader_blueprint": args.leader,
                        "follower_blueprint": args.follower,
                    }
                    try:
                        capture_one(world, args.leader, args.follower,
                                    dist, run_dir, frame_id, meta)
                    except Exception as e:
                        print(f"  [ERR] {frame_id}: {e}")

        # Restore async mode before switching town (avoids hang on load)
        settings.synchronous_mode = False
        world.apply_settings(settings)

    print(f"\nDone. {counter} frame(s) written to {run_dir}")


if __name__ == "__main__":
    main()
