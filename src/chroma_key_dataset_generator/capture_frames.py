"""Capture still frames of a leader vehicle wearing the yellow chroma-key marker.

Generates a balanced grid dataset for adversarial-patch training. For every
combination in the product of:
  (town × weather × sun_altitude × distance × lateral_offset × heading_offset)
the script:
  1. Spawns leader + follower on a long-enough lane, with optional lateral
     offset of the leader and rotation of the follower's heading.
  2. Optionally spawns NPC traffic in a radius around the scene, started in
     autopilot so the frame contains plausibly moving vehicles.
  3. Ticks the world a few times so weather/physics/rendering settle.
  4. Grabs one frame from the follower's front camera + saves a sidecar JSON
     with all the generation parameters.

Output:
    data/chroma_key_dataset/capture_<ts>/
        000001.png + 000001.json
        ...
        captures_index.csv      one row per frame, all params

Usage (on Vortex, with CARLA server running on :2000):
    python src/chroma_key_dataset_generator/capture_frames.py \\
        --towns Town04 Town06 Town10HD_Opt \\
        --weather ClearNoon CloudyNoon WetNoon MidRainyNoon \\
        --sun-altitudes 70 55 40 25 \\
        --distances 6 10 14 18 \\
        --lateral-offsets -1.5 0 1.5 \\
        --heading-offsets -5 0 5 \\
        --npc-count 10 --npc-radius 60

Defaults give a ~1000-frame grid.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
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

DEFAULT_WEATHER = ["ClearNoon", "CloudyNoon", "WetNoon", "MidRainyNoon"]


# ---------- helpers --------------------------------------------------------

def make_camera_bp(world):
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(IMAGE_W))
    bp.set_attribute("image_size_y", str(IMAGE_H))
    bp.set_attribute("fov", str(IMAGE_FOV))
    return bp


def save_frame(image: carla.Image, path: Path):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    bgr = arr[..., :3].copy()
    import cv2
    cv2.imwrite(str(path), bgr)


def resolve_weather(name: str) -> carla.WeatherParameters:
    if not hasattr(carla.WeatherParameters, name):
        raise SystemExit(f"Unknown weather preset '{name}'. Examples: "
                         f"ClearNoon, CloudyNoon, WetNoon, MidRainyNoon, "
                         f"ClearSunset, HardRainNoon, ...")
    return getattr(carla.WeatherParameters, name)


def find_all_straight_spawns(world, min_lane_length=80.0):
    """Return every spawn point whose lane keeps going forward >= min_lane_length."""
    spawns = world.get_map().get_spawn_points()
    carla_map = world.get_map()
    good = []
    for sp in spawns:
        wp = carla_map.get_waypoint(sp.location, project_to_road=True,
                                    lane_type=carla.LaneType.Driving)
        if wp is None:
            continue
        walked = 0.0
        cur = wp
        while walked < min_lane_length:
            nxt = cur.next(2.0)
            if not nxt:
                break
            cur = nxt[0]
            walked += 2.0
        if walked >= min_lane_length:
            good.append(wp)
    if not good:
        raise RuntimeError("No driving lane with sufficient length found.")
    return good


def walk_along_lane(wp: carla.Waypoint, distance_m: float) -> carla.Waypoint:
    walked = 0.0
    cur = wp
    while walked < distance_m:
        nxt = cur.next(1.0)
        if not nxt:
            break
        cur = nxt[0]
        walked += 1.0
    return cur


def offset_transform(tf: carla.Transform, lateral: float, heading_deg: float) -> carla.Transform:
    """Return a copy of `tf` shifted laterally (perpendicular to its forward axis)
    and rotated in yaw by `heading_deg`."""
    yaw = math.radians(tf.rotation.yaw)
    # Right vector in CARLA: (cos(yaw+90°), sin(yaw+90°)) = (-sin yaw, cos yaw)
    right_x = -math.sin(yaw)
    right_y = math.cos(yaw)
    new_loc = carla.Location(
        x=tf.location.x + lateral * right_x,
        y=tf.location.y + lateral * right_y,
        z=tf.location.z,
    )
    new_rot = carla.Rotation(
        pitch=tf.rotation.pitch,
        yaw=tf.rotation.yaw + heading_deg,
        roll=tf.rotation.roll,
    )
    return carla.Transform(new_loc, new_rot)


def spawn_npc_traffic(world, anchor_loc: carla.Location, count: int, radius_m: float,
                      tm_port: int = 8000) -> list:
    """Spawn `count` random vehicles within `radius_m` of `anchor_loc`, enable autopilot.

    Returns the list of spawned actors (caller must destroy)."""
    if count <= 0:
        return []
    bplib = world.get_blueprint_library()
    vehicle_bps = [bp for bp in bplib.filter("vehicle.*")
                   if int(bp.get_attribute("number_of_wheels")) == 4]
    spawn_points = world.get_map().get_spawn_points()
    # Pick spawn points within radius
    nearby = [sp for sp in spawn_points
              if sp.location.distance(anchor_loc) <= radius_m]
    random.shuffle(nearby)

    tm = None
    try:
        tm = world.get_blueprint_library()  # noqa: dummy
    except Exception:
        pass

    spawned = []
    for sp in nearby[:count * 3]:  # try up to 3x to handle collisions
        if len(spawned) >= count:
            break
        bp = random.choice(vehicle_bps)
        actor = world.try_spawn_actor(bp, sp)
        if actor is not None:
            spawned.append(actor)
            try:
                actor.set_autopilot(True, tm_port)
            except Exception:
                pass
    return spawned


# ---------- per-frame capture ---------------------------------------------

def capture_one(world, leader_bp, follower_bp,
                follower_wp: carla.Waypoint,
                distance_m: float, lateral_offset: float, heading_offset_deg: float,
                npc_count: int, npc_radius: float,
                out_dir: Path, frame_id: str, meta: dict):
    """Spawn leader+follower(+NPCs), grab one camera frame, destroy everything."""
    leader_wp = walk_along_lane(follower_wp, distance_m)

    # Base transforms, slight z bump to avoid colliding with ground
    follower_tf = follower_wp.transform
    follower_tf.location.z += 0.5
    follower_tf = offset_transform(follower_tf, lateral=0.0, heading_deg=heading_offset_deg)

    leader_tf = leader_wp.transform
    leader_tf.location.z += 0.5
    leader_tf = offset_transform(leader_tf, lateral=lateral_offset, heading_deg=0.0)

    follower = world.try_spawn_actor(follower_bp, follower_tf)
    if follower is None:
        print(f"  [SKIP] {frame_id}: follower spawn failed")
        return
    leader = world.try_spawn_actor(leader_bp, leader_tf)
    if leader is None:
        follower.destroy()
        print(f"  [SKIP] {frame_id}: leader spawn failed")
        return

    npcs = spawn_npc_traffic(world, anchor_loc=follower_tf.location,
                             count=npc_count, radius_m=npc_radius)

    cam = None
    saved = {"received": False}
    try:
        cam_bp = make_camera_bp(world)
        cam_tf = carla.Transform(carla.Location(x=1.6, z=1.7))
        cam = world.spawn_actor(cam_bp, cam_tf, attach_to=follower)

        out_dir.mkdir(parents=True, exist_ok=True)
        img_path = out_dir / f"{frame_id}.png"

        def on_image(image):
            if saved["received"]:
                return
            save_frame(image, img_path)
            saved["received"] = True

        cam.listen(on_image)

        # Let weather/physics/rendering settle
        for _ in range(6):
            world.tick()

        deadline = time.time() + 2.0
        while not saved["received"] and time.time() < deadline:
            world.tick()
            time.sleep(0.02)

        if not saved["received"]:
            print(f"  [WARN] no image for {frame_id}")
            return

        with open(out_dir / f"{frame_id}.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  [OK] {frame_id}.png  ({meta['town']}, {meta['weather']}, "
              f"sun={meta['sun_altitude']:.0f}°, dist={meta['distance_m']:.0f}m, "
              f"lat={meta['lateral_offset']:+.1f}, hdg={meta['heading_offset']:+.0f}°, "
              f"npcs={len(npcs)})")

    finally:
        if cam is not None:
            cam.stop()
            cam.destroy()
        for n in npcs:
            try:
                n.destroy()
            except Exception:
                pass
        if leader is not None:
            leader.destroy()
        if follower is not None:
            follower.destroy()


# ---------- main -----------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--towns", nargs="+",
                   default=["Town04", "Town06", "Town10HD_Opt"])
    p.add_argument("--weather", nargs="+", default=DEFAULT_WEATHER)
    p.add_argument("--sun-altitudes", type=float, nargs="+",
                   default=[70.0, 55.0, 40.0, 25.0])
    p.add_argument("--distances", type=float, nargs="+",
                   default=[6.0, 10.0, 14.0, 18.0])
    p.add_argument("--lateral-offsets", type=float, nargs="+",
                   default=[-1.5, 0.0, 1.5])
    p.add_argument("--heading-offsets", type=float, nargs="+",
                   default=[-5.0, 0.0, 5.0])
    p.add_argument("--npc-count", type=int, default=10,
                   help="NPCs to spawn around the scene (autopilot on).")
    p.add_argument("--npc-radius", type=float, default=60.0,
                   help="Radius around the follower in which to spawn NPCs (m).")
    p.add_argument("--spawn-pool-size", type=int, default=4,
                   help="How many distinct starting waypoints to cycle through "
                        "per town (gives more visual variety).")
    p.add_argument("--leader", default="vehicle.carlamotors.carlacola",
                   help="Leader blueprint id (must wear the yellow TGA in UE).")
    p.add_argument("--follower", default="vehicle.tesla.model3")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=None,
                   help="Optional cap. If grid > this, the script stops early.")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / f"capture_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # CSV index — one row per frame
    csv_path = run_dir / "captures_index.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_id", "town", "spawn_idx", "weather",
                         "sun_altitude", "distance_m", "lateral_offset",
                         "heading_offset", "npc_count", "leader_bp",
                         "follower_bp"])

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    print(f"Connecting to CARLA at {args.host}:{args.port} ...")

    weather_presets = [(name, resolve_weather(name)) for name in args.weather]

    # Total grid size (assuming spawn_pool_size waypoints per town)
    n_combo_per_spawn = (len(weather_presets) * len(args.sun_altitudes)
                        * len(args.distances) * len(args.lateral_offsets)
                        * len(args.heading_offsets))
    total = len(args.towns) * args.spawn_pool_size * n_combo_per_spawn
    if args.max_frames:
        total = min(total, args.max_frames)
    print(f"Grid: {len(args.towns)} towns × {args.spawn_pool_size} spawns × "
          f"{n_combo_per_spawn} per-spawn combos = {total} frame(s)")
    print(f"Out:  {run_dir}\n")

    counter = 0
    try:
        for town in args.towns:
            print(f"\n--- Loading {town} ---")
            world = client.load_world(town)
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)

            bplib = world.get_blueprint_library()
            try:
                leader_bp = bplib.filter(args.leader)[0]
                follower_bp = bplib.filter(args.follower)[0]
            except IndexError:
                print(f"  [ERR] blueprint not found in {town}")
                continue

            all_spawns = find_all_straight_spawns(world, min_lane_length=80.0)
            random.shuffle(all_spawns)
            spawn_pool = all_spawns[:args.spawn_pool_size]

            for spawn_idx, follower_wp in enumerate(spawn_pool):
                for weather_name, weather_obj in weather_presets:
                    for sun_alt in args.sun_altitudes:
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
                            for lat in args.lateral_offsets:
                                for hdg in args.heading_offsets:
                                    if args.max_frames and counter >= args.max_frames:
                                        raise StopIteration
                                    counter += 1
                                    frame_id = f"{counter:06d}"
                                    meta = {
                                        "town": town,
                                        "spawn_idx": spawn_idx,
                                        "weather": weather_name,
                                        "sun_altitude": float(sun_alt),
                                        "distance_m": float(dist),
                                        "lateral_offset": float(lat),
                                        "heading_offset": float(hdg),
                                        "npc_count_requested": args.npc_count,
                                        "leader_blueprint": args.leader,
                                        "follower_blueprint": args.follower,
                                    }
                                    try:
                                        capture_one(world, leader_bp, follower_bp,
                                                    follower_wp, dist, lat, hdg,
                                                    args.npc_count, args.npc_radius,
                                                    run_dir, frame_id, meta)
                                        csv_writer.writerow([frame_id, town, spawn_idx,
                                                             weather_name, sun_alt,
                                                             dist, lat, hdg,
                                                             args.npc_count,
                                                             args.leader, args.follower])
                                        csv_file.flush()
                                    except Exception as e:
                                        print(f"  [ERR] {frame_id}: {e}")

            # async mode before town swap, prevents hang
            settings.synchronous_mode = False
            world.apply_settings(settings)

    except StopIteration:
        pass
    finally:
        csv_file.close()

    print(f"\nDone. {counter} frame(s) written to {run_dir}")
    print(f"Index: {csv_path}")


if __name__ == "__main__":
    main()
