"""Deterministic dataset capture for Fase 1 of the July 2026 plan.

Walks a chosen spawn point forward along the road at fixed step, captures one
frame per position with the follower's front camera. Every frame comes with a
sidecar JSON that lists ALL factor values (map, spawn, walk offset, sun,
weather, no_leader flag, ...), so the dataset can be re-sliced arbitrarily
later.

Design differences vs capture_frames.py (the older random-sampling generator):
  - No factor product loop — a single run captures ONE map × ONE light × ONE
    weather × ONE leader-mode (clean/marker/noleader). Combinatorial factors
    are handled at the shell level (call this script 6 times, once per combo).
  - Follower position walks along the road via wp.next(step) instead of being
    fixed at a "straight spawn". Matches test_leader_drive's new placement.
  - Leader placed at wp.next(follower_gap) — same road, same lane — so the
    "follower spawns in the previous junction" problem never occurs.
  - Zero NPCs (Fase 1 requirement: controlled scenario, no ambient traffic).
  - Long warmup (200 ticks) after town load so texture streaming settles;
    additional per-frame settle ticks handle spawn+physics settling.

Usage:
    python src/chroma_key_dataset_generator/capture_fase1.py \\
        --town Town04 --spawn 273 \\
        --walk-max 20 --walk-step 2 \\
        --sun-altitude 45 --sun-azimuth 90 --weather ClearNoon \\
        --leader-mode clean \\
        --out-dir data/chroma_key_dataset

Leader modes:
    clean       — CarlaCola with whatever texture the current CARLA package has
                  (start with the "clean" package for the clean pass)
    marker      — same as clean, but rename in the output folder — the actual
                  clean/marker difference is which package is currently
                  running on the server
    noleader    — skip the leader spawn entirely (follower alone on the road)
"""
from __future__ import annotations

import argparse
import json
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
CLIENT_TIMEOUT_S = 180.0


def make_camera_bp(world):
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(IMAGE_W))
    bp.set_attribute("image_size_y", str(IMAGE_H))
    bp.set_attribute("fov", str(IMAGE_FOV))
    return bp


def save_frame(image: carla.Image, path: Path) -> None:
    # CARLA's built-in save handles BGRA -> PNG conversion. No cv2/pillow needed.
    image.save_to_disk(str(path))


def resolve_weather(name: str) -> carla.WeatherParameters:
    if not hasattr(carla.WeatherParameters, name):
        raise SystemExit(f"unknown weather preset '{name}'")
    return getattr(carla.WeatherParameters, name)


def walk_forward(wp: carla.Waypoint, distance_m: float, step: float = 2.0) -> carla.Waypoint | None:
    cur = wp
    total = 0.0
    while total < distance_m:
        nxts = cur.next(step)
        if not nxts:
            return None
        cur = nxts[0]
        total += step
    return cur


def cleanup(world: carla.World) -> None:
    for actor in world.get_actors():
        try:
            if actor.type_id.startswith(("vehicle.", "sensor.")):
                actor.destroy()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--town", required=True)
    p.add_argument("--spawn", type=int, required=True, help="follower spawn index — leader goes 10 m ahead")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)

    p.add_argument("--walk-max", type=float, default=20.0, help="how far along the road to walk (m)")
    p.add_argument("--walk-step", type=float, default=2.0, help="distance between poses (m)")
    p.add_argument("--leader-gap-m", type=float, default=10.0)

    p.add_argument("--sun-altitude", type=float, default=45.0)
    p.add_argument("--sun-azimuth", type=float, default=90.0)
    p.add_argument("--weather", default="ClearNoon", help="carla.WeatherParameters preset name")

    p.add_argument("--leader-mode", choices=["clean", "marker", "noleader"], required=True,
                   help="just a folder-naming hint; whether the leader shows the yellow marker or"
                        " a clean texture depends on which CARLA package is currently running")
    p.add_argument("--leader-bp", default="vehicle.carlamotors.carlacola")
    p.add_argument("--follower-bp", default="vehicle.tesla.model3")

    p.add_argument("--settle-ticks", type=int, default=30, help="ticks between spawn and camera capture, for physics + rendering to settle")
    p.add_argument("--warmup-ticks", type=int, default=200, help="ticks after town load, for texture streaming")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--run-tag", default=None, help="folder tag; defaults to timestamp_town_spawn")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.run_tag or f"{ts}_{args.town}_spawn{args.spawn}_{args.leader_mode}"
    run_dir = args.out_dir / f"fase1_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output: {run_dir}")

    print(f"[INFO] Connecting to CARLA at {args.host}:{args.port}")
    client = carla.Client(args.host, args.port)
    client.set_timeout(CLIENT_TIMEOUT_S)

    print(f"[INFO] Loading world '{args.town}'")
    world = client.load_world(args.town)
    world.tick()

    # synchronous mode for deterministic tick sequence
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # weather
    w = resolve_weather(args.weather)
    w = carla.WeatherParameters(
        cloudiness=w.cloudiness,
        precipitation=w.precipitation,
        precipitation_deposits=w.precipitation_deposits,
        wind_intensity=w.wind_intensity,
        sun_azimuth_angle=args.sun_azimuth,
        sun_altitude_angle=args.sun_altitude,
        fog_density=w.fog_density,
        fog_distance=w.fog_distance,
        wetness=w.wetness,
    )
    world.set_weather(w)
    print(f"[INFO] Weather '{args.weather}' with sun_alt={args.sun_altitude:.1f} sun_az={args.sun_azimuth:.1f}")

    # warmup for texture streaming
    print(f"[INFO] Warmup {args.warmup_ticks} ticks...")
    for _ in range(args.warmup_ticks):
        world.tick()

    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    if args.spawn >= len(spawn_points):
        raise SystemExit(f"spawn {args.spawn} out of range (max {len(spawn_points)-1})")
    base_sp = spawn_points[args.spawn]
    base_wp = carla_map.get_waypoint(
        base_sp.location, project_to_road=True, lane_type=carla.LaneType.Driving
    )
    if base_wp is None:
        raise SystemExit(f"spawn {args.spawn} does not project to a driving lane")

    print(f"[INFO] Spawn {args.spawn}: loc=({base_sp.location.x:.1f},{base_sp.location.y:.1f}) "
          f"road={base_wp.road_id} lane={base_wp.lane_id}")

    bplib = world.get_blueprint_library()
    leader_bp = bplib.filter(args.leader_bp)[0]
    follower_bp = bplib.filter(args.follower_bp)[0]

    # sanity: walk to the end and see how many poses we can afford
    n_poses = int(args.walk_max // args.walk_step) + 1
    print(f"[INFO] Capturing {n_poses} poses (walk 0..{args.walk_max} m step {args.walk_step} m)")
    print(f"[INFO] Leader gap = {args.leader_gap_m} m ahead of follower")
    print(f"[INFO] Leader mode = {args.leader_mode}")

    ok_count = 0
    for i in range(n_poses):
        walk_m = i * args.walk_step
        frame_id = f"{i+1:06d}"
        print(f"\n[{frame_id}] walk={walk_m:.1f}m", flush=True)

        cleanup(world)
        world.tick()

        follower_wp = walk_forward(base_wp, walk_m, args.walk_step)
        if follower_wp is None:
            print(f"  [SKIP] can't walk {walk_m:.1f}m from spawn")
            continue
        if args.leader_mode != "noleader":
            leader_wp = walk_forward(follower_wp, args.leader_gap_m, args.walk_step)
            if leader_wp is None:
                print(f"  [SKIP] can't place leader {args.leader_gap_m}m ahead")
                continue

        # spawn follower
        follower_tf = follower_wp.transform
        follower_tf.location.z += 0.10
        follower = world.try_spawn_actor(follower_bp, follower_tf)
        if follower is None:
            print(f"  [SKIP] follower spawn failed")
            continue
        world.tick()

        leader = None
        if args.leader_mode != "noleader":
            leader_tf = leader_wp.transform
            leader_tf.location.z += 0.10
            leader = world.try_spawn_actor(leader_bp, leader_tf)
            if leader is None:
                follower.destroy()
                print(f"  [SKIP] leader spawn failed")
                continue
            world.tick()

        # attach camera to follower
        cam_bp = make_camera_bp(world)
        cam_tf = carla.Transform(carla.Location(x=1.2, y=0.0, z=1.5))
        cam = world.spawn_actor(cam_bp, cam_tf, attach_to=follower)

        got = {"img": None}
        def _on(image, container=got):
            container["img"] = image
        cam.listen(_on)

        # settle
        for _ in range(args.settle_ticks):
            world.tick()

        img = got["img"]
        cam.stop()
        cam.destroy()

        if img is None:
            print(f"  [SKIP] no camera image received")
            if leader is not None:
                leader.destroy()
            follower.destroy()
            continue

        png_path = run_dir / f"{frame_id}.png"
        json_path = run_dir / f"{frame_id}.json"
        save_frame(img, png_path)

        meta = {
            "frame_id": frame_id,
            "town": args.town,
            "spawn_idx": args.spawn,
            "walk_offset_m": float(walk_m),
            "leader_gap_m": float(args.leader_gap_m),
            "leader_mode": args.leader_mode,
            "leader_bp": args.leader_bp,
            "follower_bp": args.follower_bp,
            "weather": args.weather,
            "sun_altitude": float(args.sun_altitude),
            "sun_azimuth": float(args.sun_azimuth),
            "follower_loc": [follower_tf.location.x, follower_tf.location.y, follower_tf.location.z],
            "follower_yaw_deg": float(follower_tf.rotation.yaw),
            "follower_road_id": follower_wp.road_id,
            "follower_lane_id": follower_wp.lane_id,
        }
        if leader is not None:
            meta["leader_loc"] = [leader_tf.location.x, leader_tf.location.y, leader_tf.location.z]
            meta["leader_yaw_deg"] = float(leader_tf.rotation.yaw)
            meta["leader_road_id"] = leader_wp.road_id
            meta["leader_lane_id"] = leader_wp.lane_id
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

        # cleanup this pose
        if leader is not None:
            leader.destroy()
        follower.destroy()
        world.tick()

        ok_count += 1
        print(f"  [OK] saved {frame_id}.png + json")

    # restore async mode so we don't leave the world stuck in sync
    try:
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
    except Exception:
        pass

    print(f"\n[INFO] Done. Saved {ok_count}/{n_poses} frames to {run_dir}")


if __name__ == "__main__":
    main()
