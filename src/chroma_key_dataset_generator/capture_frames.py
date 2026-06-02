"""Capture still frames of a leader vehicle wearing the yellow chroma-key marker.

Generates a balanced grid dataset for adversarial-patch training. For every
combination in the product of:
  (town × spawn × weather × sun_altitude × distance × lateral_offset × heading_offset)
the script:
  1. Spawns leader + follower on a long-enough lane, with optional lateral
     offset of the leader and rotation of the follower's heading.
  2. Optionally spawns NPC traffic in a radius around the scene, in autopilot.
  3. Ticks the world a few times so weather/physics/rendering settle.
  4. Grabs one frame from the follower's front camera + saves a sidecar JSON
     with all generation parameters.

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
        --npc-count 10 --npc-radius 60 \\
        [--shuffle] [--max-frames 1000]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
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

# 180 s timeout covers heavy town loads (Town10HD, Town12) where the server
# stops answering RPCs for a minute or more while UE compiles shaders.
CLIENT_TIMEOUT_S = 180.0

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
    """Shift `tf` perpendicular to its forward axis by `lateral` meters and rotate
    its yaw by `heading_deg`. Used only for the heading offset of the follower
    now; lateral shift of the leader is done via lane shifts (`shift_lanes`)."""
    yaw = math.radians(tf.rotation.yaw)
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


def shift_lanes(wp: carla.Waypoint, shift: int):
    """Move `shift` lanes left (negative) or right (positive) from `wp`.
    Returns the resulting waypoint, or None if no valid Driving lane exists
    that far in the requested direction.

    Lane direction convention follows the waypoint's own forward axis:
        shift = -1 -> one lane to the left of the vehicle
        shift = +1 -> one lane to the right
    """
    if shift == 0:
        return wp
    cur = wp
    for _ in range(abs(shift)):
        if cur is None:
            return None
        nxt = cur.get_right_lane() if shift > 0 else cur.get_left_lane()
        if nxt is None or nxt.lane_type != carla.LaneType.Driving:
            return None
        cur = nxt
    return cur


def spawn_npc_traffic(world, anchor_loc: carla.Location, count: int, radius_m: float,
                      tm_port: int = 8000) -> list:
    """Spawn up to `count` random vehicles within `radius_m` of `anchor_loc`,
    each in autopilot. Returns the spawned actors (caller destroys them)."""
    if count <= 0:
        return []
    bplib = world.get_blueprint_library()
    vehicle_bps = [bp for bp in bplib.filter("vehicle.*")
                   if int(bp.get_attribute("number_of_wheels")) == 4]
    spawn_points = world.get_map().get_spawn_points()
    nearby = [sp for sp in spawn_points
              if sp.location.distance(anchor_loc) <= radius_m]
    random.shuffle(nearby)

    spawned = []
    for sp in nearby[:count * 3]:
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


def safe_destroy(actor):
    """Destroy an actor only if it's still alive on the server. Mirrors the
    PCLA.cleanup() pattern: check is_listening on sensors, is_alive on vehicles."""
    if actor is None:
        return
    try:
        if hasattr(actor, "is_listening") and actor.is_listening():
            actor.stop()
    except Exception:
        pass
    try:
        if not hasattr(actor, "is_alive") or actor.is_alive:
            actor.destroy()
    except Exception:
        pass


def cleanup_world(world):
    """Wipe stray sensors from the world via server's actor list.
    Do NOT touch vehicles: any vehicle the TrafficManager owns would crash the
    server on its next tick. We trust the per-frame finally for vehicles."""
    if world is None:
        return
    try:
        for sensor in world.get_actors().filter("*sensor*"):
            try:
                if sensor.is_listening():
                    sensor.stop()
                sensor.destroy()
            except Exception:
                pass
    except Exception:
        pass


# ---------- per-frame capture ---------------------------------------------

def capture_one(world, leader_bp, follower_bp,
                follower_wp: carla.Waypoint,
                distance_m: float, lateral_offset: float, heading_offset_deg: float,
                npc_count: int, npc_radius: float,
                out_dir: Path, frame_id: str, meta: dict,
                settle_ticks: int = 50):
    """Spawn leader+follower(+NPCs), grab one camera frame, destroy everything."""
    # PCLA-style preflight: wipe any leftover sensor/vehicle from a previous
    # frame whose finally block didn't fully complete. Source of truth is the
    # CARLA server, not our Python references.
    cleanup_world(world)
    try:
        world.tick()
    except Exception:
        pass

    leader_wp = walk_along_lane(follower_wp, distance_m)

    # Apply lateral as lane shift (integer): 0 = same lane, -1/+1 = adjacent.
    # Skip combos that fall off the road network.
    shift = int(round(lateral_offset))
    shifted_leader_wp = shift_lanes(leader_wp, shift)
    if shifted_leader_wp is None:
        print(f"  [SKIP] {frame_id}: no valid lane at shift={shift:+d} from follower lane",
              flush=True)
        return False
    leader_wp = shifted_leader_wp

    # Small z bump to avoid spawning *inside* the road mesh; large bump means
    # vehicles have to fall a long way during settle.
    Z_BUMP = 0.10

    follower_tf = follower_wp.transform
    follower_tf.location.z += Z_BUMP
    follower_tf = offset_transform(follower_tf, lateral=0.0, heading_deg=heading_offset_deg)

    leader_tf = leader_wp.transform
    leader_tf.location.z += Z_BUMP

    print(f"  [..] {frame_id}: spawn follower", flush=True)
    follower = world.try_spawn_actor(follower_bp, follower_tf)
    if follower is None:
        print(f"  [SKIP] {frame_id}: follower spawn failed", flush=True)
        return False
    try:
        follower.set_simulate_physics(True)
    except Exception:
        pass
    world.tick()   # commit follower

    print(f"  [..] {frame_id}: spawn leader", flush=True)
    leader = world.try_spawn_actor(leader_bp, leader_tf)
    if leader is None:
        safe_destroy(follower)
        print(f"  [SKIP] {frame_id}: leader spawn failed", flush=True)
        return False
    try:
        leader.set_simulate_physics(True)
    except Exception:
        pass
    world.tick()   # commit leader

    print(f"  [..] {frame_id}: spawn NPCs (count={npc_count})", flush=True)
    npcs = spawn_npc_traffic(world, anchor_loc=follower_tf.location,
                             count=npc_count, radius_m=npc_radius)
    world.tick()   # commit NPCs

    cam = None
    saved = {"received": False}
    try:
        print(f"  [..] {frame_id}: spawn camera", flush=True)
        cam_bp = make_camera_bp(world)
        cam_tf = carla.Transform(carla.Location(x=1.6, z=1.7))
        cam = world.spawn_actor(cam_bp, cam_tf, attach_to=follower)
        world.tick()   # commit camera before .listen()

        out_dir.mkdir(parents=True, exist_ok=True)
        img_path = out_dir / f"{frame_id}.png"

        # Counter to skip the first N camera frames the renderer produces
        # (they're stale: the render thread is behind sync ticks; the first
        # ~15 frames after `cam.listen()` show the pre-settle state, including
        # shader-not-yet-compiled artifacts in the very first town session).
        SKIP_FRAMES = 20
        captured = {"count": 0}

        def on_image(image):
            captured["count"] += 1
            if captured["count"] <= SKIP_FRAMES:
                return     # discard early stale frames
            if saved["received"]:
                return
            save_frame(image, img_path)
            saved["received"] = True

        # FIRST settle physics WITHOUT a listener so we don't accidentally
        # capture the very first frame (vehicles still in mid-air).
        print(f"  [..] {frame_id}: settle (no listener)", flush=True)
        for i in range(settle_ticks):
            world.tick()
            if (i + 1) % 10 == 0:
                print(f"  [..] {frame_id}: tick {i+1}/{settle_ticks}", flush=True)

        # NOW register listener. The first SKIP_FRAMES camera frames are
        # discarded (renderer is still catching up); we save the next one.
        print(f"  [..] {frame_id}: cam.listen (skip {SKIP_FRAMES} stale frames)", flush=True)
        cam.listen(on_image)

        deadline = time.time() + 6.0
        while not saved["received"] and time.time() < deadline:
            world.tick()
            time.sleep(0.02)

        if not saved["received"]:
            print(f"  [WARN] no image for {frame_id}", flush=True)
            return False

        with open(out_dir / f"{frame_id}.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  [OK] {frame_id}.png  ({meta['town']}, {meta['weather']}, "
              f"sun={meta['sun_altitude']:.0f}°, dist={meta['distance_m']:.0f}m, "
              f"lat={meta['lateral_offset']:+.1f}, hdg={meta['heading_offset']:+.0f}°, "
              f"npcs={len(npcs)})", flush=True)
        return True

    finally:
        print(f"  [..] {frame_id}: finally start", flush=True)
        if cam is not None:
            try:
                if cam.is_listening():
                    cam.stop()
            except Exception:
                pass
        print(f"  [..] {frame_id}: drain tick", flush=True)
        try:
            world.tick()
        except Exception as e:
            print(f"  [..] {frame_id}: drain tick raised {e}", flush=True)
        print(f"  [..] {frame_id}: destroy cam", flush=True)
        safe_destroy(cam)
        print(f"  [..] {frame_id}: destroy {len(npcs)} npcs", flush=True)
        for n in npcs:
            safe_destroy(n)
        print(f"  [..] {frame_id}: destroy leader", flush=True)
        safe_destroy(leader)
        print(f"  [..] {frame_id}: destroy follower", flush=True)
        safe_destroy(follower)
        print(f"  [..] {frame_id}: post-cleanup tick", flush=True)
        try:
            world.tick()
        except Exception as e:
            print(f"  [..] {frame_id}: post-cleanup tick raised {e}", flush=True)
        print(f"  [..] {frame_id}: finally end", flush=True)


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
                   default=[-1, 0, 1],
                   help="Lane shifts (NOT meters): integer count of lanes to "
                        "move the LEADER relative to the follower's lane. "
                        "-1 = one lane left, +1 = one lane right. Combos that "
                        "have no valid driving lane in that direction are "
                        "skipped. Non-integer values are rounded.")
    p.add_argument("--heading-offsets", type=float, nargs="+",
                   default=[-5.0, 0.0, 5.0])
    p.add_argument("--npc-count", type=int, default=0,
                   help="NPC vehicles to spawn in autopilot. WARNING: with "
                        "autopilot on, the TrafficManager owns these actors "
                        "and can crash the server if they collide / disappear "
                        "during the settle ticks. Keep this 0 unless tested.")
    p.add_argument("--npc-radius", type=float, default=60.0)
    p.add_argument("--settle-ticks", type=int, default=50,
                   help="World ticks (each 0.05 s sim time) between spawn and "
                        "frame capture. Higher = vehicles fall to ground more, "
                        "weather/lighting fully apply. 50 = 2.5 s of sim time.")
    p.add_argument("--spawn-pool-size", type=int, default=4,
                   help="Distinct starting waypoints per town.")
    p.add_argument("--leader", default="vehicle.carlamotors.carlacola")
    p.add_argument("--follower", default="vehicle.tesla.model3")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=None,
                   help="Cap. With --shuffle, this samples uniformly across "
                        "the grid; without, takes the first max_frames combos.")
    p.add_argument("--shuffle", action="store_true",
                   help="Sample combinations randomly from the full grid "
                        "(then re-sort by town so each town loads once).")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / f"capture_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / "captures_index.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_id", "town", "spawn_idx", "weather",
                         "sun_altitude", "distance_m", "lateral_offset",
                         "heading_offset", "npc_count", "leader_bp",
                         "follower_bp"])

    client = carla.Client(args.host, args.port)
    client.set_timeout(CLIENT_TIMEOUT_S)
    print(f"Connecting to CARLA at {args.host}:{args.port} "
          f"(timeout={CLIENT_TIMEOUT_S:.0f}s) ...")

    weather_presets = [(name, resolve_weather(name)) for name in args.weather]

    # Build the FULL grid of combinations. Each combo carries every value it
    # needs so we can shuffle / slice independent of the town iteration order.
    combos = []
    town_order = {t: i for i, t in enumerate(args.towns)}
    for town in args.towns:
        for spawn_idx in range(args.spawn_pool_size):
            for weather_name, weather_obj in weather_presets:
                for sun_alt in args.sun_altitudes:
                    for dist in args.distances:
                        for lat in args.lateral_offsets:
                            for hdg in args.heading_offsets:
                                combos.append({
                                    "town": town,
                                    "spawn_idx": spawn_idx,
                                    "weather_name": weather_name,
                                    "weather_obj": weather_obj,
                                    "sun_alt": sun_alt,
                                    "dist": dist,
                                    "lat": lat,
                                    "hdg": hdg,
                                })

    print(f"Full grid: {len(combos)} combinations across {len(args.towns)} towns")

    if args.shuffle:
        random.shuffle(combos)
    if args.max_frames:
        combos = combos[:args.max_frames]
    # Group by town so each town is loaded only once
    combos.sort(key=lambda c: town_order[c["town"]])

    # Count per town after sampling (informational)
    per_town = {}
    for c in combos:
        per_town[c["town"]] = per_town.get(c["town"], 0) + 1
    print(f"Sampled : {len(combos)} frame(s) "
          f"({'shuffled' if args.shuffle else 'deterministic'})")
    for t in args.towns:
        n = per_town.get(t, 0)
        if n > 0:
            print(f"   {t:20s}  {n}")
    print(f"Out     : {run_dir}\n")

    counter = 0
    current_town = None
    world = None
    spawn_pool = None
    leader_bp = follower_bp = None

    try:
        for combo in combos:
            # Town switch
            if combo["town"] != current_town:
                if world is not None:
                    # Cleanup any stray sensors/NPCs before unloading the world
                    cleanup_world(world)
                    try:
                        settings = world.get_settings()
                        settings.synchronous_mode = False
                        world.apply_settings(settings)
                    except Exception:
                        pass
                print(f"\n--- Loading {combo['town']} ---")
                world = client.load_world(combo["town"])
                # Give UE a moment to settle the new map before we touch it
                time.sleep(3.0)
                settings = world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
                world.apply_settings(settings)
                # Long warmup so UE finishes shader compile / texture streaming
                # before the first capture. Without this, frames 1-4 come back
                # half-rendered (stale buffers).
                print(f"  warmup: 100 ticks", flush=True)
                for _ in range(100):
                    world.tick()

                bplib = world.get_blueprint_library()
                try:
                    leader_bp = bplib.filter(args.leader)[0]
                    follower_bp = bplib.filter(args.follower)[0]
                except IndexError:
                    print(f"  [ERR] blueprint not found in {combo['town']}, skipping town")
                    current_town = combo["town"]
                    continue

                all_spawns = find_all_straight_spawns(world, min_lane_length=80.0)
                random.shuffle(all_spawns)
                spawn_pool = all_spawns[:args.spawn_pool_size]
                current_town = combo["town"]

            # Apply weather
            wo = combo["weather_obj"]
            w = carla.WeatherParameters(
                cloudiness=wo.cloudiness,
                precipitation=wo.precipitation,
                precipitation_deposits=wo.precipitation_deposits,
                wind_intensity=wo.wind_intensity,
                sun_azimuth_angle=wo.sun_azimuth_angle,
                sun_altitude_angle=float(combo["sun_alt"]),
                fog_density=wo.fog_density,
                fog_distance=wo.fog_distance,
                wetness=wo.wetness,
            )
            world.set_weather(w)

            # Pick the spawn
            spawn_idx = combo["spawn_idx"] % len(spawn_pool)
            follower_wp = spawn_pool[spawn_idx]

            counter += 1
            frame_id = f"{counter:06d}"
            meta = {
                "town": combo["town"],
                "spawn_idx": spawn_idx,
                "weather": combo["weather_name"],
                "sun_altitude": float(combo["sun_alt"]),
                "distance_m": float(combo["dist"]),
                "lateral_offset": float(combo["lat"]),
                "heading_offset": float(combo["hdg"]),
                "npc_count_requested": args.npc_count,
                "leader_blueprint": args.leader,
                "follower_blueprint": args.follower,
            }
            try:
                ok = capture_one(world, leader_bp, follower_bp, follower_wp,
                                 combo["dist"], combo["lat"], combo["hdg"],
                                 args.npc_count, args.npc_radius,
                                 run_dir, frame_id, meta,
                                 settle_ticks=args.settle_ticks)
                if ok:
                    csv_writer.writerow([frame_id, combo["town"], spawn_idx,
                                         combo["weather_name"], combo["sun_alt"],
                                         combo["dist"], combo["lat"], combo["hdg"],
                                         args.npc_count, args.leader, args.follower])
                    csv_file.flush()
            except Exception as e:
                import traceback
                print(f"  [ERR] {frame_id}: {type(e).__name__}: {e}")
                traceback.print_exc()
    finally:
        csv_file.close()

    print(f"\nDone. {counter} frame(s) attempted → {run_dir}")
    print(f"Index: {csv_path}")


if __name__ == "__main__":
    main()
