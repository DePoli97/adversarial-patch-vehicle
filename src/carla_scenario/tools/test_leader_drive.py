"""Interactive verification: spawn a leader vehicle at the chosen spawn point,
wait 5 s so the user can look at it in the CARLA spectator, then hand control
to the Traffic Manager with an explicit route derived from
`waypoint.next(step)`. The leader will then drive deterministically along the
road following the OpenDRIVE lane centerline, so the user can visually confirm
the road is followed correctly (no wrong turns at junctions, no drift off the
lane).

The intended route is also drawn in the world with red debug spheres, so
before the vehicle starts moving the user can already see where it is
supposed to go.

Usage on Vortex (with CARLA server running):
    python src/carla_scenario/tools/test_leader_drive.py --town Town04 --spawn 273
    python src/carla_scenario/tools/test_leader_drive.py --town Town07 --spawn 38  --route-m 200
    python src/carla_scenario/tools/test_leader_drive.py --town Town04 --spawn 273 --speed-kmh 30 --wait-s 8
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import carla

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)


def build_route(wp: carla.Waypoint, distance_m: float, step: float = 2.0) -> list[carla.Location]:
    """Walk forward along the OpenDRIVE lane centerline collecting waypoint
    locations. TrafficManager.set_path will drive through them in order."""
    route = []
    cur = wp
    total = 0.0
    while total < distance_m:
        nxts = cur.next(step)
        if not nxts:
            break
        cur = nxts[0]
        route.append(cur.transform.location)
        total += step
    return route


def draw_route(world: carla.World, route: list[carla.Location], life_time: float = 60.0) -> None:
    debug = world.debug
    for loc in route:
        debug.draw_point(
            carla.Location(x=loc.x, y=loc.y, z=loc.z + 0.5),
            size=0.08,
            color=carla.Color(r=255, g=40, b=40),
            life_time=life_time,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--town", required=True)
    p.add_argument("--spawn", type=int, required=True, help="spawn point index")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--speed-kmh", type=float, default=40.0)
    p.add_argument("--route-m", type=float, default=300.0, help="how far to walk forward")
    p.add_argument("--step-m", type=float, default=2.0, help="waypoint sampling step")
    p.add_argument("--wait-s", type=float, default=5.0, help="idle time before releasing TM")
    p.add_argument("--drive-s", type=float, default=30.0, help="seconds of driving after release")
    p.add_argument("--vehicle", default="model3", help="blueprint filter (model3, carlacola, ...)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[INFO] Connecting to CARLA at {args.host}:{args.port}")
    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)

    print(f"[INFO] Loading world '{args.town}' (may take a while for Large Maps)")
    world = client.load_world(args.town)
    world.tick()

    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    if args.spawn >= len(spawn_points):
        raise SystemExit(f"spawn index {args.spawn} out of range (max {len(spawn_points)-1})")
    sp = spawn_points[args.spawn]
    wp = carla_map.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp is None:
        raise SystemExit(f"spawn {args.spawn} does not project to a driving lane")
    print(
        f"[INFO] Spawn {args.spawn}: loc=({sp.location.x:.1f},{sp.location.y:.1f}) "
        f"yaw={sp.rotation.yaw:.1f} road={wp.road_id} lane={wp.lane_id}"
    )

    route = build_route(wp, args.route_m, args.step_m)
    print(f"[INFO] Route: {len(route)} waypoints over ~{args.route_m:.0f} m")
    draw_route(world, route, life_time=args.wait_s + args.drive_s + 10)

    bplib = world.get_blueprint_library()
    bp = bplib.filter(args.vehicle)[0]

    spawn_t = wp.transform
    spawn_t.location.z += 0.5  # lift so the wheels aren't clipped in the road
    leader = world.try_spawn_actor(bp, spawn_t)
    if leader is None:
        raise SystemExit("failed to spawn leader — try a different spawn or check the road is clear")
    print(f"[INFO] Spawned {bp.id} at spawn {args.spawn}")

    print(f"[INFO] Waiting {args.wait_s:.1f} s so you can look at the vehicle in the spectator...")
    time.sleep(args.wait_s)

    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(False)
    tm.ignore_lights_percentage(leader, 100)
    tm.ignore_signs_percentage(leader, 100)
    tm.auto_lane_change(leader, False)
    # NOTE: set_desired_speed uses absolute km/h in 0.9.15+
    tm.set_desired_speed(leader, args.speed_kmh)
    tm.set_path(leader, route)
    leader.set_autopilot(True, tm.get_port())
    print(f"[INFO] TM enabled: speed={args.speed_kmh} km/h, {len(route)}-point path, autopilot ON")

    t_start = time.monotonic()
    try:
        while time.monotonic() - t_start < args.drive_s:
            v = leader.get_velocity()
            speed_kmh = 3.6 * (v.x ** 2 + v.y ** 2 + v.z ** 2) ** 0.5
            loc = leader.get_location()
            print(
                f"\r[INFO] t={time.monotonic()-t_start:5.1f}s  "
                f"speed={speed_kmh:5.1f} km/h  "
                f"loc=({loc.x:7.1f},{loc.y:7.1f})",
                end="",
                flush=True,
            )
            time.sleep(0.5)
        print()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        print("[INFO] Destroying leader.")
        try:
            leader.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    main()
