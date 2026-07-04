"""Interactive helper to locate a one-way country road in CARLA.

Loads the requested town via the Python API (mandatory — Masoud's rule: never
trust the town the server is already on, because spawn point indices are
map-scoped and get out of sync otherwise), then draws every spawn point's
index directly in the 3D world. Optionally filters to spawn points that sit
on a single-direction road (no opposite-direction driving lane sibling).

Usage on Vortex (with CARLA server running on port 2000):
    python src/carla_scenario/tools/find_one_way_road.py --town Town07
    python src/carla_scenario/tools/find_one_way_road.py --town Town07 --one-way-only
    python src/carla_scenario/tools/find_one_way_road.py --town Town04 --one-way-only --duration 600

While the script runs, fly around with the CARLA spectator (the editor window,
or a spectator client). Every spawn point shows its index as a floating red
number, plus a small green arrow pointing along the road forward direction.
When you find the road you want, note its index and Ctrl-C to exit.

The recommended workflow after that:
  1. Cache the chosen index with `scan_spawn.py` (or edit spawn_cache.json)
  2. Verify by running scenario_two_vehicles.py --town <T> to make sure the
     leader spawns where you expected
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


def is_one_way(wp: carla.Waypoint) -> bool:
    """A road is one-way if walking left across driving lanes never crosses
    into a lane with opposite sign of lane_id (CARLA's convention: lanes on
    opposite sides of a two-way road have opposite lane_id signs)."""
    current = wp
    while True:
        left = current.get_left_lane()
        if left is None or left.lane_type != carla.LaneType.Driving:
            break
        if left.lane_id * current.lane_id < 0:
            return False
        current = left
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--town", required=True, help="e.g. Town07, Town04")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument(
        "--one-way-only",
        action="store_true",
        help="draw only spawn points that sit on a one-way road",
    )
    p.add_argument(
        "--skip-junction",
        action="store_true",
        default=True,
        help="skip spawn points inside intersections (default: on)",
    )
    p.add_argument(
        "--no-skip-junction",
        dest="skip_junction",
        action="store_false",
        help="also draw spawn points in junctions",
    )
    p.add_argument(
        "--duration",
        type=int,
        default=300,
        help="seconds to keep the labels visible (default 300 = 5 min)",
    )
    p.add_argument(
        "--refresh",
        type=float,
        default=10.0,
        help="how often (seconds) to redraw labels — CARLA's debug strings decay after life_time",
    )
    p.add_argument(
        "--z-offset",
        type=float,
        default=1.5,
        help="vertical lift (m) of the label above the road, so it isn't clipped by the asphalt",
    )
    p.add_argument(
        "--max-labels",
        type=int,
        default=0,
        help="hard cap on labels drawn (default 0 = no cap). Use on huge maps (Town11/12) to avoid overloading the renderer",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[INFO] Connecting to CARLA at {args.host}:{args.port}")
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    print(f"[INFO] Loading world '{args.town}' via API (mandatory to sync spawn indices)")
    world = client.load_world(args.town)
    world.tick()

    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    print(f"[INFO] {args.town} has {len(spawn_points)} spawn points")

    # Filter set up front so the console output already tells you the count.
    keep: list[tuple[int, carla.Transform, carla.Waypoint]] = []
    for idx, sp in enumerate(spawn_points):
        wp = carla_map.get_waypoint(
            sp.location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if wp is None:
            continue
        if args.skip_junction and wp.is_junction:
            continue
        if args.one_way_only and not is_one_way(wp):
            continue
        keep.append((idx, sp, wp))

    if args.max_labels > 0 and len(keep) > args.max_labels:
        step = max(1, len(keep) // args.max_labels)
        keep = keep[::step][: args.max_labels]
        print(f"[INFO] Capped labels to {len(keep)} (every {step}th spawn)")

    print(
        f"[INFO] Drawing {len(keep)} spawn labels "
        f"(one_way_only={args.one_way_only}, skip_junction={args.skip_junction})"
    )
    if not keep:
        print("[WARN] No spawn points matched the filters. Try --no-skip-junction or drop --one-way-only.")
        return

    debug = world.debug
    life_time = args.refresh * 1.2  # slight overlap so labels never blink

    print(f"[INFO] Fly around in the CARLA spectator to inspect labels.")
    print(f"[INFO] Refreshing every {args.refresh:.1f}s for up to {args.duration}s. Ctrl-C to stop.")

    start = time.monotonic()
    try:
        while time.monotonic() - start < args.duration:
            for idx, sp, wp in keep:
                label_loc = carla.Location(
                    x=sp.location.x,
                    y=sp.location.y,
                    z=sp.location.z + args.z_offset,
                )
                debug.draw_string(
                    label_loc,
                    str(idx),
                    draw_shadow=True,
                    color=carla.Color(r=255, g=40, b=40),
                    life_time=life_time,
                    persistent_lines=False,
                )
                # Small forward arrow so you can tell which way the road flows.
                fwd = wp.transform.get_forward_vector()
                head = carla.Location(
                    x=sp.location.x + fwd.x * 3.0,
                    y=sp.location.y + fwd.y * 3.0,
                    z=sp.location.z + 0.5,
                )
                debug.draw_arrow(
                    carla.Location(x=sp.location.x, y=sp.location.y, z=sp.location.z + 0.5),
                    head,
                    thickness=0.08,
                    arrow_size=0.25,
                    color=carla.Color(r=60, g=220, b=60),
                    life_time=life_time,
                )
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted. Note the spawn index of the road you picked.")


if __name__ == "__main__":
    main()
