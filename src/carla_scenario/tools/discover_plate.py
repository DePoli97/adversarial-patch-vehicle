"""Diagnostic: find the license-plate material/object of a CARLA vehicle.

Run on Vortex with a CARLA server listening on :2000, e.g.

    python src/carla_scenario/tools/discover_plate.py --town Town06

Writes a report to experiments/carla_scenarios/plate_discovery.txt so we can
decide which texture-swap strategy is feasible (runtime API vs attached quad
vs UE editor bake).
"""

import argparse
import datetime
import os

import carla


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
OUT_PATH = os.path.join(
    REPO_ROOT, "experiments", "carla_scenarios", "plate_discovery.txt"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--town", default="Town06")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument(
        "--vehicle", default="vehicle.tesla.model3",
        help="Blueprint id to inspect (same one the scenario uses)."
    )
    args = p.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    client.load_world(args.town)
    world = client.get_world()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    lines: list[str] = []

    def log(s: str = ""):
        print(s)
        lines.append(s)

    log(f"plate_discovery — {datetime.datetime.now().isoformat(timespec='seconds')}")
    log(f"town={args.town}  vehicle={args.vehicle}")
    log("=" * 70)

    # (1) World objects containing plate/license in the name. If anything shows
    # up here we can try world.apply_color_texture_to_object(...) directly.
    log("\n[1] World objects with 'plate' or 'license' in the name")
    log("-" * 70)
    try:
        names = world.get_names_of_all_objects()
    except Exception as e:
        log(f"  get_names_of_all_objects() not supported: {e}")
        names = []
    matches = sorted(
        n for n in names if "plate" in n.lower() or "licens" in n.lower()
    )
    if matches:
        for n in matches:
            log(f"  {n}")
    else:
        log("  (none found — plate is likely baked into the vehicle mesh)")

    # (2) Vehicle blueprint attributes — some builds expose license_plate
    # colour/id/etc. If we see one, swapping may be possible via the BP.
    log("\n[2] Blueprint attributes for the ego vehicle")
    log("-" * 70)
    bplib = world.get_blueprint_library()
    bp = bplib.find(args.vehicle)
    for attr in bp:
        try:
            recommended = list(attr.recommended_values)
        except Exception:
            recommended = []
        log(f"  {attr.id:30s} type={attr.type}  modifiable={attr.is_modifiable}"
            f"  recommended={recommended[:3]}")

    # (3) Spawn the vehicle and enumerate its Unreal components. The plate
    # usually appears as a material slot on one of these.
    log("\n[3] Attached components of a spawned instance")
    log("-" * 70)
    spawn_points = world.get_map().get_spawn_points()
    vehicle = world.try_spawn_actor(bp, spawn_points[0])
    if vehicle is None:
        log("  could not spawn vehicle for inspection (try another spawn idx)")
    else:
        log(f"  actor_id={vehicle.id}  type_id={vehicle.type_id}")
        # attributes on the spawned actor (may differ from BP in 0.9.15)
        for k, v in vehicle.attributes.items():
            log(f"    attr {k} = {v}")
        # light state (bones/lights are enumerable)
        try:
            log(f"  vehicle.get_physics_control().wheels -> "
                f"{len(vehicle.get_physics_control().wheels)} wheels")
        except Exception as e:
            log(f"  physics inspect failed: {e}")
        vehicle.destroy()

    # (4) Probe: can we call apply_color_texture_to_object at all, and on
    # which material parameters?
    log("\n[4] world.apply_color_texture_to_object API probe")
    log("-" * 70)
    api_ok = hasattr(world, "apply_color_texture_to_object")
    log(f"  apply_color_texture_to_object available: {api_ok}")
    if api_ok:
        try:
            params = list(carla.MaterialParameter.values)
            log(f"  MaterialParameter enum values: {[str(v) for v in params]}")
        except Exception:
            log("  (MaterialParameter enum not introspectable — try Diffuse/Normal/Emissive)")

    log("\n=" * 1)
    log(f"Report saved to: {OUT_PATH}")

    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
