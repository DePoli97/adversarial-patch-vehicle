"""Capture a CARLA dataset in TransFuser-v6's native 6-camera composite format.

Why a new capture at all
------------------------
The Fase 1 chroma-key dataset is a single 1280x720 FOV-90 front camera. tfv6
consumes a 360-degree surround rig of SIX 384x384 FOV-60 cameras stitched along
the width into one (384, 2304) image. The two are geometrically incompatible:
a patch optimised on the old frames is warped through the wrong intrinsics and
lands on the wrong pixels once the model sees it. Hence this script, which
captures through tfv6's exact calibration.

Calibration is never hard-coded. It is read at runtime from the checkpoint's
own `config.json` through `TrainingConfig(...).camera_calibration[idx]`, so it
cannot drift away from the model. For the shipped `visiononly_resnet34`
checkpoint the resolved rig is:

    idx  pos (x, y, z)          rot (roll, pitch, yaw)   size      fov
    1    ( 0.00, -0.30, 2.25)   (0, 0,  -57.5)           384x384   60
    2    ( 0.25,  0.00, 2.25)   (0, 0,    0.0)           384x384   60   <- FRONT
    3    ( 0.00,  0.30, 2.25)   (0, 0,  +57.5)           384x384   60
    4    (-0.30,  0.30, 2.25)   (0, 0, +122.5)           384x384   60
    5    (-0.55,  0.00, 2.25)   (0, 0,  180.0)           384x384   60
    6    (-0.30, -0.30, 2.25)   (0, 0, -122.5)           384x384   60

Camera 2 is the front view that sees the leader truck: in the composite it is
the width slice [384:768].

Colour convention
-----------------
`lead/common/base_agent.py:287-291` stitches the raw CARLA buffers, which are
BGRA, keeping the first three channels -> BGR. `lead/inference/sensor_agent.py`
then applies `cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)` before the tensor reaches
the network, so the model is fed RGB. This script writes the composite with
`cv2.imwrite`, i.e. BGR on disk, exactly like every other PNG in this repo;
`ChromaKeyDataset` reads it back with `cv2.imread` + `COLOR_BGR2RGB` and lands
on the same RGB the model sees. Do not "fix" the channel order here.

(Note for the loss/EoT side: the closed loop also round-trips the composite
through JPEG quality 90 — `sensor_agent.py:307-314`, `config_closed_loop.py:27`
— to mimic its training data. That compression is NOT applied here; if the
trainer wants to model it, it belongs in EoT, not in the capture.)

Determinism
-----------
The patch quad is not detected here. It is recovered afterwards by differencing
a `clean` capture against a `patched` one (`build_quads.py`), so the two passes
must differ ONLY in the truck texture. Everything that could introduce drift is
therefore pinned:
  - synchronous mode, fixed_delta_seconds = 0.05 (= tfv6's carla_fps 20)
  - physics disabled on both vehicles, poses written with `set_transform`, so
    there is no suspension settling to diverge between the two passes
  - the full pose list (including any jitter) is precomputed from `--seed`
    BEFORE the first spawn, and a pose that fails still consumes its frame id
  - identical weather, identical spawn, zero NPCs

Texture streaming
-----------------
A past 577-run matrix was invalidated because the CarlaCola's 4K texture had
not streamed to its full mip when the shutter fired, so 261/263 frames rendered
a blurred patch. Three mitigations here:
  1. spectator warmup at the spawn area before any actor or sensor exists
     (mandatory on Town11, a 133-tile Large Map, where spawning a camera into
     unstreamed tiles also segfaults the server) — see
     `carla_scenario/scenario_two_vehicles.py:426-435`;
  2. a texture warmup with the leader parked at the CLOSEST distance of the
     sweep, which is the pose that demands the top mip, plus the sweep itself
     always running near -> far so streaming only ever degrades gracefully;
  3. a Laplacian-variance sharpness number for the front camera stored in every
     sidecar, so a blurred run is detectable from the metadata alone instead of
     by eye 500 frames later.

Usage (on Vortex, with CARLA already running):

    conda activate PCLA15
    python src/tfv6_chroma_attack/capture_tfv6.py \\
        --town Town04 --light day --mode clean \\
        --dist-min 5 --dist-max 25 --dist-step 2.5 \\
        --walk-max 20 --walk-step 4

Then swap the CarlaCola .ubulk, restart CARLA, and re-run with
`--mode patched`. Output goes to

    data/chroma_key_dataset/tfv6/<town>_spawn<N>_<light>/<mode>/
        000001.png      (384, 2304, 3) BGR composite
        000001.json     sidecar: pose, distances, light, calibration, sharpness
        capture_manifest.json
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import time
from datetime import datetime
from pathlib import Path

import carla
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "data" / "chroma_key_dataset"

# The spawn points the Fase 1 capture and the closed-loop scenario both use.
# Mirrors carla_scenario/scenario_two_vehicles.py:93 — keep them in sync.
FASE1_SPAWN = {"Town04": 273, "Town07": 38, "Town11": 1713}

DEFAULT_PCLA_DIR = "/home/vortex/PCLA"
DEFAULT_CKPT_REL = "pcla_agents/transfuserv6_pretrained/visiononly_resnet34"

SIM_DELTA = 0.05          # = 1 / tfv6 cfg.carla_fps (20)
CLIENT_TIMEOUT_S = 180.0
SENSOR_TIMEOUT_S = 20.0


# ---------------------------------------------------------------------------
# tfv6 calibration
# ---------------------------------------------------------------------------
def load_camera_calibration(ckpt_dir: Path, pcla_dir: Path,
                            tfv6_dir: Path | None,
                            calib_json: Path | None) -> tuple[dict, int, dict]:
    """Return (calibration, num_cameras, provenance).

    `calibration` is {camera_idx (int, 1-based): {pos, rot, width, height, fov,
    cropped_height}} exactly as tfv6 itself reads it, so the rig can never
    silently diverge from the model.

    Primary source is the model: `TrainingConfig(config.json).camera_calibration`.
    `--calib-json` is an escape hatch for a machine without PCLA importable; it
    must be a file previously dumped by this script (`camera_calibration.json`
    in any run dir), and it is recorded as such in the manifest.
    """
    if calib_json is not None:
        with open(calib_json) as f:
            raw = json.load(f)
        calib = {int(k): v for k, v in raw["camera_calibration"].items()}
        return calib, int(raw["num_cameras"]), {
            "source": "calib_json", "path": str(calib_json),
        }

    tfv6_dir = tfv6_dir or (pcla_dir / "pcla_agents" / "transfuserv6")
    for p in (str(pcla_dir), str(tfv6_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from lead.training.config_training import TrainingConfig  # noqa: E402
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            f"Cannot import tfv6's TrainingConfig ({exc}).\n"
            f"Run inside `conda activate PCLA15` with the tfv6 sources present at\n"
            f"  {pcla_dir}\n  {tfv6_dir}\n"
            f"or pass --calib-json <a camera_calibration.json dumped by a previous run>."
        ) from exc

    cfg_path = ckpt_dir / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"Missing checkpoint config: {cfg_path}")
    with open(cfg_path) as f:
        cfg = TrainingConfig(json.load(f))

    calib = {int(i): dict(cfg.camera_calibration[i])
             for i in range(1, cfg.num_cameras + 1)}
    provenance = {
        "source": "TrainingConfig",
        "checkpoint_config": str(cfg_path),
        "num_cameras": int(cfg.num_cameras),
        "num_used_cameras": int(cfg.num_used_cameras),
        "final_image_width": int(cfg.final_image_width),
        "final_image_height": int(cfg.final_image_height),
        "crop_height": int(cfg.crop_height),
        "horizontal_fov_reduction": int(cfg.horizontal_fov_reduction),
    }
    # Anything but a plain 6-camera full-height concat means the composite this
    # script writes is not what the model eats. Fail loudly rather than produce
    # a dataset that is subtly off.
    if cfg.num_used_cameras != cfg.num_cameras:
        raise SystemExit(
            f"Checkpoint sub-selects cameras ({cfg.num_used_cameras}/{cfg.num_cameras}); "
            f"this capture script assumes all cameras are used."
        )
    if cfg.crop_height != 0 or cfg.horizontal_fov_reduction != 0:
        raise SystemExit(
            f"Checkpoint crops images (crop_height={cfg.crop_height}, "
            f"horizontal_fov_reduction={cfg.horizontal_fov_reduction}); "
            f"this capture script assumes no cropping."
        )
    return calib, int(cfg.num_cameras), provenance


def composite_shape(calib: dict, num_cameras: int) -> tuple[int, int]:
    """(height, width) of the stitched composite."""
    h = int(calib[1]["height"])
    w = sum(int(calib[i]["width"]) for i in range(1, num_cameras + 1))
    return h, w


def camera_slices(calib: dict, num_cameras: int) -> dict[int, list[int]]:
    """{camera_idx: [x_start, x_end]} of each camera inside the composite."""
    out = {}
    x = 0
    for i in range(1, num_cameras + 1):
        w = int(calib[i]["width"])
        out[i] = [x, x + w]
        x += w
    return out


def front_camera_index(calib: dict, num_cameras: int) -> int:
    """The camera whose yaw is closest to 0 — the one that sees the leader."""
    return min(range(1, num_cameras + 1),
               key=lambda i: abs(float(calib[i]["rot"][2])))


# ---------------------------------------------------------------------------
# CARLA helpers
# ---------------------------------------------------------------------------
def walk_forward(wp: carla.Waypoint, distance_m: float,
                 step: float = 1.0) -> carla.Waypoint | None:
    """Walk `distance_m` along the lane in <= `step` increments.

    Unlike capture_fase1.walk_forward this lands on the requested distance
    exactly (the last increment is the remainder) instead of overshooting to the
    next multiple of `step` — the leader gap sweep needs the real value.
    Branches at junctions take `next()[0]`, same as capture_fase1.py and
    scenario_two_vehicles.py, so all three place actors identically.
    """
    cur = wp
    remaining = float(distance_m)
    while remaining > 1e-3:
        d = min(step, remaining)
        nxts = cur.next(d)
        if not nxts:
            return None
        cur = nxts[0]
        remaining -= d
    return cur


def place(wp: carla.Waypoint, dz: float = 0.10, lateral_m: float = 0.0,
          yaw_deg: float = 0.0) -> carla.Transform:
    """A BRAND NEW spawn transform on `wp`, raised by `dz`, optionally shifted
    sideways and rotated.

    Always constructs a fresh `carla.Transform` instead of mutating
    `wp.transform` in place: the same waypoint is visited once per leader gap,
    and an accidentally shared object would accumulate the z offset frame after
    frame, silently lifting the ego off the road halfway through the sweep.
    """
    tf = wp.transform
    out = carla.Transform(
        carla.Location(tf.location.x, tf.location.y, tf.location.z + dz),
        carla.Rotation(pitch=tf.rotation.pitch, yaw=tf.rotation.yaw + yaw_deg,
                       roll=tf.rotation.roll),
    )
    if abs(lateral_m) > 1e-6:
        right = tf.get_right_vector()
        out.location.x += right.x * lateral_m
        out.location.y += right.y * lateral_m
        out.location.z += right.z * lateral_m
    return out


def make_weather(light: str, azimuth: float) -> tuple[carla.WeatherParameters, float]:
    """ClearNoon with only the sun angles overridden — identical to the
    closed-loop scenario (scenario_two_vehicles.py:351-368) and to
    capture_fase1.py, so train and deploy see the same lighting."""
    alt = 45.0 if light == "day" else -30.0
    w = carla.WeatherParameters.ClearNoon
    return carla.WeatherParameters(
        cloudiness=w.cloudiness,
        precipitation=w.precipitation,
        precipitation_deposits=w.precipitation_deposits,
        wind_intensity=w.wind_intensity,
        sun_azimuth_angle=azimuth,
        sun_altitude_angle=alt,
        fog_density=w.fog_density,
        fog_distance=w.fog_distance,
        wetness=w.wetness,
    ), alt


def cleanup_world(world: carla.World) -> None:
    for actor in world.get_actors():
        try:
            if actor.type_id.startswith(("vehicle.", "sensor.")):
                actor.destroy()
        except Exception:
            pass


def to_bgr(image: carla.Image) -> np.ndarray:
    """CARLA raw BGRA buffer -> contiguous (H, W, 3) BGR uint8."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    return np.ascontiguousarray(arr[:, :, :3])


def drain(queues: dict) -> None:
    for q in queues.values():
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break


def tick_n(world: carla.World, queues: dict, n: int) -> None:
    """Tick `n` times, discarding sensor output (keeps memory bounded: six
    384x384 BGRA frames per tick is ~3.5 MB, which adds up fast over a 150-tick
    warmup)."""
    for _ in range(n):
        world.tick()
        drain(queues)


def grab_synced(world: carla.World, queues: dict) -> tuple[int, dict]:
    """Tick once and return (frame, {camera_idx: carla.Image}) for that frame."""
    frame = world.tick()
    images = {}
    for idx, q in queues.items():
        while True:
            try:
                img = q.get(timeout=SENSOR_TIMEOUT_S)
            except queue.Empty as exc:
                raise RuntimeError(
                    f"camera {idx} produced no image for frame {frame} "
                    f"within {SENSOR_TIMEOUT_S}s"
                ) from exc
            if img.frame >= frame:
                images[idx] = img
                break
    return frame, images


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian — the standard blur proxy. Used as the
    tripwire for the 4K-texture-not-streamed bug."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--town", default="Town04", choices=sorted(FASE1_SPAWN),
                   help="map; the Fase 1 spawn index is looked up from it")
    p.add_argument("--spawn", type=int, default=None,
                   help="override the Fase 1 spawn index (exploratory runs only)")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)

    p.add_argument("--mode", choices=["clean", "patched"], required=True,
                   help="LABEL ONLY. Which texture the CarlaCola actually wears "
                        "depends on the .ubulk currently deployed in the running "
                        "CARLA package — swap it and restart CARLA between modes.")
    p.add_argument("--light", choices=["day", "night"], default="day")
    p.add_argument("--sun-azimuth", type=float, default=90.0)

    p.add_argument("--dist-min", type=float, default=5.0,
                   help="closest leader gap, metres along the lane")
    p.add_argument("--dist-max", type=float, default=25.0)
    p.add_argument("--dist-step", type=float, default=2.5)
    p.add_argument("--walk-max", type=float, default=20.0,
                   help="how far the ego walks along the road (pose diversity)")
    p.add_argument("--walk-step", type=float, default=4.0)

    p.add_argument("--lateral-jitter-m", type=float, default=0.0,
                   help="uniform +/- lateral offset of the ego, seeded so clean "
                        "and patched draw identical values")
    p.add_argument("--yaw-jitter-deg", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--warmup-ticks", type=int, default=200,
                   help="spectator warmup after town load, BEFORE any actor or "
                        "sensor exists (mandatory on Town11)")
    p.add_argument("--texture-warmup-ticks", type=int, default=150,
                   help="ticks with the leader parked at --dist-min, so the 4K "
                        "truck texture is asked for its top mip")
    p.add_argument("--settle-ticks", type=int, default=12,
                   help="ticks between moving the actors and the shutter")
    p.add_argument("--first-frame-extra-ticks", type=int, default=60)

    p.add_argument("--leader-bp", default="vehicle.carlamotors.carlacola")
    p.add_argument("--follower-bp", default="vehicle.tesla.model3")

    p.add_argument("--pcla-dir", type=Path, default=Path(DEFAULT_PCLA_DIR))
    p.add_argument("--tfv6-dir", type=Path, default=None,
                   help="defaults to <pcla-dir>/pcla_agents/transfuserv6")
    p.add_argument("--ckpt-dir", type=Path, default=None,
                   help=f"defaults to <pcla-dir>/{DEFAULT_CKPT_REL}")
    p.add_argument("--calib-json", type=Path, default=None,
                   help="fallback calibration dumped by a previous run")

    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--run-tag", default=None,
                   help="folder tag; defaults to <town>_spawn<N>_<light>")
    p.add_argument("--save-per-camera", action="store_true",
                   help="also dump the six 384x384 views separately (debug)")
    return p.parse_args()


def build_pose_list(args: argparse.Namespace) -> list[dict]:
    """Precompute every pose BEFORE touching CARLA.

    Ordering is ego walk offset (outer) x leader gap ASCENDING (inner): the
    truck always approaches from near to far, so mip streaming only ever goes
    high-detail -> low-detail, never the risky other way round.

    Jitter is drawn here, in one pass, from `--seed`. Drawing it inside the
    capture loop would let a skipped pose shift every subsequent draw and break
    the clean/patched correspondence.
    """
    n_walk = int(round(args.walk_max / args.walk_step)) + 1 if args.walk_step > 0 else 1
    walks = [i * args.walk_step for i in range(n_walk)]

    n_dist = int(round((args.dist_max - args.dist_min) / args.dist_step)) + 1 \
        if args.dist_step > 0 else 1
    gaps = [args.dist_min + i * args.dist_step for i in range(n_dist)]
    gaps = [g for g in gaps if g <= args.dist_max + 1e-6]

    rng = np.random.default_rng(args.seed)
    poses = []
    for walk_m in walks:
        for gap_m in gaps:
            lat = float(rng.uniform(-args.lateral_jitter_m, args.lateral_jitter_m)) \
                if args.lateral_jitter_m > 0 else 0.0
            yaw = float(rng.uniform(-args.yaw_jitter_deg, args.yaw_jitter_deg)) \
                if args.yaw_jitter_deg > 0 else 0.0
            poses.append({
                "frame_id": f"{len(poses) + 1:06d}",
                "walk_m": float(walk_m),
                "gap_m": float(gap_m),
                "lateral_m": lat,
                "yaw_deg": yaw,
            })
    return poses


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    ckpt_dir = args.ckpt_dir or (args.pcla_dir / DEFAULT_CKPT_REL)
    calib, num_cameras, calib_provenance = load_camera_calibration(
        ckpt_dir, args.pcla_dir, args.tfv6_dir, args.calib_json)
    comp_h, comp_w = composite_shape(calib, num_cameras)
    slices = camera_slices(calib, num_cameras)
    front_idx = front_camera_index(calib, num_cameras)

    spawn_idx = args.spawn if args.spawn is not None else FASE1_SPAWN[args.town]
    if args.spawn is not None:
        print(f"[WARN] Spawn overridden to {args.spawn} (Fase 1 spawn for "
              f"{args.town} is {FASE1_SPAWN[args.town]}) — exploratory run.")

    parent_tag = args.run_tag or f"{args.town}_spawn{spawn_idx}_{args.light}"
    run_dir = args.out_dir / "tfv6" / parent_tag / args.mode
    run_dir.mkdir(parents=True, exist_ok=True)

    poses = build_pose_list(args)

    print(f"{'=' * 68}")
    print(f"  tfv6 composite capture")
    print(f"  town/spawn : {args.town} / {spawn_idx}")
    print(f"  mode       : {args.mode}   light: {args.light}")
    print(f"  rig        : {num_cameras} cameras -> composite {comp_h}x{comp_w}")
    print(f"  front cam  : index {front_idx}, composite slice {slices[front_idx]}")
    print(f"  calib from : {calib_provenance['source']}")
    print(f"  poses      : {len(poses)}")
    print(f"  output     : {run_dir}")
    print(f"{'=' * 68}\n")

    # Dump the resolved rig next to the frames: it is both the sidecar's source
    # of truth and the --calib-json fallback for a machine without PCLA.
    with open(run_dir / "camera_calibration.json", "w") as f:
        json.dump({"num_cameras": num_cameras,
                   "camera_calibration": {str(k): v for k, v in calib.items()},
                   "composite_shape": [comp_h, comp_w, 3],
                   "camera_slices": {str(k): v for k, v in slices.items()},
                   "front_camera_index": front_idx,
                   "provenance": calib_provenance}, f, indent=2)

    print(f"[INFO] Connecting to CARLA at {args.host}:{args.port}")
    client = carla.Client(args.host, args.port)
    client.set_timeout(CLIENT_TIMEOUT_S)
    print(f"[INFO] Loading world '{args.town}'")
    world = client.load_world(args.town)

    settings = world.get_settings()
    original_settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = SIM_DELTA
    # Large Maps (Town11/12) unload tiles — and their textures — around the
    # streaming origin. Widen it so the whole sweep stays resident.
    if hasattr(settings, "tile_stream_distance"):
        settings.tile_stream_distance = 2000.0
    if hasattr(settings, "actor_active_distance"):
        settings.actor_active_distance = 2000.0
    world.apply_settings(settings)

    weather, sun_alt = make_weather(args.light, args.sun_azimuth)
    world.set_weather(weather)
    print(f"[INFO] Weather ClearNoon, sun_altitude={sun_alt} azimuth={args.sun_azimuth}")

    cleanup_world(world)
    world.tick()

    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    if spawn_idx >= len(spawn_points):
        raise SystemExit(f"spawn {spawn_idx} out of range for {args.town} "
                         f"({len(spawn_points)} spawn points)")
    base_sp = spawn_points[spawn_idx]
    base_wp = carla_map.get_waypoint(base_sp.location, project_to_road=True,
                                     lane_type=carla.LaneType.Driving)
    if base_wp is None:
        raise SystemExit(f"spawn {spawn_idx} does not project to a driving lane")
    print(f"[INFO] Spawn {spawn_idx}: loc=({base_sp.location.x:.1f},"
          f"{base_sp.location.y:.1f}) road={base_wp.road_id} lane={base_wp.lane_id}")

    # Tile/texture warmup with the spectator parked over the spawn, BEFORE any
    # actor or sensor exists. On Town11 spawning a camera into unstreamed tiles
    # segfaults the server; on small maps this costs a couple of seconds.
    world.get_spectator().set_transform(carla.Transform(
        carla.Location(base_sp.location.x, base_sp.location.y,
                       base_sp.location.z + 30.0),
        carla.Rotation(pitch=-40.0)))
    print(f"[INFO] Spectator warmup {args.warmup_ticks} ticks ...")
    for _ in range(args.warmup_ticks):
        world.tick()

    bplib = world.get_blueprint_library()
    follower_bp = bplib.filter(args.follower_bp)[0]
    leader_bp = bplib.filter(args.leader_bp)[0]

    ego = None
    leader = None
    cameras: dict[int, carla.Actor] = {}
    queues: dict[int, queue.Queue] = {}
    t0 = time.time()
    ok_count = 0
    skipped: list[dict] = []

    try:
        ego = world.try_spawn_actor(follower_bp, place(base_wp))
        if ego is None:
            raise SystemExit(f"failed to spawn ego at spawn {spawn_idx}")
        leader_wp0 = walk_forward(base_wp, args.dist_min)
        if leader_wp0 is None:
            raise SystemExit(f"cannot place leader {args.dist_min} m ahead of spawn")
        leader = world.try_spawn_actor(leader_bp, place(leader_wp0))
        if leader is None:
            raise SystemExit("failed to spawn leader")
        world.tick()

        # Physics off: the poses we write are the poses we get, bit-identical
        # between the clean and the patched pass. With physics on, suspension
        # settling would put the same nominal pose a few millimetres apart in
        # the two runs and the diff would smear along every silhouette edge.
        ego.set_simulate_physics(False)
        leader.set_simulate_physics(False)
        world.tick()
        print("[INFO] Ego + leader spawned, physics disabled")

        for idx in range(1, num_cameras + 1):
            c = calib[idx]
            bp = bplib.find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", str(int(c["width"])))
            bp.set_attribute("image_size_y", str(int(c["height"])))
            bp.set_attribute("fov", str(float(c["fov"])))
            tf = carla.Transform(
                carla.Location(x=float(c["pos"][0]), y=float(c["pos"][1]),
                               z=float(c["pos"][2])),
                carla.Rotation(roll=float(c["rot"][0]), pitch=float(c["rot"][1]),
                               yaw=float(c["rot"][2])),
            )
            cam = world.spawn_actor(bp, tf, attach_to=ego)
            q: queue.Queue = queue.Queue()
            cam.listen(q.put)
            cameras[idx] = cam
            queues[idx] = q
        print(f"[INFO] {len(cameras)} cameras attached with tfv6 calibration")

        # Texture warmup: the leader is already at the closest gap of the sweep,
        # which is the pose that requests the highest mip of its 4K texture.
        print(f"[INFO] Texture warmup {args.texture_warmup_ticks} ticks at "
              f"{args.dist_min:.1f} m ...")
        tick_n(world, queues, args.texture_warmup_ticks)

        for pose in poses:
            fid = pose["frame_id"]
            ego_wp = walk_forward(base_wp, pose["walk_m"])
            if ego_wp is None:
                skipped.append({**pose, "reason": "ego walk failed"})
                print(f"[{fid}] SKIP — cannot walk {pose['walk_m']:.1f} m from spawn")
                continue
            leader_wp = walk_forward(ego_wp, pose["gap_m"])
            if leader_wp is None:
                skipped.append({**pose, "reason": "leader walk failed"})
                print(f"[{fid}] SKIP — cannot place leader {pose['gap_m']:.1f} m ahead")
                continue

            ego.set_transform(place(ego_wp, lateral_m=pose["lateral_m"],
                                    yaw_deg=pose["yaw_deg"]))
            leader.set_transform(place(leader_wp))

            extra = args.first_frame_extra_ticks if ok_count == 0 else 0
            tick_n(world, queues, args.settle_ticks + extra)
            frame, images = grab_synced(world, queues)

            views = [to_bgr(images[i]) for i in range(1, num_cameras + 1)]
            composite = np.concatenate(views, axis=1)  # BGR, (384, 2304, 3)
            if composite.shape[:2] != (comp_h, comp_w):
                raise RuntimeError(
                    f"composite is {composite.shape[:2]}, expected {(comp_h, comp_w)}"
                )

            cv2.imwrite(str(run_dir / f"{fid}.png"), composite)
            if args.save_per_camera:
                cam_dir = run_dir / "per_camera" / fid
                cam_dir.mkdir(parents=True, exist_ok=True)
                for i, v in enumerate(views, start=1):
                    cv2.imwrite(str(cam_dir / f"cam{i}.png"), v)

            x0, x1 = slices[front_idx]
            front_gray = cv2.cvtColor(composite[:, x0:x1], cv2.COLOR_BGR2GRAY)
            ego_loc = ego.get_transform().location
            lead_loc = leader.get_transform().location
            euclid = float(np.linalg.norm([lead_loc.x - ego_loc.x,
                                           lead_loc.y - ego_loc.y,
                                           lead_loc.z - ego_loc.z]))

            meta = {
                "frame_id": fid,
                "carla_frame": int(frame),
                "town": args.town,
                "spawn_idx": spawn_idx,
                "mode": args.mode,
                "light": args.light,
                "weather": "ClearNoon",
                "sun_altitude": float(sun_alt),
                "sun_azimuth": float(args.sun_azimuth),
                "walk_offset_m": pose["walk_m"],
                "leader_gap_m": pose["gap_m"],
                "leader_distance_m": euclid,
                "lateral_jitter_m": pose["lateral_m"],
                "yaw_jitter_deg": pose["yaw_deg"],
                "ego_bp": args.follower_bp,
                "leader_bp": args.leader_bp,
                "ego_loc": [ego_loc.x, ego_loc.y, ego_loc.z],
                "ego_yaw_deg": float(ego.get_transform().rotation.yaw),
                "ego_road_id": int(ego_wp.road_id),
                "ego_lane_id": int(ego_wp.lane_id),
                "leader_loc": [lead_loc.x, lead_loc.y, lead_loc.z],
                "leader_yaw_deg": float(leader.get_transform().rotation.yaw),
                "leader_road_id": int(leader_wp.road_id),
                "leader_lane_id": int(leader_wp.lane_id),
                "composite_shape": [comp_h, comp_w, 3],
                "channel_order": "BGR",
                "num_cameras": num_cameras,
                "front_camera_index": front_idx,
                "camera_slices": {str(k): v for k, v in slices.items()},
                "camera_calibration": {str(k): v for k, v in calib.items()},
                "front_sharpness": sharpness(front_gray),
                "seed": args.seed,
            }
            with open(run_dir / f"{fid}.json", "w") as f:
                json.dump(meta, f, indent=2)

            ok_count += 1
            print(f"[{fid}] walk={pose['walk_m']:5.1f}m gap={pose['gap_m']:5.1f}m "
                  f"euclid={euclid:5.1f}m sharp={meta['front_sharpness']:7.1f}",
                  flush=True)

    finally:
        for cam in cameras.values():
            try:
                cam.stop()
                cam.destroy()
            except Exception:
                pass
        for actor in (leader, ego):
            try:
                if actor is not None:
                    actor.destroy()
            except Exception:
                pass
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass

    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "town": args.town,
        "spawn_idx": spawn_idx,
        "mode": args.mode,
        "light": args.light,
        "sun_altitude": float(sun_alt),
        "sun_azimuth": float(args.sun_azimuth),
        "seed": args.seed,
        "num_poses": len(poses),
        "num_frames": ok_count,
        "skipped": skipped,
        "composite_shape": [comp_h, comp_w, 3],
        "channel_order": "BGR",
        "num_cameras": num_cameras,
        "front_camera_index": front_idx,
        "camera_slices": {str(k): v for k, v in slices.items()},
        "camera_calibration": {str(k): v for k, v in calib.items()},
        "calibration_provenance": calib_provenance,
        "sim_delta": SIM_DELTA,
        "physics_disabled": True,
        "closed_loop_jpeg_quality_note": (
            "the tfv6 closed loop re-encodes the composite at JPEG quality 90 "
            "before inference (sensor_agent.py:307-314); not applied here — "
            "model it in EoT if wanted"
        ),
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
    }
    with open(run_dir / "capture_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    dt = time.time() - t0
    print(f"\n[INFO] Done. {ok_count}/{len(poses)} frames in {dt:.0f}s -> {run_dir}")
    if skipped:
        print(f"[WARN] {len(skipped)} poses skipped (recorded in capture_manifest.json)")
    print("[INFO] Next: capture the other --mode, then run build_quads.py on the pair.")


if __name__ == "__main__":
    main()
