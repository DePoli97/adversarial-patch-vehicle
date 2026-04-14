# Phase A — Two-vehicle CARLA scenario

Script: `scenario_two_vehicles.py`

## What it does

Spawns two Tesla Model 3 in Town02:
- **Leader**: CARLA autopilot (Traffic Manager), drives ahead
- **Follower**: PCLA agent (default `tfv4_l6_0`) follows with a pre-computed route

Collects per-tick telemetry (CSV) and camera images (JPG) from the follower's
front camera. Run once per condition: `none`, `raw`, `camouflaged`.

---

## Prerequisites on Vortex

```
Terminal 1 (CARLA server):
  conda activate carla
  cd /home/vortex/carla
  make launch         # or: ./CarlaUE4.sh -RenderOffScreen

Terminal 2 (this script):
  conda activate PCLA310
  cd /home/vortex/adversarial-patch-vehicle
```

---

## Changing the license plate texture

CARLA loads textures at startup. To switch condition you must:

1. **Identify the CARLA plate texture path on Vortex**:
   ```bash
   find /home/vortex/carla -name "T_LicensePlate_d.TGA" 2>/dev/null
   # typically: /home/vortex/carla/Unreal/CarlaUE4/Content/Static/Vehicles/2Wheeled/...
   # or inside the packaged build Content/ folder
   ```

2. **Backup the original**:
   ```bash
   cp /path/to/T_LicensePlate_d.TGA /path/to/T_LicensePlate_d.TGA.bak
   ```

3. **Copy the patch texture**:
   ```bash
   # Baseline (no patch) — restore original:
   cp /path/to/T_LicensePlate_d.TGA.bak /path/to/T_LicensePlate_d.TGA

   # Raw adversarial patch:
   cp ~/assets/T_LicensePlate_d_raw.TGA /path/to/T_LicensePlate_d.TGA

   # Camouflaged patch:
   cp ~/assets/T_LicensePlate_d_camo.TGA /path/to/T_LicensePlate_d.TGA
   ```

4. **Restart the CARLA server**, then run the script.

---

## Running the three conditions

```bash
# Condition 1: no patch (baseline)
python src/carla_scenario/scenario_two_vehicles.py --condition none

# Condition 2: raw adversarial patch
python src/carla_scenario/scenario_two_vehicles.py --condition raw

# Condition 3: camouflaged patch
python src/carla_scenario/scenario_two_vehicles.py --condition camouflaged
```

### Optional arguments

| Flag | Default | Description |
|---|---|---|
| `--agent` | `tfv4_l6_0` | PCLA agent for follower |
| `--town` | `Town02` | CARLA map |
| `--num_ticks` | `600` | Simulation steps (~30s) |
| `--save_interval` | `10` | Save image every N ticks |
| `--leader_speed` | `30` | Leader target speed (km/h) |
| `--host` | `localhost` | CARLA server host |
| `--port` | `2000` | CARLA server port |

---

## Output

```
experiments/carla_scenarios/<condition>_<timestamp>/
  telemetry.csv        — per-tick: distance, TTC, speed, steering, collisions
  images/              — front-camera JPGs every --save_interval ticks
  follower_route.xml   — generated PCLA route
  summary.json         — aggregate stats: collisions, mean/min dist, images saved
```

Key telemetry columns:
- `distance_m` — 3D distance leader→follower
- `ttc_s` — time-to-collision (seconds), -1 if not closing
- `collision_detected` — 1 if collision sensor fired that tick
- `follower_throttle/steer/brake` — PCLA control output

---

## Suggested agents to compare (Phase B)

| Agent | Modality | Why interesting |
|---|---|---|
| `tfv4_l6_0` | Camera + LiDAR | Top Leaderboard 1 performer |
| `tfv6_visiononly` | Vision only | No LiDAR — most vulnerable to 2D patch |
| `tfv4_aim_0` | Vision only | AIM method, simple architecture |
| `carl_carlv11` | Camera + LiDAR | Best RL planner |
| `simlingo_simlingo` | Vision + LLM | Leaderboard 2 winner, VLM-based |
| `lmdrive_llava` | Vision + LLM | LLaVA-based, explicit language reasoning |
