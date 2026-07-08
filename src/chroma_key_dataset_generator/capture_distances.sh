#!/bin/bash
# Multi-distance Fase 1 capture: add 6 m and 20 m leader gaps (10 m already
# captured and moved under dist10m/). Distance variety is REAL pose variety
# (truck fills the frame at 6 m, is small at 20 m) — the proper cure for the
# low-pose-variety overfitting that made single-distance patches noisy, and
# physically relevant for the closed-loop collision test where the follower
# closes the gap.
#
# CarlaCola half-length ~2.6 m + Model3 half-length ~2.4 m => bumpers touch at
# ~5 m centroid-to-centroid; 6 m leaves ~1 m clearance (near-contact, realistic).
#
# Layout: fase1/<town>_spawn<N>/<light>/dist<M>m/<mode>/  (matches moved 10 m).
# Package strategy (same as overnight_capture.sh):
#   pass 1 carla_lite            -> clean + noleader
#   pass 2 CARLA_Shipping        -> marker (yellow marker baked into that build)

set -u
REPO=/home/vortex/adversarial-patch-vehicle
cd "$REPO" || exit 1

LOGDIR="$REPO/experiments/fase1_dataset_gen"
mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOGDIR/distances_$STAMP.log"
echo "log -> $LOG"

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh

TOWNS=(Town04:273 Town07:38 Town11:1713)
LIGHTS=(day night)
DISTS=(6 20)

log() { echo "[$(date +'%F %T')] $*" | tee -a "$LOG"; }

kill_carla() { pkill -9 -f CarlaUE4-Linux 2>/dev/null || true; sleep 3; }

start_carla() {
    # $1 = lite | shipping
    kill_carla
    local dir
    if [ "$1" = "lite" ]; then
        dir=/home/vortex/carla_lite
    else
        dir=/home/vortex/carla/Dist/CARLA_Shipping_0.9.15.2/LinuxNoEditor
    fi
    log "starting CARLA ($1) from $dir"
    (cd "$dir" && DISPLAY=:1 ./CarlaUE4.sh -RenderOffScreen >>"$LOG" 2>&1 &)
    # CARLA lite is slow to cold-start: wait up to 300 s for port 2000.
    for _ in $(seq 1 60); do
        sleep 5
        if ss -tln 2>/dev/null | grep -q ':2000'; then
            log "CARLA up ($1)"
            sleep 8   # extra grace for full warmup
            return 0
        fi
    done
    log "ERROR: CARLA ($1) did not open port 2000 within 300 s"
    return 1
}

run_cap() {
    # $1 town  $2 spawn  $3 light  $4 mode  $5 dist
    local town=$1 spawn=$2 light=$3 mode=$4 dist=$5 sunalt
    if [ "$light" = "day" ]; then sunalt=45; else sunalt=-30; fi
    local tag="${town}_spawn${spawn}/${light}/dist${dist}m"
    log "capture $town $light dist${dist}m $mode -> $tag/$mode"
    for attempt in 1 2 3; do
        conda run -n PCLA15 python src/chroma_key_dataset_generator/capture_fase1.py \
            --town "$town" --spawn "$spawn" \
            --walk-max 300 --walk-step 2 \
            --leader-mode "$mode" --leader-gap-m "$dist" \
            --sun-altitude "$sunalt" --sun-azimuth 90 --weather ClearNoon \
            --run-tag "$tag" >>"$LOG" 2>&1
        if [ $? -eq 0 ]; then
            log "  OK ($town $light dist${dist}m $mode) attempt=$attempt"
            return 0
        fi
        log "  attempt $attempt FAILED — restarting CARLA and retrying"
        start_carla "$CURRENT_PKG" || return 1
    done
    log "  GIVING UP: $town $light dist${dist}m $mode"
    return 1
}

# ---------- PASS 1: carla_lite -> clean + noleader ----------
CURRENT_PKG=lite
start_carla lite || exit 1
for dist in "${DISTS[@]}"; do
    for spec in "${TOWNS[@]}"; do
        IFS=":" read -r town spawn <<< "$spec"
        for light in "${LIGHTS[@]}"; do
            run_cap "$town" "$spawn" "$light" clean "$dist"
            run_cap "$town" "$spawn" "$light" noleader "$dist"
        done
    done
done

# ---------- PASS 2: CARLA_Shipping -> marker ----------
CURRENT_PKG=shipping
start_carla shipping || exit 1
for dist in "${DISTS[@]}"; do
    for spec in "${TOWNS[@]}"; do
        IFS=":" read -r town spawn <<< "$spec"
        for light in "${LIGHTS[@]}"; do
            run_cap "$town" "$spawn" "$light" marker "$dist"
        done
    done
done

kill_carla
log "=== DONE ==="
find "$REPO/data/chroma_key_dataset/fase1" -name '*.png' 2>/dev/null | \
    awk -F/ '{print $(NF-4)"/"$(NF-3)"/"$(NF-2)"/"$(NF-1)}' | \
    sort | uniq -c | tee -a "$LOG"
echo "log at $LOG"
