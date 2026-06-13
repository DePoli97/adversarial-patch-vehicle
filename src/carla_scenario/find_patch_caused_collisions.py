"""Identify simlingo seeds where the patch causes a collision that does NOT
happen in clean conditions (clean=no_coll AND patch=coll). For each such seed,
list the spawn index, distances, and the simlingo language excerpts so the user
can review the cases one by one.

Usage:
    python src/carla_scenario/find_patch_caused_collisions.py \\
        --root experiments/carla_scenarios/multi_agent_20260614_001906 \\
        --agent simlingo_simlingo --town Town04
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def has_leader_collision(run_dir: Path, dist_threshold_m: float = 5.0) -> bool:
    """Same rule as the compare notebook: collision flag AND distance < 5m."""
    tel = pd.read_csv(run_dir / "telemetry.csv")
    mask = (tel["collision_detected"] > 0) & (tel["distance_m"] < dist_threshold_m)
    return bool(mask.any())


def run_metrics(run_dir: Path) -> dict:
    tel = pd.read_csv(run_dir / "telemetry.csv")
    summary = json.loads((run_dir / "summary.json").read_text())
    post = tel[tel["sim_time_s"] >= 10.0]
    return {
        "seed": summary.get("seed"),
        "spawn_idx": summary.get("leader_spawn_index"),
        "spawn_xy": summary.get("leader_spawn_xy"),
        "leader_collision": has_leader_collision(run_dir),
        "min_dist_after_brake": float(post["distance_m"].min()) if len(post) else None,
        "total_collisions_flag": int(tel["collision_detected"].sum()),
        "run_dir": str(run_dir),
    }


def load_agent_town(root: Path, agent: str, town: str) -> tuple[dict, dict]:
    """Return {seed: row} for clean and patch conditions."""
    out = {}
    for cond in ("clean", "patch"):
        cond_root = root / agent / cond / town
        seeded = {}
        if not cond_root.exists():
            out[cond] = seeded
            continue
        for run_dir in sorted(cond_root.iterdir()):
            if not run_dir.is_dir():
                continue
            if not (run_dir / "summary.json").exists():
                continue
            try:
                r = run_metrics(run_dir)
            except Exception as e:
                print(f"[skip] {run_dir.name}: {e}")
                continue
            if r["seed"] is None:
                continue
            seeded[int(r["seed"])] = r
        out[cond] = seeded
    return out["clean"], out["patch"]


def excerpt_language(run_dir: Path, ticks=(0, 100, 150, 200, 220, 240, 260, 280)) -> str:
    """Return a short multi-line excerpt of the simlingo language log."""
    tsv = run_dir / "simlingo_language.tsv"
    if not tsv.exists():
        return "(no simlingo_language.tsv)"
    df = pd.read_csv(tsv, sep="\t")
    # Strip the "Waypoints:" tail for readability.
    df["language"] = df["language"].str.split("Waypoints:").str[0].str.strip()
    lines = []
    for tick in ticks:
        row = df[df["step"] == tick]
        if len(row) == 0:
            continue
        lines.append(f"  t={tick/20:.1f}s  {row['language'].iloc[0]}")
    return "\n".join(lines) if lines else "(no matching ticks)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--agent", default="simlingo_simlingo")
    p.add_argument("--town", default="Town04")
    p.add_argument("--show-language", action="store_true",
                   help="Also print simlingo language excerpts for the matching seeds.")
    args = p.parse_args()

    clean, patch = load_agent_town(args.root, args.agent, args.town)
    common_seeds = sorted(set(clean) & set(patch))
    print(f"\n{args.agent} on {args.town} (root={args.root.name})")
    print(f"clean runs : {len(clean)}")
    print(f"patch runs : {len(patch)}")
    print(f"common seeds: {len(common_seeds)}")

    rows_summary = []
    transition_seeds = []
    for s in common_seeds:
        c = clean[s]; p_ = patch[s]
        rows_summary.append({
            "seed": s,
            "spawn_idx": c["spawn_idx"],
            "clean_coll": c["leader_collision"],
            "clean_min_dist": round(c["min_dist_after_brake"], 2) if c["min_dist_after_brake"] else None,
            "patch_coll": p_["leader_collision"],
            "patch_min_dist": round(p_["min_dist_after_brake"], 2) if p_["min_dist_after_brake"] else None,
        })
        if (not c["leader_collision"]) and p_["leader_collision"]:
            transition_seeds.append((s, c, p_))

    df = pd.DataFrame(rows_summary)
    print("\nPer-seed overview:")
    print(df.to_string(index=False))

    print(f"\n=== Seeds where clean=no_coll AND patch=coll ({len(transition_seeds)}) ===")
    for s, c, p_ in transition_seeds:
        print(f"\nseed {s}  spawn=[{c['spawn_idx']}] {c['spawn_xy']}")
        print(f"  CLEAN: min_dist={c['min_dist_after_brake']:.2f} m  -> {c['run_dir']}")
        print(f"  PATCH: min_dist={p_['min_dist_after_brake']:.2f} m  -> {p_['run_dir']}")
        if args.show_language:
            print("  --- simlingo language (PATCH run) ---")
            print(excerpt_language(Path(p_["run_dir"])))


if __name__ == "__main__":
    main()
