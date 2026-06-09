"""Evaluate all patches under multi_start root, rank by detection rate, copy
the best one to multi_<ts>/best/."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--multi-root", type=Path, required=True)
    p.add_argument("--marker-dir", type=Path, required=True,
                   help="paired marker dataset (for warping patch at eval)")
    p.add_argument("--clean-dir", type=Path, required=True,
                   help="paired clean dataset (for true baseline)")
    args = p.parse_args()

    results = []
    seeds = sorted(d for d in args.multi_root.iterdir()
                   if d.is_dir() and d.name.startswith("seed"))
    print(f"Evaluating {len(seeds)} patches…")
    for d in seeds:
        patch_file = d / "patch_final.pt"
        if not patch_file.exists():
            print(f"  [skip] no patch_final.pt in {d}")
            continue
        eval_out = d / "eval.json"
        cmd = [
            "python", "-m", "src.yolo_chroma_attack.evaluate",
            "--run-dir", str(args.marker_dir),
            "--clean-run-dir", str(args.clean_dir),
            "--patch", str(patch_file),
            "--out", str(eval_out),
        ]
        print(f"  [eval] {d.name}", flush=True)
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        summary = json.loads(eval_out.read_text())
        results.append({
            "seed": d.name,
            "trained_rate": summary["trained"]["detection_rate"],
            "clean_rate": summary["clean"]["detection_rate"],
            "random_rate": summary["random"]["detection_rate"],
        })
    results.sort(key=lambda r: r["trained_rate"])
    print("\n=== Ranking by trained-patch detection rate (lower = stronger attack) ===")
    for r in results:
        print(f"  {r['seed']:10s}  trained={r['trained_rate']*100:5.1f}%  "
              f"clean={r['clean_rate']*100:5.1f}%  random={r['random_rate']*100:5.1f}%")
    best = results[0]
    best_dir = args.multi_root / best["seed"]
    out_best = args.multi_root / "best"
    out_best.mkdir(exist_ok=True)
    for f in ("patch_final.pt", "patch_final.png", "eval.json", "args.json"):
        if (best_dir / f).exists():
            shutil.copy(best_dir / f, out_best / f)
    (out_best / "from.txt").write_text(best["seed"] + "\n")
    print(f"\nBest = {best['seed']}  → copied to {out_best}")


if __name__ == "__main__":
    main()
