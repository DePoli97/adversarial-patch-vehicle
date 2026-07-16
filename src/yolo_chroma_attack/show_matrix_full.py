"""Print the full eval matrix (13 patches x 3 roads) of trained detection_rate."""
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
roads = ["Town04", "Town07", "Town11"]
patches = ["Town04_spawn273_day", "Town04_spawn273_night", "Town04_spawn273",
           "Town07_spawn38_day", "Town07_spawn38_night", "Town07_spawn38",
           "Town11_spawn1713_day", "Town11_spawn1713_night", "Town11_spawn1713",
           "pooled_all_day", "pooled_all_night", "pooled", "generalist"]


def rate(p, r):
    f = out / f"patch_{p}__road_{r}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())["trained"]["detection_rate"] * 100


print("\n=== detection_rate TRAINED, illum-fix (lower = better attack) ===")
hdr = "patch \\ road"
print(f"{hdr:24s}" + "".join(f"{r:>9s}" for r in roads))
for p in patches:
    row = f"{p:24s}"
    for r in roads:
        v = rate(p, r)
        row += f"{v:8.1f}%" if v is not None else "     -  "
    print(row)

# clean reference (same per road, read from any patch's json)
print("\nclean (no patch):")
for r in roads:
    for p in patches:
        f = out / f"patch_{p}__road_{r}.json"
        if f.exists():
            c = json.loads(f.read_text()).get("clean", {}).get("detection_rate")
            if c is not None:
                print(f"  {r}: {c*100:.1f}%")
            break

print("\n=== VERDETTO per strada: per-road vs pooled vs generalist ===")
for r in roads:
    own = {"Town04": "Town04_spawn273", "Town07": "Town07_spawn38",
           "Town11": "Town11_spawn1713"}[r]
    o, pl, g = rate(own, r), rate("pooled", r), rate("generalist", r)
    parts = []
    if o is not None: parts.append(f"per-strada={o:.1f}%")
    if pl is not None: parts.append(f"pooled={pl:.1f}%")
    if g is not None: parts.append(f"generalist={g:.1f}%")
    best = min((v, n) for v, n in [(o, "per-strada"), (pl, "pooled"), (g, "generalist")] if v is not None)
    print(f"  {r}: " + "  ".join(parts) + f"  -> vince {best[1]}")
