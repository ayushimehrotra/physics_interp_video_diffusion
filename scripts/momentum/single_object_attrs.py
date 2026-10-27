#!/usr/bin/env python
"""APPENDIX A: single-object attributes across depth, and what a frame-wide pool holds.

Two measurements on `conservation_set_v1`'s 150 momentum-conserving clips, so the
appendix quotes numbers from the same set as the main text.

1. DEPTH SWEEP of per-object attributes (velocity, speed, direction, radius, mass) read
   from that object's own token pool -- the multi-object counterpart of the single-object
   probes in the interpretability literature.
2. WHAT A FRAME-WIDE POOL CONTAINS. With two objects present, the frame-wide mean is a
   SUM: `v_A + v_B` reads far better than either part, and `v_A - v_B` reads at nothing.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))  # noqa: E401,E702
import _momentum_path  # noqa: F401,E402
from probe_energy import load_captures, cell_oof_regression, r2  # noqa: E402

SEEDS = 12
def depth(site): return 0 if site.startswith("embed") else int(re.search(r"block\.(\d+)", site).group(1)) + 1

def targets(labels, idx):
    c = [labels[i]["collision"]["pre"] for i in idx]
    vA = np.array([x["velocity_A"][:2] for x in c], float)
    vB = np.array([x["velocity_B"][:2] for x in c], float)
    spd = np.linalg.norm(vA, axis=1, keepdims=True)
    hd  = vA / np.maximum(spd, 1e-9)
    rad = np.array([[labels[i]["objects"]["A"]["radius"]] for i in idx], float)
    lm  = np.array([[np.log(labels[i]["objects"]["A"]["mass"])] for i in idx], float)
    return {"velocity": vA, "speed": spd, "direction": hd, "radius": rad, "log mass": lm}, vA, vB

def score(X, Y, groups):
    return float(np.mean([r2(Y, cell_oof_regression(X, Y, groups, s)) for s in range(SEEDS)])), \
           float(np.std([r2(Y, cell_oof_regression(X, Y, groups, s)) for s in range(SEEDS)], ddof=1))

out = {}
for outdir, tag in (("outputs/cons_videomae-base", "VideoMAE-base"),
                    ("outputs/cons_vjepa2-vitl", "V-JEPA2-ViT-L")):
    acts, labels, pools, _ = load_captures(outdir)
    keep = [i for i, L in enumerate(labels) if not L["violates_momentum"]]
    groups = np.array([labels[i]["seed"] for i in keep])
    T, vA, vB = targets(labels, keep)
    sites = sorted(acts, key=depth)
    rec = {"depth_sweep": {}, "pools": {}}

    # ---- 1. per-object attributes, read from A's own pool, across depth ----------
    for name, Y in T.items():
        rec["depth_sweep"][name] = {}
        for site in sites:
            X = acts[site][:, pools.index("A.pre"), :][keep]
            m, s = score(X, Y, groups)
            rec["depth_sweep"][name][site] = {"mean": m, "sd": s, "depth": depth(site)}
        best = max(rec["depth_sweep"][name], key=lambda k: rec["depth_sweep"][name][k]["mean"])
        print(f"  {tag:14s} {name:10s} best {rec['depth_sweep'][name][best]['mean']:+.3f} at {best}", flush=True)

    # ---- 2. what a frame-wide pool holds: the SUM, not the parts -----------------
    combo = {"$v_A$": vA, "$v_B$": vB, "$v_A + v_B$": vA + vB, "$v_A - v_B$": vA - vB}
    for pool in ("A.pre", "B.pre", "all.pre", "all.all"):
        rec["pools"][pool] = {}
        for cname, Y in combo.items():
            best = max(score(acts[site][:, pools.index(pool), :][keep], Y, groups)[0]
                       for site in sites if not site.startswith("embed"))
            rec["pools"][pool][cname] = best
        print(f"  {tag:14s} {pool:8s} " +
              "  ".join(f"{k} {v:+.3f}" for k, v in rec["pools"][pool].items()), flush=True)
    out[tag] = rec

Path("docs/artifacts").mkdir(parents=True, exist_ok=True)
json.dump(out, open("docs/artifacts/single_object_attrs.json", "w"), indent=1)
print("\nwrote docs/artifacts/single_object_attrs.json")
