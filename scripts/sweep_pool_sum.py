#!/usr/bin/env python
"""Depth sweep of v_A, v_B, their SUM and their DIFFERENCE, at the frame-wide pool.

A frame-wide mean over an object-agnostic per-token code is a SUM over whatever objects
fall in the pool. The prediction is that `v_A + v_B` survives frame-wide while `v_A - v_B`
does not, and that neither individual velocity does as well as the sum.
"""
import json, re, sys
import numpy as np
sys.path.insert(0, 'scripts')
from probe_energy import load_captures, cell_oof_regression, r2

SEEDS = 12
depth = lambda s: 0 if s.startswith("embed") else int(re.search(r"block\.(\d+)", s).group(1)) + 1
out = {}
for outdir, tag in (("outputs_cons_videomae-base", "VideoMAE-base"),
                    ("outputs_cons_vjepa2-vitl", "V-JEPA2-ViT-L")):
    acts, labels, pools, _ = load_captures(outdir)
    keep = [i for i, L in enumerate(labels) if not L["violates_momentum"]]
    g = np.array([labels[i]["seed"] for i in keep])
    c = [labels[i]["collision"]["pre"] for i in keep]
    vA = np.array([x["velocity_A"][:2] for x in c], float)
    vB = np.array([x["velocity_B"][:2] for x in c], float)
    T = {"$v_A$": vA, "$v_B$": vB, "$v_A + v_B$": vA + vB, "$v_A - v_B$": vA - vB}
    sites = sorted(acts, key=depth); dmax = depth(sites[-1])
    rec = {}
    print(f"\n{tag}   pool all.pre")
    for name, Y in T.items():
        curve = {}
        for s_ in sites:
            X = acts[s_][:, pools.index("all.pre"), :][keep]
            v = [r2(Y, cell_oof_regression(X, Y, g, k)) for k in range(SEEDS)]
            curve[s_] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                         "depth": depth(s_)}
        rec[name] = curve
        pk = max(curve.values(), key=lambda d: d["mean"])
        print(f"   {name:12s} embed {curve[sites[0]]['mean']:+.3f} -> peak "
              f"{pk['mean']:+.3f} at depth {pk['depth']/dmax:.2f}")
    out[tag] = {"depth_sweep": rec}
json.dump(out, open("docs/artifacts/pool_sum_sweep.json", "w"), indent=1)
print("\nwrote docs/artifacts/pool_sum_sweep.json")
