#!/usr/bin/env python
"""KE CONFOUND + SUBSPACE, on the full 450-clip energy set.

PART 1 -- CONFOUND. The impossible ending's post-contact velocities sit exactly k times
further from elastic than the legal ending's, so a detector of RAW DEPARTURE from elastic
reproduces the label with no notion of impossibility. We build that detector explicitly,
score it, and measure how much of the model's own surprise ordering it accounts for.

PART 2 -- SUBSPACE. The paper argues the encoders read energy AS ELASTICITY. That is
testable geometrically: if so, the probe subspace for `energy created` and the subspace for
restitution `e` should coincide, and both should be separate from an unrelated factor.
Brackets follow scripts/momentum/geometry_v1.py -- a split-half CEILING (the same factor fit twice
is not perfectly aligned) and a shuffled-label NULL (ridge on noise still loads on dominant
variance). Cross-factor fits use DISJOINT halves so no shared noise inflates the overlap.
    align = (overlap - null) / (ceiling - null)
"""
from __future__ import annotations
import json, re, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, 'scripts')
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))  # noqa: E401,E702
import _momentum_path  # noqa: F401,E402
from probe_energy import load_captures, auc
from geometry_v1 import fit_dirs, overlap, halves, shuffled, linear_dependence

ALPHA, REPS = 100.0, 40
rng = np.random.default_rng(0)

def scene_targets(labels):
    """Per-clip physical quantities, from the simulator state."""
    T = defaultdict(list)
    for L in labels:
        c = L["collision"]; pf = {p["frame"]: p for p in L["per_frame"]}
        g = lambda o, f: np.array(pf[f][o]["velocity"][:2], float)
        n = np.array(c["contact_normal"][:2], float); n /= np.linalg.norm(n)
        vA, vB, wA, wB = g("A",18), g("B",18), g("A",22), g("B",22)
        e = abs((wA-wB) @ n) / max(abs((vA-vB) @ n), 1e-12)
        T["restitution"].append([e])
        T["energy created"].append([float(e > 1.0)])
        T["speed"].append([np.linalg.norm(vA)])
        T["post speed"].append([np.linalg.norm(wA)])
    return {k: np.array(v, float) for k, v in T.items()}

for outdir, tag in (("outputs/evoe450_videomae-base", "VideoMAE-base"),
                    ("outputs/evoe450_vjepa2-vitl",   "V-JEPA2-ViT-L")):
    acts, labels, pools, ids = load_captures(outdir)
    T = scene_targets(labels)
    arm = np.array([("impossible" if "super" in i else "legal" if "sub" in i else "anchor")
                    for i in ids])
    seed = np.array([re.search(r"seed(\d+)$", i).group(1) for i in ids])

    # ---------- PART 1: the raw-departure detector ---------------------------
    pf = [{p["frame"]: p for p in L["per_frame"]} for L in labels]
    post = np.stack([np.r_[np.array(pf[i][22]["A"]["velocity"][:2], float),
                           np.array(pf[i][22]["B"]["velocity"][:2], float)]
                     for i in range(len(labels))])
    anc = {s: post[(seed == s) & (arm == "anchor")][0] for s in np.unique(seed)
           if ((seed == s) & (arm == "anchor")).any()}
    dep = np.array([np.linalg.norm(post[i] - anc[seed[i]]) for i in range(len(labels))])
    m = arm != "anchor"
    y = (arm[m] == "impossible")
    a = auc(y, dep[m])
    print(f"\n{'='*74}\n{tag}")
    print(f"  PART 1  raw departure from elastic, impossible vs legal: AUC {max(a,1-a):.3f}")
    ratio = np.array([dep[(seed==s)&(arm=='impossible')][0] / dep[(seed==s)&(arm=='legal')][0]
                      for s in np.unique(seed) if ((seed==s)&(arm=='impossible')).any()
                      and ((seed==s)&(arm=='legal')).any() and dep[(seed==s)&(arm=='legal')][0] > 0])
    print(f"          departure ratio impossible/legal: {ratio.mean():.3f} +- {ratio.std():.3f}"
          f"   (design predicts k = 1.5)")

    # ---------- PART 2: subspaces -------------------------------------------
    site = sorted(acts, key=lambda s: 0 if s.startswith("embed")
                  else int(re.search(r"block\.(\d+)", s).group(1))+1)[len(acts)//2]
    X = acts[site][:, pools.index("all.pre"), :].astype(np.float64)
    names = ["energy created", "restitution", "speed", "post speed"]
    print(f"  PART 2  subspace alignment at {site}, pool all.pre")
    print(f"          label algebra first (R2 of predicting one target from another):")
    for i, a_ in enumerate(names):
        for b_ in names[i+1:]:
            print(f"            {a_:15} <- {b_:12} R2 {linear_dependence(T[a_], T[b_]):+.3f}")
    print(f"\n          {'pair':34} {'overlap':>8} {'null':>7} {'ceiling':>8} {'align':>7}")
    res = {}
    for i, A_ in enumerate(names):
        for B_ in names[i:]:
            ov, ce, nu = [], [], []
            for _ in range(REPS):
                h1, h2 = halves(len(labels), rng)
                qa1 = fit_dirs(X, T[A_], h1, ALPHA); qb2 = fit_dirs(X, T[B_], h2, ALPHA)
                qa2 = fit_dirs(X, T[A_], h2, ALPHA)
                ov.append(overlap(qa1, qb2))
                ce.append(overlap(qa1, qa2))
                nu.append(overlap(fit_dirs(X, shuffled(T[A_], h1, rng), h1, ALPHA),
                                  fit_dirs(X, shuffled(T[B_], h2, rng), h2, ALPHA)))
            o, c, n0 = np.mean(ov), np.mean(ce), np.mean(nu)
            al = (o - n0) / (c - n0) if (c - n0) > .05 else float("nan")
            res[f"{A_}|{B_}"] = {"overlap": o, "null": n0, "ceiling": c, "align": al}
            star = "  <--" if A_ != B_ and al == al and al > .8 else ""
            print(f"          {A_+' | '+B_:34} {o:>8.3f} {n0:>7.3f} {c:>8.3f} {al:>7.2f}{star}")
    json.dump(res, open(f"docs/artifacts/energy_subspace_{tag}.json", "w"), indent=1)
