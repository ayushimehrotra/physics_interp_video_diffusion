#!/usr/bin/env python
"""Momentum failure, stratified by MASS RATIO.

Delta-p = m_A dv_A + m_B dv_B is mass-weighted, so how well it can be detected may depend
on how unequal the two masses are. The confound to control is violation SIZE: detection is
already known to scale with |Delta-p|, and |Delta-p| is not independent of the mass ratio.
We therefore report, per mass-ratio tercile:

  * the AUC of the norm route,
  * the mean |Delta-p| in that tercile (is the comparison confounded?),
  * and an AUC recomputed after matching violation size ACROSS terciles, which is the
    only comparison that isolates the mass ratio itself.
"""
from __future__ import annotations
import json, sys
import numpy as np
sys.path.insert(0, 'scripts')
from probe_energy import load_captures, cell_oof_regression, r2, auc

SEEDS = 8
BEST = {"outputs_cons_videomae-base": ("block.9.resid_post", "VideoMAE-base"),
        "outputs_cons_vjepa2-vitl":   ("block.17.resid_post", "V-JEPA2-ViT-L")}
rng = np.random.default_rng(0)
out = {}
for outdir, (site, tag) in BEST.items():
    acts, labels, pools, _ = load_captures(outdir)
    g  = np.array([L["seed"] for L in labels])
    y  = np.array([bool(L["violates_momentum"]) for L in labels])
    pf = [{p["frame"]: p for p in L["per_frame"]} for L in labels]
    v  = lambda i, o, f: np.array(pf[i][f][o]["velocity"][:2], float)
    mA = np.array([L["objects"]["A"]["mass"] for L in labels])
    mB = np.array([L["objects"]["B"]["mass"] for L in labels])
    dp = np.stack([mA[i]*(v(i,"A",22)-v(i,"A",18)) + mB[i]*(v(i,"B",22)-v(i,"B",18))
                   for i in range(len(labels))])
    lr = np.abs(np.log(mA/mB))                       # 0 = equal masses
    nd = np.linalg.norm(dp, axis=1)
    X  = acts[site][:, pools.index("all.pre"), :]
    P  = np.mean([cell_oof_regression(X, dp, g, s) for s in range(SEEDS)], axis=0)
    s  = np.linalg.norm(P, axis=1)

    q = np.quantile(lr, [1/3, 2/3])
    print(f"\n{'='*74}\n{tag}   pool all.pre   site {site}")
    print(f"  overall AUC {max(auc(y,s),1-auc(y,s)):.3f}   Delta-p R2 {r2(dp,P):+.3f}")
    print(f"\n  {'mass-ratio tercile':26} {'n':>4} {'mean |dp|':>10} {'AUC':>7} {'dp R2':>8}")
    rec = {}
    for lo, hi, nm in ((0, q[0], "near-equal"), (q[0], q[1], "moderate"), (q[1], np.inf, "very unequal")):
        sel = (lr >= lo) & (lr < hi)
        a = auc(y[sel], s[sel])
        rec[nm] = {"n": int(sel.sum()), "auc": float(max(a, 1-a)),
                   "mean_dp": float(nd[sel & y].mean()), "r2": float(r2(dp[sel], P[sel]))}
        print(f"  |log m_A/m_B| in [{lo:.2f},{hi if hi<9 else 9.99:.2f})  "
              f"{sel.sum():>4} {nd[sel & y].mean():>10.3f} {max(a,1-a):>7.3f} {r2(dp[sel],P[sel]):>8.3f}")

    # ---- the comparison that isolates mass ratio: match violation size -------
    print("\n  after matching violation size across terciles (resampled to a common"
          " |dp| distribution, 200 draws):")
    viol = np.where(y)[0]; cons = np.where(~y)[0]
    edges = np.quantile(nd[viol], np.linspace(0, 1, 6))
    for lo, hi, nm in ((0, q[0], "near-equal"), (q[0], q[1], "moderate"), (q[1], np.inf, "very unequal")):
        aucs = []
        for _ in range(200):
            pick = []
            for b in range(5):                      # equal count from each |dp| bin
                cand = [i for i in viol if lr[i] >= lo and lr[i] < hi
                        and edges[b] <= nd[i] < (edges[b+1] if b < 4 else np.inf)]
                if cand: pick += list(rng.choice(cand, size=min(8, len(cand)), replace=True))
            if not pick: continue
            yy = np.r_[np.ones(len(pick)), np.zeros(len(cons))].astype(bool)
            ss = np.r_[s[pick], s[cons]]
            a = auc(yy, ss); aucs.append(max(a, 1-a))
        print(f"    {nm:14} AUC {np.mean(aucs):.3f} +- {np.std(aucs):.3f}")
        rec[nm]["auc_size_matched"] = float(np.mean(aucs))
    out[tag] = rec
json.dump(out, open("docs/artifacts/momentum_by_massratio.json", "w"), indent=1)
print("\nwrote docs/artifacts/momentum_by_massratio.json")
