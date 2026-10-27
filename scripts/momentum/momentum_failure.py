#!/usr/bin/env python
"""WHY the momentum norm route is weak: localise the failure.

The route is two stages -- regress the Delta-p VECTOR, then take |.| outside the probe.
Four diagnostics, each isolating one candidate cause.

  Q1  Is stage 1 the bottleneck?  Corrupt the TRUE Delta-p to the model's own measured
      fidelity, take the norm, score AUC. If that matches the observed AUC, everything
      is explained by how well Delta-p is decoded and nothing else is wrong.
  Q2  Is the error in the DIRECTION of Delta-p or in its LENGTH? Only length survives
      the norm, so a model that gets direction right and length wrong fails this route
      for a reason invisible in the R^2.
  Q3  Does detection improve with violation SIZE? Flat => representational; rising =>
      a sensitivity floor.
  Q4  Are the INGREDIENTS there? Compose Delta-p from the model's own decoded masses and
      decoded velocity changes and compare with the direct probe (the S26 test).
"""
from __future__ import annotations
import json, sys
import numpy as np
sys.path.insert(0, 'scripts')
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))  # noqa: E401,E702
import _momentum_path  # noqa: F401,E402
from probe_energy import load_captures, cell_oof_regression, r2, auc

SEEDS = 8
BEST = {"outputs/cons_videomae-base": ("block.9.resid_post", "VideoMAE-base"),
        "outputs/cons_vjepa2-vitl":   ("block.17.resid_post", "V-JEPA2-ViT-L")}
rng = np.random.default_rng(0)

def norm_auc(y, v):
    a = auc(y, np.linalg.norm(v, axis=1)); return max(a, 1 - a)

for outdir, (site, tag) in BEST.items():
    acts, labels, pools, _ = load_captures(outdir)
    g   = np.array([L["seed"] for L in labels])
    y   = np.array([bool(L["violates_momentum"]) for L in labels])
    pf  = [{p["frame"]: p for p in L["per_frame"]} for L in labels]
    vel = lambda i, o, f: np.array(pf[i][f][o]["velocity"][:2], float)
    mA  = np.array([L["objects"]["A"]["mass"] for L in labels])
    mB  = np.array([L["objects"]["B"]["mass"] for L in labels])
    dvA = np.stack([vel(i,"A",22) - vel(i,"A",18) for i in range(len(labels))])
    dvB = np.stack([vel(i,"B",22) - vel(i,"B",18) for i in range(len(labels))])
    dp  = mA[:,None]*dvA + mB[:,None]*dvB
    X   = acts[site][:, pools.index("all.pre"), :]

    print(f"\n{'='*72}\n{tag}   site {site}   pool all.pre   n={len(y)}")

    # ---- observed, and stage-1 fidelity -------------------------------------
    P = np.mean([cell_oof_regression(X, dp, g, s) for s in range(SEEDS)], axis=0)
    R2 = r2(dp, P); obs = norm_auc(y, P)
    print(f"  stage 1  Delta-p regression R2 = {R2:+.3f}")
    print(f"  stage 2  AUC from |prediction|  = {obs:.3f}   <- OBSERVED")

    # ---- Q1: what AUC does that fidelity ALLOW? -----------------------------
    sd = np.sqrt(max(1 - R2, 0) * dp.var(axis=0).sum() / 2)
    sim = [norm_auc(y, dp + rng.normal(0, sd, dp.shape)) for _ in range(200)]
    print(f"  Q1  true Delta-p corrupted to that same fidelity -> AUC "
          f"{np.mean(sim):.3f} +- {np.std(sim):.3f}")
    print(f"      observed - attainable = {obs - np.mean(sim):+.3f}")

    # ---- Q2: radial vs tangential error -------------------------------------
    n_true = np.linalg.norm(dp, axis=1)
    ok = n_true > 1e-9                       # violating clips only: direction defined
    u  = dp[ok] / n_true[ok][:,None]
    rad  = (P[ok] * u).sum(1) - n_true[ok]                       # error along Delta-p
    tan  = P[ok] - (n_true[ok] + rad)[:,None]*u                  # error across it
    print(f"  Q2  on violating clips: |radial err| {np.abs(rad).mean():.3f}   "
          f"|tangential err| {np.linalg.norm(tan,axis=1).mean():.3f}   "
          f"(true |Delta-p| {n_true[ok].mean():.3f})")
    print(f"      cos(predicted, true) = {np.mean((P[ok]*u).sum(1)/np.maximum(np.linalg.norm(P[ok],axis=1),1e-9)):+.3f}")

    # ---- Q3: detection vs violation size ------------------------------------
    s = np.linalg.norm(P, axis=1)
    q = np.quantile(n_true[ok], [1/3, 2/3])
    print("  Q3  AUC by violation size (each bin vs ALL conserving clips):")
    for lo, hi, name in ((0, q[0], "small"), (q[0], q[1], "medium"), (q[1], np.inf, "large")):
        sel = ok & (n_true >= lo) & (n_true < hi)
        yy = np.r_[np.ones(sel.sum()), np.zeros((~ok).sum())]
        ss = np.r_[s[sel], s[~ok]]
        a = auc(yy.astype(bool), ss)
        print(f"        {name:7s} |Delta-p| in [{lo:.2f}, {hi if hi<9 else 9.99:.2f})  "
              f"n={sel.sum():>3}   AUC {max(a,1-a):.3f}")

    # ---- Q4: are the ingredients there? -------------------------------------
    parts = {}
    for nm, Y in (("log mass_A", np.log(mA)[:,None]), ("log mass_B", np.log(mB)[:,None]),
                  ("dv_A", dvA), ("dv_B", dvB)):
        Pp = np.mean([cell_oof_regression(X, Y, g, s) for s in range(4)], axis=0)
        parts[nm] = Pp
        print(f"  Q4  {nm:11s} R2 {r2(Y, Pp):+.3f}")
    comp = np.exp(parts["log mass_A"])*parts["dv_A"] + np.exp(parts["log mass_B"])*parts["dv_B"]
    print(f"      COMPOSED Delta-p from those parts: R2 {r2(dp, comp):+.3f}   "
          f"AUC {norm_auc(y, comp):.3f}    (direct: R2 {R2:+.3f}, AUC {obs:.3f})")
