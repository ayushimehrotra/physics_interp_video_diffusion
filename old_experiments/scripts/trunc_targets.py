#!/usr/bin/env python
"""WHICH post-contact quantity should carry the elastic-vs-inelastic answer?

For a frictionless two-sphere impact EVERY post-contact quantity is AFFINE in
the restitution e, because the impulse is a single scalar along the contact
normal:  J = (1+e)*mu*u_n,  v_A_post = v_A - (J/m_A) n,  v_B_post = v_B + (J/m_B) n.

So any target T can be written   T(e) = T0 + e * T1   and T1 -- the derivative of
the target with respect to restitution -- is that target's ENTIRE sensitivity to
the elastic/inelastic question. Targets differ enormously in it:

    target                        T1                       e-leverage
    v_A_post                      -c_A n                   c_A = mu u_n/m_A
    v_B_post                      +c_B n                   c_B = mu u_n/m_B
    v_A_post + v_B_post           (c_B - c_A) n            vanishes at m_A = m_B
    (v_A_post-v_B_post).n         -u_n                     u_n, the approach speed
    m_A v_A_post + m_B v_B_post   0                        NONE -- conserved

That last row is the point of the negative control: total momentum post EQUALS
total momentum pre, so it is e-free by construction and cannot answer the
question no matter how well it is decoded. The mass-weighted sum is the wrong
thing to probe. The UNWEIGHTED sum does carry e, but its leverage collapses
whenever the two masses are similar.

The relative normal velocity is the natural target: (v_A-v_B).n after contact is
exactly -e * u_n, so e is 100% of its variance once u_n is known, and its
leverage u_n is never small (5th pct 0.32) where c_A reaches 0.017.

ESTIMATOR (same for every target, so they are comparable)
---------------------------------------------------------
Project the held-out prediction P onto the scene's own T1 direction:

    y = (P - T0).T1/|T1|,   x = |T1|          truth: y = e * x

    regress   y = b0 * x + b1 * (e_true * x)

    b0 = 0, b1 = 1   the forecast TRACKS this scene's restitution
    b0 = d, b1 = 0   the forecast ignores it, committing to a fixed default d
                     (d = 1 perfectly elastic, d = 0 perfectly inelastic)

POWER is reported for every row, because a null is meaningless without it: the
observed residuals are permuted across scenes (real magnitude, no e-dependence)
and added to the true target, simulating a probe that tracks e PERFECTLY at the
observed error level. If that simulation does not return b1 ~ 1, the row has no
power and its null says nothing.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import sys
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))  # noqa: E401,E702
import _momentum_path  # noqa: F401,E402
from trunc_restitution import (ridge, r2, oof_predict, scene_geometry, post)   # noqa: E402


def targets(g):
    """name -> (T0, T1) with T(e) = T0 + e*T1, all as 1-D arrays."""
    n, cA, cB = g["n"], g["cA"], g["cB"]
    vA, vB, mA, mB, un = g["vA"], g["vB"], g["mA"], g["mB"], g["un"]
    return {
        "v_A_post":            (vA - cA * n,                 -cA * n),
        "v_B_post":            (vB + cB * n,                 +cB * n),
        "v_A_post + v_B_post": (vA + vB + (cB - cA) * n,     (cB - cA) * n),
        "(v_A-v_B).n  post":   (np.array([0.0]),             np.array([-un])),
        "total momentum post": (mA * vA + mB * vB,           np.zeros(2)),
    }


def project(G, P, name):
    y, x = [], []
    for g, p in zip(G, P):
        T0, T1 = targets(g)[name]
        L = np.linalg.norm(T1)
        if L < 1e-12:
            return None, None
        y.append((p - T0) @ T1 / L)
        x.append(L)
    return np.array(y), np.array(x)


def fit(y, x, e):
    b, *_ = np.linalg.lstsq(np.stack([x, e * x], 1), y, rcond=None)
    return float(b[0]), float(b[1])


def boot(y, x, e, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    out = [fit(*(v[i] for v in (y, x, e))) for i in
           (rng.choice(len(y), len(y), replace=True) for _ in range(n))]
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="outputs/trunc_videomae-base_acts.npz")
    ap.add_argument("--pool", default="all.all")
    ap.add_argument("--ks", default="19,20,48")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    z = np.load(args.acts, allow_pickle=True)
    pools = list(z["pools"]); pi = [pools.index(p) for p in args.pool.split("+")]
    labels = [json.loads(s) for s in z["labels"]]
    G = [scene_geometry(l) for l in labels]
    e = np.array([g["e"] for g in G])
    names = list(targets(G[0]))
    ks = [int(v) for v in args.ks.split(",")]

    print(f"{args.acts}  n={len(G)}  pool '{args.pool}'   contact at frame 20.03")
    print(f"true e: min {e.min():.3f} median {np.median(e):.3f} max {e.max():.3f} "
          f"mean {e.mean():.3f}\n")
    print("e-LEVERAGE of each target (|T1|, the target's whole sensitivity to restitution)")
    print(f"  {'target':<22}{'median |T1|':>12}{'5th pct':>10}{'e-share of target var':>23}")
    for nm in names:
        T1 = np.array([np.linalg.norm(targets(g)[nm][1]) for g in G])
        T = np.stack([np.atleast_1d(targets(g)[nm][0] + g["e"] * targets(g)[nm][1]) for g in G])
        share = np.var(e * T1) / max(T.var(0).sum(), 1e-12)
        print(f"  {nm:<22}{np.median(T1):>12.3f}{np.percentile(T1, 5):>10.3f}{share:>22.1%}")

    rep = {"acts": args.acts, "pool": args.pool, "rows": {}}
    print(f"\n{'='*104}")
    print(f"  {'k':>3} {'target':<22}{'R2':>7}{'b0 = implied default':>22}{'b1 = tracks e':>22}"
          f"{'power: b1 if perfect':>22}")
    print("=" * 104)
    rng = np.random.default_rng(0)
    for k in ks:
        A = np.concatenate([z[f"k{k}"][:, :, q, :] for q in pi], axis=2).astype(np.float64)
        for nm in names:
            Y = np.stack([np.atleast_1d(targets(g)[nm][0] + g["e"] * targets(g)[nm][1]) for g in G])
            P = oof_predict(A, Y)
            y, x = project(G, P, nm)
            if y is None:
                print(f"  {k:>3} {nm:<22}{r2(Y, P):>7.3f}{'--  e-free by construction':>44}")
                rep["rows"][f"{k}|{nm}"] = {"r2": r2(Y, P), "e_free": True}
                continue
            b0, b1 = fit(y, x, e)
            bs = boot(y, x, e)
            c0 = np.percentile(bs[:, 0], [2.5, 97.5]); c1 = np.percentile(bs[:, 1], [2.5, 97.5])
            R = P - Y
            sim = []
            for _ in range(300):
                ys, xs = project(G, Y + R[rng.permutation(len(G))], nm)
                sim.append(fit(ys, xs, e)[1])
            sim = np.array(sim)
            print(f"  {k:>3} {nm:<22}{r2(Y, P):>7.3f}"
                  f"{f'{b0:+.2f} [{c0[0]:+.2f},{c0[1]:+.2f}]':>22}"
                  f"{f'{b1:+.2f} [{c1[0]:+.2f},{c1[1]:+.2f}]':>22}"
                  f"{f'{sim.mean():+.2f} sd {sim.std():.2f}':>22}")
            rep["rows"][f"{k}|{nm}"] = {
                "r2": r2(Y, P), "b0_implied_default": b0, "b0_ci95": c0.tolist(),
                "b1_tracks_e": b1, "b1_ci95": c1.tolist(),
                "power_b1_if_perfect": float(sim.mean()), "power_sd": float(sim.std())}
        print()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(rep, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
