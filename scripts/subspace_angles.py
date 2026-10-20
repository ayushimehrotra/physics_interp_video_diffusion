#!/usr/bin/env python
"""Do A and B occupy different subspaces inside the *same* tokens?

Re-runs the §18b claim from scratch on the momentum dataset captures. Both
targets are read out of the A-local pool, so the question is purely geometric:
are the directions that carry A's velocity the same ones that carry B's?

Three numbers are needed, not one. A raw principal angle means nothing without
its two brackets:

  null     two random k-dim subspaces of the same activation space. This is what
           "unrelated" looks like at this k and this dim -- NOT 90 deg.
  ceiling  the same target, fit twice on disjoint halves of train at matched
           sample size. This is what "identical subspace, estimated twice from
           finite data" looks like -- NOT 0 deg.
  measured velocity_A vs velocity_B.

If measured sits at the null, the subspaces are as unrelated as chance allows.
If it sits at the ceiling, they are the same subspace and the apparent
separation was estimation noise.

Also checks the §34b rule before believing any of it (is one target a linear
function of the other?) and closes with the functional test: strip A's velocity
subspace out of the activations and re-probe for B's velocity.

CPU only -- reads cached activations, fits ridge, no model load.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steer_velocity import (                                        # noqa: E402
    fit_readout, load_captures, r2_of, split_indices,
)
from patch_velocity import inlp_subspace                            # noqa: E402
from physics_interp.config import load_config                       # noqa: E402
from physics_interp.interp import extract_label                     # noqa: E402


def principal_angles(Q1, Q2):
    """Degrees, ascending. Q1, Q2 orthonormal [d, k]."""
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def fmt(a, overlap):
    return (f"min {a[0]:5.1f}  med {np.median(a):5.1f}  mean {a.mean():5.1f}  "
            f"max {a[-1]:5.1f}   overlap {overlap:.3f}")


def overlap(Q1, Q2):
    """Mean cos^2 of the principal angles: the fraction of a random unit vector
    in span(Q1) that survives projection onto span(Q2). 1 = same subspace,
    k/dim = chance."""
    return float((np.linalg.norm(Q1.T @ Q2, "fro") ** 2) / Q1.shape[1])


def linear_dependence(Ya, Yb):
    """R2 of predicting Yb from Ya by OLS. ~1.0 => the §34b trap."""
    A = np.hstack([Ya, np.ones((len(Ya), 1))])
    pred = A @ np.linalg.lstsq(A, Yb, rcond=None)[0]
    ss_res = ((Yb - pred) ** 2).sum()
    ss_tot = ((Yb - Yb.mean(0)) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", nargs="?", default="configs/momentum_probe.yaml")
    p.add_argument("--capture-dir", default="outputs_v3_videomae-base")
    p.add_argument("--block", type=int, default=6)
    p.add_argument("--pool", default="A.pre")
    p.add_argument("--rounds", type=int, default=8, help="INLP rounds -> k = 2*rounds")
    p.add_argument("--seeds", type=int, default=5, help="split seeds to repeat over")
    args = p.parse_args()

    cfg = load_config(args.config)
    acts, labels, pools, _ = load_captures(Path(args.capture_dir), cfg.name)
    n = len(labels)
    pi = {q: i for i, q in enumerate(pools)}
    site = f"block.{args.block}.resid_post"
    X = acts[site][:, pi[args.pool], :].astype(np.float64)
    k = 2 * args.rounds

    Y = {t: np.stack([extract_label(l, t) for l in labels])
         for t in ("velocity_A_pre", "velocity_B_pre", "radius_A")}
    vA, vB = Y["velocity_A_pre"], Y["velocity_B_pre"]

    print(f"{args.capture_dir}  {site}  pool {args.pool}  "
          f"n={n}  dim={X.shape[1]}  k={k}\n")

    # --- §34b guard: is either target a linear function of the other? -------- #
    print("label check (a target that is a linear combination of another makes")
    print("every subspace comparison vacuous -- §34b):")
    print(f"  R2(v_B <- v_A)  {linear_dependence(vA, vB):+.4f}")
    print(f"  R2(v_A <- v_B)  {linear_dependence(vB, vA):+.4f}")
    for i, c in enumerate("xy"):
        print(f"  corr(v_A{c}, v_B{c}) {np.corrcoef(vA[:, i], vB[:, i])[0, 1]:+.3f}", end="")
    print("\n")

    rng = np.random.default_rng(0)
    rows = {key: [] for key in ("A_vs_B", "A_vs_radius", "B_vs_radius",
                                "ceiling_A", "ceiling_B", "null_random")}
    ovl = {key: [] for key in rows}

    def record(key, Q1, Q2):
        rows[key].append(principal_angles(Q1, Q2))
        ovl[key].append(overlap(Q1, Q2))

    for seed in range(args.seeds):
        tr, te = split_indices(n, split_seed=seed)
        VA = inlp_subspace(X, vA, tr, args.rounds)
        VB = inlp_subspace(X, vB, tr, args.rounds)
        U = inlp_subspace(X, Y["radius_A"], tr, k)      # 1 col/round -> k cols

        # ceiling: same target, disjoint halves of train, matched size
        h = len(tr) // 2
        tr1, tr2 = tr[:h], tr[h:2 * h]
        record("ceiling_A", inlp_subspace(X, vA, tr1, args.rounds),
               inlp_subspace(X, vA, tr2, args.rounds))
        record("ceiling_B", inlp_subspace(X, vB, tr1, args.rounds),
               inlp_subspace(X, vB, tr2, args.rounds))
        # null: two random k-dim subspaces of the same space
        R1, _ = np.linalg.qr(rng.normal(size=(X.shape[1], k)))
        R2, _ = np.linalg.qr(rng.normal(size=(X.shape[1], k)))
        record("null_random", R1, R2)

        record("A_vs_B", VA, VB)
        record("A_vs_radius", VA, U)
        record("B_vs_radius", VB, U)

    print(f"principal angles, mean over {args.seeds} split seeds (90 = orthogonal):")
    for key in ("ceiling_A", "ceiling_B", "A_vs_B", "A_vs_radius",
                "B_vs_radius", "null_random"):
        a = np.stack(rows[key]).mean(0)
        print(f"  {key:14s} {fmt(a, float(np.mean(ovl[key])))}")

    # --- the functional test: does removing one object's subspace touch the
    #     other? Run it deep enough that the target itself actually dies --
    #     velocity is redundantly coded, so a shallow removal proves nothing.
    tr, te = split_indices(n, split_seed=0)
    rng = np.random.default_rng(1)

    def probe_all(Xw):
        return [r2_of(fit_readout(Xw, Y[t], tr), Xw, Y[t], te)
                for t in ("velocity_A_pre", "velocity_B_pre", "radius_A")]

    depths = (0, 2, 4, 8, 16, 24, 32)
    for owner, y in (("A", vA), ("B", vB)):
        print(f"\nfunctional test -- project {owner}'s velocity subspace out of the")
        print("activations, then re-probe everything. Held-out R2:")
        print(f"  {'removed':>8s}  {'vel_A':>7s}  {'vel_B':>7s}  {'radius_A':>8s}")
        for r_rounds in depths:
            Xw = X
            if r_rounds:
                Q = inlp_subspace(X, y, tr, r_rounds)
                Xw = X - (X @ Q) @ Q.T
            a, b, c = probe_all(Xw)
            tag = f"{2 * r_rounds}d" if r_rounds else "none"
            print(f"  {tag:>8s}  {a:7.3f}  {b:7.3f}  {c:8.3f}")
        # control: the same number of dimensions, chosen at random
        Q, _ = np.linalg.qr(rng.normal(size=(X.shape[1], 2 * depths[-1])))
        a, b, c = probe_all(X - (X @ Q) @ Q.T)
        print(f"  {'random':>8s}  {a:7.3f}  {b:7.3f}  {c:8.3f}   <- control, "
              f"{2 * depths[-1]} random dims")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
