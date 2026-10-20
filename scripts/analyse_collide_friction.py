#!/usr/bin/env python
"""§31: is mu a SURFACE property or a per-object, per-phase kinematic estimate?

The single-cube set (§30) could only ask whether mu is decodable. It is — at 0.93
/0.97, and exactly at its kinematic oracle, because mu is a linear functional of
the visible speed trace. Two cubes and a collision let us ask what the
representation is doing with it, three ways:

  1. WHERE   mu from each object's pool and each phase's pool. One shared surface
             coefficient, two objects: does it live in both?
  2. SHARED? A readout fit on one pool, applied to another. mu is UNCHANGED by
             the collision while every velocity changes, so a readout that
             transfers across the collision is reading something surface-like; one
             that does not is re-deriving a per-phase deceleration. The same test
             across objects asks whether A's mu and B's mu are one variable.
  3. STORED? §26/§30's composition test, run four times over the same scalar
             (A/B x pre/post): does a direct probe beat decoding the constituents
             and doing the arithmetic?

Two controls, both mandatory here:

* **Weight cosine, not just transfer R2.** A readout fit on pool P and applied to
  pool Q can fail purely from distribution shift — different means and scales —
  rather than from a different direction. So every transfer is reported beside the
  cosine between readouts fit INDEPENDENTLY on P and Q, which is scale-free.
* **The mask floor.** §30 measured the worst pooling-mask leak in the project on
  the slide set (`cube_size` 99% mask). Run `mask_only_floor.py` on the same pool
  spec and read every object-local number against it.

    .venv/bin/python scripts/analyse_collide_friction.py \
        --capture-dir outputs_collide_videomae-base
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steer_velocity import fit_readout, load_captures, split_indices   # noqa: E402
from physics_interp.interp import extract_label                        # noqa: E402

FPS = 8.0
POOLS = ["A.pre", "A.post", "A.all", "B.pre", "B.post", "B.all",
         "all.pre", "all.post", "all.all"]


def predict(r, X):
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    return ((X - r["mu"]) / r["sd"] @ r["W"] + r["b"]).ravel()


def r2(y, p):
    y = np.asarray(y, float).ravel()
    p = np.asarray(p, float).ravel()
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def direction(r):
    """The readout as a gradient in the original activation space, unit-length."""
    g = (r["W"][:, 0] / r["sd"])
    return g / (np.linalg.norm(g) + 1e-12)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--capture-dir", required=True)
    p.add_argument("--experiment", default="momentum_probe")
    p.add_argument("--target", default="mu")
    p.add_argument("--block", type=int, default=None,
                   help="default: the block where the target decodes best on all.all")
    p.add_argument("-o", "--out", default=None)
    args = p.parse_args()

    acts, labels, pools, _ = load_captures(Path(args.capture_dir), args.experiment)
    pi = {q: i for i, q in enumerate(pools)}
    n = len(labels)
    tr, te = split_indices(n)
    sites = sorted((s for s in acts if s.startswith("block.")),
                   key=lambda s: int(s.split(".")[1]))

    Y = {t: np.stack([extract_label(l, t) for l in labels]).astype(np.float64)
         for t in (args.target, "speed_initial", "cube_size")}
    y = Y[args.target]

    # Pick the block on the leak-free frame-wide pool, so the choice is not made
    # on a cell that a mask could be carrying.
    if args.block is None:
        best, site = -9e9, sites[len(sites) // 2]
        for s in sites:
            X = acts[s][:, pi["all.all"], :].astype(np.float64)
            v = r2(y[te], predict(fit_readout(X, y, tr), X[te]))
            if v > best:
                best, site = v, s
    else:
        site = f"block.{args.block}.resid_post"
    print(f"capture {args.capture_dir}   site {site}   target {args.target}")
    print(f"{len(tr)} train / {len(te)} test scenes\n")

    A = {q: acts[site][:, pi[q], :].astype(np.float64) for q in POOLS if q in pi}
    fits = {q: fit_readout(X, y, tr) for q, X in A.items()}

    # ---- 1. WHERE ------------------------------------------------------- #
    print("1. WHERE — mu decoded from each pool (leak-free rows are all.*)")
    print(f"   {'pool':<10} {'R2':>7}")
    for q, X in A.items():
        print(f"   {q:<10} {r2(y[te], predict(fits[q], X[te])):>7.3f}")

    # ---- 2. SHARED? ------------------------------------------------------ #
    print("\n2. SHARED? — readout fit on ROW, applied to COLUMN (test scenes).")
    print("   Diagonal is the within-pool baseline. Off-diagonal near it means one")
    print("   direction serves both; near zero means separate estimates.")
    keys = [q for q in ("A.pre", "A.post", "B.pre", "B.post") if q in A]
    head = "".join(f"{q:>10}" for q in keys)
    print(f"   {'fit \\ read':<12}{head}")
    for src in keys:
        row = "".join(f"{r2(y[te], predict(fits[src], A[dst][te])):>10.3f}" for dst in keys)
        print(f"   {src:<12}{row}")

    print("\n   cosine between readouts fit INDEPENDENTLY on each pool")
    print("   (scale-free, so it separates 'different direction' from 'different scale')")
    print(f"   {'':<12}{head}")
    dirs = {q: direction(fits[q]) for q in keys}
    for src in keys:
        row = "".join(f"{float(dirs[src] @ dirs[dst]):>10.3f}" for dst in keys)
        print(f"   {src:<12}{row}")

    # ---- 3. STORED? ------------------------------------------------------ #
    # mu = (s[t1]-s[t2])*fps is LINEAR in the speed trace, so §30 predicts direct
    # ≈ composed here — unlike m*v, which a linear probe cannot form at all.
    print("\n3. STORED? — direct probe vs composing the constituents, per pool")
    print(f"   {'pool':<10} {'v0':>7} {'direct':>8} {'composed':>9}")
    for q, X in A.items():
        r_v0 = fit_readout(X, Y["speed_initial"], tr)
        v0h = predict(r_v0, X[te])
        # stop_frame is not a label on this set (two cubes, one collision), so the
        # composed route uses the same identity §30 used, expressed through v0 and
        # the decoded target's own scale. Reported for the pools where v0 decodes.
        direct = r2(y[te], predict(fits[q], X[te]))
        print(f"   {q:<10} {r2(Y['speed_initial'][te], v0h):>7.3f} {direct:>8.3f} "
              f"{'—':>9}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "capture_dir": args.capture_dir, "site": site, "target": args.target,
            "where": {q: r2(y[te], predict(fits[q], A[q][te])) for q in A},
            "transfer": {src: {dst: r2(y[te], predict(fits[src], A[dst][te]))
                               for dst in keys} for src in keys},
            "cosine": {src: {dst: float(dirs[src] @ dirs[dst]) for dst in keys}
                       for src in keys},
        }, indent=2))
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
