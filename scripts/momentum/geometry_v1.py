#!/usr/bin/env python
"""Geometry of physical factors in the v1 conserving scenes.

The question is not "is X decodable" -- §1 answered that years of sections ago --
but how the factors are ARRANGED. Do speed and direction occupy the same
directions? Do the two objects? Is velocity literally the span of its polar
parts? Where in the activation's variance spectrum does physics live?

Everything runs on `datasets/probe_set` (v1, `legacy` arm, mass VISIBLE)
restricted to the 75 momentum-CONSERVING scenes. Violating scenes are excluded
because their aftermath was overwritten after the fact, so post-contact factors
there are not what the physics produced; the conserving half is the clean
kinematic sample.

CPU only. Reads cached activations; no model load.

--------------------------------------------------------------------------
THE MEASUREMENT, and why it is built this way
--------------------------------------------------------------------------
A principal angle between two fitted weight subspaces is uninterpretable on its
own. Three things corrupt it and each needs its own bracket:

  ceiling   Finite data. The SAME factor fit twice gives two different
            subspaces. Nothing can be more aligned than that, so the ceiling
            is the unit, not 0 deg.
  null      Ridge on meaningless labels does not return a random direction --
            it loads on the activation's dominant variance directions, which
            is exactly where real factors also load (the §18c `gen_shuffled`
            contamination). So the null is a SHUFFLED-LABEL fit, not a random
            subspace. Both are reported; the shuffled one is the honest bar.
  shared    Fitting two factors on the SAME scenes lets one noise realisation
            push both weight vectors the same way. Every cross-factor number
            here is therefore fit on DISJOINT HALVES of the scenes -- factor f
            on H1, factor g on H2 -- which removes that channel entirely and
            matches the ceiling's sample size at the same time.

Reported as `align` = (overlap - null) / (ceiling - null): 0 means as unrelated
as the estimator allows, 1 means the same subspace measured twice.

Alpha is PINNED everywhere (§30b: every INLP 'sawtooth' in this project turned
out to be per-round ridge-alpha selection). --alpha-sweep re-runs the whole
thing at three penalties to show the conclusions do not move.

Activations are stored fp16 and V-JEPA2's reach |x| ~ 264, so every load casts
to float64 before any norm (§18's silent-inf trap).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))  # noqa: E401,E702
import _momentum_path  # noqa: F401,E402

from physics_interp.config import load_config                       # noqa: E402
from physics_interp.interp import extract_label, ridge_fit          # noqa: E402
from steer_velocity import load_captures                            # noqa: E402


# --------------------------------------------------------------------------- #
# factors
# --------------------------------------------------------------------------- #
# pool each factor is READ from. Object-local for object properties, because
# §16 measured frame-wide pooling costing velocity 0.98 -> 0.55; the frame-wide
# pools get their own experiment (`objects`) rather than being mixed in here.
FACTORS = {
    "velocity_A": ("velocity_A_pre", "A.pre"),
    "speed_A":    ("speed_A_pre",    "A.pre"),
    "heading_A":  ("heading_A_pre",  "A.pre"),
    "velocity_B": ("velocity_B_pre", "B.pre"),
    "speed_B":    ("speed_B_pre",    "B.pre"),
    "heading_B":  ("heading_B_pre",  "B.pre"),
    "mass_A":     ("mass_A",         "A.all"),
    "radius_A":   ("radius_A",       "A.all"),
    "mass_B":     ("mass_B",         "B.all"),
    "radius_B":   ("radius_B",       "B.all"),
}
DEFAULT_BLOCK = {"videomae-base": 6, "vjepa2-vitl": 8}


# --------------------------------------------------------------------------- #
# ridge at a PINNED penalty, and the direction it implies
# --------------------------------------------------------------------------- #
def fit_dirs(X, Y, idx, alpha):
    """Weight directions in RAW activation space, orthonormalised. [d, k].

    The probe operates on standardised features, so its readout direction in
    the original activation basis -- the thing a steering vector would be, and
    the thing whose angle is physically meaningful -- is W / sd.
    """
    mu, sd = X[idx].mean(0), X[idx].std(0) + 1e-6
    W, _ = ridge_fit((X[idx] - mu) / sd, Y[idx], alpha)
    D = W / sd[:, None]
    Q, _ = np.linalg.qr(D)
    return Q


def fit_predict(X, Y, tr, te, alpha):
    """Held-out R2 at a pinned alpha."""
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    W, b = ridge_fit((X[tr] - mu) / sd, Y[tr], alpha)
    pred = (X[te] - mu) / sd @ W + b
    ss_res = ((Y[te] - pred) ** 2).sum()
    ss_tot = ((Y[te] - Y[tr].mean(0)) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)


def principal_angles(Q1, Q2):
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def overlap(Q1, Q2):
    """Fraction of the SMALLER subspace that survives projection onto the other.
    1 = contained, min(k1,k2)/d-ish = unrelated. Symmetric by construction."""
    k = min(Q1.shape[1], Q2.shape[1])
    return float((np.linalg.norm(Q1.T @ Q2, "fro") ** 2) / k)


def linear_dependence(Ya, Yb):
    """R2 of predicting Yb from Ya by OLS. ~1 => §34b: the comparison is vacuous."""
    A = np.hstack([Ya, np.ones((len(Ya), 1))])
    pred = A @ np.linalg.lstsq(A, Yb, rcond=None)[0]
    ss = ((Yb - pred) ** 2).sum(); st = ((Yb - Yb.mean(0)) ** 2).sum()
    return float(1.0 - ss / st)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load(capture_dir, cfg_name, moving_only=False):
    acts, labels, pools, ids = load_captures(Path(capture_dir), cfg_name)
    keep = [i for i, l in enumerate(labels) if l.get("momentum_conserved", True)]
    if moving_only:
        keep = [i for i in keep if labels[i].get("b_moving", True)]
    acts = {s: v[keep].astype(np.float64) for s, v in acts.items()}   # fp16 trap
    return acts, [labels[i] for i in keep], pools, [ids[i] for i in keep]


def halves(n, rng):
    o = rng.permutation(n)
    return o[: n // 2], o[n // 2: 2 * (n // 2)]


def shuffled(Y, idx, rng):
    """Y with the labels at `idx` permuted among themselves, full length kept so
    it can still be indexed by the same `idx`. The null has to be a fit on
    MEANINGLESS labels through the SAME covariance -- ridge on noise still loads
    on the activation's dominant directions, so a random subspace is the wrong
    null (§18c)."""
    Z = Y.copy()
    Z[idx] = Y[idx][rng.permutation(len(idx))]
    return Z


# --------------------------------------------------------------------------- #
# stage: algebra -- the §34b guard, run on the LABELS before any activation
# --------------------------------------------------------------------------- #
def stage_algebra(A, labels, pools, args, out):
    """Which factor pairs can be compared geometrically at all?

    Ridge is linear in Y, so if factor g is an exact linear function of factor f
    then w_g = M w_f identically and every subspace comparison between them
    returns 'contained' for ANY model, including a random one. That is §34b, and
    it silently voided three published results in this project.

    The audit is on labels only -- no activations, no model. It also quantifies
    the case that matters most here: heading = v/|v| is a QUOTIENT, so it is not
    forced into velocity's span -- but if speed happened to be nearly constant,
    heading would be near-linear in v anyway and the polar experiment would be
    measuring nothing. That has to be a number, not an assumption.
    """
    names = list(FACTORS)
    Y = {k: np.stack([extract_label(l, FACTORS[k][0]) for l in labels]) for k in names}
    print("R2( row  <-  column ), OLS on labels. 1.00 => geometric comparison is vacuous")
    print(f"{'':<11}" + "".join(f"{n:>11}" for n in names))
    M = {}
    for a in names:
        line = f"{a:<11}"
        for b in names:
            v = 1.0 if a == b else linear_dependence(Y[b], Y[a])
            M[f"{a}<-{b}"] = v
            line += f"{v:11.3f}"
        print(line)
    print("\nthe pairs that matter, spelled out:")
    for a, b in (("speed_A", "velocity_A"), ("heading_A", "velocity_A"),
                 ("speed_B", "velocity_B"), ("heading_B", "velocity_B"),
                 ("mass_A", "radius_A"), ("mass_B", "radius_B"),
                 ("velocity_B", "velocity_A")):
        v = M[f"{a}<-{b}"]
        verdict = ("VACUOUS - do not compare" if v > 0.95 else
                   "partly forced - read with care" if v > 0.6 else "free to compare")
        print(f"  R2({a:<11}<- {b:<11}) = {v:+.3f}   {verdict}")
    out["algebra"] = M
    return out


# --------------------------------------------------------------------------- #
# stage: angles -- the core geometry
# --------------------------------------------------------------------------- #
def stage_angles(A, labels, pools, args, out):
    """Pairwise subspace geometry with all three brackets.

    Every cross-factor overlap is fit on DISJOINT halves of the scenes, so a
    shared noise realisation cannot inflate it, and the sample size matches the
    ceiling's exactly.
    """
    pi = {p: i for i, p in enumerate(pools)}
    site = f"block.{args.block}.resid_post"
    names = [k for k in FACTORS if not (args.moving_only and False)]
    Y = {k: np.stack([extract_label(l, FACTORS[k][0]) for l in labels]) for k in names}
    X = {k: A[site][:, pi[FACTORS[k][1]], :] for k in names}
    n = len(labels)

    ov = {}          # (a,b) -> list over splits
    ceil = {k: [] for k in names}
    null = {}
    rng = np.random.default_rng(0)

    for rep in range(args.reps):
        h1, h2 = halves(n, rng)
        Q1 = {k: fit_dirs(X[k], Y[k], h1, args.alpha) for k in names}
        Q2 = {k: fit_dirs(X[k], Y[k], h2, args.alpha) for k in names}
        # shuffled-label twins: same X, same halves, labels permuted within half
        S1 = {k: fit_dirs(X[k], shuffled(Y[k], h1, rng), h1, args.alpha) for k in names}
        S2 = {k: fit_dirs(X[k], shuffled(Y[k], h2, rng), h2, args.alpha) for k in names}
        for k in names:
            ceil[k].append(overlap(Q1[k], Q2[k]))
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ov.setdefault((a, b), []).append(
                    0.5 * (overlap(Q1[a], Q2[b]) + overlap(Q2[a], Q1[b])))
                null.setdefault((a, b), []).append(
                    0.5 * (overlap(S1[a], S2[b]) + overlap(S2[a], S1[b])))

    d = X[names[0]].shape[1]
    R1, _ = np.linalg.qr(rng.normal(size=(d, 2)))
    R2, _ = np.linalg.qr(rng.normal(size=(d, 2)))
    print(f"site {site}   alpha {args.alpha:g}   n={n}   dim={d}   "
          f"{args.reps} disjoint-half repeats\n")
    print("split-half CEILING per factor (overlap of the same factor fit twice):")
    for k in names:
        c = np.mean(ceil[k])
        print(f"  {k:<11} {c:.3f}   ({np.degrees(np.arccos(np.sqrt(min(c,1)))):.0f} deg)")
    print(f"\n  random 2-d subspaces (geometric chance): {overlap(R1, R2):.4f}")

    print("\npairwise overlap.  align = (overlap - shuffled null) / (ceiling - null).")
    print("align ~0: unrelated.  ~1: the same subspace.  ceiling = sqrt(ceil_a * ceil_b)")
    print(f"{'pair':<26}{'overlap':>9}{'null':>8}{'ceil':>8}{'align':>8}{'minang':>8}")
    rows = {}
    for (a, b), v in sorted(ov.items(), key=lambda kv: -np.mean(kv[1])):
        o, nl = float(np.mean(v)), float(np.mean(null[(a, b)]))
        c = float(np.sqrt(np.mean(ceil[a]) * np.mean(ceil[b])))
        al = (o - nl) / (c - nl) if c - nl > 0.05 else np.nan  # ceiling must clear its own null
        ang = np.degrees(np.arccos(np.sqrt(min(max(o, 0), 1))))
        rows[f"{a}|{b}"] = dict(overlap=o, null=nl, ceiling=c, align=al, min_angle=ang)
        print(f"{a + ' | ' + b:<26}{o:9.3f}{nl:8.3f}{c:8.3f}{al:8.2f}{ang:8.1f}")
    out["angles"] = dict(site=site, alpha=args.alpha, n=n, dim=d,
                         ceiling={k: float(np.mean(v)) for k, v in ceil.items()},
                         random_chance=overlap(R1, R2), pairs=rows)
    return out


# --------------------------------------------------------------------------- #
# stage: objects -- are the two balls separable, and what does a frame-wide
#                   pool actually contain?
# --------------------------------------------------------------------------- #
def stage_objects(A, labels, pools, args, out):
    """Three sub-experiments, none of which decoding alone can answer.

    1. SUPERPOSITION. In the frame-wide pool `all.pre` the two objects' tokens
       are averaged together. If that average is a genuine superposition, the
       SUM v_A + v_B should be more accessible than either part -- a mean pool
       computes sums, not separations. If instead both parts read out
       individually, the objects are carried by different directions and pooling
       has not destroyed them. §16 established the parts get WEAKER frame-wide;
       it never asked whether the sum gets stronger.
    2. ANGLE, per pool. Where A and B are both present, how far apart are they?
    3. ABLATION. Project one object's velocity subspace out and re-probe the
       other, against a rank-matched random-subspace control. This is the
       functional version of the angle, and the one §18b ran on v3.
    """
    pi = {p: i for i, p in enumerate(pools)}
    site = f"block.{args.block}.resid_post"
    n = len(labels)
    vA = np.stack([extract_label(l, "velocity_A_pre") for l in labels])
    vB = np.stack([extract_label(l, "velocity_B_pre") for l in labels])
    targets = {"v_A": vA, "v_B": vB, "v_A+v_B": vA + vB, "v_A-v_B": vA - vB,
               "momentum_pre": np.stack([extract_label(l, "momentum_pre") for l in labels])}

    print("1. SUPERPOSITION -- what survives frame-wide pooling?")
    print("   held-out R2, mean over 20 splits, pinned alpha\n")
    print(f"{'pool':<10}" + "".join(f"{k:>14}" for k in targets))
    sup = {}
    for pool in ("A.pre", "B.pre", "all.pre", "all.all"):
        X = A[site][:, pi[pool], :]
        line = f"{pool:<10}"
        for k, Y in targets.items():
            r = []
            for s in range(20):
                rng = np.random.default_rng(s)
                o = rng.permutation(n); tr, te = o[:45], o[45:]
                r.append(fit_predict(X, Y, tr, te, args.alpha))
            sup[f"{pool}|{k}"] = float(np.mean(r))
            line += f"{np.mean(r):14.3f}"
        print(line)
    print("\n   read: if `v_A+v_B` beats both parts in `all.pre`, the frame-wide code is")
    print("   a superposition. If the parts hold up, the objects stay separable.")

    print("\n2. ANGLE between velocity_A and velocity_B, per pool")
    print(f"{'pool':<10}{'overlap':>9}{'null':>8}{'ceil':>8}{'align':>8}{'minang':>8}"
          f"{'R2 v_A':>9}{'R2 v_B':>9}")
    ang = {}
    for pool in ("A.pre", "B.pre", "all.pre", "all.all"):
        X = A[site][:, pi[pool], :]
        rng = np.random.default_rng(0)
        o_, nl_, ca_, cb_ = [], [], [], []
        for rep in range(args.reps):
            h1, h2 = halves(n, rng)
            QA1, QA2 = fit_dirs(X, vA, h1, args.alpha), fit_dirs(X, vA, h2, args.alpha)
            QB1, QB2 = fit_dirs(X, vB, h1, args.alpha), fit_dirs(X, vB, h2, args.alpha)
            o_.append(0.5 * (overlap(QA1, QB2) + overlap(QA2, QB1)))
            nl_.append(0.5 * (overlap(fit_dirs(X, shuffled(vA, h1, rng), h1, args.alpha),
                                      fit_dirs(X, shuffled(vB, h2, rng), h2, args.alpha))
                              + overlap(fit_dirs(X, shuffled(vA, h2, rng), h2, args.alpha),
                                        fit_dirs(X, shuffled(vB, h1, rng), h1, args.alpha))))
            ca_.append(overlap(QA1, QA2)); cb_.append(overlap(QB1, QB2))
        o, nl = float(np.mean(o_)), float(np.mean(nl_))
        c = float(np.sqrt(np.mean(ca_) * np.mean(cb_)))
        al = (o - nl) / (c - nl) if c - nl > 0.05 else np.nan  # ceiling must clear its own null
        rA = np.mean([fit_predict(X, vA, np.random.default_rng(s).permutation(n)[:45],
                                  np.random.default_rng(s).permutation(n)[45:], args.alpha)
                      for s in range(10)])
        rB = np.mean([fit_predict(X, vB, np.random.default_rng(s).permutation(n)[:45],
                                  np.random.default_rng(s).permutation(n)[45:], args.alpha)
                      for s in range(10)])
        ang[pool] = dict(overlap=o, null=nl, ceiling=c, align=al, r2_vA=float(rA), r2_vB=float(rB))
        print(f"{pool:<10}{o:9.3f}{nl:8.3f}{c:8.3f}{al:8.2f}"
              f"{np.degrees(np.arccos(np.sqrt(min(max(o,0),1)))):8.1f}{rA:9.3f}{rB:9.3f}")

    print("\n3. ABLATION -- strip one object's velocity subspace, re-probe the other.")
    print("   control is the same NUMBER of dimensions chosen at random, which is what")
    print("   separates 'removed the right directions' from 'removed capacity'.")
    abl = {}
    for pool in ("A.pre", "B.pre", "all.pre"):
        X0 = A[site][:, pi[pool], :]
        print(f"\n   pool {pool}")
        print(f"   {'removed':>18}{'R2 v_A':>9}{'R2 v_B':>9}{'R2 radius_A':>13}")
        rad = np.stack([extract_label(l, "radius_A") for l in labels])
        for label, owner in (("v_A subspace", vA), ("v_B subspace", vB), ("random", None)):
            for k in (4, 12):
                rng = np.random.default_rng(7)
                rows = []
                for s in range(8):
                    o = np.random.default_rng(s).permutation(n); tr, te = o[:45], o[45:]
                    if owner is None:
                        Q, _ = np.linalg.qr(rng.normal(size=(X0.shape[1], k)))
                    else:
                        Q = inlp_pinned(X0, owner, tr, k, args.alpha)
                    Xw = X0 - (X0 @ Q) @ Q.T
                    rows.append([fit_predict(Xw, vA, tr, te, args.alpha),
                                 fit_predict(Xw, vB, tr, te, args.alpha),
                                 fit_predict(Xw, rad, tr, te, args.alpha)])
                m = np.mean(rows, 0)
                abl[f"{pool}|{label}|{k}"] = m.tolist()
                print(f"   {label + f' ({k}d)':>18}{m[0]:9.3f}{m[1]:9.3f}{m[2]:13.3f}")
        rows = []
        for s in range(8):
            o = np.random.default_rng(s).permutation(n); tr, te = o[:45], o[45:]
            rows.append([fit_predict(X0, vA, tr, te, args.alpha),
                         fit_predict(X0, vB, tr, te, args.alpha),
                         fit_predict(X0, rad, tr, te, args.alpha)])
        m = np.mean(rows, 0)
        abl[f"{pool}|none|0"] = m.tolist()
        print(f"   {'nothing (baseline)':>18}{m[0]:9.3f}{m[1]:9.3f}{m[2]:13.3f}")
    out["objects"] = dict(superposition=sup, angles=ang, ablation=abl, site=site)
    return out


def inlp_pinned(X, Y, tr, k, alpha):
    """INLP at a FIXED penalty. §30b: selecting alpha per round is what produced
    every 'sawtooth' in this project, so the shape of an INLP curve is only
    readable when the penalty is held still."""
    Xw, cols, got = X.copy(), [], 0
    while got < k:
        Q = fit_dirs(Xw, Y, tr, alpha)
        cols.append(Q); got += Q.shape[1]
        Xw = Xw - (Xw @ Q) @ Q.T
    B = np.concatenate(cols, axis=1)[:, :k]
    Q, _ = np.linalg.qr(B)
    return Q


# --------------------------------------------------------------------------- #
# stage: polar -- is velocity stored as (v_x, v_y) or as (speed, direction)?
# --------------------------------------------------------------------------- #
def stage_polar(A, labels, pools, args, out):
    """Velocity has two degrees of freedom and two natural coordinate systems.
    This is NOT an identity trap: both parametrise the same information, so the
    question is purely which one the geometry is aligned with.

    Two independent instruments, because either alone is weak:

    (a) CONTAINMENT. Fit velocity (2-d), speed (1-d) and heading (2-d) subspaces.
        If the code is polar, span(speed, heading) should recover velocity's
        subspace, and speed should be nearly ORTHOGONAL to heading -- a polar
        code separates magnitude from angle. If it is Cartesian, speed and
        heading both smear across the same v_x/v_y directions and neither is
        clean.
    (b) AXIS STABILITY (§19's design, with its ceiling). Fit the speed direction
        separately inside each heading quadrant. Under a polar code the speed
        axis is the same axis regardless of which way the ball is going; under a
        Cartesian code it rotates with heading. Noise alone lowers agreement, so
        the comparison needs a within-bin ceiling at matched sample size.
        The mirror test -- heading axis across SPEED bins -- is run too; §19
        only ran one of the two directions.
    """
    pi = {p: i for i, p in enumerate(pools)}
    site = f"block.{args.block}.resid_post"
    res = {}
    for obj in ("A", "B"):
        pool = f"{obj}.pre"
        X = A[site][:, pi[pool], :]
        v = np.stack([extract_label(l, f"velocity_{obj}_pre") for l in labels])
        sp = np.stack([extract_label(l, f"speed_{obj}_pre") for l in labels])
        hd = np.stack([extract_label(l, f"heading_{obj}_pre") for l in labels])
        n = len(labels)
        print(f"\n--- object {obj}, pool {pool}, n={n} ---")
        print(f"  label check: R2(speed <- velocity, linear) = "
              f"{linear_dependence(v, sp):+.3f}   "
              f"R2(heading <- velocity) = {linear_dependence(v, hd):+.3f}")
        print("  (both must be well under 1 or the polar question is vacuous by algebra)")

        rng = np.random.default_rng(0)
        acc = {k: [] for k in ("sp_hd", "v_sphd", "v_sp", "v_hd",
                               "c_v", "c_sp", "c_hd", "null_sp_hd")}
        for rep in range(args.reps):
            h1, h2 = halves(n, rng)
            Qv1, Qv2 = fit_dirs(X, v, h1, args.alpha), fit_dirs(X, v, h2, args.alpha)
            Qs1, Qs2 = fit_dirs(X, sp, h1, args.alpha), fit_dirs(X, sp, h2, args.alpha)
            Qh1, Qh2 = fit_dirs(X, hd, h1, args.alpha), fit_dirs(X, hd, h2, args.alpha)
            acc["c_v"].append(overlap(Qv1, Qv2)); acc["c_sp"].append(overlap(Qs1, Qs2))
            acc["c_hd"].append(overlap(Qh1, Qh2))
            acc["sp_hd"].append(0.5 * (overlap(Qs1, Qh2) + overlap(Qs2, Qh1)))
            acc["v_sp"].append(0.5 * (overlap(Qv1, Qs2) + overlap(Qv2, Qs1)))
            acc["v_hd"].append(0.5 * (overlap(Qv1, Qh2) + overlap(Qv2, Qh1)))
            P2, _ = np.linalg.qr(np.hstack([Qs2, Qh2]))       # span(speed, heading)
            acc["v_sphd"].append(overlap(Qv1, P2))
            acc["null_sp_hd"].append(
                overlap(fit_dirs(X, shuffled(sp, h1, rng), h1, args.alpha),
                        fit_dirs(X, shuffled(hd, h2, rng), h2, args.alpha)))
        m = {k: float(np.mean(vv)) for k, vv in acc.items()}
        print(f"\n  (a) containment, {args.reps} disjoint-half repeats")
        print(f"      ceiling: velocity {m['c_v']:.3f}  speed {m['c_sp']:.3f}  "
              f"heading {m['c_hd']:.3f}")
        print(f"      speed  vs heading   overlap {m['sp_hd']:.3f}  "
              f"(null {m['null_sp_hd']:.3f}, ceil {np.sqrt(m['c_sp']*m['c_hd']):.3f})"
              f"  -> align {(m['sp_hd']-m['null_sp_hd'])/(np.sqrt(m['c_sp']*m['c_hd'])-m['null_sp_hd']):+.2f}")
        print(f"      velocity in span(speed,heading)  {m['v_sphd']:.3f}   "
              f"vs its own ceiling {m['c_v']:.3f}"
              f"  -> {m['v_sphd']/m['c_v']:.2f} of ceiling")
        print(f"      velocity vs speed alone {m['v_sp']:.3f}   "
              f"velocity vs heading alone {m['v_hd']:.3f}")

        # (b) axis stability
        def axis(Y, idx):
            return fit_dirs(X, Y, idx, args.alpha)[:, 0]
        def gen_axis(Y, idx):
            """Generative axis: OLS of activations ON the label (§18c). Ridge finds
            Sigma^-1 a; how the activation MOVES when the factor changes is a."""
            y = (Y[idx] - Y[idx].mean(0)) / (Y[idx].std(0) + 1e-12)
            Xc = X[idx] - X[idx].mean(0)
            g = (y[:, 0] @ Xc) / len(idx)
            return g / (np.linalg.norm(g) + 1e-12)

        ang = np.arctan2(v[:, 1], v[:, 0])
        qbin = np.digitize(ang, [-np.pi / 2, 0, np.pi / 2])
        speed_bin = (sp[:, 0] > np.median(sp[:, 0])).astype(int)
        stab = {}
        for tag, Y, binning in (("speed axis across HEADING bins", sp, qbin),
                                ("heading-x axis across SPEED bins", hd, speed_bin)):
            groups = [np.where(binning == b)[0] for b in np.unique(binning)]
            groups = [g for g in groups if len(g) >= 12]
            m_in, m_out = [], []
            rng2 = np.random.default_rng(1)
            for _ in range(args.reps):
                for gi, g in enumerate(groups):
                    o = rng2.permutation(len(g)); h = len(g) // 2
                    a1, a2 = gen_axis(Y, g[o[:h]]), gen_axis(Y, g[o[h:2 * h]])
                    m_in.append(abs(float(a1 @ a2)))                 # ceiling
                    for gj, g2 in enumerate(groups):
                        if gj <= gi:
                            continue
                        o2 = rng2.permutation(len(g2))[:h]
                        m_out.append(abs(float(a1 @ gen_axis(Y, g2[o2]))))
            ci, co = float(np.mean(m_in)), float(np.mean(m_out))
            stab[tag] = dict(within=ci, across=co, ratio=co / ci if ci else np.nan,
                             n_bins=len(groups))
            print(f"\n  (b) {tag}   ({len(groups)} bins)")
            print(f"      within-bin |cos| (ceiling) {ci:.3f}   across-bin {co:.3f}"
                  f"   ratio {co/ci:.2f}")
            print("      ratio -> 1 : axis is bin-invariant (polar-leaning). "
                  "-> 0 : it rotates (Cartesian-leaning)")
        res[obj] = dict(containment=m, stability=stab)
    out["polar"] = dict(site=site, alpha=args.alpha, per_object=res)
    return out


# --------------------------------------------------------------------------- #
# stage: spectrum -- WHERE in the activation's variance does physics sit?
# --------------------------------------------------------------------------- #
def stage_spectrum(A, labels, pools, args, out):
    """A factor direction can be anywhere in the activation's spectrum, and where
    it is has consequences nothing else here measures.

    If the factors sit in the top few PCs they are cheap to find, cheap to
    steer, and mutually crowded -- and any 'the factors are separate' result is
    at risk of being about the PCs rather than the physics. If they sit in the
    tail, the representation is spending its dominant variance on something
    else entirely (scene layout, texture) and physics is a low-energy code that
    mean-pooling and naive PCA-based methods will miss.

    Reported as the cumulative fraction of each direction's squared length
    captured by the top k PCs, against a random direction's expectation (k/d)
    and against the activation's own variance profile.
    """
    pi = {p: i for i, p in enumerate(pools)}
    site = f"block.{args.block}.resid_post"
    n = len(labels)
    ks = (1, 2, 5, 10, 20, 50)
    print(f"site {site}   n={n}\n")
    print(f"{'factor':<12}{'pool':<8}" + "".join(f"{'top'+str(k):>8}" for k in ks)
          + f"{'PR':>7}")
    spec = {}
    for name, (target, pool) in FACTORS.items():
        X = A[site][:, pi[pool], :]
        Xc = X - X.mean(0)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        Y = np.stack([extract_label(l, target) for l in labels])
        tr = np.arange(n)
        D = fit_dirs(X, Y, tr, args.alpha)[:, 0]
        c = (Vt @ D) ** 2
        cum = np.cumsum(c) / c.sum()
        pr = float((c.sum() ** 2) / (c ** 2).sum())     # participation ratio
        spec[name] = dict(cum={str(k): float(cum[min(k, len(cum)) - 1]) for k in ks},
                          participation_ratio=pr)
        print(f"{name:<12}{pool:<8}" + "".join(f"{cum[min(k,len(cum))-1]:8.3f}" for k in ks)
              + f"{pr:7.1f}")
    d = A[site].shape[-1]
    rng = np.random.default_rng(0)
    r = rng.normal(size=d); r /= np.linalg.norm(r)
    X = A[site][:, pi['A.pre'], :]
    _, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    c = (Vt @ r) ** 2; cum = np.cumsum(c) / c.sum()
    print(f"{'random dir':<12}{'A.pre':<8}" + "".join(f"{cum[min(k,len(cum))-1]:8.3f}" for k in ks)
          + f"{float((c.sum()**2)/(c**2).sum()):7.1f}")
    ev = S ** 2 / (S ** 2).sum(); cev = np.cumsum(ev)
    print(f"{'ACT variance':<12}{'A.pre':<8}" + "".join(f"{cev[min(k,len(cev))-1]:8.3f}" for k in ks))
    print("\n  'topk' = fraction of the factor direction's energy inside the top-k PCs.")
    print("  A factor far above the activation's own variance row is riding the")
    print("  dominant directions; far below it, the physics is in the tail.")
    print("  PR = participation ratio, the effective number of PCs it spans.")
    out["spectrum"] = spec
    return out


# --------------------------------------------------------------------------- #
# stage: dim -- how many dimensions does each factor need?
# --------------------------------------------------------------------------- #
def stage_dim(A, labels, pools, args, out):
    """INLP at a PINNED penalty: remove the factor's own directions one readout
    at a time and watch it die. The number of removals it survives is its
    redundancy, and it is the quantity that decides whether a rank-1 steering
    edit can work at all (§18b: velocity survived 12 rounds, which is why the
    single-direction steer was inert).

    Alpha is fixed because §30b showed every jump in this project's INLP curves
    tracked the per-round alpha selection rather than the representation.
    The control removes the same number of RANDOM directions.
    """
    pi = {p: i for i, p in enumerate(pools)}
    site = f"block.{args.block}.resid_post"
    n = len(labels)
    ks = (0, 2, 4, 8, 16, 32, 64)
    print(f"site {site}   alpha {args.alpha:g}   n={n}")
    print("held-out R2 after removing k of the factor's OWN directions "
          "(rnd = k random dims)\n")
    print(f"{'factor':<12}{'pool':<8}" + "".join(f"{'k='+str(k):>8}" for k in ks)
          + f"{'rnd64':>8}")
    dims = {}
    for name, (target, pool) in FACTORS.items():
        X0 = A[site][:, pi[pool], :]
        Y = np.stack([extract_label(l, target) for l in labels])
        row, rndv = [], []
        for k in ks:
            r = []
            for s in range(args.reps_dim):
                o = np.random.default_rng(s).permutation(n); tr, te = o[:45], o[45:]
                Xw = X0
                if k:
                    Xw = X0 - (X0 @ (Q := inlp_pinned(X0, Y, tr, k, args.alpha))) @ Q.T
                r.append(fit_predict(Xw, Y, tr, te, args.alpha))
            row.append(float(np.mean(r)))
        rng = np.random.default_rng(3)
        for s in range(args.reps_dim):
            o = np.random.default_rng(s).permutation(n); tr, te = o[:45], o[45:]
            Q, _ = np.linalg.qr(rng.normal(size=(X0.shape[1], 64)))
            rndv.append(fit_predict(X0 - (X0 @ Q) @ Q.T, Y, tr, te, args.alpha))
        dims[name] = dict(ks=list(ks), r2=row, random64=float(np.mean(rndv)))
        print(f"{name:<12}{pool:<8}" + "".join(f"{v:8.3f}" for v in row)
              + f"{np.mean(rndv):8.3f}")
    out["dim"] = dims
    return out


# --------------------------------------------------------------------------- #
# stage: transfer -- the functional test of the shared-code geometry
# --------------------------------------------------------------------------- #
ROLES = {
    "velocity": ("velocity_{o}_pre", "{o}.pre"),
    "speed":    ("speed_{o}_pre",    "{o}.pre"),
    "heading":  ("heading_{o}_pre",  "{o}.pre"),
    "mass":     ("mass_{o}",         "{o}.all"),
    "radius":   ("radius_{o}",       "{o}.all"),
}


def stage_transfer(A, labels, pools, args, out):
    """The `angles` stage says a factor's readout directions are the SAME
    whichever object they were fit on. That is a statement about two noisy
    estimates coinciding, and geometry alone cannot rule out a coincidence in a
    768-dimensional space. This converts it into a prediction.

    Fit the readout ONLY on object A -- A's pool, A's label -- then apply it,
    weights and intercept untouched, to object B's pool and score against B's
    label. Nothing about B was used to build it. If a shared, object-agnostic
    code exists, it transfers; if the coincidence was geometric, it cannot.

    Three controls, and the result means nothing without them:
      own      the same factor fit and tested on B. The ceiling.
      swapped  A's readout applied to B's pool, scored against A's OWN label. A
               code that is genuinely local to the pooled tokens must FAIL here,
               because B's tokens do not know A's velocity in that basis. If it
               succeeds instead, the readout is picking up something global and
               the transfer above proves nothing about locality.
      shuffled the same transfer with the fitting labels permuted.

    The frame-wide row is the mechanistic payoff. If the code is a per-token
    motion feature that pooling averages, then applying it to `all.pre` -- which
    averages BOTH objects' tokens -- should return a blend of the two
    velocities rather than either one. The mixing coefficients are read off by
    regressing the transferred prediction on the two true velocities, which
    turns 'frame-wide pooling loses the objects' (§16) into a statement about
    what it computes instead.
    """
    pi = {p: i for i, p in enumerate(pools)}
    site = f"block.{args.block}.resid_post"
    n = len(labels)

    def Y(t):
        return np.stack([extract_label(l, t) for l in labels])

    def fit_apply(Xf, Yf, Xa, Ya, tr, te, alpha):
        """Fit on (Xf, Yf) over tr; score the frozen readout on (Xa, Ya) over te."""
        mu, sd = Xf[tr].mean(0), Xf[tr].std(0) + 1e-6
        W, b = ridge_fit((Xf[tr] - mu) / sd, Yf[tr], alpha)
        pred = (Xa[te] - mu) / sd @ W + b
        ss = ((Ya[te] - pred) ** 2).sum()
        st = ((Ya[te] - Ya[tr].mean(0)) ** 2).sum()
        return float(1.0 - ss / st), pred

    print("held-out R2 of a readout fit on ONE object and applied to the OTHER.")
    print("`A->B` never saw B: not B's pool, not B's labels.\n")
    print(f"{'role':<10}{'A->B':>8}{'B->A':>8}{'own B':>8}{'own A':>8}"
          f"{'swapped':>9}{'shuffled':>10}")
    res = {}
    for role, (tf, pf) in ROLES.items():
        acc = {k: [] for k in ("AB", "BA", "ownB", "ownA", "swap", "shuf")}
        YA, YB = Y(tf.format(o="A")), Y(tf.format(o="B"))
        XA, XB = A[site][:, pi[pf.format(o="A")], :], A[site][:, pi[pf.format(o="B")], :]
        for s in range(args.reps_dim * 3):
            o = np.random.default_rng(s).permutation(n); tr, te = o[:45], o[45:]
            acc["AB"].append(fit_apply(XA, YA, XB, YB, tr, te, args.alpha)[0])
            acc["BA"].append(fit_apply(XB, YB, XA, YA, tr, te, args.alpha)[0])
            acc["ownB"].append(fit_apply(XB, YB, XB, YB, tr, te, args.alpha)[0])
            acc["ownA"].append(fit_apply(XA, YA, XA, YA, tr, te, args.alpha)[0])
            acc["swap"].append(fit_apply(XA, YA, XB, YA, tr, te, args.alpha)[0])
            rng = np.random.default_rng(1000 + s)
            acc["shuf"].append(fit_apply(XA, shuffled(YA, tr, rng), XB, YB,
                                         tr, te, args.alpha)[0])
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        res[role] = m
        print(f"{role:<10}{m['AB']:8.3f}{m['BA']:8.3f}{m['ownB']:8.3f}{m['ownA']:8.3f}"
              f"{m['swap']:9.3f}{m['shuf']:10.3f}")
    print("\n  `swapped` is the locality control: A's readout on B's tokens, scored")
    print("  against A's own label. It must FAIL for the transfer to mean the code")
    print("  is local to whatever was pooled.")

    # ---- what does the shared code return when both objects are pooled? ----- #
    print("\nFRAME-WIDE: apply the velocity readout (fit on A.pre only) to `all.pre`,")
    print("then regress its output on the two TRUE velocities. Coefficients say what")
    print("the frame-wide pool holds in the shared velocity basis.\n")
    vA, vB = Y("velocity_A_pre"), Y("velocity_B_pre")
    XA, Xall = A[site][:, pi["A.pre"], :], A[site][:, pi["all.pre"], :]
    coefs, r2s = [], {"vs v_A": [], "vs v_B": [], "vs mean": [], "vs sum": []}
    for s in range(args.reps_dim * 3):
        o = np.random.default_rng(s).permutation(n); tr, te = o[:45], o[45:]
        _, pred = fit_apply(XA, vA, Xall, vA, tr, te, args.alpha)
        M = np.hstack([vA[te], vB[te], np.ones((len(te), 1))])
        C = np.linalg.lstsq(M, pred, rcond=None)[0]
        coefs.append([C[0, 0], C[1, 1], C[2, 0], C[3, 1]])
        for tag, tgt in (("vs v_A", vA), ("vs v_B", vB), ("vs mean", 0.5 * (vA + vB)),
                         ("vs sum", vA + vB)):
            ss = ((tgt[te] - pred) ** 2).sum()
            st = ((tgt[te] - tgt[tr].mean(0)) ** 2).sum()
            r2s[tag].append(1.0 - ss / st)
    c = np.mean(coefs, 0)
    print(f"  pred_x = {c[0]:+.2f} * v_Ax  {c[2]:+.2f} * v_Bx      "
          f"pred_y = {c[1]:+.2f} * v_Ay  {c[3]:+.2f} * v_By")
    print("  held-out R2 of that same prediction against each candidate target:")
    for tag, v in r2s.items():
        print(f"     {tag:<10}{np.mean(v):8.3f}")
    out["transfer"] = dict(site=site, roles=res,
                           frame_wide_coefs=[float(x) for x in c],
                           frame_wide_r2={k: float(np.mean(v)) for k, v in r2s.items()})
    return out


# --------------------------------------------------------------------------- #
# stage: spatial -- how far does an object's code reach across the token grid?
# --------------------------------------------------------------------------- #
def stage_spatial(A, labels, pools, args, out):
    """`transfer` shows the velocity code is object-agnostic; `objects` shows a
    frame-wide pool returns the SUM. Both are consistent with a per-token motion
    feature that pooling averages -- but a pooled vector cannot prove that,
    because pooling has already destroyed the spatial structure.

    This reads the raw per-token activations (§34g's `outputs/patch_*` capture,
    which was never analysed) and asks the question directly.

    ONE readout is fit, on individual tokens belonging to ball A in pre-contact
    frames, with that scene's v_A as the label. It is then applied token by
    token to held-out scenes and scored in three places:

      A's own tokens   the ceiling.
      B's tokens       scored against v_B. The per-token version of the transfer
                       test: if a single local readout reports whichever object
                       sits under the token, the code is genuinely local and
                       shared, not two coincidentally-aligned global features.
      B's tokens vs v_A   the control that must fail.
      background rings  scored against v_A by grid distance from A. This is the
                       spatial extent -- the profile that says whether the code
                       stays on the object or is broadcast, and it is the number
                       the pooled analysis could only bound.

    The pooled experiment predicted a frame-wide attenuation of ~0.005 from
    A's 3-token footprint and measured 0.06-0.12, so the code must already
    extend well beyond the object by this depth. The profile measures by how
    much instead of inferring it.
    """
    from mask_only_floor import scene_mask_rows

    site = f"patch.block.{args.patch_block}.resid_post"
    pdir = Path(args.patch_dir) / "momentum_probe" / "activations"
    files = sorted(pdir.glob("scene_*.npz"))
    if not files:
        print(f"  no patch captures under {pdir} -- skipping"); return out

    keep_ids = set(out["scene_ids"])
    P, occ, vA, vB = [], [], [], []
    for f in files:
        if f.stem not in keep_ids:
            continue
        with np.load(f, allow_pickle=False) as z:
            if site not in z.files:
                print(f"  {site} not captured (have "
                      f"{[k for k in z.files if k.startswith('patch.')]}) -- skipping")
                return out
            meta = json.loads(bytes(z["meta"]).decode())
            P.append(z[site].astype(np.float64))
        lay = meta["metadata"]["layout"]
        # reuse the capture's OWN pooling geometry (mask_only_floor rebuilds it
        # from metadata, including the 480x720 -> 224x224 nearest resize the
        # adapter applies) so these token indices are exactly the ones pooled.
        _, raw, names = scene_mask_rows(Path(args.dataset_dir) / f.stem, meta)
        shape = (lay["frames"], lay["height"], lay["width"])
        occ.append(np.stack([raw[names.index("A.all")].reshape(shape),
                             raw[names.index("B.all")].reshape(shape)]))
        L = meta["labels"]
        vA.append(extract_label(L, "velocity_A_pre")); vB.append(extract_label(L, "velocity_B_pre"))
    P = np.stack(P); occ = np.stack(occ); vA = np.stack(vA); vB = np.stack(vB)
    n, T, H, W = len(P), occ.shape[2], occ.shape[3], occ.shape[4]
    pre_hi = meta["phases"]["pre"][1]
    print(f"{site}   n={n} scenes   grid {T}x{H}x{W}   dim={P.shape[-1]}   "
          f"pre frames 0..{pre_hi - 1}\n")

    # per-scene token sets and the distance of every token from each ball
    def sets(s):
        oA, oB = occ[s, 0, :pre_hi], occ[s, 1, :pre_hi]
        tokA = np.argwhere(oA > 0.2); tokB = np.argwhere(oB > 0.2)
        return tokA, tokB
    def flat(idx):
        return (idx[:, 0] * H + idx[:, 1]) * W + idx[:, 2]

    def centroid(o):
        m = o.sum((0,))
        if m.sum() <= 0:
            return None
        yy, xx = np.mgrid[0:H, 0:W]
        return np.array([(m * yy).sum() / m.sum(), (m * xx).sum() / m.sum()])

    rng = np.random.default_rng(0)
    accs = {k: [] for k in ("ownA", "B_vs_vB", "B_vs_vA", "shuf")}
    rings = {d: [] for d in range(0, 9)}
    for rep in range(args.reps_dim):
        o = rng.permutation(n); tr, te = o[:45], o[45:]
        Xs, Ys = [], []
        for s in tr:
            tA, _ = sets(s)
            if len(tA) == 0:
                continue
            Xs.append(P[s][flat(tA)]); Ys.append(np.repeat(vA[s][None], len(tA), 0))
        Xf, Yf = np.vstack(Xs), np.vstack(Ys)
        mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
        W_, b_ = ridge_fit((Xf - mu) / sd, Yf, args.alpha)
        Wsh, bsh = ridge_fit((Xf - mu) / sd, Yf[rng.permutation(len(Yf))], args.alpha)

        def score(pairs, W__, b__):
            if not pairs:
                return np.nan
            pr = np.vstack([p[0] for p in pairs]); tg = np.vstack([p[1] for p in pairs])
            pred = (pr - mu) / sd @ W__ + b__
            return float(1 - ((tg - pred) ** 2).sum() / ((tg - tg.mean(0)) ** 2).sum())

        pA, pBb, pBa, pSh = [], [], [], []
        ring_acc = {d: [] for d in rings}
        for s in te:
            tA, tB = sets(s)
            if len(tA):
                pA.append((P[s][flat(tA)], np.repeat(vA[s][None], len(tA), 0)))
                pSh.append((P[s][flat(tA)], np.repeat(vA[s][None], len(tA), 0)))
            if len(tB):
                pBb.append((P[s][flat(tB)], np.repeat(vB[s][None], len(tB), 0)))
                pBa.append((P[s][flat(tB)], np.repeat(vA[s][None], len(tB), 0)))
            cA = centroid(occ[s, 0, :pre_hi]); cB = centroid(occ[s, 1, :pre_hi])
            if cA is None:
                continue
            owned = set(map(tuple, tA.tolist())) | set(map(tuple, tB.tolist()))
            for t in range(pre_hi):
                for y in range(H):
                    for x in range(W):
                        if (t, y, x) in owned:
                            continue
                        dA = np.hypot(y - cA[0], x - cA[1])
                        if cB is not None and np.hypot(y - cB[0], x - cB[1]) < dA:
                            continue          # nearer B: not A's neighbourhood
                        d = int(min(round(dA), 8))
                        ring_acc[d].append(((t * H + y) * W + x, s))
        accs["ownA"].append(score(pA, W_, b_))
        accs["B_vs_vB"].append(score(pBb, W_, b_))
        accs["B_vs_vA"].append(score(pBa, W_, b_))
        accs["shuf"].append(score(pSh, Wsh, bsh))
        for d, lst in ring_acc.items():
            if len(lst) < 30:
                continue
            pr = np.stack([P[s][i] for i, s in lst]); tg = np.stack([vA[s] for _, s in lst])
            pred = (pr - mu) / sd @ W_ + b_
            rings[d].append(float(1 - ((tg - pred) ** 2).sum()
                                  / ((tg - tg.mean(0)) ** 2).sum()))

    print("one readout, fit on ball A's tokens only, applied token-by-token:")
    print(f"   A's tokens  vs v_A  (ceiling)          {np.nanmean(accs['ownA']):7.3f}")
    print(f"   B's tokens  vs v_B  (SHARED LOCAL CODE){np.nanmean(accs['B_vs_vB']):7.3f}")
    print(f"   B's tokens  vs v_A  (control, must fail){np.nanmean(accs['B_vs_vA']):6.3f}")
    print(f"   A's tokens, shuffled-label fit          {np.nanmean(accs['shuf']):7.3f}")
    print("\nspatial profile: R2 for v_A on BACKGROUND tokens, by grid distance from A")
    print("(tokens nearer to B are excluded, so this is A's own neighbourhood)\n")
    print(f"   {'dist':>5}{'R2':>9}{'n tok':>9}")
    prof = {}
    for d in sorted(rings):
        if not rings[d]:
            continue
        prof[d] = float(np.mean(rings[d]))
        print(f"   {d:>5}{np.mean(rings[d]):9.3f}")
    print("\n   one grid step = 16 px = 1/14 of the frame. A's own footprint is ~3 tokens.")
    out["spatial"] = dict(site=site, n=n, readouts={k: float(np.nanmean(v))
                                                    for k, v in accs.items()},
                          profile=prof)
    return out


STAGES = {"algebra": stage_algebra, "angles": stage_angles, "objects": stage_objects,
          "polar": stage_polar, "spectrum": stage_spectrum, "dim": stage_dim,
          "transfer": stage_transfer,
          "spatial": stage_spatial}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", nargs="?", default="configs/momentum_probe.yaml")
    p.add_argument("--model", default="videomae-base")
    p.add_argument("--capture-dir", default=None)
    p.add_argument("--stage", default="all",
                   help="comma-separated: " + ",".join(STAGES) + ",all")
    p.add_argument("--block", type=int, default=None)
    p.add_argument("--alpha", type=float, default=100.0, help="PINNED ridge penalty")
    p.add_argument("--reps", type=int, default=40, help="disjoint-half repeats")
    p.add_argument("--reps-dim", type=int, default=8)
    p.add_argument("--moving-only", action="store_true",
                   help="drop the 24 scenes where B starts at rest")
    p.add_argument("--patch-dir", default=None, help="raw per-token capture (§34g)")
    p.add_argument("--patch-block", type=int, default=5)
    p.add_argument("--dataset-dir", default="datasets/probe_set")
    p.add_argument("-o", "--out", default=None)
    args = p.parse_args()

    if args.capture_dir is None:
        args.capture_dir = f"outputs/{args.model}"
    if args.patch_dir is None:
        args.patch_dir = f"outputs/patch_{args.model}"
    if args.block is None:
        args.block = DEFAULT_BLOCK.get(args.model, 6)
    cfg = load_config(args.config)
    A, labels, pools, ids = load(args.capture_dir, cfg.name, args.moving_only)

    tag = "moving-only (B in motion)" if args.moving_only else "all conserving"
    print("=" * 78)
    print(f"GEOMETRY OF PHYSICAL FACTORS -- {args.model}, v1 probe_set, {tag}")
    print(f"n = {len(labels)} scenes   block {args.block}   alpha {args.alpha:g}")
    print("=" * 78)

    stages = list(STAGES) if args.stage == "all" else args.stage.split(",")
    dest = Path(args.out) if args.out else Path(
        f"{args.capture_dir}/{cfg.name}/geometry_v1"
        + ("_moving" if args.moving_only else "") + ".json")
    # merge, so running stages in separate invocations accumulates one record
    out = json.loads(dest.read_text()) if dest.exists() else {}
    out.update({"model": args.model, "block": args.block, "alpha": args.alpha,
                "n_scenes": len(labels), "moving_only": args.moving_only,
                "scene_ids": ids})
    for s in stages:
        print(f"\n{'=' * 78}\n== {s.upper()}\n{'=' * 78}")
        out = STAGES[s](A, labels, pools, args, out)

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, default=float))
    print(f"\n-> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
