#!/usr/bin/env python
"""Does the pre-contact forecast commit to an ELASTIC or an INELASTIC outcome?

§43 established that post-contact velocity is decodable from a clip truncated
BEFORE contact (0.584 / 0.539 at k=20). That says the model anticipates *an*
outcome. It does not say WHICH -- and the outcome is a one-parameter family:

    v_A_post(e) = v_A - (1+e)*mu*u_n/m_A * n        u_n = (v_A - v_B).n
    v_B_post(e) = v_B + (1+e)*mu*u_n/m_B * n        mu  = m_A m_B/(m_A+m_B)

Both AFFINE in the restitution e. Verified against the rendered labels at 5.6e-17
on the conserving half, and the inversion below round-trips true velocities to
their true e at 1e-15.

WHY THIS IS NOT DONE PER SCENE
------------------------------
The e-dependent part of the target is small: the ENTIRE swing from perfectly
inelastic to perfectly elastic, |v_post(1) - v_post(0)|, is 30% of the
across-scene spread of v_post. A probe at R2 = 0.55 has residual error ~0.67 of
that spread -- about 2.2x the whole swing. Dividing a per-scene prediction error
by c = mu*u_n/m (5th percentile 0.057) then explodes it. Measured: per-scene
e_hat is flat at -0.06 with corr(e_hat, e_true) = 0.03 EVEN AT k=48, where the
aftermath is fully on screen. That is a failed positive control, i.e. a broken
estimator, not a finding about the model.

THE POOLED ESTIMATOR
--------------------
Along the contact normal the prediction implies, per scene and per object,

    a_A = v_A.n - p_A.n  =  (1 + e) * c_A ,     c_A = mu*u_n/m_A
    a_B = p_B.n - v_B.n  =  (1 + e) * c_B ,     c_B = mu*u_n/m_B

Stack both objects of all scenes and regress, WITHOUT dividing:

    a  =  b0 * c  +  b1 * (e_true * c)

  b1 ~ 1, b0 ~ 1   the forecast TRACKS the scene's true restitution
  b1 ~ 0, b0 ~ 1+d the forecast ignores it and commits to a FIXED default d
                   (d = 1 elastic, d = 0 perfectly inelastic)

Weighting by c is the point: scenes with a grazing approach carry almost no
restitution signal and are downweighted instead of amplified.

Ridge shrinkage attenuates b1 toward 0, so its ABSOLUTE value is not meaningful.
k=48 supplies the scale: it is the same estimator on the same scenes with the
aftermath visible, so b1(k) / b1(48) is the fraction of the recoverable
restitution signal present at truncation k.

CONTROLS
--------
  C2  POSITIVE CONTROL at k=48 -- aftermath on screen, b1 MUST be well above 0.
  C3  DIRECT PROBE for e itself at every k.
  C1  TRAINING-DISTRIBUTION SWAP -- fit on e < LO only, then on e > HI only, and
      predict the SAME held-out mid-band. A least-squares readout with no
      information about e reports its TRAINING mean, which would be a fact about
      our protocol, not the model. If the two agree, the default is the model's.

Targets are the RECONSTRUCTED aftermath (§43a) so all 150 scenes are usable: the
violation edits post-contact velocities only, leaving frames 0-20 honest physics.
Pool is `all.all` -- an all-ones mask, so the §20 pooling-mask leak is exactly
zero by construction, and it is also §43's best pool at k=20.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

ALPHAS = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4]


def ridge(Xtr, Ytr, a):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Z = (Xtr - mu) / sd
    xm, ym = Z.mean(0), Ytr.mean(0)
    Zc, Yc = Z - xm, Ytr - ym
    n, d = Zc.shape
    if n <= d:
        W = Zc.T @ np.linalg.solve(Zc @ Zc.T + a * np.eye(n), Yc)
    else:
        W = np.linalg.solve(Zc.T @ Zc + a * np.eye(d), Zc.T @ Yc)
    return lambda Xte: ((Xte - mu) / sd - xm) @ W + ym


def r2(Y, P):
    den = ((Y - Y.mean(0)) ** 2).sum()
    return float(1 - ((Y - P) ** 2).sum() / den) if den > 1e-12 else float("nan")


def _pick(A, Y, itr, iva):
    best = (-np.inf, 0, ALPHAS[0])
    for s in range(A.shape[1]):
        for a in ALPHAS:
            v = r2(Y[iva], ridge(A[itr, s], Y[itr], a)(A[iva, s]))
            if v > best[0]:
                best = (v, s, a)
    return best[1], best[2]


def oof_predict(A, Y, seed=0, n_fold=5):
    """Out-of-fold predictions; (site, alpha) chosen on an inner split of the
    TRAINING fold only -- never on the test fold."""
    n = Y.shape[0]
    P = np.zeros_like(Y)
    order = np.random.default_rng(seed).permutation(n)
    for f in range(n_fold):
        te = order[f::n_fold]
        tr = np.setdiff1d(order, te)
        cut = int(0.8 * len(tr))
        s, a = _pick(A, Y, tr[:cut], tr[cut:])
        P[te] = ridge(A[tr, s], Y[tr], a)(A[te, s])
    return P


def fit_predict_on(A, Y, tr, te, seed=0):
    rng = np.random.default_rng(seed)
    tr = rng.permutation(tr)
    cut = min(max(int(0.8 * len(tr)), 5), len(tr) - 5)
    s, a = _pick(A, Y, tr[:cut], tr[cut:])
    return ridge(A[tr, s], Y[tr], a)(A[te, s])


def scene_geometry(lab):
    c = lab["collision"]
    vA = np.asarray(c["pre"]["velocity_A"][:2], float)
    vB = np.asarray(c["pre"]["velocity_B"][:2], float)
    n = np.asarray(c["contact_normal"][:2], float)
    n = n / np.linalg.norm(n)
    if (vA - vB) @ n < 0:                     # §42b: orient n along the approach
        n = -n
    mA, mB = lab["objects"]["A"]["mass"], lab["objects"]["B"]["mass"]
    mu = mA * mB / (mA + mB)
    un = (vA - vB) @ n
    return dict(vA=vA, vB=vB, n=n, mA=mA, mB=mB, mu=mu, un=un,
                cA=mu * un / mA, cB=mu * un / mB, e=lab["restitution"])


def post(g, e, which):
    J = (1.0 + e) * g["mu"] * g["un"]
    return g["vA"] - (J / g["mA"]) * g["n"] if which == "A" else g["vB"] + (J / g["mB"]) * g["n"]


def stack_rows(G, PA, PB):
    """Per scene, two rows: (a, c, e*c) for object A and for object B."""
    a, c, ec, sid = [], [], [], []
    for i, (g, pa, pb) in enumerate(zip(G, PA, PB)):
        a += [g["vA"] @ g["n"] - pa @ g["n"], pb @ g["n"] - g["vB"] @ g["n"]]
        c += [g["cA"], g["cB"]]
        ec += [g["e"] * g["cA"], g["e"] * g["cB"]]
        sid += [i, i]
    return np.array(a), np.array(c), np.array(ec), np.array(sid)


def pooled_fit(a, c, ec):
    b, *_ = np.linalg.lstsq(np.stack([c, ec], 1), a, rcond=None)
    b0c, *_ = np.linalg.lstsq(c[:, None], a, rcond=None)   # constant-e model
    return float(b[0]), float(b[1]), float(b0c[0]) - 1.0


def boot(a, c, ec, sid, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(sid)
    out = []
    for _ in range(n):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([np.where(sid == p)[0] for p in pick])
        out.append(pooled_fit(a[idx], c[idx], ec[idx]))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="outputs_trunc_videomae-base_acts.npz")
    ap.add_argument("--pool", default="all.all")
    ap.add_argument("--lo", type=float, default=0.40)
    ap.add_argument("--hi", type=float, default=0.70)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    z = np.load(args.acts, allow_pickle=True)
    sites, pools = list(z["sites"]), list(z["pools"])
    labels = [json.loads(s) for s in z["labels"]]
    ks = sorted(int(k[1:]) for k in z.files if k.startswith("k"))
    pi = pools.index(args.pool)
    G = [scene_geometry(l) for l in labels]
    e_true = np.array([g["e"] for g in G])
    YA = np.stack([post(g, g["e"], "A") for g in G])
    YB = np.stack([post(g, g["e"], "B") for g in G])
    E = e_true[:, None]

    print(f"{args.acts}   n={len(G)} scenes, {len(sites)} sites, pool '{args.pool}'")
    print(f"true restitution: min {e_true.min():.3f}  median {np.median(e_true):.3f}  "
          f"max {e_true.max():.3f}  mean {e_true.mean():.3f}")
    print("contact at frame 20.03 -> k <= 20 is STRICTLY pre-contact\n")

    rep = {"acts": args.acts, "pool": args.pool, "ladder": ks,
           "e_true_mean": float(e_true.mean()), "rows": {}}
    hdr = (f"  {'k':>4} {'R2 vA_post':>10} {'R2 vB_post':>10} {'R2 probe e':>11} | "
           f"{'b1 (tracks e)':>22} {'b0':>6} {'implied default':>16}")
    print("=" * len(hdr)); print(hdr); print("=" * len(hdr))
    store = {}
    for k in ks:
        A = z[f"k{k}"][:, :, pi, :].astype(np.float64)
        PA, PB = oof_predict(A, YA), oof_predict(A, YB)
        PE = oof_predict(A, E)
        a, c, ec, sid = stack_rows(G, PA, PB)
        b0, b1, dflt = pooled_fit(a, c, ec)
        bs = boot(a, c, ec, sid, args.n_boot)
        lo1, hi1 = np.percentile(bs[:, 1], [2.5, 97.5])
        store[k] = (b1, lo1, hi1)
        mark = "  <- pre-contact" if k <= 20 else ""
        print(f"  {k:>4} {r2(YA, PA):>10.3f} {r2(YB, PB):>10.3f} {r2(E, PE):>11.3f} | "
              f"{f'{b1:+.3f} [{lo1:+.2f},{hi1:+.2f}]':>22} {b0:>6.3f} {dflt:>16.3f}{mark}")
        rep["rows"][k] = {"r2_vA_post": r2(YA, PA), "r2_vB_post": r2(YB, PB),
                          "r2_probe_restitution": r2(E, PE),
                          "b1_tracks_e": b1, "b1_ci95": [float(lo1), float(hi1)],
                          "b0": b0, "implied_default_restitution": dflt}

    b1_48 = store[ks[-1]][0]
    print(f"\n  b1 relative to the k={ks[-1]} positive control "
          f"(fraction of recoverable restitution signal present):")
    print("   " + "  ".join(f"k={k}:{store[k][0]/b1_48:+.2f}" for k in ks))
    rep["b1_relative_to_full_clip"] = {k: float(store[k][0] / b1_48) for k in ks}

    lo_i = np.where(e_true < args.lo)[0]
    hi_i = np.where(e_true > args.hi)[0]
    mid_i = np.where((e_true >= args.lo) & (e_true <= args.hi))[0]
    print(f"\nC1  TRAINING-DISTRIBUTION SWAP -- same held-out band both times")
    print(f"    train LOW  e < {args.lo}: n={len(lo_i)}, mean e {e_true[lo_i].mean():.3f}   |   "
          f"train HIGH e > {args.hi}: n={len(hi_i)}, mean e {e_true[hi_i].mean():.3f}")
    print(f"    test  MID  {args.lo}-{args.hi}: n={len(mid_i)}, mean e {e_true[mid_i].mean():.3f}")
    print(f"\n  {'k':>4} {'default | trained LOW':>22} {'default | trained HIGH':>23} {'gap':>7}   verdict")
    rep["C1"] = {"lo_mean": float(e_true[lo_i].mean()), "hi_mean": float(e_true[hi_i].mean()),
                 "mid_mean": float(e_true[mid_i].mean()),
                 "n": [len(lo_i), len(hi_i), len(mid_i)], "rows": {}}
    gm = [G[i] for i in mid_i]
    for k in ks:
        A = z[f"k{k}"][:, :, pi, :].astype(np.float64)
        dd = {}
        for tag, tr in (("lo", lo_i), ("hi", hi_i)):
            pa = fit_predict_on(A, YA, tr, mid_i)
            pb = fit_predict_on(A, YB, tr, mid_i)
            aa, cc, ee, _ = stack_rows(gm, pa, pb)
            dd[tag] = pooled_fit(aa, cc, ee)[2]
        gap = dd["hi"] - dd["lo"]
        verdict = "follows TRAINING set" if abs(gap) > 0.20 else "training-independent"
        print(f"  {k:>4} {dd['lo']:>22.3f} {dd['hi']:>23.3f} {gap:>7.3f}   {verdict}")
        rep["C1"]["rows"][k] = {"default_trained_low": dd["lo"],
                                "default_trained_high": dd["hi"], "gap": float(gap)}

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(rep, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
