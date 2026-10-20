"""Is kinetic energy STORED, or composed at readout time? Two tests, one question.

§44 ended on a structural wall: for a fully observed two-body collision, momentum
conservation gives `m_A/m_B = |dv_B|/|dv_A|`, so the energy ratio is a mass-free
function of the visible deflections (R2 = 1.000000) and no probe on those captures
can separate an energy concept from kinematics.

`slide_set_v2` does not have that wall. One cube slides onto a friction floor; the
momentum goes into the floor, whose recoil is unobservable, so nothing hands the
observer a mass. And the factors are clean:

    mass = size^3 EXACTLY (density pinned at 1.0)
    corr(log size, log speed) = -0.053          -- independent by design
    log KE from speed alone R2 0.207, from size alone 0.749   -- neither is a shortcut

So `KE = 1/2 m v^2` there is a genuine product of two independent visible factors
with no identity bypass. Two tests follow.

TEST B -- COMPOSITION, within the slide set. Does a linear readout give KE better
than decoding mass and speed separately and multiplying? §26's criterion, and here
it has teeth. Direct >= composed means energy is held as a unit.

    THE TRAP THIS AVOIDS, and it would have faked a positive. Everything is done in
    LINEAR space. In log space `log KE = log(1/2) + log m + 2 log v` is a LINEAR
    combination, and ridge is linear in Y, so `w_KE = w_m + 2 w_v` to 1e-15 and
    direct-vs-composed stops being a measurement and becomes an algebraic identity.
    That is §34b, which already cost this project the mass-ratio result once.

TEST A -- CROSS-SCENARIO TRANSFER. Fit the readout on `slide_set_v2` (cubes, side
camera, energy going to friction heat) and apply it UNCHANGED to
`conservation_set_v1` (spheres, different camera, energy going to deformation). The
kinematic shortcut differs completely between the two -- deceleration x distance
versus `(1-e^2)` -- so a single direction that reads KE in both is reading energy
rather than a scenario-specific trick.

    Scored by SPEARMAN, not R2. The two domains put KE on different scales and a
    rank statistic is invariant to that; raw R2 is reported beside it but would
    punish an affine mismatch that says nothing about whether the quantity is there.

    THE CELL IS PICKED ON THE SOURCE SET ONLY, by grouped CV within the slide
    captures, so the target set contributes nothing to selection and there is no
    winner's curse to correct.

    SPEED AND APPARENT SIZE TRANSFER FIRST, as controls. Cubes-vs-spheres and a
    different camera are a real domain gap; if those two do not cross it then a KE
    failure is uninformative and the test is void rather than negative. Apparent
    size is the SEGMENTATION PIXEL COUNT, which is the same visible quantity in both
    domains -- a cube's edge length and a sphere's radius are not.

    `B.*` POOLS ARE EXCLUDED from the transfer. In the slide set object B is the
    FLOOR; in the collision set it is the second sphere. The pool name matches and
    the content does not.

    .venv/bin/python scripts/energy_composition.py --test B
    .venv/bin/python scripts/energy_composition.py --test A
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
ALL_POOLS = ["all.pre", "all.post", "all.all", "A.pre", "A.post", "A.all",
             "B.pre", "B.post", "B.all"]
TRANSFER_POOLS = ["A.pre", "A.post", "A.all", "all.pre", "all.post", "all.all"]


# --------------------------------------------------------------------------- #
# ridge, shared with probe_energy.py's conventions
# --------------------------------------------------------------------------- #

def r2(y, p):
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean(0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def spearman(a, b):
    def rank(x):
        o = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), float)
        xs = x[o]
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and xs[j + 1] == xs[i]:
                j += 1
            r[o[i:j + 1]] = (i + j) / 2.0 + 1
            i = j + 1
        return r
    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


class Ridge:
    """Standardise on TRAIN, dual solve, dead features zeroed (see probe_energy)."""

    def __init__(self, Xtr, Xte):
        Xtr = np.asarray(Xtr, np.float64); Xte = np.asarray(Xte, np.float64)
        mu, sd = Xtr.mean(0), Xtr.std(0)
        dead = sd <= 1e-9 * (np.abs(Xtr).mean(0) + 1e-30)
        scale = np.where(dead, 1.0, sd + 1e-6)
        Ztr, Zte = (Xtr - mu) / scale, (Xte - mu) / scale
        Ztr[:, dead] = 0.0; Zte[:, dead] = 0.0
        xm = Ztr.mean(0); Zc = Ztr - xm
        w, V = np.linalg.eigh(Zc @ Zc.T)
        self.w, self.V = w, V
        self.Kte = ((Zte - xm) @ Zc.T) @ V

    def predict(self, Y):
        ym = Y.mean(0)
        A = self.V.T @ (Y - ym)
        return {a: self.Kte @ (A / (self.w + a)[:, None]) + ym for a in ALPHAS}


def folds(n, k, seed):
    rng = np.random.default_rng(seed)
    return np.array_split(rng.permutation(n), k)


def oof(X, y, seed=0, n_out=5, n_in=3):
    """Out-of-fold predictions, alpha chosen inside each training half."""
    out = np.zeros_like(y)
    fo = folds(len(y), n_out, seed)
    for k in range(n_out):
        te = fo[k]
        tr = np.concatenate([fo[j] for j in range(n_out) if j != k])
        fi = folds(len(tr), n_in, seed + 100 + k)
        score = {a: 0.0 for a in ALPHAS}
        for f in range(n_in):
            iva = tr[fi[f]]
            itr = np.concatenate([tr[fi[j]] for j in range(n_in) if j != f])
            for a, P in Ridge(X[itr], X[iva]).predict(y[itr]).items():
                score[a] += r2(y[iva], P)
        best = max(ALPHAS, key=lambda a: score[a])
        out[te] = Ridge(X[tr], X[te]).predict(y[tr])[best]
    return out


def source_cv_r2(X, y, seed=0, n_out=5):
    """Grouped CV R2 within the source set, used ONLY to pick the transfer cell."""
    return r2(y, oof(X, y, seed, n_out))


def fit_all_predict(Xs, ys, Xt, alpha):
    return Ridge(Xs, Xt).predict(ys)[alpha]


def pick_alpha(X, y, seed=0, n_out=5):
    fo = folds(len(y), n_out, seed)
    score = {a: 0.0 for a in ALPHAS}
    for k in range(n_out):
        te = fo[k]
        tr = np.concatenate([fo[j] for j in range(n_out) if j != k])
        for a, P in Ridge(X[tr], X[te]).predict(y[tr]).items():
            score[a] += r2(y[te], P)
    return max(ALPHAS, key=lambda a: score[a])


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def load(outdir):
    paths = sorted(glob.glob(f"{outdir}/momentum_probe/activations/*.npz"))
    if not paths:
        raise FileNotFoundError(outdir)
    store, labels, pools, sites, meta = {}, [], None, None, None
    for p in paths:
        z = np.load(p, allow_pickle=True)
        m = json.loads(bytes(z["meta"]).decode())
        meta = meta or m
        pools = m["pools"]; labels.append(m["labels"])
        sites = sites or [k[4:] for k in z.files if k.startswith("act.")]
        for s in sites:
            store.setdefault(s, []).append(z["act." + s][0])
    return {s: np.stack(v).astype(np.float32) for s, v in store.items()}, labels, pools, meta


def pre_window(meta):
    """The source frames the capture's `pre` pool actually averaged over."""
    idx = meta["frame_indices"]
    tub = len(idx) // meta["metadata"]["layout"]["frames"]
    return idx[: meta["phases"]["pre"][1] * tub]


def slide_targets(labels, meta):
    """Object A (the cube); object B is the FLOOR. mass = size^3, density pinned to 1.

    TWO READ CONVENTIONS, because on this set they are not the same test and picking
    one silently would be a choice worth hiding. The cube DECELERATES through the pre
    window, so where the target is read decides how much of it is mass:

      `ke`       at the FIRST pre frame. log-KE variance: 0.653 from mass, 0.218 from
                 speed. Mass-dominated, so a speed-only readout is capped near 0.22 --
                 the sharper test of whether the representation uses a mass at all.
      `ke_mean`  averaged over the whole pre window, which is what a temporally pooled
                 vector most plausibly encodes. 0.334 from mass, 0.617 from speed --
                 more faithful to the pooling, weaker as a test.

    Both are reported. Neither has a zero-speed clip; reading at the pre window's END
    instead would put 70 of 300 cubes at rest and collapse the target.
    """
    win = pre_window(meta)
    m, v, vm, px, kem = [], [], [], [], []
    for L in labels:
        A = L["objects"]["A"]
        pf = {p["frame"]: p for p in L["per_frame"]}
        sp = [pf[f]["speed"] for f in win if f in pf]
        cnt = L["segmentation_stats"]["A"]["count"]
        m.append(A["mass"]); v.append(sp[0]); vm.append(float(np.mean(sp)))
        kem.append(float(np.mean([0.5 * A["mass"] * s ** 2 for s in sp])))
        px.append(np.mean([cnt[min(f, len(cnt) - 1)] for f in win]))
    m, v, vm, px = (np.array(x, float) for x in (m, v, vm, px))
    return {"mass": m, "speed": v, "speed_mean": vm, "pixels": px,
            "ke": 0.5 * m * v ** 2, "ke_mean": np.array(kem)}


def collision_targets(labels, meta):
    """Sphere A. No gravity and no friction before contact, so its speed is CONSTANT
    across the whole pre window -- the two slide read conventions collapse to one here,
    which is why the comparison is well posed at all."""
    win = pre_window(meta)
    m, v, px = [], [], []
    for L in labels:
        A = L["objects"]["A"]
        c = L["collision"]
        pf = {p["frame"]: p for p in L["per_frame"]}
        m.append(A["mass"])
        v.append(np.linalg.norm(np.array(c["pre"]["velocity_A"][:2], float)))
        px.append(np.mean([pf[f]["A"]["segmentation_pixel_count"] for f in win if f in pf]))
    m, v, px = np.array(m), np.array(v), np.array(px, float)
    return {"mass": m, "speed": v, "speed_mean": v, "pixels": px,
            "ke": 0.5 * m * v ** 2, "ke_mean": 0.5 * m * v ** 2}


def cells_of(acts, pools, want):
    return {f"{s}/{p}": acts[s][:, pools.index(p), :] for s in acts for p in want}


# --------------------------------------------------------------------------- #
# TEST B -- composition within the slide set
# --------------------------------------------------------------------------- #

def test_b(outdir, short, args, conv="ke"):
    acts, labels, pools, meta = load(outdir)
    T = slide_targets(labels, meta)
    cells = cells_of(acts, pools, ALL_POOLS)
    ke = T[conv].reshape(-1, 1)
    mass = T["mass"].reshape(-1, 1)
    spd = T["speed" if conv == "ke" else "speed_mean"].reshape(-1, 1)
    n = len(ke)
    print(f"\n{'=' * 86}\nTEST B — COMPOSITION   {short}   target `{conv}`   n={n}, {len(cells)} cells")
    print(f"  KE = ½·mass·speed²  in LINEAR space (log space would make this an identity, §34b)")
    print(f"  KE {ke.min():.4f}–{ke.max():.4f}   mass {mass.min():.3f}–{mass.max():.3f}   "
          f"speed {spd.min():.3f}–{spd.max():.3f}")

    rows, preds = {}, {}
    for cname, X in cells.items():
        p_ke = oof(X, ke, args.seed)
        p_m = np.clip(oof(X, mass, args.seed), 1e-6, None)   # mass and speed are
        p_v = np.clip(oof(X, spd, args.seed), 0.0, None)     # physically non-negative
        composed = 0.5 * p_m * p_v ** 2
        # affine calibration of the composed value, fitted on TRAINING folds only, so
        # the comparison is about SHAPE and not about the product's arbitrary scale
        cal = np.zeros_like(composed)
        fo = folds(n, 5, args.seed)
        for k in range(5):
            te = fo[k]; tr = np.concatenate([fo[j] for j in range(5) if j != k])
            Amat = np.hstack([composed[tr], np.ones((len(tr), 1))])
            coef, *_ = np.linalg.lstsq(Amat, ke[tr], rcond=None)
            cal[te] = np.hstack([composed[te], np.ones((len(te), 1))]) @ coef
        rows[cname] = {
            "direct": r2(ke, p_ke), "composed_raw": r2(ke, composed),
            "composed_cal": r2(ke, cal), "mass": r2(mass, p_m), "speed": r2(spd, p_v),
            "direct_rho": spearman(ke.ravel(), p_ke.ravel()),
            "composed_rho": spearman(ke.ravel(), composed.ravel()),
        }
        preds[cname] = (p_ke, cal)

    bd = max(rows, key=lambda c: rows[c]["direct"])
    bc = max(rows, key=lambda c: rows[c]["composed_cal"])
    print(f"\n  best DIRECT cell    {bd}")
    for k in ("direct", "composed_cal", "composed_raw", "mass", "speed"):
        print(f"      {k:14s} R2 {rows[bd][k]:+.3f}")
    print(f"  best COMPOSED cell  {bc}")
    for k in ("direct", "composed_cal", "composed_raw", "mass", "speed"):
        print(f"      {k:14s} R2 {rows[bc][k]:+.3f}")
    d, c = rows[bd]["direct"], rows[bc]["composed_cal"]
    # BOOTSTRAP THE GAP. On this set both routes land at 0.95-0.99 and the raw gap
    # runs a few thousandths, which is small enough that its SIGN flips between the
    # two read conventions. Printing a verdict off that sign would be inventing a
    # result, so the verdict is gated on an interval that excludes zero.
    pk = preds[bd][0]; cl = preds[bc][1]
    rng = np.random.default_rng(args.seed)
    gaps = []
    for _ in range(args.n_boot):
        i = rng.integers(0, n, n)
        gaps.append(r2(ke[i], pk[i]) - r2(ke[i], cl[i]))
    g = np.array(gaps)
    lo, hi = np.percentile(g, 2.5), np.percentile(g, 97.5)
    print(f"\n  BEST-vs-BEST   direct {d:+.3f}   composed {c:+.3f}   gap {d - c:+.3f}"
          f"   95% CI [{lo:+.3f}, {hi:+.3f}]")
    if lo > 0:
        verdict = "ENERGY IS HELD AS A UNIT"
    elif hi < 0:
        verdict = "energy is COMPOSED at readout, not stored"
    else:
        verdict = "INCONCLUSIVE — the gap is not distinguishable from zero"
    print(f"  → {verdict}")
    if min(d, c) > 0.95:
        print(f"  ⚠  CEILING. Both routes clear 0.95 because the factors decode at "
              f"mass {rows[bd]['mass']:+.3f} / speed {rows[bd]['speed']:+.3f} — near\n"
              f"     perfect, so composition is near perfect too and the comparison has "
              f"little room to resolve.\n     §26's momentum version had teeth because "
              f"its factors sat at 0.73-0.97 and the product FELL to 0.43-0.66.")
    print(f"\n  per-pool (best site), direct / composed_cal:")
    for p in ALL_POOLS:
        sub = [c2 for c2 in rows if c2.endswith("/" + p)]
        j = max(sub, key=lambda c2: rows[c2]["direct"])
        k = max(sub, key=lambda c2: rows[c2]["composed_cal"])
        print(f"      {p:9s} {rows[j]['direct']:+.3f} ({j.split('/')[0]})   "
              f"{rows[k]['composed_cal']:+.3f} ({k.split('/')[0]})")
    return {"cells": rows, "best_direct": bd, "best_composed": bc,
            "direct": d, "composed": c, "gap": d - c,
            "gap_ci": [float(lo), float(hi)], "verdict": verdict}


# --------------------------------------------------------------------------- #
# TEST A -- cross-scenario transfer
# --------------------------------------------------------------------------- #

def test_a(src_dir, tgt_dir, short, args):
    sa, sl, sp, sm = load(src_dir)
    ta, tl, tp, tm = load(tgt_dir)
    S, T = slide_targets(sl, sm), collision_targets(tl, tm)
    scells = cells_of(sa, sp, TRANSFER_POOLS)
    tcells = cells_of(ta, tp, TRANSFER_POOLS)
    common = [c for c in scells if c in tcells]
    print(f"\n{'=' * 86}\nTEST A — TRANSFER   {short}")
    print(f"  fit on {src_dir} (n={len(sl)}), predict {tgt_dir} (n={len(tl)}), "
          f"{len(common)} shared cells")
    print(f"  B.* pools excluded: object B is the FLOOR in the slide set and the "
          f"second sphere in the collision set")

    rng = np.random.default_rng(args.seed)
    out = {}
    order = [("speed", "CONTROL"), ("pixels", "CONTROL"), ("ke", "THE TEST")]
    for tname, kind in order:
        ys = S[tname].reshape(-1, 1)
        yt = T[tname]
        # cell + alpha chosen on the SOURCE only
        scores = {c: source_cv_r2(scells[c], ys, args.seed) for c in common}
        best = max(scores, key=lambda c: scores[c])
        alpha = pick_alpha(scells[best], ys, args.seed)
        pred = fit_all_predict(scells[best], ys, tcells[best], alpha).ravel()
        rho = spearman(yt, pred)
        null = np.array([spearman(rng.permutation(yt), pred) for _ in range(args.n_perm)])
        p = float((np.abs(null) >= abs(rho)).mean())
        print(f"\n  [{kind}] {tname}")
        print(f"      source-set CV R2 at chosen cell   {scores[best]:+.3f}   {best}")
        print(f"      TRANSFERRED to target: Spearman ρ {rho:+.3f}   "
              f"(raw R2 {r2(yt.reshape(-1,1), pred.reshape(-1,1)):+.3f})   "
              f"null |ρ| p95 {np.percentile(np.abs(null),95):.3f}   p = {p:.3f}")
        out[tname] = {"cell": best, "source_cv_r2": scores[best], "rho": rho,
                      "p": p, "null_p95": float(np.percentile(np.abs(null), 95))}

    # the composed route, transferred the same way
    if all(k in out for k in ("pixels", "speed", "ke")):
        cm, cv = out["pixels"]["cell"], out["speed"]["cell"]
        am = pick_alpha(scells[cm], S["pixels"].reshape(-1, 1), args.seed)
        av = pick_alpha(scells[cv], S["speed"].reshape(-1, 1), args.seed)
        pm = np.clip(fit_all_predict(scells[cm], S["pixels"].reshape(-1, 1), tcells[cm], am).ravel(), 1e-9, None)
        pv = np.clip(fit_all_predict(scells[cv], S["speed"].reshape(-1, 1), tcells[cv], av).ravel(), 0.0, None)
        comp = pm * pv ** 2      # apparent size stands in for mass, up to a monotone map
        rho_c = spearman(T["ke"], comp)
        print(f"\n  COMPOSED transfer (apparent size × speed², both transferred "
              f"separately): ρ {rho_c:+.3f}")
        print(f"  direct ρ {out['ke']['rho']:+.3f}  vs  composed ρ {rho_c:+.3f}   "
              f"gap {out['ke']['rho'] - rho_c:+.3f}")
        out["composed_rho"] = rho_c

    if abs(out["speed"]["rho"]) < 0.2 and abs(out["pixels"]["rho"]) < 0.2:
        print("\n  ⚠  BOTH CONTROLS FAILED TO CROSS THE DOMAIN GAP. The KE number below "
              "is VOID,\n     not negative — nothing transfers between these two renders.")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", choices=["A", "B", "both"], default="both")
    ap.add_argument("--slide", nargs=2,
                    default=["outputs_slide_videomae-base", "outputs_slide_vjepa2-vitl"])
    ap.add_argument("--collision", nargs=2,
                    default=["outputs_cons_videomae-base", "outputs_cons_vjepa2-vitl"])
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", default="docs/artifacts/energy_composition.json")
    args = ap.parse_args()

    report = {}
    names = ["VideoMAE", "V-JEPA2"]
    if args.test in ("B", "both"):
        for sd, nm in zip(args.slide, names):
            got = {}
            for conv in ("ke", "ke_mean"):
                got[conv] = test_b(sd, nm, args, conv)
                report.setdefault("test_B", {})[f"{sd}/{conv}"] = got[conv]
            # THE GUARD THAT MATTERS, and a CI is not it. Both read conventions gave
            # intervals EXCLUDING zero on VideoMAE -- [-0.015,-0.006] and
            # [+0.000,+0.008] -- i.e. two significant answers pointing OPPOSITE ways,
            # from the same captures, differing only in where inside the pre window
            # the target was read. Significance at the ceiling certifies precision,
            # not robustness. The claim only survives if the sign does.
            g1, g2 = got["ke"]["gap"], got["ke_mean"]["gap"]
            agree = (g1 > 0) == (g2 > 0)
            print(f"\n  {'=' * 78}\n  ROBUSTNESS across read conventions   "
                  f"ke {g1:+.3f}   ke_mean {g2:+.3f}")
            if not agree:
                print("  ✗ THE SIGN FLIPS. Both intervals exclude zero and they disagree, so\n"
                      "    this set CANNOT answer the composition question: the answer is set\n"
                      "    by an arbitrary analysis choice, not by the representation.\n"
                      "    Report as INCONCLUSIVE. Fix by finding a set where the factors do\n"
                      "    not decode at 0.99 — here mass is size^3 on a plainly visible cube.")
            else:
                print(f"  ✓ sign agrees across conventions "
                      f"({'direct' if g1 > 0 else 'composed'} wins in both)")
            report.setdefault("test_B_robustness", {})[nm] = {
                "gap_ke": g1, "gap_ke_mean": g2, "sign_agrees": bool(agree)}
    if args.test in ("A", "both"):
        for sd, td, nm in zip(args.slide, args.collision, names):
            report.setdefault("test_A", {})[nm] = test_a(sd, td, nm, args)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
