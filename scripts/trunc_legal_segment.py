#!/usr/bin/env python
"""Does the PRE-CONTACT representation encode the set of LEGAL collision outcomes?

    python scripts/trunc_legal_segment.py                    # both encoders, all rungs
    python scripts/trunc_legal_segment.py --models videomae-base --honest-only

Reads §43's truncation captures (old/outputs_trunc_<model>_acts.npz): every clip
re-encoded from frames [0, k] only, so for k <= 20 the aftermath does not exist
and cannot be attended to (§4's confound removed by construction).

THE LEGAL SET
-------------
Given the pre-contact state (v_A, v_B, m_A, m_B, contact normal n), physics
allows a one-parameter family of outcomes, one per restitution e in [0, 1]:

    J(e) = (1+e) mu u_n,   v_A' = v_A - J/m_A n,   v_B' = v_B + J/m_B n
    mu = m_A m_B / M,  u_n = (v_A - v_B).n > 0,  M = m_A + m_B

A segment in the 4-d outcome space (v_A', v_B'). Write any outcome in physical
coordinates instead:

    c   = (m_A v_A' + m_B v_B') / M      centre-of-mass velocity (2-d)
    r_t = (v_B' - v_A').t                relative velocity, tangential
    r_n = (v_B' - v_A').n                relative velocity, normal

and the legal set is exactly

    c   = c_pre          MOMENTUM CONSERVATION   (mass-weighted: needs m_A/m_B)
    r_t = r_t_pre        no tangential impulse   (mass-free)
    r_n = e u_n, e in [0,1]    the one free direction (mass-free)

This decomposition is unique (c and r together are an invertible linear map of
the outcome for fixed masses), so any decoded outcome splits into a MOMENTUM
error, a TANGENTIAL error and a RESTITUTION component, and nothing else.

!! IDENTITY CAVEAT (§12). Post-contact c EQUALS pre-contact c for every legal
outcome. So "does the anticipated outcome conserve momentum" and "does the
representation encode the pre-contact CM velocity" are the SAME estimator here
-- a truncated clip cannot separate them. The momentum row reports how well the
mass-weighted quantity is available, not whether the law is applied.

READOUTS (all 5-fold CV, site / pool / alpha chosen by LOO inside each training
fold -- no winner's curse)
  direct      activations -> 4-d outcome, then decomposed
  composed    activations -> v_A, v_B, log m_A, log m_B, n separately, then the
              impulse law at the TRAINING-mean e. On the legal line by construction
              (w.r.t. its own decoded masses).
  oracles     true pre-contact features, same ridge pipeline:
              lin_visible  [v_A, v_B, log m, radii, n]  -> outcome
              lin_massfree [v_A, v_B, n]                -> outcome
              law_mean_e   the exact law with e at its training mean: the ceiling
                           for any observer that cannot see the impact

Targets are the RECONSTRUCTED legal outcome at the clip's simulated e (§43a), so
the violating half is usable pre-contact (frames 0-20 are honest in all 150).
For k > 20 the violating clips SHOW an edited aftermath; use --honest-only there.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = np.array([1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5])
POOLS = ["A.all", "B.all", "all.all", "A.all+B.all"]


# --------------------------------------------------------------------------- #
# physics
# --------------------------------------------------------------------------- #
def pre_state(labels):
    """Arrays over scenes: vA, vB [n,2], mA, mB, rA, rB [n], n_hat, t_hat [n,2], e."""
    S = {k: [] for k in ("vA", "vB", "mA", "mB", "rA", "rB", "n", "e")}
    for l in labels:
        c = l["collision"]
        vA = np.array(c["pre"]["velocity_A"][:2]); vB = np.array(c["pre"]["velocity_B"][:2])
        n = np.array(c["contact_normal"][:2]); n /= np.linalg.norm(n)
        if (vA - vB) @ n < 0:
            n = -n                                  # orient: A approaches B along +n
        S["vA"].append(vA); S["vB"].append(vB); S["n"].append(n)
        S["mA"].append(l["objects"]["A"]["mass"]); S["mB"].append(l["objects"]["B"]["mass"])
        S["rA"].append(l["objects"]["A"]["radius"]); S["rB"].append(l["objects"]["B"]["radius"])
        S["e"].append(l["sampled_parameters"]["restitution"])
    S = {k: np.array(v, dtype=np.float64) for k, v in S.items()}
    S["t"] = np.stack([-S["n"][:, 1], S["n"][:, 0]], 1)
    return S


def law(vA, vB, mA, mB, n, e):
    mu = mA * mB / (mA + mB)
    un = np.einsum("ij,ij->i", vA - vB, n)
    J = ((1 + e) * mu * un)[:, None]
    return np.hstack([vA - J / mA[:, None] * n, vB + J / mB[:, None] * n])


def physical_coords(o, S):
    """outcome [n,4] -> c [n,2], r_t [n], r_n [n] using the TRUE masses / normal."""
    vA, vB = o[:, :2], o[:, 2:]
    M = (S["mA"] + S["mB"])[:, None]
    c = (S["mA"][:, None] * vA + S["mB"][:, None] * vB) / M
    r = vB - vA
    return c, np.einsum("ij,ij->i", r, S["t"]), np.einsum("ij,ij->i", r, S["n"])


def segment_distance(o, S):
    """Euclidean distance (4-d velocity space) from o to the scene's legal segment."""
    lo = law(S["vA"], S["vB"], S["mA"], S["mB"], S["n"], np.zeros(len(o)))
    hi = law(S["vA"], S["vB"], S["mA"], S["mB"], S["n"], np.ones(len(o)))
    d = hi - lo
    s = np.clip(np.einsum("ij,ij->i", o - lo, d) / np.einsum("ij,ij->i", d, d), 0, 1)
    return np.linalg.norm(o - (lo + s[:, None] * d), axis=1), np.linalg.norm(d, axis=1)


def r2(y, yhat):
    y = y.reshape(len(y), -1); yhat = yhat.reshape(len(y), -1)
    return float(1 - ((y - yhat) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


# --------------------------------------------------------------------------- #
# ridge with LOO model selection
# --------------------------------------------------------------------------- #
def _prep(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    return (Xtr - mu) / sd, (Xte - mu) / sd


class FoldCache:
    """One SVD per (outer fold, feature set), shared by every target fit on it.
    Stores U, s of the z-scored train block and the test block projected onto V."""

    def __init__(self, feats, folds, n):
        self.folds, self.names, self.n = folds, list(feats), n
        self.svd = []
        for te in folds:
            tr = np.setdiff1d(np.arange(n), te)
            per = {}
            for name, X in feats.items():
                Xtr, Xte = _prep(X[tr], X[te])        # z-scored on train: centred
                U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
                per[name] = (U, s, Xte @ Vt.T)
            self.svd.append(per)


def loo_mse(U, s, Y):
    """LOO MSE for every alpha, X/Y already centred. Returns [len(ALPHAS)]."""
    UtY = U.T @ Y
    U2 = U ** 2
    n = len(U)
    out = []
    for a in ALPHAS:
        f = s ** 2 / (s ** 2 + a)
        pred = U @ (f[:, None] * UtY)
        h = U2 @ f + 1.0 / n                        # +1/n: the intercept
        out.append((((Y - pred) / (1 - h)[:, None]) ** 2).mean())
    return np.array(out)


def cv_select_predict(cache, Y):
    """Per outer fold choose (feature set, alpha) by LOO on train (targets z-scored
    on train so every column counts), predict test."""
    pred = np.zeros_like(Y)
    chosen = []
    for te, per in zip(cache.folds, cache.svd):
        tr = np.setdiff1d(np.arange(cache.n), te)
        ym = Y[tr].mean(0)
        Yc = (Y[tr] - ym) / (Y[tr].std(0) + 1e-12)
        best = (np.inf, None, None)
        for name, (U, s, _) in per.items():
            m = loo_mse(U, s, Yc)
            i = int(np.argmin(m))
            if m[i] < best[0]:
                best = (m[i], name, ALPHAS[i])
        _, name, a = best
        U, s, Pte = per[name]
        pred[te] = Pte @ ((s / (s ** 2 + a))[:, None] * (U.T @ (Y[tr] - ym))) + ym
        chosen.append(name)
    return pred, max(set(chosen), key=chosen.count)



# --------------------------------------------------------------------------- #
def evaluate(o_hat, o_true, S):
    c_t, rt_t, rn_t = physical_coords(o_true, S)
    c_h, rt_h, rn_h = physical_coords(o_hat, S)
    dist, seglen = segment_distance(o_hat, S)
    un = np.einsum("ij,ij->i", S["vA"] - S["vB"], S["n"])
    e_impl = rn_h / un
    spread = np.sqrt(((o_true - o_true.mean(0)) ** 2).sum(1).mean())
    return {
        "R2_outcome": r2(o_true, o_hat),
        "R2_momentum_c": r2(c_t, c_h),
        "R2_tangential_rt": r2(rt_t, rt_h),
        "R2_normal_rn": r2(rn_t, rn_h),
        "rmse_momentum_c": float(np.sqrt(((c_t - c_h) ** 2).sum(1).mean())),
        "rmse_tangential_rt": float(np.sqrt(((rt_t - rt_h) ** 2).mean())),
        "rmse_normal_rn": float(np.sqrt(((rn_t - rn_h) ** 2).mean())),
        "median_dist_to_segment_over_spread": float(np.median(dist) / spread),
        "median_dist_to_segment_over_seglen": float(np.median(dist / seglen)),
        "frac_implied_e_in_0_1": float(np.mean((e_impl >= 0) & (e_impl <= 1))),
        "corr_implied_e_true_e": float(np.corrcoef(e_impl, S["e"])[0, 1]),
    }


def paired_boot(fn, n, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.array([fn(rng.integers(0, n, n)) for _ in range(B)])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


class OldRungs:
    """§43's single-file truncation capture: old/outputs_trunc_<model>_acts.npz."""

    def __init__(self, model):
        self.z = np.load(ROOT / f"old/outputs_trunc_{model}_acts.npz", allow_pickle=True)
        self.labels = [json.loads(s) for s in self.z["labels"]]
        self.sites, self.pools = list(self.z["sites"]), list(self.z["pools"])
        self.ladder = sorted(int(k[1:]) for k in self.z.files if k.startswith("k"))

    def acts(self, k, keep):
        return self.z[f"k{k}"][keep].astype(np.float64)            # [n, site, pool, d]


class DirRungs:
    """capture_encoder.py --resample-to output: ROOT/k<k>/<clip>.npz, clips aligned by id."""

    def __init__(self, root):
        self.root = Path(root)
        self.ladder = sorted(int(d.name[1:]) for d in self.root.glob("k*") if d.is_dir())
        ids = [set(q.stem for q in (self.root / f"k{k}").glob("*.npz")) for k in self.ladder]
        self.ids = sorted(set.intersection(*ids))
        self.labels, self.sites, self.pools = [], None, ["A.all", "B.all", "all.all"]
        for c in self.ids:
            with np.load(self.root / f"k{self.ladder[0]}" / f"{c}.npz") as z:
                meta = json.loads(bytes(z["meta"]).decode())
            self.labels.append(meta["labels"])
            if self.sites is None:
                self.sites, self._pools9 = meta["sites"], meta["pools"]
        self._pi = [self._pools9.index(p) for p in self.pools]

    def acts(self, k, keep):
        out = []
        for c, kp in zip(self.ids, keep):
            if not kp:
                continue
            with np.load(self.root / f"k{k}" / f"{c}.npz") as z:
                out.append(np.stack([z[f"act.{s}"][self._pi] for s in self.sites]))
        return np.stack(out).astype(np.float64)


def folds_grouped(groups, k=5, seed=0):
    """Outer folds that never split a seed (twins share their pre-contact frames)."""
    u = np.random.default_rng(seed).permutation(np.unique(groups))
    return [np.flatnonzero(np.isin(groups, g)) for g in np.array_split(u, k)]


def run_model(model, honest_only, ladder_only=None, source=None):
    src = DirRungs(source) if source else OldRungs(model)
    labels, sites, pools = src.labels, src.sites, src.pools
    keep = np.array([not l["violates_momentum"] for l in labels]) if honest_only else np.ones(len(labels), bool)
    labels = [l for l, k in zip(labels, keep) if k]
    S = pre_state(labels)
    n = len(labels)
    o_true = law(S["vA"], S["vB"], S["mA"], S["mB"], S["n"], S["e"])
    folds = folds_grouped(np.array([l["seed"] for l in labels]))
    rd = sorted({str(l.get("scene_constants", {}).get("render_device")) for l in labels})
    print(f"  n={n}  seeds={len(set(l['seed'] for l in labels))}  render devices {rd}")

    # ---- oracles: rung-independent ------------------------------------------------
    vis = np.hstack([S["vA"], S["vB"], np.log(S["mA"])[:, None], np.log(S["mB"])[:, None],
                     S["rA"][:, None], S["rB"][:, None], S["n"]])
    mf = np.hstack([S["vA"], S["vB"], S["n"]])
    o_lawmean = np.zeros_like(o_true)
    for te in folds:
        tr = np.setdiff1d(np.arange(n), te)
        o_lawmean[te] = law(S["vA"][te], S["vB"][te], S["mA"][te], S["mB"][te], S["n"][te],
                            np.full(len(te), S["e"][tr].mean()))
    oracles = {
        "lin_visible": evaluate(cv_select_predict(FoldCache({"vis": vis}, folds, n), o_true)[0], o_true, S),
        "lin_massfree": evaluate(cv_select_predict(FoldCache({"mf": mf}, folds, n), o_true)[0], o_true, S),
        "law_mean_e": evaluate(o_lawmean, o_true, S),
    }
    c_t, rt_t, rn_t = physical_coords(o_true, S)
    spread = {"sd_momentum_c": float(np.sqrt(((c_t - c_t.mean(0)) ** 2).sum(1).mean())),
              "sd_tangential_rt": float(rt_t.std()), "sd_normal_rn": float(rn_t.std()),
              "sd_normal_rn_from_e_alone": float((S["e"] * np.einsum("ij,ij->i", S["vA"] - S["vB"], S["n"])
                                                  - (S["e"].mean() * np.einsum("ij,ij->i", S["vA"] - S["vB"], S["n"]))).std())}
    print("  target spread: " + "  ".join(f"{k} {v:.3f}" for k, v in spread.items()))

    rungs = {}
    for k in ladder_only or src.ladder:
        A = src.acts(k, keep)                               # [n, site, pool, d]
        feats = {}
        for si, s in enumerate(sites):
            for p in POOLS:
                if p == "A.all+B.all":
                    X = np.hstack([A[:, si, pools.index("A.all")], A[:, si, pools.index("B.all")]])
                else:
                    X = A[:, si, pools.index(p)]
                feats[f"{s}|{p}"] = X
        cache = FoldCache(feats, folds, n)
        del feats, A
        o_dir, where = cv_select_predict(cache, o_true)
        ing = {}
        for name, Y in (("vA", S["vA"]), ("vB", S["vB"]),
                        ("logm", np.stack([np.log(S["mA"]), np.log(S["mB"])], 1)),
                        ("n", S["n"])):
            ing[name], _ = cv_select_predict(cache, Y)
        n_hat = ing["n"] / np.linalg.norm(ing["n"], axis=1, keepdims=True)
        o_comp = np.zeros_like(o_true)
        o_comp_n = np.zeros_like(o_true)        # same, but the TRUE contact normal
        for te in folds:
            tr = np.setdiff1d(np.arange(n), te)
            args_ = (ing["vA"][te], ing["vB"][te], np.exp(ing["logm"][te, 0]), np.exp(ing["logm"][te, 1]))
            e_bar = np.full(len(te), S["e"][tr].mean())
            o_comp[te] = law(*args_, n_hat[te], e_bar)
            o_comp_n[te] = law(*args_, S["n"][te], e_bar)
        # direct readout of the CM velocity itself (best-vs-best for the momentum row)
        c_true = physical_coords(o_true, S)[0]
        c_dir, _ = cv_select_predict(cache, c_true)
        # paired bootstrap: composed minus direct, on the momentum component
        c_comp = physical_coords(o_comp, S)[0]
        c_from_dir = physical_coords(o_dir, S)[0]
        ci = paired_boot(lambda i: r2(c_true[i], c_comp[i]) - r2(c_true[i], c_from_dir[i]), n)
        rungs[k] = {
            "direct": evaluate(o_dir, o_true, S), "direct_site": where,
            "composed": evaluate(o_comp, o_true, S),
            "composed_true_normal": evaluate(o_comp_n, o_true, S),
            "R2_momentum_c_direct_target": r2(c_true, c_dir),
            "ingredients_R2": {"vA": r2(S["vA"], ing["vA"]), "vB": r2(S["vB"], ing["vB"]),
                               "logm": r2(np.log(np.stack([S["mA"], S["mB"]], 1)), ing["logm"]),
                               "n": r2(S["n"], n_hat)},
            "momentum_composed_minus_direct_ci95": ci,
        }
        d, c = rungs[k]["direct"], rungs[k]["composed"]
        print(f"  k={k:2d}  outcome R2 {d['R2_outcome']:.3f}/{c['R2_outcome']:.3f}   "
              f"c {d['R2_momentum_c']:+.3f}/{c['R2_momentum_c']:+.3f} (direct-target {rungs[k]['R2_momentum_c_direct_target']:+.3f})  "
              f"r_t {d['R2_tangential_rt']:+.3f}/{c['R2_tangential_rt']:+.3f}  "
              f"r_n {d['R2_normal_rn']:+.3f}/{c['R2_normal_rn']:+.3f}  "
              f"seg-dist {d['median_dist_to_segment_over_spread']:.3f}/{c['median_dist_to_segment_over_spread']:.3f}  "
              f"e in[0,1] {d['frac_implied_e_in_0_1']:.2f}  [{where}]", flush=True)
    return {"n": n, "honest_only": honest_only, "target_spread": spread, "oracles": oracles, "rungs": rungs}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["videomae-base", "vjepa2-vitl"])
    p.add_argument("--honest-only", action="store_true")
    p.add_argument("--rungs", nargs="+", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--source", default=None,
                   help="capture_encoder.py --resample-to root (with k*/ subdirs) instead of §43's npz; "
                        "use with a single --models entry")
    args = p.parse_args()
    out = {"protocol": "trunc_legal_segment", "source": args.source or "old/outputs_trunc_<model>_acts.npz",
           "models": {}}
    for m in args.models:
        print(f"\n[{m}]  honest_only={args.honest_only}   cells: direct/composed")
        res = run_model(m, args.honest_only, args.rungs, args.source)
        for name, o in res["oracles"].items():
            print(f"  oracle {name:12s} outcome R2 {o['R2_outcome']:.3f}  c {o['R2_momentum_c']:+.3f}  "
                  f"r_t {o['R2_tangential_rt']:+.3f}  r_n {o['R2_normal_rn']:+.3f}  "
                  f"seg-dist {o['median_dist_to_segment_over_spread']:.3f}")
        out["models"][m] = res
    tag = "honest" if args.honest_only else "all"
    path = Path(args.out) if args.out else ROOT / f"docs/artifacts/trunc_legal_segment_{tag}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
