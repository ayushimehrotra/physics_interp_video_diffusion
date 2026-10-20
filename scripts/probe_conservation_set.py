"""Probes for `conservation_set_v1`: paired scoring, grouped folds, exact paired null.

THREE THINGS THIS DOES THAT `--mode probe` DOES NOT, each forced by the set's design.

1. GROUPED FOLDS. Every clip has a twin that shares its entire pre-contact segment.
   Split by clip and the probe sees a scene's honest twin in train and its violating
   twin in test, memorises the honest deflection split, and flags the deviation. On
   simulated twins that reads AUC 0.213 -- 0.787 discrimination -- from data with no
   leak in it at all. Grouped by seed: 0.498. Every fold here is grouped.

2. PAIRED SCORING. The probe still reads one clip and emits one number; the pairing
   is in the comparison. Asking "within this pair, is the violating twin scored
   higher" removes all between-scene variance, which is the whole reason the pairs
   were rendered. Joseph et al. use matched pairs for the DATASET and grouped CV for
   the folds but score per clip, so this is a strict improvement on their protocol.

3. THE EXACT PAIRED NULL. Under H0 the two twins of a pair are exchangeable, so
   flipping which twin carries the `violating` label -- independently per pair, with
   probability 1/2 -- generates the null distribution exactly, with no asymptotic
   assumption. The whole selection procedure is re-run inside each permutation, so
   the null prices in the cell search the observed value enjoyed.

TWO READOUTS, because the label's algebra caps the first one:
  direct  ridge on the 0/1 label. A linear map cannot form `|m_A dv_A + m_B dv_B|`
          -- handed the EXACT ingredients it scores AUC 0.49 -- so this is reported
          as the baseline it is, not as the measurement.
  norm    ridge on the Delta-p VECTOR, then |prediction| taken outside the probe.
          The nonlinearity lives in numpy, so the label's shape stops being a ceiling.

Pools are frame-wide only. Object-local pooling carries the §20 ground-truth-mask
leak and §38d measured it buying nothing on this target.
"""
from __future__ import annotations
import argparse, glob, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from physics_interp.interp import ridge_fit, _r2          # noqa: E402

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

# Frame-wide pools win on the BINARY target (§38d) but a frame-wide mean cannot
# hold a ratio: §34d measured `log_mass_ratio` at ~0 in every frame-wide pool and
# 0.75-0.81 object-local, because averaging the two spheres maps them onto the
# same coordinates and the quotient is gone before the probe sees anything. This
# set's whole route runs through the mass ratio, so the object-local CONCATS have
# to be candidates -- '+' joins on the feature axis, which is what lets a linear
# readout form a difference between the two objects. Single-object pools are left
# out: one of them cannot hold a relation between two objects either way.
POOLS = ["all.pre", "all.all", "A.pre+B.pre", "A.pre+A.post+B.pre+B.post"]


def load(outdir: str):
    paths = sorted(glob.glob(f"{outdir}/momentum_probe/activations/*.npz"))
    if not paths:
        raise FileNotFoundError(f"no captures in {outdir}")
    store, labels, pools, sites = {}, [], None, None
    for p in paths:
        z = np.load(p, allow_pickle=True)
        m = json.loads(bytes(z["meta"]).decode())
        pools = m["pools"]; labels.append(m["labels"])
        if sites is None:
            sites = [k[4:] for k in z.files if k.startswith("act.")]
        for s in sites:
            store.setdefault(s, []).append(z["act." + s][0])
    return {s: np.stack(v).astype(np.float64) for s, v in store.items()}, labels, pools


def scene_facts(L):
    pf = {p["frame"]: p for p in L["per_frame"]}
    v = lambda t, fr: np.array(pf[fr][t]["velocity"][:2], float)
    dp = (np.asarray(L["collision"]["post"]["total_momentum"][:2])
          - np.asarray(L["collision"]["pre"]["total_momentum"][:2]))
    return dict(seed=L["seed"], e=L["restitution"],
                mixed=L["objects"]["A"]["material"] != L["objects"]["B"]["material"],
                viol=L["momentum_conserved"] < 0.5, dp=dp,
                vA_pre=v("A", 18), mass_A=L["objects"]["A"]["mass"],
                metal_A=float(L["objects"]["A"]["material"] == "metal"))


def fit_predict(Xtr, Ytr, Xte, alpha):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    W, b = ridge_fit((Xtr - mu) / sd, Ytr, alpha)
    return (Xte - mu) / sd @ W + b


def predict_all_alphas(Xtr, Ytr, Xte):
    """Dual-ridge predictions for every alpha from a single eigendecomposition.

    The permutation null re-runs the whole cell search inside each of N draws, so
    the fit count is (cells x alphas x folds x draws) and the constant matters.
    Ridge is W = Xc'(G + aI)^-1 Yc with G = Xc Xc'; eigendecomposing G once makes
    every alpha a diagonal rescale, which is the difference between a tractable
    null and an untractable one.
    """
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    xm, ym = Ztr.mean(0), Ytr.mean(0)
    Zc, Yc = Ztr - xm, Ytr - ym
    G = Zc @ Zc.T
    w, V = np.linalg.eigh(G)
    Kte = (Zte - xm) @ Zc.T
    A = V.T @ Yc
    KV = Kte @ V
    return {a: KV @ (A / (w + a)[:, None]) + ym for a in ALPHAS}


def paired_acc(score, pairs, viol):
    """Fraction of pairs whose violating twin outscores its honest twin."""
    wins = [0.5 if score[i] == score[j] else float((score[i] > score[j]) == viol[i])
            for i, j in pairs]
    return float(np.mean(wins))


def auc_score(score, viol):
    """Rank-based AUC over ALL clips, ignoring which twin is which."""
    o = np.argsort(score); r = np.empty(len(score), float); r[o] = np.arange(1, len(score) + 1)
    n1 = int(viol.sum()); n0 = len(viol) - n1
    a = (r[viol].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(a)


def _stat(score, pairs, viol, stat):
    return _stat(score, pairs, viol, stat) if stat == "paired" else auc_score(score, viol)


def run(cells, pairs, viol, dp, mode, rng, n_out=5, n_in=3, stat="paired"):
    """Out-of-fold score with the cell chosen inside training only.

    `stat` picks the statistic, for the inner model selection as well as the
    final number: 'paired' compares each violating clip with its own twin;
    'auc' ranks all clips and ignores the pairing.
    """
    P = len(pairs)
    score = np.zeros(sum(len(p) for p in pairs))
    outer = np.array_split(rng.permutation(P), n_out)
    Y = viol.reshape(-1, 1).astype(float) if mode == "direct" else dp
    for k in range(n_out):
        te_p = outer[k]
        tr_p = np.concatenate([outer[j] for j in range(n_out) if j != k])
        tr_i = np.concatenate([list(pairs[p]) for p in tr_p])
        te_i = np.concatenate([list(pairs[p]) for p in te_p])
        inner = np.array_split(rng.permutation(tr_p), n_in)
        best = (-np.inf, None, None)
        for name, X in cells.items():
            acc = {a: [] for a in ALPHAS}
            for f in range(n_in):
                iva = inner[f]
                itr = np.concatenate([inner[j] for j in range(n_in) if j != f])
                ii = np.concatenate([list(pairs[p]) for p in itr])
                vi = np.concatenate([list(pairs[p]) for p in iva])
                preds = predict_all_alphas(X[ii], Y[ii], X[vi])
                for a, Pv in preds.items():
                    sc = np.zeros(len(viol))
                    sc[vi] = -Pv.ravel() if mode == "direct" else np.linalg.norm(Pv, axis=1)
                    acc[a].append(_stat(sc, [pairs[p] for p in iva], viol, stat))
            for a in ALPHAS:
                m = float(np.mean(acc[a]))
                if m > best[0]:
                    best = (m, name, a)
        _, name, a = best
        Pv = fit_predict(cells[name][tr_i], Y[tr_i], cells[name][te_i], a)
        score[te_i] = -Pv.ravel() if mode == "direct" else np.linalg.norm(Pv, axis=1)
    return _stat(score, pairs, viol, stat)


def control(cells, y, groups, rng, binary=False):
    """Simple grouped 5-fold readout for a positive control."""
    uniq = np.unique(groups); folds = np.array_split(rng.permutation(uniq), 5)
    best = -np.inf
    for name, X in cells.items():
        oof = np.zeros((len(y), y.shape[1]))
        for g in folds:
            te = np.isin(groups, g)
            oof[te] = fit_predict(X[~te], y[~te], X[te], 10.0)
        best = max(best, _r2(y, oof))
    return best


def mask_cells(outdir: str, dataset_dir: str):
    """§20's control, run through the IDENTICAL paired protocol.

    Object-local pooling picks its tokens with the ground-truth segmentation, so
    the choice of region is an information channel independent of the tokens'
    contents. On this set the two twins share a pre-contact segment but diverge
    after it, so their POST masks differ -- which means a probe handed
    `A.pre+A.post+B.pre+B.post` could in principle read the class off the mask
    alone and never look at an activation. Fitting the same probe on the pooling
    weights with no activations anywhere gives the floor the object-local numbers
    have to clear. The `all.*` masks are all-ones and identical in every scene, so
    they are the built-in sanity check: they must score chance.
    """
    from mask_only_floor import scene_mask_rows            # noqa: PLC0415
    paths = sorted(glob.glob(f"{outdir}/momentum_probe/activations/*.npz"))
    rows, raws, names = [], [], None
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(bytes(z["meta"]).decode())
        w, raw, nm = scene_mask_rows(Path(dataset_dir) / Path(path).stem, meta)
        names = names or nm
        rows.append(w); raws.append(raw)
    M, R = np.stack(rows), np.stack(raws)
    idx = {n: i for i, n in enumerate(names)}
    out = {}
    for spec in POOLS:
        for tag, src in (("normalised", M), ("occupancy", R)):
            out[f"mask-{tag}/{spec}"] = np.concatenate(
                [src[:, idx[q], :] for q in spec.split("+")], axis=-1).astype(np.float64)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", nargs="+",
                    default=["outputs_cons_videomae-base", "outputs_cons_vjepa2-vitl"])
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset-dir", default="momentum_dataset/conservation_set_v1")
    ap.add_argument("--pools", default=None,
                    help="comma-separated pool specs, overriding POOLS. "
                         "'all.pre,all.all' is the leak-free set: those masks are "
                         "all-ones and identical in every scene, so §20's floor is "
                         "exactly 0.5 by construction rather than by measurement "
                         "(the object-local concats measure 0.560 on the norm route)")
    ap.add_argument("--mask-floor", action="store_true",
                    help="replace activations with the pooling masks (§20 control)")
    ap.add_argument("-o", "--out", default="docs/artifacts/conservation_probes.json")
    args = ap.parse_args()
    global POOLS
    if args.pools:
        POOLS = [p.strip() for p in args.pools.split(",")]
    report = {"pools": POOLS}

    for outdir in args.outputs:
        acts, labels, pools = load(outdir)
        F = [scene_facts(L) for L in labels]
        if args.mask_floor:
            cells = mask_cells(outdir, args.dataset_dir)
        else:
            cells = {f"{s}/{p}": np.concatenate(
                         [acts[s][:, pools.index(q), :] for q in p.split("+")], axis=-1)
                     for s in acts for p in POOLS}
        viol = np.array([f["viol"] for f in F])
        dp = np.stack([f["dp"] for f in F])
        seeds = np.array([f["seed"] for f in F])
        print(f"\n{'='*78}\n{outdir}   {len(F)} clips, {len(cells)} candidate cells")

        # positive controls, grouped, so the captures are known to be sound
        rng = np.random.default_rng(args.seed)
        ctl = {
            "velocity_A_pre": control(cells, np.stack([f["vA_pre"] for f in F]), seeds, rng),
            "log mass_A": control(cells, np.log(np.array([[f["mass_A"]] for f in F])), seeds, rng),
            "material_A (metal)": control(cells, np.array([[f["metal_A"]] for f in F]), seeds, rng),
            "momentum_delta (vector)": control(cells, dp, seeds, rng),
        }
        print("  positive controls (grouped 5-fold, best cell, R2):")
        for k, v in ctl.items():
            print(f"     {k:26s} {v:+.3f}")

        report[outdir] = {"controls": ctl, "cells": {}}
        groups = defaultdict(list)
        for i, f in enumerate(F):
            groups[(f["e"], f["mixed"], f["seed"])].append(i)
        cellsets = {"elastic_mixed": (1.0, True), "inelastic_mixed": (0.15, True),
                    "elastic_same": (1.0, False), "inelastic_same": (0.15, False)}
        for cname in list(cellsets) + ["ALL"]:
            want = cellsets.get(cname)
            pairs = [tuple(v) for k, v in sorted(groups.items())
                     if len(v) == 2 and (want is None or (k[0] == want[0] and k[1] == want[1]))]
            if not pairs:
                continue
            sel = np.concatenate([list(p) for p in pairs])
            sub = {n: X[sel] for n, X in cells.items()}
            remap = {g: i for i, g in enumerate(sel)}
            pr = [(remap[i], remap[j]) for i, j in pairs]
            vsub, dsub = viol[sel], dp[sel]
            print(f"\n  --- {cname}  ({len(pairs)} pairs)")
            report[outdir]["cells"][cname] = {"n_pairs": len(pairs)}
            for mode in ("direct", "norm"):
                obs = run(sub, pr, vsub, dsub, mode, np.random.default_rng(args.seed))
                null = []
                for t in range(args.n_perm):
                    r = np.random.default_rng(1000 + t)
                    vp, dpp = vsub.copy(), dsub.copy()
                    for i, j in pr:
                        if r.random() < 0.5:
                            # Exchange the twins COMPLETELY. Flipping only the
                            # scoring label while stage 1 still regresses the true
                            # Delta-p leaves the regressor trained to fire on the
                            # genuinely violating clip, so the flipped label
                            # mismatches systematically and the null lands BELOW
                            # chance -- measured at 0.37-0.47 before this fix,
                            # which would have turned every result into a false
                            # positive. Under H0 the twins are exchangeable, so
                            # everything derived from the label moves with it.
                            vp[i], vp[j] = vp[j], vp[i]
                            dpp[[i, j]] = dpp[[j, i]]
                    null.append(run(sub, pr, vp, dpp, mode, np.random.default_rng(args.seed)))
                null = np.array(null)
                p = float((null >= obs).mean())
                print(f"     {mode:6s} paired accuracy {obs:.3f}   null {null.mean():.3f} "
                      f"(95th {np.percentile(null,95):.3f})   p = {p:.3f}", flush=True)
                report[outdir]["cells"][cname][mode] = {
                    "paired_accuracy": obs, "null_mean": float(null.mean()),
                    "null_p95": float(np.percentile(null, 95)), "p": p}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
