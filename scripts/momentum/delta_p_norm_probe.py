"""Can a LINEAR probe read |Delta p| -- the size of the conservation violation?

    python scripts/momentum/delta_p_norm_probe.py \
        --outputs outputs/consnoise_cogvideox-2b \
        --dataset-dir datasets/conservation_set_1000

|Delta p| = |p_post - p_pre| is exactly 0 on every honest clip and 0.02-9.7 on the
violating ones. Two readouts, both scored out-of-fold:

  direct  ridge on the scalar |Delta p|.
  norm    ridge on the Delta-p VECTOR, |prediction| taken outside (§34a's route).

Protocol, each part forced by the set (see probe_conservation_set.py's docstring):
  * GROUPED folds by seed -- twins share their pre-contact segment, so splitting a
    pair across train/test leaks the honest deflection split into the test twin.
  * The cell (site x noise level x pool) and the ridge alpha are chosen by inner
    grouped CV on TRAIN only; the outer fold never sees the selection.
  * Permutation null: |Delta p| is shuffled across seed groups (twins keep their
    pairing) and the WHOLE selection is re-run, so the null prices in the search.
  * References, because |Delta p| is a norm of products (§12, §38e):
      vae     the same probe on the VAE latent -- signal already in the pixels;
      oracle  ridge on the TRUE ingredients (m_A, m_B, dv_A, dv_B), linear and
              with all pairwise products. The linear oracle is the ceiling a
              linear readout of perfect kinematics can reach.

Writes docs/artifacts/delta_p_norm_<model>.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np

ALPHAS = [1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5]
POOLS = ["all.pre", "all.post", "all.all", "A.post+B.post", "A.pre+A.post+B.pre+B.post"]


def load(outdir: str, pools: list[str]):
    paths = sorted(glob.glob(f"{outdir}/momentum_probe/activations/*.npz"))
    if not paths:
        raise FileNotFoundError(f"no captures in {outdir}")
    feats, labels, meta0 = {}, [], None
    for p in paths:
        z = np.load(p, allow_pickle=False)
        m = json.loads(bytes(z["meta"]).decode())
        meta0 = meta0 or m
        pi = {name: i for i, name in enumerate(m["pools"])}
        labels.append(m["labels"])
        for k in z.files:
            if not k.startswith("act."):
                continue
            a = z[k]                                   # [levels, pools, dim]
            for li in range(a.shape[0]):
                for pool in pools:
                    x = np.concatenate([a[li, pi[q]] for q in pool.split("+")])
                    feats.setdefault((k[4:], li, pool), []).append(x)
    cells = {c: np.stack(v).astype(np.float32) for c, v in feats.items()}
    return cells, labels, meta0


def facts(L):
    c = L["collision"]
    dp = np.subtract(c["post"]["total_momentum"][:2], c["pre"]["total_momentum"][:2])
    dvA = np.subtract(c["post"]["velocity_A"][:2], c["pre"]["velocity_A"][:2])
    dvB = np.subtract(c["post"]["velocity_B"][:2], c["pre"]["velocity_B"][:2])
    mA, mB = L["objects"]["A"]["mass"], L["objects"]["B"]["mass"]
    return dict(seed=L["seed"], viol=not L["momentum_conserved"], dp=dp,
                ingredients=np.r_[mA, mB, dvA, dvB])


class Cell:
    """Dual-ridge predictions for every alpha from one eigendecomposition per split.

    The permutation null only changes Y, so caching (V, w, K) per split makes each
    null draw a few matrix-vector products instead of a refit.
    """

    def __init__(self, X, tr, te):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Ztr, Zte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        self.xm = Ztr.mean(0)
        Zc = (Ztr - self.xm).astype(np.float64)
        self.w, self.V = np.linalg.eigh(Zc @ Zc.T)
        self.KV = ((Zte - self.xm).astype(np.float64) @ Zc.T) @ self.V

    def predict(self, Ytr):
        ym = Ytr.mean(0)
        A = self.V.T @ (Ytr - ym)
        return {a: self.KV @ (A / (self.w + a)[:, None]) + ym for a in ALPHAS}


def r2(y, p):
    y, p = y.reshape(len(y), -1), p.reshape(len(p), -1)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


def auc(score, viol):
    o = np.argsort(score); r = np.empty(len(score)); r[o] = np.arange(1, len(score) + 1)
    n1 = int(viol.sum()); n0 = len(viol) - n1
    return float((r[viol].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def make_splits(groups, rng, n_out=5, n_in=3):
    """Outer grouped folds, each with inner grouped folds over its train part."""
    uniq = rng.permutation(np.unique(groups))
    out = []
    for g_te in np.array_split(uniq, n_out):
        te = np.where(np.isin(groups, g_te))[0]
        tr = np.where(~np.isin(groups, g_te))[0]
        g_tr = rng.permutation(np.unique(groups[tr]))
        inner = []
        for g_va in np.array_split(g_tr, n_in):
            va_mask = np.isin(groups[tr], g_va)
            inner.append((tr[~va_mask], tr[va_mask]))
        out.append((tr, te, inner))
    return out


def nested(cells, splits, Y, readout, cache):
    """Out-of-fold |Delta p| prediction with cell + alpha chosen on train only."""
    y = np.linalg.norm(Y, axis=1) if readout == "norm" else Y[:, 0]
    oof, chosen = np.zeros(len(y)), []
    for k, (tr, te, inner) in enumerate(splits):
        best = (-np.inf, None, None)
        for name in cells:
            scores = {a: [] for a in ALPHAS}
            for f, (itr, iva) in enumerate(inner):
                preds = cache[(name, k, f)].predict(Y[itr])
                for a, P in preds.items():
                    ph = np.linalg.norm(P, axis=1) if readout == "norm" else P[:, 0]
                    scores[a].append(r2(y[iva], ph))
            for a in ALPHAS:
                s = float(np.mean(scores[a]))
                if s > best[0]:
                    best = (s, name, a)
        _, name, a = best
        P = cache[(name, k, "outer")].predict(Y[tr])[a]
        oof[te] = np.linalg.norm(P, axis=1) if readout == "norm" else P[:, 0]
        chosen.append({"cell": list(map(str, name)), "alpha": a})
    return oof, chosen


def build_cache(cells, splits):
    cache = {}
    for name, X in cells.items():
        for k, (tr, te, inner) in enumerate(splits):
            cache[(name, k, "outer")] = Cell(X, tr, te)
            for f, (itr, iva) in enumerate(inner):
                cache[(name, k, f)] = Cell(X, itr, iva)
    return cache


def per_cell_grid(cells, splits, Y, cache):
    """OOF R2 of the direct readout per cell at a fixed alpha -- the depth x noise map."""
    y = Y[:, 0]; grid = {}
    for name in cells:
        best = -np.inf
        for a in ALPHAS:
            oof = np.zeros(len(y))
            for k, (tr, te, _) in enumerate(splits):
                oof[te] = cache[(name, k, "outer")].predict(Y[tr])[a][:, 0]
            best = max(best, r2(y, oof))
        grid["|".join(map(str, name))] = round(best, 4)
    return grid


def oracle(F, y, viol, splits, quadratic):
    X = F.copy()
    if quadratic:
        i, j = np.triu_indices(F.shape[1])
        X = np.c_[F, F[:, i] * F[:, j]]
    best = (-np.inf, float("nan"))
    for a in ALPHAS:
        oof = np.zeros(len(y))
        for tr, te, _ in splits:
            oof[te] = Cell(X, tr, te).predict(y[tr, None])[a][:, 0]
        best = max(best, (r2(y, oof), auc(oof, viol)))
    return {"r2": round(best[0], 4), "auc": round(best[1], 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", required=True, help="capture dir, e.g. outputs/consnoise_cogvideox-2b")
    ap.add_argument("--pools", default=",".join(POOLS))
    ap.add_argument("--sites", default=None, help="comma-separated subset of sites (default: all)")
    ap.add_argument("--n-perm", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()
    t0 = time.time()

    cells, labels, meta = load(args.outputs, args.pools.split(","))
    if args.sites:
        keep = set(args.sites.split(","))
        cells = {c: X for c, X in cells.items() if c[0] in keep}
    F = [facts(L) for L in labels]
    groups = np.array([f["seed"] for f in F]); viol = np.array([f["viol"] for f in F])
    dp = np.stack([f["dp"] for f in F]); ydp = np.linalg.norm(dp, axis=1)
    ing = np.stack([f["ingredients"] for f in F])
    vae = {c: X for c, X in cells.items() if c[0] == "vae.latent"}
    model_cells = {c: X for c, X in cells.items() if c[0] != "vae.latent"}
    print(f"{len(F)} clips ({viol.sum()} violating), {len(model_cells)} model cells, "
          f"{len(vae)} VAE cells; levels {meta['noise_levels']}", flush=True)

    rng = np.random.default_rng(args.seed)
    splits = make_splits(groups, rng)
    cache = build_cache(cells, splits)
    print(f"cache built in {time.time() - t0:.0f}s", flush=True)

    res = {"n_clips": len(F), "n_violating": int(viol.sum()), "noise_levels": meta["noise_levels"],
           "inversion": meta.get("metadata", {}), "pools": args.pools.split(",")}
    for readout, Y in (("direct", ydp[:, None]), ("norm", dp)):
        for tag, cs in (("model", model_cells), ("vae", vae)):
            if not cs:
                continue
            oof, chosen = nested(cs, splits, Y, readout, cache)
            res[f"{readout}.{tag}"] = {
                "r2": round(r2(ydp, oof), 4),
                "r2_violating_only": round(r2(ydp[viol], oof[viol]), 4),
                "spearman": round(float(np.corrcoef(np.argsort(np.argsort(ydp)), np.argsort(np.argsort(oof)))[0, 1]), 4),
                "auc_honest_vs_violating": round(auc(oof, viol), 4),
                "chosen": chosen,
            }
            print(readout, tag, {k: v for k, v in res[f"{readout}.{tag}"].items() if k != "chosen"}, flush=True)

    # Null for the headline (direct, model cells): shuffle |dp| across seed groups.
    null = []
    uniq = np.unique(groups); gidx = {g: np.where(groups == g)[0] for g in uniq}
    for i in range(args.n_perm):
        perm = rng.permutation(uniq)
        Yp = np.empty_like(ydp)
        for g, h in zip(uniq, perm):
            a, b = gidx[g], gidx[h]
            if len(a) == len(b):
                Yp[a] = ydp[b]
            else:                               # a truncated twin: fall back to own label
                Yp[a] = ydp[a]
        oof, _ = nested(model_cells, splits, Yp[:, None], "direct", cache)
        null.append(r2(Yp, oof))
        if (i + 1) % 10 == 0:
            print(f"  null {i + 1}/{args.n_perm}: max so far {max(null):.4f}", flush=True)
    obs = res["direct.model"]["r2"]
    res["direct.model"]["null_r2"] = {"mean": round(float(np.mean(null)), 4), "max": round(float(np.max(null)), 4),
                                      "p": round(float((1 + sum(n >= obs for n in null)) / (1 + len(null))), 4)}

    res["oracle.linear"] = oracle(ing, ydp, viol, splits, quadratic=False)
    res["oracle.quadratic"] = oracle(ing, ydp, viol, splits, quadratic=True)
    res["grid.direct"] = per_cell_grid(model_cells, splits, ydp[:, None], cache)
    res["elapsed_s"] = round(time.time() - t0)
    print("oracle", res["oracle.linear"], res["oracle.quadratic"], "null", res["direct.model"]["null_r2"])

    model = Path(args.outputs).name.split("_", 1)[-1]
    out = Path(args.out or f"docs/artifacts/delta_p_norm_{model}.json")
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
