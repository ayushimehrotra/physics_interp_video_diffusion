#!/usr/bin/env python3
"""The valid-collision subspace, found WITHOUT LABELS from pre-collision clips.

    python scripts/momentum/momentum_unsup_subspace.py \
        --model vjepa2-vitl \
        --full  outputs/cons_vjepa2-vitl/momentum_probe/activations \
        --trunc outputs/construnc_vjepa2-vitl_a100/activations

§62 (register §47) found the valid subspace with a READOUT: ridge from activations
to the physical outcome ``(dv_A, dv_B)``, then the legal line from the labels. This
script asks the same question with no physics label anywhere in the construction.
Only activations go in:

  P = the representation of the PRE-COLLISION clip  (resample rung ``k*/``, k <= 20,
      pool ``A.all+B.all``; the encoder never saw contact)
  Q = the POST-COLLISION representation of the full clip  (``A.post+B.post`` or
      ``all.post``)

For each scene the valid subspace is AFFINE: a point predicted from P, plus the
directions valid outcomes still vary along once P is known.

  1. Fit, on honest (valid) clips only, an activation-to-activation map
     ``f: P -> Q`` (PCA both sides, ridge between them, alpha by inner grouped
     CV on reconstruction error, no labels).
  2. Take the OUT-OF-FOLD honest residuals ``R = Q - f(P)`` (in-sample residuals
     would be shrunk by the fit) and PCA them. The top ``r`` components are the
     valid outcome variation the pre-collision clip cannot pin down. For a
     frictionless pair that should be ~1-D: restitution, latent until impact,
     sets where on the legal line the outcome lands.
  3. Score a held-out clip by how far its residual leaves ``span(U_r)``:
     Mahalanobis over the complement (each direction scaled by its honest residual
     variance), and a Euclidean share as a second form.

``r`` is chosen by parallel analysis on the training residuals (columns permuted
independently), so it is label-free too. Fixed ``r`` in {0, 1, 2, 4, 8} is
reported beside it; ``r = 0`` is "any residual at all".

Labels enter only AFTER the subspace is built, to evaluate it:

  * detection — AUC honest vs violating, and the paired sign test over twins
    (twins share a bit-identical pre-contact segment, so ``f(P)`` is the same for
    both and the paired difference is purely post-collision). Grid-max null by
    flipping which twin carries the label, per pair (§46d: group permutation is a
    no-op on a balanced twin design).
  * meaning — does the top residual direction read restitution? (AUC elastic vs
    inelastic on honest held-out clips, and the 2x2 cell breakdown.)
  * direction — share of the twin activation difference ``Q_v - Q_h`` inside
    ``span(U_r)`` vs a random r-dim subspace and vs the honest residuals' own share.
  * controls — the UNCONDITIONED version (``f`` = training mean: a plain PCA of
    valid post-collision activations), and rungs k = 22/25 where the aftermath is
    already visible to the "pre" clip.

Cell (site x post pool) is chosen by a label-free criterion: held-out honest
reconstruction R^2 of Q from P. The per-site curve and the grid-max over cells are
both reported. CPU only, reads captures.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))  # noqa: E401,E702
import _momentum_path  # noqa: F401

import argparse
import math
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from physics_interp.momentum import cli
from physics_interp.momentum.captures import load_clipset
from physics_interp.momentum.stats import bootstrap_ci, grouped_folds

PRE_POOL = "A.all+B.all"                  # resample rungs: only *.all is exact
POST_POOLS = ["A.post+B.post", "all.post"]
RANKS = [0, 1, 2, 4, 8]
ALPHAS = [1e-1, 1.0, 1e1, 1e2, 1e3, 1e4]

_G: dict = {}     # worker globals, filled before the fork


# --------------------------------------------------------------------------- #
# small numerics
# --------------------------------------------------------------------------- #


def fast_auc(s: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUC; ties broken by average rank."""
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    ss = s[order]
    if np.any(ss[1:] == ss[:-1]):                       # rare: average tied ranks
        _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
        csum = np.cumsum(cnt)
        ranks = (csum - (cnt - 1) / 2.0)[inv]
    npos = int(y.sum())
    nneg = len(y) - npos
    return float((ranks[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def sign_test(diff: np.ndarray) -> dict:
    """Exact one-sided binomial test that ``diff > 0`` more often than not (ties dropped)."""
    d = diff[np.isfinite(diff) & (diff != 0)]
    n, k = len(d), int((d > 0).sum())
    p = sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n if n else float("nan")
    return {"n": n, "positive": k, "fraction": k / n if n else float("nan"), "p_one_sided": p}


def pca_fit(X: np.ndarray, q: int):
    """Centre and project to the top ``q`` principal directions. Returns ``(mu, V [d, q])``."""
    mu = X.mean(0)
    Xc = X - mu
    # n << d: eigendecompose the Gram matrix
    G = Xc @ Xc.T
    w, U = np.linalg.eigh(G)
    w, U = w[::-1][:q], U[:, ::-1][:, :q]
    w = np.maximum(w, 1e-12)
    V = Xc.T @ U / np.sqrt(w)
    return mu, V


def standardiser(X: np.ndarray):
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 1e-8, sd, np.inf)                 # constant features -> 0 (§44e)
    return mu, sd


class Ridge:
    """Ridge from Zp to Zq, every alpha off one eigendecomposition."""

    def __init__(self, Zp, Zq):
        self.mp, self.mq = Zp.mean(0), Zq.mean(0)
        A = Zp - self.mp
        self.w, self.V = np.linalg.eigh(A.T @ A)
        self.AtY = self.V.T @ (A.T @ (Zq - self.mq))

    def predict(self, Zp, alpha):
        B = self.V @ (self.AtY / (self.w + alpha)[:, None])
        return (Zp - self.mp) @ B + self.mq


def inner_oof(Zp, Zq, groups, alpha=None, n_folds=5):
    """Out-of-fold predictions of Zq from Zp; alpha by inner CV MSE when not given."""
    folds = grouped_folds(groups, n_folds, seed=1)
    preds = {a: np.zeros_like(Zq) for a in (ALPHAS if alpha is None else [alpha])}
    for te in folds:
        tr = np.setdiff1d(np.arange(len(Zq)), te)
        m = Ridge(Zp[tr], Zq[tr])
        for a in preds:
            preds[a][te] = m.predict(Zp[te], a)
    best = min(preds, key=lambda a: ((Zq - preds[a]) ** 2).sum())
    return best, preds[best]


def parallel_analysis(R: np.ndarray, draws: int = 20, seed: int = 0) -> int:
    """Number of residual PCs whose variance exceeds the 95th pct of column-permuted data."""
    rng = np.random.default_rng(seed)
    obs = np.linalg.svd(R - R.mean(0), compute_uv=False) ** 2
    null = []
    for _ in range(draws):
        Rp = np.stack([rng.permutation(c) for c in R.T], 1)
        null.append(np.linalg.svd(Rp - Rp.mean(0), compute_uv=False) ** 2)
    thr = np.quantile(np.stack(null), 0.95, axis=0)
    k = 0
    while k < len(obs) and obs[k] > thr[k]:
        k += 1
    return k


# --------------------------------------------------------------------------- #
# one cell: (rung, site, post pool) -> held-out scores for every clip
# --------------------------------------------------------------------------- #


def fit_score(P, Q, honest, seed, folds, q, conditioned=True):
    """Held-out scores for every clip. Returns a dict of arrays [n] plus per-fold info."""
    n = len(Q)
    out = {f"maha_r{r}": np.full(n, np.nan) for r in RANKS + ["pa"]}
    out.update({f"eucl_r{r}": np.full(n, np.nan) for r in RANKS + ["pa"]})
    out["r2_recon"] = np.full(n, np.nan)        # per-clip held-out SSE / SST pieces
    out["coords"] = np.full((n, 8), np.nan)     # residual coords on the top 8 directions
    out["Zq"] = np.full((n, q), np.nan)
    info = []
    for te in folds:
        tr = np.setdiff1d(np.arange(n), te)
        tr = tr[honest[tr]]
        # standardise + PCA on the training honest clips only
        mq, sq = standardiser(Q[tr]); mp, sp = standardiser(P[tr])
        Qs, Ps = (Q - mq) / sq, (P - mp) / sp
        muq, Vq = pca_fit(Qs[tr], q)
        Zq = (Qs - muq) @ Vq
        if conditioned:
            mup, Vp = pca_fit(Ps[tr], q)
            Zp = (Ps - mup) @ Vp
            alpha, oof = inner_oof(Zp[tr], Zq[tr], seed[tr])
            pred_te = Ridge(Zp[tr], Zq[tr]).predict(Zp[te], alpha)
        else:
            alpha, oof = None, np.broadcast_to(Zq[tr].mean(0), Zq[tr].shape).copy()
            pred_te = np.broadcast_to(Zq[tr].mean(0), (len(te), q))
        Rtr = Zq[tr] - oof
        Rte = Zq[te] - pred_te
        # residual PCA on OUT-OF-FOLD honest residuals
        c0 = Rtr.mean(0)
        _, S, Vt = np.linalg.svd(Rtr - c0, full_matrices=False)
        lam = S ** 2 / (len(Rtr) - 1)
        lam = lam + 1e-3 * lam.mean()                    # light shrinkage for the tail
        r_pa = parallel_analysis(Rtr - c0)
        C = (Rte - c0) @ Vt.T                            # coords in the residual basis
        for r in RANKS + ["pa"]:
            rr = r_pa if r == "pa" else r
            out[f"maha_r{r}"][te] = (C[:, rr:] ** 2 / lam[rr:]).sum(1)
            out[f"eucl_r{r}"][te] = (C[:, rr:] ** 2).sum(1) / (Zq[te] ** 2).sum(1)
        out["coords"][te] = C[:, :8]
        # held-out reconstruction (label-free selection criterion), honest only
        sst = ((Zq[te] - Zq[tr].mean(0)) ** 2).sum(1)
        sse = (Rte ** 2).sum(1)
        out["r2_recon"][te] = 1 - sse / np.where(sst > 0, sst, np.nan)
        out["Zq"][te] = Zq[te]
        info.append({"alpha": alpha, "r_pa": int(r_pa),
                     "residual_var_top8": (lam[:8] / lam.sum()).tolist()})
        # direction-of-violation needs the basis; keep the last fold's for twins in te
        out.setdefault("_bases", []).append((te, Vt[:8], lam))
    return out, info


def _cell(args):
    key, rung, site, post_pool, conditioned = args
    full, trunc, honest, seed, folds, q = _G[key]
    Q, vq = full.features(site, post_pool)
    if conditioned:
        P, vp = trunc[rung].features(site, PRE_POOL)
    else:
        P, vp = Q, vq
    out, info = fit_score(P, Q, honest, seed, folds, q, conditioned)
    return (rung, site, post_pool, conditioned), out, info


# --------------------------------------------------------------------------- #
# evaluation (labels enter here and only here)
# --------------------------------------------------------------------------- #


def evaluate(score, viol, pairs):
    d = np.array([score[v] - score[h] for h, v in pairs])
    return {"auc": fast_auc(score, viol), "paired": sign_test(d)}


def paired_flip_null(score_mat: np.ndarray, pairs, draws=200, seed=0):
    """Grid-max null: for each draw flip which twin is 'violating', per pair, independently.

    ``score_mat`` is [cells, n]. Returns the null distribution of the max paired
    fraction over cells (the AUC on a twin design moves with the same flips).
    """
    rng = np.random.default_rng(seed)
    h = np.array([p[0] for p in pairs]); v = np.array([p[1] for p in pairs])
    D = score_mat[:, v] - score_mat[:, h]                 # [cells, pairs]
    nz = D != 0
    null = []
    for _ in range(draws):
        s = rng.choice([-1.0, 1.0], size=D.shape[1])
        frac = ((D * s) > 0).sum(1) / nz.sum(1)
        null.append(frac.max())
    return np.array(null)


def twin_direction(rec, pairs, rank, seed=0, draws=200):
    """Share of ||Zq_v - Zq_h||^2 inside the top-``rank`` residual directions vs random."""
    Zq = rec["Zq"]
    shares, null = [], []
    rng = np.random.default_rng(seed)
    pos = {}
    for te, Vt, lam in rec["_bases"]:
        for i in te:
            pos[i] = Vt[:rank]
    q = Zq.shape[1]
    for h, v in pairs:
        U = pos[h]
        d = Zq[v] - Zq[h]
        shares.append(((U @ d) ** 2).sum() / (d @ d))
    for _ in range(draws):
        R = np.linalg.qr(rng.normal(size=(q, rank)))[0]
        null.append(np.median([((R.T @ (Zq[v] - Zq[h])) ** 2).sum() / ((Zq[v] - Zq[h]) ** 2).sum()
                               for h, v in pairs[:200]]))
    return {"rank": rank, "twin_diff_share_median": float(np.median(shares)),
            "random_subspace_share_median": float(np.mean(null)),
            "random_expected": rank / q}


def meaning(rec, e, honest, cell):
    """Does each top residual coordinate read restitution? Honest held-out clips only."""
    out = {}
    elastic = e > 0.5
    for j in range(4):
        c = rec["coords"][honest, j]
        a = fast_auc(c, elastic[honest])
        out[f"pc{j + 1}_auc_elastic"] = float(max(a, 1 - a))
    # by material cell, PC1
    for m in ("mixed", "same"):
        mask = honest & np.array([c.endswith(m) for c in cell])
        c = rec["coords"][mask, 0]
        a = fast_auc(c, elastic[mask])
        out[f"pc1_auc_elastic_{m}"] = float(max(a, 1 - a))
    return out


# --------------------------------------------------------------------------- #


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--full", required=True, type=Path)
    p.add_argument("--trunc", required=True, type=Path)
    p.add_argument("--rungs", default="10,14,17,19,20,22,25")
    p.add_argument("--q", type=int, default=64, help="PCA dims on each side")
    p.add_argument("--sites", default=None, help="comma list; default every shared site")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    t0 = time.time()
    rungs = [int(k) for k in args.rungs.split(",")]
    sites = args.sites.split(",") if args.sites else None
    full = load_clipset(args.full, sites=sites)
    trunc = {k: load_clipset(args.trunc / f"k{k}", sites=sites) for k in rungs}
    # align every rung to the full clip order
    ids = [c for c in full.clip_ids if all(c in set(t.clip_ids) for t in trunc.values())]
    full = full.subset(np.isin(full.clip_ids, ids))
    for k, t in trunc.items():
        pos = {c: i for i, c in enumerate(t.clip_ids)}
        order = np.array([pos[c] for c in full.clip_ids])
        trunc[k] = type(t)(root=t.root, clip_ids=[t.clip_ids[i] for i in order],
                           labels=[t.labels[i] for i in order], metas=[t.metas[i] for i in order],
                           pools=t.pools, acts={s: a[order] for s, a in t.acts.items()})
    shared = [s for s in full.sites if all(s in t.acts for t in trunc.values())]
    print(f"[{args.model}] {full.describe()}\n  rungs {rungs}, {len(shared)} shared sites, "
          f"loaded in {time.time() - t0:.0f}s", flush=True)

    honest = ~full.violates
    viol = full.violates
    seed = full.seed
    e = full.restitution
    cell = np.array(["_".join(c.split("_")[:2]) for c in full.clip_ids])
    pairs = []
    for s in np.unique(seed):
        idx = np.flatnonzero(seed == s)
        h = [i for i in idx if honest[i]]; v = [i for i in idx if viol[i]]
        if len(h) == 1 and len(v) == 1:
            pairs.append((h[0], v[0]))
    folds = grouped_folds(seed, 5, seed=0)
    print(f"  {len(pairs)} twin pairs, {int(honest.sum())} honest / {int(viol.sum())} violating", flush=True)

    # twins must encode identically before contact (k <= 20)
    twin_rel = {}
    for k in rungs:
        X, _ = trunc[k].features(shared[len(shared) // 2], PRE_POOL)
        twin_rel[k] = float(np.max([np.linalg.norm(X[h] - X[v]) / np.linalg.norm(X[h]) for h, v in pairs]))

    _G["g"] = (full, trunc, honest, seed, folds, args.q)
    jobs = [("g", k, s, pp, True) for k in rungs for s in shared for pp in POST_POOLS]
    jobs += [("g", None, s, pp, False) for s in shared for pp in POST_POOLS]
    with Pool(args.workers) as pool:
        res = pool.map(_cell, jobs, chunksize=1)
    recs = {key: (out, info) for key, out, info in res}
    print(f"  {len(jobs)} cells fit in {time.time() - t0:.0f}s", flush=True)

    def summary(key):
        out, info = recs[key]
        row = {"r2_recon_honest": float(np.nanmedian(out["r2_recon"][honest])),
               "r_pa_folds": [i["r_pa"] for i in info],
               "residual_var_top8": np.mean([i["residual_var_top8"] for i in info], 0).tolist()}
        for r in RANKS + ["pa"]:
            row[f"maha_r{r}"] = evaluate(out[f"maha_r{r}"], viol, pairs)
            row[f"eucl_r{r}"] = evaluate(out[f"eucl_r{r}"], viol, pairs)
        return row

    result = {"protocol": "unsupervised valid-collision subspace: f(pre)->post on honest clips, "
                          "residual PCA, off-subspace score; labels only for evaluation",
              "model": args.model, "q": args.q, "pre_pool": PRE_POOL, "n": full.n,
              "n_pairs": len(pairs), "twin_pre_activation_max_rel_diff": twin_rel,
              "rungs": {}, "unconditioned": {}}

    # ---- per rung: label-free cell choice, then evaluate ---------------------- #
    def per_block(cond_rung):
        cond = cond_rung is not None
        keys = [kk for kk in recs if kk[0] == cond_rung and kk[3] == cond]
        rows = {f"{kk[2]} @ {kk[1]}": summary(kk) for kk in keys}
        best = max(keys, key=lambda kk: rows[f"{kk[2]} @ {kk[1]}"]["r2_recon_honest"])
        out, _ = recs[best]
        name = f"{best[2]} @ {best[1]}"
        r_pa = int(np.median(rows[name]["r_pa_folds"]))
        blk = {
            "chosen_cell_label_free": name,
            "chosen": rows[name],
            "meaning": meaning(out, e, honest, cell),
            "twin_direction": twin_direction(out, pairs, max(1, r_pa)),
            "by_cell_maha_pa": {c: evaluate(out["maha_rpa"], viol, [pr for pr in pairs if cell[pr[0]] == c])["paired"]
                                for c in np.unique(cell)},
            "all_cells": rows,
        }
        # grid-max null over every cell at this rung, primary score
        mat = np.stack([recs[kk][0]["maha_rpa"] for kk in keys])
        null = paired_flip_null(mat, pairs)
        obs_max = max(rows[n_]["maha_rpa"]["paired"]["fraction"] for n_ in rows)
        blk["grid_max_paired_fraction"] = {"observed": obs_max, "null_p95": float(np.quantile(null, 0.95)),
                                           "p": float((1 + (null >= obs_max).sum()) / (len(null) + 1))}
        c = rows[name]
        print(f"  {'k=' + str(cond_rung) if cond else 'UNCOND':7s} {name:28s} recon R2 {c['r2_recon_honest']:+.3f}  "
              f"r_pa {c['r_pa_folds']}  maha(pa) AUC {c['maha_rpa']['auc']:.3f} paired "
              f"{c['maha_rpa']['paired']['positive']}/{c['maha_rpa']['paired']['n']} p={c['maha_rpa']['paired']['p_one_sided']:.1e}  "
              f"r0 {c['maha_r0']['auc']:.3f}  eucl(pa) {c['eucl_rpa']['auc']:.3f}  "
              f"PC1~e {blk['meaning']['pc1_auc_elastic']:.3f}  twin-in-U {blk['twin_direction']['twin_diff_share_median']:.3f} "
              f"(rand {blk['twin_direction']['random_subspace_share_median']:.3f})  "
              f"grid-max {obs_max:.3f} vs null95 {blk['grid_max_paired_fraction']['null_p95']:.3f}", flush=True)
        return blk

    for k in rungs:
        result["rungs"][str(k)] = per_block(k)
    result["unconditioned"] = per_block(None)
    cli.write_artifact(result, args.out, f"momentum_unsup_subspace_{args.model}.json")
    print(f"  total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
