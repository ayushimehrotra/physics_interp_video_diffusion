"""Batched form of ``ridge.fit_probe`` + ``ridge.permutation_null`` for many targets at once.

Same protocol and the same folds (``grouped_kfold`` with the same seeds, inner folds with
``seed + 10_000``, nested alpha by pooled R^2 per target), so a single-target call returns
exactly what ``fit_probe`` / ``permutation_null`` would. What changes is cost: the ridge is
solved in dual form from ONE eigendecomposition of the training Gram matrix per split, and
that decomposition is shared by every alpha, every target block and every permuted copy of
the targets. ``tests/test_grid_probe.py`` checks equality with the reference functions.

Targets are passed as ``blocks``: name -> [n, k] array; a block is scored by pooled R^2 over
its k columns and gets its own alpha (a 2-D velocity is one block).

Permutation null (spec §5): ``perms`` is [P, n] row permutations (draw them at the GROUP level
with ``group_block_perms`` so within-group twin structure survives). For each draw the caller
takes the max over the searched grid; this module returns the per-cell scores it needs.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .ridge import DEFAULT_ALPHAS, grouped_kfold


def _r2_cols(Y: np.ndarray, P: np.ndarray, slices: list[slice]) -> np.ndarray:
    """Pooled R^2 per block. Y, P [n, m]; returns [n_blocks]."""
    sse = ((Y - P) ** 2).sum(0)
    sst = ((Y - Y.mean(0)) ** 2).sum(0)
    out = np.empty(len(slices))
    for i, s in enumerate(slices):
        t = sst[s].sum()
        out[i] = 1.0 - sse[s].sum() / t if t > 0 else np.nan
    return out


class _Dual:
    """Standardise X on train rows, eigh the Gram once; predict any (alpha, Y) cheaply."""

    def __init__(self, X: np.ndarray, tr: np.ndarray, standardize: bool = True):
        mx = X[tr].mean(0)
        sx = X[tr].std(0) if standardize else np.ones(X.shape[1])
        sx = np.where(sx == 0, 1.0, sx)
        self.mx, self.sx, self.tr = mx, sx, tr
        self.Xs = (X[tr] - mx) / sx
        ev, self.U = np.linalg.eigh(self.Xs @ self.Xs.T)
        self.ev = np.clip(ev, 0.0, None)

    def kernel(self, X: np.ndarray, rows: np.ndarray) -> np.ndarray:
        return (((X[rows] - self.mx) / self.sx) @ self.Xs.T) @ self.U          # [n_te, n_tr]

    def predict(self, KU: np.ndarray, Ytr: np.ndarray, alpha: float) -> np.ndarray:
        my = Ytr.mean(0)
        return KU @ ((self.U.T @ (Ytr - my)) / (self.ev + alpha)[:, None]) + my


def _cv_all_alphas(X, Y, groups, rows, alphas, k, fold_seed, standardize):
    """Out-of-fold predictions on ``rows`` for every alpha: [A, len(rows), m]."""
    out = np.zeros((len(alphas), len(rows), Y.shape[1]))
    for te_l in grouped_kfold(groups[rows], k, fold_seed):
        tr_l = np.setdiff1d(np.arange(len(rows)), te_l)
        d = _Dual(X, rows[tr_l], standardize)
        KU = d.kernel(X, rows[te_l])
        my = Y[rows[tr_l]].mean(0)
        UtY = d.U.T @ (Y[rows[tr_l]] - my)
        for ai, a in enumerate(alphas):
            out[ai, te_l] = KU @ (UtY / (d.ev + a)[:, None]) + my
    return out


def _pick(Y, preds, slices):
    """Per block, index of the best alpha (first max, like ``max`` over an ordered dict)."""
    sc = np.stack([_r2_cols(Y, p, slices) for p in preds])               # [A, B]
    sc = np.where(np.isnan(sc), -np.inf, sc)
    return sc.argmax(0)


def grid_probe(X: np.ndarray, blocks: Mapping[str, np.ndarray], groups: np.ndarray, *,
               alphas: Sequence[float] = DEFAULT_ALPHAS, k: int = 5, fold_seeds: int = 12,
               standardize: bool = True, perms: np.ndarray | None = None) -> dict:
    """Full nested protocol for every block on one feature matrix (one grid cell).

    Returns {block: {"score_mean", "score_sd", "scores", "alphas_chosen",
                     "quick": observed single-seed score at the null's alpha,
                     "null": [P] scores under the permutations (if perms given)}}.
    """
    X = np.asarray(X, np.float64)
    groups = np.asarray(groups)
    names = list(blocks)
    Ys = [np.asarray(blocks[b], np.float64).reshape(len(X), -1) for b in names]
    widths = [y.shape[1] for y in Ys]
    edges = np.cumsum([0] + widths)
    slices = [slice(edges[i], edges[i + 1]) for i in range(len(names))]
    Y = np.concatenate(Ys, 1)
    n, B = len(X), len(names)
    allr = np.arange(n)

    scores = np.zeros((fold_seeds, B))
    chosen = [[] for _ in range(B)]
    for seed in range(fold_seeds):
        oof = np.zeros_like(Y)
        for te in grouped_kfold(groups, k, seed):
            tr = np.setdiff1d(allr, te)
            inner = _cv_all_alphas(X, Y, groups, tr, alphas, k, seed + 10_000, standardize)
            best = _pick(Y[tr], inner, slices)
            d = _Dual(X, tr, standardize)
            KU = d.kernel(X, te)
            for ai in np.unique(best):
                cols = np.concatenate([np.arange(slices[b].start, slices[b].stop) for b in np.where(best == ai)[0]])
                oof[np.ix_(te, cols)] = d.predict(KU, Y[np.ix_(tr, cols)], alphas[ai])
            for b in range(B):
                chosen[b].append(float(alphas[best[b]]))
        scores[seed] = _r2_cols(Y, oof, slices)

    # reduced protocol of permutation_null: alpha by inner CV on all rows (seed 0 -> 10_000),
    # one fold seed (0), fixed alpha; observed and null use the same.
    a_null = _pick(Y, _cv_all_alphas(X, Y, groups, allr, alphas, k, 10_000, standardize), slices)
    quick = np.zeros(B)
    null = np.zeros((0 if perms is None else len(perms), B))
    oof_q = np.zeros_like(Y)
    if perms is not None:
        P = len(perms)
        Yp = np.concatenate([Y[p] for p in perms], 1)                   # [n, P*m]
        oof_p = np.zeros_like(Yp)
        m = Y.shape[1]
    for te in grouped_kfold(groups, k, 0):
        tr = np.setdiff1d(allr, te)
        d = _Dual(X, tr, standardize)
        KU = d.kernel(X, te)
        for ai in np.unique(a_null):
            bs = np.where(a_null == ai)[0]
            cols = np.concatenate([np.arange(slices[b].start, slices[b].stop) for b in bs])
            oof_q[np.ix_(te, cols)] = d.predict(KU, Y[np.ix_(tr, cols)], alphas[ai])
            if perms is not None:
                pc = (np.arange(P)[:, None] * m + cols[None]).ravel()
                oof_p[np.ix_(te, pc)] = d.predict(KU, Yp[np.ix_(tr, pc)], alphas[ai])
    quick = _r2_cols(Y, oof_q, slices)
    if perms is not None:
        for i in range(P):
            sl = slice(i * m, (i + 1) * m)
            null[i] = _r2_cols(Yp[:, sl], oof_p[:, sl], slices)

    return {b: {"score_mean": float(scores[:, i].mean()), "score_sd": float(scores[:, i].std()),
                "scores": scores[:, i].tolist(), "alphas_chosen": chosen[i],
                "alpha_null": float(alphas[a_null[i]]), "quick": float(quick[i]),
                "null": null[:, i]}
            for i, b in enumerate(names)}


def group_block_perms(groups: np.ndarray, twins: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """[n, len(groups)] row permutations that shuffle whole groups and keep each twin slot:
    row (g, t) takes the target of row (pi(g), t). Requires every group to hold the same twin set."""
    groups, twins = np.asarray(groups), np.asarray(twins)
    ug = np.unique(groups)
    index = {(g, t): i for i, (g, t) in enumerate(zip(groups, twins))}
    rng = np.random.default_rng(seed)
    out = np.empty((n, len(groups)), int)
    for j in range(n):
        pi = dict(zip(ug, rng.permutation(ug)))
        out[j] = [index[(pi[g], t)] for g, t in zip(groups, twins)]
    return out


def grid_max_p(observed: Mapping[str, float], null: Mapping[str, np.ndarray]) -> tuple[str, float, np.ndarray]:
    """Best cell by observed score, and its p against the max-over-cells null distribution."""
    cells = list(observed)
    best = max(cells, key=lambda c: -np.inf if np.isnan(observed[c]) else observed[c])
    null_max = np.nanmax(np.stack([null[c] for c in cells]), 0)
    p = (1 + np.sum(null_max >= observed[best])) / (len(null_max) + 1)
    return best, float(p), null_max
