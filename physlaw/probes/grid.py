"""Batched form of ``ridge.fit_probe`` for many targets at once.

Same protocol and the same folds (``grouped_kfold`` with the same seeds, inner folds with
``seed + 10_000``, nested alpha by per-output-standardised R^2 per target), so a single-target call returns
exactly what ``fit_probe`` would. What changes is cost: the ridge is solved in dual form from ONE
eigendecomposition of the training Gram matrix per split, and that decomposition is shared by every
alpha and every target block. ``tests/test_probes.py`` checks equality with the reference function.

Targets are passed as ``blocks``: name -> [n, k] array; a block is scored by R^2 on its k columns standardised per output
(the mean per-output R^2) and gets its own alpha (a 2-D velocity is one block).
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .ridge import DEFAULT_ALPHAS, feature_scale, grouped_kfold


def _r2_cols(Y: np.ndarray, P: np.ndarray, slices: list[slice]) -> np.ndarray:
    """R^2 per block on targets standardised per output (ridge.r2_score: the mean of the block's per-output R^2 over
    outputs with non-zero variance). Y, P [n, m]; returns [n_blocks]."""
    sse = ((Y - P) ** 2).sum(0)
    sst = ((Y - Y.mean(0)) ** 2).sum(0)
    out = np.empty(len(slices))
    for i, s in enumerate(slices):
        ok = sst[s] > 0
        out[i] = float(np.mean(1.0 - sse[s][ok] / sst[s][ok])) if ok.any() else np.nan
    return out


class _Dual:
    """Standardise X on train rows, eigh the Gram once; predict any (alpha, Y) cheaply."""

    def __init__(self, X: np.ndarray, tr: np.ndarray, standardize: bool = True):
        mx = X[tr].mean(0)
        sx = feature_scale(X[tr]) if standardize else np.ones(X.shape[1])
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
               standardize: bool = True) -> dict:
    """Full nested protocol for every block on one feature matrix (one grid cell).

    Returns {block: {"score_mean", "score_sd", "scores", "alphas_chosen",
                     "oof": [fold_seeds, n, k] out-of-fold predictions per fold seed (for held-out selection)}}.
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
    oof_all = np.zeros((fold_seeds, *Y.shape))
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
        oof_all[seed] = oof

    return {b: {"score_mean": float(scores[:, i].mean()), "score_sd": float(scores[:, i].std()),
                "scores": scores[:, i].tolist(), "alphas_chosen": chosen[i], "oof": oof_all[:, :, slices[i]]}
            for i, b in enumerate(names)}

