"""Grouped-CV ridge with every alpha read off one eigendecomposition per split.

Same protocol as ``physics_interp.momentum.stats.cv_predict`` — grouped outer
folds, alpha chosen on an inner grouped split of the training rows by pooled
R^2, standardisation fit on the rows being trained — but the dual Gram matrix
is decomposed once per split and every alpha costs a diagonal rescale. That is
what makes a full (site x pool) grid on the 41-site giant captures affordable.

One difference, deliberate: a feature that is constant over the training rows
is zeroed rather than divided by ``std + 1e-6`` (§44e — the guard turns float
noise into a ranking feature).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from physics_interp.momentum.stats import grouped_folds, r2

ALPHAS = (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4)


def _standardise(X: np.ndarray, rows: np.ndarray):
    mu = X[rows].mean(0)
    sd = X[rows].std(0)
    live = sd > 1e-8
    scale = np.where(live, 1.0 / np.where(live, sd, 1.0), 0.0)
    return mu, scale


class DualRidge:
    """Ridge fit on ``train`` rows, evaluable at any alpha without refitting."""

    def __init__(self, X: np.ndarray, Y: np.ndarray, train: np.ndarray):
        self.mu, self.scale = _standardise(X, train)
        self.Xs = (X[train] - self.mu) * self.scale
        self.y_mean = Y[train].mean(0)
        G = self.Xs @ self.Xs.T
        self.evals, self.U = np.linalg.eigh(G)
        self.evals = np.clip(self.evals, 0.0, None)
        self.UtY = self.U.T @ (Y[train] - self.y_mean)

    def predict(self, X: np.ndarray, rows: np.ndarray, alphas: Sequence[float]) -> list[np.ndarray]:
        K = ((X[rows] - self.mu) * self.scale) @ self.Xs.T
        KU = K @ self.U
        return [KU @ (self.UtY / (self.evals + a)[:, None]) + self.y_mean for a in alphas]

    def weights(self, alpha: float) -> np.ndarray:
        """Primal ``W [d, k]`` in standardised feature units."""
        return self.Xs.T @ (self.U @ (self.UtY / (self.evals + alpha)[:, None]))


def cv_predict(X: np.ndarray, Y: np.ndarray, groups: Sequence, *, n_folds: int = 5,
               alphas: Sequence[float] = ALPHAS, seed: int = 0,
               fit_mask: np.ndarray | None = None, return_alphas: bool = False):
    """Out-of-fold predictions; every row is predicted by a readout that never saw its group."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    n = X.shape[0]
    g = np.asarray(groups)
    fit_mask = np.ones(n, bool) if fit_mask is None else np.asarray(fit_mask, bool)
    pred = np.full_like(Y, np.nan)
    chosen = []
    for test in grouped_folds(g, n_folds, seed):
        train = np.setdiff1d(np.arange(n), test)
        train = train[fit_mask[train]]
        if len(train) < 8:
            raise ValueError("fewer than 8 training rows in a fold")
        best = select_alpha(X, Y, g, train, alphas=alphas, seed=seed + 1)
        chosen.append(best)
        pred[test] = DualRidge(X, Y, train).predict(X, test, [best])[0]
    return (pred, chosen) if return_alphas else pred


def select_alpha(X: np.ndarray, Y: np.ndarray, g: np.ndarray, train: np.ndarray, *,
                 alphas: Sequence[float] = ALPHAS, seed: int = 1) -> float:
    """Alpha by pooled R^2 on an inner grouped split of ``train`` only."""
    inner = grouped_folds(g[train], n_folds=min(4, len(np.unique(g[train]))), seed=seed)
    score = np.zeros(len(alphas))
    for val_local in inner:
        val = train[val_local]
        fit = np.setdiff1d(train, val)
        for ai, p in enumerate(DualRidge(X, Y, fit).predict(X, val, alphas)):
            score[ai] += r2(Y[val], p)
    return float(alphas[int(np.nanargmax(score))])


def fit_full(X: np.ndarray, Y: np.ndarray, g: np.ndarray, rows: np.ndarray, *,
             alphas: Sequence[float] = ALPHAS, seed: int = 1) -> tuple[DualRidge, float]:
    """One readout on ``rows`` (alpha chosen inside them), for transfer to other rows."""
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    best = select_alpha(X, Y, np.asarray(g), rows, alphas=alphas, seed=seed)
    return DualRidge(X, Y, rows), best
