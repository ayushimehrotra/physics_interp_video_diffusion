"""Ridge probes with grouped CV, nested penalty selection and grid-max permutation nulls (spec §5).

Protocol
--------
* Ridge from a pooled activation vector to a target (multi-output allowed).
  Features and targets are standardised on the TRAINING fold only.
* Outer: grouped K-fold by twin group ID (K = 5), repeated over ``fold_seeds``
  seeds; every clip gets one out-of-fold prediction per seed.
* Inner: the penalty is chosen by grouped K-fold on the outer training fold only.
* Score: out-of-fold R^2 (continuous) or AUC of the ridge score (binary),
  mean +- sd over fold seeds.
* Null: refit on shuffled targets ``n_permutations`` times; for each draw take the
  MAX score over the whole searched grid (layers x pools x noise levels), and
  compare the observed grid max against that distribution. This is the
  winner's-curse correction the old project's §2 found necessary.

Solved through an SVD of the centred training matrix, so all penalties cost one
decomposition per fold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

DEFAULT_ALPHAS = (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5)


# --------------------------------------------------------------------------- #
# folds and scores
# --------------------------------------------------------------------------- #


def grouped_kfold(groups: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    """Test-index arrays; whole groups per fold, groups shuffled by ``seed``."""
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    order = rng.permutation(uniq)
    fold_of = {g: i % k for i, g in enumerate(order)}
    f = np.array([fold_of[g] for g in groups])
    return [np.where(f == i)[0] for i in range(k)]


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    """Pooled R^2 over all outputs: 1 - SSE / SST (targets centred per output)."""
    y, yhat = np.atleast_2d(y.T).T, np.atleast_2d(yhat.T).T
    sse = ((y - yhat) ** 2).sum()
    sst = ((y - y.mean(0)) ** 2).sum()
    return float(1.0 - sse / sst) if sst > 0 else float("nan")


def auc_score(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUC with tie correction; y in {0, 1}."""
    y = np.asarray(y).ravel().astype(bool)
    s = np.asarray(s).ravel()
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks); ranks = (sums / cnt)[inv]
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# --------------------------------------------------------------------------- #
# ridge
# --------------------------------------------------------------------------- #


class _SVDRidge:
    """Standardise on train, SVD once, predict for any alpha."""

    def __init__(self, X: np.ndarray, Y: np.ndarray, standardize: bool = True):
        self.mx = X.mean(0)
        self.sx = X.std(0) if standardize else np.ones(X.shape[1])
        self.sx[self.sx == 0] = 1.0
        self.my = Y.mean(0)
        Xs = (X - self.mx) / self.sx
        self.U, self.s, self.Vt = np.linalg.svd(Xs, full_matrices=False)
        self.UtY = self.U.T @ (Y - self.my)

    def coef(self, alpha: float) -> np.ndarray:
        d = self.s / (self.s ** 2 + alpha)
        return self.Vt.T @ (d[:, None] * self.UtY)            # [D, k] in standardised-x units

    def predict(self, X: np.ndarray, alpha: float) -> np.ndarray:
        return ((X - self.mx) / self.sx) @ self.coef(alpha) + self.my


def _score(y, yhat, binary):
    return auc_score(y, yhat[:, 0]) if binary else r2_score(y, yhat)


def _inner_alpha(X, Y, groups, alphas, k, seed, binary, standardize):
    folds = grouped_kfold(groups, k, seed + 10_000)
    preds = {a: np.zeros_like(Y, dtype=np.float64) for a in alphas}
    for te in folds:
        tr = np.setdiff1d(np.arange(len(Y)), te)
        m = _SVDRidge(X[tr], Y[tr], standardize)
        for a in alphas:
            preds[a][te] = m.predict(X[te], a)
    scores = {a: _score(Y, preds[a], binary) for a in alphas}
    return max(scores, key=lambda a: -np.inf if np.isnan(scores[a]) else scores[a])


@dataclass
class ProbeResult:
    score_mean: float
    score_sd: float
    scores: list[float]
    alphas_chosen: list[float]
    oof: np.ndarray                    # [n_seeds, n, k] out-of-fold predictions
    metric: str
    direction: np.ndarray | None = None  # full-data coefficient (standardised x units), first output
    extra: dict = field(default_factory=dict)


def fit_probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray, *, binary: bool = False,
              alphas: Sequence[float] = DEFAULT_ALPHAS, k: int = 5, fold_seeds: int = 12,
              standardize: bool = True, nested: bool = True) -> ProbeResult:
    X = np.asarray(X, np.float64)
    Y = np.asarray(y, np.float64).reshape(len(X), -1)
    groups = np.asarray(groups)
    oof = np.zeros((fold_seeds, *Y.shape))
    scores, chosen = [], []
    for seed in range(fold_seeds):
        for te in grouped_kfold(groups, k, seed):
            tr = np.setdiff1d(np.arange(len(Y)), te)
            a = (_inner_alpha(X[tr], Y[tr], groups[tr], alphas, k, seed, binary, standardize)
                 if nested and len(alphas) > 1 else alphas[0])
            chosen.append(a)
            oof[seed, te] = _SVDRidge(X[tr], Y[tr], standardize).predict(X[te], a)
        scores.append(_score(Y, oof[seed], binary))
    a_full = float(np.median(chosen))
    direction = _SVDRidge(X, Y, standardize).coef(a_full)[:, 0]
    return ProbeResult(float(np.mean(scores)), float(np.std(scores)), scores, chosen, oof,
                       "auc" if binary else "r2", direction)


# --------------------------------------------------------------------------- #
# grid-max permutation null
# --------------------------------------------------------------------------- #


def permutation_null(grid: Mapping[str, np.ndarray], y: np.ndarray, groups: np.ndarray, *,
                     binary: bool = False, n_permutations: int = 500, seed: int = 0,
                     alphas: Sequence[float] = DEFAULT_ALPHAS, k: int = 5,
                     standardize: bool = True) -> dict:
    """Observed best cell vs the distribution of the grid-max under shuffled targets.

    ``grid`` maps a cell name (e.g. "layer12/roi_A.post/snr0") to its [n, D] features.
    The null uses one fold seed and a fixed alpha per cell (the cell's own nested
    choice on the real labels) to keep 500 x |grid| refits affordable; the
    observed scores use the same reduced protocol so the comparison is fair.
    """
    Y = np.asarray(y, np.float64).reshape(len(groups), -1)
    rng = np.random.default_rng(seed)

    def quick(X, Yv, alpha):
        oof = np.zeros_like(Yv)
        for te in grouped_kfold(groups, k, 0):
            tr = np.setdiff1d(np.arange(len(Yv)), te)
            oof[te] = _SVDRidge(X[tr], Yv[tr], standardize).predict(X[te], alpha)
        return _score(Yv, oof, binary)

    cell_alpha = {c: _inner_alpha(np.asarray(X, np.float64), Y, groups, alphas, k, 0, binary, standardize)
                  for c, X in grid.items()}
    observed = {c: quick(np.asarray(X, np.float64), Y, cell_alpha[c]) for c, X in grid.items()}
    best = max(observed, key=observed.get)
    null_max = np.empty(n_permutations)
    for i in range(n_permutations):
        Yp = Y[rng.permutation(len(Y))]
        null_max[i] = max(quick(np.asarray(X, np.float64), Yp, cell_alpha[c]) for c, X in grid.items())
    p = (1 + np.sum(null_max >= observed[best])) / (n_permutations + 1)
    return {"observed": observed, "best_cell": best, "best_score": observed[best],
            "null_max": null_max.tolist(), "p_corrected": float(p), "alphas": cell_alpha}
