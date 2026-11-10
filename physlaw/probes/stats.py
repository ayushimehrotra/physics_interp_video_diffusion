"""Statistics shared by the momentum protocols.

Three rules, enforced here so no script can drift from them:

* **Raw AUC over every clip** is the detection statistic (§55). Paired accuracy
  — does the violating clip beat its own twin — is not implemented anywhere.
* **Grouped folds by seed.** Twins and ladder arms share a bit-identical
  pre-contact segment; a twin in training lets a readout memorise its partner
  (0.787 on leak-free data, §55). :func:`grouped_folds` never splits a seed.
* **The ridge penalty is chosen inside the training split**, on an inner
  grouped split, never on the fold being scored.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np

from ..interp import ridge_fit

__all__ = [
    "auc",
    "permutation_null",
    "bootstrap_ci",
    "grouped_folds",
    "cv_predict",
    "r2",
    "spearman",
    "participation_ratio",
    "pca",
    "random_cosine_reference",
    "polyfit_curve",
    "eval_curve",
    "nearest_on_curve",
]


# --------------------------------------------------------------------------- #
# detection statistics
# --------------------------------------------------------------------------- #


def auc(scores: np.ndarray, positives: np.ndarray) -> float:
    """Raw AUC: P(score of a random positive > score of a random negative), ties 1/2.

    Mann-Whitney form, O(n log n), exact with ties.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(positives).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    # average ranks for ties
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def permutation_null(scores: np.ndarray, positives: np.ndarray, *, draws: int = 4000,
                     seed: int = 0, groups: np.ndarray | None = None) -> dict[str, float]:
    """Label-permutation null for :func:`auc`.

    Shuffles the labels ``draws`` times and reports how often chance reaches the
    observed AUC (one-sided, ``>=``, with the +1 correction so p is never 0
    exactly — the register's ``p = 0.0000`` is this quantity rounded).

    With ``groups`` (one id per clip) labels are permuted *within* each group,
    which is the twin-exchangeability null the register keeps (§55 "not
    removed") — it asks whether the score tells the twins apart, holding the
    scene fixed. Without it the null is the plain label shuffle.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(positives).astype(bool)
    observed = auc(scores, y)
    if not np.isfinite(observed):
        return {"auc": observed, "p": float("nan"), "null_mean": float("nan"),
                "null_sd": float("nan"), "draws": draws}
    null = np.empty(draws)
    if groups is None:
        for k in range(draws):
            null[k] = auc(scores, rng.permutation(y))
    else:
        g = np.asarray(groups)
        idx_by_group = [np.flatnonzero(g == u) for u in np.unique(g)]
        yp = y.copy()
        for k in range(draws):
            for idx in idx_by_group:
                yp[idx] = y[rng.permutation(idx)]
            null[k] = auc(scores, yp)
    return {
        "auc": observed,
        "p": float((1 + np.sum(null >= observed)) / (draws + 1)),
        "null_mean": float(null.mean()),
        "null_sd": float(null.std()),
        "draws": int(draws),
    }


def bootstrap_ci(values: np.ndarray, stat: Callable[[np.ndarray], float] = np.median, *,
                 draws: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float, float]:
    """Percentile bootstrap of ``stat`` over the rows of ``values``. Returns ``(point, lo, hi)``."""
    v = np.asarray(values)
    rng = np.random.default_rng(seed)
    n = len(v)
    boots = np.array([stat(v[rng.integers(0, n, n)]) for _ in range(draws)])
    a = (1 - level) / 2
    return float(stat(v)), float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


# --------------------------------------------------------------------------- #
# folds and readouts
# --------------------------------------------------------------------------- #


def grouped_folds(groups: Sequence, n_folds: int = 5, seed: int = 0) -> list[np.ndarray]:
    """Test-index arrays for K folds that never split a group.

    Groups are shuffled, then dealt round-robin so the folds are balanced in
    group count. Leave-one-group-out is ``n_folds = number of groups``.
    """
    g = np.asarray(groups)
    uniq = np.unique(g)
    rng = np.random.default_rng(seed)
    uniq = uniq[rng.permutation(len(uniq))]
    n_folds = min(n_folds, len(uniq))
    folds = []
    for k in range(n_folds):
        members = uniq[k::n_folds]
        folds.append(np.flatnonzero(np.isin(g, members)))
    return folds


def _standardise(X: np.ndarray, idx: np.ndarray):
    mu = X[idx].mean(0)
    sd = X[idx].std(0) + 1e-6
    return mu, sd


def cv_predict(X: np.ndarray, Y: np.ndarray, groups: Sequence, *, n_folds: int = 5,
               alphas: Sequence[float] = (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4),
               seed: int = 0, fit_mask: np.ndarray | None = None,
               return_alphas: bool = False):
    """Out-of-fold ridge predictions with grouped folds and inner alpha selection.

    ``X [n, d]``, ``Y [n, k]``. Every clip receives a prediction from a readout
    that never saw its seed. ``fit_mask`` restricts the *training* rows (e.g.
    honest clips only, §49 ``--fit-honest``); every row is still predicted.

    The penalty is selected per outer fold on an inner grouped split of that
    fold's training rows, by pooled R^2 — the fold being scored is never
    consulted.
    """
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
        inner = grouped_folds(g[train], n_folds=min(4, len(np.unique(g[train]))), seed=seed + 1)
        score = np.zeros(len(alphas))
        for val_local in inner:
            val = train[val_local]
            fit = np.setdiff1d(train, val)
            mu, sd = _standardise(X, fit)
            Xf = (X[fit] - mu) / sd
            Xv = (X[val] - mu) / sd
            for ai, a in enumerate(alphas):
                W, b = ridge_fit(Xf, Y[fit], a)
                score[ai] += r2(Y[val], Xv @ W + b)
        best = float(alphas[int(np.nanargmax(score))])
        chosen.append(best)
        mu, sd = _standardise(X, train)
        W, b = ridge_fit((X[train] - mu) / sd, Y[train], best)
        pred[test] = (X[test] - mu) / sd @ W + b
    if return_alphas:
        return pred, chosen
    return pred


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pooled multi-output R^2 (see :func:`physics_interp.interp._r2`)."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.ndim == 1:
        yt, yp = yt[:, None], yp[:, None]
    ss_res = float(((yt - yp) ** 2).sum())
    ss_tot = float(((yt - yt.mean(0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# --------------------------------------------------------------------------- #
# geometry helpers (§54b, §56, §60)
# --------------------------------------------------------------------------- #


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = _rank(np.asarray(x, float)), _rank(np.asarray(y, float))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def pca(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre ``X [n, d]`` and return ``(scores [n, r], components [r, d], variances [r])``."""
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / max(len(X) - 1, 1)
    return U * S, Vt, var


def participation_ratio(variances: np.ndarray) -> float:
    """``(sum l)^2 / sum l^2`` — 1 for one direction, r for r equal directions."""
    v = np.asarray(variances, dtype=np.float64)
    return float(v.sum() ** 2 / (v ** 2).sum()) if (v ** 2).sum() > 0 else float("nan")


def random_cosine_reference(dim: int, draws: int = 20000, seed: int = 0) -> float:
    """Expected ``|cos|`` between two random directions in ``dim`` dimensions."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(draws, dim))
    b = rng.normal(size=(draws, dim))
    c = np.einsum("ij,ij->i", a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return float(np.abs(c).mean())


def polyfit_curve(e: np.ndarray, X: np.ndarray, degree: int) -> np.ndarray:
    """Fit ``X[i] ~ poly(e[i])`` per dimension. Returns coefficients ``[degree + 1, d]``.

    ``degree`` is lowered when there are too few knots for it, so a seed with
    three legal arms gets a line rather than an error; callers that need a
    quadratic check ``len(e) >= 3`` themselves.
    """
    e = np.asarray(e, dtype=np.float64)
    degree = min(degree, len(e) - 1)
    V = np.vander(e, degree + 1, increasing=True)
    coef, *_ = np.linalg.lstsq(V, X, rcond=None)
    return coef


def eval_curve(coef: np.ndarray, e) -> np.ndarray:
    e = np.atleast_1d(np.asarray(e, dtype=np.float64))
    V = np.vander(e, coef.shape[0], increasing=True)
    return V @ coef


def nearest_on_curve(coef: np.ndarray, x: np.ndarray, e_lo: float, e_hi: float,
                     grid: int = 401, refine: int = 3) -> tuple[float, float]:
    """``min_e |x - curve(e)|`` over ``[e_lo, e_hi]`` by grid search with refinement.

    Returns ``(distance, e_at_min)``. A move along the family costs nothing (§60).
    """
    lo, hi = float(e_lo), float(e_hi)
    best_e, best_d = lo, float("inf")
    for _ in range(refine):
        es = np.linspace(lo, hi, grid)
        d = np.linalg.norm(eval_curve(coef, es) - x[None, :], axis=1)
        k = int(np.argmin(d))
        best_e, best_d = float(es[k]), float(d[k])
        step = (hi - lo) / (grid - 1)
        lo, hi = max(e_lo, best_e - 2 * step), min(e_hi, best_e + 2 * step)
    return best_d, best_e
