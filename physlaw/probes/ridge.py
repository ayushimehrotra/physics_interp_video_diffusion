"""Ridge probes with grouped CV and nested penalty selection (spec §5), and the R1
same-instrument decoder (spec §5.0.3: :func:`fit_instrument`, :func:`instrument_folds`, :func:`apply_instrument`,
:func:`r1_decode`) shared by L0, L1, L3 and L4.

Protocol
--------
* Ridge from a pooled activation vector to a target (multi-output allowed).
  Features are standardised and targets centred on the TRAINING fold only. Ridge solves each output column
  independently and linearly in it, so scaling a target column leaves its predictions (in its own units) unchanged;
  what scaling would change is the score. R² of a multi-output block is therefore computed on targets STANDARDISED
  PER OUTPUT (:func:`r2_score`: the mean of the per-output R², outputs with zero variance skipped), so a block's R²
  -- and the penalty / cell chosen by it -- does not weight its outputs by their variance (a 3-D velocity whose z
  component carries most of the variance is not scored on z alone).
* Outer: grouped K-fold by twin group ID (K = 5), repeated over ``fold_seeds``
  seeds; every clip gets one out-of-fold prediction per seed.
* Inner: the penalty is chosen by grouped K-fold on the outer training fold only.
* Score: out-of-fold R^2 (continuous) or AUC of the ridge score (binary),
  mean +- sd over fold seeds. Binary: every fold's scores are centred on that
  fold's training-mean score (its intercept, the training class balance) before
  the folds are pooled into one AUC; otherwise fold-to-fold intercept shifts leak
  into the pooled ranking and push the AUC below chance when fold class balance
  varies. ``oof`` holds these centred scores.
* Near-constant features (sd <= SD_REL_TOL x the largest feature sd on the
  training fold) are zeroed instead of being divided by a tiny sd.
* No permutation null (cut 2026-11-11): the winner's curse over the searched grid is handled by
  held-out selection (``probes.select``).

Solved through an SVD of the centred training matrix, so all penalties cost one
decomposition per fold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

DEFAULT_ALPHAS = (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5)
SD_REL_TOL = 1e-8


def feature_scale(X: np.ndarray) -> np.ndarray:
    """Per-column sd for standardising X, with near-constant columns (sd <= SD_REL_TOL * max sd, or every column
    when all are constant) set to inf so they standardise to exactly 0 rather than blowing up rounding noise."""
    sx = X.std(0)
    return np.where(sx > SD_REL_TOL * (sx.max() if sx.size else 0.0), sx, np.inf)


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
    """R^2 over all outputs on targets standardised per output: the mean over outputs with non-zero variance of
    1 - SSE_j / SST_j (= the pooled 1 - SSE / SST after dividing each output by its sd). One output: the usual R^2.
    NaN when no output varies."""
    y, yhat = np.atleast_2d(y.T).T, np.atleast_2d(yhat.T).T
    sse = ((y - yhat) ** 2).sum(0)
    sst = ((y - y.mean(0)) ** 2).sum(0)
    ok = sst > 0
    return float(np.mean(1.0 - sse[ok] / sst[ok])) if ok.any() else float("nan")


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
        self.sx = feature_scale(X) if standardize else np.ones(X.shape[1])
        self.my = Y.mean(0)
        Xs = (X - self.mx) / self.sx
        self.U, self.s, self.Vt = np.linalg.svd(Xs, full_matrices=False)
        self.UtY = self.U.T @ (Y - self.my)

    def coef(self, alpha: float) -> np.ndarray:
        d = self.s / (self.s ** 2 + alpha)
        return self.Vt.T @ (d[:, None] * self.UtY)            # [D, k] in standardised-x units

    def predict(self, X: np.ndarray, alpha: float, centred: bool = False) -> np.ndarray:
        """Predictions; ``centred`` drops the intercept (the training-mean score), for pooled AUCs."""
        return ((X - self.mx) / self.sx) @ self.coef(alpha) + (0.0 if centred else self.my)


def _score(y, yhat, binary):
    return auc_score(y, yhat[:, 0]) if binary else r2_score(y, yhat)


def _inner_alpha(X, Y, groups, alphas, k, seed, binary, standardize):
    folds = grouped_kfold(groups, k, seed + 10_000)
    preds = {a: np.zeros_like(Y, dtype=np.float64) for a in alphas}
    for te in folds:
        tr = np.setdiff1d(np.arange(len(Y)), te)
        m = _SVDRidge(X[tr], Y[tr], standardize)
        for a in alphas:
            preds[a][te] = m.predict(X[te], a, binary)
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
            oof[seed, te] = _SVDRidge(X[tr], Y[tr], standardize).predict(X[te], a, binary)
        scores.append(_score(Y, oof[seed], binary))
    a_full = float(np.median(chosen))
    direction = _SVDRidge(X, Y, standardize).coef(a_full)[:, 0]
    return ProbeResult(float(np.mean(scores)), float(np.std(scores)), scores, chosen, oof,
                       "auc" if binary else "r2", direction)


# --------------------------------------------------------------------------- #
# R1 instrument (spec §5.0.3): one calibrated linear map, cross-fitted by group
# --------------------------------------------------------------------------- #
#
# The ONE implementation of the same-instrument decoder, used by L0 (same_instrument), L1 (forecast decoding), L3
# (Galilean) and L4 (steered forecasts). An instrument is a ridge map from features to a target, with the cell
# (feature matrix) and penalty chosen by grouped inner CV on its training rows, followed by a per-output INVERSE
# (classical) calibration fitted on the inner out-of-fold predictions of the training rows: y_hat ~ c + d y, decode
# = (y_hat - c) / d. Ridge shrinks toward the mean (y_hat = mu + gamma (y - mu) + noise); the inverse map restores
# the measurement scale, E[decode | y] = y, so ratio statistics are read on the true scale. The forward regression
# y ~ y_hat would instead give E[y | y_hat], which attenuates by ~rho² (rho = corr(y, y_hat)) toward mu: a shrinkage
# toward a non-zero mean that no ratio statistic cancels. The price is noise amplified by 1/d.
# R1 joint: pre- and post-event rows (or every token-frame of a clip) are rows of ONE map; a clip's rows share
# its group, so they always share a fold.


def fit_select(Xc: Mapping[str, np.ndarray], Y: np.ndarray, groups: np.ndarray, rows, alphas=DEFAULT_ALPHAS,
               k: int = 5, seed: int = 0):
    """Choose (cell, alpha) by grouped inner CV (R², standardised per output) on ``rows`` only; refit there. Returns (cell, predict(X),
    inner R², alpha). Fewer than two groups: the first cell and the middle penalty, R² NaN."""
    Y = np.asarray(Y, np.float64).reshape(len(Y), -1)
    rows = np.asarray(rows)
    g = groups[rows]
    best = (-np.inf, None, alphas[len(alphas) // 2])
    if len(set(g)) >= 2:
        folds = grouped_kfold(g, min(k, len(set(g))), seed + 10_000)
        for c, X in Xc.items():
            P = {a: np.zeros((len(rows), Y.shape[1])) for a in alphas}
            for te in folds:
                tr = np.setdiff1d(np.arange(len(rows)), te)
                m = _SVDRidge(X[rows[tr]], Y[rows[tr]], True)
                for a in alphas:
                    P[a][te] = m.predict(X[rows[te]], a)
            for a in alphas:
                s = r2_score(Y[rows], P[a])
                if np.isfinite(s) and s > best[0]:
                    best = (s, c, a)
    s, c, a = best
    c = next(iter(Xc)) if c is None else c
    m = _SVDRidge(np.asarray(Xc[c], np.float64)[rows], Y[rows], True)
    return c, (lambda X, m=m, a=a: m.predict(np.asarray(X, np.float64), a)), (float(s) if np.isfinite(s) else float("nan")), a


def _affine(P: np.ndarray, Y: np.ndarray, groups: np.ndarray, t_min: float = 3.0):
    """Per-output inverse calibration: OLS P_j ~ c_j + d_j Y_j (prediction on truth), decode (Z - c) / d. Only where
    the slope is significantly positive (t = d / se(d) >= ``t_min``, se clustered by group since a group's rows are
    not independent): a probe without signal has d ~ 0, and dividing by it would blow its noise up without bound, so
    such an output keeps the raw ridge prediction (identity map, ~ the training mean). Constant columns: identity."""
    d, c = np.ones(Y.shape[1]), np.zeros(Y.shape[1])
    _, gi = np.unique(groups, return_inverse=True)
    for j in range(Y.shape[1]):
        y, p = Y[:, j], P[:, j]
        if gi.max() < 2 or np.ptp(y) <= 1e-12 or np.ptp(p) <= 1e-12:
            continue
        dj, cj = np.polyfit(y, p, 1)
        yc = y - y.mean()
        se = np.sqrt((np.bincount(gi, yc * (p - cj - dj * y)) ** 2).sum()) / (yc @ yc)
        if dj > t_min * se:
            d[j], c[j] = dj, cj
    return lambda Z: (Z - c) / d


def fit_instrument(Xc: Mapping[str, np.ndarray], Y: np.ndarray, groups: np.ndarray, rows=None, alphas=DEFAULT_ALPHAS,
                   k: int = 5, seed: int = 0):
    """The instrument on ``rows`` (default all): cell and penalty by :func:`fit_select`, then the inverse calibration
    (:func:`_affine`) on grouped inner out-of-fold predictions at that (cell, penalty). Returns (cell, decode(X) -> [m, d], inner R²)."""
    Xc = {c: np.asarray(X, np.float64) for c, X in Xc.items()}
    Y = np.asarray(Y, np.float64).reshape(len(Y), -1)
    rows = np.arange(len(Y)) if rows is None else np.asarray(rows)
    c, f, r2, a = fit_select(Xc, Y, groups, rows, alphas, k, seed)
    g = groups[rows]
    if len(set(g)) < 2:
        return c, f, r2
    P = np.zeros((len(rows), Y.shape[1]))
    for te in grouped_kfold(g, min(k, len(set(g))), seed + 20_000):
        tr = np.setdiff1d(np.arange(len(rows)), te)
        P[te] = _SVDRidge(Xc[c][rows[tr]], Y[rows[tr]], True).predict(Xc[c][rows[te]], a * len(tr) / len(rows))
    cal = _affine(P, Y[rows], g)
    return c, (lambda X, f=f, cal=cal: cal(f(X))), r2


def _stack(Xc, Y, groups, fit_rows):
    """R1 joint: Xc values (X_pre, X_post) and Y = (Y_pre, Y_post) [n, ...] -> rows stacked [2n, ...] (row i and
    n + i are clip i's two phases, same group); anything else is returned as is."""
    if not isinstance(Y, tuple):
        return Xc, np.asarray(Y, np.float64).reshape(len(Y), -1), np.asarray(groups), fit_rows
    n = len(Y[0])
    Xs = {c: np.concatenate([np.asarray(x[0]), np.asarray(x[1])]) for c, x in Xc.items()}
    Ys = np.concatenate([np.asarray(y, np.float64).reshape(n, -1) for y in Y])
    fr = np.arange(n) if fit_rows is None else np.asarray(fit_rows)
    return Xs, Ys, np.concatenate([groups, groups]), np.concatenate([fr, fr + n])


def instrument_folds(Xc: Mapping, Y, groups: np.ndarray, fit_rows=None, k: int = 5, seeds: int = 1,
                     alphas=DEFAULT_ALPHAS) -> list[tuple]:
    """Cross-fitted instruments over the groups of ``fit_rows`` (default all rows): [(test groups, training groups,
    cell, decode)] per outer fold and seed (folds by ``grouped_kfold`` with seeds 0 .. seeds-1). ``Xc`` {cell: X
    [m, D]} with ``Y`` [m, d]; or R1 joint: {cell: (X_pre, X_post)} with Y = (Y_pre, Y_post)."""
    Xc, Y, groups, fit_rows = _stack(Xc, Y, np.asarray(groups), fit_rows)
    fit_rows = np.arange(len(Y)) if fit_rows is None else np.asarray(fit_rows)
    fg = groups[fit_rows]
    folds = []
    for s in range(seeds):
        for te in grouped_kfold(fg, min(k, len(set(fg))), s):
            test = set(fg[te])
            c, f, _ = fit_instrument(Xc, Y, groups, fit_rows[~np.isin(fg, list(test))], alphas, k, s)
            folds.append((test, set(fg) - test, c, f))
    return folds


def apply_instrument(folds, X, groups: np.ndarray, transform=None) -> np.ndarray:
    """Decode X ({cell: [m, D]}, or one [m, D] matrix for single-cell instruments) row by row with the folds that
    held the row's group out, averaged over seeds. ``transform(X_rows, training groups)`` (optional) maps the rows
    first, e.g. a fold-proper moment match. Rows of a group no fold held out raise ValueError."""
    groups = np.asarray(groups)
    out, hit = None, np.zeros(len(groups))
    for test, train, c, f in folds:
        r = np.where(np.isin(groups, list(test)))[0]
        if not len(r):
            continue
        x = (X[c] if isinstance(X, Mapping) else X)[r]
        p = f(transform(np.asarray(x, np.float64), train) if transform else x)
        out = np.zeros((len(groups), p.shape[1])) if out is None else out
        out[r] += p
        hit[r] += 1
    if out is None or (hit == 0).any():
        raise ValueError(f"apply_instrument: {int((hit == 0).sum())} rows belong to groups no fold held out")
    return out / hit[:, None]


def moment_match(M: np.ndarray, T: np.ndarray, groups: np.ndarray):
    """transform for :func:`apply_instrument`: maps rows by the per-feature affine map sending the mean / sd of M's
    per-group means onto those of T's per-group means, both over the fold's training groups (no labels; group means
    because a forecast is one per group: matching to every twin would stretch it by a ladder's spread)."""
    groups = np.asarray(groups)

    def tf(x, train):
        gm = lambda Z: np.array([Z[groups == g].mean(0) for g in sorted(train)])
        Mg, Tg = gm(M), gm(T)
        return (x - Mg.mean(0)) / (Mg.std(0) + 1e-8) * (Tg.std(0) + 1e-8) + Tg.mean(0)
    return tf


def r1_decode(X_pre, X_post, Y_pre, Y_post, groups, applies=None, k=5, n_seeds=4, alphas=DEFAULT_ALPHAS,
              joint=True) -> dict:
    """R1 decoding of one (space, pool, object) in true units. ``joint`` (pre and post rows are features of the same
    space): ONE instrument trained on both rows. ``joint=False`` (two spaces, e.g. encoder pre rows and predictor
    pred-real post rows): one instrument per space, never one linear map across both. ``applies``: {name: (X [n, D],
    M [n, D] or None)} decoded by the post instrument of the fold that held the clip's group out; with M, X is first
    moment-matched (:func:`moment_match`, M onto X_post). Returns {"pre", "post", *applies: [n, d], "r2_pre",
    "r2_post"} (calibrated out-of-fold; constant target columns are passed through at their mean, not fitted)."""
    applies, groups = applies or {}, np.asarray(groups)
    Y = np.vstack([np.asarray(Y_pre, float).reshape(len(groups), -1), np.asarray(Y_post, float).reshape(len(groups), -1)])
    n, var = len(groups), np.ptp(Y, 0) > 1e-12
    Yp, Yq = Y[:n, var], Y[n:, var]
    if joint:
        fp = fq = instrument_folds({"x": (X_pre, X_post)}, (Yp, Yq), groups, None, k, n_seeds, alphas)
    else:
        fp = instrument_folds({"x": X_pre}, Yp, groups, None, k, n_seeds, alphas)
        fq = instrument_folds({"x": X_post}, Yq, groups, None, k, n_seeds, alphas)
    o_pre, o_post = apply_instrument(fp, X_pre, groups), apply_instrument(fq, X_post, groups)
    app = {a: apply_instrument(fq, Xa, groups, None if M is None else moment_match(M, X_post, groups))
           for a, (Xa, M) in applies.items()}

    def full(D):
        out = np.tile(Y.mean(0), (len(D), 1))
        out[:, var] = D
        return out
    return {"pre": full(o_pre), "post": full(o_post), **{a: full(v) for a, v in app.items()},
            "r2_pre": r2_score(Yp, o_pre), "r2_post": r2_score(Yq, o_post)}


# --------------------------------------------------------------------------- #
# selectivity control tasks (Hewitt & Liang 2019)
# --------------------------------------------------------------------------- #
#
# A probe that reads WHICH scene or WHICH object a clip shows can fit a physical target without representing it. The
# control tasks keep the data's structure and drop the physics: ``scene`` gives every clip of a twin group the target
# values of another group (whole blocks, so within-group structure survives), ``object`` gives every object identity
# class (shape, material, colour, size class: the caller's ids) one fixed random value drawn from the target's own
# values. The same probe (same features, penalty search and grouped folds) is fitted on each; selectivity = real R²
# - control R² (out of fold), with a group-bootstrap CI. A high control R² means the probe reads identity. Under
# grouped folds the scene control is learnable only by a probe that carries group-level information across folds
# (expected <= 0); the object control is learnable from appearance alone, since identity classes recur across groups.
#
# Rows with a position in the clip (every token-frame, ``strata``): the scene control matches donor rows by position
# inside the group, so it keeps everything all groups share at a position -- the target's time course (a falling
# ball's velocity is mostly time) and its twin design (each twin slot's rule) -- and any feature that knows the
# position (time-in-clip, which twin) fits that part of BOTH targets: the control R² is inflated and the selectivity
# understated by that share, even for a feature that encodes the quantity exactly. So with ``strata`` (the position
# key, e.g. twin slot x token-frame index) the shared part is partialled out of features and targets alike before
# either probe is fitted (Frisch-Waugh): each row minus the mean of the rows of OTHER groups in its stratum
# (:func:`partial_strata`), real and control. Both R² then score the variation across scenes within a stratum: a
# feature that knows only the position gets ~0 on both (selectivity ~0); a feature that encodes the quantity gets its
# real R² and a control R² ~0 (selectivity ~ real R²).

CONTROLS = ("scene", "object")


def control_targets(Y: np.ndarray, groups: np.ndarray, object_ids=None, seed: int = 0) -> dict[str, np.ndarray]:
    """{"scene": Y with whole groups' blocks permuted across groups (a derangement; rows matched by position inside
    the group, cyclically when sizes differ), "object": a fixed random target row per object-identity id} (``object``
    only when ``object_ids`` [n] is given)."""
    Y = np.asarray(Y, np.float64)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    rows = {g: np.where(groups == g)[0] for g in ug}
    order = rng.permutation(len(ug))
    src = {ug[order[i]]: ug[order[(i + 1) % len(ug)]] for i in range(len(ug))}         # a cycle: never itself
    scene = np.empty_like(Y)
    for g, r in rows.items():
        s = rows[src[g]]
        scene[r] = Y[s[np.arange(len(r)) % len(s)]]
    out = {"scene": scene}
    if object_ids is not None:
        ids = np.asarray(object_ids)
        pick = {i: Y[rng.integers(len(Y))] for i in np.unique(ids)}
        out["object"] = np.stack([pick[i] for i in ids])
    return out


def partial_strata(A: np.ndarray, strata, groups) -> np.ndarray:
    """A [m, ...] minus, per row, the mean of A over the rows of OTHER groups in the same stratum (a row's own group
    never enters its correction; a stratum no other group has is left as is). Returns [m, d] float64."""
    A = np.asarray(A, np.float64).reshape(len(A), -1)
    strata, groups = np.asarray(strata), np.asarray(groups)
    out = A.copy()
    for t in np.unique(strata):
        r = np.where(strata == t)[0]
        S = A[r].sum(0)
        for g in np.unique(groups[r]):
            own = r[groups[r] == g]
            if len(r) > len(own):
                out[own] -= (S - A[own].sum(0)) / (len(r) - len(own))
    return out


def selectivity(Y, P, Yc, Pc, groups, n_boot: int = 500, seed: int = 0) -> dict:
    """Real out-of-fold R² (targets Y, predictions P) minus the control task's (Yc, Pc), with a group-bootstrap 95%
    CI of the difference (the same resampled groups for both)."""
    Y, P, Yc, Pc = (np.asarray(a, np.float64).reshape(len(groups), -1) for a in (Y, P, Yc, Pc))
    groups = np.asarray(groups)
    r, c = r2_score(Y, P), r2_score(Yc, Pc)
    out = {"r2": r, "control_r2": c, "selectivity": r - c, "ci95": [float("nan")] * 2}
    ug = np.unique(groups)
    if n_boot and len(ug) > 1:
        by = [np.where(groups == g)[0] for g in ug]
        rng = np.random.default_rng(seed)
        b = []
        for _ in range(n_boot):
            ix = np.concatenate([by[j] for j in rng.integers(0, len(by), len(by))])
            b.append(r2_score(Y[ix], P[ix]) - r2_score(Yc[ix], Pc[ix]))
        b = np.array(b, float)
        if np.isfinite(b).any():
            out["ci95"] = [float(np.nanquantile(b, 0.025)), float(np.nanquantile(b, 0.975))]
    return out


def instrument_selectivity(Xc, Y, groups, object_ids=None, k: int = 5, seeds: int = 1, alphas=DEFAULT_ALPHAS,
                           n_boot: int = 500, seed: int = 0, strata=None) -> dict:
    """{control: :func:`selectivity`} of the R1 instrument (same arguments as :func:`instrument_folds`; R1 joint:
    Xc {cell: (X_pre, X_post)}, Y = (Y_pre, Y_post), the clip's phases share its control value mapping). ``strata``
    [m] (single-matrix rows only: the row's position key, e.g. twin slot x token-frame index): features, real and
    control targets are partialled within strata first (:func:`partial_strata`; see the section comment)."""
    groups = np.asarray(groups)
    joint = isinstance(Y, tuple)
    Ys = np.concatenate([np.asarray(y, np.float64).reshape(len(groups), -1) for y in Y], 1) if joint else Y
    ctl = control_targets(Ys, groups, object_ids, seed)
    if strata is not None:
        if joint:
            raise ValueError("instrument_selectivity: strata label single-matrix rows, not R1 joint (pre, post) pairs")
        Xc = {c: partial_strata(x, strata, groups) for c, x in Xc.items()}
        Ys, ctl = partial_strata(Ys, strata, groups), {n: partial_strata(T, strata, groups) for n, T in ctl.items()}
    split = (lambda A: (A[:, :A.shape[1] // 2], A[:, A.shape[1] // 2:])) if joint else (lambda A: A)
    Xa = {c: np.concatenate([x[0], x[1]]) for c, x in Xc.items()} if joint else Xc
    ga = np.concatenate([groups, groups]) if joint else groups
    stack = (lambda A: np.concatenate(split(A))) if joint else (lambda A: A)

    def oof(T):
        return apply_instrument(instrument_folds(Xc, split(T), groups, None, k, seeds, alphas), Xa, ga)
    P = oof(Ys)
    return {name: selectivity(stack(Ys), P, stack(T), oof(T), ga, n_boot, seed) for name, T in ctl.items()}


# --------------------------------------------------------------------------- #
# RigidBench velocity protocol (spec §5.1 velocity protocol check)
# --------------------------------------------------------------------------- #


def adam_linear_probe(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, *, k: int = 5, fold_seeds: int = 4,
                      epochs: int = 20, lr: float = 1e-3, batch: int = 32, seed: int = 0) -> dict:
    """Unregularised linear probe trained with Adam (RigidBench's protocol): MSE, ``epochs`` passes of
    shuffled minibatches, zero init, default Adam betas. Features and targets standardised on the
    training fold (a preprocessing step; it leaves the probe unregularised). Same grouped folds as
    ``fit_probe``. Returns {score_mean, score_sd, scores, oof [fold_seeds, n, m]}."""
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64).reshape(len(X), -1)
    rng = np.random.default_rng(seed)
    scores, oofs = [], []
    for fs in range(fold_seeds):
        oof = np.zeros_like(Y)
        for te in grouped_kfold(groups, k, fs):
            tr = np.setdiff1d(np.arange(len(Y)), te)
            mx, sx = X[tr].mean(0), feature_scale(X[tr])
            my, sy = Y[tr].mean(0), Y[tr].std(0)
            sy[sy == 0] = 1.0
            Xs, Ys = (X[tr] - mx) / sx, (Y[tr] - my) / sy
            W, b = np.zeros((X.shape[1], Y.shape[1])), np.zeros(Y.shape[1])
            mW, vW, mb, vb, t = np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b), 0
            for _ in range(epochs):
                order = rng.permutation(len(tr))
                for s in range(0, len(tr), batch):
                    i = order[s:s + batch]
                    r = Xs[i] @ W + b - Ys[i]
                    gW, gb = 2 * Xs[i].T @ r / len(i), 2 * r.mean(0)
                    t += 1
                    for p, g, m_, v_ in ((W, gW, mW, vW), (b, gb, mb, vb)):
                        m_ *= 0.9; m_ += 0.1 * g
                        v_ *= 0.999; v_ += 0.001 * g * g
                        p -= lr * (m_ / (1 - 0.9 ** t)) / (np.sqrt(v_ / (1 - 0.999 ** t)) + 1e-8)
            oof[te] = ((X[te] - mx) / sx @ W + b) * sy + my
        scores.append(r2_score(Y, oof))
        oofs.append(oof)
    return {"score_mean": float(np.mean(scores)), "score_sd": float(np.std(scores)), "scores": scores,
            "oof": np.stack(oofs)}
