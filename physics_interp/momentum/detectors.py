"""The momentum detectors, each a function from a clip set to one score per clip.

Every detector here returns ``Scores`` — a per-clip score where *higher means
more violating* — and the reporting layer (:func:`report`) turns that into raw
AUC, per-stratum AUC and permutation nulls. No detector sees paired accuracy.

The four routes compared in §50:

* :func:`direct_binary`  — §2a. Ridge straight to the label; score = prediction.
* :func:`fitted_norm`    — §2c. Ridge to the dp vector; score = |prediction|.
* :func:`composed_norm`  — §48. Decode log m_A, log m_B, dv_A, dv_B on their own
  targets, do the momentum arithmetic outside the probe; score = |dp_hat|.
* :func:`subspace_membership` — §49. Decode the outcome (dv_A, dv_B) and score its
  distance from the scene's legal line over its length. ``fit_honest`` trains
  every readout on honest clips only.

Plus §2g's :func:`failure_localisation` and §51's :func:`lawful_errors`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from . import physics
from .captures import ClipSet
from .stats import auc, cv_predict, permutation_null, r2

__all__ = [
    "Scores", "report", "ingredient_targets", "decode", "choose_site",
    "direct_binary", "fitted_norm", "composed_norm", "subspace_membership",
    "failure_localisation", "lawful_errors", "ROUTES",
]


@dataclass
class Scores:
    score: np.ndarray                 # [n], higher = more violating
    valid: np.ndarray                 # [n] bool, clips the detector could score
    extra: dict[str, Any] = field(default_factory=dict)


def report(cs: ClipSet, sc: Scores, *, draws: int = 4000, seed: int = 0,
           strata: Sequence[str] = ("elastic", "inelastic")) -> dict[str, Any]:
    """Raw AUC over every valid clip, per stratum, with label-permutation nulls."""
    out: dict[str, Any] = {"n_scored": int(sc.valid.sum()), "n_dropped": int((~sc.valid).sum())}
    y = cs.violates
    pooled = permutation_null(sc.score[sc.valid], y[sc.valid], draws=draws, seed=seed)
    out["pooled"] = pooled
    for s in strata:
        m = sc.valid & (cs.stratum == s)
        if m.sum() >= 8 and y[m].any() and (~y[m]).any():
            out[s] = permutation_null(sc.score[m], y[m], draws=draws, seed=seed)
    out["extra"] = {k: v for k, v in sc.extra.items() if not isinstance(v, np.ndarray)}
    return out


# --------------------------------------------------------------------------- #
# targets and decoding
# --------------------------------------------------------------------------- #


def _pre_states(cs: ClipSet) -> list[physics.PreState]:
    return [physics.pre_state_from_labels(l) for l in cs.labels]


def ingredient_targets(cs: ClipSet) -> dict[str, np.ndarray]:
    """The visible quantities every route decodes: masses and velocity changes."""
    outcome = np.stack([physics.outcome_from_labels(l) for l in cs.labels])
    pres = _pre_states(cs)
    return {
        "log_m_a": np.log([p.m_a for p in pres])[:, None],
        "log_m_b": np.log([p.m_b for p in pres])[:, None],
        "dv_a": outcome[:, :2],
        "dv_b": outcome[:, 2:],
        "dp": np.stack([physics.delta_p(o, p) for o, p in zip(outcome, pres)]),
        "outcome": outcome,
    }


def decode(cs: ClipSet, site: str, pool: str, targets: dict[str, np.ndarray], *,
           fit_mask: np.ndarray | None = None, n_folds: int = 5, seed: int = 0
           ) -> tuple[dict[str, np.ndarray], dict[str, float], np.ndarray]:
    """Out-of-fold ridge predictions for several targets at one site/pool.

    Returns ``(pred_by_target, r2_by_target, valid)``. R^2 is reported on the
    fit population (honest clips only under ``fit_mask``) so a label-free
    quantity stays label-free.
    """
    X, valid = cs.features(site, pool)
    idx = np.flatnonzero(valid)
    fm = None if fit_mask is None else np.asarray(fit_mask, bool)[idx]
    preds, scores = {}, {}
    for name, Y in targets.items():
        p = np.full_like(Y, np.nan)
        p[idx] = cv_predict(X[idx], Y[idx], cs.seed[idx], n_folds=n_folds, seed=seed, fit_mask=fm)
        preds[name] = p
        pop = idx if fit_mask is None else idx[fm]
        scores[name] = r2(Y[pop], p[pop])
    return preds, scores, valid


def choose_site(cs: ClipSet, pool: str, *, rule: str = "ingredients", seed: int = 0) -> tuple[str, dict]:
    """Pick the site to report at **without looking at the violation label**.

    ``rule='last'`` is the pre-declared last block; ``'ingredients'`` is the
    site where the mean ingredient R^2 (log m_A, log m_B, dv_A, dv_B) is highest.
    ``'embed'`` is the patch embedding. Any other string names a site.
    """
    sites = cs.sites
    if rule == "last":
        return sites[-1], {"rule": "last"}
    if rule == "embed":
        return sites[0], {"rule": "embed"}
    if rule != "ingredients":
        if rule not in sites:
            raise KeyError(f"site {rule!r} not in {sites}")
        return rule, {"rule": "named"}
    t = ingredient_targets(cs)
    ing = {k: t[k] for k in ("log_m_a", "log_m_b", "dv_a", "dv_b")}
    table = {}
    for s in sites:
        _, sc, _ = decode(cs, s, pool, ing, seed=seed)
        table[s] = float(np.mean(list(sc.values())))
    best = max(table, key=table.get)
    return best, {"rule": "ingredients", "mean_ingredient_r2": table}


# --------------------------------------------------------------------------- #
# the four routes
# --------------------------------------------------------------------------- #


def direct_binary(cs: ClipSet, site: str, pool: str, *, seed: int = 0, folded: bool = False) -> Scores:
    X, valid = cs.features(site, pool)
    idx = np.flatnonzero(valid)
    y = cs.violates.astype(np.float64)[:, None]
    pred = np.full(cs.n, np.nan)
    pred[idx] = cv_predict(X[idx], y[idx], cs.seed[idx], seed=seed)[:, 0]
    score = np.abs(pred - 0.5) if folded else pred
    return Scores(score, valid, {"r2": r2(y[idx], pred[idx, None])})


def fitted_norm(cs: ClipSet, site: str, pool: str, *, seed: int = 0) -> Scores:
    t = ingredient_targets(cs)
    preds, sc, valid = decode(cs, site, pool, {"dp": t["dp"]}, seed=seed)
    return Scores(np.linalg.norm(preds["dp"], axis=1), valid,
                  {"dp_r2": sc["dp"], "dp_pred": preds["dp"], "dp_true": t["dp"]})


def composed_norm(cs: ClipSet, site: str, pool: str, *, seed: int = 0,
                  fit_mask: np.ndarray | None = None) -> Scores:
    t = ingredient_targets(cs)
    ing = {k: t[k] for k in ("log_m_a", "log_m_b", "dv_a", "dv_b")}
    preds, sc, valid = decode(cs, site, pool, ing, seed=seed, fit_mask=fit_mask)
    dp_hat = np.exp(preds["log_m_a"]) * preds["dv_a"] + np.exp(preds["log_m_b"]) * preds["dv_b"]
    score = np.linalg.norm(dp_hat, axis=1)
    honest = valid & ~cs.violates
    viol = valid & cs.violates
    return Scores(score, valid, {
        "ingredient_r2": sc,
        "noise_floor_honest": float(np.median(score[honest])) if honest.any() else float("nan"),
        "signal_violating": float(np.median(score[viol])) if viol.any() else float("nan"),
        "dp_hat": dp_hat,
    })


def subspace_membership(cs: ClipSet, site: str, pool: str, *, seed: int = 0,
                        fit_honest: bool = False, decoded_pre: bool = False,
                        segment: tuple[float, float] | None = None) -> Scores:
    t = ingredient_targets(cs)
    fit_mask = ~cs.violates if fit_honest else None
    targets = {"dv_a": t["dv_a"], "dv_b": t["dv_b"]}
    if decoded_pre:
        targets.update(log_m_a=t["log_m_a"], log_m_b=t["log_m_b"])
    preds, sc, valid = decode(cs, site, pool, targets, seed=seed, fit_mask=fit_mask)
    pres = _pre_states(cs)
    score = np.full(cs.n, np.nan)
    offline = np.full(cs.n, np.nan)
    for i in np.flatnonzero(valid):
        pre = pres[i]
        if decoded_pre:
            pre = physics.PreState(float(np.exp(preds["log_m_a"][i, 0])),
                                   float(np.exp(preds["log_m_b"][i, 0])), pre.v_a, pre.v_b, pre.n)
        o = np.concatenate([preds["dv_a"][i], preds["dv_b"][i]])
        score[i] = physics.membership_score(o, pre, segment=segment)
        offline[i] = physics.distance_to_legal_line(o, pre, segment=segment)
    return Scores(score, valid, {"ingredient_r2": sc, "fit_honest": fit_honest,
                                 "decoded_pre": decoded_pre, "offline_distance": offline})


ROUTES: dict[str, Callable[..., Scores]] = {
    "direct_binary": direct_binary,
    "fitted_norm": fitted_norm,
    "composed_norm": composed_norm,
    "subspace": subspace_membership,
}


# --------------------------------------------------------------------------- #
# §2g failure localisation
# --------------------------------------------------------------------------- #


def _folded_auc(score, y):
    return auc(score, y)


def failure_localisation(cs: ClipSet, site: str, pool: str, *, seed: int = 0,
                         noise_draws: int = 50) -> dict[str, Any]:
    """Q1–Q4 of §2g at one site.

    Q1 corrupts the true dp with isotropic noise at the variance that matches the
    probe's R^2 and scores |.| — a reference, not a ceiling (§2g-bis).
    Q2 splits the prediction error on violating clips into radial (along the
    true dp) and tangential (across it), in units of the mean true |dp|.
    Q3 scores each |dp| tercile of violating clips against all honest clips.
    Q4 is the composed norm at the same site.
    """
    fn = fitted_norm(cs, site, pool, seed=seed)
    y = cs.violates
    valid = fn.valid
    pred, true = fn.extra["dp_pred"], fn.extra["dp_true"]
    observed = auc(fn.score[valid], y[valid])
    out: dict[str, Any] = {"site": site, "pool": pool, "dp_r2": fn.extra["dp_r2"], "auc": observed,
                           "auc_by_stratum": {}}
    for s in ("elastic", "inelastic"):
        m = valid & (cs.stratum == s)
        if m.sum() >= 8:
            out["auc_by_stratum"][s] = auc(fn.score[m], y[m])

    # Q1 — attainable AUC at the measured fidelity
    rng = np.random.default_rng(seed)
    r2v = max(fn.extra["dp_r2"], 0.0)
    var_true = true[valid].var(0).sum()
    noise_var = var_true * (1.0 - r2v) / max(r2v, 1e-9) if r2v > 0 else None
    if noise_var is None or not np.isfinite(noise_var):
        attain = 0.5
    else:
        sd = np.sqrt(noise_var / 2.0)
        vals = []
        for _ in range(noise_draws):
            corrupted = true[valid] + sd * rng.normal(size=true[valid].shape)
            # the ridge at R^2 shrinks toward the mean; match that scale too
            corrupted = r2v * corrupted + (1 - r2v) * true[valid].mean(0)
            vals.append(auc(np.linalg.norm(corrupted, axis=1), y[valid]))
        attain = float(np.mean(vals))
    out["q1"] = {"attainable_auc": attain, "excess": float(observed - attain)}

    # Q2 — error geometry on violating clips
    v = valid & y
    err = pred[v] - true[v]
    tn = np.linalg.norm(true[v], axis=1)
    unit = true[v] / tn[:, None]
    radial = np.einsum("ij,ij->i", err, unit)
    tang = err - radial[:, None] * unit
    pn = np.linalg.norm(pred[v], axis=1)
    cos = np.einsum("ij,ij->i", pred[v], unit) / np.where(pn > 0, pn, np.nan)
    scale = float(tn.mean())
    out["q2"] = {
        "radial_abs": float(np.abs(radial).mean() / scale),
        "tangential_abs": float(np.linalg.norm(tang, axis=1).mean() / scale),
        "cos": float(np.nanmean(cos)),
        "pred_over_true_length": float(pn.mean() / scale),
        "true_dp_mean_norm": scale,
    }

    # Q3 — sensitivity by violation size
    terciles = np.quantile(tn, [1 / 3, 2 / 3])
    bins = np.digitize(tn, terciles)
    honest_idx = np.flatnonzero(valid & ~y)
    viol_idx = np.flatnonzero(v)
    q3 = {}
    for b, name in enumerate(("small", "medium", "large")):
        sel = np.r_[honest_idx, viol_idx[bins == b]]
        q3[name] = {"auc": auc(fn.score[sel], y[sel]), "n_violating": int((bins == b).sum())}
    out["q3"] = q3

    # Q4 — composition
    cn = composed_norm(cs, site, pool, seed=seed)
    out["q4"] = {"composed_auc": auc(cn.score[cn.valid], y[cn.valid]),
                 "ingredient_r2": cn.extra["ingredient_r2"]}
    return out


# --------------------------------------------------------------------------- #
# §51 lawful errors
# --------------------------------------------------------------------------- #


def lawful_errors(cs: ClipSet, site: str, pool: str, *, seed: int = 0, draws: int = 2000
                  ) -> dict[str, Any]:
    """Does the model suppress the impossible part of a violation?

    For every twin pair, the true displacement ``D = o_v - o_h`` in outcome space
    splits into its component along the legal direction (``D_on``) and the rest
    (``D_off``). The decoded displacement ``D_hat`` is projected onto the same two
    unit vectors. Slopes of recovered-vs-true through the origin, over seeds,
    give ``k_on`` and ``k_off``; the statistic is ``k_off / k_on``. Shrinkage hits
    both equally and cancels; a ratio below 1 means the off-line (impossible)
    part is suppressed — the model applies the law. Bootstrap CI over seeds.
    """
    t = ingredient_targets(cs)
    preds, sc, valid = decode(cs, site, pool, {"dv_a": t["dv_a"], "dv_b": t["dv_b"]}, seed=seed)
    o_hat = np.concatenate([preds["dv_a"], preds["dv_b"]], axis=1)
    o_true = t["outcome"]
    pres = _pre_states(cs)
    rows = []
    for h, v in cs.twins():
        if not (valid[h] and valid[v]):
            continue
        b = physics.legal_direction(pres[h])
        b /= np.linalg.norm(b)
        D = o_true[v] - o_true[h]
        on = float(D @ b)
        off_vec = D - on * b
        off = float(np.linalg.norm(off_vec))
        u_off = off_vec / off if off > 0 else np.zeros(4)
        Dh = o_hat[v] - o_hat[h]
        rows.append((on, float(Dh @ b), off, float(Dh @ u_off), cs.stratum[h]))
    arr = np.array([r[:4] for r in rows], dtype=np.float64)
    strata = np.array([r[4] for r in rows])

    def slopes(a):
        k_on = float((a[:, 0] * a[:, 1]).sum() / (a[:, 0] ** 2).sum())
        k_off = float((a[:, 2] * a[:, 3]).sum() / (a[:, 2] ** 2).sum())
        return k_on, k_off

    def ratio(a):
        k_on, k_off = slopes(a)
        return k_off / k_on if k_on != 0 else float("nan")

    rng = np.random.default_rng(seed)
    n = len(arr)
    boots = np.array([ratio(arr[rng.integers(0, n, n)]) for _ in range(draws)])
    k_on, k_off = slopes(arr)
    out = {
        "site": site, "pool": pool, "n_pairs": n, "ingredient_r2": sc,
        "k_on": k_on, "k_off": k_off, "ratio": ratio(arr),
        "ratio_ci95": [float(np.nanquantile(boots, 0.025)), float(np.nanquantile(boots, 0.975))],
        "corr_off": float(np.corrcoef(arr[:, 2], arr[:, 3])[0, 1]),
        "corr_on": float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]) if arr[:, 0].std() > 0 else float("nan"),
        "by_stratum": {},
    }
    for s in ("elastic", "inelastic"):
        m = strata == s
        if m.sum() >= 8:
            bs = np.array([ratio(arr[m][rng.integers(0, m.sum(), m.sum())]) for _ in range(draws)])
            out["by_stratum"][s] = {"n": int(m.sum()), "ratio": ratio(arr[m]),
                                    "ratio_ci95": [float(np.nanquantile(bs, 0.025)),
                                                   float(np.nanquantile(bs, 0.975))],
                                    "p_below_1": float((bs < 1).mean())}
    return out
