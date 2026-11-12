"""physlaw.probes: ridge probe pipeline on synthetic linear data, the two probe pitfalls (near-constant features,
pooled out-of-fold AUC), and grid_probe == ridge.fit_probe exactly."""
from __future__ import annotations

import numpy as np
import pytest

from physlaw.probes.grid import grid_probe
from physlaw.probes.ridge import _SVDRidge, auc_score, fit_probe, grouped_kfold


def synthetic(n=400, d=64, noise=0.3, seed=0, group_size=5):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=d); w /= np.linalg.norm(w)
    X = rng.normal(size=(n, d))
    y = X @ w + noise * rng.normal(size=n)
    groups = np.repeat(np.arange(n // group_size), group_size)
    return X, y, groups, w


def test_probe_recovers_known_direction():
    X, y, g, w = synthetic()
    res = fit_probe(X, y, g, fold_seeds=3)
    expected_r2 = 1 / (1 + 0.3 ** 2)
    assert res.score_mean == pytest.approx(expected_r2, abs=0.05)
    cos = abs(res.direction @ w) / np.linalg.norm(res.direction)
    assert cos > 0.98


def test_probe_on_noise_scores_near_zero():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 64)); y = rng.normal(size=300)
    g = np.repeat(np.arange(60), 5)
    assert fit_probe(X, y, g, fold_seeds=3).score_mean < 0.05


def test_groups_never_split_across_folds():
    g = np.repeat(np.arange(37), 5)
    for seed in range(5):
        folds = grouped_kfold(g, 5, seed)
        assert sorted(np.concatenate(folds)) == list(range(len(g)))
        for f in folds:
            others = np.setdiff1d(np.arange(len(g)), f)
            assert not set(g[f]) & set(g[others])


def test_auc_matches_definition():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 200); s = y + rng.normal(size=200)
    brute = np.mean([(a > b) + 0.5 * (a == b) for a in s[y == 1] for b in s[y == 0]])
    assert auc_score(y, s) == pytest.approx(brute, abs=1e-12)


# ------------------------------------------------------------------ grid_probe


def _data(seed=0, n_groups=20, twins=5, d=40):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), twins)
    t = np.tile(np.arange(twins), n_groups)
    X = rng.normal(size=(len(g), d))
    y1 = X[:, :2] @ [[1.0, 0.5], [0.2, -1.0]] + 0.5 * rng.normal(size=(len(g), 2))
    y2 = X[:, 3] * 0.3 + rng.normal(size=len(g))
    return X, {"vel": y1, "spd": y2}, g, t


def test_matches_fit_probe():
    X, blocks, g, _ = _data()
    out = grid_probe(X, blocks, g, fold_seeds=3)
    for name, y in blocks.items():
        ref = fit_probe(X, y, g, fold_seeds=3)
        assert np.allclose(out[name]["scores"], ref.scores, atol=1e-8)
        assert out[name]["alphas_chosen"] == [float(a) for a in ref.alphas_chosen]


# ------------------------------------------------------------------ pitfalls

def test_near_constant_features_are_zeroed_not_blown_up():
    """A column with sd ~1e-12 (rounding noise around a constant) must not be standardised to unit variance."""
    X, y, g, _ = synthetic(n=200, d=16, seed=4)
    rng = np.random.default_rng(5)
    noisy = np.c_[X, 3.0 + 1e-12 * rng.normal(size=len(X))]
    m = _SVDRidge(noisy, y[:, None])
    assert np.isinf(m.sx[-1]) and np.allclose(((noisy - m.mx) / m.sx)[:, -1], 0.0)
    a = fit_probe(X, y, g, fold_seeds=2)
    b = fit_probe(noisy, y, g, fold_seeds=2)
    assert np.allclose(a.scores, b.scores, atol=1e-9)
    out = grid_probe(noisy, {"y": y}, g, fold_seeds=2)
    assert np.allclose(out["y"]["scores"], a.scores, atol=1e-8)


def test_pooled_auc_has_no_fold_intercept_leak():
    """Pure-noise features, labels whose class balance differs strongly between groups: the pooled out-of-fold AUC
    must sit at chance. Without centring each fold on its training-mean score, the fold intercepts (training class
    balance, anti-correlated with the test fold's) rank whole folds and pull the pooled AUC below 0.5."""
    aucs = []
    for seed in range(5):                    # uncorrected pooling gives ~0.43 on these draws
        rng = np.random.default_rng(seed)
        g = np.repeat(np.arange(60), 6)
        y = (rng.uniform(size=len(g)) < np.repeat(rng.uniform(0.05, 0.95, 60), 6)).astype(float)
        X = rng.normal(size=(len(g), 8))
        aucs.append(fit_probe(X, y, g, binary=True, fold_seeds=4, alphas=(1e5,), nested=False).score_mean)
    assert abs(np.mean(aucs) - 0.5) < 0.04


# ------------------------------------------------------------------ R1 instrument (shared by L0, L1, L3, L4)


def test_instrument_is_one_map_and_undoes_shrinkage():
    from physlaw.probes.ridge import apply_instrument, instrument_folds
    rng = np.random.default_rng(0)
    n, D = 200, 12
    W = rng.normal(size=(2, D))
    vp, vq = rng.normal(size=(n, 2)), rng.normal(size=(n, 2)) * 2
    Xc = {"c": (vp @ W + 0.8 * rng.normal(size=(n, D)), vq @ W + 0.8 * rng.normal(size=(n, D)))}
    groups = np.repeat(np.arange(50), 4)
    folds = instrument_folds(Xc, (vp, vq), groups, None, k=5, seeds=1, alphas=(1e2, 1e3))   # heavy ridge
    dq = apply_instrument(folds, {"c": Xc["c"][1]}, groups)
    slope = np.polyfit(vq[:, 0], dq[:, 0], 1)[0]
    assert slope == pytest.approx(1.0, abs=0.1)                   # calibrated: no shrinkage toward the mean
    with pytest.raises(ValueError):                               # a group no fold held out
        apply_instrument(folds, {"c": Xc["c"][1][:3]}, np.array([999, 999, 999]))


def test_inverse_calibration_restores_scale_under_shrinkage_and_noise():
    """Heavy ridge + feature noise (corr(y, y_hat) ~ 0.8): the forward calibration y ~ y_hat would leave
    E[decode | y] attenuated by ~rho² toward the mean (slope ~0.66 here) and bias a ratio statistic toward 1; the
    inverse calibration (y_hat ~ c + d y, decode (y_hat - c) / d) gives decodes with slope ~1 on the truth, and
    eps = s'/s read from them is unbiased in the median (both below and above the ladder's middle)."""
    from physlaw.probes.ridge import apply_instrument, instrument_folds
    rng = np.random.default_rng(1)
    n, D = 400, 16
    w = rng.normal(size=D)
    s = rng.uniform(2, 3, n)
    eps = rng.uniform(0.5, 1.5, n)
    X = lambda y: np.outer(y, w) + rng.normal(size=(n, D))
    Xc = {"c": (X(s), X(eps * s))}
    groups = np.repeat(np.arange(100), 4)
    folds = instrument_folds(Xc, (s[:, None], (eps * s)[:, None]), groups, None, k=5, seeds=1, alphas=(1e3, 1e4))
    dp = apply_instrument(folds, {"c": Xc["c"][0]}, groups)[:, 0]
    dq = apply_instrument(folds, {"c": Xc["c"][1]}, groups)[:, 0]
    y, yh = np.r_[s, eps * s], np.r_[dp, dq]
    assert np.corrcoef(y, yh)[0, 1] < 0.9                          # a noisy instrument, not the noise-free limit
    assert np.polyfit(y, yh, 1)[0] == pytest.approx(1.0, abs=0.05)
    e = dq / dp
    assert abs(np.median(e - eps)) < 0.05
    assert abs(np.median((e - eps)[eps < 0.8])) < 0.04
    assert np.polyfit(eps, e, 1)[0] == pytest.approx(1.0, abs=0.1)


def test_selectivity_controls_flag_identity_probes():
    """A feature that encodes only the object's identity class fits a class-determined target AND the object control
    (control R² high, selectivity ~ 0); a feature that encodes the quantity itself keeps selectivity ~ its R². The scene
    control (whole groups' values swapped) is unlearnable on held-out groups for either (control R² <= ~0)."""
    from physlaw.probes.ridge import control_targets, instrument_selectivity
    rng = np.random.default_rng(1)
    G, T, C = 60, 4, 6
    groups = np.repeat(np.arange(G), T)
    ids = rng.integers(0, C, G * T)                                     # object identity class per clip
    y = rng.normal(size=C)[ids] + 0.3 * rng.normal(size=G * T)         # quantity: class mean + within-class spread
    identity = np.eye(C)[ids] @ rng.normal(size=(C, 30))               # identity code only
    quantity = np.outer(y, rng.normal(size=30)) + 0.05 * rng.normal(size=(G * T, 30))
    c = control_targets(y[:, None], groups, ids)
    assert not np.allclose(c["scene"][groups == 0], y[groups == 0][:, None])
    assert all(len(np.unique(c["object"][ids == i])) == 1 for i in range(C))
    kw = dict(k=5, seeds=1, alphas=(1e-2, 1.0, 1e2), n_boot=50)
    s_id = instrument_selectivity({"x": identity}, y[:, None], groups, ids, **kw)
    s_q = instrument_selectivity({"x": quantity}, y[:, None], groups, ids, **kw)
    assert s_id["object"]["r2"] > 0.7 and s_id["object"]["control_r2"] > 0.9 and abs(s_id["object"]["selectivity"]) < 0.3
    assert s_q["object"]["selectivity"] > 0.4 and s_q["object"]["selectivity"] == pytest.approx(
        s_q["object"]["r2"] - s_q["object"]["control_r2"])
    assert s_q["scene"]["control_r2"] < 0.1 and s_q["scene"]["selectivity"] > 0.9 and s_q["scene"]["ci95"][0] > 0.8


def test_selectivity_on_token_frames_partials_out_time():
    """Token-frame rows (a time index each) of a target whose course is mostly time (a ball falling from a
    clip-specific start speed). The scene control keeps the donor's time course, so without ``strata`` even a feature
    that encodes the quantity exactly fits the control (control R² ~ the time share: selectivity understated). With
    ``strata`` = the time index, features and both targets are partialled on time: a feature that knows only time-in-clip scores real R²
    ~ control R² ~ 0 (selectivity ~ 0); the quantity's own feature scores selectivity ~ its real R²."""
    from physlaw.probes.ridge import instrument_selectivity, partial_strata
    rng = np.random.default_rng(3)
    G, C, T = 50, 2, 12                                                 # groups, clips per group, token-frames
    groups, times = np.repeat(np.arange(G), C * T), np.tile(np.arange(T), G * C)
    y = (np.repeat(rng.normal(size=G * C), T) + times)[:, None]       # ~92% of the variance is time
    clock = np.eye(T)[times] @ rng.normal(size=(T, 20))                # time-in-clip only
    assert np.allclose(partial_strata(clock, times, groups), 0, atol=1e-12)
    noise = lambda: 0.01 * rng.normal(size=(len(y), 20))
    quantity = y @ rng.normal(size=(1, 20)) + noise()
    kw = dict(k=5, seeds=1, alphas=(1e-2, 1.0, 1e2, 1e4), n_boot=0)
    raw = instrument_selectivity({"x": quantity}, y, groups, **kw)["scene"]
    assert raw["control_r2"] > 0.7 and raw["selectivity"] < 0.3                          # the inflated control
    s_t = instrument_selectivity({"x": clock + noise()}, y, groups, strata=times, **kw)["scene"]
    s_q = instrument_selectivity({"x": quantity}, y, groups, strata=times, **kw)["scene"]
    assert abs(s_t["r2"]) < 0.1 and abs(s_t["control_r2"]) < 0.1 and abs(s_t["selectivity"]) < 0.1
    assert s_q["r2"] > 0.9 and abs(s_q["control_r2"]) < 0.1 and s_q["selectivity"] == pytest.approx(s_q["r2"], abs=0.1)
    with pytest.raises(ValueError, match="joint"):
        instrument_selectivity({"x": (quantity, quantity)}, (y, y), groups, strata=times, **kw)


def test_r2_is_standardised_per_output():
    """A multi-output R² is the mean of the per-output R² (targets standardised per output), so a low-variance output
    counts as much as a high-variance one; ridge predictions themselves do not depend on the target scale."""
    from physlaw.probes.ridge import _SVDRidge, r2_score
    rng = np.random.default_rng(0)
    y = np.stack([100 * rng.normal(size=200), rng.normal(size=200)], 1)
    yhat = np.stack([y[:, 0], np.zeros(200)], 1)                  # perfect on the big output, nothing on the small
    assert r2_score(y, yhat) == pytest.approx(0.5, abs=0.02)
    X = rng.normal(size=(200, 5))
    Y = X @ rng.normal(size=(5, 2))
    a = _SVDRidge(X, Y).predict(X, 10.0)
    b = _SVDRidge(X, Y * np.array([1.0, 1e3])).predict(X, 10.0) / np.array([1.0, 1e3])
    assert np.allclose(a, b)
