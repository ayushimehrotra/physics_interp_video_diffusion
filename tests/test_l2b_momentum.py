"""L2b momentum measurement: kinematic baseline, transfer split, norm routes, null diagnostics."""
from __future__ import annotations

import numpy as np
import pytest

LAMS = (1e-2, 1.0, 1e2)


def _labels(m, v_pre, v_post):
    return {"physlaw": {"family": "f1", "objects": {"A": {"mass": m[0], "radius": 0.1}, "B": {"mass": m[1], "radius": 0.1}},
                        "contact_normal": [1.0, 0.0], "pre_velocity": {"A": list(v_pre[0]), "B": list(v_pre[1])},
                        "post_velocity": {"A": list(v_post[0]), "B": list(v_post[1])}},
            "collision": {"contact_point_world": [0.0, 0.0, 0.0]}}


def _kin_dataset(separable: bool, n_groups=40, seed=0):
    from physlaw.data import collision_physics as cp
    from physlaw.eval.l2b_detection import kinematic_features
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    for i in range(n_groups):
        m = rng.uniform(1, 3, 2)
        st = cp.PreState(m[0], m[1], np.array([1.0, rng.normal()]), np.array([-1.0, rng.normal()]), np.array([1.0, 0.0]))
        v0 = np.stack([st.v_a, st.v_b])
        o = cp.legal_outcome(st, 0.5).reshape(2, 2)
        vcm = (m @ (v0 + o)) / m.sum()
        drift = -0.6 * vcm                                     # violator slower
        if not separable:      # violator: same kinematics distribution as the legal twin (relabelled twin)
            o2 = cp.legal_outcome(st, rng.uniform(0, 1)).reshape(2, 2)
            pair = [(v0 + o2, 1.0), (v0 + cp.legal_outcome(st, rng.uniform(0, 1)).reshape(2, 2), 0.0)]
        else:
            pair = [(v0 + o + drift, 1.0), (v0 + o, 0.0)]
        for vp, lab in pair:
            X.append(kinematic_features(_labels(m, v0, vp))); y.append(lab); g.append(i)
    return np.array(X), np.array(y), np.array(g)


def test_kinematic_baseline_separable_vs_not():
    from physlaw.eval.l2b_detection import cv_auc
    X, y, g = _kin_dataset(True)
    assert X.shape[1] == 6
    a, _, _, _ = cv_auc(X, y, g, LAMS, 5, 2)
    assert np.mean(a) > 0.9
    X, y, g = _kin_dataset(False)
    a, _, _, _ = cv_auc(X, y, g, LAMS, 5, 2)
    assert 0.25 < np.mean(a) < 0.75


def _contrast(rng, n_groups, w, signal):
    y = np.tile([1.0, 0.0], n_groups)
    g = np.repeat(np.arange(n_groups), 2)
    X = rng.normal(size=(len(y), 8))
    X[:, 0] += signal * (2 * y - 1) * w
    return X, y, g


def test_transfer_split():
    from physlaw.eval.l2b_detection import transfer_auc
    rng = np.random.default_rng(0)
    Xa, ya, ga = _contrast(rng, 40, 1.0, 1.5)
    Xb, yb, gb = _contrast(rng, 40, 1.0, 1.5)
    assert np.mean(transfer_auc(Xa, ya, ga, Xb, yb, gb, LAMS, 5, 2)) > 0.8          # shared signal transfers
    Xc, yc, gc = _contrast(rng, 40, -1.0, 1.5)
    assert np.mean(transfer_auc(Xa, ya, ga, Xc, yc, gc, LAMS, 5, 2)) < 0.2          # reversed shortcut fails
    Xd, yd, gd = _contrast(rng, 40, 1.0, 0.0)
    assert 0.3 < np.mean(transfer_auc(Xa, ya, ga, Xd, yd, gd, LAMS, 5, 2)) < 0.7    # no signal in test


def test_transfer_holds_out_groups(monkeypatch):
    """No group's rows of the training contrast are used when that group's test rows are scored."""
    import physlaw.eval.l2b_detection as m
    seen = []
    real = m._project

    def spy(X, tr):
        seen.append(len(tr))
        return real(X, tr)
    rng = np.random.default_rng(0)
    Xa, ya, ga = _contrast(rng, 20, 1.0, 1.0)
    Xb, yb, gb = _contrast(rng, 20, 1.0, 1.0)
    # outer fits must use only the 4/5 of training-contrast groups not held out: 32 of 40 rows (inner fits fewer)
    monkeypatch.setattr(m, "_project", spy)
    m.transfer_auc(Xa, ya, ga, Xb, yb, gb, (1.0,), 5, 1)
    assert max(seen) == 32


def test_norm_route_beats_logistic_on_norm_label():
    """'conserved' = |Δp| == 0: linear logistic on activations encoding Δp is near chance; |Δp̂| separates."""
    from physlaw.eval.l2b_detection import cv_auc
    from physlaw.probes.ridge import auc_score, fit_probe
    rng = np.random.default_rng(0)
    n = 60
    y = np.tile([1.0, 0.0], n)
    g = np.repeat(np.arange(n), 2)
    ang = rng.uniform(0, 2 * np.pi, len(y))
    dp = y[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1)          # violators: unit |Δp|, random direction
    X = np.hstack([dp @ rng.normal(size=(2, 10)), 0.05 * rng.normal(size=(len(y), 10))])
    la, _, _, _ = cv_auc(X, y, g, LAMS, 5, 2)
    pr = fit_probe(X, dp, g, alphas=LAMS, k=5, fold_seeds=2)
    fn = np.mean([auc_score(y, np.linalg.norm(o, axis=1)) for o in pr.oof])
    assert np.mean(la) < 0.75 and fn > 0.95


def test_null_diagnostics_moves():
    from physlaw.eval.l2b_detection import null_diagnostics, swap_perms
    g = np.repeat(np.arange(30), 2)
    y = np.tile([1.0, 0.0], 30)
    P = swap_perms(g, y, 200, seed=0)
    d = null_diagnostics(P, y, np.random.default_rng(0).uniform(0.4, 0.6, 200))
    assert d["share_differ_from_identity_and_flip"] == 1.0 and d["n_distinct"] == 200 and d["null_sd"] > 0
    assert np.all(P.reshape(-1, 30, 2).sum(-1) == 1)            # still one violator per group
