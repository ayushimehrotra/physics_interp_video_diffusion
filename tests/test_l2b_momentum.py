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
    a, _, _ = cv_auc(X, y, g, LAMS, 5, 2)
    assert np.mean(a) > 0.9
    X, y, g = _kin_dataset(False)
    a, _, _ = cv_auc(X, y, g, LAMS, 5, 2)
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
    la, _, _ = cv_auc(X, y, g, LAMS, 5, 2)
    pr = fit_probe(X, dp, g, alphas=LAMS, k=5, fold_seeds=2)
    fn = np.mean([auc_score(y, np.linalg.norm(o, axis=1)) for o in pr.oof])
    assert np.mean(la) < 0.75 and fn > 0.95


def test_paired_auc_diff_and_speed_matched_stratum():
    """The paired group bootstrap of model − kinematic AUC, overall and on a group subset (the speed-matched stratum)."""
    from physlaw.eval.l2b_detection import ke_ratio_to_legal, paired_auc_diff
    rng = np.random.default_rng(0)
    G = 60
    groups = np.repeat(np.arange(G), 2)
    y = np.tile([1.0, 0.0], G)
    model = np.stack([y + 0.5 * rng.normal(size=2 * G) for _ in range(2)])        # good everywhere
    matched = np.arange(G) < 30                                                     # speed-matched groups
    kin = np.stack([np.where(np.repeat(matched, 2), rng.normal(size=2 * G), y + 0.3 * rng.normal(size=2 * G))])
    d = paired_auc_diff(y, groups, model, kin, n_boot=300)
    assert d["n_groups"] == G and d["auc_b"] > 0.6 and "auc_a_ci95" in d
    sm = paired_auc_diff(y, groups, model, kin, keep_groups=np.arange(30), n_boot=300)
    assert sm["n_groups"] == 30 and abs(sm["auc_b"] - 0.5) < 0.2 and sm["diff_ci95"][0] > 0   # speed gone: model wins
    assert sm["diff"] > d["diff"]
    lo, hi = paired_auc_diff(y, groups, model, n_boot=300)["auc_a_ci95"]
    assert lo <= paired_auc_diff(y, groups, model, n_boot=0)["auc_a"] <= hi
    lab = lambda ke, r=None: {"physlaw": {"summary": {"ke_post": ke}, **({"ke_ratio_to_legal": r} if r else {})}}
    assert ke_ratio_to_legal(lab(2.0), lab(1.0)) == 2.0 and ke_ratio_to_legal(lab(2.0, 1.01), lab(1.0)) == 1.01


def test_window_kinematic_features_non_collision():
    """Non-collision families: per non-aux object, mean speed / end speed / displacement / mean spin over the clean
    window after the event (a viscous slide and a legal slide share the entry speed but not the window means)."""
    from physlaw.eval.l2b_detection import kinematic_feature_names, kinematic_features

    def labels(decel):
        pf = []
        for f in range(48):
            t = f / 24
            tau = max(0.0, t - 0.5)
            v = 2.0 - decel(tau)
            pf.append({"time_s": t, "A": {"position": [2.0 * min(t, 0.5) + 2.0 * tau - 0.5 * tau ** 2, 0, 0],
                                          "velocity": [v, 0, 0], "angular_velocity": [0, 0, 0]},
                       "P": {"position": [0, 0, 0], "velocity": [9, 9, 9], "angular_velocity": [0, 0, 0]}})
        return {"scene_constants": {"frame_rate": 24}, "per_frame": pf,
                "physlaw": {"family": "f6", "first_post_contact_frame": 12, "objects": {"A": {}, "P": {"aux": True}}}}
    lab = labels(lambda tau: tau)
    assert kinematic_feature_names(lab) == ["mean_speed_A", "end_speed_A", "displacement_A", "mean_spin_A"]
    a, b = kinematic_features(lab, t_end=1.0), kinematic_features(labels(lambda tau: 2 * tau), t_end=1.0)
    assert a.shape == (4,) and a[1] == pytest.approx(1.5) and b[1] == pytest.approx(1.0) and a[0] > b[0]
    with pytest.raises(ValueError, match="per_frame"):
        kinematic_features({**lab, "per_frame": []})
