"""Tests for physics_interp.momentum — the identities the register quotes as controls.

    python tests/test_momentum.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics_interp.momentum import physics as ph
from physics_interp.momentum import stats
from physics_interp.momentum import synthetic as syn


def _random_pre(rng):
    m_a, m_b = rng.uniform(0.2, 5.0, size=2)
    v_a, v_b = rng.normal(size=2), rng.normal(size=2)
    th = rng.uniform(0, 2 * np.pi)
    n = np.array([np.cos(th), np.sin(th)])
    if np.dot(v_a - v_b, n) <= 0:
        n = -n
    return ph.PreState(m_a, m_b, v_a, v_b, n)


def test_legal_outcome_conserves_momentum_and_realises_e():
    rng = np.random.default_rng(0)
    for _ in range(200):
        pre = _random_pre(rng)
        e = rng.uniform(0, 1)
        o = ph.legal_outcome(pre, e)
        assert np.linalg.norm(ph.delta_p(o, pre)) < 1e-12
        assert abs(ph.implied_restitution(o, pre) - e) < 1e-12
        d = ph.decompose(o, pre)
        # the §47 geometry control: drift ~0, tangential ~0
        assert abs(d["drift_n"]) < 1e-12 and abs(d["drift_t"]) < 1e-12
        assert abs(d["J_tangential"]) < 1e-12
        assert d["J_normal"] > 0
        assert ph.distance_to_legal_line(o, pre) < 1e-12
        assert ph.membership_score(o, pre) < 1e-12


def test_basis_decomposition_roundtrips():
    rng = np.random.default_rng(1)
    for _ in range(100):
        pre = _random_pre(rng)
        B = ph.outcome_basis(pre)
        coef = rng.normal(size=4)
        o = B @ coef
        d = ph.decompose(o, pre)
        assert np.allclose([d["J_normal"], d["J_tangential"], d["drift_n"], d["drift_t"]], coef)
        # drift is exactly the momentum change, impulses contribute nothing
        dp = ph.delta_p(o, pre)
        expect = (pre.m_a + pre.m_b) * (coef[2] * pre.n + coef[3] * ph.perp(pre.n))
        assert np.allclose(dp, expect)


def test_split_edit_is_a_common_drift_along_n_keeping_separation_speed():
    rng = np.random.default_rng(2)
    for _ in range(100):
        pre = _random_pre(rng)
        e = rng.uniform(0, 1)
        o = ph.legal_outcome(pre, e)
        pa, pb = pre.v_a + o[:2], pre.v_b + o[2:]
        ds = rng.uniform(-0.4, 0.4)
        pa2, pb2, rec = ph.split_edit(pre, pa, pb, ds)
        o2 = np.concatenate([pa2 - pre.v_a, pb2 - pre.v_b])
        # restitution unchanged (a + b held)
        assert abs(ph.implied_restitution(o2, pre) - e) < 1e-12
        d = ph.decompose(o2, pre)
        assert abs(d["drift_t"]) < 1e-12 and abs(d["J_tangential"]) < 1e-12
        assert abs(d["drift_n"] + rec["delta_speed"]) < 1e-12
        # it breaks momentum unless ds == 0
        assert np.linalg.norm(ph.delta_p(o2, pre)) > 1e-6
        # the §57h closed form: cos between the violation and the legal line
        viol = o2 - o
        b = ph.legal_direction(pre)
        cos = abs(viol @ b) / (np.linalg.norm(viol) * np.linalg.norm(b))
        assert abs(cos - ph.drift_legal_cosine(pre.m_a, pre.m_b)) < 1e-9


def test_restitution_edit_is_exact():
    rng = np.random.default_rng(3)
    for _ in range(100):
        pre = _random_pre(rng)
        e_sim, e_new = rng.uniform(0, 1), rng.uniform(0, 5)
        o = ph.legal_outcome(pre, e_sim)
        pa, pb = pre.v_a + o[:2], pre.v_b + o[2:]
        pa2, pb2, rec = ph.restitution_edit(pre, pa, pb, e_sim, e_new)
        o2 = np.concatenate([pa2 - pre.v_a, pb2 - pre.v_b])
        assert np.allclose(o2, ph.legal_outcome(pre, e_new))
        assert np.linalg.norm(ph.delta_p(o2, pre)) < 1e-12
        assert abs(ph.implied_restitution(o2, pre) - e_new) < 1e-9
        ke = lambda va, vb: 0.5 * pre.m_a * va @ va + 0.5 * pre.m_b * vb @ vb
        dke = ke(pa2, pb2) - ke(pre.v_a, pre.v_b)
        assert abs(dke - rec["dke_closed_form"]) < 1e-9
        assert rec["impossible"] == (e_new > 1)
    # identity at e_new == e_sim
    pre = _random_pre(rng)
    o = ph.legal_outcome(pre, 0.4)
    pa, pb = pre.v_a + o[:2], pre.v_b + o[2:]
    pa2, pb2, _ = ph.restitution_edit(pre, pa, pb, 0.4, 0.4)
    assert np.allclose(pa, pa2) and np.allclose(pb, pb2)


def test_three_arms_are_collinear_with_chord_position_from_e():
    rng = np.random.default_rng(4)
    pre = _random_pre(rng)
    e = 0.5
    o = ph.legal_outcome(pre, e)
    pa, pb = pre.v_a + o[:2], pre.v_b + o[2:]
    sub = np.concatenate(ph.restitution_edit(pre, pa, pb, e, e - 0.2)[:2]) - np.concatenate([pre.v_a, pre.v_b])
    sup = np.concatenate(ph.restitution_edit(pre, pa, pb, e, e + 0.3)[:2]) - np.concatenate([pre.v_a, pre.v_b])
    chord = sup - sub
    t = (o - sub) @ chord / (chord @ chord)
    assert abs(t - 0.4) < 1e-12           # physics fixes it at 0.400 (§54b)
    assert np.linalg.norm(o - sub - t * chord) < 1e-12


def test_labels_roundtrip_through_accessors():
    rng = np.random.default_rng(5)
    clips = syn.make_conservation_set(n_seeds=10, rng_seed=1)
    for lab in clips:
        pre = ph.pre_state_from_labels(lab)
        o = ph.outcome_from_labels(lab)
        if lab["violates_momentum"]:
            assert np.linalg.norm(ph.delta_p(o, pre)) > 1e-6
            assert ph.membership_score(o, pre) > 1e-3
        else:
            assert np.linalg.norm(ph.delta_p(o, pre)) < 1e-10
            assert ph.membership_score(o, pre) < 1e-10
        assert abs(ph.clip_restitution(lab) - lab["restitution"]) < 1e-9
    assert sum(l["violates_momentum"] for l in clips) == 10
    assert {ph.clip_seed(l) for l in clips} == set(range(101, 111))


def test_restitution_label_trap_is_caught():
    clips = syn.make_ladder_set(n_seeds=2, rng_seed=2)
    arm = [c for c in clips if c["arm"] == "e2"][0]
    assert arm["restitution"] == 1.0                       # the trap: simulator value
    assert abs(ph.clip_restitution(arm) - 2.0) < 1e-12     # the right field
    broken = {k: v for k, v in arm.items() if k != "restitution_realised"}
    try:
        ph.clip_restitution(broken)
    except KeyError:
        pass
    else:
        raise AssertionError("an edited arm without restitution_realised must refuse")


def test_auc_and_null():
    rng = np.random.default_rng(6)
    y = np.r_[np.zeros(100), np.ones(100)].astype(bool)
    s = rng.normal(size=200) + 2.0 * y
    assert 0.9 < stats.auc(s, y) <= 1.0
    assert stats.auc(y.astype(float), y) == 1.0
    assert stats.auc(np.zeros(200), y) == 0.5
    null = stats.permutation_null(rng.normal(size=200), y, draws=300)
    assert null["p"] > 0.05 and abs(null["null_mean"] - 0.5) < 0.05
    sig = stats.permutation_null(s, y, draws=300)
    assert sig["p"] < 0.01
    # grouped null: within-twin permutation
    g = np.repeat(np.arange(100), 2)
    yy = np.tile([False, True], 100)
    ss = rng.normal(size=200) + 1.5 * yy
    assert stats.permutation_null(ss, yy, draws=200, groups=g)["p"] < 0.01


def test_grouped_folds_never_split_a_seed_and_cv_predict_recovers_signal():
    rng = np.random.default_rng(7)
    g = np.repeat(np.arange(60), 2)
    folds = stats.grouped_folds(g, 5)
    assert sorted(np.concatenate(folds)) == list(range(120))
    for f in folds:
        seeds = set(g[f])
        assert all(np.isin(np.flatnonzero(np.isin(g, list(seeds))), f).all() for _ in [0])
    Y = rng.normal(size=(120, 2))
    X = Y @ rng.normal(size=(2, 40)) + 0.3 * rng.normal(size=(120, 40))
    pred = stats.cv_predict(X, Y, g)
    assert stats.r2(Y, pred) > 0.8
    pred_noise = stats.cv_predict(rng.normal(size=(120, 40)), Y, g)
    assert stats.r2(Y, pred_noise) < 0.2
    # fit_mask: train on even groups only, still predicts everything
    pred_h = stats.cv_predict(X, Y, g, fit_mask=(g % 2 == 0))
    assert np.isfinite(pred_h).all() and stats.r2(Y, pred_h) > 0.7


def test_curve_helpers():
    e = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    X = np.stack([e, e ** 2, np.ones_like(e)], axis=1)
    coef = stats.polyfit_curve(e, X, 2)
    assert np.allclose(stats.eval_curve(coef, e), X)
    d, e_at = stats.nearest_on_curve(coef, stats.eval_curve(coef, 0.55)[0], 0.0, 1.2)
    assert d < 1e-6 and abs(e_at - 0.55) < 1e-3
    assert abs(stats.participation_ratio([1, 1, 1, 1]) - 4) < 1e-12
    assert abs(stats.participation_ratio([1, 0, 0]) - 1) < 1e-12
    assert abs(stats.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-12
    assert abs(stats.spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-12


def test_synthetic_sets_have_the_register_structure():
    cons = syn.make_conservation_set(n_seeds=6)
    assert len(cons) == 12
    es = sorted({c["restitution_realised"] for c in cons})
    assert es == [0.15, 1.0]
    lad = syn.make_ladder_set(n_seeds=2)
    assert len(lad) == 22 and sum(c["arm"] == "honest" for c in lad) == 2
    three = syn.make_three_arm_set(n_seeds=3)
    assert len(three) == 9 and {c["arm"] for c in three} == {"sub", "anchor", "super"}
    scene = syn.make_ladder_set(n_seeds=4, arms=syn.LADDER_LEGAL + syn.LADDER_IMPOSSIBLE,
                                with_violating=True, pinned=(1.0, 0.15))
    assert len(scene) == 4 * 13
    # twins share the pre-contact segment bit for bit
    for i in range(4):
        group = [c for c in scene if c["seed_group"] == 9500 + i]
        pre0 = group[0]["collision"]["pre"]
        assert all(c["collision"]["pre"] == pre0 for c in group)
        assert all(c["per_frame"][18] == group[0]["per_frame"][18] for c in group)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
