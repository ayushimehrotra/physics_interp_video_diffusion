"""Momentum package on generated outcomes (physlaw.eval.outputs): track A S_mom, track B mass-free checks."""
import json

import numpy as np
import pytest

from physlaw.eval import outputs as O


def _legal_post(va, vb, n, ma, mb, e):
    va, vb, n = (np.asarray(x, float) for x in (va, vb, n))
    n = n / np.linalg.norm(n)
    J = (1 + e) * ma * mb / (ma + mb) * float((va - vb) @ n)
    return va - J * n / ma, vb + J * n / mb


def _labels(va, vb, post_a, post_b, n=(1.0, 0.0), ma=1.0, mb=2.0, ra=0.4, rb=0.5):
    return {"physlaw": {"family": "f1", "pre_velocity": {"A": list(va), "B": list(vb)},
                        "post_velocity": {"A": list(post_a), "B": list(post_b)}, "contact_normal": list(n),
                        "objects": {"A": {"mass": ma, "radius": ra}, "B": {"mass": mb, "radius": rb}}},
            "collision": {"contact_point_world": [0.0, 0.0, 0.0]}}


# ---------------------------------------------------------------- track A

VA, VB, N, MA, MB = (1.0, 0.0), (0.0, 0.0), (1.0, 0.0), 1.0, 2.0
TWINS = {"conserve_e1": 1.0, "conserve_e015": 0.15}
GROUPS = [f"g{i}" for i in range(6)]


def _lab_fn():
    lab = {t: _labels(VA, VB, *_legal_post(VA, VB, N, MA, MB, e), N, MA, MB) for t, e in TWINS.items()}
    return lambda g, t: lab[t]


def _real_rows(noise=0.01, blind=False):
    rng = np.random.default_rng(1)
    rows = []
    for g in GROUPS:
        for t, e in TWINS.items():
            pa, pb = _legal_post(VA, VB, N, MA, MB, e)
            if blind:                                               # equal and opposite velocity changes
                h = ((pa - np.array(VA)) - (pb - np.array(VB))) / 2
                pa, pb = np.array(VA) + h, np.array(VB) - h
            v = np.stack([pa, pb]) + rng.normal(0, noise, (2, 2))
            rows.append({"group": g, "twin": t, "measurable": True, "v_post": v.tolist()})
    return rows


def _gen_rows(kind):
    rows = []
    for g in GROUPS:
        for seed in range(2):
            pa, pb = _legal_post(VA, VB, N, MA, MB, 0.5)
            if kind == "blind":
                h = ((pa - np.array(VA)) - (pb - np.array(VB))) / 2
                pa, pb = np.array(VA) + h, np.array(VB) - h
            elif kind == "pass":
                pa, pb = np.array(VA), np.array(VB)
            rows.append({"group": g, "seed": seed, "measurable": True, "v_post": [list(pa), list(pb)]})
    return rows


def test_track_a_perfect_momentum_scores_one():
    m = O.track_a_momentum(_gen_rows("legal"), _real_rows(), _lab_fn(), list(TWINS), n_boot=200)
    for t in (*TWINS, "legal"):
        d = m[t]
        assert d["interaction_rate"] == 1.0 and d["n_scored_clips"] == (24 if t == "legal" else 12)
        assert d["eta_generated"] == pytest.approx(0.0, abs=1e-12)
        assert 0 < d["eta_real"] < 0.1 and d["eta_blind"] == pytest.approx(1 / 3)
        assert d["S_mom"] == pytest.approx(1.0, abs=1e-9) and d["S_mom_defined"]
        assert d["antiparallel_generated"] == pytest.approx(1.0)


def test_track_a_mass_blind_scores_zero():
    m = O.track_a_momentum(_gen_rows("blind"), _real_rows(), _lab_fn(), list(TWINS), n_boot=200)
    for t in (*TWINS, "legal"):
        assert m[t]["eta_generated"] == pytest.approx(1 / 3) and m[t]["S_mom"] == pytest.approx(0.0, abs=1e-9)
        assert m[t]["S_mom_ci95"] == pytest.approx([0.0, 0.0], abs=1e-9)
        assert m[t]["antiparallel_generated"] == pytest.approx(1.0)        # mass-free check cannot see mass-blindness


def test_track_a_pass_through_is_gated_out():
    m = O.track_a_momentum(_gen_rows("pass"), _real_rows(), _lab_fn(), list(TWINS), n_boot=200)
    d = m["conserve_e1"]
    assert d["interaction_rate"] == 0.0 and d["n_scored_clips"] == 0 and np.isnan(d["S_mom"])
    assert d["interaction_rate_real"] == 1.0


def test_track_a_undefined_when_floor_reaches_blind():
    m = O.track_a_momentum(_gen_rows("legal"), _real_rows(noise=0.0, blind=True), _lab_fn(), list(TWINS), n_boot=50)
    assert not m["conserve_e1"]["S_mom_defined"] and np.isnan(m["conserve_e1"]["S_mom"])


def test_track_a_summary_has_momentum_and_legacy(tmp_path):
    lab = _lab_fn()
    for g in GROUPS:
        for t in TWINS:
            (tmp_path / g / t).mkdir(parents=True)
            (tmp_path / g / t / "labels.json").write_text(json.dumps(lab(g, t)))
    res = O.track_a_summary(_gen_rows("legal"), _real_rows(), tmp_path, ["momentum", "energy"], n_boot=50)
    d = res["twins"]["conserve_e1"]
    assert d["momentum"]["S_mom"] == pytest.approx(1.0) and "S_outcome" in d["legacy"] and "r_p_generated" in d["legacy"]
    assert "r_p_generated" not in d and res["momentum_legal"]["n_scored_clips"] == 24


# ---------------------------------------------------------------- track B

def _row_b(va, vb, post, n, ra, rb, dt=0.02, group="g0", seed=0):
    """Readout row: positions at a 'contact frame' dt after the true touch (the pre lines overlap there)."""
    va, vb, n = (np.asarray(x, float) for x in (va, vb, n))
    n = n / np.linalg.norm(n)
    c = np.array([0.3, -0.2])
    xa, xb = c - ra * n + va * dt, c + rb * n + vb * dt
    return {"group": group, "seed": seed, "measurable": True, "velocity_pre": [va.tolist(), vb.tolist()],
            "velocity_post": [list(post[0]), list(post[1])], "position_pre": [xa.tolist(), xb.tolist()]}


def test_track_b_legal_outcomes_any_masses():
    rng = np.random.default_rng(0)
    for _ in range(50):
        ma, mb = rng.uniform(0.2, 5.0, 2)
        e = rng.uniform(0, 1)
        ra, rb = rng.uniform(0.2, 0.6, 2)
        th = rng.uniform(0, 2 * np.pi)
        n = np.array([np.cos(th), np.sin(th)])
        t = np.array([-n[1], n[0]])
        va = n * rng.uniform(0.5, 2) + t * rng.uniform(-1, 1)
        vb = -n * rng.uniform(0, 1) + t * rng.uniform(-1, 1)
        post = _legal_post(va, vb, n, ma, mb, e)
        lab = _labels(va, vb, *post, n, ma, mb, ra, rb)
        f = O.mass_free_clip(_row_b(va, vb, post, n, ra, rb), lab)
        assert f["mf_interacts"] and not f["mf_contact_miss"]
        assert f["mf_antiparallel"] == pytest.approx(1.0)
        assert f["mf_normal"] == pytest.approx(1.0) and f["mf_normal_err_deg"] == pytest.approx(0.0, abs=1e-4)
        assert f["mf_e_hat"] == pytest.approx(e, abs=1e-9)
        assert f["mf_implied_ratio"] == pytest.approx(ma / mb) and f["mf_log_ratio_vs_label"] == pytest.approx(0.0, abs=1e-9)


def test_track_b_same_direction_push_and_pass_through():
    va, vb, n = (1.0, 0.0), (0.0, 0.0), (1.0, 0.0)
    lab = _labels(va, vb, va, vb, n)
    push = O.mass_free_clip(_row_b(va, vb, ([1.5, 0.0], [0.5, 0.0]), n, 0.4, 0.5), lab)
    assert push["mf_antiparallel"] == pytest.approx(-1.0)
    still = O.mass_free_clip(_row_b(va, vb, (va, vb), n, 0.4, 0.5), lab)
    assert not still["mf_interacts"]
    assert O.mass_free_clip({"measurable": False}, lab) == {}


def test_track_b_stats_e_share_and_ratio_consistency():
    va, vb, n = (1.0, 0.0), (0.0, 0.0), (1.0, 0.0)
    rows = []
    for g, (ma, mb) in {"g0": (1.0, 2.0), "g1": (3.0, 1.0)}.items():
        lab = _labels(va, vb, va, vb, n, ma, mb)
        for s, e in enumerate((0.3, 0.9)):                     # seeds of one scene: one mass ratio, any e
            rows.append({**_row_b(va, vb, _legal_post(va, vb, n, ma, mb, e), n, 0.4, 0.5, group=g, seed=s), "group": g})
            rows[-1].update(O.mass_free_clip(rows[-1], lab))
    st = O.mass_free_stats(rows, n_boot=50)
    assert st["interaction_rate"] == 1.0 and st["e_le1_share"] == 1.0 and st["antiparallel_mean"] == pytest.approx(1.0)
    assert st["ratio_consistency_groups"] == 2 and st["ratio_log_sd"] == pytest.approx(0.0, abs=1e-9)
    # one seed implies a different ratio, and creates energy (e = 1.5)
    lab = _labels(va, vb, va, vb, n, 1.0, 1.0)
    bad = {**_row_b(va, vb, _legal_post(va, vb, n, 1.0, 1.0, 1.5), n, 0.4, 0.5, group="g0", seed=2)}
    bad.update(O.mass_free_clip(bad, lab))
    st = O.mass_free_stats(rows + [bad], n_boot=50)
    assert st["e_le1_share"] == pytest.approx(0.8) and st["ratio_log_sd"] > 0.3
