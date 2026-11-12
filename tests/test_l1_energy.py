"""L1 energy (l1_forecast_laws): eps of the exchange event decoded with the R1 instrument, creation / lower-tail rates
against the floor, the instrument check, and run() end to end on a fake F1 cache (S_mom, observers, energy)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from physlaw.data import collision_physics as cp
from physlaw.eval import l1_forecast_laws as L1
from physlaw.probes.ridge import _SVDRidge

LADDER = np.round(np.arange(0.5, 1.51, 0.1), 2)


def _ladder_arrays(n_groups=200, noise=0.03, seed=0):
    rng = np.random.default_rng(seed)
    eps = np.tile(LADDER, n_groups)
    groups = np.repeat([f"g{i}" for i in range(n_groups)], len(LADDER))
    return rng, eps, eps + noise * rng.normal(size=eps.size), groups


@pytest.mark.parametrize("e_model, flagged", [(1.4, True), (0.8, False)])
def test_creation_is_flagged_at_1_4_not_at_0_8(e_model, flagged):
    rng, eps, fl, groups = _ladder_arrays()
    fc = e_model + 0.03 * rng.normal(size=eps.size)       # the model's own eps, whatever the twin
    ok = np.ones(eps.size, bool)
    r = L1.energy_scores(eps, fl, fc, ok, ok, groups, n_boot=200, rng=rng)
    assert 0 < r["tau"] < 0.1 and r["eps_lo"] == 0.5 and r["status"] == "measured"
    assert r["eps_lo_floor_median"] == pytest.approx(0.5, abs=0.05)       # the lower tail reads the floor decode
    assert r["floor"]["creation_rate"] <= 0.06 and r["floor"]["lower_tail_rate"] <= 0.06
    assert r["forecast"]["eps_median"] == pytest.approx(e_model, abs=0.02)
    if flagged:
        assert r["forecast"]["creation_rate"] > 0.95 and r["forecast"]["creation_ci95"][0] > 0.9
    else:
        assert r["forecast"]["creation_rate"] < 0.02 and r["forecast"]["lower_tail_rate"] < 0.02
    assert "necessary, not sufficient" in r["note"]


def test_excess_noise_inflates_both_tails_equally():
    rng, eps, fl, groups = _ladder_arrays(n_groups=600)
    ok = np.ones(eps.size, bool)
    fc = eps + 0.3 * rng.normal(size=eps.size)            # law-perfect on average, but noisy
    r = L1.energy_scores(eps, fl, fc, ok, ok, groups)
    up, lo = r["forecast"]["creation_rate"], r["forecast"]["lower_tail_rate"]
    assert up > 4 * r["floor"]["creation_rate"] and lo > 4 * r["floor"]["lower_tail_rate"]
    assert up == pytest.approx(lo, abs=0.03)
    # a dissipative bias moves only the lower tail
    r = L1.energy_scores(eps, fl, eps - 0.4, ok, ok, groups)
    assert r["forecast"]["creation_rate"] == 0 and r["forecast"]["lower_tail_rate"] > 0.3


def test_instrument_check_marks_a_blind_floor_unmeasurable():
    rng, eps, fl, groups = _ladder_arrays(n_groups=60)
    ok = np.ones(eps.size, bool)
    r = L1.energy_scores(eps, fl, fl, ok, ok, groups, n_boot=100, rng=rng)
    assert r["status"] == "measured" and r["floor_check"]["slope"] == pytest.approx(1, abs=0.05)
    blind = 0.9 + 0.05 * rng.normal(size=eps.size)        # the floor does not see eps
    r = L1.energy_scores(eps, blind, blind, ok, ok, groups, n_boot=100, rng=rng)
    assert r["status"] == "unmeasurable" and r["floor_check"]["r2"] < 0.1
    r = L1.energy_scores(eps, fl, fl, np.zeros(eps.size, bool), ok, groups)
    assert r["status"] == "no gated floor clips"


def _bounce(n_groups=80, seed=0):
    """F7-bounce-like ladder: per group one fall (shared pre state), post v' = (v_x, -eps v_z) for every rung."""
    rng = np.random.default_rng(seed)
    pre, post, eps, groups = [], [], [], []
    for g in range(n_groups):
        v = np.array([rng.uniform(-1.5, 1.5), 0.0, -rng.uniform(2.0, 6.0)])
        for e in LADDER:
            pre.append(v)
            post.append(v * np.array([1.0, 1.0, -e]))
            eps.append(e)
            groups.append(f"g{g:03d}")
    return np.array(pre), np.array(post), np.array(eps), np.array(groups), rng


def _emb(rng, D=24, noise=0.6):
    W = rng.normal(size=(3, D))
    return lambda v: v @ W + noise * rng.normal(size=(len(v), D))


@pytest.mark.parametrize("gain", [1.0, 0.3])
def test_r1_calibration_common_gain_neither_creates_nor_hides_energy(gain):
    """Noisy features make ridge shrink hard; after the fold-proper calibration the floor reads eps without bias. A
    forecast in a gained / shifted feature space (moment-matched) at eps = 1 does not read as creation."""
    vp, vq, eps, groups, rng = _bounce()
    emb = _emb(rng)
    ctx = {"event": "bounce", "g": np.array([0.0, 0.0, -9.81])}
    v_fc = vp * np.array([1.0, 1.0, -1.0])                # the model: eps = 1 for every rung
    X_fc = gain * emb(v_fc) + 3.0
    d = L1.r1_decode(emb(vp), emb(vq), vp, vq, groups, {"forecast": (X_fc, X_fc)}, k=5, n_seeds=1)
    e = lambda a, b: np.array([L1._eps(a[i:i + 1], b[i:i + 1], ctx) for i in range(len(a))])
    e_fl, e_fc = e(d["pre"], d["post"]), e(d["pre"], d["forecast"])
    assert d["r2_post"] > 0.8 and np.median(e_fl - eps) == pytest.approx(0, abs=0.03)
    raw = _SVDRidge(emb(vq), vq, True).predict(emb(vq), 1e5)            # uncalibrated, heavily shrunk
    assert np.ptp(raw[:, 2]) < 0.5 * np.ptp(vq[:, 2])
    ok = np.ones(len(eps), bool)
    r = L1.energy_scores(eps, e_fl, e_fc, ok, ok, groups)
    assert r["status"] == "measured" and r["floor_check"]["slope"] == pytest.approx(1, abs=0.1)
    assert r["forecast"]["eps_median"] == pytest.approx(1.0, abs=0.08)
    assert r["forecast"]["creation_rate"] < 0.25 and r["forecast"]["lower_tail_rate"] < 0.05
    # every decode scaled by a common gain: eps unchanged
    assert np.allclose(e(0.6 * d["pre"], 0.6 * d["forecast"]), e_fc)


def test_two_spaces_need_one_calibrated_probe_each():
    """pred_real: pre rows in the encoder space, post rows in the predictor space = a rotation of it. One linear map
    across both spaces cannot fit (3-D features: it would have to equal its own rotation); a calibrated probe per space
    recovers eps for the floor and for a forecast in the predictor space."""
    vp, vq, eps, groups, rng = _bounce()
    W = rng.normal(size=(3, 3))
    R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    enc = lambda v: v @ W + 0.05 * rng.normal(size=v.shape)
    v_fc = vp * np.array([1.0, 1.0, -0.8])                # the model: eps = 0.8
    Xa, Xb, Xf = enc(vp), enc(vq) @ R, enc(v_fc) @ R
    ctx = {"event": "bounce", "g": np.array([0.0, 0.0, -9.81])}
    e = lambda a, b: np.array([L1._eps(a[i:i + 1], b[i:i + 1], ctx) for i in range(len(a))])
    sep = L1.r1_decode(Xa, Xb, vp, vq, groups, {"forecast": (Xf, None)}, k=5, n_seeds=1, joint=False)
    jnt = L1.r1_decode(Xa, Xb, vp, vq, groups, {"forecast": (Xf, None)}, k=5, n_seeds=1, joint=True)
    assert sep["r2_pre"] > 0.95 and sep["r2_post"] > 0.95 and jnt["r2_post"] < sep["r2_post"] - 0.05
    err = lambda d: np.median(np.abs(e(d["pre"], d["post"]) - eps))
    assert err(sep) < 0.05 and err(jnt) > 3 * err(sep)
    assert np.median(e(sep["pre"], sep["forecast"])) == pytest.approx(0.8, abs=0.05)


def _scenes(n_groups=30, seed=0):
    """{clip: (pre, post, ctx)}: random 2-D collisions (unequal masses), F1's five twins."""
    rng = np.random.default_rng(seed)
    twins = {"conserve_e1": 1.0, "conserve_e015": 0.15, "energy_create": 1.5, "mom_violate_e1": 1.0}
    states, st_of = {}, {}
    for g in range(n_groups):
        mA, mB = rng.uniform(0.3, 1.0), rng.uniform(1.3, 3.0)
        n = rng.normal(size=2); n /= np.linalg.norm(n)
        vB = rng.normal(size=2)
        vA = vB + n * rng.uniform(0.8, 3.0) + np.array([-n[1], n[0]]) * rng.normal()
        st = cp.PreState(mA, mB, vA, vB, n)
        pre = {"v": np.stack([vA, vB])}
        for t, e in twins.items():
            dv = cp.legal_outcome(st, e).reshape(2, 2)
            if t.startswith("mom_violate"):
                dv = dv - 0.3 * n                          # common drift: momentum created, eps unchanged
            states[f"g{g:03d}/{t}"] = (pre, {"v": pre["v"] + dv}, {"m": np.array([mA, mB]), "n": n})
            st_of[f"g{g:03d}/{t}"] = st
    return states, st_of


def _labels(g, t, pre, post, ctx):
    n = ctx["n"]
    return {"physlaw": {"family": "f1", "twin": t, "twin_group_id": g, "pre_segment": "0",
                        "objects": {o: {"mass": float(ctx["m"][i]), "radius": 0.1} for i, o in enumerate("AB")},
                        "contact_normal": [float(n[0]), float(n[1]), 0.0],
                        "pre_velocity": {o: [*map(float, pre["v"][i]), 0.0] for i, o in enumerate("AB")},
                        "post_velocity": {o: [*map(float, post["v"][i]), 0.0] for i, o in enumerate("AB")}},
            "collision": {"contact_point_world": [0.0, 0.0, 0.0]}}


@pytest.fixture(scope="module")
def f1_cache(tmp_path_factory):
    states, st_of = _scenes()
    root = tmp_path_factory.mktemp("f1")
    data, cache = root / "data", root / "cache"
    cache.mkdir()
    recs = []
    for c, (pre, post, ctx) in states.items():
        g, t = c.split("/")
        (data / c).mkdir(parents=True)
        (data / c / "labels.json").write_text(json.dumps(_labels(g, t, pre, post, ctx)))
        recs.append({"clip": c, "group": g, "twin": t, "family": "f1", "file": f"{c}.npz"})
    (cache / "_index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    np.savez(cache / "_maskfeat_v2.npz", clips=np.array(sorted(states)))
    return states, st_of, data, cache


@pytest.mark.parametrize("e_model, pred_real", [(1.4, False), (0.5, False), (1.4, True), (0.5, True)])
def test_run_f1_end_to_end(f1_cache, monkeypatch, e_model, pred_real):
    """Features linear in each ball's velocity; the forecast (shared by a group's twins) is the impulse law at the
    model's e: law-perfect for momentum, so S_mom ~ 1 when it is read as is. pred_real: the post rows are the model's
    own (noisier) forecast-space outputs for the real post frames, the primary instrument when cached."""
    from physlaw.config import load_config
    states, st_of, data, cache = f1_cache
    clips = sorted(states)
    rng = np.random.default_rng(0)
    W = rng.normal(size=(4, 16))
    emb = lambda v: v.reshape(-1) @ W + 0.02 * rng.normal(size=16)
    fc_v = {c: states[c][0]["v"] + cp.legal_outcome(st_of[c], e_model).reshape(2, 2) for c in clips}
    real = {"frame.post": np.array([emb(states[c][1]["v"]) for c in clips]),
            "frame.pre": np.array([emb(states[c][0]["v"]) for c in clips])}
    fc = {"frame.post": np.array([emb(fc_v[c]) for c in clips])}      # the same layer as the real pools
    sp = {"real": real, "fc": fc}
    if pred_real:
        sp["pr"] = {"frame.post": np.array([emb(states[c][1]["v"]) + 0.05 * rng.normal(size=16) for c in clips])}
    monkeypatch.setattr(L1, "_load_spaces", lambda *a, **k: {"final": sp})
    cfg = load_config("configs/eval/l1_forecast_laws.yaml")
    cfg["fold_seeds_l1"] = 1
    cfg["bootstrap"] = {"n": 50}
    cfg["selection"] = {"n_splits": 4}
    res = L1.run(cache, data, cfg, {}, log=lambda *a: None)
    v = res["variants"]["energy|final|frame"]
    assert v["probe_r2"]["A"]["pre"] > 0.9 and v["probe_r2"]["A"]["post"] > 0.9
    m, e = v["momentum"], v["energy"]
    assert v["instrument"] == ("pred_real" if pred_real else "as_is")
    assert m["interaction_rate"] > 0.9 and m["S"] > 0.9 and m["S_ci95"][0] > 0.8
    assert m["mass_free"]["normal_A"]["forecast"] > 0.9 and m["family"]["S"] > 0.5
    assert e["status"] == "measured" and e["floor_check"]["slope"] == pytest.approx(1, abs=0.15)
    want = {"forecast_mm", *L1.CONTROLS} | ({"floor_encoder"} if pred_real else set())
    assert set(e["observers"]) == want and set(m["observers"]) == want
    # M4: S_mom reads decoded pre AND post velocities (the old true-pre mix is a diagnostic)
    assert m["pre_velocities"].startswith("decoded") and m["S_true_pre"] > 0.9
    # H4: the static control moves no momentum (gated out), other sets' outcomes break momentum: the forecast beats all
    o = m["observers"]
    assert o["ctl_static"]["n_scored_clips"] == 0 and o["ctl_other_real"]["S"] < 0.5 and o["ctl_other_forecast"]["S"] < 0.5
    assert m["verdict"]["claim_allowed"] and m["verdict"]["verdict"] == "beats every negative control"
    assert 0.9 < m["pass_share"] <= 1 and m["S_sensitivity"] <= m["S"] + 1e-9          # M1
    assert "cannot detect the drift edit" in m["mass_free"]["role"]                     # N1
    assert e["floor"]["creation_rate"] < 0.1
    if e_model > 1:
        assert e["forecast"]["creation_rate"] > 0.8
    else:
        assert e["forecast"]["creation_rate"] < 0.05 and e["forecast"]["eps_median"] == pytest.approx(0.5, abs=0.15)
    assert set(res["selected"]["energy|frame"]) == {"momentum", "floor_best"}
    md = L1.table(res)
    assert "S_mom" in md and "creation rate" in md and "necessary, not sufficient" in md
    json.dumps(res, default=float)
