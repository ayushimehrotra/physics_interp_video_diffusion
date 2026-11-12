"""L2 energy kink (surprise along the ε ladder), clean window, stop-dose exclusion and L2-rep (representational kink):
synthetic curves / activations, the labels-only observers (K = 0 by construction, so not run at runtime), end-to-end
records and md, and the F2 / F7 / F8 config."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from physlaw.data import twins as TW
from physlaw.eval import l2_surprise as L

REPO = Path(__file__).resolve().parents[1]
X = np.round(np.arange(16) * 0.1, 1)          # F2 ladder e0.0 .. e1.5


def _groups(f, G=30, seed=0, noise=0.0):
    """G groups of a*f(x) + b (+ noise): per-group scale and offset, which standardisation removes."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(G, 1)) + rng.uniform(0.5, 2.0, (G, 1)) * (f[None] + noise * rng.normal(size=(G, len(f))))


def test_pure_hinge_gives_positive_K():
    st = L.kink_stat(_groups(np.maximum(0.0, X - 1.0)), X, 1.0)
    K = np.array(st["K_g"])
    assert not st["inconclusive"] and st["x_hat"] == pytest.approx(0.0)
    assert np.all(K > 0.5)


def test_convex_parabola_at_08_gives_zero_K():
    st = L.kink_stat(_groups((X - 0.8) ** 2), X, 1.0)
    assert st["x_hat"] == pytest.approx(0.8, abs=1e-9)
    assert not st["inconclusive"]                       # |0.8 - 1| = 0.2 is not < 0.2
    assert np.allclose(st["K_g"], 0.0, atol=1e-9)


def test_linear_motion_trend_gives_zero_K():
    st = L.kink_stat(_groups(2.0 + 3.0 * X), X, 1.0)
    assert not st["inconclusive"]
    assert np.allclose(st["K_g"], 0.0, atol=1e-9)


def test_v_at_x0_is_inconclusive():
    st = L.kink_stat(_groups(np.abs(X - 1.0)), X, 1.0)
    assert st["inconclusive"] and "cannot be told apart" in st["reason"]
    assert st["x_hat"] == pytest.approx(1.0, abs=1e-9)
    # a V away from x0 is excluded from the reference points: K = 0
    st = L.kink_stat(_groups(np.abs(X - 0.5)), X, 1.0)
    assert not st["inconclusive"] and np.allclose(st["K_g"], 0.0, atol=1e-9)


def test_ladder_without_both_sides_and_bad_steps():
    x = X[:11]                                          # e0.0..e1.0: x0 is the last point
    st = L.kink_stat(_groups(x ** 2), x, 1.0)
    assert st["inconclusive"] and "both sides" in st["reason"]
    with pytest.raises(ValueError):
        L.kink_stat(np.zeros((2, 4)), [0.0, 0.1, 0.3, 0.4], 0.1)


def test_x_hat_is_held_out():
    C = _groups((X - 0.4) ** 2, G=5)
    C[0] = (X - 1.3) ** 2                               # one odd group: its own x_hat comes from the others
    st = L.kink_stat(C, X, 1.0)
    assert st["x_hat_fit"] == "leave-one-group-out"
    assert st["x_hat_g"][0] == pytest.approx(0.4, abs=1e-9)
    assert L.kink_stat(C[:1], X, 1.0)["x_hat_fit"].startswith("in-sample")
    # logo_x_hat (the ONE x̂ per group shared by the speed-match exclusion and K): equal to kink_stat's own
    # leave-one-group-out fit when every group is a fit group; a group outside the fit set gets the all-fit-group x̂
    xh = L.logo_x_hat({g: C[g] for g in range(5)}, list(range(5)), X)
    assert [xh[g] for g in range(5)] == pytest.approx(st["x_hat_g"])
    xh = L.logo_x_hat({g: C[g] for g in range(5)}, [1, 2, 3, 4], X)
    assert xh[0] == pytest.approx(L.fit_x_hat(L._standardise(C[1:]), X))
    given = L.kink_compare(C, X, 1.0, x_hat=[xh[g] for g in range(5)])["model"]
    assert given["x_hat_fit"] == "given" and given["x_hat"] == pytest.approx(np.median([xh[g] for g in range(5)]))


def _observers(x, x0, x_hat, speeds):
    """Labels-only reference observers (not run at runtime: these tests pin what they would give). law: distance to
    the nearest legal outcome max(0, x − x0) (a hinge); novelty: |x − x̂| and (x − x̂)² (expects x̂ only); speed_only:
    |S(x) − S(x̂)| with S the total post-event speed; motion: S(x)."""
    G = len(speeds)
    rep = lambda c: np.repeat(np.asarray(c, float)[None], G, 0)
    S = np.asarray(speeds, float)
    s_hat = np.array([np.interp(x_hat, x, r) for r in S])
    return {"law": rep(np.maximum(0.0, x - x0)), "novelty_abs": rep(np.abs(x - x_hat)),
            "novelty_sq": rep((x - x_hat) ** 2), "speed_only": np.abs(S - s_hat[:, None]), "motion": S}


def test_observers_are_zero_by_construction():
    """x̂ exclusion and median-D² subtraction put the novelty, speed-only and motion observers at K = 0 on affine
    kinematics; the law hinge has K > 0."""
    obs = _observers(X, 1.0, 0.5, np.tile(1.0 + 0.8 * X, (4, 1)))
    K = {k: np.nanmean(L.kink_stat(c, X, 1.0)["K_g"]) for k, c in obs.items()}
    assert K["law"] > 0.5
    for k in ("novelty_abs", "novelty_sq", "speed_only", "motion"):
        assert abs(K[k]) < 1e-9, k


def test_smooth_varying_curvature_is_detrended():
    """K subtracts the smooth curvature trend, not just a constant: exp(ε) and |v'| hyperbolas (a ball stopping beyond
    the ladder, or inside it away from 1 with its neighbourhood left out of the fit) give |K| < 0.01 sd units."""
    assert abs(L.kink_stat(np.exp(X)[None], X, 1.0)["K_g"][0]) < 0.01
    assert abs(L.kink_stat(np.exp(2 * X)[None], X, 1.0)["K_g"][0]) < 0.01
    for stop, c in ((2.1, 0.05), (2.1, 0.3), (-0.6, 0.1), (0.5, 0.02), (1.25, 0.05)):
        h = np.sqrt((X - stop) ** 2 + c ** 2)[None]
        assert abs(L.kink_stat(h, X, 1.0, x_hat=0.0, stops=[[stop]])["K_g"][0]) < 0.01, stop
    hinge_on_exp = np.maximum(0.0, X - 1.0) + np.exp(X)
    K = L.kink_stat(hinge_on_exp[None], X, 1.0)["K_g"][0]
    assert K == pytest.approx(0.1 / np.std(hinge_on_exp), rel=0.05)    # the hinge's Δ² (h = 0.1) in sd units
    assert L.curvature_trend([0.1, 0.2, 0.3], [1.0, 2.0, 3.0], 1.0) == pytest.approx(10.0)
    assert L.curvature_trend([0.5], [2.0], 1.0) == 2.0


def _f2_ladder(rng):
    from tests.test_l2_matched import random_pre
    pre = random_pre(rng)
    dv = TW.CP.legal_outcome(pre, 1.0)
    tw = TW.f2_twins(pre, pre.v_a + dv[:2], pre.v_b + dv[2:], 1.0, TW.F2_LEVELS, (1.1, 1.2, 1.3, 1.4, 1.5))
    labs = [{"physlaw": {"objects": {"A": {"mass": pre.m_a}, "B": {"mass": pre.m_b}},
                         "post_velocity": {"A": tw[f"e{e:.1f}"].v_a, "B": tw[f"e{e:.1f}"].v_b}}} for e in X]
    vA, vB = (np.array([np.linalg.norm(getattr(tw[f"e{e:.1f}"], v)) for e in X]) for v in ("v_a", "v_b"))
    return labs, vA, vB


def test_speed_observers_on_f2_collision_ladders():
    """Real F2 ladders (twins.f2_twins + impossible levels), 200 groups, the kinematic-V rule with the observer's
    x̂ = 0.5: every kept, scorable group has |K| < 0.1 sd units for the motion observer S(x) (total speed), each ball's |v'| (a
    hyperbola) and the speed-only observer |S(x) − S(x̂)|; the law hinge > 0.5. Every group whose speed-only K is
    large (the second V of S returning to S(x̂) near 1, e.g. S bottoming near ε = 1 while both balls stop far from
    it: stops 1.64 / −1.93, K = 1.22 before this rule) is excluded."""
    rng = np.random.default_rng(1)
    K = {"motion": [], "speed_only": [], "ball": []}
    kept, reasons, whys = [], {"stop": 0, "speed_min": 0, "speed_match": 0}, []
    for _ in range(200):
        labs, vA, vB = _f2_ladder(rng)
        sd, kin = L.stop_doses(labs, X), L.ladder_kinematics(labs, X)
        mins, match = list(L.kinematic_minima(kin, X).values()), L.speed_match_doses(kin, X, 0.5)
        why = {"stop": L._near(sd, 1.0, 0.2), "speed_min": L._near(mins, 1.0, 0.2), "speed_match": L._near(match, 1.0, 0.2)}
        whys.append((sd, mins, why))
        for k, v in why.items():
            reasons[k] += v
        kept.append(not any(why.values()))
        obs = _observers(X, 1.0, 0.5, (vA + vB)[None])
        for k, c in (("motion", obs["motion"]), ("speed_only", obs["speed_only"]), ("ball", vA[None])):
            K[k].append(L.kink_stat(c, X, 1.0, x_hat=0.5, stops=[sd + mins + match])["K_g"][0])
    kept = np.array(kept)
    K = {k: np.abs(v) for k, v in K.items()}
    scored = kept & np.isfinite(K["motion"])              # NaN: every reference dose is near a kinematic V
    # measured: 18% excluded (stop 27, speed_min 8, speed_match 7 groups); 16 kept groups have no reference dose
    # left (x̂ = 0.5 and the V neighbourhoods leave only 0.1, 0.2, 1.3, 1.4); max |K| over the scored groups:
    # motion 0.060, speed-only 0.093, ball 0.034 (95th percentiles <= 0.02)
    assert 0.1 < 1 - kept.mean() < 0.3 and all(reasons.values()) and (kept & ~scored).sum() <= 0.12 * kept.sum()
    for k, v in K.items():
        assert v[scored].max() < 0.1 and np.quantile(v[scored], 0.95) < 0.03, k
    assert (K["speed_only"][~kept] > 0.3).any()                                         # what the rule removes
    # the case from the report: both balls stop far from 1 (1.64, -1.93) but S bottoms at 0.97 -> speed_min
    sd, mins, why = whys[118]
    assert sd == pytest.approx([1.64, -1.93], abs=0.01) and mins[0] == pytest.approx(0.972, abs=0.002)
    assert why == {"stop": False, "speed_min": True, "speed_match": False} and not kept[118]
    assert L.kink_stat(_observers(X, 1.0, 0.5, np.ones((1, len(X))))["law"], X, 1.0)["K_g"][0] > 0.5


def test_kinematic_v_helpers():
    """Analytic speeds from affine velocities: minima and the speed-match doses."""
    k = {"a": np.array([[-1.0, 0.0], [0.0, 1.0]]), "b": np.array([[1.0, 0.0], [0.0, 0.0]]), "m": np.array([1.0, 3.0])}
    # |v_A| = |x - 1| (stops at 1), |v_B| = 1: S = |x - 1| + 1 minimal at 1; S(0.5) = S(1.5)
    assert L.kinematic_minima(k, X) == pytest.approx({"S": 1.0, "KE": 1.0})
    assert L.speed_match_doses(k, X, 0.5) == pytest.approx([1.5], abs=1e-6)
    assert L.speed_match_doses(k, X, 1.4) == pytest.approx([0.6], abs=1e-6)
    assert L.speed_match_doses({**k, "a": np.array([[1.0, 0.0], [0.0, 1.0]])}, X, 0.5) == []      # monotone S
    labs = [{"physlaw": {"objects": {"A": {"mass": 1.0}, "B": {"mass": 3.0}},
                         "post_velocity": {"A": [e - 1.0, 0.0], "B": [0.0, 1.0]}}} for e in X]
    kin = L.ladder_kinematics(labs, X)
    assert np.allclose(kin["a"], k["a"]) and np.allclose(kin["b"], k["b"]) and np.allclose(kin["m"], [1.0, 3.0])


def test_kink_compare_model_beats_pixel_baselines():
    G = 40
    motion = 0.5 * X + 0.3 * X ** 2
    model = _groups(np.maximum(0.0, X - 1.0) + motion, G, seed=4, noise=0.02)
    pix = {"copy_last": _groups(motion, G, seed=5, noise=0.02), "const_velocity": _groups(motion, G, seed=6, noise=0.02)}
    rec = L.kink_compare(model, X, 1.0, 0.2, pix, n_boot=200)
    json.dumps(rec)
    assert rec["model"]["K_ci95"][0] > 0 and rec["model"]["share_K_pos"] > 0.9
    assert set(rec["vs_baselines"]) == set(L.KINK_BASELINES) == {"copy_last", "const_velocity"}
    assert rec["verdict"] == "kink beyond baselines"
    assert L.kink_compare(model, X, 1.0, 0.2, {"copy_last": pix["copy_last"]}, n_boot=50)["verdict"].startswith(
        "kink, not beyond")                                   # a missing pixel baseline never passes
    rec = L.kink_compare(_groups(motion, G, seed=7, noise=0.02), X, 1.0, 0.2, pix, n_boot=200)
    assert rec["verdict"] in ("no kink", "kink, not beyond every baseline") and abs(rec["model"]["K"]) < 0.2


def _fake_labels(g, t, x, v_stop=-1.0):
    """Collision-like labels whose ball A stops at x = v_stop (|v'_A| = |x - v_stop|), B moves 0.5 (1 + x)."""
    return {"physlaw": {"family": "f2", "objects": ["A", "B"], "twin": t, "epsilon": x,
                        "post_velocity": {"A": [x - v_stop, 0.0, 0.0], "B": [0.5 * (1 + x), 0.0, 0.0]},
                        "summary": {"ke_post": 0.5 * (1 + x) ** 2}}}


def _ladder(G=12, stop=lambda i: -1.0, seed=0):
    twins = [f"e{e:.1f}" for e in X]
    rng = np.random.default_rng(seed)
    sur, pix, labs = {}, {}, {}
    groups = [f"g{i:02d}" for i in range(G)]
    for i, g in enumerate(groups):
        a = rng.uniform(0.5, 2)
        for t, x in zip(twins, X):
            c = f"{g}/{t}"
            labs[c] = _fake_labels(g, t, x, stop(i))
            mot = (1.0 + x) * rng.uniform(0.95, 1.05)
            sur[c] = {"l2_obj": a * mot * np.exp(max(0.0, x - 1.0))}
            pix[c] = {"copy_last_mse_obj": mot, "constvel_px": mot * rng.uniform(0.9, 1.1)}
    return twins, groups, sur, pix, labs


def test_energy_kink_end_to_end_and_md():
    twins, groups, sur, pix, labs = _ladder()
    spec = {"name": "energy_kink", "kind": "energy_kink", "twins": twins, "x0": 1.0, "exclude_radius": 0.2}
    res = L.energy_kink(spec, groups, labs.__getitem__, sur, "model", pix, n_boot=100, log=lambda *a: None)
    json.dumps(res)
    assert res["n_groups_complete"] == 12 and res["status"] == "ok"
    assert res["excluded"] == {"stop": {}, "speed_min": {}} and res["groups_scored"] == groups
    assert res["distances"]["l2_obj"]["excluded_speed_match"] == {} and "excluded_groups" not in res["distances"]["l2_obj"]
    assert res["epsilon_by_dose"]["e1.3"] == pytest.approx(1.3)
    rec = res["distances"]["l2_obj"]
    assert rec["model"]["K"] > 0 and set(rec["baselines"]) == {"copy_last", "const_velocity"}
    assert rec["model"]["x_hat_fit"].startswith("leave-one-group-out over the groups kept")
    md = L.kink_table({"energy_kink": res})
    assert "Energy kink" in md and "l2_obj" in md
    empty = L.energy_kink({**spec, "twins": [f"eps{x:.1f}" for x in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1)]}, groups,
                          labs.__getitem__, sur, "model", pix, log=lambda *a: None)
    assert empty["n_groups_complete"] == 0 and empty["status"].startswith("no complete")


def test_stop_dose_exclusion():
    """Groups whose ball stops near ε = 1 (from labels stop_dose, or fitted from post_velocity) are excluded and
    scored apart."""
    x = X
    labs = [_fake_labels("g", f"e{e:.1f}", e, 0.95) for e in x]
    assert L.stop_doses(labs, x)[0] == pytest.approx(0.95)            # fitted: v'_A = x - 0.95
    labs[3]["physlaw"]["stop_dose"] = {"A": 0.4, "B": None}
    assert L.stop_doses(labs, x) == [0.4]                             # the label wins
    assert L.stop_doses([{"physlaw": {"objects": ["A"]}}], x[:1]) is None
    twins, groups, sur, pix, labs = _ladder(G=10, stop=lambda i: 1.05 if i < 3 else -1.0)
    spec = {"twins": twins, "x0": 1.0, "exclude_radius": 0.2}
    res = L.energy_kink(spec, groups, labs.__getitem__, sur, "model", pix, n_boot=50, log=lambda *a: None)
    assert set(res["excluded"]["stop"]) == set(groups[:3]) and res["n_groups_scored"] == 7
    assert res["groups_scored"] == groups[3:]
    r = res["distances"]["l2_obj"]
    assert r["n_groups_kept"] == 7 and r["excluded_groups"]["groups"] == groups[:3] and "model" in r["excluded_groups"]
    assert res["excluded"]["stop"]["g00"][0] == pytest.approx(1.05)
    assert res["excluded"]["speed_min"]["g00"]["S"] == pytest.approx(1.05)        # S = |x − 1.05| + 0.5 (1 + x)
    assert "[excluded]" in L.kink_table({"k": res})


def _pf(t, posA, posB=(9.0, 0.0), inA=True, r=0.5):
    return {"time_s": t, "A": {"position": [*posA, 0.0], "velocity": [1.0, 0.0, 0.0], "in_frame": inA},
            "B": {"position": [*posB, 0.0], "velocity": [0.0, 0.0, 0.0], "in_frame": True}}


def test_clean_until_from_per_frame_and_label():
    obj = {"A": {"radius": 0.5}, "B": {"radius": 0.5}}
    base = {"physlaw": {"objects": obj, "first_post_contact_frame": 1}, "scene_constants": {"frame_rate": 10}}
    # A leaves contact, separates, then touches B again at frame 4
    pf = [_pf(0.0, (0.0, 0.0), (1.0, 0.0)), _pf(0.1, (-0.5, 0.0), (1.0, 0.0)), _pf(0.2, (-1.0, 0.0), (1.0, 0.0)),
          _pf(0.3, (-0.5, 0.0), (1.0, 0.0)), _pf(0.4, (0.0, 0.0), (1.0, 0.0)), _pf(0.5, (0.0, 0.0), (1.0, 0.0))]
    assert L.clean_until_s({**base, "per_frame": pf}) == pytest.approx(0.4)
    pf2 = [_pf(0.1 * i, (-0.3 * i, 0.0), inA=i < 3) for i in range(6)]          # A leaves the frame at 0.3 s
    assert L.clean_until_s({**base, "per_frame": pf2}) == pytest.approx(0.3)
    pf3 = [_pf(0.1 * i, (-0.3 * i, 0.0)) for i in range(6)]
    assert L.clean_until_s({**base, "per_frame": pf3}) == pytest.approx(0.5)  # nothing happens: the clip end
    assert L.clean_until_s({"physlaw": {"clean_until_s": 0.25}}) == 0.25
    assert L.clean_until_s({"physlaw": {"objects": obj}}) is None
    assert L.event_time_s(base) == pytest.approx(0.1)


def test_window_surprise_and_apply_clean_window():
    rec = {"group": "g", "surprise": {"m": 9.0},
           "per_frame": {"t_s": [0.1, 0.2, 0.3, 0.4], "surprise": {"m": [1.0, 3.0, 100.0, None], "n": [1.0, 1.0, 1.0, 1.0]},
                         "weight": {"m": [1.0, 3.0, 1.0, 1.0]}}}
    assert L.window_surprise(rec, 0.2) == pytest.approx({"m": 2.5, "n": 1.0})       # (1 + 9) / 4
    assert L.window_surprise(rec, 0.05) is None
    idx = {"model": {"g/a": {"group": "g", "surprise": {"m": 5.0}}},
           "pixel_baselines": {"g/a": rec, "h/a": {**rec, "group": "h"}, "k/a": {**rec, "group": "k"}}}
    new, info = L.apply_clean_window(idx, {"g": {"t_event": 0.1, "t_end": 0.2}, "h": None, "k": {"t_event": 0.0, "t_end": 0.0}})
    assert new["model"]["g/a"]["surprise"] == {"m": 5.0}                           # scalar only: unrestricted
    assert new["pixel_baselines"]["g/a"]["surprise"]["m"] == pytest.approx(2.5)
    assert new["pixel_baselines"]["h/a"]["surprise"] == {"m": 9.0} and "k/a" not in new["pixel_baselines"]
    assert info["sources"]["pixel_baselines"] == {"restricted": 1, "unrestricted": 1, "dropped_empty_window": 1}
    assert info["sources"]["model"]["unrestricted"] == 1 and info["n_groups_with_window"] == 2


def test_rep_hinge_affine_convex_and_v_at_x_hat():
    rng = np.random.default_rng(0)
    G, D = 20, 32
    P = rng.normal(size=(G, 3, D))                      # per group: a(x) = P0 + P1 x + P2 f(x)
    acts = lambda f: P[:, :1] + P[:, 1:2] * X[None, :, None] + P[:, 2:3] * f[None, :, None]
    st = L.rep_kink_stat(acts(np.maximum(0.0, X - 1.0)), X, 1.0, 0.2)
    assert not st["inconclusive"] and np.min(st["K_g"]) > 10                        # a hinge at x = 1
    st = L.rep_kink_stat(acts(np.zeros_like(X)), X, 1.0, 0.2)                       # affine in ε (kinematics)
    assert np.allclose(st["K_g"], 0.0, atol=1e-6)
    st = L.rep_kink_stat(acts(X ** 2), X, 1.0, 0.2)                                 # convex, smooth
    assert np.allclose(st["K_g"], 0.0, atol=1e-6)
    st = L.rep_kink_stat(acts(np.exp(X)), X, 1.0, 0.2)                              # smoothly varying curvature
    assert np.max(np.abs(st["K_g"])) < 0.05
    v = acts(np.abs(X - 0.5))                           # a V at the model's x̂ = 0.5
    assert np.allclose(L.rep_kink_stat(v, X, 1.0, 0.2, x_hat=0.5)["K_g"], 0.0, atol=1e-6)
    v1 = acts(np.abs(X - 1.0))                          # a V exactly at x0: indistinguishable from a hinge...
    assert np.min(L.rep_kink_stat(v1, X, 1.0, 0.2)["K_g"]) > 10
    assert L.rep_kink_stat(v1, X, 1.0, 0.2, x_hat=0.95)["inconclusive"]   # ...so x̂ near x0 is inconclusive
    assert L.rep_kink_stat([a for a in acts(X)], X, 1.0, 0.2)["n_reference_points"] == 9    # |x − 1| > 0.2; list input


def test_energy_kink_rep_end_to_end(tmp_path):
    """Fake stage-4 cache: layer L1 has a hinge at ε = 1, layer L0 is affine; label kinematics affine."""
    twins = [f"eps{e:.1f}" for e in np.round(np.arange(0.5, 1.51, 0.1), 1)]
    x = L.ladder_doses(twins)
    rng = np.random.default_rng(1)
    groups = [f"g{i}" for i in range(10)]
    recs, labs = {}, {}
    for g in groups:
        P = rng.normal(size=(3, 16))
        for t, e in zip(twins, x):
            a0 = P[0] + e * P[1]
            a1 = P[0] + e * P[1] + max(0.0, e - 1.0) * P[2] * 5 + 0.01 * rng.normal(size=16)
            f = tmp_path / g / f"{t}.npz"
            f.parent.mkdir(exist_ok=True)
            np.savez(f, pooled=np.stack([np.stack([a0, a1]), np.zeros((2, 16))]), pools=np.array(["frame.post", "roi_pair.post"]),
                     layers=np.array([0, 1]), phases=np.array(["pre", "post"]))
            recs[f"{g}/{t}"] = {"group": g, "file": f"{g}/{t}.npz"}
            labs[f"{g}/{t}"] = {"physlaw": {"objects": ["A"]}, "per_frame": [
                {"time_s": 0.1 * k, "A": {"position": [e * 0.1 * k, 0.0, 0.0], "velocity": [e, 0.0, 0.0]}} for k in range(8)]}
    windows = {g: {"t_event": 0.2, "t_end": 0.5} for g in groups}
    spec = {"twins": twins, "x0": 1.0, "exclude_radius": 0.2, "rep_pixels": False}
    res = L.energy_kink_rep(spec, groups, labs.__getitem__, tmp_path, tmp_path, recs, windows, x_hat=0.6,
                            n_boot=100, n_splits=5, log=lambda *a: None)
    json.dumps(res)
    assert res["status"] == "ok" and res["model"]["cell"] == "L1" and res["window"]["unrestricted"] == len(recs)
    assert res["model"]["heldout_K_rep"] > 1 and res["model"]["K_rep_ci95"][0] > 1
    assert abs(res["baselines"]["kinematics"]["K_rep"]) < 1e-3                      # affine by construction
    assert res["vs_baselines"]["kinematics"]["diff_ci95"][0] > 0
    assert res["verdict"] == "representational kink beyond baselines"
    assert "L2-rep" in L.rep_table({"energy_kink": res})
    # restricted window: per-token-frame frame pools, the window drops the token-frame after t_end
    z = dict(np.load(tmp_path / groups[0] / f"{twins[0]}.npz"))
    z["phases"] = np.array(["post", "post"])
    np.savez(tmp_path / groups[0] / f"{twins[0]}.npz", **z, pooled_frame_t=np.stack([z["pooled"][0], 1 + z["pooled"][0]]),
             token_t_s=np.array([0.3, 0.9]))                  # the second token-frame ends after t_end = 0.5
    A, cells, cnt = L._rep_model_acts(groups[:1], twins[:1], tmp_path, recs, windows)
    assert cnt == {"restricted": 1, "unrestricted": 0} and cells == ["L0", "L1"] and A.shape == (1, 1, 2, 16)
    assert np.allclose(A[0, 0], z["pooled"][0], atol=1e-3)


FMT7 = {"fps": 48, "num_frames": 289, "resolution": [832, 480], "sensor_width_mm": 41.6, "aspect": 832 / 480,
        "event_frame_range": [112, 128]}


def test_one_primary_per_law_and_f10_contrasts():
    """Every family of the shipped L2 config has exactly one primary contrast per law (F2: the KE-reducing e0.15
    violator); a second primary, or a primary flag on a group contrast, is refused. F10's three violators are
    contrasted with legal under law rolling, in L2 and L2b, and every name exists in the data family."""
    from physlaw.config import load_config
    from physlaw.eval.l2b_detection import L2B_LAWS
    cfg = load_config(REPO / "configs" / "eval" / "l2_surprise.yaml")
    for fam in cfg["contrasts"]:
        _, cons = L.family_contrasts(cfg, fam)
        laws = {c.get("law", "momentum") for c in cons}
        assert sorted(c.get("law", "momentum") for c in cons if c.get("primary")) == sorted(laws), fam
    _, cons2 = L.family_contrasts(cfg, "f2")
    assert [L.contrast_key(c) for c in cons2 if c.get("primary")] == ["mom_violate_e0.15_neg vs e0.15"]
    two = {"contrasts": {"fx": {"list": [{"viol": "a", "legal": "l", "primary": True},
                                         {"viol": "b", "legal": "l", "primary": True}]}}}
    with pytest.raises(ValueError, match="exactly one primary"):
        L.family_contrasts(two, "fx")
    flag = {"contrasts": {"fx": {"list": [{"viol": "a", "legal": "l", "primary": True}],
                                 "group_contrasts": [{"name": "k", "kind": L.KINK_KIND, "primary": True}]}}}
    with pytest.raises(ValueError, match="group_contrasts take no primary"):
        L.family_contrasts(flag, "fx")
    cal10, cons10 = L.family_contrasts(cfg, "f10")
    assert cal10 is None and L.control_pairs(cfg, "f10") == []           # one legal twin per group: no control
    assert [L.contrast_key(c) for c in cons10] == ["hollow_first vs legal", "tie vs legal", "slip_w2 vs legal"]
    assert {c["law"] for c in cons10} == {"rolling"} and cons10[0]["primary"] and "rolling" in L2B_LAWS
    assert "f10" in cfg.families and "f10" in load_config(REPO / "configs" / "eval" / "l2b_detection.yaml").families
    from physlaw.data.families import f10_rolling as F10
    sc = next(sc for seed in range(40) for sc in [F10.sample(np.random.default_rng([seed, 0]), {}, FMT7)]
              if not isinstance(sc, str))
    assert {"legal", *(c["viol"] for c in cons10)} <= set(sc.twins)
    assert all(sc.twins[c["viol"]].violated_law == "rolling" and not sc.twins[c["viol"]].legal for c in cons10)
    assert sc.twins["legal"].legal


def test_config_kinks_and_new_families():
    from physlaw.config import load_config
    from physlaw.data.families import eps_name
    cfg = load_config(REPO / "configs" / "eval" / "l2_surprise.yaml")
    (f2,) = L.kink_specs(cfg, "f2")
    assert f2["twins"] == [f"e{e:.1f}" for e in X]
    assert L.ladder_doses(f2["twins"]).tolist() == pytest.approx(X.tolist())
    eps = [round(0.5 + 0.05 * i, 2) for i in range(21)]                 # 21 rungs eps0.5, eps0.55, ..., eps1.5
    for fam, name in (("f7", "energy_kink_bounce"), ("f8", "energy_kink"), ("f9", "energy_kink_swing")):
        (k,) = L.kink_specs(cfg, fam)
        assert k["name"] == name and k["x0"] == 1.0 and k["kind"] == L.KINK_KIND
        assert k["twins"] == [eps_name(e) for e in eps]
        assert L.ladder_doses(k["twins"]).tolist() == pytest.approx(eps)
    for fam in ("f8", "f11"):
        assert L.group_spec(cfg, fam) is None and L.family_contrasts(cfg, fam) == (None, [])
    # F7 gravity contrasts (projectile, drop) beside the bounce kink; no legal-vs-legal control (user, 2026-11-12:
    # legal and legal_e04 are identical inside the clean window, which ends at the first floor contact)
    cal7, cons7 = L.family_contrasts(cfg, "f7")
    assert L.group_spec(cfg, "f7") is None and cal7 is None and L.control_pairs(cfg, "f7") == []
    assert [L.contrast_key(c) for c in cons7] == ["gravity_x15 vs legal", "vx_x06 vs legal", "heavy_x13 vs legal"]
    assert {c["law"] for c in cons7} == {"gravity"} and cons7[0].get("primary")
    from physlaw.data.families import f7_freefall as F7
    for sub in ("projectile", "drop"):                   # every configured twin exists in the data family
        sc = next(sc for seed in range(40) for sc in [F7.sample(np.random.default_rng([seed, 0]), {}, FMT7, sub)]
                  if not isinstance(sc, str))
        assert {"legal", "legal_e04"} <= set(sc.twins)
        assert {c["viol"] for c in cons7 if c["viol"] in sc.twins} == {"projectile": {"gravity_x15", "vx_x06"},
                                                                       "drop": {"heavy_x13"}}[sub]
        assert all(sc.twins[c["viol"]].violated_law == "gravity" for c in cons7 if c["viol"] in sc.twins)
    from physlaw.eval.l2b_detection import L2B_LAWS
    assert "f7" in load_config(REPO / "configs" / "eval" / "l2b_detection.yaml").families and "gravity" in L2B_LAWS
    (k11,) = L.kink_specs(cfg, "f11")
    assert k11["kind"] == L.GROUP_KINK_KIND and k11["dose"] == "overhang_x" and set(k11["twins"]) == {"stay", "tip"}
    assert L.kink_specs(cfg, "f1") == []
    _, cons = L.family_contrasts(cfg, "f1")
    assert all(c["viol"].startswith("mom_violate") for c in cons)       # no energy contrasts (R5)
    laws = {f: {c.get("law", "momentum") for c in L.family_contrasts(cfg, f)[1]} for f in ("f6", "f9", "f12")}
    assert laws == {"f6": {"momentum", "friction"}, "f9": {"oscillation"}, "f12": {"angular_momentum"}}
    cal6, cons6 = L.family_contrasts(cfg, "f6")
    assert cal6 is None and L.contrast_key(cons6[0]) == "speed_x0.9+speed_x1.1 vs legal"      # F6: no legal-vs-legal pair
    assert {L.contrast_key(c) for c in cons6} >= {"reverse vs legal", "viscous vs legal",
                                                  "decel_x0.85+decel_x1.15 vs legal", "heavy_x13 vs legal", "turn_20 vs legal"}
    assert not any("no_friction" in L.contrast_key(c) for c in cons6)          # twin dropped 2026-11-12 (user)
    assert [L.contrast_key(c) for c in L.family_contrasts(cfg, "f12")[1] if c.get("primary")] == [
        "spin_ke_x0.8+spin_ke_x1.2 vs legal"]                                  # energy-neutral primary (audit M3)
    assert [L.contrast_key(c) for c in cons6 if c.get("primary")] == ["speed_x0.9+speed_x1.1 vs legal", "reverse vs legal"]
    assert L.family_contrasts(cfg, "f9")[0] is None and L.control_pairs(cfg, "f12") == [("legal_e04", "legal")]


# --------------------------------------------------------------------------- across-group kink (F11)

XG = np.round(0.55 + 0.1 * np.arange(10), 2)       # the F11 overhang grid, symmetric about 1 (1 not on it)


def _f11(observer, reps=8, seed=0, noise=0.02):
    """Per-group log surprise of the stay / tip twins under an observer; returns (labels getter, surprise, groups)."""
    rng = np.random.default_rng(seed)
    labs, sur, groups = {}, {}, []
    for i, x in enumerate(np.repeat(XG, reps)):
        g = f"g{i:03d}"
        groups.append(g)
        s_stay, s_tip = observer(x)
        for t, v in (("stay", s_stay), ("tip", s_tip)):
            labs[f"{g}/{t}"] = {"physlaw": {"legal": (t == "tip") == (x > 1), "constants": {"overhang_x": float(x)}}}
            sur[f"{g}/{t}"] = {"l2_obj": float(np.exp(v + noise * rng.normal()))}
    return labs.__getitem__, sur, groups


def test_group_kink_law_vs_motion():
    """The law observer (the violating twin more surprising by |x − 1|) gives a kink at x = 1 (midpoint stencil) and a
    positive contrast on both sides; a motion correlate (only the tip twin moves, motion ~ |x − 1|: contrast linear
    through 1) and a constant motion step (contrast ±m) give K = 0 and opposite-sign sides."""
    law = lambda x: (abs(x - 1) if x > 1 else 0.0, abs(x - 1) if x < 1 else 0.0)        # (stay, tip)
    motion = lambda x: (0.0, 2.0 * abs(x - 1))
    step = lambda x: (0.0, 0.5)
    spec = {"name": "statics_kink", "kind": "group_kink", "twins": ["stay", "tip"], "dose": "overhang_x",
            "x0": 1.0, "exclude_radius": 0.15}
    res = {}
    for name, ob in (("law", law), ("motion", motion), ("step", step)):
        lab, sur, groups = _f11(ob)
        res[name] = L.group_kink(spec, groups, lab, sur, "model", None, n_boot=200, log=lambda *a: None)
        json.dumps(res[name])
    r = res["law"]["distances"]["l2_obj"]
    assert res["law"]["status"] == "ok" and res["law"]["doses"] == pytest.approx(XG.tolist())
    assert r["model"]["stencil"] == "midpoint" and r["model"]["K_ci95"][0] > 0 and r["verdict"].startswith("kink")
    assert r["model"]["below_ci95"][0] > 0 and r["model"]["above_ci95"][0] > 0
    assert r["sides_verdict"].startswith("violation more surprising on both sides")
    for name in ("motion", "step"):
        m = res[name]["distances"]["l2_obj"]["model"]
        assert abs(m["K"]) < 0.1 and m["below"] > 0 > m["above"], name
        assert res[name]["distances"]["l2_obj"]["sides_verdict"].startswith("not")
    md = L.group_kink_table({"statics_kink": res["law"]})
    assert "Across-group kink" in md and "l2_obj" in md
    # exact curves: a V at the midpoint gives K > 0, a line or a step at x0 gives K = 0
    assert L.kink_stat(np.abs(XG - 1)[None], XG, 1.0, 0.15, novelty=False)["K_g"][0] > 0.5
    for c in (1 - XG, np.where(XG < 1, 1.0, -1.0) * 0.3 + 0.2 * XG):
        st = L.kink_stat(c[None], XG, 1.0, 0.15, novelty=False)
        assert abs(st["K_g"][0]) < 1e-9 and not st["inconclusive"] and st["x_hat_fit"].startswith("none")


def test_listed_violators_pairwise(tmp_path):
    """A contrast with two violators (the glide's speed-matched pair): every (violator, legal) row, group bootstrap;
    a speed-only observer (surprise = post speed) wins exactly half, a law observer every row."""
    from physlaw.eval.l2_surprise import run
    data = tmp_path / "f6"
    rows = []
    for i in range(12):
        g = f"g{i:02d}"
        for t, s in (("legal", 1.0), ("speed_x0.9", 0.9), ("speed_x1.1", 1.1)):
            (data / g / t).mkdir(parents=True)
            (data / g / t / "labels.json").write_text(json.dumps({"physlaw": {"legal": t == "legal",
                                                                               "energy_physical": t != "speed_x1.1"}}))
            law = 1.0 if t == "legal" else 2.0
            rows.append({"clip": f"{g}/{t}", "group": g, "twin": t, "surprise": {"speed_obj": s, "law_obj": law}})
    (tmp_path / "idx.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    cfg = {"contrasts": {"f6": {"list": [{"viol": ["speed_x0.9", "speed_x1.1"], "legal": "legal", "primary": True}]}},
           "group_n_boot": 50}
    res = run({"model": tmp_path / "idx.jsonl"}, data, cfg, log=lambda *a: None)
    c = res["contrasts"]["speed_x0.9+speed_x1.1 vs legal"]["model"]
    assert c["speed_obj"]["win_rate"] == 0.5 and c["speed_obj"]["n"] == 24 and c["speed_obj"]["n_groups"] == 12
    assert c["law_obj"]["win_rate"] == 1.0 and res["primary_by_law"] == {"momentum": "speed_x0.9+speed_x1.1 vs legal"}
    assert res["calibration"] is None and res["controls"] == []
    # audit M3: each half of the speed-matched pair alone (speed-only: the slower half never wins, the faster always),
    # and the faster half flagged as also creating energy
    h = res["contrasts"]
    assert h["speed_x0.9+speed_x1.1 vs legal [half speed_x0.9]"]["model"]["speed_obj"]["win_rate"] == 0.0
    assert h["speed_x0.9+speed_x1.1 vs legal [half speed_x1.1]"]["model"]["speed_obj"]["win_rate"] == 1.0
    assert h["speed_x0.9+speed_x1.1 vs legal [half speed_x1.1]"]["model"]["speed_obj"]["n"] == 12
    info = res["contrast_info"]["speed_x0.9+speed_x1.1 vs legal"]
    assert info["energy_creating_share"] == {"speed_x0.9": 0.0, "speed_x1.1": 1.0}
    assert info["also_breaks_energy"] == ["speed_x1.1"] and "also breaks energy" in L.table(res)
