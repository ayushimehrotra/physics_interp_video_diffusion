"""physlaw.eval.laws (spec §5.0): every statistic invariant to a length scale, a time scale and a common gain on the
measured vectors (documented exceptions for the gain: slip, L_ratio, alpha_ratio);
legal outcomes give legal values, nulls give null values; the energy identity
KE - KE' = 1/2 mu u_n^2 (1 - eps^2); eps boost-invariant; S_mom ingredients; io.law_inputs on synthetic labels of
every family; agreement with the rendered data/f1 labels (skipped if absent)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from physlaw.data import collision_physics as cp
from physlaw.data.io import law_inputs
from physlaw.eval import laws as L

F1 = Path(__file__).resolve().parents[1] / "data" / "f1"
G = np.array([0.0, 0.0, -9.81])


def _rand_collision(rng, e=0.5):
    mA, mB = rng.uniform(0.2, 3.0, 2)
    n = rng.normal(size=2); n /= np.linalg.norm(n)
    vB = rng.normal(size=2)
    vA = vB + n * rng.uniform(0.2, 3.0) + np.array([-n[1], n[0]]) * rng.normal()
    st = cp.PreState(mA, mB, vA, vB, n)
    pre = {"v": np.stack([vA, vB]), "p": rng.normal(size=(2, 2))}
    post = {"v": pre["v"] + cp.legal_outcome(st, e).reshape(2, 2), "p": pre["p"].copy()}
    return pre, post, {"event": "collision", "m": np.array([mA, mB]), "n": n, "e_range": (0.0, 1.0)}, st


# --------------------------------------------------------------------------- one random case per law

def _cases(rng):
    """(law, pre, post, ctx) with generic (non-legal) values, so invariance is tested away from fixed points."""
    out = []
    pre, post, ctx, _ = _rand_collision(rng, e=0.7)
    post = {**post, "v": post["v"] + rng.normal(scale=0.2, size=(2, 2))}
    out.append((L.momentum, pre, post, ctx))
    out.append((L.energy, pre, post, ctx))
    # bounce and pad energy events
    v = np.array([[0.7, 0.2, -3.0]])
    out.append((L.energy, {"v": v}, {"v": np.array([[0.6, 0.2, 2.1]])}, {"event": "bounce", "g": G}))
    out.append((L.energy, {"v": v}, {"v": 0.8 * v + 0.1}, {"event": "pad"}))
    # galilean: boosted, perturbed clip vs its unboosted partner
    pre0, post0, c0, _ = _rand_collision(rng, e=0.4)
    u = rng.normal(size=2)
    out.append((L.galilean, {"v": pre0["v"] + u}, {"v": post0["v"] + u + rng.normal(scale=0.1, size=(2, 2))},
                {**c0, "ref": (pre0, post0)}))
    # gravity: two objects, accelerations along and across g, heavy one off
    a0 = np.array([[0.1, 0.0, -9.81], [0.0, 0.2, -9.81]])
    out.append((L.gravity, {"v": np.array([[1.0, 0.5, 2.0], [0.3, -0.2, 1.0]]), "a": a0},
                {"v": np.array([[0.9, 0.6, 1.0], [0.3, -0.1, 0.5]]), "a": a0 * np.array([[1, 1, 1.3], [1, 1, 1.1]])},
                {"g": G, "m": np.array([1.0, 3.0])}))
    # friction: two slides, decelerations before / after the branch, a creep over the stop window
    out.append((L.friction, {"v": np.array([[1.0, 0.5, 0.0], [0.8, 0.4, 0.0]]),
                             "a": np.array([[-1.1, -0.55, 0.0], [-1.0, -0.5, 0.0]])},
                {"v": np.array([[0.9, 0.45, 0.0], [0.7, 0.35, 0.0]]),
                 "a": np.array([[-1.0, -0.5, 0.0], [-0.96, -0.48, 0.0]]),
                 "v_stop": np.array([[0.05, 0.02, 0.0], [-0.03, -0.01, 0.0]])},
                {"m": np.array([2.0, 1.0])}))
    # momentum / Newton I on a force-free glide (one body)
    out.append((L.momentum, {"v": np.array([[1.2, -0.4, 0.0]])}, {"v": np.array([[0.9, 0.1, 0.0]])},
                {"force_free": True}))
    # oscillation: two bobs, periods before / after
    out.append((L.oscillation, {"T": np.array([2.0, 2.0])}, {"T": np.array([2.1, 1.7])}, {"m": np.array([1.0, 3.0])}))
    # statics: rotation over the window and the initial angular acceleration of a tipping block, g from the clip
    out.append((L.statics, {"v": np.zeros((1, 3))},
                {"v": np.zeros((1, 3)), "alpha": np.array([[0.0, 1.7, 0.0]]), "phi": np.array([0.3])},
                {"overhang_x": 1.25, "aspect_ba": 0.5, "g_clip": 9.6, "half_length": 0.7}))
    # statics without "phi": the rotation from the spins and dt
    out.append((L.statics, {"v": np.zeros((1, 3)), "w": np.zeros((1, 3))},
                {"v": np.zeros((1, 3)), "w": np.array([[0.0, 0.3, 0.0]])},
                {"overhang_x": 0.85, "aspect_ba": 0.5, "dt": 0.5}))
    # angular momentum: puck + bar, generic post state
    pre12 = {"p": np.array([[0.0, 0.3, 0.0], [0.5, 0.0, 0.0]]), "v": np.array([[2.0, 0.1, 0.0], [0.0, 0.0, 0.0]]),
             "w": np.zeros((2, 3))}
    post12 = {"p": pre12["p"].copy(), "v": np.array([[0.4, 0.1, 0.0], [0.9, 0.05, 0.0]]),
              "w": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.3]])}
    out.append((L.angular_momentum, pre12, post12, {"m": np.array([1.0, 1.5]), "k": np.array([0.5, 0.36]),
                                                    "r": np.array([0.25, 1.2])}))
    # energy: pendulum damper (swing) from the apex states around it (heights only)
    out.append((L.energy, {"p": np.array([[0.4, 0.0, 1.2]]), "v": np.zeros((1, 3))},
                {"p": np.array([[-0.35, 0.0, 1.13]]), "v": np.zeros((1, 3))},
                {"event": "swing", "g": G, "p_bottom": np.array([0.0, 0.0, 1.0])}))
    # rolling: generic accelerations and spins
    k = np.array([0.5, 1.0])
    out.append((L.rolling, {"v": np.array([[0.2, 0.0], [0.1, 0.0]])},
                {"v": np.array([[1.4, 0.1], [0.9, 0.05]]), "w": np.array([[0, 0, 4.1], [0, 0, 2.2]])},
                {"k": k, "r": np.array([0.3, 0.35]), "dt": 0.4}))
    return out


# slip, L_ratio: w vs v; statics alpha_ratio: an angular acceleration vs a measured g_clip / half-length
GAIN_EXEMPT = {"slip_s", "slip_h", "L_ratio", "alpha_ratio"}


def _scale(st: dict, ctx: dict, s=1.0, tau=1.0, gain=1.0):
    """Length scale s, time scale tau (t -> tau t), and a common gain on measured velocities / accelerations."""
    f = {"p": s, "v": gain * s / tau, "v_stop": gain * s / tau, "a": gain * s / tau ** 2, "w": 1.0 / tau, "T": tau, "alpha": 1.0 / tau ** 2,
         "phi": 1.0}
    return {k: (np.asarray(x) * f[k] if k in f else x) for k, x in st.items()}


def _scale_ctx(ctx, s, tau, gain):
    c = dict(ctx)
    if "dt" in c:
        c["dt"] = c["dt"] * tau
    if "r" in c:
        c["r"] = np.asarray(c["r"]) * s
    if "g" in c:
        c["g"] = np.asarray(c["g"]) * s / tau ** 2
    if "g_clip" in c:                         # measured in the clip: a fall's acceleration
        c["g_clip"] = c["g_clip"] * gain * s / tau ** 2
    for key in ("p_bottom", "c", "half_length"):
        if key in c:
            c[key] = np.asarray(c[key]) * s
    if "ref" in c:
        c["ref"] = tuple(_scale(x, c, s, tau, gain) for x in c["ref"])
    return c


@pytest.mark.parametrize("s,tau,gain", [(37.0, 1.0, 1.0), (1.0, 0.13, 1.0), (1.0, 1.0, 0.42), (5.0, 2.5, 0.6)])
def test_every_statistic_is_scale_time_gain_invariant(s, tau, gain):
    rng = np.random.default_rng(0)
    for law, pre, post, ctx in _cases(rng):
        ref = law.stats(pre, post, ctx)
        got = law.stats(_scale(pre, ctx, s, tau, gain), _scale(post, ctx, s, tau, gain), _scale_ctx(ctx, s, tau, gain))
        assert set(ref) == set(law.legal) | ({"rho_star", "creation"} if law is L.energy else set())
        for k, v in ref.items():
            if gain != 1.0 and k in GAIN_EXEMPT:
                continue                      # a measured quantity compared with an unmeasured / unscaled one
            if not np.isfinite(v):            # undefined on this state (e.g. g_ratio without pre accelerations)
                assert np.isnan(got[k]), (law.NAME, k)
                continue
            assert got[k] == pytest.approx(v, rel=1e-9, abs=1e-12), (law.NAME, k, s, tau, gain)


# --------------------------------------------------------------------------- legal and null values

def _is_legal(law, stats):
    """Every statistic the state defines is at its legal value (NaN: not defined on this state, e.g. the inertia
    statistics on a collision); at least one is defined."""
    n = 0
    for k, (op, v) in law.legal.items():
        x = stats[k]
        if np.isnan(x):
            continue
        n += 1
        assert (x == pytest.approx(v, abs=1e-9)) if op == "eq" else (x <= v + 1e-9), (law.NAME, k, x)
    assert n, (law.NAME, "no statistic defined")


def test_legal_outcomes_give_legal_values_and_nulls_null():
    rng = np.random.default_rng(1)
    for _ in range(50):
        for e in (0.0, 0.5, 1.0):
            pre, post, ctx, _ = _rand_collision(rng, e)
            _is_legal(L.momentum, L.momentum.stats(pre, post, ctx))
            _is_legal(L.energy, L.energy.stats(pre, post, ctx))
            assert L.epsilon(pre, post, ctx) == pytest.approx(e, abs=1e-9)
            assert L.momentum.family_distance(post, pre, ctx)[0] == pytest.approx(0.0, abs=1e-9)
        null = L.momentum.null_outcome(pre, ctx)
        assert L.epsilon(pre, null, ctx) == pytest.approx(-1.0)
        assert L.momentum.family_distance(null, pre, ctx)[0] == pytest.approx(1.0)
    hot = L.energy.stats(pre, {"v": pre["v"] + 1.3 / 1.0 * (post["v"] - pre["v"])}, ctx)   # e = 1.0 -> eps 1.6
    assert hot["epsilon"] == pytest.approx(1.6) and hot["creation"] == pytest.approx(0.6)
    assert hot["rho_star"] == pytest.approx(1.6 ** 2)
    # galilean: a boost of a legal clip
    pre, post, ctx, _ = _rand_collision(rng, 0.3)
    u = np.array([0.7, -0.3])
    _is_legal(L.galilean, L.galilean.stats({"v": pre["v"] + u}, {"v": post["v"] + u}, {**ctx, "ref": (pre, post)}))
    e, th = L.collision.invariants(pre, post, ctx)
    s = L.galilean.stats({"v": pre["v"] + u}, {"v": post["v"] * 1.2 + u}, {**ctx, "e_ref": e, "theta_ref": th})
    assert s["d_epsilon"] > 0.05
    # gravity: same g after the branch, horizontal kept, equal accelerations
    pre = {"v": np.array([[2.0, 0.0, 1.0], [1.0, 1.0, 0.0]]), "a": np.stack([G, G])}
    post = {"v": pre["v"] + G * 0.3, "a": np.stack([G, G])}
    ctx = {"g": G, "m": np.array([1.0, 2.0]), "dt": 0.3}
    _is_legal(L.gravity, L.gravity.stats(pre, post, ctx))
    ns = L.gravity.stats(pre, L.gravity.null_outcome(pre, ctx), ctx)
    assert ns["g_ratio"] == 0.0 and ns["axis_ratio"] == 0.0 and np.isnan(ns["mass_ratio"])  # Aristotle: v_h' = 0
    st = L.gravity.stats({"v": pre["v"]}, {"v": pre["v"]}, {"g": G})
    assert np.isnan(st["g_ratio"]) and np.isnan(st["mass_ratio"])           # no accelerations: undefined
    # rolling: legal accelerations and spins; null = inertia ignored, no rotation
    g, th, dt, r = 9.81, 0.3, 0.5, np.array([0.3, 0.4])
    ctx = {"g_mag": g, "theta": th, "k": np.array([0.5, 1.0]), "dt": dt, "r": r}
    sp = g * np.sin(th) / (1 + ctx["k"]) * dt
    pre = {"v": np.zeros((2, 3)), "w": np.zeros((2, 3))}
    post = {"v": np.c_[sp, 0 * sp, 0 * sp], "w": np.c_[0 * sp, sp / r, 0 * sp]}
    _is_legal(L.rolling, L.rolling.stats(pre, post, ctx))
    ns = L.rolling.stats(pre, L.rolling.null_outcome(pre, ctx), ctx)
    assert ns == {"order": pytest.approx(0.0, abs=1e-12), "slip_s": 0.0, "slip_h": 0.0}
    swap = L.rolling.stats(pre, {"v": post["v"][::-1], "w": post["w"]}, ctx)
    assert swap["order"] == pytest.approx(-1.0)
    # rolling at the release instant (dt = 0): order from the post accelerations
    acc = {"v": np.zeros((2, 3)), "a": np.c_[sp / dt, 0 * sp, 0 * sp]}
    assert L.rolling.stats(pre, acc, {**ctx, "dt": 0.0})["order"] == pytest.approx(1.0)


def test_new_laws_legal_values_and_nulls():
    """Newton I, friction, oscillation, statics, angular momentum and the swing epsilon: legal -> legal values,
    nulls -> their null values, violations -> off the legal value in the right direction."""
    # Newton I (momentum on a force-free glide)
    v = np.array([[1.5, -0.5, 0.0]])
    ff = {"force_free": True}
    _is_legal(L.momentum, L.momentum.stats({"v": v}, {"v": v.copy()}, ff))
    st = L.momentum.stats({"v": v}, {"v": 1.3 * v}, ff)
    assert st["inertia_speed"] == pytest.approx(1.3) and st["inertia_turn"] == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(st["eta"])                                              # collision statistics undefined here
    c, s_ = math.cos(math.radians(20)), math.sin(math.radians(20))
    turned = np.array([[c * v[0, 0] - s_ * v[0, 1], s_ * v[0, 0] + c * v[0, 1], 0.0]])
    assert L.momentum.stats({"v": v}, {"v": turned}, ff)["inertia_turn"] == pytest.approx(20 / 180)
    assert L.momentum.stats({"v": v}, L.momentum.null_outcome({"v": v}, ff), ff)["inertia_speed"] == 0.0
    pre, post, ctx, _ = _rand_collision(np.random.default_rng(5), 0.5)       # collisions: inertia stats NaN
    assert np.isnan(L.momentum.stats(pre, post, ctx)["inertia_speed"])
    # friction: Coulomb (constant deceleration, then rest) legal; viscous null < 1 with the lighter body slowed more
    u = np.array([[0.6, 0.8, 0.0], [0.6, 0.8, 0.0]])
    m = np.array([1.0, 4.0])
    early = {"v": 2.0 * u, "a": -1.5 * u}
    late = {"v": 1.4 * u, "a": -1.5 * u, "v_stop": 0.0 * u}
    st = L.friction.stats(early, late, {"m": m})
    _is_legal(L.friction, st)
    assert st["stop_hold"] == 0.0
    # reverse (the deceleration continued through v = 0: back-slide), no friction (glides on at the branch speed)
    assert L.friction.stats(early, {**late, "v_stop": -0.6 * u}, {"m": m})["stop_hold"] == pytest.approx(0.3)
    assert L.friction.stats(early, {**late, "a": 0 * u, "v_stop": 2.0 * u}, {})["decel_const"] == 0.0
    assert L.friction.stats(early, {**late, "a": 0 * u, "v_stop": 2.0 * u}, {})["stop_hold"] == pytest.approx(1.0)
    one = {"v": 2.0 * u[:1], "a": -1.5 * u[:1]}                             # Stokes null, one body: e^-1 left at the stop
    assert L.friction.stats(one, L.friction.null_outcome(one, {"dt": 0.3}), {})["stop_hold"] == pytest.approx(math.exp(-1))
    nul = L.friction.null_outcome(early, {"m": m, "dt": 0.5})
    ns = L.friction.stats(early, nul, {"m": m})
    assert ns["decel_const"] < 1.0 and ns["mass_ratio"] < 1.0                # F = -c v: the lighter body slows more
    n0 = L.friction.null_outcome(early, {"m": m, "dt": 0.0})
    assert L.friction.stats(early, n0, {"m": m})["mass_ratio"] == pytest.approx(m[0] / m[1])
    visc = {"v": 1.4 * u, "a": -1.5 * 0.7 * u}                              # a ~ v
    assert L.friction.stats(early, visc, {"m": m})["decel_const"] == pytest.approx(0.7)
    # oscillation
    T = np.array([1.8, 1.8])
    _is_legal(L.oscillation, L.oscillation.stats({"T": T}, {"T": T.copy()}, {"m": m}))
    ns = L.oscillation.stats({"T": T}, L.oscillation.null_outcome({"T": T}, {"m": m}), {"m": m})
    assert ns["mass_period_ratio"] == pytest.approx(math.sqrt(m[0] / m[1]))
    assert L.oscillation.stats({"T": T}, {"T": T * [1.0, 0.8]}, {"m": m})["mass_period_ratio"] == pytest.approx(0.8)
    # statics: legal tip at x > 1, legal stay at x < 1, violations, the stay null; g measured in the clip (in
    # pixels / frame^2 here: only the ratio to the half-length in the same units enters)
    g_px, a_px = 9.81 * 125.0 / 48.0 ** 2, 0.8 * 125.0
    for x in (0.55, 0.95, 1.05, 1.45):
        ctx = {"overhang_x": x, "aspect_ba": 0.5, "g_clip": g_px, "half_length": a_px}
        al = g_px * float(L.tip_rate(x, 0.5)) / a_px
        assert al * abs(x - 1) * a_px == pytest.approx(g_px * float(L.tip_factor(x, 0.5)))    # |a_cm| = alpha d
        rest = {"v": np.zeros((1, 3)), "alpha": np.zeros((1, 3)), "phi": [0.0]}
        tip = {"v": np.zeros((1, 3)), "alpha": np.array([[0.0, al, 0.0]]), "phi": [0.4]}
        legal, viol = (tip, rest) if x > 1 else (rest, tip)
        s_l, s_v = L.statics.stats(rest, legal, ctx), L.statics.stats(rest, viol, ctx)
        assert s_l["tip_consistency"] == 1.0 and s_v["tip_consistency"] == 0.0
        if x > 1:
            assert s_l["alpha_ratio"] == pytest.approx(1.0) and s_v["alpha_ratio"] == 0.0
            assert L.statics.stats(rest, L.statics.null_outcome(rest, ctx), ctx)["tip_consistency"] == 0.0
            assert np.isnan(L.statics.stats(rest, legal, {**ctx, "g_clip": float("nan")})["alpha_ratio"])
        else:
            assert np.isnan(s_l["alpha_ratio"])
    # angular momentum: the rigid impulse conserves L and p; spin x k scales the bar's share; the null drops it
    ctx = {"m": np.array([0.8, 1.6]), "I": np.array([0.02, 0.9]), "n": np.array([1.0, 0.0]),
           "c": np.array([0.0, 0.4]), "e": 0.7}
    pre = {"p": np.array([[-0.25, 0.4, 0.0], [0.15, 0.0, 0.0]]), "v": np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
           "w": np.zeros((2, 3))}
    for e in (0.0, 0.5, 1.0):
        post = L.rigid_impulse(pre, ctx, e)
        _is_legal(L.angular_momentum, L.angular_momentum.stats(pre, post, ctx))
        assert np.allclose(ctx["m"] @ post["v"], ctx["m"] @ pre["v"])
        assert post["w"][0, 2] == 0.0 and post["w"][1, 2] != 0.0
    spun = {**post, "w": post["w"] * 1.2}
    L0 = L.angular_momentum_z(pre, ctx)
    want = 1.0 + 0.2 * ctx["I"][1] * post["w"][1, 2] / L0
    assert L.angular_momentum.stats(pre, spun, ctx)["L_ratio"] == pytest.approx(want)
    nul = L.angular_momentum.null_outcome(pre, ctx)
    assert L.angular_momentum.stats(pre, nul, ctx)["L_ratio"] == pytest.approx(
        1.0 - ctx["I"][1] * L.rigid_impulse(pre, ctx, 0.7)["w"][1, 2] / L0)
    # swing epsilon: apex heights only (no g), any length unit
    pb = np.array([0.0, 0.0, 1.0])
    sw = {"event": "swing", "p_bottom": pb}
    assert L.epsilon({"p": np.array([[0.3, 0.0, 1.2]]), "v": np.zeros((1, 3))},
                     {"p": np.array([[-0.3, 0.0, 1.05]]), "v": np.zeros((1, 3))}, sw) == pytest.approx(math.sqrt(0.25))
    assert L.epsilon({"p": np.array([[37.0, 0.0, 44.4]]), "v": np.zeros((1, 3))},
                     {"p": np.array([[-37.0, 0.0, 44.4]]), "v": np.zeros((1, 3))},
                     {**sw, "p_bottom": 37.0 * pb, "g": G}) == pytest.approx(1.0)


def test_epsilon_events_and_dispatch():
    v = np.array([[1.0, 0.0, -4.0]])
    for e in (0.4, 0.8, 1.2):
        post = {"v": np.array([[1.0, 0.0, 4.0 * e]])}
        assert L.epsilon({"v": v}, post, {"event": "bounce", "g": G}) == pytest.approx(e)
        assert L.epsilon({"v": v}, {"v": e * v}, {"event": "pad"}) == pytest.approx(e)
    pre, post, ctx, _ = _rand_collision(np.random.default_rng(2), 0.6)
    assert L.epsilon(pre, post, {"n": ctx["n"]}) == pytest.approx(0.6)                       # inferred, mass-free
    p3 = {"v": np.c_[pre["v"], np.zeros(2)]}
    q3 = {"v": np.c_[post["v"], np.zeros(2)]}
    assert L.epsilon(p3, q3, {"event": "collision", "n": np.r_[ctx["n"], 0.0]}) == pytest.approx(0.6)
    assert L.rho_star(pre, post, ctx) == pytest.approx(0.36)
    with pytest.raises(ValueError):
        L.epsilon(pre, post, {"g": G})
    with pytest.raises(ValueError):
        L.epsilon(pre, post, {"event": "spring"})


@pytest.mark.parametrize("e", [0.0, 0.3, 0.8, 1.0, 1.2, 1.5])
def test_energy_identity_and_boost_invariance(e):
    rng = np.random.default_rng(3)
    pre, post, ctx, st = _rand_collision(rng, e)
    ke = lambda v: 0.5 * float((ctx["m"] * (v ** 2).sum(-1)).sum())
    eps = L.epsilon(pre, post, ctx)
    assert ke(pre["v"]) - ke(post["v"]) == pytest.approx(0.5 * st.mu * st.approach_speed ** 2 * (1 - eps ** 2),
                                                         rel=1e-9, abs=1e-12)
    u = rng.normal(size=2) * 3
    assert L.epsilon({"v": pre["v"] + u}, {"v": post["v"] + u}, {"n": ctx["n"]}) == pytest.approx(eps, abs=1e-12)


def test_ratio_score():
    assert L.ratio_score(1.0, 1.0, 0.5) == 1.0 and L.ratio_score(0.5, 1.0, 0.5) == 0.0
    assert L.ratio_score(2.0, 1.0, 0.5) == pytest.approx(0.0)                # log-symmetric
    out = L.ratio_score(np.array([1.0, np.nan, -1.0, 1.0]), 1.0, np.array([0.5, 0.5, 0.5, 1.0]))
    assert out[0] == 1.0 and np.isnan(out[1:]).all()


def test_registry_api():
    assert set(L.LAWS) == {"momentum", "energy", "gravity", "friction", "oscillation", "rolling", "statics", "galilean",
                           "angular_momentum"}
    for name, law in L.LAWS.items():
        assert law.NAME == name and callable(law.stats) and callable(law.null_outcome)
        assert all(op in ("eq", "le") for op, _ in law.legal.values())
    assert L.LAWS["momentum"].family_distance is not None and L.energy.family_distance is None


# --------------------------------------------------------------------------- S_mom ingredients

def test_momentum_exchange_isolates_momentum():
    rng = np.random.default_rng(1)
    for _ in range(200):
        pre, _, ctx, st = _rand_collision(rng)
        mA, mB = ctx["m"]
        post = lambda o: {"v": pre["v"] + np.asarray(o).reshape(2, 2)}
        for e in (0.0, 0.4, 1.0, 1.5):
            for scale in (1.0, 0.3):
                assert L.momentum_exchange(pre, post(scale * cp.legal_outcome(st, e)), ctx)["eta"] == \
                    pytest.approx(0.0, abs=1e-9)
        j = rng.normal(size=2)
        assert L.momentum_exchange(pre, post(np.r_[-j / mA, j / mB]), ctx)["eta"] == pytest.approx(0.0, abs=1e-9)
        x0 = L.momentum_exchange(pre, post(cp.legal_outcome(st, 0.0)), ctx)
        assert x0["den"] == pytest.approx(x0["d_min"], rel=1e-9)
        assert L.momentum_exchange(pre, post(np.r_[1.0, 0.5, 0.0, 0.0]), ctx)["eta"] == pytest.approx(1.0)
        o = cp.legal_outcome(st, 0.5) + np.r_[0.2, -0.1, 0.2, -0.1]
        hand = np.linalg.norm(mA * o[:2] + mB * o[2:]) / (mA * np.linalg.norm(o[:2]) + mB * np.linalg.norm(o[2:]))
        x = L.momentum_exchange(pre, post(o), ctx)
        assert x["eta"] == pytest.approx(hand, rel=1e-9)
        assert L.momentum.stats(pre, post(o), ctx)["eta"] == pytest.approx(hand, rel=1e-9)
        legal = post(cp.legal_outcome(st, 0.7))
        bl = L.mass_blind_outcome(pre, legal, ctx)
        assert L.momentum_exchange(pre, bl, ctx)["eta"] == pytest.approx(abs(mA - mB) / (mA + mB), rel=1e-9)
        mf = L.mass_free_momentum(pre, legal, ctx)
        assert mf["antiparallel"] == pytest.approx(1.0) and mf["log_ratio_err"] == pytest.approx(0.0, abs=1e-9)
    x = L.momentum_exchange(pre, {"v": pre["v"].copy()}, ctx)
    assert x["den"] == 0 and np.isnan(x["eta"])
    assert np.isnan(L.momentum.stats(pre, post(o), {"n": ctx["n"]})["eta"])          # eta needs masses


# --------------------------------------------------------------------------- io.law_inputs on synthetic labels

FPS = 48


def _labels(family, subfamily, objs, traj, te, constants, event_kind=None, extra=None, T=96):
    """labels-2 dict from analytic trajectories. traj[name] = (side_pre, side_post), each (p(te), v(te), a, w):
    p(t) = p + v (t - te) + a (t - te)^2 / 2 on that side of te."""
    names = list(objs)

    def st(n, t):
        p, v, a, w = traj[n][0 if t < te else 1]
        d = t - te
        return np.asarray(p) + np.asarray(v) * d + 0.5 * np.asarray(a) * d * d, np.asarray(v) + np.asarray(a) * d, w

    per_frame = []
    for f in range(T):
        e = {"frame": f, "time_s": f / FPS}
        for n in names:
            p, v, w = st(n, f / FPS)
            e[n] = {"position": p.tolist(), "velocity": v.tolist(), "angular_velocity": list(w)}
        per_frame.append(e)
    side = lambda i: {"t_s": te, "p": [list(traj[n][i][0]) for n in names], "v": [list(traj[n][i][1]) for n in names],
                      "w": [list(traj[n][i][3]) for n in names]}
    pl = {"family": family, "subfamily": subfamily, "twin": "t", "legal": True, "violated_law": None,
          "violation_magnitude": 1.0, "event_time_s": te, "first_post_contact_frame": int(math.floor(te * FPS)) + 1,
          "constants": {"g": 9.81, "fps": FPS, "masses": {n: o["mass"] for n, o in objs.items()},
                        "sizes": {n: o["radius"] for n, o in objs.items()}, **constants},
          "event_state": {"pre": side(0), "post": side(1)}, "objects": objs, **(extra or {})}
    if event_kind:
        pl["event_kind"] = event_kind
    return {"per_frame": per_frame, "scene_constants": {"frame_rate": FPS}, "physlaw": pl}


TE = 50.4 / FPS
Z3 = [0.0, 0.0, 0.0]


def _tipping_labels(x, tau, beta=0.5, a_half=0.8, T=240):
    """F11 labels of a block released at TE on a ledge with overhang x, in slow motion by tau (g -> g / tau^2: the
    same motion, tau times slower): spin about y growing at the legal alpha (tip twin) or 0 (stay), the prop (aux)
    falling at 3 g from TE (the clip's own g reference)."""
    g = 9.81 / tau ** 2
    al = g * float(L.tip_rate(x, beta)) / a_half if x > 1 else 0.0
    per_frame = []
    for f in range(T):
        d = max(f / FPS - TE, 0.0)
        per_frame.append({"frame": f, "time_s": f / FPS,
                          "A": {"position": [0.0, 0.0, 1.0], "velocity": Z3, "angular_velocity": [0.0, al * d, 0.0]},
                          "P": {"position": [0.5, 0.0, 0.9 - 1.5 * g * d * d], "velocity": [0.0, 0.0, -3 * g * d],
                                "angular_velocity": Z3}})
    st = lambda: {"t_s": TE, "objects": ["A", "P"], "p": [[0.0, 0.0, 1.0], [0.5, 0.0, 0.9]], "v": [Z3, Z3],
                  "w": [Z3, Z3]}
    objs = {"A": {"mass": 2.0, "radius": a_half}, "P": {"mass": 1.0, "radius": 0.05, "aux": True}}
    return {"per_frame": per_frame, "scene_constants": {"frame_rate": FPS},
            "physlaw": {"family": "f11", "subfamily": "support", "twin": "tip", "legal": x > 1, "event_kind": "release",
                        "event_time_s": TE, "first_post_contact_frame": int(math.floor(TE * FPS)) + 1, "objects": objs,
                        "constants": {"g": g, "fps": FPS, "masses": {"A": 2.0, "P": 1.0},
                                      "sizes": {"A": a_half, "P": 0.05}, "overhang_x": x, "aspect_ba": beta,
                                      "half_length": a_half, "prop_acc_over_g": 3.0},
                        "event_state": {"pre": st(), "post": st()}}}


@pytest.mark.parametrize("x", [1.05, 1.25, 0.95])
def test_statics_tip_window_is_time_scale_invariant(x):
    """io.law_inputs measures phi over the tip window 2 sqrt(a / g_clip) (clip time units), so phi and tip_consistency
    are unchanged by slow motion (tau = 3: g_clip / 9); a fixed 0.5 s window would see a near-threshold tip (x = 1.05)
    turn by 0.02 rad < TIP_ANGLE at tau = 3 and score it as staying."""
    res = {}
    for tau in (1.0, 3.0):
        pre, post, ctx = law_inputs(_tipping_labels(x, tau))
        assert ctx["g_clip"] == pytest.approx(9.81 / tau ** 2, rel=1e-6)
        res[tau] = (float(post["phi"][0]), L.statics.stats(pre, post, ctx)["tip_consistency"])
    assert res[3.0][0] == pytest.approx(res[1.0][0], rel=0.1, abs=1e-12)
    assert res[1.0][1] == res[3.0][1] == 1.0
    if x == 1.05:                                     # the legal start turns 2 tip_rate(x) rad over the window
        assert res[3.0][0] == pytest.approx(2 * float(L.tip_rate(x, 0.5)), rel=0.1)


def test_law_inputs_camera_boost_frames():
    """F4 camera boost (labels frame == camera): io.law_inputs hands the laws the camera-frame states (what the clip
    shows: the ball-boosted motion), so the Galilean law against the u = 0 partner is a real test (0 for a legal clip);
    frame="world" gives the world states (= the u0 twin's, the trivially exact case). io.world_view swaps per-frame
    states too."""
    from physlaw.data.io import world_view
    rng = np.random.default_rng(7)
    _, _, _, st = _rand_collision(rng, 0.5)
    n = np.r_[st.n, 0.0]
    cA, cB = np.zeros(3), 0.75 * n
    vA, vB = np.r_[st.v_a, 0.0], np.r_[st.v_b, 0.0]
    o = cp.legal_outcome(st, 0.5).reshape(2, 2)
    objs = {"A": {"mass": st.m_a, "radius": 0.3}, "B": {"mass": st.m_b, "radius": 0.45}}
    u = np.array([0.4, -0.3, 0.0])

    def lab(du):
        traj = {"A": ((cA, vA + du, Z3, Z3), (cA, vA + du + np.r_[o[0], 0], Z3, Z3)),
                "B": ((cB, vB + du, Z3, Z3), (cB, vB + du + np.r_[o[1], 0], Z3, Z3))}
        return _labels("f4", "", objs, traj, TE, {"contact_normal": list(n), "e": 0.5}, event_kind="collision")

    world, cam = lab(np.zeros(3)), lab(u)
    pc, pw = cam["physlaw"], world["physlaw"]
    pc.update(frame="camera", boost_kind="camera", ref_event_state=pw["event_state"], event_state_world=pw["event_state"])
    for fc, fw in zip(cam["per_frame"], world["per_frame"]):
        for k in "AB":
            fc[k]["position_world"], fc[k]["velocity_world"] = fw[k]["position"], fw[k]["velocity"]
    pre, post, ctx = law_inputs(cam)
    assert ctx["frame"] == "camera" and ctx["boost_kind"] == "camera"
    assert np.allclose(pre["v"], np.stack([vA, vB]) + u)
    assert L.galilean.stats(pre, post, ctx)["d_epsilon"] == pytest.approx(0.0, abs=1e-9)
    prew, postw, ctxw = law_inputs(cam, frame="world")
    pre0, post0, _ = law_inputs(world)
    assert ctxw["frame"] == "world" and np.allclose(prew["v"], pre0["v"]) and np.allclose(postw["v"], post0["v"])
    wv = world_view(cam)
    assert wv["per_frame"][3]["A"]["position"] == world["per_frame"][3]["A"]["position"]
    assert world_view(world) is world and law_inputs(world)[2]["frame"] == "world"


def test_law_inputs_collision_normal_from_centres():
    rng = np.random.default_rng(4)
    _, _, _, st = _rand_collision(rng, 0.6)
    n = np.r_[st.n, 0.0]
    rA, rB = 0.3, 0.45
    cA = np.zeros(3)
    cB = cA + (rA + rB) * n
    vA, vB = np.r_[st.v_a, 0.0], np.r_[st.v_b, 0.0]
    o = cp.legal_outcome(st, 0.6).reshape(2, 2)
    objs = {"A": {"mass": st.m_a, "radius": rA}, "B": {"mass": st.m_b, "radius": rB}}
    traj = {"A": ((cA, vA, Z3, Z3), (cA, vA + np.r_[o[0], 0], Z3, Z3)),
            "B": ((cB, vB, Z3, Z3), (cB, vB + np.r_[o[1], 0], Z3, Z3))}
    off = n + np.r_[0.0, 0.0, 0.0] + 0.02 * np.r_[-n[1], n[0], 0.0]          # a labelled normal ~1 deg off
    lab = _labels("f2", "", objs, traj, TE, {"contact_normal": list(off / np.linalg.norm(off)), "e": 0.6},
                  event_kind="contact")
    pre, post, ctx = law_inputs(lab)
    assert ctx["event"] == "collision" and np.allclose(ctx["n"], n, atol=1e-9)
    assert np.allclose(pre["p"], np.stack([cA, cB]), atol=1e-9) and np.allclose(post["p"], pre["p"])
    assert L.epsilon(pre, post, ctx) == pytest.approx(0.6, abs=1e-9)
    _is_legal(L.momentum, L.momentum.stats(pre, post, ctx))
    # without per-frame / state positions the labelled normal is the fallback
    lab2 = json.loads(json.dumps(lab))
    for s in ("pre", "post"):
        lab2["physlaw"]["event_state"][s]["p"] = [[0.0, 0.0, 0.0]] * 2
    assert np.allclose(law_inputs(lab2)[2]["n"], off / np.linalg.norm(off))
    # from_labels keeps the 2-D collision form on the same normal
    lab["physlaw"].update(pre_velocity={"A": list(vA), "B": list(vB)},
                          post_velocity={"A": list(vA + np.r_[o[0], 0]), "B": list(vB + np.r_[o[1], 0])})
    p2, q2, c2 = L.from_labels(lab)
    assert c2["event"] == "collision" and np.allclose(c2["n"], st.n, atol=1e-9)
    assert L.epsilon(p2, q2, c2) == pytest.approx(0.6, abs=1e-9)


def test_law_inputs_gravity_bounce_pad_rolling():
    g3 = np.array([0.0, 0.0, -9.81])
    ball = {"A": {"mass": 1.0, "radius": 0.1}}
    # f7 projectile: gravity x1.3 after the branch
    p0, v0 = np.array([0.0, 0.0, 2.0]), np.array([1.5, 0.5, 1.0])
    lab = _labels("f7", "projectile", ball, {"A": ((p0, v0, g3, Z3), (p0, v0, 1.3 * g3, Z3))}, TE, {},
                  event_kind="branch")
    pre, post, ctx = law_inputs(lab)
    assert "event" not in ctx or ctx["event"] == "branch"
    assert np.allclose(pre["a"], g3[None], atol=1e-6) and np.allclose(post["a"], 1.3 * g3[None], atol=1e-6)
    s = L.gravity.stats(pre, post, ctx)
    assert s["g_ratio"] == pytest.approx(1.3, rel=1e-6) and s["axis_ratio"] == pytest.approx(1.0)
    _, post_dt, ctx_dt = law_inputs(lab, dt=0.2)
    assert ctx_dt["dt"] == pytest.approx(0.2, abs=0.6 / FPS)
    assert np.allclose(post_dt["a"], 1.3 * g3[None], atol=1e-6)
    # f6 pair: two blocks decelerating at 2 before the branch; after it A keeps 2, B (heavy) 2.4 and both slide on
    # through v = 0 (the analytic sides do not stop): moving window up to B's stop - 0.1 s, stop window from the
    # legal stop (te + 1 s) + 0.1 s
    u = np.array([0.6, 0.8, 0.0])
    blocks = {"A": {"mass": 1.0, "radius": 0.1}, "B": {"mass": 3.0, "radius": 0.1}}
    traj = {"A": ((Z3, 2 * u, -2.0 * u, Z3), (Z3, 2 * u, -2.0 * u, Z3)),
            "B": ((u, 2 * u, -2.0 * u, Z3), (u, 2 * u, -2.4 * u, Z3))}
    lab = _labels("f6", "pair", blocks, traj, TE, {"t_stop_s": TE + 1.0, "t_stop_twin_s": TE + 2.0 / 2.4},
                  event_kind="branch", T=120)
    pre, post, ctx = law_inputs(lab)
    assert ctx["move_end"] == pytest.approx(TE + 2.0 / 2.4 - 0.1) and ctx["stop_start"] == pytest.approx(TE + 1.1)
    st = L.friction.stats(pre, post, ctx)
    assert st["mass_ratio"] == pytest.approx(1.2, rel=1e-6) and st["decel_const"] == pytest.approx(1.1, rel=1e-6)
    t_mid = 0.5 * (TE + 1.1 + 119 / FPS) - TE                               # B back-slides: |v| / |v_pre| at mid-window
    assert st["stop_hold"] == pytest.approx(max(2.0 * t_mid - 2.0, 2.4 * t_mid - 2.0) / 2.0, rel=0.02)
    assert np.allclose(post["p"], pre["p"]) and post["t_s"] == pytest.approx(TE)
    # f7 bounce
    v0, v1 = np.array([0.7, 0.0, -4.0]), np.array([0.7, 0.0, 3.2])
    lab = _labels("f7", "bounce", ball, {"A": ((p0, v0, g3, Z3), (p0, v1, g3, Z3))}, TE, {"e_floor": 0.8},
                  event_kind="contact")
    pre, post, ctx = law_inputs(lab)
    assert ctx["event"] == "bounce" and L.energy.stats(pre, post, ctx)["epsilon"] == pytest.approx(0.8)
    # f8 pad (labels may or may not carry event_kind / the dose fields)
    block = {"A": {"mass": 2.0, "radius": 0.1}}
    v = np.array([2.0, 0.0, 0.0])
    traj = {"A": ((Z3, v, Z3, Z3), (Z3, 1.2 * v, Z3, Z3))}
    for kind, extra in ((None, {}), ("pad", {"epsilon": 1.2, "clean_until_s": 1.6, "stop_dose": 0.0})):
        pre, post, ctx = law_inputs(_labels("f8", "ramp", block, traj, TE, {}, event_kind=kind, extra=extra))
        assert ctx["event"] == "pad" and L.energy.stats(pre, post, ctx)["creation"] == pytest.approx(0.2)
        assert "a" not in pre and all(ctx.get(k) == x for k, x in extra.items())
    # f10 rolling from rest: order and slip
    th, k, r = 0.25, np.array([0.5, 1.0]), np.array([0.2, 0.25])
    a = 9.81 * np.sin(th) / (1 + k)
    down = np.array([math.cos(th), 0.0, -math.sin(th)])
    objs = {n: {"mass": 1.0, "radius": float(r[i]), "inertia": float(k[i] * r[i] ** 2)} for i, n in enumerate("AB")}
    spin = lambda i: list(np.array([0.0, a[i] / r[i], 0.0]))                  # w grows with v; only |w| / |v| matters
    traj = {n: ((np.r_[0.0, i, 1.0], Z3, Z3, Z3), (np.r_[0.0, i, 1.0], Z3, a[i] * down, spin(i)))
            for i, n in enumerate("AB")}
    lab = _labels("f10", "rolling", objs, traj, TE, {"theta_rad": th}, event_kind="release")
    # per-frame w must be |v| / r: rewrite it from the velocities
    for fr in lab["per_frame"]:
        for i, n in enumerate("AB"):
            fr[n]["angular_velocity"] = [0.0, float(np.linalg.norm(fr[n]["velocity"]) / r[i]), 0.0]
    pre, post, ctx = law_inputs(lab, dt=0.25)
    assert np.allclose(ctx["k"], k) and np.allclose(pre["a"], 0.0, atol=1e-9)
    st = L.rolling.stats(pre, post, ctx)
    assert st["order"] == pytest.approx(1.0, rel=1e-6)
    assert st["slip_s"] == pytest.approx(1.0) and st["slip_h"] == pytest.approx(1.0)
    pre, post, ctx = law_inputs(lab)                                          # event instant: from post["a"]
    assert L.rolling.stats(pre, post, ctx)["order"] == pytest.approx(1.0, rel=1e-6)


def test_law_inputs_new_families():
    """f6 glide (force_free), f9 pair periods and swing epsilon, f11 exact event_state accelerations, f12 contact point
    and inertias; aux bodies (rods, props) are dropped."""
    # f6 glide: speed x 0.8 at the branch
    puck = {"A": {"mass": 1.0, "radius": 0.3, "inertia": 0.045}}
    v = np.array([1.2, 0.5, 0.0])
    pre, post, ctx = law_inputs(_labels("f6", "glide", puck, {"A": ((Z3, v, Z3, Z3), (Z3, 0.8 * v, Z3, Z3))}, TE, {},
                                        event_kind="branch"))
    assert ctx["force_free"] and L.momentum.stats(pre, post, ctx)["inertia_speed"] == pytest.approx(0.8)
    # f9 pair: periods from the per-frame bob positions (rods are aux)
    from physlaw.data.families import f9_pendulum as P9
    Lp, r = 1.0, 0.12
    w0 = P9.omega0(Lp, r)
    te9 = 125 / FPS
    objs = {"A": {"mass": 1.0, "radius": r}, "B": {"mass": 3.0, "radius": r},
            "RA": {"mass": 1e-6, "radius": 0.4, "aux": True}}
    T = 289
    t = np.arange(T) / FPS
    piv = {"A": [-1.0, 0.0, 6.0], "B": [1.0, 0.0, 6.0], "RA": [-1.0, 0.0, 6.0]}
    per_frame = []
    th = {n: P9.free_swing(t, te9, 0.5, w0) for n in "AB"}
    th["B"] = tuple(np.where(t <= te9, a, b) for a, b in zip(th["B"], P9.free_swing(t, te9, 0.5, w0 / 0.8)))
    th["RA"] = th["A"]
    for f in range(T):
        e = {"frame": f, "time_s": t[f]}
        for n in objs:
            a, ad = th[n][0][f], th[n][1][f]
            e[n] = {"position": [piv[n][0] + Lp * math.sin(a), 0.0, piv[n][2] - Lp * math.cos(a)],
                    "velocity": [Lp * ad * math.cos(a), 0.0, Lp * ad * math.sin(a)], "angular_velocity": [0.0, -ad, 0.0]}
        per_frame.append(e)
    st = lambda f: {"t_s": te9, "objects": list(objs), "p": [per_frame[f][n]["position"] for n in objs],
                    "v": [per_frame[f][n]["velocity"] for n in objs], "w": [[0.0, 0.0, 0.0]] * 3}
    lab = {"per_frame": per_frame, "scene_constants": {"frame_rate": FPS},
           "physlaw": {"family": "f9", "subfamily": "pair", "twin": "heavy_x0.8", "event_kind": "branch",
                       "event_time_s": te9, "first_post_contact_frame": 126, "objects": objs,
                       "constants": {"g": 9.81, "fps": FPS, "masses": {n: o["mass"] for n, o in objs.items()},
                                     "sizes": {n: o["radius"] for n, o in objs.items()}, "pivots": piv},
                       "event_state": {"pre": st(125), "post": st(125)}}}
    pre, post, ctx = law_inputs(lab)
    assert pre["v"].shape == (2, 3) and ctx["m"].tolist() == [1.0, 3.0]                  # rod dropped
    Tl = P9.period(0.5, w0)
    assert np.allclose(pre["T"], Tl, rtol=1e-6) and np.allclose(post["T"], [Tl, 0.8 * Tl], rtol=1e-6)
    s = L.oscillation.stats(pre, post, ctx)
    assert s["mass_period_ratio"] == pytest.approx(0.8, rel=1e-6)
    # f9 swing: epsilon from the apex states around the damper (labels edit.apex_states; event_kind fallback on the
    # sub-family); the damper entry / exit event_state is not what the law reads
    h0, zb = Lp * (1 - math.cos(0.5)), 6.0 - Lp
    apex = lambda x, h, t_s: {"t_s": t_s, "objects": ["A"], "p": [[x, 0.0, zb + h]], "v": [Z3], "w": [Z3]}
    side = lambda x: ((np.array([x, 0.0, zb + 0.01]), np.array([-1.0, 0.0, 0.0]), Z3, Z3))
    lab = _labels("f9", "swing", {"A": {"mass": 1.0, "radius": r}}, {"A": (side(0.1), side(-0.1))}, TE,
                  {"p_bottom": [0.0, 0.0, zb]},
                  extra={"edit": {"apex_states": {"pre": apex(0.48, h0, 0.2), "post": apex(-0.5, 1.44 * h0, 1.6)}}})
    pre, post, ctx = law_inputs(lab)
    assert ctx["event"] == "swing" and pre["t_s"] == 0.2 and post["t_s"] == 1.6
    assert L.epsilon(pre, post, ctx) == pytest.approx(1.2, rel=1e-9)
    # f11: exact accelerations from the event_state, prop (aux) dropped from the states but its 3 g descent
    # measures g in the clip; the rotation angle over the tip window from the per-frame spins
    x, beta, a_half = 1.25, 0.5, 0.8
    a0 = -9.81 * float(L.tip_factor(x, beta))
    al = 9.81 * float(L.tip_rate(x, beta)) / a_half
    objs = {"A": {"mass": 2.0, "radius": a_half}, "P": {"mass": 1.0, "radius": 0.05, "aux": True}}
    traj = {"A": ((Z3, Z3, Z3, Z3), (Z3, Z3, [0.0, 0.0, a0], [0.0, 0.6, 0.0])),
            "P": ((Z3, Z3, Z3, Z3), (Z3, Z3, [0.0, 0.0, -3 * 9.81], Z3))}
    lab = _labels("f11", "support", objs, traj, TE, {"overhang_x": x, "aspect_ba": beta, "half_length": a_half,
                                                     "prop_acc_over_g": 3.0}, event_kind="release")
    for sd, a, alpha in (("pre", [Z3] * 2, [Z3] * 2), ("post", [[0.0, 0.0, a0], [0.0, 0.0, -29.43]],
                                                          [[0.0, al, 0.0], Z3])):
        lab["physlaw"]["event_state"][sd].update(objects=["A", "P"], a=a, alpha=alpha)
    pre, post, ctx = law_inputs(lab)
    assert post["a"].shape == (1, 3) and post["a"][0, 2] == a0 and ctx["g_clip"] == pytest.approx(9.81, rel=1e-9)
    assert post["phi"][0] == pytest.approx(0.6 * L.tip_window(9.81, a_half), rel=0.05)    # 2 sqrt(a / g_clip)
    s = L.statics.stats(pre, post, ctx)
    assert s == {"tip_consistency": 1.0, "alpha_ratio": pytest.approx(1.0, rel=1e-9)}
    del lab["physlaw"]["constants"]["prop_acc_over_g"]                       # no in-clip fall: only tip_consistency
    s = L.statics.stats(*law_inputs(lab))
    assert s["tip_consistency"] == 1.0 and np.isnan(s["alpha_ratio"])
    # f12: contact point and inertias from the labels
    objs = {"A": {"mass": 0.8, "radius": 0.25, "inertia": 0.025}, "B": {"mass": 1.6, "radius": 1.2, "inertia": 0.8}}
    ctx12 = {"m": np.array([0.8, 1.6]), "I": np.array([0.025, 0.8]), "n": np.array([1.0, 0.0]), "c": np.array([0.0, 0.5])}
    p12 = {"p": np.array([[-0.25, 0.5, 0.0], [0.15, 0.0, 0.0]]), "v": np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
           "w": np.zeros((2, 3))}
    q12 = L.rigid_impulse(p12, ctx12, 0.9)
    traj = {n: ((p12["p"][i], p12["v"][i], Z3, Z3), (p12["p"][i], q12["v"][i], Z3, list(q12["w"][i])))
            for i, n in enumerate("AB")}
    lab = _labels("f12", "spin", objs, traj, TE, {"contact_point": [0.0, 0.5, 0.0], "contact_normal": [1.0, 0.0, 0.0],
                                                  "e": 0.9}, event_kind="contact")
    pre, post, ctx = law_inputs(lab)
    assert ctx["event"] == "contact" and np.allclose(ctx["c"], [0.0, 0.5, 0.0])
    assert np.allclose(L.collision.inertias(ctx), [0.025, 0.8])
    assert L.angular_momentum.stats(pre, post, ctx)["L_ratio"] == pytest.approx(1.0, rel=1e-9)
    nul = L.angular_momentum.null_outcome(pre, ctx)
    assert L.angular_momentum.stats(pre, nul, ctx)["L_ratio"] < 1.0


# --------------------------------------------------------------------------- rendered F1 (labels-1)

@pytest.fixture(scope="module")
def f1():
    clips = sorted(F1.glob("f1_*/*/labels.json")) if F1.exists() else []
    if not clips:
        pytest.skip("data/f1 not rendered here")
    return [json.loads(p.read_text()) for p in clips]


def test_f1_agrees_with_labels_and_collision_physics(f1):
    for lab in f1:
        pl = lab["physlaw"]
        twin = pl["twin"]
        pre, post, ctx = L.from_labels(lab)
        st = cp.PreState(*ctx["m"], *pre["v"], ctx["n"])
        o = (post["v"] - pre["v"]).reshape(4)
        d, e_hat = L.momentum.family_distance(post, pre, ctx)
        ref = cp.distance_to_legal_line(o, st, segment=(0.0, 1.0)) / np.linalg.norm(cp.legal_outcome(st, 0.0))
        assert d == pytest.approx(ref, rel=1e-9, abs=1e-12)
        ms, es = L.momentum.stats(pre, post, ctx), L.energy.stats(pre, post, ctx)
        if twin.startswith("conserve"):
            # the e = 1 outcome is Bullet's: its impulse is up to ~0.4 deg off the contact normal
            assert ms["eta"] < 1e-6 and ms["antiparallel"] > 1 - 1e-6 and min(ms["normal_A"], ms["normal_B"]) > 1 - 1e-4
            assert es["epsilon"] == pytest.approx(pl["e_target"], abs=1e-3) and d < 0.02
        if twin.startswith("mom_violate"):
            assert ms["eta"] > 1e-3 and d > 1e-3
        if twin == "energy_create":
            assert es["creation"] > 1e-3 and ms["eta"] < 1e-3
