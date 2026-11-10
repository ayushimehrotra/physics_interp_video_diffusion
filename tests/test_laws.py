"""physlaw.eval.laws: closed-form legal / violating outcomes per law, and agreement of the F1 laws
with physlaw.data.collision_physics and the twin labels of the rendered data/f1 (skipped if absent)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from physlaw.data import collision_physics as cp
from physlaw.eval import laws as L

F1 = Path(__file__).resolve().parents[1] / "data" / "f1"
G = np.array([0.0, 0.0, -9.81])


def _f1_clips():
    if not F1.exists():
        return []
    return sorted(F1.glob("f1_*/*/labels.json"))


@pytest.fixture(scope="module")
def f1():
    clips = _f1_clips()
    if not clips:
        pytest.skip("data/f1 not rendered here")
    return [json.loads(p.read_text()) for p in clips]


def test_f1_agrees_with_labels_and_collision_physics(f1):
    for lab in f1:
        pl = lab["physlaw"]
        twin, s = pl["twin"], pl["summary"]
        pre, post, ctx = L.from_labels(lab)
        st = cp.PreState(*ctx["m"], *pre["v"], ctx["n"])
        o = (post["v"] - pre["v"]).reshape(4)
        assert L.momentum.residual(pre, post, ctx) == pytest.approx(s["momentum_residual"], abs=1e-9)
        assert L.energy.residual(pre, post, ctx) == pytest.approx(max(0.0, s["ke_ratio"] - 1), abs=1e-9)
        d, e_hat = L.momentum.family_distance(post, pre, ctx)
        ref = cp.distance_to_legal_line(o, st, segment=(0.0, 1.0)) / np.linalg.norm(cp.legal_outcome(st, 0.0))
        assert d == pytest.approx(ref, rel=1e-9, abs=1e-12)
        null = L.momentum.null_outcome(pre, ctx)
        assert L.momentum.family_distance(null, pre, ctx)[0] == pytest.approx(1.0)
        if twin.startswith("conserve"):
            # the e = 1 outcome is Bullet's: its impulse is up to ~0.4 deg off the analytic normal
            # (tangential fraction <= 0.007), so legal twins sit at d <= 0.015, not 0
            assert d < 0.02
            assert e_hat == pytest.approx(cp.implied_restitution(o, st), abs=1e-3)
            assert e_hat == pytest.approx(pl["e_target"], abs=1e-3)
            for law in (L.com, L.angular_momentum, L.energy):
                assert law.residual(pre, post, ctx) < 0.02, (law.NAME, twin)   # Bullet: KE up to +1.2%
        else:
            assert d > 1e-3, twin
        if twin.startswith("mom_violate"):
            assert L.com.residual(pre, post, ctx) > 1e-3
            assert L.angular_momentum.residual(pre, post, ctx) > 1e-4
            assert L.energy.residual(pre, post, ctx) > 0 or pl["energy_physical"]
        if twin == "energy_create":
            assert L.energy.residual(pre, post, ctx) > 1e-3 and e_hat == 1.0
            assert L.momentum.residual(pre, post, ctx) < 1e-3


def test_angular_momentum_point_dependence():
    """A drift along n is invisible about the contact point, visible about the origin."""
    m, n = np.array([1.0, 2.0]), np.array([1.0, 0.0])
    c = np.array([3.0, 1.0])
    p = np.stack([c - 0.5 * n, c + 0.5 * n])
    pre = {"v": np.array([[1.0, 0.2], [0.0, 0.0]]), "p": p}
    post = {"v": pre["v"] + 0.1 * n, "p": p}
    assert L.angular_momentum.residual(pre, post, {"m": m, "point": c}) < 1e-12
    assert L.angular_momentum.residual(pre, post, {"m": m, "point": np.zeros(2)}) > 1e-3


def test_galilean_boost():
    m, n = np.array([1.0, 3.0]), np.array([0.6, 0.8])
    st = cp.PreState(1.0, 3.0, np.array([1.0, 0.5]), np.array([-0.2, 0.1]), n)
    o = cp.legal_outcome(st, 0.5)
    pre = {"v": np.stack([st.v_a, st.v_b])}
    post = {"v": pre["v"] + o.reshape(2, 2)}
    u = np.array([0.7, -0.3])
    bpre, bpost = {"v": pre["v"] + u}, {"v": post["v"] + u}
    ctx = {"m": m, "n": n, "ref": (pre, post)}
    assert L.galilean.residual(bpre, bpost, ctx) < 1e-12
    assert L.collision.invariants(pre, post, ctx)[0] == pytest.approx(0.5)
    bad = {"v": bpost["v"] * 1.2}                 # not a boost of the base outcome
    assert L.galilean.residual(bpre, bad, ctx) > 0.05


def test_gravity_axis_mass():
    dt = 0.5
    pre = {"v": np.array([[2.0, 0.0, 1.0]]), "p": np.array([[0.0, 0.0, 1.0]])}
    ctx = {"g": G, "dt": dt}
    legal = L.gravity.null_outcome(pre, ctx)
    legal = {"v": pre["v"] + G * dt, "p": pre["p"] + pre["v"] * dt + 0.5 * G * dt ** 2}
    assert L.gravity.family_distance(legal, pre, ctx) == (pytest.approx(0.0, abs=1e-12), None)
    assert L.gravity.residual(pre, L.gravity.null_outcome(pre, ctx), ctx) == pytest.approx(1.0)
    vel = {"v": pre["v"] + 1.5 * G * dt}          # gravity x1.5 (velocities only)
    assert L.gravity.residual(pre, vel, ctx) == pytest.approx(0.5)
    coupled = {"v": legal["v"] * np.array([0.6, 1, 1])}      # horizontal x0.6 at the branch
    assert L.axis_independence.residual(pre, legal, ctx) == pytest.approx(0.0, abs=1e-12)
    assert L.axis_independence.residual(pre, coupled, ctx) == pytest.approx(0.4)
    pair = {"v": np.zeros((2, 3))}
    assert L.mass_independence.residual(pair, {"v": np.stack([G, G]) * dt}, ctx) == pytest.approx(0.0)
    assert L.mass_independence.residual(pair, {"v": np.stack([G, 1.3 * G]) * dt}, ctx) == pytest.approx(0.3)


def test_bounce():
    pre = {"v": np.array([[1.0, 0.0, -4.0]])}
    ctx = {"g": G}
    for e in (0.8, 0.4):
        post = {"v": np.array([[1.0, 0.0, 4.0 * e]])}
        d, e_hat = L.bounce.family_distance(post, pre, ctx)
        assert d == pytest.approx(0.0, abs=1e-12) and e_hat == pytest.approx(e)
        assert L.freefall.height_ratio(pre, post, ctx) == pytest.approx(e ** 2)
    hot = {"v": np.array([[1.0, 0.0, 4.0 * np.sqrt(1.3)]])}      # rebound height ratio 1.3
    d, e_hat = L.bounce.family_distance(hot, pre, ctx)
    assert e_hat == 1.0 and d == pytest.approx((np.sqrt(1.3) - 1) * 4 / 4)
    assert L.bounce.residual(pre, L.bounce.null_outcome(pre, ctx), ctx) == pytest.approx(1.0)


def test_ramp_and_rolling():
    g, h = 9.81, 0.8
    pre = {"v": np.zeros((1, 2))}
    s = np.sqrt(2 * g * h)
    ctx = {"g_mag": g, "h": h}
    assert L.ramp_energy.residual(pre, {"v": np.array([[s * 0.6, -s * 0.8]])}, ctx) == pytest.approx(0, abs=1e-12)
    assert L.ramp_energy.residual(pre, {"v": np.array([[s * 1.25, 0]])}, ctx) == pytest.approx(0.25)
    assert L.ramp_energy.residual(pre, L.ramp_energy.null_outcome(pre, ctx), ctx) == pytest.approx(1.0)
    th, dt = 0.3, 1.0
    ctx = {"g_mag": g, "theta": th, "k": np.array([0.5, 1.0]), "dt": dt}
    a = g * np.sin(th) / (1 + ctx["k"])
    pre = {"v": np.zeros((2, 2))}
    legal = {"v": np.stack([[a[0] * dt, 0], [a[1] * dt, 0]])}
    assert L.rolling.residual(pre, legal, ctx) == pytest.approx(0.0)
    assert L.rolling.family_distance(legal, pre, ctx)[0] == pytest.approx(0.0)
    assert L.rolling.residual(pre, {"v": np.stack([[a[0], 0], [a[0], 0]])}, ctx) == pytest.approx(1.0)
    assert L.rolling.residual(pre, {"v": np.stack([[a[1], 0], [a[0], 0]])}, ctx) == pytest.approx(2.0)


def test_pendulum():
    Lp, g = 1.0, 9.81
    T = 2 * np.pi * np.sqrt(Lp / g)
    pre = {"p": np.array([0.1]), "w": np.array([0.0])}
    ctx = {"L": Lp, "g_mag": g, "dt": T / 4}
    legal = {"p": np.array([0.0]), "w": np.array([-0.1 * 2 * np.pi / T])}
    assert L.pendulum.residual(pre, legal, ctx) == pytest.approx(0.0, abs=1e-12)
    assert L.pendulum.residual(pre, L.pendulum.null_outcome(pre, ctx), ctx) == pytest.approx(1.0)
    w = 2 * np.pi / (0.8 * T)                          # heavier bob with a 0.8x period
    fast = {"p": 0.1 * np.cos(w * T / 4), "w": -0.1 * w * np.sin(w * T / 4)}
    assert L.pendulum.residual(pre, fast, ctx) > 0.2


def test_registry_api():
    for name, law in L.LAWS.items():
        assert law.NAME == name and callable(law.residual) and callable(law.null_outcome)
