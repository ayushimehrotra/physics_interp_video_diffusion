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


# --------------------------------------------------------------------------- S_mom (momentum only)

def _rand_collision(rng):
    mA, mB = rng.uniform(0.2, 3.0, 2)
    n = rng.normal(size=2); n /= np.linalg.norm(n)
    vB = rng.normal(size=2)
    vA = vB + n * rng.uniform(0.2, 3.0) + np.array([-n[1], n[0]]) * rng.normal()
    pre = {"v": np.stack([vA, vB])}
    ctx = {"m": np.array([mA, mB]), "n": n, "e_range": (0.0, 1.0)}
    return pre, ctx, cp.PreState(mA, mB, vA, vB, n)


def test_momentum_exchange_isolates_momentum():
    rng = np.random.default_rng(1)
    for _ in range(200):
        pre, ctx, st = _rand_collision(rng)
        mA, mB = ctx["m"]
        post = lambda o: {"v": pre["v"] + np.asarray(o).reshape(2, 2)}
        for e in (0.0, 0.4, 1.0, 1.5):                       # any restitution, energy-creating included
            for scale in (1.0, 0.3):                         # uniform shrinkage does not move eta
                x = L.momentum_exchange(pre, post(scale * cp.legal_outcome(st, e)), ctx)
                assert x["eta"] == pytest.approx(0.0, abs=1e-9)
        j = rng.normal(size=2)                               # tangential / any-direction impulse conserves too
        assert L.momentum_exchange(pre, post(np.r_[-j / mA, j / mB]), ctx)["eta"] == pytest.approx(0.0, abs=1e-9)
        # the gate's unit is D of the e = 0 collision
        x0 = L.momentum_exchange(pre, post(cp.legal_outcome(st, 0.0)), ctx)
        assert x0["den"] == pytest.approx(x0["d_min"], rel=1e-9)
        # only A changes: no transfer, eta = 1
        assert L.momentum_exchange(pre, post(np.r_[1.0, 0.5, 0.0, 0.0]), ctx)["eta"] == pytest.approx(1.0)
        # drift on a legal outcome: eta = |dp| / D by hand
        o = cp.legal_outcome(st, 0.5) + np.r_[0.2, -0.1, 0.2, -0.1]
        dvA, dvB = o[:2], o[2:]
        hand = np.linalg.norm(mA * dvA + mB * dvB) / (mA * np.linalg.norm(dvA) + mB * np.linalg.norm(dvB))
        assert L.momentum_exchange(pre, post(o), ctx)["eta"] == pytest.approx(hand, rel=1e-9)
        # mass-blind reference: same relative-velocity change, eta = |mA - mB| / (mA + mB)
        legal = post(cp.legal_outcome(st, 0.7))
        bl = L.mass_blind_outcome(pre, legal, ctx)
        dA, dB = bl["v"] - pre["v"]
        assert dA - dB == pytest.approx((legal["v"] - pre["v"])[0] - (legal["v"] - pre["v"])[1])
        assert L.momentum_exchange(pre, bl, ctx)["eta"] == pytest.approx(abs(mA - mB) / (mA + mB), rel=1e-9)
    # the no-event null moves nothing: eta undefined (must be gated out)
    pre, ctx, _ = _rand_collision(rng)
    x = L.momentum_exchange(pre, {"v": pre["v"].copy()}, ctx)
    assert x["den"] == 0 and np.isnan(x["eta"])


def test_l1_momentum_scores_package():
    """S_mom = 1 for a law-perfect (shrunk) forecast, ~0 for the mass-blind rule, gated null not scored."""
    from physlaw.eval import l1_forecast_laws as l1
    rng = np.random.default_rng(2)
    per, rows = {}, []
    kinds = {"perfect": lambda st, o, pre, post, ctx: o * 0.6,
             "blind": lambda st, o, pre, post, ctx: (L.mass_blind_outcome(pre, post, ctx)["v"] - pre["v"]).reshape(4),
             "null": lambda st, o, pre, post, ctx: np.zeros(4)}
    res = {}
    for kind, fc in kinds.items():
        per, rows = {}, []
        for g in range(40):
            pre, ctx, st = _rand_collision(rng)
            if max(ctx["m"]) / min(ctx["m"]) < 1.3:
                ctx["m"] = ctx["m"] * np.array([1.0, 2.0])
                st = cp.PreState(*ctx["m"], *pre["v"], ctx["n"])
            o = cp.legal_outcome(st, 0.5)
            post = {"v": pre["v"] + o.reshape(2, 2)}
            v_fc = pre["v"] + fc(st, o, pre, post, ctx).reshape(2, 2)
            v_fl = post["v"] + rng.normal(scale=1e-3, size=(2, 2))
            c = f"g{g}/conserve_e1"
            per[c] = {"mom": l1._mom_clip({**pre, "p": np.zeros((2, 2))}, post, ctx, v_fc, v_fl)}
            rows.append([c])
        res[kind] = l1._momentum_scores(per, rows, 200, rng, l1.MOM_DEFAULTS)
    assert res["perfect"]["interaction_rate"] == 1.0 and res["perfect"]["S"] == pytest.approx(1.0, abs=0.02)
    assert res["blind"]["S"] == pytest.approx(0.0, abs=0.02)
    assert res["null"]["interaction_rate"] == 0.0 and res["null"]["n_scored_clips"] == 0 and np.isnan(res["null"]["S"])
    common = l1.momentum_on_common_groups([res["perfect"], res["blind"]])
    assert common[0]["n_common_groups"] == 40 and common[0]["S"] > common[1]["S"]
