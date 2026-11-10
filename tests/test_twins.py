"""Spec §2.2 twin identities, checked on the twin algebra directly (no rendering).

Random frictionless pre-contact states; the honest outcome is the exact legal
outcome at e = 1, which is what the simulator delivers to ~1e-8.
"""
from __future__ import annotations

import numpy as np
import pytest

from physlaw.data import twins as TW
from physlaw.data.splits import split_for_group

CP = TW.CP


def random_pre(rng) -> "TW.PreState":
    while True:
        m_a, m_b = np.exp(rng.uniform(np.log(0.1), np.log(3.0), 2))
        n = rng.normal(size=2); n /= np.linalg.norm(n)
        v_a, v_b = rng.normal(size=2), rng.normal(size=2) * 0.5
        pre = TW.PreState(m_a=m_a, m_b=m_b, v_a=v_a, v_b=v_b, n=n)
        if pre.approach_speed > 0.05:
            return pre


def honest(pre, e=1.0):
    dv = CP.legal_outcome(pre, e)
    return pre.v_a + dv[:2], pre.v_b + dv[2:]


@pytest.fixture(params=range(200))
def case(request):
    rng = np.random.default_rng(request.param)
    pre = random_pre(rng)
    pa, pb = honest(pre)
    sign = TW.ke_reducing_sign(pre)
    delta = TW.choose_split_delta(rng, pre.m_a / (pre.m_a + pre.m_b), (0.10, 0.35), 0.05, sign=sign)
    if delta is None:
        pytest.skip("no admissible split delta for this mass ratio")
    return pre, TW.f1_twins(pre, pa, pb, 1.0, delta), delta


def test_sum_and_ratio_identities(case):
    pre, tw, _ = case
    u = pre.approach_speed
    for name, e in (("conserve_e1", 1.0), ("conserve_e015", 0.15), ("energy_create", 1.5)):
        a, b = TW.deflections(pre, tw[name].v_a, tw[name].v_b)
        assert a + b == pytest.approx((1 + e) * u, rel=1e-10), name          # |dvA|+|dvB| = (1+e) u_n
        assert b / a == pytest.approx(pre.m_a / pre.m_b, rel=1e-10), name    # |dvB|/|dvA| = mA/mB


def test_momentum_violation_keeps_sum_changes_ratio(case):
    pre, tw, delta = case
    for tag in ("e1", "e015"):
        a0, b0 = TW.deflections(pre, tw[f"conserve_{tag}"].v_a, tw[f"conserve_{tag}"].v_b)
        a1, b1 = TW.deflections(pre, tw[f"mom_violate_{tag}"].v_a, tw[f"mom_violate_{tag}"].v_b)
        assert a1 + b1 == pytest.approx(a0 + b0, rel=1e-10)                  # restitution unchanged
        assert b1 / a1 != pytest.approx(b0 / a0, rel=1e-3)                   # split moved
        assert a1 > 0 and b1 > 0                                              # nobody deflected backwards
        assert TW.realised_restitution(pre, tw[f"mom_violate_{tag}"].v_a, tw[f"mom_violate_{tag}"].v_b) == \
            pytest.approx(TW.realised_restitution(pre, tw[f"conserve_{tag}"].v_a, tw[f"conserve_{tag}"].v_b), abs=1e-10)
        assert TW.momentum_residual(pre, tw[f"mom_violate_{tag}"].v_a, tw[f"mom_violate_{tag}"].v_b) > 1e-3


def test_conserving_twins_conserve_momentum(case):
    pre, tw, _ = case
    for name in ("conserve_e1", "conserve_e015", "energy_create"):
        assert TW.momentum_residual(pre, tw[name].v_a, tw[name].v_b) < 1e-10, name


def test_energy_bookkeeping(case):
    pre, tw, _ = case
    ke0 = TW.ke_pre(pre)
    assert TW.kinetic_energy(pre, tw["conserve_e1"].v_a, tw["conserve_e1"].v_b) == pytest.approx(ke0, rel=1e-10)
    assert TW.kinetic_energy(pre, tw["conserve_e015"].v_a, tw["conserve_e015"].v_b) < ke0
    assert TW.kinetic_energy(pre, tw["energy_create"].v_a, tw["energy_create"].v_b) > ke0      # KE_post/KE_pre > 1
    lam = tw["energy_create"].record["lambda"]
    assert lam == pytest.approx(2.5 / 2.0)
    # flags agree with the bookkeeping
    for t in tw.values():
        assert t.energy_physical == (TW.kinetic_energy(pre, t.v_a, t.v_b) <= ke0 + 1e-12)


def test_ke_reducing_sign_lowers_ke_relative_to_legal_base():
    rng = np.random.default_rng(0)
    for _ in range(500):
        pre = random_pre(rng)
        pa, pb = honest(pre, 0.15)
        s = TW.ke_reducing_sign(pre)
        small = 1e-3 * s
        va, vb, _ = CP.split_edit(pre, pa, pb, small)
        assert TW.kinetic_energy(pre, va, vb) <= TW.kinetic_energy(pre, pa, pb) + 1e-12


def test_split_delta_respects_margin():
    rng = np.random.default_rng(1)
    for s in np.linspace(0.02, 0.98, 49):
        for sign in (None, 1.0, -1.0):
            d = TW.choose_split_delta(rng, s, (0.10, 0.35), 0.05, sign=sign)
            if d is not None:
                assert 0.05 - 1e-12 <= s - d <= 0.95 + 1e-12
                assert abs(d) >= 0.10 - 1e-12


def test_f2_ladder_is_legal_and_ordered():
    rng = np.random.default_rng(2)
    pre = random_pre(rng)
    pa, pb = honest(pre)
    tw = TW.f2_twins(pre, pa, pb, 1.0)
    assert len(tw) == 11
    kes = [TW.kinetic_energy(pre, t.v_a, t.v_b) for t in tw.values()]
    assert all(np.diff(kes) > 0)
    for t in tw.values():
        assert TW.momentum_residual(pre, t.v_a, t.v_b) < 1e-10
        assert TW.realised_restitution(pre, t.v_a, t.v_b) == pytest.approx(t.e_target, abs=1e-10)


def test_split_is_by_group_and_balanced():
    ids = [f"f1_{100000 + i:06d}" for i in range(4000)]
    s = [split_for_group(g) for g in ids]
    assert split_for_group(ids[0]) == s[0]                     # deterministic
    frac = {k: s.count(k) / len(s) for k in ("train", "val", "test")}
    assert abs(frac["train"] - 0.70) < 0.03 and abs(frac["val"] - 0.15) < 0.03
