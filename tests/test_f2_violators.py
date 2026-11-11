"""F2 twin groups: momentum violators (both drift signs) and energy-creating arms beside the legal ladder.

Physics only: twins.py on sampled pre-states; one optional end-to-end check runs generate.py --no-render in
the Kubric venv (skipped when that venv is absent).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from physlaw.data import twins as TW

CP = TW.CP
REPO = Path(__file__).resolve().parents[1]
KUBRIC_PY = Path("/root/envs/kubric_venv/bin/python")
E_VIOL = (1.0, 0.5, 0.15)


def random_pre(rng):
    while True:
        m_a, m_b = np.exp(rng.uniform(np.log(0.1), np.log(3.0), 2))
        n = rng.normal(size=2); n /= np.linalg.norm(n)
        pre = TW.PreState(m_a=m_a, m_b=m_b, v_a=rng.normal(size=2), v_b=rng.normal(size=2) * 0.5, n=n)
        if pre.approach_speed > 0.05:
            return pre


def honest(pre, e=1.0):
    dv = CP.legal_outcome(pre, e)
    return pre.v_a + dv[:2], pre.v_b + dv[2:]


def eta_ref(pre, va, vb):
    """η the way laws.momentum_exchange defines it."""
    q = np.array([pre.m_a, pre.m_b])[:, None] * (np.stack([va, vb]) - np.stack([pre.v_a, pre.v_b]))
    return np.linalg.norm(q.sum(0)) / np.linalg.norm(q, axis=1).sum()


@pytest.fixture(params=range(150))
def group(request):
    rng = np.random.default_rng(request.param)
    pre = random_pre(rng)
    pa, pb = honest(pre)
    mag = float(rng.uniform(0.10, 0.35))
    vt, skipped = TW.f2_violators(pre, pa, pb, 1.0, E_VIOL, mag)
    return pre, pa, pb, mag, vt, skipped


def test_violators_break_momentum_by_the_intended_edit(group):
    pre, pa, pb, mag, vt, skipped = group
    assert set(vt) | set(skipped) == {TW.violator_name(e, s) for e in E_VIOL for s in ("pos", "neg")}
    M = pre.m_a + pre.m_b
    for name, t in vt.items():
        e = t.e_target
        la, lb = honest(pre, e)
        r = t.record
        # the drift is exactly -delta_speed * n on both balls: momentum created = -M delta n
        np.testing.assert_allclose(t.v_a - la, -r["delta_speed"] * pre.n, atol=1e-12)
        np.testing.assert_allclose(t.v_b - lb, -r["delta_speed"] * pre.n, atol=1e-12)
        dp = pre.m_a * (t.v_a - pre.v_a) + pre.m_b * (t.v_b - pre.v_b)
        np.testing.assert_allclose(dp, -M * r["delta_speed"] * pre.n, atol=1e-10)
        assert r["delta_speed"] == pytest.approx(r["delta_split"] * (1 + e) * pre.approach_speed, rel=1e-10)
        # the recorded η is laws.momentum_exchange's η; restitution and the deflection sum are untouched
        assert r["eta"] == pytest.approx(eta_ref(pre, t.v_a, t.v_b), rel=1e-12) and r["eta"] > 0.05
        assert TW.realised_restitution(pre, t.v_a, t.v_b) == pytest.approx(e, abs=1e-10)
        a, b = TW.deflections(pre, t.v_a, t.v_b)
        assert a + b == pytest.approx((1 + e) * pre.approach_speed, rel=1e-10)
        assert 0.05 - 1e-9 <= r["split_edited"] <= 0.95 + 1e-9          # nobody deflected backwards
        assert not t.momentum_conserved and TW.momentum_residual(pre, t.v_a, t.v_b) > 1e-6
        assert t.energy_physical == (TW.kinetic_energy(pre, t.v_a, t.v_b) <= TW.ke_pre(pre) + 1e-12)
        assert r["ke_ratio_vs_legal"] == pytest.approx(TW.kinetic_energy(pre, t.v_a, t.v_b) / TW.kinetic_energy(pre, la, lb))
        assert r["faster_than_legal"] == (r["speed_ratio_vs_legal"] > 1)
        assert abs(r["delta_split"]) <= mag + 1e-12 and (abs(r["delta_split"]) >= 0.10 - 1e-12)


def test_sign_both_is_two_sided_in_kinetic_energy(group):
    pre, pa, pb, mag, vt, _ = group
    v_cm = (pre.m_a * pre.v_a + pre.m_b * pre.v_b) / (pre.m_a + pre.m_b)
    for e in E_VIOL:
        pos, neg = vt.get(TW.violator_name(e, "pos")), vt.get(TW.violator_name(e, "neg"))
        if pos is not None:      # drift along v_cm . n: always more KE than the legal base
            assert pos.record["ke_ratio_vs_legal"] > 1.0
            assert -pos.record["delta_speed"] * (v_cm @ pre.n) >= 0
        if neg is not None:      # F1's KE-reducing sign
            assert -neg.record["delta_speed"] * (v_cm @ pre.n) <= 0
        if pos is not None and neg is not None:
            assert np.sign(pos.record["delta_split"]) == -np.sign(neg.record["delta_split"])


def test_both_signs_mostly_fit_and_speeds_are_two_sided():
    rng = np.random.default_rng(7)
    n_both, faster, total = 0, 0, 0
    for _ in range(400):
        pre = random_pre(rng)
        pa, pb = honest(pre)
        vt, _ = TW.f2_violators(pre, pa, pb, 1.0, E_VIOL, float(rng.uniform(0.10, 0.35)))
        n_both += all(TW.violator_name(1.0, s) in vt for s in ("pos", "neg"))
        faster += sum(t.record["faster_than_legal"] for t in vt.values())
        total += len(vt)
    assert n_both / 400 > 0.5
    assert 0.3 < faster / total < 0.7


def test_ladder_alone_is_unchanged():
    rng = np.random.default_rng(3)
    pre = random_pre(rng)
    pa, pb = honest(pre)
    a = TW.f2_twins(pre, pa, pb, 1.0)
    b = TW.f2_twins(pre, pa, pb, 1.0, TW.F2_LEVELS, impossible_levels=())
    assert list(a) == list(b) == [f"e{e:.1f}" for e in TW.F2_LEVELS]
    for k in a:
        np.testing.assert_array_equal(a[k].v_a, b[k].v_a)
        assert a[k].momentum_conserved and a[k].energy_physical and a[k].record == b[k].record


@pytest.mark.parametrize("seed", range(50))
def test_impossible_arms_create_energy_and_conserve_momentum(seed):
    rng = np.random.default_rng(seed)
    pre = random_pre(rng)
    pa, pb = honest(pre)
    levels = (1.1, 1.2, 1.3, 1.4, 1.5)
    tw = TW.f2_twins(pre, pa, pb, 1.0, impossible_levels=levels)
    assert list(tw)[11:] == ["e1.1", "e1.2", "e1.3", "e1.4", "e1.5"]
    ke0 = TW.ke_pre(pre)
    prev = ke0
    for e in levels:
        t = tw[f"e{e:.1f}"]
        ke = TW.kinetic_energy(pre, t.v_a, t.v_b)
        assert ke - ke0 == pytest.approx(0.5 * pre.mu * (e * e - 1) * pre.approach_speed ** 2, rel=1e-9)
        assert t.record["ke_ratio_vs_pre"] == pytest.approx(ke / ke0)
        assert t.record["kind"] == "energy_create" and t.record["lambda"] == pytest.approx((1 + e) / 2)
        assert ke > prev and not t.energy_physical and t.momentum_conserved
        assert TW.momentum_residual(pre, t.v_a, t.v_b) < 1e-10
        assert TW.realised_restitution(pre, t.v_a, t.v_b) == pytest.approx(e, abs=1e-10)
        assert t.pre_segment == tw["e0.0"].pre_segment and t.masses is None and t.boost is None   # same pre frames
        prev = ke
    with pytest.raises(ValueError):
        TW.f2_twins(pre, pa, pb, 1.0, impossible_levels=(1.0,))


def test_f2_config_keys():
    from physlaw.config import load_config
    t = load_config(REPO / "configs" / "data" / "f2_restitution.yaml").twins
    assert list(t.violators) == [1.0, 0.5, 0.15] and list(t.violator_signs) == ["pos", "neg"]
    assert list(t.impossible_levels) == [1.1, 1.2, 1.3, 1.4, 1.5]


@pytest.mark.skipif(not KUBRIC_PY.exists(), reason="Kubric venv absent")
def test_generate_no_render_ladder_backward_compatible(tmp_path):
    """generate.py --no-render: with violators and impossible arms off, the ladder labels equal the full run's."""
    seed = 200003
    out = {}
    for tag, extra in (("full", []), ("off", ["--set", "twins.violators=[]", "twins.impossible_levels=[]"])):
        d = tmp_path / tag
        r = subprocess.run([str(KUBRIC_PY), str(REPO / "physlaw/data/generate.py"), "--config",
                            str(REPO / "configs/data/f2_restitution.yaml"), "--seed", str(seed), "--out", str(d),
                            "--no-render", *extra], capture_output=True, text=True, timeout=600,
                           env={**os.environ, "PYTHONPATH": str(REPO)})
        line = next(x for x in r.stdout.splitlines() if x.startswith("RESULT "))
        out[tag] = (json.loads(line[7:]), d / f"f2_{seed}")
    full, off = out["full"][0], out["off"][0]
    assert full["status"] == off["status"] == "ok" and full["accepted_attempt"] == off["accepted_attempt"]
    assert off["twins"] == [f"e{e:.1f}" for e in TW.F2_LEVELS]
    skip = {"config_hash", "render_seconds", "sampling_attempts"}
    strip = lambda x: ({k: strip(v) for k, v in x.items() if k not in skip} if isinstance(x, dict)
                       else [strip(v) for v in x] if isinstance(x, list) else x)
    for t in off["twins"]:
        a, b = (json.loads((out[k][1] / t / "labels.json").read_text()) for k in ("full", "off"))
        assert strip(a) == strip(b), t
    for t in full["twins"]:
        pl = json.loads((out["full"][1] / t / "labels.json").read_text())["physlaw"]
        if t.startswith("mom_violate"):
            assert pl["legal"] is False and pl["violated_law"] == "momentum" and "eta" in pl["edit"]
            assert pl["edit"]["drift_sign"] == t.rsplit("_", 1)[1]
        elif float(t[1:]) > 1:
            assert pl["legal"] is False and pl["violated_law"] == "energy" and pl["energy_physical"] is False
            assert pl["momentum_conserved"] is True and pl["summary"]["ke_ratio"] > 1 and pl["e_target"] == float(t[1:])
        else:
            assert pl["legal"] is True
