"""Spec §2.2 twin identities, checked on the twin algebra directly (no rendering).

Random frictionless pre-contact states; the honest outcome is the exact legal
outcome at e = 1, which is what the simulator delivers to ~1e-8.
"""
from __future__ import annotations

import math

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


# --------------------------------------------------------------------------- #
# F3 / F4 algebra
# --------------------------------------------------------------------------- #


def test_f3_hidden_mass_twins_are_legal(case):
    pre, _, _ = case
    tw = TW.f3_twins(pre, 1.0)
    assert [t.record["mass_ratio"] for t in tw.values()] == list(TW.F3_RATIOS)
    for t in tw.values():
        p = t.pre(pre)
        assert p.m_a / p.m_b == pytest.approx(t.record["mass_ratio"], rel=1e-12)
        assert p.m_a * p.m_b == pytest.approx(pre.m_a * pre.m_b, rel=1e-12)     # only the ratio moves
        assert TW.momentum_residual(p, t.v_a, t.v_b) < 1e-10
        assert TW.realised_restitution(p, t.v_a, t.v_b) == pytest.approx(1.0, abs=1e-10)


def test_f4_boosts_keep_restitution_and_com_angle(case):
    pre, _, _ = case
    pa, pb = honest(pre)
    u = np.array([0.4, -0.3])
    tw = TW.f4_twins(pre, pa, pb, 1.0, {"u0": np.zeros(2), "usmall": 0.5 * u, "ularge": u})
    for tag in ("e1", "e015"):
        ref = tw[f"u0_{tag}"]
        r0 = TW.realised_restitution(ref.pre(pre), ref.v_a, ref.v_b)
        a0 = TW.com_frame_angle(ref.pre(pre), ref.v_a, ref.v_b)
        for b in ("usmall", "ularge"):
            t = tw[f"{b}_{tag}"]
            p = t.pre(pre)
            assert TW.realised_restitution(p, t.v_a, t.v_b) == pytest.approx(r0, abs=1e-10)
            assert TW.com_frame_angle(p, t.v_a, t.v_b) == pytest.approx(a0, abs=1e-10)
            assert TW.momentum_residual(p, t.v_a, t.v_b) < 1e-10
            assert t.pre_segment == b


# --------------------------------------------------------------------------- #
# closed-form families (spec §2.4 on the twin trajectories, no rendering)
# --------------------------------------------------------------------------- #

from physlaw.data import families as FAM  # noqa: E402

FMT = {"fps": 48, "num_frames": 289, "resolution": [832, 480], "sensor_width_mm": 41.6, "aspect": 832 / 480,
       "event_frame_range": [112, 128]}
SUBS = [(f, s) for f in FAM.FAMILY_MODULES for s in FAM.family_module(f).SUBFAMILIES]


def scenes(fam, sub, n=6):
    mod = FAM.family_module(fam)
    out = []
    for seed in range(40):
        sc = mod.sample(np.random.default_rng([seed, 0]), {}, FMT, sub)
        if isinstance(sc, str) or sc.inframe_reason():
            continue
        out.append(sc)
        if len(out) == n:
            break
    return out


@pytest.mark.parametrize("fam,sub", SUBS)
def test_family_twins_pass_acceptance(fam, sub):
    scs = scenes(fam, sub)
    assert len(scs) >= 3, "sampler rejects almost everything"
    for sc in scs:
        sc.check_pre_identity()
        lo, hi = FMT["event_frame_range"]
        assert lo - 1 <= sc.t_event * sc.fps <= hi
        assert any(t.legal for t in sc.twins.values())
        for tw in sc.twins.values():
            assert FAM.acceptance(FAM.View.from_scene(sc, tw)) == [], (sc.subfamily, tw.name)
            assert tw.legal == (tw.violated_law is None)
            assert "energy_physical" in tw.flags


@pytest.mark.parametrize("fam,sub", SUBS)
def test_family_violations_are_detected(fam, sub):
    """Relabelling a violating twin as legal must fail acceptance: the checks see the violation."""
    n = 0
    for sc in scenes(fam, sub, 3):
        for tw in sc.twins.values():
            if tw.legal:
                continue
            v = FAM.View.from_scene(sc, tw)
            v.legal, v.violated_law, v.magnitude = True, None, 0.0
            v.record = {**v.record, "speed_factor": 1.0}
            assert FAM.acceptance(v), (sub, tw.name)
            n += 1
    if not any(not t.legal for sc in scenes(fam, sub, 1) for t in sc.twins.values()):
        pytest.skip("sub-family has only legal twins")
    assert n


def test_ballistic_bounce_heights():
    t = np.linspace(0, 3, 3001)
    p, v, b = FAM.ballistic(t, 0.0, [0, 0, 5.0], [1.0, 0, 0], 9.81, 0.0, 0.8)
    assert b[0] == pytest.approx(math.sqrt(2 * 5.0 / 9.81), rel=1e-12)
    vz_in = 9.81 * b[0]
    i = np.searchsorted(t, b[0])
    assert v[i, 2] == pytest.approx(0.8 * vz_in - 9.81 * (t[i] - b[0]), rel=1e-9)
    assert np.allclose(p[:, 0], t, atol=1e-12)                 # frictionless floor keeps vx


def test_ramp_bottom_speed_is_sqrt_2gh():
    from physlaw.data.families import f8_ramp as R
    for angles, drops in (([0.6, 0.0], [2.0]), ([1.0, 0.4, 0.0], [1.0, 1.0])):
        path = R.Path(np.array([0.0, 0.0, 2.3]), angles, drops, 0.3)
        *_, vb = R.slide(path, np.linspace(0, 5, 241), 0.0)
        assert vb == pytest.approx(math.sqrt(2 * 9.81 * 2.0), rel=1e-9)
        assert path.point(path.bottom_s)[2] == pytest.approx(0.3, abs=1e-12)


def test_camera_matches_f1_camera_block():
    cam = FAM.top_camera(FMT, height=20.0, z_plane=0.65)
    assert np.allclose(cam.world_to_camera()[:3, :3], np.diag([1.0, -1.0, -1.0]))
    uv = cam.project(np.array([0.0, 0.0, 0.65]))
    assert uv[:2] == pytest.approx([416.0, 240.0])
    assert cam.px_per_m(20.0) == pytest.approx(50.0)
    side = FAM.side_camera(FMT)
    assert side.project(np.array(side.look_at))[:2] == pytest.approx([416.0, 240.0])
    q = np.array(side.blender_quaternion())
    assert np.linalg.norm(q) == pytest.approx(1.0)


def test_squashed_camera_keeps_the_view():
    """PC1 256x256: same world extent as the 832x480 master, non-square pixels."""
    sq = FAM.side_camera({**FMT, "resolution": [256, 256]})
    full = FAM.side_camera(FMT)
    X = np.array([3.0, 0.0, 6.0])
    a, b = full.project(X), sq.project(X)
    assert a[0] / 832 == pytest.approx(b[0] / 256) and a[1] / 480 == pytest.approx(b[1] / 256)


def test_nuisance_selection_and_camera():
    from physlaw.data import nuisance as NZ
    frac = np.mean([NZ.selected(s, 0.2) for s in range(5000)])
    assert abs(frac - 0.2) < 0.02
    rng = np.random.default_rng(0)
    for cam in (FAM.top_camera(FMT), FAM.side_camera(FMT)):
        look = NZ.sample(rng, ["A", "B"], identical=True)
        assert look["bodies"]["A"] == look["bodies"]["B"]
        c2 = NZ.perturb_camera(cam, look)
        assert c2.look_at == cam.look_at
        assert c2.project(np.array(cam.look_at))[:2] == pytest.approx([416.0, 240.0], abs=1e-6)
        assert not np.allclose(c2.position, cam.position)


def test_subfamily_blocks():
    from physlaw.config import Cfg
    from physlaw.data.generate import subfamily_for
    cfg = Cfg(seed_offset=700000, subfamilies=Cfg(projectile=3, drop=2, bounce=1))
    got = [subfamily_for(cfg, 700000 + i) for i in range(8)]
    assert got == ["projectile"] * 3 + ["drop"] * 2 + ["bounce"] + ["projectile"] * 2


def test_pc1_seeds_disjoint_from_benchmark():
    import argparse
    import importlib.util
    from pathlib import Path
    from physlaw.config import load_config
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("gen01", root / "scripts" / "01_generate_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pc = load_config(root / "configs" / "data" / "pc1_train.yaml")
    jobs = mod.pc1_jobs(pc, argparse.Namespace(families=None, n_groups=None, out=None, set=[]))
    assert {Path(c).name for c, *_ in jobs} == {s.config for s in pc.families.values()}
    for c, sets, _, n in jobs:
        cfg = load_config(c, sets)
        assert cfg.legal_only and cfg.master.width == 256 and cfg.nuisance_frac == 0.0
