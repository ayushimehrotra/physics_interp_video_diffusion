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
    tw = TW.f4_twins(pre, pa, pb, 1.0, {"u0": np.zeros(2), "usmall": 0.5 * u, "ularge": u},
                     camera_boosts=TW.F4_CAMERA_BOOSTS)
    assert len(tw) == 10
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
            # camera boost: world = the u0 twin, the camera's rest frame = the ball-boost twin
            c = tw[f"cam_{b}_{tag}"]
            assert c.pre_segment == f"cam_{b}" and c.boost is None and c.record["boost_kind"] == "camera"
            assert np.allclose(c.v_a, ref.v_a) and np.allclose(c.pre(pre).v_a, ref.pre(pre).v_a)
            cp_, ca, cb = c.apparent(pre)
            assert np.allclose(ca, t.v_a, atol=1e-14) and np.allclose(cb, t.v_b, atol=1e-14)
            assert np.allclose(cp_.v_a, p.v_a, atol=1e-14) and np.allclose(cp_.v_b, p.v_b, atol=1e-14)
            assert np.allclose(c.record["camera_velocity"], -np.asarray(t.boost))


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
def test_every_twin_has_a_clean_window(fam, sub):
    """Every closed-form twin carries clean_until_s (the first secondary event): after the event, inside the clip,
    every non-aux body wholly in frame up to it; F7 projectile / drop end it at the first floor contact."""
    for sc in scenes(fam, sub, 3):
        T = next(iter(sc.twins.values())).tracks[sc.bodies[0].name].p.shape[0]
        for tw in sc.twins.values():
            assert tw.clean_until_s is not None, (sub, tw.name)
            assert sc.t_event < tw.clean_until_s <= (T - 1) / sc.fps + 1e-12, (sub, tw.name)
            stop = int(math.floor(tw.clean_until_s * sc.fps + 1e-9)) + 1
            for b in (b for b in sc.bodies if not b.aux):
                assert not FAM.outside(sc.camera, b, tw.tracks[b.name].p[sc.first_post:stop]).any(), (sub, tw.name)
            if fam == "f7" and sub in ("projectile", "drop"):
                land = tw.record["first_landing_s"] if sub == "projectile" else \
                    sc.constants["release_time_s"] + math.sqrt(2 * sc.constants["drop_height"] / sc.gravity)
                assert tw.clean_until_s <= land + 1e-12, (sub, tw.name)
                if not FAM.outside(sc.camera, sc.bodies[0], tw.tracks[sc.bodies[0].name].p[sc.first_post:]).any():
                    # in frame throughout: the window ends exactly at the first floor contact (heavy_x13: earlier)
                    assert tw.clean_until_s == pytest.approx(land) or tw.name == "heavy_x13"


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


def test_ramp_speed_is_sqrt_2g_dz_both_ways():
    """F8 block on a surface path down, flat, up (concave fillets at both feet), no pad: the energy (translation +
    the block's rotation while it bridges a fillet + potential) is conserved, the block turns back at its release
    height and returns to the release point; the closed-form geometry's u-derivatives match finite differences."""
    from physlaw.data.families import f8_ramp as R
    surf = R.Path(np.array([0.0, 0.0, 2.0]), [0.8, 0.0, -0.9], [2.0 / math.sin(0.8), 4.0, 12.0], 0.0, 2.4)
    ct = R.Contact(surf, 0.4)
    for u in np.linspace(0.3, 11.0, 37):
        g, gp, gm = ct.geo(u), ct.geo(u + 1e-6), ct.geo(u - 1e-6)
        assert np.allclose(np.subtract(gp[0], gm[0]) / 2e-6, g[1], atol=1e-6)
        assert np.allclose(np.subtract(gp[1], gm[1]) / 2e-6, g[2], atol=1e-4)
        assert (gp[3] - gm[3]) / 2e-6 == pytest.approx(g[4], abs=1e-6)
    tau = np.linspace(0, 6, 601)
    U, V, ev = R.motion(ct, tau, 6.0)
    st = ct.state(U)
    E = 0.5 * ((st["dc"] ** 2).sum(1) + ct.k * st["dth"] ** 2) * V ** 2 + 9.81 * st["c"][:, 2]
    assert np.allclose(E, E[0], rtol=1e-8)
    apex = [e for e in ev if e[0] == "apex"]
    assert len(apex) >= 2 and ct.geo(apex[0][2])[0][1] == pytest.approx(st["c"][0, 2], abs=1e-7)
    assert apex[1][2] == pytest.approx(0.0, abs=1e-6)                 # back at the release point


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


def _static_uv(cam, st):
    c, h = np.asarray(st.position, float), np.asarray(st.half, float)
    return cam.project(np.array([c + h * np.array([i, j, k]) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]))


def test_scene_framing():
    """Render-check framing (2026-11-11), at the master's 832 x 480: F9 bob >= 70 px with the beam (and the damper)
    wholly in frame and the pendulum filling most of the binding dimension; F7 bounce ball >= 50 px with floor contact
    .. the eps = 1.5 apex filling most of the height; F10 camera without yaw (equal world x -> equal screen x up to the
    long lens' perspective); F12 never touches again within the clip and its bar stays in frame to the clip end; F4
    spheres >= 45 px; F8 block >= 70 px with release .. the eps = 1.5 apex in frame and filling most of the frame."""
    W, H = FMT["resolution"]
    for sub in ("pair", "swing"):
        for sc in scenes("f9", sub, 4):
            cam = sc.camera
            depth = -cam.position[1]
            for n in sc.constants["bobs"]:
                assert 2 * sc.body(n).size * cam.px_per_m(depth) >= 70.0
            v = [_static_uv(cam, st)[:, 1] for st in sc.statics]
            for st in sc.statics:
                uv = _static_uv(cam, st)
                assert uv[:, 0].min() >= 0 and uv[:, 0].max() <= W and uv[:, 1].min() >= 0 and uv[:, 1].max() <= H, st.name
            bob_v = np.concatenate([cam.project(tw.tracks[n].p)[:, 1] for tw in sc.twins.values()
                                    for n in sc.constants["bobs"]])
            r_px = sc.body("A").size * cam.px_per_m(depth)
            span_v = max(bob_v.max() + r_px, max(x.max() for x in v)) - min(x.min() for x in v)
            bob_u = np.concatenate([cam.project(tw.tracks[n].p)[:, 0] for tw in sc.twins.values()
                                    for n in sc.constants["bobs"]])
            assert span_v >= 0.75 * H or (bob_u.max() - bob_u.min() + 2 * r_px) >= 0.75 * W
    for sc in scenes("f7", "bounce", 4):
        cam, b = sc.camera, sc.bodies[0]
        depth = -cam.position[1]
        assert 2 * b.size * cam.px_per_m(depth) >= 50.0
        top = cam.project(np.array([0.0, 0.0, 2 * b.size + 1.5 ** 2 * sc.constants["drop_height"]]))[1]
        floor = cam.project(np.array([0.0, 0.0, 0.0]))[1]
        assert 0 < top and floor < H and floor - top >= 0.75 * H
    for sc in scenes("f10", "rolling", 4):
        cam = sc.camera
        assert np.allclose(cam.rotation()[0], [1.0, 0.0, 0.0])                    # image x = world x: no yaw
        tw = sc.twins["tie"]                                                     # same world x in both lanes
        stop = int(math.floor(tw.clean_until_s * sc.fps + 1e-9)) + 1
        ua, ub = (cam.project(tw.tracks[n].p[:stop])[:, 0] for n in "AB")
        assert np.all(np.abs(ua - ub) <= 0.03 * np.abs(ua - W / 2) + 1e-6)
    from physlaw.data.families import f12_spin as F12
    for sc in scenes("f12", "spin", 4):
        k, fps = sc.first_post, sc.fps
        c = sc.constants
        for tw in sc.twins.values():
            psi = np.array([2 * math.atan2(q[3], q[0]) for q in tw.tracks["B"].q])
            assert not F12._recontact(tw.tracks["A"].p[k:], tw.tracks["B"].p[k:], psi[k:], sc.body("A").size,
                                      c["bar_half_length"], c["bar_half_width"]).any(), tw.name
            assert not FAM.outside(sc.camera, sc.body("B"), tw.tracks["B"].p[k - fps:]).any(), tw.name
    from physlaw.config import load_config
    from physlaw.data import kubric_scene as KS
    from pathlib import Path
    d = load_config(Path(__file__).resolve().parents[1] / "configs" / "data" / "f4_galilean.yaml").camera_distance
    assert 2 * KS.RADIUS_RANGE[0] * FAM.top_camera(FMT, height=d).px_per_m(d) >= 45.0
    # F8 (2026-11-12): fitted camera; block >= 70 px; the block wholly in frame from frame 0 (release) to every rung's
    # clean_until_s (the eps = 1.5 apex), pad wholly in frame; the motion fills most of the binding dimension
    for sc in scenes("f8", "ramp", 6):
        cam, b = sc.camera, sc.bodies[0]
        depth = -cam.position[1]
        assert 2 * b.size * cam.px_per_m(depth) >= 70.0
        assert sc.inframe_reason() is None and sc.inframe["start"] == 0
        pad = next(st for st in sc.statics if st.name == "pad")
        uv = _static_uv(cam, pad)
        assert uv[:, 0].min() >= 0 and uv[:, 0].max() <= W and uv[:, 1].max() <= H
        top = max(sc.twins.values(), key=lambda t: t.epsilon)
        stop = int(math.floor(top.clean_until_s * sc.fps + 1e-9)) + 1
        p = cam.project(top.tracks["A"].p[:stop])
        r = math.sqrt(3) * b.size * cam.px_per_m(depth)
        assert (p[:, 1].max() - p[:, 1].min() + 2 * r) >= 0.75 * H or (p[:, 0].max() - p[:, 0].min() + 2 * r) >= 0.75 * W


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


# --------------------------------------------------------------------------- #
# F6 glide / slide / pair, F9, F11, F12: closed forms
# --------------------------------------------------------------------------- #


def test_f6_glide_twins():
    from physlaw.eval import laws as L
    for sc in scenes("f6", "glide", 3):
        k = sc.first_post
        v0 = sc.twins["legal"].tracks["A"].v[k - 1]
        for name, tw in sc.twins.items():
            v1 = tw.tracks["A"].v[k]
            st = L.momentum.stats({"v": v0[None]}, {"v": v1[None]}, {"force_free": True})
            want = 1.0 if tw.legal else tw.record["speed_factor"]
            assert st["inertia_speed"] == pytest.approx(want, rel=1e-12), name
            assert st["inertia_turn"] == pytest.approx(0.0 if "turn" not in name else tw.violation_magnitude, abs=1e-12)
            assert tw.clean_until_s >= sc.t_event + 1.0


def _scene_labels(sc, name):
    import json
    from physlaw.data import generate as GEN
    tw = sc.twins[name]
    return json.loads(json.dumps({"per_frame": GEN.per_frame_labels(sc, tw, sc.camera, None),
                                  "scene_constants": {"num_frames": len(tw.tracks["A"].p), "frame_rate": sc.fps},
                                  "physlaw": {"family": sc.family, "twin": name, **GEN.law_fields(sc, tw, sc.camera)}},
                                 default=GEN._json_default))


@pytest.mark.parametrize("sub", ["slide", "pair"])
def test_f6_slide_closed_form_and_law_statistics(sub):
    """F6 slide / pair: the block slides at frame 0 (pre-event deceleration = mu g, shared by every twin), the branch
    is mid-slide, the legal stop is inside the clip POST_STOP_S[sub] before its end (slide 2.1-2.4 s, pair 0.5-0.6 s:
    >= 0.5 s at rest), every twin in frame to the clip end with the block >= MIN_BLOCK_PX; the effects are visible at
    832 px (slide: reverse >= 150 px back from the legal stop at the clip end; pair: heavy_x13's heavier block stops
    >= 35 px short; no no_friction twin since 2026-11-12); the closed form integrates (positions = trapezoid of
    velocities wherever the acceleration is constant); law statistics from the labels (io.law_inputs) at their legal
    values for the legal twin and off them as designed for each violator."""
    from physlaw.data.families import f6_friction as F
    from physlaw.data.io import law_inputs
    from physlaw.eval import laws as L
    for sc in scenes("f6", sub, 3):
        T, fps, k, te = 289, sc.fps, sc.first_post, sc.t_event
        c = sc.constants
        assert sc.event_kind == "branch" and te == pytest.approx((k - 1) / fps)
        lo, hi = F.POST_STOP_S[sub]
        assert (T - 1) / fps - hi - 1e-9 <= c["t_stop_s"] <= (T - 1) / fps - lo + 1e-9 and c["t_stop_s"] > te + 0.9
        assert "no_friction" not in sc.twins
        u_end = lambda name, n: float(sc.camera.project(sc.twins[name].tracks[n].p[-1])[0])
        if sub == "slide":
            assert abs(u_end("reverse", "A") - u_end("legal", "A")) >= 150.0
        else:
            h = c["heavier"]
            assert abs(u_end("heavy_x13", h) - u_end("legal", h)) >= 35.0
            assert sc.twins["heavy_x13"].constants["t_stop_twin_s"] <= (T - 1) / fps - 0.5
        assert sc.params["block_px"] >= F.MIN_BLOCK_PX and not sc.inframe_reason()
        assert all(tw.clean_until_s == pytest.approx((T - 1) / fps) for tw in sc.twins.values())
        leg = sc.twins["legal"]
        for n, tr in leg.tracks.items():
            sp = np.linalg.norm(tr.v, axis=1)
            assert sp[0] == pytest.approx(c["v0"]) and sp[k - 1] > 0.3 * c["v0"]           # sliding at frame 0 and the branch
            moving = np.nonzero(sp > 0)[0]
            assert moving[-1] == int(math.floor(c["t_stop_s"] * fps - 1e-9))              # at rest from t_stop_s
            const = [f for f in range(T - 1) if f != moving[-1]]                         # constant acceleration pairs
            dp = (tr.p[1:] - tr.p[:-1])[const] * fps - 0.5 * (tr.v[1:] + tr.v[:-1])[const]
            assert np.abs(dp).max() < 1e-9
        stats = {name: L.friction.stats(*law_inputs(_scene_labels(sc, name))) for name in sc.twins}
        assert stats["legal"]["decel_const"] == pytest.approx(1.0, abs=1e-6) and stats["legal"]["stop_hold"] < 1e-9
        if sub == "pair":                       # the near block (B) never hides the far one (A): disjoint image rows
            box = lambda n: sc.camera.project(np.array(c["start"][n]) + c["half_size"] * np.array(
                [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]))[:, 1]
            assert box("A").max() < box("B").min()
            assert stats["legal"]["mass_ratio"] == pytest.approx(1.0, abs=1e-6)
            assert stats["heavy_x13"]["mass_ratio"] == pytest.approx(F.HEAVY_FACTOR, rel=1e-6)
            continue
        assert stats["reverse"]["decel_const"] == pytest.approx(1.0, abs=1e-6) and stats["reverse"]["stop_hold"] > 0.2
        assert 0.4 < stats["viscous"]["decel_const"] < 0.9 and 0.05 < stats["viscous"]["stop_hold"] < math.exp(-1)
        for kf in F.DECEL_LADDER:
            assert stats[f"decel_x{kf:g}"]["decel_const"] == pytest.approx(kf, rel=1e-6)
        assert not sc.twins["reverse"].flags["energy_physical"] and sc.twins["viscous"].flags["energy_physical"]


def test_f9_exact_pendulum_matches_integration():
    from scipy.integrate import solve_ivp
    from physlaw.data.families import f9_pendulum as P
    L_, r, amp = 1.1, 0.13, 0.6
    w0 = P.omega0(L_, r)
    t = np.linspace(0.0, 3.0, 61)
    th, td = P.free_swing(t, 0.0, amp, w0)
    sol = solve_ivp(lambda _, y: [y[1], -w0 * w0 * math.sin(y[0])], (0, 3), [amp, 0.0], method="DOP853",
                    rtol=1e-12, atol=1e-12, t_eval=t)
    assert np.allclose(th, sol.y[0], atol=1e-9) and np.allclose(td, sol.y[1], atol=1e-9)
    T = P.period(amp, w0)
    assert P.free_swing([T], 0.0, amp, w0)[0][0] == pytest.approx(amp, abs=1e-9)
    assert T > 2 * math.pi / w0                                    # large-angle period exceeds the small-angle one
    tau = P.time_to_angle(amp, 0.2, w0)
    assert abs(P.free_swing([tau], 0.0, amp, w0)[0][0]) == pytest.approx(0.2, abs=1e-12)
    # pair twins: the branch at an apex keeps theta and theta' continuous; the scaled bob's period is k T
    for sc in scenes("f9", "pair", 3):
        k = sc.first_post
        for tw in sc.twins.values():
            for n in sc.constants["bobs"]:
                assert np.linalg.norm(tw.tracks[n].v[k - 1]) < 1e-9          # apex at the branch frame
                assert tw.record["period_post_s"][n] == pytest.approx(
                    sc.constants["period_legal_s"] * tw.record["period_factor"][n])


def test_f9_swing_energy_ladder():
    from physlaw.eval import laws as L
    for sc in scenes("f9", "swing", 3):
        pb = np.array(sc.constants["p_bottom"])
        assert set(sc.stop_dose) == {"A"} and math.isnan(sc.stop_dose["A"])
        for tw in sc.twins.values():
            es, ap = tw.event_state, tw.record["apex_states"]
            pre, post = ({"p": np.array(ap[s]["p"][:1]), "v": np.array(ap[s]["v"][:1])} for s in ("pre", "post"))
            assert np.abs(pre["v"]).max() == 0.0 and np.abs(post["v"]).max() < 1e-12      # apexes: at rest
            eps = L.epsilon(pre, post, {"event": "swing", "p_bottom": pb})                   # heights only, no g
            assert eps == pytest.approx(tw.epsilon, abs=1e-9) and tw.legal == (tw.epsilon <= 1.0)
            assert ap["post"]["t_s"] == pytest.approx(tw.record["t_apex_after_s"])
            i = int(round(ap["post"]["t_s"] * sc.fps))                                       # the track agrees
            assert tw.tracks["A"].p[i] == pytest.approx(ap["post"]["p"][0], abs=0.02)
            assert es["pre"]["t_s"] == pytest.approx(sc.t_event) and tw.clean_until_s > tw.record["t_apex_after_s"]


def test_f11_tipping_closed_form():
    from physlaw.data.families import f11_support as P
    from physlaw.eval import laws as L
    a, beta = 0.8, 0.5
    for x in (0.55, 0.85, 1.15, 1.45):
        C, V, PH, W, info = P.tip_motion(np.array([0.0]), a, beta * a, x)
        assert np.linalg.norm(info["a0"]) == pytest.approx(9.81 * float(L.tip_factor(x, beta)), rel=1e-12)
        assert info["a0"][0] == pytest.approx(0.0, abs=1e-12)              # frictionless edge: vertical reaction
        assert math.copysign(1.0, info["a0"][1]) == (-1.0 if x > 1 else 1.0)
    mirrored = [np.linalg.norm(P.tip_motion(np.array([0.0]), a, beta * a, x)[4]["a0"]) for x in (0.75, 1.25)]
    assert mirrored[0] == pytest.approx(mirrored[1], rel=1e-12)            # smooth in |x - 1| across the boundary
    seen = set()
    for sc in scenes("f11", "support", 6):
        x = sc.constants["overhang_x"]
        seen.add(x > 1)
        assert sc.twins["tip"].legal == (x > 1) and sc.twins["stay"].legal == (x < 1)
        post = sc.twins["tip"].event_state["post"]
        a0 = np.linalg.norm(post["a"][0])
        assert a0 == pytest.approx(sc.gravity * float(L.tip_factor(x, sc.constants["aspect_ba"])), rel=1e-12)
        for tw in sc.twins.values():
            k, f1 = sc.first_post, sc.first_post + int(0.5 * sc.fps)
            q0, q1 = tw.tracks["A"].q[k - 1], tw.tracks["A"].q[f1]
            phi = 2.0 * math.acos(min(1.0, abs(float(np.dot(q0, q1)))))       # rotation over the tip window
            post = {"v": np.zeros((1, 3)), "alpha": np.array(tw.event_state["post"]["alpha"][:1]), "phi": [phi]}
            ctx = {"overhang_x": x, "aspect_ba": sc.constants["aspect_ba"], "g_clip": sc.gravity,
                   "half_length": sc.constants["half_length"]}
            st = L.statics.stats({"v": np.zeros((1, 3))}, post, ctx)
            assert st["tip_consistency"] == float(tw.legal)
            if x > 1:
                assert st["alpha_ratio"] == (pytest.approx(1.0, rel=1e-9) if tw.name == "tip" else 0.0)
            else:
                assert math.isnan(st["alpha_ratio"])
        T = sc.twins["stay"].tracks["A"].p.shape[0]
        assert sc.body("P").aux and sc.twins["stay"].clean_until_s == pytest.approx((T - 1) / sc.fps)
    assert seen == {True, False}


def test_f12_impulse_conserves_momentum_and_angular_momentum():
    from physlaw.eval import laws as L
    for sc in scenes("f12", "spin", 3):
        k, te, fps = sc.first_post, sc.t_event, sc.fps
        m = np.array([b.mass for b in sc.bodies])
        ctx = {"m": m, "I": np.array([b.inertia for b in sc.bodies])}
        assert sc.body("B").inertia == pytest.approx(sc.body("B").mass * (sc.constants["bar_half_length"] ** 2 +
                                                                          sc.constants["bar_half_width"] ** 2) / 3)
        for tw in sc.twins.values():
            at = lambda f: {"p": np.stack([tw.tracks[n].p[f] + tw.tracks[n].v[f] * (te - f / fps) for n in "AB"]),
                            "v": np.stack([tw.tracks[n].v[f] for n in "AB"]),
                            "w": np.stack([tw.tracks[n].w[f] for n in "AB"])}
            pre, post = at(k - 1), at(k)
            assert np.allclose(m @ post["v"], m @ pre["v"], atol=1e-12)
            Lr = L.angular_momentum.stats(pre, post, ctx)["L_ratio"]
            assert (Lr == pytest.approx(1.0, abs=1e-12)) == tw.legal, tw.name
            if not tw.legal:
                assert (Lr > 1) == (tw.violation_magnitude > 1)
            assert tw.clean_until_s >= te + 0.6
            # audit M3: energy_physical = (KE after <= KE before) for every twin; spin_ke_x<k> keep the legal KE
            # (bar spin x k paid by the radial relative translation): they break angular momentum only
            ke = lambda st: 0.5 * float(m @ (st["v"][:, :2] ** 2).sum(-1) + ctx["I"] @ st["w"][:, 2] ** 2)
            assert tw.flags["energy_physical"] == (ke(post) <= ke(pre) * (1 + 1e-12)), tw.name
            if tw.name.startswith("spin_ke_x"):
                assert ke(post) == pytest.approx(tw.record["ke_legal"], rel=1e-9) and tw.flags["energy_physical"]
                assert Lr == pytest.approx(tw.record["L_ratio"], rel=1e-9)
                assert Lr == pytest.approx(sc.twins[f"spin_x{tw.violation_magnitude:g}"].record["L_ratio"], rel=1e-9)
        assert {"spin_ke_x0.8", "spin_ke_x1.2"} <= set(sc.twins)


def test_f10_solid_assignment_and_kappa_vary_across_groups():
    """L4's rolling manifold is fitted across groups: kappa_A must vary (A is the solid one in some groups, a tube of
    random wall thickness in others); the legal twin's solid always leads, the violators keep their semantics."""
    kA, solids = set(), set()
    for sc in scenes("f10", "rolling", 6):
        c = sc.constants
        S, H = c["solid"], c["hollow"]
        solids.add(S)
        kA.add(round(c["inertia_factors"]["A"], 6))
        assert c["inertia_factors"][S] == pytest.approx(0.5)
        assert c["inertia_factors"][H] == pytest.approx((1 + c["tube_inner_ratio"] ** 2) / 2)
        assert sc.body(S).mass == pytest.approx(sc.body(H).mass)
        acc = {n: tw.record["accel"] for n, tw in sc.twins.items()}
        assert acc["legal"][S] > acc["legal"][H]
        assert acc["hollow_first"][H] > acc["hollow_first"][S] and acc["tie"][H] == acc["tie"][S]
    assert solids == {"A", "B"} and len(kA) > 2


@pytest.mark.parametrize("cfg_name,fam,sub,key", [("f7_freefall.yaml", "f7", "bounce", "bounce_ladder"),
                                                  ("f8_ramp.yaml", "f8", "ramp", "epsilon_ladder")])
def test_energy_ladders_from_the_configs(cfg_name, fam, sub, key):
    """The configured ladders (21 rungs, eps 0.50 ... 1.50 step 0.05): every rung passes acceptance and the in-frame
    check, all rungs share the pre-event frames, twin names are eps<e> (eps0.5, eps0.55, ..., eps1.0, ..., eps1.5)
    and parse back to the rung."""
    from pathlib import Path
    from physlaw.config import load_config
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "data" / cfg_name)
    ladder = [float(e) for e in cfg.twins[key]]
    assert ladder == pytest.approx([0.5 + 0.05 * i for i in range(21)])
    names = [FAM.eps_name(e) for e in ladder]
    assert names[:3] == ["eps0.5", "eps0.55", "eps0.6"] and names[10] == "eps1.0" and names[-1] == "eps1.5"
    assert [float(n[3:]) for n in names] == pytest.approx(ladder) and len(set(names)) == 21
    mod, n_ok = FAM.family_module(fam), 0
    for seed in range(40):
        sc = mod.sample(np.random.default_rng([seed, 0]), cfg, FMT, sub)
        if isinstance(sc, str) or sc.inframe_reason():
            continue
        sc.check_pre_identity()
        assert list(sc.twins) == names
        for tw in sc.twins.values():
            assert FAM.acceptance(FAM.View.from_scene(sc, tw)) == [], tw.name
            assert tw.legal == (tw.epsilon <= 1.0) and float(tw.name[3:]) == pytest.approx(tw.epsilon)
        n_ok += 1
        if n_ok == 4:
            break
    assert n_ok >= 3


@pytest.mark.parametrize("width,height", [(416, 240), (1248, 720)])
def test_collision_labels_camera_matches_render_geometry(width, height, monkeypatch):
    """F1-F4 at a non-master width without ``master.sensor_width_mm``: the Kubric scene (render camera and in-frame
    check, ``KS.SENSOR_WIDTH`` / ``KS.half_extents``) and the labels' camera (``FAM.top_camera(fmt)``) share one
    field of view, so labelled pixel positions are where the render draws the spheres (closed-form pinhole check)."""
    from physlaw.config import load_config
    from physlaw.data import generate as GEN
    from physlaw.data import kubric_scene as KS
    cfg = load_config(GEN.REPO / "configs" / "data" / "f1_collisions.yaml",
                      [f"master.width={width}", f"master.height={height}"])
    assert "sensor_width_mm" not in cfg.master
    fmt = GEN.master_fmt(cfg)
    for k in ("NUM_FRAMES", "FRAME_RATE", "RESOLUTION", "TARGET_CONTACT_FRAME", "PRE_FRAME", "POST_FRAME",
              "SENSOR_WIDTH", "SAMPLES_PER_PIXEL", "CONTACT_TOLERANCE", "CAMERA_DISTANCE", "ASPECT", "DENOISE"):
        monkeypatch.setattr(KS, k, getattr(KS, k))      # restore the module's scene constants afterwards
    GEN.configure_kubric(cfg, fmt, 120)
    assert KS.SENSOR_WIDTH == pytest.approx(GEN.MASTER_SENSOR_WIDTH_MM) == pytest.approx(fmt["sensor_width_mm"])
    cam = FAM.top_camera(fmt, height=KS.CAMERA_DISTANCE, z_plane=KS.CENTER_HEIGHT)
    # Kubric's straight-down camera (identity quaternion, image-up = +y) at the render's focal length / sensor width
    fx = KS.FOCAL_LENGTH / KS.SENSOR_WIDTH * width
    fy = fx * (height / width) * KS.ASPECT
    rng = np.random.default_rng(0)
    half_w, half_h = KS.half_extents()
    for _ in range(20):
        x, y = rng.uniform(-half_w, half_w), rng.uniform(-half_h, half_h)
        u, v, depth = cam.project(np.array([x, y, KS.CENTER_HEIGHT]))
        assert depth == pytest.approx(KS.CAMERA_DISTANCE)
        assert u == pytest.approx(width / 2 + fx * x / KS.CAMERA_DISTANCE, abs=1e-6)
        assert v == pytest.approx(height / 2 - fy * y / KS.CAMERA_DISTANCE, abs=1e-6)
    # the in-frame check's visible region is exactly the labels' image: its corners project to the image corners
    u, v, _ = cam.project(np.array([half_w, -half_h, KS.CENTER_HEIGHT]))
    assert (u, v) == pytest.approx((width, height), abs=1e-6)
    # the labels' visible box (families' in-frame check) is the Kubric in-frame check's
    assert FAM.visible_box(cam, KS.CAMERA_DISTANCE) == pytest.approx((half_w, half_h))
