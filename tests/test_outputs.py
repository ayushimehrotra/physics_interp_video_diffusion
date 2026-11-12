"""Output-level law statistics (physlaw.eval.outputs): track A S_mom, epsilon along the line of centres, track B
mass-free checks, energy shares, and scale-free trajectory statistics and readouts (F6-F12)."""
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


def test_track_a_summary_epsilon_eta_and_energy_share(tmp_path):
    lab = _lab_fn()
    for g in GROUPS:
        for t in TWINS:
            (tmp_path / g / t).mkdir(parents=True)
            (tmp_path / g / t / "labels.json").write_text(json.dumps(lab(g, t)))
    res = O.track_a_summary(_gen_rows("legal"), _real_rows(noise=0.0), tmp_path, n_boot=50)
    assert res["momentum"]["conserve_e1"]["S_mom"] == pytest.approx(1.0) and res["momentum"]["legal"]["n_scored_clips"] == 24
    g = res["generated"]
    assert g["median_epsilon"] == pytest.approx(0.5) and g["median_eta"] == pytest.approx(0.0, abs=1e-9)
    assert res["real"]["conserve_e015"]["median_epsilon"] == pytest.approx(0.15)
    e = res["energy"]
    assert e["generated"]["share_le1"] == 1.0 and e["real"]["conserve_e1"]["eps_true"] == pytest.approx(1.0)
    assert e["real_legal"]["n"] == 12 and e["real_creating"]["n"] == 0
    assert not any(k.startswith(("r_p", "S_outcome", "res_")) for k in json.dumps(res).split('"'))


def test_energy_shares_against_the_real_distribution():
    gen = [{"group": f"g{i}", "epsilon": e} for i, e in enumerate((0.8, 1.05, 1.3, 1.6))]
    real = [{"group": "g0", "twin": t, "epsilon": e + d, "eps_true": e}
            for t, e in (("eps0.5", 0.5), ("eps1.5", 1.5)) for d in (-0.05, 0.05)]
    s = O.energy_shares(gen, real, tol=0.1, n_boot=20)
    assert s["generated"]["share_le1"] == pytest.approx(0.5)
    assert s["real_legal"]["share_le1"] == 1.0 and s["real_creating"]["share_le1"] == 0.0
    assert s["real"]["eps1.5"]["eps_true"] == 1.5 and s["generated"]["share_le1_ci95"] is not None


# ---------------------------------------------------------------- track B

def _row_b(va, vb, post, n, ra, rb, dt=0.02, group="g0", seed=0, ma=1.0, mb=2.0):
    """A readout_self row: positions at a 'contact frame' dt after the true touch (the pre lines overlap there, the
    post lines are run from the touch), n = the line of centres from them, statistics = collision_stats."""
    va, vb, n = (np.asarray(x, float) for x in (va, vb, n))
    pa, pb = (np.asarray(x, float) for x in post)
    n = n / np.linalg.norm(n)
    c = np.array([0.3, -0.2])
    x0 = np.stack([c - ra * n + va * dt, c + rb * n + vb * dt])
    x1 = np.stack([c - ra * n + pa * dt, c + rb * n + pb * dt])
    nn, miss = O.line_of_centres(x0, [va, vb], x1, [pa, pb], ra + rb)
    return {"group": group, "seed": seed, "measurable": True, "velocity_pre": [va.tolist(), vb.tolist()],
            "velocity_post": [pa.tolist(), pb.tolist()], "position_pre": x0.tolist(), "contact_normal": nn.tolist(),
            "contact_miss": miss, **O.collision_stats([va, vb], [pa, pb], nn, [ma, mb])}


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
        r = _row_b(va, vb, post, n, ra, rb, ma=ma, mb=mb)
        f = O.mass_free_clip(r, lab)
        assert f["mf_interacts"] and not f["mf_contact_miss"]
        assert r["antiparallel"] == pytest.approx(1.0) and r["eta"] == pytest.approx(0.0, abs=1e-9)
        assert f["mf_normal"] == pytest.approx(1.0) and f["mf_normal_err_deg"] == pytest.approx(0.0, abs=1e-4)
        assert r["epsilon"] == pytest.approx(e, abs=1e-9)
        assert f["mf_implied_ratio"] == pytest.approx(ma / mb) and f["mf_log_ratio_vs_label"] == pytest.approx(0.0, abs=1e-9)


def test_track_b_same_direction_push_and_pass_through():
    va, vb, n = (1.0, 0.0), (0.0, 0.0), (1.0, 0.0)
    lab = _labels(va, vb, va, vb, n)
    push = _row_b(va, vb, ([1.5, 0.0], [0.5, 0.0]), n, 0.4, 0.5)
    assert push["antiparallel"] == pytest.approx(-1.0)
    still = O.mass_free_clip(_row_b(va, vb, (va, vb), n, 0.4, 0.5), lab)
    assert not still["mf_interacts"]
    assert O.mass_free_clip({"measurable": False}, lab) == {}


def test_track_b_stats_e_share_and_ratio_consistency():
    va, vb, n = (1.0, 0.0), (0.0, 0.0), (1.0, 0.0)
    rows = []
    for g, (ma, mb) in {"g0": (1.0, 2.0), "g1": (3.0, 1.0)}.items():
        lab = _labels(va, vb, va, vb, n, ma, mb)
        for s, e in enumerate((0.3, 0.9)):                     # seeds of one scene: one mass ratio, any e
            rows.append(_row_b(va, vb, _legal_post(va, vb, n, ma, mb, e), n, 0.4, 0.5, group=g, seed=s, ma=ma, mb=mb))
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


def test_line_of_centres_not_the_impulse_direction():
    """A sideways kick (equal and opposite tangential dv on top of a legal e = 0.6 exchange along n): momentum is
    conserved and the line of centres still gives eps = 0.6; the impulse direction would give a biased e."""
    va, vb, n, ma, mb = np.array([1.0, 0.3]), np.array([-0.2, 0.0]), np.array([1.0, 0.0]), 1.0, 1.0
    pa, pb = _legal_post(va, vb, n, ma, mb, 0.6)
    kick = np.array([0.0, 0.4])
    pa, pb = pa + kick, pb - kick
    r = _row_b(va, vb, (pa, pb), n, 0.4, 0.5, ma=ma, mb=mb)
    assert r["epsilon"] == pytest.approx(0.6, abs=1e-9) and r["eta"] == pytest.approx(0.0, abs=1e-9)
    j = (pb - vb) - (pa - va)
    j /= np.linalg.norm(j)
    e_impulse = float(-(pa - pb) @ j / ((va - vb) @ j))
    assert abs(e_impulse - 0.6) > 0.05


# ---------------------------------------------------------------- trajectories: scale-free

def _parabola_track(g=1.0, vx=2.0, vz=3.0, t_branch=1.0, g_post=None, T=60, fps=30.0, n_obj=1, g_obj=None):
    t = np.arange(T) / fps
    P = np.zeros((T, n_obj, 2))
    for k in range(n_obj):
        gk = (g_obj or [g] * n_obj)[k]
        gp = (g_post or gk) if k == 0 else gk
        x = vx * t
        z = np.where(t < t_branch, vz * t - 0.5 * gk * t ** 2,
                     vz * t_branch - 0.5 * gk * t_branch ** 2 + (vz - gk * t_branch) * (t - t_branch) - 0.5 * gp * (t - t_branch) ** 2)
        P[:, k] = np.stack([x, z], 1)
    return t, P, np.ones((T, n_obj), bool), int(np.searchsorted(t, t_branch)), t_branch


@pytest.mark.parametrize("s,tau", [(1.0, 1.0), (37.0, 1.0), (1.0, 2.5), (0.01, 0.3)])
def test_traj_stats_scale_free(s, tau):
    """Pixel size (length unit) and fps (time unit) rescalings leave every statistic unchanged."""
    t, P, ok, ke, te = _parabola_track(g_post=1.5)
    base = O.traj_stats("projectile", t, P, ok, ke, te, 10)
    sc = O.traj_stats("projectile", t * tau, P * s, ok, ke, te * tau, 10)
    assert base["g_ratio"] == pytest.approx(1.5, rel=1e-6) and base["axis_ratio"] == pytest.approx(1.0, rel=1e-6)
    for k in ("g_ratio", "axis_ratio"):
        assert sc[k] == pytest.approx(base[k], rel=1e-9)
    t, P, ok, ke, te = _parabola_track(vx=0.0, vz=0.0, n_obj=2, g_obj=[1.0, 1.3])          # drop pair, heavy falls faster
    d = O.traj_stats("drop", t * tau, P * s, ok, ke, te * tau, 10, masses=[2.0, 1.0])
    assert d["mass_ratio"] == pytest.approx(1 / 1.3, rel=1e-6) and np.isnan(d["axis_ratio"])


def test_traj_stats_drop_pre_window_starts_at_the_release():
    """F7 drop balls are held at rest until the release (here 0.2 s before the branch, inside the 10-frame pre
    window): fitting the pre parabola over the rest frames biases a_pre and g_ratio on a legal drop; with
    ``t_release`` only free-flight frames enter (g_ratio = 1)."""
    fps, t_rel, t_b, g = 30.0, 0.8, 1.0, 9.81
    t = np.arange(60) / fps
    z = np.where(t < t_rel, 0.0, -0.5 * g * (t - t_rel) ** 2)
    P = np.stack([np.zeros_like(t), z], 1)[:, None].repeat(2, 1)
    ok, ke = np.ones((60, 2), bool), int(np.searchsorted(t, t_b))
    held = O.traj_stats("drop", t, P, ok, ke, t_b, 10, masses=[2.0, 1.0])
    free = O.traj_stats("drop", t, P, ok, ke, t_b, 10, masses=[2.0, 1.0], t_release=t_rel)
    assert abs(held["g_ratio"] - 1.0) > 0.05 and free["g_ratio"] == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("s,tau", [(1.0, 1.0), (50.0, 0.4)])
def test_bounce_and_galileo_ramp_epsilon_scale_free(s, tau):
    eps, fps, g = 0.8, 60.0, 9.81
    t = np.arange(120) / fps
    tc = 0.6                                                    # fall from h0 = g tc^2 / 2, rebound eps * g tc
    z = np.where(t < tc, 0.5 * g * tc ** 2 - 0.5 * g * t ** 2, eps * g * tc * (t - tc) - 0.5 * g * (t - tc) ** 2)
    P = np.stack([0.5 * t, np.maximum(z, 0)], 1)[:, None]
    ke = int(np.searchsorted(t, tc))
    b = O.traj_stats("bounce", t * tau, P * s, np.ones((120, 1), bool), ke, tc * tau, 12)
    assert b["epsilon"] == pytest.approx(eps, abs=1e-6) and b["epsilon_height"] == pytest.approx(eps, abs=1e-3)
    # Galileo's ramp: release at h = 1 at x = 0, descend to the pad level 0 at x = 1, flat run x 1..3 (pad at x ~ 1.5),
    # climb to eps^2 h; speed on the run-in 1, after the pad eps
    x = np.r_[np.linspace(0, 1, 20), 1 + np.arange(1, 21) * 0.05, 2 + np.arange(1, 21) * 0.05 * eps, 3 + np.linspace(0, eps ** 2, 16)[1:]]
    zz = np.r_[np.linspace(1, 0, 20), np.zeros(40), np.linspace(0, eps ** 2, 16)[1:]]
    P = np.stack([x, zz], 1)[:, None]
    tt = np.arange(len(x)) / fps
    r = O.traj_stats("ramp", tt * tau, P * s, np.ones((len(x), 1), bool), 30, tt[30] * tau, 8)
    assert r["epsilon"] == pytest.approx(eps, abs=1e-6) and r["epsilon_speed"] == pytest.approx(eps, rel=1e-6)


@pytest.mark.parametrize("s,tau", [(1.0, 1.0), (80.0, 0.5)])
def test_rolling_stats_order_and_slip_scale_free(s, tau):
    k = np.array([0.5, 0.905])
    a = 1.0 / (1 + k)
    t = np.linspace(0, 1, 25)
    S = 0.5 * a * t[:, None] ** 2
    r = np.array([0.3, 0.3])
    ang = S / r                                                  # rolling without slipping: angle = s / r
    st = O.rolling_stats(t * tau, S * s, r * s, k, ang)
    assert st["order"] == pytest.approx(1.0, abs=1e-6)
    assert st["slip_s"] == pytest.approx(1.0, rel=0.1) and st["slip_h"] == pytest.approx(1.0, rel=0.1)
    skid = O.rolling_stats(t * tau, S * s, r * s, k, 0.0 * ang)
    assert skid["slip_s"] == pytest.approx(0.0, abs=1e-9)


def _rolling_clip(slip=1.0, s=1.0, n_master=96, fps=48, release=24):
    """Two fronto-parallel discs (pixel radius 14 s) with a 90-degree marker wedge (dark on the light body, bright on
    the dark one), released from rest at master frame ``release``, sliding down-screen-right with a_i = A / (1 + k_i)
    and spinning at w = slip v / r: an F10-like render for the track-B rotation readout."""
    from types import SimpleNamespace
    H, W, R = int(160 * s), int(320 * s), 14.0 * s
    k = {"A": 0.5, "B": 0.905}
    body = {"A": (200, 60, 60), "B": (40, 40, 160)}
    wedge = {"A": (5, 5, 5), "B": (242, 242, 242)}
    start = {"A": np.array([45.0, 30.0]) * s, "B": np.array([115.0, 30.0]) * s}           # (row, col)
    u = np.array([0.2, 1.0]) / np.hypot(0.2, 1.0)
    t = np.maximum(np.arange(n_master) - release, 0) / fps
    yy, xx = np.mgrid[:H, :W]
    frames = np.full((n_master, H, W, 3), 128, np.uint8)
    masks = np.zeros((n_master, H, W), np.uint8)
    pf = [{} for _ in range(n_master)]
    for sid, o in enumerate("AB", 1):
        S = 0.5 * 250.0 * s / (1 + k[o]) * t ** 2
        th = slip * S / R + 0.3 * sid
        for f in range(n_master):
            c = start[o] + S[f] * u
            dr, dc = yy - c[0], xx - c[1]
            disc = dr ** 2 + dc ** 2 <= R ** 2
            phi = np.mod(np.arctan2(dr, dc) - th[f], 2 * np.pi)
            frames[f][disc] = body[o]
            frames[f][disc & (phi < np.pi / 2)] = wedge[o]
            masks[f][disc] = sid
            pf[f][o] = {"pixel": [c[1], c[0]], "position": [0.0, 0.0, 0.0]}
    objects = {o: {"mass": 1.0, "radius": 0.4, "size": 0.4, "inertia": k[o] * 0.16, "segmentation_id": i}
               for i, o in enumerate("AB", 1)}
    labels = {"physlaw": {"family": "f10", "subfamily": "rolling", "first_post_contact_frame": release + 3,
                          "objects": objects, "constants": {"fps": fps}, "clean_until_s": (n_master - 1) / fps,
                          "camera": {"resolution": [W, H]}},
              "scene_constants": {"frame_rate": fps}, "per_frame": pf}
    return SimpleNamespace(labels=labels, frames=frames, masks=masks, pl=labels["physlaw"])


@pytest.mark.parametrize("slip,s,fps", [(1.0, 1.0, 16.0), (2.0, 1.0, 16.0), (1.0, 2.0, 24.0), (2.0, 0.75, 24.0)])
def test_f10_rotation_readout_on_a_rendered_like_sequence(slip, s, fps):
    """Wedge angle + centre track + mask chord: slip = |w| r / |v| and the rolling order, scale-free (pixel scale s,
    frame rate fps)."""
    c = _rolling_clip(slip, s)
    assert O.READOUT_SELF["f10"] is O.readout_rolling
    r = O.readout_rolling(c.frames[:: 48 // int(fps)], c, fps)
    assert r["measurable"], r
    assert r["slip_s"] == pytest.approx(slip, rel=0.06) and r["slip_h"] == pytest.approx(slip, rel=0.06)
    assert r["order"] == pytest.approx(1.0, abs=0.1)
    assert r["r_px"] == pytest.approx([14 * s, 14 * s], rel=0.06)
    assert set(O.row_stats(r, c.labels)) == {"order", "slip_s", "slip_h"}


def test_world_maps_letterbox():
    pf = [{o: {"pixel": [x, y], "position": [x / 10, y / 10, 0.0]} for o, (x, y) in
           (("A", (10.0 * f, 5.0 + f)), ("B", (3.0 * f + 7, 40.0 - 2 * f)))} for f in range(6)]
    lab = {"scene_constants": {"resolution": [400, 100]}, "per_frame": pf}
    m = O._world_maps(lab, list(range(6)), 100, 100, "letterbox")        # scale 0.25, 37-px bars top and bottom
    p = pf[3]["A"]
    assert np.array([p["pixel"][1] * 0.25 + 37, p["pixel"][0] * 0.25, 1.0]) @ m["A"] == pytest.approx(p["position"][:2])
    with pytest.raises(ValueError):
        O._world_maps(lab, list(range(6)), 100, 100, "crop")


# ---------------------------------------------------------------- F6 / F9 / F11 / F12 readouts: scale-free statistics

@pytest.mark.parametrize("s,tau", [(1.0, 1.0), (60.0, 0.4)])
def test_glide_and_friction_stats(s, tau):
    t = np.arange(60) / 30.0
    ke, v = 30, np.array([1.2, 0.5, 0.0])
    for k, turn, want in ((1.0, 0.0, (1.0, 0.0)), (1.3, 0.0, (1.3, 0.0)), (1.0, 20.0, (1.0, 20 / 180))):
        c, sn = np.cos(np.radians(turn)), np.sin(np.radians(turn))
        v1 = k * np.array([c * v[0] - sn * v[1], sn * v[0] + c * v[1], 0.0])
        X = np.where((t < t[ke])[:, None], v * (t[:, None] - t[ke]), v1 * (t[:, None] - t[ke]))[:, None]
        st = O.glide_stats(t * tau, X * s, np.ones((60, 1), bool), ke, 60, 15)
        assert (st["inertia_speed"], st["inertia_turn"]) == (pytest.approx(want[0]), pytest.approx(want[1], abs=1e-9))
    # slides (F6 closed form): decelerating from frame 0, branch at te, legal stop at 4.4 s; legal (and the heavier
    # x1.3 in a pair), reverse, viscous -> decel_const, stop_hold, mass_ratio from the tracks
    from physlaw.data.families.f6_friction import slide_motion
    tt, te, t_stop, a = np.arange(180) / 30.0, 2.5, 4.4, 0.5
    d = np.array([0.6, 0.8, 0.0])
    win = {k: x * tau for k, x in {"move_end": t_stop - 0.1, "stop_start": t_stop + 0.1, "end": tt[-1]}.items()}
    X = lambda kind, k=1.0: slide_motion(tt, te, a * t_stop, a, kind, k)[0][:, None] * d * s
    ok = np.ones((180, 2), bool)
    st = lambda A, B, w=win: O.friction_stats(tt * tau, np.stack([A, B], 1), ok, te * tau, w, [3.0, 1.0])
    leg = st(X("decel"), X("decel"))
    assert leg["decel_const"] == pytest.approx(1.0, abs=1e-6) and leg["mass_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert leg["stop_hold"] == pytest.approx(0.0, abs=1e-9)
    w13 = {**win, "move_end": (te + (t_stop - te) / 1.3 - 0.1) * tau}       # io.friction_windows: the twin's own stop
    assert st(X("decel", 1.3), X("decel"), w13)["mass_ratio"] == pytest.approx(1.3, rel=1e-6)
    rev = st(X("reverse"), X("reverse"))
    assert rev["decel_const"] == pytest.approx(1.0, abs=1e-6) and rev["stop_hold"] > 0.3
    vs = st(X("viscous"), X("viscous"))
    assert vs["decel_const"] < 0.8 and 0.05 < vs["stop_hold"] < 0.4


@pytest.mark.parametrize("s,tau", [(1.0, 1.0), (125.0, 0.5)])
def test_pendulum_periods_and_swing_epsilon(s, tau):
    from physlaw.data.families import f9_pendulum as P9
    fps, Lp, w0 = 16.0, 1.0, P9.omega0(1.0, 0.12)
    t = np.arange(96) / fps
    te = 3.0
    ke = int(np.searchsorted(t, te))
    piv = [-1.0, 1.0]

    def bobs(fac_heavy):
        X = np.zeros((96, 2, 2))
        for k, f in enumerate((1.0, fac_heavy)):
            th = np.where(t <= te, P9.free_swing(t, te, 0.5, w0)[0], P9.free_swing(t, te, 0.5, w0 / f)[0])
            X[:, k] = np.stack([piv[k] + Lp * np.sin(th), -Lp * np.cos(th)], 1)
        return X
    ok = np.ones((96, 2), bool)
    leg = O.pendulum_stats(t * tau, bobs(1.0) * s, ok, np.array(piv) * s, te * tau, 96, [1.0, 3.0])
    assert leg["period_ratio"] == pytest.approx(1.0, abs=2e-3) and leg["mass_period_ratio"] == pytest.approx(1.0, abs=2e-3)
    hv = O.pendulum_stats(t * tau, bobs(0.8) * s, ok, np.array(piv) * s, te * tau, 96, [1.0, 3.0])
    assert hv["mass_period_ratio"] == pytest.approx(0.8, abs=5e-3)
    # swing: held at A0, released, apex after the damper at eps^2 h0 (heights only; the bottom from the track)
    for eps in (0.7, 1.0, 1.3):
        a0 = 0.5
        a1 = np.arccos(1 - eps ** 2 * (1 - np.cos(a0)))
        tq = P9.time_to_angle(a0, 0.0, w0)                         # release apex -> bottom
        th = np.where(t < 1.0, a0, np.where(t < 1.0 + tq, P9.free_swing(t, 1.0, a0, w0)[0],
                                            P9.free_swing(t, 1.0 + tq + P9.time_to_angle(a1, 0.0, w0), -a1, w0)[0]))
        z = -Lp * np.cos(th)
        ke_ = int(np.searchsorted(t, 1.0 + tq))
        end = int(np.searchsorted(t, 1.0 + tq + 2 * P9.time_to_angle(a1, 0.0, w0)))
        sw = O.swing_stats(t * tau, z * s, np.ones(96, bool), ke_, end)
        assert sw["epsilon"] == pytest.approx(eps, abs=0.02)


def _tip_track(x, beta=0.5, a=0.8, fps=24.0, T=72, te=1.0, tipped=True):
    """Block axis angle and CoM height (metres) of the closed-form F11 tip, the prop's top edge falling at 3 g."""
    from physlaw.data.families import f11_support as P
    t = np.arange(T) / fps
    C, V, PH, W, info = P.tip_motion(np.clip(t - te, 0, None), a, beta * a, x)
    phi = np.where(t > te, -PH, 0.0) if tipped else np.zeros(T)
    zb = np.where(t > te, C[:, 1], beta * a) if tipped else np.full(T, beta * a)
    zp = -0.5 * 3 * 9.81 * np.clip(t - te, 0, None) ** 2
    return t, phi, zb, zp, info


@pytest.mark.parametrize("s,tau", [(1.0, 1.0), (125.0, 1.0)])
def test_support_stats_scale_free(s, tau):
    """Length-scale free (pixels vs metres); the tip and fit windows are in the clip's own seconds (clip time is
    frame index / master rate, never a model's fps), so the time unit is fixed by the clip."""
    for x in (0.75, 1.25, 1.45):
        for tipped in (True, False):
            t, phi, zb, zp, info = _tip_track(x, tipped=tipped)
            legal = tipped == (x > 1)
            st = O.support_stats(t * tau, phi, zb * s, zp * s, 1.0 * tau, t[-1] * tau, 0.8 * s, x, 0.5,
                                 prop_ratio=3.0)
            assert st["tip_consistency"] == float(legal) and st["g_clip"] == pytest.approx(9.81 * s / tau ** 2, rel=1e-6)
            if x > 1:
                assert st["alpha_ratio"] == (pytest.approx(1.0, abs=0.02) if tipped else 0.0)
            else:
                assert np.isnan(st["alpha_ratio"])
    # g from the block's own free fall after the separation (no prop reference)
    t, phi, zb, zp, info = _tip_track(1.45, T=120)
    st = O.support_stats(t * tau, phi, zb * s, zp * s, 1.0 * tau, t[-1] * tau, 0.8 * s, 1.45, 0.5,
                         t_sep=(1.0 + info["t_sep"]) * tau)
    assert st["g_clip"] == pytest.approx(9.81 * s / tau ** 2, rel=1e-6) and st["alpha_ratio"] == pytest.approx(1.0, abs=0.02)
    assert np.isnan(O.support_stats(t, phi, zb, zp, 1.0, 1.5, 0.8, 1.45, 0.5)["alpha_ratio"])      # no in-clip fall


def _top_camera(Hc, f, W, H):
    """Top-down pinhole camera at height Hc (world x right, world y up the image)."""
    return {"K": [[f, 0.0, W / 2], [0.0, f, H / 2], [0.0, 0.0, 1.0]], "resolution": [W, H],
            "world_to_camera": [[1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, -1.0, Hc], [0, 0, 0, 1.0]]}


def _spin_clip(k_spin=1.0, n_master=144, fps=48, W=320, H=200, f=500.0, Hc=10.0):
    """F12-like top-down render: a red puck hits a light bar (dark marker wedge on its local x > 0 half) at rest,
    closed-form frictionless impulse (e = 0.9) and the bar's spin x k_spin afterwards."""
    from types import SimpleNamespace
    from physlaw.eval import laws
    r, l, w, mA, mB = 0.25, 1.0, 0.15, 1.0, 1.5
    I = np.array([0.5 * mA * r * r, mB * (l * l + w * w) / 3])
    b = 0.5
    ctx = {"m": np.array([mA, mB]), "I": I, "n": np.array([1.0, 0.0]), "c": np.array([-w, b])}
    pre = {"p": np.array([[-w - r, b, 0.0], [0.0, 0.0, 0.0]]), "v": np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
           "w": np.zeros((2, 3))}
    post = laws.rigid_impulse(pre, ctx, 0.9)
    post["w"] = post["w"] * k_spin
    te = 1.52
    t = np.arange(n_master) / fps
    yy, xx = np.mgrid[:H, :W]
    frames = np.full((n_master, H, W, 3), 128, np.uint8)
    masks = np.zeros((n_master, H, W), np.uint8)
    pf = []
    cam = _top_camera(Hc, f, W, H)
    to_px = lambda p: (H / 2 - f * p[1] / Hc, W / 2 + f * p[0] / Hc)          # (row, col)
    for i, ti in enumerate(t):
        st, d = (pre, ti - te) if ti < te else (post, ti - te)
        pA = st["p"][0, :2] + st["v"][0, :2] * d
        pB = st["p"][1, :2] + st["v"][1, :2] * d
        psi = np.pi / 2 + st["w"][1, 2] * d
        e = {}
        for o, pc in (("A", pA), ("B", pB)):
            e[o] = {"position": [pc[0], pc[1], 0.0], "pixel": list(to_px(pc))[::-1]}
        pf.append(e)
        X, Y = (xx - W / 2) * Hc / f, (H / 2 - yy) * Hc / f                      # world coordinates of each pixel
        lx = (X - pB[0]) * np.cos(psi) + (Y - pB[1]) * np.sin(psi)
        ly = -(X - pB[0]) * np.sin(psi) + (Y - pB[1]) * np.cos(psi)
        bar = (np.abs(lx) <= l) & (np.abs(ly) <= w)
        frames[i][bar] = (217, 166, 26)
        frames[i][bar & (lx > 0)] = (5, 5, 5)
        masks[i][bar] = 2
        puck = (X - pA[0]) ** 2 + (Y - pA[1]) ** 2 <= r * r
        frames[i][puck] = (200, 30, 30)
        masks[i][puck] = 1
    objects = {"A": {"mass": mA, "inertia": I[0], "segmentation_id": 1}, "B": {"mass": mB, "inertia": I[1], "segmentation_id": 2}}
    labels = {"physlaw": {"family": "f12", "subfamily": "spin", "first_post_contact_frame": int(np.floor(te * fps)) + 1,
                          "event_time_s": te, "objects": objects, "constants": {"fps": fps},
                          "clean_until_s": 2.6, "camera": cam},
              "scene_constants": {"frame_rate": fps}, "per_frame": pf}
    L0 = laws.angular_momentum_z(pre, ctx)
    want = 1.0 + (k_spin - 1.0) * I[1] * post["w"][1, 2] / k_spin / L0
    return SimpleNamespace(labels=labels, frames=frames, masks=masks, pl=labels["physlaw"]), want


@pytest.mark.parametrize("k_spin,fps", [(1.0, 24.0), (1.3, 24.0), (1.0, 16.0)])
def test_f12_spin_readout_on_a_rendered_like_sequence(k_spin, fps):
    c, want = _spin_clip(k_spin)
    assert O.READOUT_SELF["f12"] is O.readout_spin
    r = O.readout_spin(c.frames[:: 48 // int(fps)], c, fps)
    assert r["measurable"], r
    assert r["L_ratio"] == pytest.approx(want, abs=0.04)
    assert set(O.row_stats(r, c.labels)) == {"L_ratio"}


def test_new_families_registered():
    assert {f: O.READOUT_SELF[f] for f in ("f6", "f9", "f11", "f12")} == {
        "f6": O.readout_friction, "f9": O.readout_pendulum, "f11": O.readout_support, "f12": O.readout_spin}
    from physlaw.config import load_config
    assert {"f6", "f9", "f11", "f12"} <= set(load_config("configs/eval/outputs.yaml").families)


def _support_clip(x=1.25, tipped=True, n_master=96, fps=48, W=320, H=200, f=400.0, dist=10.0):
    """F11-like side-view render: a red block (half-length a, beta 0.5) on a shelf edge at world (-1, 1.5), the
    closed-form tip (or none), a dark prop under the overhang falling at 3 g from the release."""
    from types import SimpleNamespace
    from physlaw.data.families import f11_support as P
    a, beta, te = 0.8, 0.5, 1.0
    b = beta * a
    O_ = np.array([-1.0, 1.5])
    t = np.arange(n_master) / fps
    tau = np.clip(t - te, 0, None)
    C, V, PH, Wv, info = P.tip_motion(tau, a, b, x)
    c0 = O_ + np.array([(x - 1) * a, b])
    cam = {"K": [[f, 0.0, W / 2], [0.0, f, H / 2], [0.0, 0.0, 1.0]], "resolution": [W, H],
           "world_to_camera": [[1.0, 0, 0, 0], [0, 0, -1.0, 0.0], [0, 1.0, 0, dist], [0, 0, 0, 1.0]]}
    yy, xx = np.mgrid[:H, :W]
    X, Z = (xx - W / 2) / f * dist, -(yy - H / 2) / f * dist
    frames = np.full((n_master, H, W, 3), 128, np.uint8)
    masks = np.zeros((n_master, H, W), np.uint8)
    pf = []
    x_p = O_[0] + 0.6 * x * a
    for i in range(n_master):
        cc, ph = (O_ + C[i], PH[i]) if tipped and t[i] > te else (c0, 0.0)
        frames[i][(X < O_[0]) & (Z <= O_[1]) & (Z > O_[1] - 0.12)] = (158, 158, 158)        # the shelf (static)
        top = O_[1] - 1.5 * 9.81 * tau[i] ** 2
        prop = (np.abs(X - x_p) <= 0.05) & (Z <= top)
        frames[i][prop], masks[i][prop] = (56, 56, 56), 2
        lx = (X - cc[0]) * np.cos(ph) - (Z - cc[1]) * np.sin(ph)
        lz = (X - cc[0]) * np.sin(ph) + (Z - cc[1]) * np.cos(ph)
        blk = (np.abs(lx) <= a) & (np.abs(lz) <= b)
        frames[i][blk], masks[i][blk] = (200, 30, 30), 1
        pf.append({"A": {"position": [cc[0], 0.0, cc[1]], "pixel": [W / 2 + f * cc[0] / dist, H / 2 - f * cc[1] / dist]},
                   "P": {"position": [x_p, 0.0, top], "pixel": [W / 2 + f * x_p / dist, H / 2 - f * top / dist]}})
    objects = {"A": {"mass": 1.0, "segmentation_id": 1}, "P": {"mass": 1.0, "segmentation_id": 2, "aux": True}}
    labels = {"physlaw": {"family": "f11", "subfamily": "support", "first_post_contact_frame": int(te * fps) + 1,
                          "event_time_s": te, "objects": objects, "camera": cam, "clean_until_s": te + 0.7,
                          "constants": {"overhang_x": x, "aspect_ba": beta, "half_length": a, "prop_acc_over_g": 3.0},
                          "edit": {"t_separate_s": te + info["t_sep"]} if tipped else {}},
              "scene_constants": {"frame_rate": fps}, "per_frame": pf}
    return SimpleNamespace(labels=labels, frames=frames, masks=masks, pl=labels["physlaw"])


@pytest.mark.parametrize("x,tipped,hw", [(1.25, True, (200, 320)), (1.25, False, (200, 320)), (0.75, True, (120, 320)),
                                         (1.45, True, (240, 240))])
def test_f11_support_readout_on_a_rendered_like_sequence(x, tipped, hw):
    """Block angle from the principal axis, half-length and heights in master pixels (isotropic after a squash),
    g from the prop's 3 g descent: tipped iff legal at x > 1, alpha_ratio ~ 1 for the legal tip, 0 for a stay."""
    from physlaw.models.base import fit_frames
    c = _support_clip(x, tipped)
    r = O.readout_support(fit_frames(c.frames[::2], hw, "squash"), c, 24.0)
    assert r["measurable"], r
    assert r["tipped"] == tipped and r["tip_consistency"] == float(tipped == (x > 1))
    assert r["g_clip"] == pytest.approx(9.81 * 400 / 10, rel=0.05) and r["half_length_px"] == pytest.approx(32, rel=0.06)
    if x > 1:
        assert r["alpha_ratio"] == (pytest.approx(1.0, abs=0.1) if tipped else 0.0)
    else:
        assert np.isnan(r["alpha_ratio"])


def _disc_clip(family, sub, tracks, te, cam, objects, n_master, fps=48, extra=None, H=200, W=320):
    """Discs (radius in px, colour) drawn at world tracks [n_master, 3] through ``cam``; minimal labels for the
    F6 / F9 readouts."""
    from types import SimpleNamespace
    K, E = np.asarray(cam["K"]), np.asarray(cam["world_to_camera"])
    yy, xx = np.mgrid[:H, :W]
    frames = np.full((n_master, H, W, 3), 128, np.uint8)
    masks = np.zeros((n_master, H, W), np.uint8)
    pf = []
    for i in range(n_master):
        e = {}
        for sid, (o, (tr, rpx, col)) in enumerate(tracks.items(), 1):
            pc = E[:3, :3] @ tr[i] + E[:3, 3]
            u, v = (K @ (pc / pc[2]))[:2]
            m = (xx - u) ** 2 + (yy - v) ** 2 <= rpx ** 2
            frames[i][m], masks[i][m] = col, sid
            e[o] = {"position": list(tr[i]), "pixel": [u, v]}
        pf.append(e)
    objs = {o: {**objects[o], "segmentation_id": sid} for sid, o in enumerate(tracks, 1)}
    labels = {"physlaw": {"family": family, "subfamily": sub, "first_post_contact_frame": int(np.floor(te * fps)) + 1,
                          "event_time_s": te, "objects": objs, "camera": cam, **(extra or {})},
              "scene_constants": {"frame_rate": fps}, "per_frame": pf}
    return SimpleNamespace(labels=labels, frames=frames, masks=masks, pl=labels["physlaw"])


@pytest.mark.parametrize("k", [1.0, 1.3])
def test_f6_glide_readout_on_a_rendered_like_sequence(k):
    t = np.arange(120) / 48.0
    te = 1.2
    v = np.array([1.2, 0.4, 0.0])
    p = np.where((t < te)[:, None], v * (t - te)[:, None], k * v * (t - te)[:, None])
    cam = _top_camera(10.0, 500.0, 320, 200)
    c = _disc_clip("f6", "glide", {"A": (p, 12.0, (200, 30, 30))}, te, cam, {"A": {"mass": 1.0}}, 120,
                   extra={"clean_until_s": 2.4})
    r = O.READOUT_SELF["f6"](c.frames[::2], c, 24.0)
    assert r["measurable"], r
    assert r["inertia_speed"] == pytest.approx(k, rel=0.02) and r["inertia_turn"] == pytest.approx(0.0, abs=0.01)


@pytest.mark.parametrize("fac", [1.0, 0.8])
def test_f9_pair_readout_on_a_rendered_like_sequence(fac):
    from physlaw.data.families import f9_pendulum as P9
    n, fps, Lp, te = 240, 48, 1.0, 2.5
    t = np.arange(n) / fps
    w0 = P9.omega0(Lp, 0.12)
    piv = {"A": np.array([-1.2, 0.0, 1.5]), "B": np.array([1.2, 0.0, 1.5])}
    tracks = {}
    for o, f, col in (("A", 1.0, (200, 30, 30)), ("B", fac, (40, 40, 200))):
        th = np.where(t <= te, P9.free_swing(t, te, 0.5, w0)[0], P9.free_swing(t, te, 0.5, w0 / f)[0])
        tracks[o] = (piv[o] + Lp * np.stack([np.sin(th), 0 * th, -np.cos(th)], 1), 8.0, col)
    cam = {"K": [[400.0, 0, 160], [0, 400.0, 100], [0, 0, 1]], "resolution": [320, 200],
           "world_to_camera": [[1.0, 0, 0, 0], [0, 0, -1.0, 0.0], [0, 1.0, 0, 10.0], [0, 0, 0, 1.0]]}
    c = _disc_clip("f9", "pair", tracks, te, cam, {"A": {"mass": 1.0}, "B": {"mass": 3.0}}, n,
                   extra={"constants": {"pivots": {o: list(piv[o]) for o in piv}}})
    r = O.READOUT_SELF["f9"](c.frames[::3], c, 16.0)
    assert r["measurable"], r
    assert r["mass_period_ratio"] == pytest.approx(fac, abs=0.03)
    assert r["period_ratio"] == pytest.approx(np.sqrt(fac), abs=0.03)
