"""L3 (physlaw.eval.l3_invariance): Galilean invariance of eps / theta_cm from DECODED velocities,
and the mass-independence ratios through laws.gravity / friction / oscillation (synthetic inputs and closed-form F9
labels; no model, no data)."""
import numpy as np
import pytest

from physlaw.eval import l3_invariance as L3

ALPHAS = (1e-2, 1.0, 1e2)


def _collision(rng, e, u=(0.0, 0.0)):
    ma, mb = rng.uniform(0.5, 3, 2)
    th = rng.uniform(-0.6, 0.6)
    n = np.array([np.cos(th), np.sin(th)])
    va, vb = np.array([rng.uniform(1, 2), rng.uniform(-0.3, 0.3)]), np.array([0.0, rng.uniform(-0.3, 0.3)])
    J = (1 + e) * ma * mb / (ma + mb) * float((va - vb) @ n)
    pa, pb = va - J * n / ma, vb + J * n / mb
    return ma, mb, n, [x + np.asarray(u) for x in (va, vb, pa, pb)]


def _f4(distort=0.0, n_scenes=30, D=10, seed=0, cam=False):
    """F4-like fake cache: scenes x rungs x boost levels; features linear in each ball's velocity. ``distort`` adds a
    boost-dependent term to A's post-event features only (a representation without Galilean structure). ``cam``: a
    camera-boost level cam_u1 whose (camera-frame) velocities equal u1's."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(2, D))
    levels = {"u0": (0.0, 0.0), "u1": (1.0, 0.5), "u2": (2.0, -1.0), **({"cam_u1": (1.0, 0.5)} if cam else {})}
    recs, labels, A = [], [], []
    for s in range(n_scenes):
        st = np.random.default_rng(1000 + s).bit_generator.state
        for e in (0.5, 1.0):
            for lv, u in levels.items():
                r = np.random.default_rng(); r.bit_generator.state = st
                ma, mb, n, (va, vb, pa, pb) = _collision(r, e, u)
                labels.append({"physlaw": {"objects": {"A": {"mass": ma}, "B": {"mass": mb}}, "contact_normal": n.tolist(),
                                           "pre_velocity": {"A": va.tolist(), "B": vb.tolist()},
                                           "post_velocity": {"A": pa.tolist(), "B": pb.tolist()},
                                           "twin_group_id": f"s{s}", "edit": {"boost_level": lv, "e": e}}})
                recs.append({"group": f"s{s}", "twin": f"{lv}_e{e}", "clip": f"s{s}/{lv}_e{e}"})
                f = lambda v: v @ W + 0.02 * rng.normal(size=D)
                ua = distort * np.linalg.norm(u)
                A.append([f(va), f(pa + ua), f(vb), f(pb), f(np.r_[va + vb]), f(pa + pb)])
    pools = ["roi_A.pre", "roi_A.post", "roi_B.pre", "roi_B.post", "frame.pre", "frame.post"]
    return recs, labels, np.array(A, np.float32)[:, :, None], pools, ["0"]


CFG = {"probe": {"alphas": list(ALPHAS), "k_folds": 3}, "fold_seeds": 1, "bootstrap": {"n": 50},
       "galilean": {"boost_key": "physlaw.edit.boost_level", "base_level": "u0", "cv_key": "physlaw.twin_group_id",
                    "pair_keys": ["physlaw.twin_group_id", "physlaw.edit.e"], "pools": ["roi_own"], "layers": "all",
                    "max_floor_eps_err": 0.25}}


def test_galilean_from_decoded_velocities_invariant_for_a_galilean_representation():
    res = L3.galilean(*_f4(), CFG, log=lambda *a: None)
    assert all(f["measurable"] for f in res["floor"].values())
    for lv in ("u1", "u2"):
        d = res["galilean"]["decoded"][lv]
        assert d["n_paired"] == 60 and d["d_epsilon"] < 0.05
        assert res["galilean"]["true"][lv]["d_epsilon"] == pytest.approx(0.0, abs=1e-9)
    assert res["stats"]["epsilon"]["by_level"]["u0"]["true_mean"] == pytest.approx(0.75)


def test_galilean_camera_boost_vs_ball_boost():
    """cam_<level> is a level of its own against u0 and is paired with its ball-boost twin (same scene, rung, u):
    identical camera-frame motion -> decoded difference ~ 0, true difference exactly 0."""
    res = L3.galilean(*_f4(cam=True), CFG, log=lambda *a: None)
    assert res["galilean"]["decoded"]["cam_u1"]["n_paired"] == 60
    cb = res["camera_vs_ball"]["cam_u1"]
    assert set(cb) == {"epsilon", "theta_cm", "eta", "antiparallel"}
    assert cb["epsilon"]["n_paired"] == 60 and abs(cb["epsilon"]["diff"]) < 0.05
    assert cb["epsilon"]["true_diff"] == pytest.approx(0.0, abs=1e-12)
    assert "camera_vs_ball" not in L3.galilean(*_f4(), CFG, log=lambda *a: None)


def test_galilean_detects_a_boost_dependent_representation():
    good = L3.galilean(*_f4(), CFG, log=lambda *a: None)["galilean"]["decoded"]["u2"]["d_epsilon"]
    bad = L3.galilean(*_f4(distort=0.4), CFG, log=lambda *a: None)["galilean"]["decoded"]["u2"]["d_epsilon"]
    assert bad > 3 * good and bad > 0.1


def test_paired_by_level_and_mass_ratios():
    level = np.array(["u0", "u1", "u0", "u1"])
    pair = np.array(["a", "a", "b", "b"])
    d = L3.paired_by_level(np.array([1.0, 1.2, 0.5, 0.6]), level, pair, np.array(["a", "a", "b", "b"]), "u0", n_boot=0)
    assert d["u1"]["diff"] == pytest.approx(0.15) and d["u1"]["abs_diff"] == pytest.approx(0.15)
    a = {"A": np.array([9.8, 9.8, 2.0]), "B": np.array([9.8, 12.74, -1.0])}
    m = {"A": np.array([1.0, 1.0, 1.0]), "B": np.array([2.0, 2.0, 2.0])}
    r = L3.mass_ratios(a, m, np.arange(3))
    assert r[0] == pytest.approx(1.0) and r[1] == pytest.approx(1.3) and np.isnan(r[2])


def test_mass_ratios_friction_and_oscillation_laws():
    m = {"A": np.array([1.0, 1.0]), "B": np.array([3.0, 3.0])}
    d = {"A": np.array([2.0, 2.0]), "B": np.array([2.0, 2.6])}             # slides: decelerations along the motion
    assert np.allclose(L3.mass_ratios(d, m, np.arange(2), "friction"), [1.0, 1.3])
    T = {"A": np.array([1.8, 1.8]), "B": np.array([1.8, 1.44])}             # pendulum pairs: periods after the branch
    assert np.allclose(L3.mass_ratios(T, m, np.arange(2), "oscillation"), [1.0, 0.8])
    with pytest.raises(KeyError):
        L3.mass_ratios(T, m, np.arange(2), "statics")


def test_f9_period_target_from_labels():
    from test_l0_momentum import _family_labels
    for lab in _family_labels("f9", "pair", 2):
        pl = lab["physlaw"]
        rec = pl["edit"]
        for o in "AB":
            assert L3.object_target(lab, o, "period") == pytest.approx(rec["period_post_s"][o], rel=1e-6)
        heavy, light = rec["heavier"], "B" if rec["heavier"] == "A" else "A"
        T = {o: np.array([L3.object_target(lab, o, "period")]) for o in (heavy, light)}
        M = {o: np.array([pl["objects"][o]["mass"]]) for o in (heavy, light)}
        want = 0.8 if pl["twin"] == "heavy_x0.8" else 1.0
        assert L3.mass_ratios(T, M, np.arange(1), "oscillation")[0] == pytest.approx(want, rel=1e-6)


def test_mass_independence_reads_both_objects_with_one_calibrated_instrument():
    """F7-drop-like rows: legal twins fall alike, the violator's heavy ball 1.3x faster; each object's ROI pool is linear
    in its own acceleration (the same code in both pools, as ROI pools of one layer) plus noise, under heavy ridge. The shared R1
    instrument (one inverse-calibrated map for both objects) reads heavy / light ~ 1 on legal clips and ~ 1.3 on the
    violator; uncalibrated per-object ridge maps shrink both terms toward the mean and the violator ratio toward 1."""
    rng = np.random.default_rng(0)
    G, D = 60, 12
    w = rng.normal(size=D)
    groups = np.repeat(np.arange(G), 2)
    viol = np.tile([False, True], G)
    g = np.repeat(rng.uniform(5, 15, G), 2)                        # each scene's own g (pixels / s^2)
    Y = {"A": g * np.where(viol, 1.3, 1.0), "B": g}                  # A is the heavier ball
    M = {"A": np.full(2 * G, 2.0), "B": np.full(2 * G, 1.0)}
    real = {"roi_A.post": np.outer(Y["A"], w) + 2.0 * rng.normal(size=(2 * G, D)),
            "roi_B.post": np.outer(Y["B"], w) + 2.0 * rng.normal(size=(2 * G, D))}
    pools = {"A": "roi_A.post", "B": "roi_B.post"}
    dec_real, _ = L3.decode_objects(real, real, pools, Y, groups, 5, 2, (1e3, 1e4))[False]
    med = lambda rows: float(np.exp(np.nanmedian(np.log(L3.mass_ratios(dec_real, M, np.where(rows)[0])))))
    assert med(~viol) == pytest.approx(1.0, abs=0.05)
    assert med(viol) == pytest.approx(1.3, abs=0.06)


def _f4_arrays(res_args):
    """(Vpre, Vpost, ctxs, level, pair, pgroup) of an _f4 fake, as galilean() builds them."""
    from physlaw.eval.l0_states import event_inputs
    recs, labels = res_args[0], res_args[1]
    ev = [event_inputs(l) for l in labels]
    level = np.array([l["physlaw"]["edit"]["boost_level"] for l in labels])
    pair = np.array([f"{l['physlaw']['twin_group_id']}/{l['physlaw']['edit']['e']}" for l in labels])
    pgroup = np.array([r["group"] for r in recs])
    return (np.stack([e[0] for e in ev]), np.stack([e[1] for e in ev]), [e[2] for e in ev], level, pair, pgroup)


def _spaces(real_args, fc_args):
    """One forecast space: real pools of one fake cache, forecast pools of another (same scenes and noise)."""
    pools = real_args[3]
    return {"final": ({p: real_args[2][:, i, 0].astype(np.float64) for i, p in enumerate(pools)},
                      {p: fc_args[2][:, i, 0].astype(np.float64) for i, p in enumerate(pools)})}


def test_galilean_forecast_vs_real_video_floor():
    """Audit H2: the law test reads the model's FORECAST of boosted vs unboosted collisions with the same instrument as
    the real-video floor. A forecast whose post-event representation depends on the boost (A's post features shifted
    by distort |u|) -> d_eps(forecast) far above the floor and verdict "not Galilean invariant"; an invariant forecast
    (here the real post-event features themselves) -> d(forecast) = d(floor), "invariant within the real-video floor";
    camera boosts are compared with their ball-boost twin too."""
    real = _f4(cam=True)
    V = _f4_arrays(real)
    run = lambda fc: L3.galilean_forecast_spaces(_spaces(real, fc), *V, "u0", ["roi_own"], 3, 1, ALPHAS, 200, 0.25)
    good = run(real)
    assert good["selected_space"] == "final" and set(good["verdict"]) == {"u1", "u2", "cam_u1", "cam_u1_vs_u1"}
    for lv, r in good["spaces"]["final"]["levels"].items():
        assert r["verdict"] == "invariant within the real-video floor", lv
        assert r["d_epsilon"]["diff"] == pytest.approx(0.0, abs=1e-12) and r["d_epsilon"]["floor"] < 0.05
    bad = run(_f4(distort=0.4, cam=True))["spaces"]["final"]["levels"]
    for lv in ("u1", "u2", "cam_u1"):
        assert bad[lv]["verdict"].startswith("not Galilean invariant"), lv
        assert bad[lv]["d_epsilon"]["forecast"] > 3 * bad[lv]["d_epsilon"]["floor"] and bad[lv]["d_epsilon"]["diff_ci95"][0] > 0
        assert abs(bad[lv]["epsilon_shift"]["forecast"]) > 0.05 and abs(bad[lv]["epsilon_shift"]["floor"]) < 0.02
    # cam_u1 and u1 get the same distortion (same |u|): the camera-vs-ball forecast pair stays at the floor
    assert bad["cam_u1_vs_u1"]["d_epsilon"]["forecast"] == pytest.approx(bad["cam_u1_vs_u1"]["d_epsilon"]["floor"], abs=0.02)
    unmeas = L3.galilean_forecast_spaces(_spaces(real, real), *V, "u0", ["roi_own"], 3, 1, ALPHAS, 50, 1e-6)
    assert all(r["verdict"].startswith("unmeasurable") for r in unmeas["spaces"]["final"]["levels"].values())


def test_mass_independence_score_must_beat_the_mask_only_score():
    """Audit M0: S(forecast) − S(mask-only) with a group bootstrap. Forecast lawful (ratio 1), floor 1, violator 1.3:
    a mask-only instrument that cannot tell the twins apart (ratio ~1.3, S_mask ~ 0) -> difference ~ 1, CI > 0; a mask
    that already moves lawfully (ratio 1: the ROI pool alone gives the answer) -> difference ~ 0, CI holds 0."""
    rng = np.random.default_rng(0)
    G = 40
    g = np.array([f"g{i}" for i in range(G)])
    noisy = lambda c: c * np.exp(0.03 * rng.normal(size=G))
    base = {"forecast": (noisy(1.0), g), "floor": (noisy(1.0), g), "violator": (noisy(1.3), g)}
    blind = L3.score_vs_mask({**base, "mask": (noisy(1.3), g)}, 300)
    assert blind["diff"] > 0.7 and blind["diff_ci95"][0] > 0
    leak = L3.score_vs_mask({**base, "mask": (noisy(1.0), g)}, 300)
    assert abs(leak["diff"]) < 0.3 and leak["diff_ci95"][0] < 0 < leak["diff_ci95"][1]
