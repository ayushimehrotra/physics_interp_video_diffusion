"""L4 manifold steering (physlaw.eval.l4_causal): manifolds and the precondition, shifts and controls, scoring against
each law's prediction, and the routine end to end on fake caches (no GPU)."""
import json

import numpy as np
import pytest

from physlaw.eval import l4_causal as L4


def _ladder(n_scenes=12, D=20, shared=True, seed=0, off_sd=3.0, cubic=False):
    """F2-like activations: per scene an offset + f(eps) along a (shared or scene-specific) direction + the
    aftermath speed along another; rungs eps 0.0 ... 1.5."""
    rng = np.random.default_rng(seed)
    eps = np.round(np.arange(0, 1.51, 0.25), 2)
    d_e, d_s = np.eye(D)[0], np.eye(D)[1]
    X, E, S, G = [], [], [], []
    for g in range(n_scenes):
        off, sp0 = rng.normal(size=D) * off_sd, rng.uniform(1, 5)
        de = d_e if shared else (lambda r: r / np.linalg.norm(r))(rng.normal(size=D))
        for e in eps:
            sp = sp0 * (1 + e) / 2 + rng.normal(0, 0.05)
            X.append(off + 2 * (e ** 3 if cubic else e) * de + sp * d_s + 0.05 * rng.normal(size=D))
            E.append(e); S.append(sp); G.append(g)
    return np.array(X), np.array(E), np.array(S), np.array(G)


def test_population_curve_passes_only_a_shared_curve():
    X, E, S, G = _ladder(shared=True)
    p = L4.population_curve(X, E, G, draws=50)
    assert p["n_scenes"] == 12 and len(p["theta_grid"]) == 7
    assert p["retention"] > 0.7 and p["p_retention"] < 0.05 and p["p_loso_line"] < 0.05 and p["loso_r2"]["line"] > 0.5
    X2, E2, _, G2 = _ladder(shared=False)
    q = L4.population_curve(X2, E2, G2, draws=50)
    assert q["retention"] < p["retention"] - 0.2 and q["loso_r2"]["line"] < 0.2


def test_across_curve_residualises_the_nuisance():
    rng = np.random.default_rng(0)
    n, D = 120, 15
    q, r = rng.normal(size=n), rng.normal(size=n)
    X = np.outer(q, np.eye(D)[0]) + np.outer(3 * r, np.eye(D)[1]) + 0.3 * rng.normal(size=(n, D))
    sc = np.arange(n)
    p = L4.across_curve(X, q, sc, r[:, None], draws=50)
    assert p["p_loso_line"] < 0.05 and p["loso_r2"]["line"] > 0.02
    man = L4.fit_manifold(X, q, sc, "across", r[:, None])
    u = L4.shift(man, 1.0)
    assert abs(u[0]) > 0.9 and abs(u[1]) < 0.1                    # mass beyond size: not the radius axis
    null = L4.across_curve(X, rng.normal(size=n), sc, r[:, None], draws=50)
    assert null["p_loso_line"] > 0.05


def test_manifold_line_or_cubic_and_norm_matched_controls():
    X, E, S, G = _ladder()
    man = L4.fit_manifold(X, E, G, "ladder")
    assert man["deg"] == 1 and L4.shift(man, 0.5)[0] == pytest.approx(1.0, abs=0.05)   # 2 per unit eps
    Xc, Ec, _, Gc = _ladder(cubic=True)
    assert L4.fit_manifold(Xc, Ec, Gc, "ladder")["deg"] == 3
    rung = (E[:, None] == np.unique(E)[None, 1:]).astype(float)
    speed = L4.fit_manifold(X, S, G, "across", rung)               # the aftermath-speed direction at fixed eps
    assert abs(L4.shift(speed, 1.0)[1]) > 0.8
    plan = L4.plan_shifts(man, [-1.0, 0.5, 2.0], {"speed": speed, "random_dir": None})
    for d, v in plan["law"].items():
        for c in ("speed", "random_dir"):
            assert np.linalg.norm(plan[c][d]) == pytest.approx(np.linalg.norm(v))
    assert plan["random_dir"][-1.0] @ plan["random_dir"][2.0] < 0   # the sign follows dtheta


def _sat_curves(G=10, sat=1.0, slope_speed=0.08, seed=0):
    rng = np.random.default_rng(seed)
    a = np.array([0, 1, 2, 4, 8, 16], float)
    base = rng.uniform(0.4, 0.7, (G, 1))
    noise = lambda: 0.01 * rng.normal(size=(G, len(a)))
    return a, {"law": sat - (sat - base) * np.exp(-a / 2) + noise(), "speed": base + slope_speed * a + noise(),
               "passive": base + 0.002 * a + noise()}


def test_energy_saturation_vs_control_and_passive():
    lc = {"predict": "saturation", "n_top": 2, "sat_tol": 0.1, "control": "speed"}
    a, c = _sat_curves()
    s = L4.score_response(c, a, list(range(10)), lc, n_boot=200)
    assert s["law"]["saturation"] == pytest.approx(1.0, abs=0.02) and s["speed"]["saturation"] > 1.3
    assert s["law"]["minus_speed"]["ci95"][1] < 0 and s["verdict"]["law_like"]
    a, c = _sat_curves(sat=1.5)                                      # energy creation: saturates above 1
    assert not L4.score_response(c, a, list(range(10)), lc, n_boot=200)["verdict"]["law_like"]
    a, c = _sat_curves()
    c["speed"] = c["law"] + 0.02 * np.random.default_rng(5).normal(size=c["law"].shape)   # same plateau
    assert not L4.score_response(c, a, list(range(10)), lc, n_boot=200)["verdict"]["differs_from_control"]


@pytest.mark.parametrize("predict,law_slope", [(1.0, 1.0), (0.0, 0.0), ("rolling", -1 / 1.5)])
def test_slope_scores_against_each_laws_prediction(predict, law_slope):
    rng = np.random.default_rng(0)
    d = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    lc = {"predict": predict, "kappa_ref": 0.5, "tol": 0.1}
    assert L4.law_value(lc) == pytest.approx(law_slope)
    G = 12
    off = rng.normal(size=(G, 1))
    curves = {"law": off + law_slope * d + 0.02 * rng.normal(size=(G, 5)), "random_dir": off + 0.02 * rng.normal(size=(G, 5)),
              "passive": off + 0.1 * d + 0.02 * rng.normal(size=(G, 5))}
    s = L4.score_response(curves, d, list(range(G)), lc, n_boot=200)
    assert s["law"]["slope"] == pytest.approx(law_slope, abs=0.05) and s["verdict"]["matches_law"]
    wrong = {**curves, "law": off + (law_slope + 0.5) * d}
    assert not L4.score_response(wrong, d, list(range(G)), lc, n_boot=200)["verdict"]["matches_law"]


def test_log_deflection_ratio_is_the_mass_ratio_for_any_e():
    vp = np.array([[1.0, 0.2], [0.0, -0.1]])
    n = np.array([1.0, 0.0])
    for e in (0.0, 0.5, 1.0):
        ma, mb = 2.0, 0.5
        J = (1 + e) * ma * mb / (ma + mb) * (vp[0] - vp[1]) @ n
        vq = np.stack([vp[0] - J * n / ma, vp[1] + J * n / mb])
        assert L4.log_deflection_ratio(vp, vq, {"n": n}) == pytest.approx(np.log(ma / mb))
    assert np.isnan(L4.log_deflection_ratio(vp, vp, {"n": n}))


# ---------------------------------------------------------------- end to end (fake caches, no GPU)

def _post(q, e=1.0):
    """Head-on, B at rest, v_A = (1, 0), m_A / m_B = q: post velocities."""
    ua = 1.0 - (1 + e) / (1 + q)
    ub = (1 + e) * q / (1 + q)
    return np.array([[ua, 0.0], [ub, 0.0]])


def test_momentum_routine_end_to_end_encoder(tmp_path, monkeypatch):
    from physlaw.config import load_config
    rng = np.random.default_rng(0)
    D, n_g = 8, 24
    M = rng.normal(size=(4, D))
    recs, labels, A, real_pre, real_post = [], [], [], [], []
    vpre = np.array([[1.0, 0.0], [0.0, 0.0]])
    feat = lambda v: v.reshape(-1) @ M
    qs = {}
    for gi in range(n_g):
        g = f"g{gi}"
        q, rr = float(np.exp(rng.uniform(-1, 1))), float(np.exp(rng.uniform(-0.3, 0.3)))
        qs[g] = q
        for t, e in (("conserve_e1", 1.0), ("conserve_e015", 0.15)):
            vq = _post(q, e)
            recs.append({"group": g, "twin": t, "clip": f"{g}/{t}", "file": f"{g}/{t}.npz"})
            labels.append({"physlaw": {"objects": {"A": {"mass": q, "radius": 0.4 * rr}, "B": {"mass": 1.0, "radius": 0.4}},
                                       "contact_normal": [1.0, 0.0], "pre_velocity": {"A": vpre[0].tolist(), "B": vpre[1].tolist()},
                                       "post_velocity": {"A": vq[0].tolist(), "B": vq[1].tolist()}}})
            A.append([np.log(q) * np.eye(D)[0] + np.log(rr) * np.eye(D)[1] + 0.05 * rng.normal(size=D)])
            real_pre.append(feat(vpre) + 0.01 * rng.normal(size=D))
            real_post.append(feat(vq) + 0.01 * rng.normal(size=D))
    A = np.array(A)[:, :, None].repeat(2, 2).astype(np.float32)               # [n, P=1, L=2, D]
    monkeypatch.setattr(L4, "load_family", lambda c, d: (recs, labels, A, ["roi_A.pre"], ["3", "-1"]))
    real = {"frame.pre": np.array(real_pre), "frame.post": np.array(real_post)}
    monkeypatch.setattr(L4, "forecast_spaces", lambda *a, **k: {"final": (real, real)})
    sites = []
    monkeypatch.setattr(L4, "site_activations", lambda m, d, r, l, positions, *a: sites.append(positions) or
                        {"layers": ["3"], "3": A[:, 0, 0].astype(np.float64)})
    work = tmp_path / "work" / "momentum"
    work.mkdir(parents=True)
    cfg = load_config("configs/eval/l4_causal.yaml")
    cfg["probe"]["k_folds"] = 3
    cfg.update(null_draws=20, bootstrap={"n": 50})
    deltas = cfg["laws"]["momentum"]["deltas"]
    calls = []

    def fake_interventions(m, mcfg, data_dir, scenes, plans, layer, wdir, positions, force, real_clips, log):
        """A law-like model: the forecast follows the mass ratio implied by the shift's component along log q."""
        calls.append(len(scenes))
        for sc in scenes:
            q = qs[sc]
            rec = {"base|0|final": feat(_post(q))}
            for c, dd in plans[sc].items():
                for d, v in dd.items():
                    rec[f"{c}|{d:g}|final"] = feat(_post(q * np.exp(float(v[0]))))
            np.savez(wdir / f"{sc}.npz", **rec)
    monkeypatch.setattr(L4, "run_interventions", fake_interventions)
    import physlaw.models.registry as reg
    monkeypatch.setattr(reg, "load_adapter", lambda *a, **k: None)
    made = []

    def fake_runner(model, mcfg, inputs, readout, positions, work, *a, **k):
        """Localization runs of the same law-like model: the readout is log q + the shift's component along log q."""
        r = _ShiftRunner(lambda i: np.log(qs[recs[i]["group"]]))
        made.append((positions, readout.space))
        return r
    monkeypatch.setattr(L4, "make_runner", fake_runner)
    cfg["localize"].update(results_root=str(tmp_path / "results"), max_groups=12, bootstrap=100,
                           das={"ks": [1, 2], "steps": 150, "decoder_steps": 300, "max_model_pairs": 4})
    # the fake stage-4 cache files carry `phases`: even groups' contact in token-frame 3, odd groups' in 4
    for r in recs:
        f = tmp_path / "cache" / r["file"]
        f.parent.mkdir(parents=True, exist_ok=True)
        ct = 3 + int(r["group"][1:]) % 2
        np.savez(f, phases=np.array(["pre"] * ct + ["post"] * (8 - ct)))
    mcfg = type("M", (dict,), {"name": "fake"})()
    res = L4.run(tmp_path / "cache", tmp_path / "f1", cfg, mcfg, tmp_path / "work", log=lambda *a: None)
    assert calls == [n_g]
    assert res["law"] == "momentum" and res["layer"] == "3" and res["mode"] == "across"
    v = res["variants"]["frame"]
    assert v["law"]["slope"] == pytest.approx(1.0, abs=0.15) and abs(v["random_dir"]["slope"]) < 0.1
    assert v["verdict"]["matches_law"] and sites == ["roi_A"] and res["positions"] == "roi_A"
    assert "passive" not in v           # a pre-event site holds no post-event state: no passive arm
    assert v["law"]["wrong_way"] == 0 and v["law"]["per_scene"]["q50"] == pytest.approx(1.0, abs=0.15)
    # input-level reference: the real clip at theta + dtheta read with the same instrument (slope 1 by construction)
    assert v["input"]["slope"] == pytest.approx(1.0, abs=0.2)
    # swap test: the base's log-q component replaced by the donor's -> the donor's mass ratio (IIA ~ 1)
    sw = v["pairs"]["swap"]
    assert sw["n_pairs"] >= n_g and sw["iia"] > 0.8 and sw["fraction_achieved"] == pytest.approx(1.0, abs=0.2)
    assert sw["error_ratio"] < 0.3 and v["pairs"]["swap_random"]["iia"] < 0.5
    assert "selectivity" in sw and np.isfinite(sw["selectivity"]["target_change_sd"])
    od = res["off_distribution"]
    assert od["real_q95"] > 0 and "law" in od["by_condition"] and "1" in od["by_condition"]["law"]
    assert res["n_pairs"]["patch"] > 0 and "patch" in v["pairs"]       # same pre velocity: no required change
    pd = res["patch_donors"]                      # donors share the base's contact token-frame; exclusions logged
    par = lambda clip: int(clip.split("/")[0][1:]) % 2
    assert pd["donors"] and all(par(d) == par(s) for s, ds in pd["donors"].items() for d in ds)
    assert set(pd["excluded_other_token_frame"]) == set(pd["donors"]) and all(
        n == n_g // 2 for n in pd["excluded_other_token_frame"].values())    # every other-parity scene (all aligned)
    assert L4.run(tmp_path / "cache", tmp_path / "f1", cfg, mcfg, tmp_path / "work", log=lambda *a: None) and calls == [n_g]
    md = L4.table(res)
    assert "law slope +1.000" in md and "random_dir" in md and "swap" in md
    # localization: gated in by the steering / swap verdicts; the site layer is flagged and localized
    loc = res["localization"]
    assert loc["gate"]["gated_in"] and set(loc["gate"]["passed"]) >= {"steering", "swap"}
    assert not loc["gate"]["l2_json_found"] and made and made[0][0] == "roi_A"
    assert loc["control_quantity"] == "log_radius_ratio" and loc["layers"]["flagged"] == ["3"]
    b3 = loc["layers"]["by_layer"]["3"]
    assert b3["law"]["iia"] > 0.9 and b3["random"]["iia"] < 0.2 and b3["size"]["iia"] < 0.2
    assert b3["shuffled"]["iia"] < b3["law"]["iia_ci95"][0]
    assert loc["knockout_layers"]["3"]["law_span"]["passes"] and loc["verdict"]["localized_layers"] == ["3"]
    assert loc["components"]["3"]["skipped"] and loc["das"]["objective"] == "readout"
    assert loc["das"]["by_k"]["1"]["iia"] > 0.8 and loc["das"]["by_k"]["1"]["model"]["n_pairs"] > 0   # wiring
    assert "L4 localization" in L4.table(res)


def test_energy_routine_end_to_end_generated_videos(tmp_path, monkeypatch):
    """DiT path: the steered videos' track-B readout rows are scored (saturation at 1)."""
    from physlaw.config import load_config
    X, E, S, G = _ladder()
    recs, labels = [], []
    for i, (e, g) in enumerate(zip(E, G)):
        recs.append({"group": f"g{g}", "twin": f"eps{e}", "clip": f"g{g}/eps{e}"})
        va, vb = np.array([S[i] / (1 + e) * 2, 0.0]), np.array([0.0, 0.0])
        labels.append({"physlaw": {"objects": {"A": {"mass": 1.0}, "B": {"mass": 1.0}}, "contact_normal": [1.0, 0.0],
                                   "twin_group_id": f"g{g}", "pre_velocity": {"A": va.tolist(), "B": vb.tolist()},
                                   "post_velocity": {"A": (va * (1 - e) / 2).tolist(), "B": (va * (1 + e) / 2).tolist()}}})
    # DiT spaces (SNR, layer): snr-8/L5 carries the eps curve, snr+8/L5 is noise (fails the precondition)
    Xn = np.random.default_rng(9).normal(size=X.shape)
    A = np.stack([np.stack([X, X], 1), np.stack([Xn, Xn], 1), np.stack([X, X], 1)], 2).astype(np.float32)
    monkeypatch.setattr(L4, "load_family", lambda c, d: (recs, labels, A, ["frame.pre", "frame.post"],
                                                         ["snr-8/L5", "snr+8/L5", "-1"]))
    cfg = load_config("configs/eval/l4_causal.yaml")
    cfg["probe"]["k_folds"] = 3
    cfg.update(null_draws=20, bootstrap={"n": 50}, localize={"enabled": False})
    lc = cfg["laws"]["energy"]
    rng = np.random.default_rng(1)
    e0 = {f"g{g}": rng.uniform(0.4, 0.7) for g in sorted(set(G))}
    d_e = np.eye(X.shape[1])[0]

    seen = {}

    def fake_interventions(m, mcfg, data_dir, scenes, plans, layer, wdir, positions, force, real_clips, log, snrs=None):
        """Track-B rows of a model whose eps saturates at 1 along the law's curve, keeps rising along the speed
        direction, and (swap) takes the donor's eps component (activations carry 2 eps along d_e). Shifts are
        per fitted SNR space ({snr_db: [D]})."""
        seen.update(layer=layer, snrs=snrs)
        for sc in scenes:
            b = e0[sc]
            rows = {"base|0": {"measurable": True, "epsilon": b}}
            for c, dd in plans[sc].items():
                for d, v in dd.items():
                    seen.setdefault("keys", set()).add(tuple(sorted(v)))
                    e = {"law": 1 - (1 - b) * np.exp(-2 * d), "speed": b + 0.3 * d, "random_dir": b}.get(
                        c, min(1.0, b + float(v[-8.0] @ d_e) / 2))
                    rows[f"{c}|{d:g}"] = {"measurable": True, "epsilon": e}
            for name in real_clips[sc]:
                rows[f"real|{name}"] = {"measurable": True, "epsilon": float(name.split("eps")[1])}
            np.savez(wdir / f"{sc}.npz", x=np.zeros(1))
            (wdir / f"{sc}_readout.json").write_text(json.dumps(rows))
            sch = L4.snr_schedule([-np.inf, -12.0, -1.0, 5.0, 30.0], snrs, [-8.0])
            (wdir / f"{sc}_steps.json").write_text(json.dumps({"law|1": sch}))
    monkeypatch.setattr(L4, "run_interventions", fake_interventions)
    import physlaw.models.registry as reg
    monkeypatch.setattr(reg, "load_adapter", lambda *a, **k: None)
    lc["readout"] = "epsilon_generated_only"           # the generated-video path only (no decoded forecasts here)
    monkeypatch.setitem(L4.READOUTS, "epsilon_generated_only", L4.READOUTS["epsilon"])
    mcfg = type("M", (dict,), {"name": "fake"})({"kind": "diffusion"})
    res = L4.run(tmp_path / "cache", tmp_path / "f2", cfg, mcfg, tmp_path / "work", log=lambda *a: None)
    assert res["law"] == "energy" and res["positions"] == "post"
    # one curve per cached SNR space that passes: snr-8 fitted, snr+8 (noise) not; steps nearest +8 are skipped
    inj = res["injection"]
    assert res["precondition"]["snr+8/L5"]["p_retention"] > 0.05 and "snr-8/L5" in res["layers_passing"]
    assert inj["block"] == 5 and inj["cached_snr_db"] == [-8.0, 8.0] and inj["fitted_snr_db"] == [-8.0]
    assert res["layer"] == "snr-8/L5" and seen["snrs"] == [-8.0, 8.0] and seen["keys"] == {(-8.0,)}
    assert [r["curve"] for r in inj["steps"]["law"]] == ["snr-8", "snr-8", "snr-8", "skipped", "skipped"]
    assert set(res["manifolds"][0]["by_snr"]) == {"snr-8"}
    v = res["variants"]["generated"]
    assert v["law"]["saturation"] == pytest.approx(1.0, abs=0.05) and v["speed"]["saturation"] > 1.3
    assert v["verdict"]["saturates_at_1"] and v["verdict"]["differs_from_control"]
    assert v["law"]["wrong_way"] == 0 and v["input"]["value_at_0"] == pytest.approx(0.0)      # the real rung at 0
    assert v["pairs"]["swap"]["n_pairs"] > 0 and v["pairs"]["swap"]["iia"] > v["pairs"]["swap_random"]["iia"]
    assert "saturation" in L4.table(res)


def test_patch_scores_physics_vs_constant_leak():
    """Collision patch: a response that follows dv_B' = k (du . n) n (any gain) has the high-k / low-k slope ratio
    physics predicts; a constant-gain leak of A's velocity into B's readout has ratio 1; doing nothing scores error
    ratio 1."""
    rng = np.random.default_rng(0)
    P = 400
    n = np.array([1.0, 0.0])
    k = rng.uniform(0.3, 3.0, P)
    vA, vD = rng.normal(size=(P, 2)), rng.normal(size=(P, 2))
    du = (vD - vA) @ n
    ctx = [{"n": n}] * P
    phys = 0.5 * (k * du)[:, None] * n + 0.01 * rng.normal(size=(P, 2))
    leak = 0.5 * du[:, None] * n + 0.01 * rng.normal(size=(P, 2))
    a = L4.patch_scores(phys, vA, vD, ctx, k)
    b = L4.patch_scores(leak, vA, vD, ctx, k)
    z = L4.patch_scores(np.zeros((P, 2)), vA, vD, ctx, k)
    assert a["slope_ratio"]["observed"] == pytest.approx(a["slope_ratio"]["physics"], rel=0.1)
    assert b["slope_ratio"]["observed"] == pytest.approx(1.0, abs=0.15) and a["slope_ratio"]["physics"] > 2
    assert z["error_ratio"] == pytest.approx(1.0) and a["error_ratio"] < 0.6 and a["p_sign_flip"] < 0.01
    full = L4.patch_scores((k * du)[:, None] * n, vA, vD, ctx, k)
    assert full["error_ratio"] < 1e-6 and full["iia"] == 1.0 and full["transfer_along_n"] == pytest.approx(1.0)


def test_patch_verdict_pass_only_for_physics_beating_controls():
    """Collision-patch pass rule: (i) B's change along n significantly closer to the momentum-required change than to
    none (group-bootstrap CI and sign-flip p), (ii) the high- / low-impulse slope ratio's CI excludes 1 and covers
    physics, (iii) beats the random-patch control on (i). Physics passes; a constant leak (ratio 1), doing nothing and
    a run without any control fail, each with its reason."""
    rng = np.random.default_rng(0)
    P = 240
    groups = np.repeat(np.arange(60), 4)
    n = np.array([1.0, 0.0])
    k = np.repeat(rng.uniform(0.4, 2.5, 60), 4)
    vA, vD = rng.normal(size=(P, 2)), rng.normal(size=(P, 2))
    du = (vD - vA) @ n
    ctx = [{"n": n}] * P
    noise = lambda: 0.2 * rng.normal(size=(P, 2))
    resp = {"phys": 0.9 * (k * du)[:, None] * n + noise(), "leak": 1.1 * du[:, None] * n + noise(),
            "none": noise()}
    rnd = noise()
    v = {}
    for name, obs in resp.items():
        arms = {"patch": L4.patch_scores(obs, vA, vD, ctx, k, groups=groups, n_boot=300),
                "patch_random": L4.patch_scores(rnd, vA, vD, ctx, k, groups=groups, n_boot=300)}
        d = {c: L4.patch_closer(o, vA, vD, ctx, k) for c, o in (("patch", obs), ("patch_random", rnd))}
        v[name] = L4.patch_verdict(arms, d, groups, n_boot=300)
        if name == "phys":
            assert arms["patch"]["closer_n"]["ci95"][0] > 0 and arms["patch"]["closer_n"]["p_sign_flip"] < 0.01
            sr = arms["patch"]["slope_ratio"]
            assert sr["observed_ci95"][0] <= sr["physics"] <= sr["observed_ci95"][1] and sr["observed_ci95"][0] > 1
            assert v[name]["controls"]["patch_passive"].startswith("not run")
            json.dumps(v[name])
            no_ctl = L4.patch_verdict({"patch": arms["patch"]}, {"patch": d["patch"]}, groups, n_boot=100)
            assert no_ctl["verdict"] == "fail" and no_ctl["reasons"] == ["beats_controls"]
    assert v["phys"]["verdict"] == "pass" and not v["phys"]["reasons"]
    assert v["leak"]["verdict"] == "fail" and "slope_ratio_matches_physics" in v["leak"]["reasons"]
    assert v["leak"]["criteria"]["closer_to_required"]           # (i) alone does not pass a leak
    assert v["none"]["verdict"] == "fail" and "closer_to_required" in v["none"]["reasons"]
    assert L4.patch_verdict({"patch": {"n_pairs": 0}}, {}, [])["verdict"] == "uninformative"
    # the full-ROI patch also swaps A's MASS in: required = k_donor (v_A,donor - v_B) . n - k (v_A - v_B) . n
    kd, vB = rng.uniform(0.4, 2.5, P), rng.normal(size=(P, 2))
    req = kd * ((vD - vB) @ n) - k * ((vA - vB) @ n)
    obs = req[:, None] * n + noise()
    don = dict(k_donor=kd, vB_base=vB)
    arms = {"patch": L4.patch_scores(obs, vA, vD, ctx, k, groups=groups, n_boot=300, **don),
            "patch_random": L4.patch_scores(rnd, vA, vD, ctx, k, groups=groups, n_boot=300, **don)}
    d = {c: L4.patch_closer(o, vA, vD, ctx, k, **don) for c, o in (("patch", obs), ("patch_random", rnd))}
    assert arms["patch"]["transfer_along_n"] == pytest.approx(1.0, abs=0.1)
    assert L4.patch_verdict(arms, d, groups, n_boot=300)["criteria"]["closer_to_required"]
    assert L4.patch_scores(obs, vA, vD, ctx, k, groups=groups, n_boot=50)["closer_n"]["mean"] < arms["patch"]["closer_n"]["mean"]


def test_null_verdict_inert_effective_and_leaking_pushes():
    """Slope-0 laws: an unchanged outcome passes only with positive evidence the push worked: (a) the model's theta
    readout moves by >= min_effect of its real-video change, or (b) a control direction (size) moves the outcome.
    Inert push -> uninformative; effective law-obeying push -> pass; leaking push -> fail."""
    rng = np.random.default_rng(0)
    G, d = 30, [-1.0, -0.5, 0.0, 0.5, 1.0]
    x = np.array(d)
    flat = lambda: 0.01 * rng.normal(size=(G, len(d)))
    lc = {"predict": 0.0, "tol": 0.1}
    cases = {"inert": ({"law": flat(), "size": flat()}, 0.0),
             "effective": ({"law": flat(), "size": flat()}, 0.9),
             "leak": ({"law": 0.6 * x + flat(), "size": flat()}, 0.9),
             "control_moves": ({"law": flat(), "size": 0.5 * x + flat()}, 0.0)}
    out = {}
    for name, (curves, th_gain) in cases.items():
        res = L4.score_response(curves, d, list(range(G)), lc, n_boot=200)
        th = {"law": th_gain * 2.0 * x + flat(), "size": flat()}          # theta readout; real-video slope 2
        L4.null_verdict(res, L4.push_effect(th, d, list(range(G)), 2.0, n_boot=200), 0.25)
        out[name] = res["verdict"]
    assert out["inert"]["result"] == "inert push (uninformative)" and not out["inert"]["law_like"]
    assert out["effective"]["result"] == "pass" and out["effective"]["law_like"]
    assert out["effective"]["effective_push"]["theta_readout_fraction_of_real"] == pytest.approx(0.9, abs=0.05)
    assert out["leak"]["result"] == "fail" and not out["leak"]["law_like"]
    assert out["control_moves"]["result"] == "pass" and out["control_moves"]["effective_push"]["control_moves_outcome"]["size"]
    res = L4.score_response(cases["inert"][0], d, list(range(G)), lc, n_boot=50)
    assert L4.null_verdict(res, None)["verdict"]["result"] == "inert push (uninformative)"   # no theta readout
    # the zero-change swap follows the same rule
    dth = rng.choice([-1.0, 1.0], 100) * rng.uniform(0.3, 1.0, 100)
    sw = lambda obs, mv, cm=False: L4.invariance_scores(obs, mv, dth, 0.5, "sd_real", real_slope=2.0, control_moved=cm)
    assert sw(0.01 * rng.normal(size=100), 0.0 * dth)["result"] == "inert swap (uninformative)"
    assert sw(0.01 * rng.normal(size=100), 1.8 * dth)["result"] == "pass"
    assert sw(0.01 * rng.normal(size=100), 1.8 * dth)["readout_moved_fraction"] == pytest.approx(0.9)
    assert sw(0.01 * rng.normal(size=100), 0.0 * dth, True)["result"] == "pass"
    assert sw(0.8 * dth, 1.8 * dth)["result"] == "fail"


def test_swap_delta_curve_basis_and_interchange_scores():
    man = {"coef": np.stack([np.eye(6)[0] * 2.0, np.ones(6)]), "theta_ref": 0.0, "deg": 1}
    Q = L4.curve_basis(man)
    assert Q.shape == (6, 1)
    xb, xd = np.arange(6.0), -np.arange(6.0) + 5
    d = L4.swap_delta(xb, xd, Q)
    assert np.allclose((xb + d)[0], xd[0]) and np.allclose((xb + d)[1:], xb[1:])     # only the curve component
    assert np.allclose(L4.swap_delta(xb, xd), xd - xb)                               # full patch
    req = np.linspace(-1, 1, 41)
    req = req[req != 0]
    good = L4.interchange_scores(req * 0.9, req, nt=np.zeros_like(req) + 0.01)
    none = L4.interchange_scores(np.zeros_like(req), req)
    assert good["iia"] == 1.0 and good["fraction_achieved"] == pytest.approx(0.9) and good["selectivity"]["ratio"] > 10
    assert none["iia"] == 0.0 and none["error_ratio"] == pytest.approx(1.0)
    assert L4.required_change({"predict": "saturation"}, 0.5, 1.4) == pytest.approx(0.5)
    assert L4.required_change({"predict": 1.0}, 0.2, -0.3) == pytest.approx(-0.5)


def test_mahalanobis_off_distribution_and_reliability():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 50))
    md = L4.mahalanobis(X)
    real = md(rng.normal(size=(100, 50)))
    od = L4.off_distribution({"law": {0.5: list(md(rng.normal(size=(20, 50)))), 4.0: list(md(rng.normal(size=(20, 50)) + 6))}},
                             real)
    assert od["by_condition"]["law"]["0.5"]["frac_above_real_q95"] < 0.3
    assert od["by_condition"]["law"]["4"]["frac_above_real_q95"] == 1.0
    sl = L4.slope_scores({"law": [[0, 1, 2], [0, -1, -2], [0, 1, 2], [0, 1, 2]]}, [0, 1, 2], list("abcd"), 1.0, n_boot=0)
    assert sl["law"]["wrong_way"] == 0.25 and sl["law"]["per_scene"]["q50"] == pytest.approx(1.0)


def test_kinematic_readout_from_token_frames():
    """Gravity readout log(a_heavy / a_light) from per-token-frame pools: the floor (real drops, mass-independent) reads
    ~0, a forecast in which the heavy ball falls 1.5x faster reads log 1.5."""
    from physlaw.eval import laws  # noqa: F401
    rng = np.random.default_rng(0)
    t = np.arange(12) * 0.1
    ph = np.where(t < 0.55, "pre", "post")
    P = rng.normal(size=(6, 24))
    emb = lambda V: V.reshape(len(V), -1) @ P + 0.005 * rng.normal(size=(len(V), 24))
    fr = {"t": [], "phase": [], "V": [], "W": [], "t_fc": []}
    real, fc, states = [], [], []
    for i in range(40):
        g = 9.81 * rng.uniform(0.8, 1.2)
        fall = lambda gs: np.stack([np.zeros((12, 2)), np.zeros((12, 2)),
                                    -np.clip(t - 0.55, 0, None)[:, None] * gs[None] - 0.5], -1)
        kind = i % 3          # real twins: legal, heavy faster, light faster (an instrument needs both objects to vary)
        V = fall(np.array([g, g * (1.0, 1.5, 0.7)[kind]]))
        fr["t"].append(t); fr["phase"].append(ph); fr["V"].append(V); fr["W"].append(np.zeros_like(V)); fr["t_fc"].append(t[6:])
        real.append(emb(V)); fc.append(emb(fall(np.array([g, 1.5 * g]))[6:]))
        states.append(({"v": V[5]}, {"v": V[6]}, {"g": np.array([0, 0, -9.81]), "m": np.array([1.0, 3.0]),
                                                   "event_time_s": 0.55, "clean_until_s": 1.2}))
    grp = np.array([f"g{i // 3}" for i in range(40)])
    rd = L4.frames_reader({"real": real, "fc": fc}, fr, grp, states, "log_mass_ratio_accel", 5, (1e-3, 1e-1, 10.0))
    f = L4.READOUTS["log_mass_ratio_accel"][0]
    legal = [i for i in range(40) if i % 3 == 0]
    floor = [f(rd["pre"](i), rd["floor"](i), states[i][2]) for i in legal]
    fcst = [f(rd["pre"](i), rd["post"](i, fc[i]), states[i][2]) for i in legal]
    assert np.median(floor) == pytest.approx(0.0, abs=0.05) and np.median(fcst) == pytest.approx(np.log(1.5), abs=0.05)


# ---------------------------------------------------------------- friction, oscillation, statics, angular momentum

def test_step_scores_statics_boundary():
    """Statics: the steered tip indicator must switch where theta + dtheta crosses x = 1."""
    rng = np.random.default_rng(0)
    d = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    th0 = np.round(rng.choice(np.arange(0.55, 1.46, 0.1), 30), 2)
    law = ((th0[:, None] + d[None]) > 1.0).astype(float)
    flip = rng.random(law.shape) < 0.05
    curves = {"law": np.where(flip, 1 - law, law), "random_dir": (th0[:, None] > 1.0).astype(float).repeat(5, 1),
              "passive": (th0[:, None] > 1.0).astype(float).repeat(5, 1)}
    lc = {"predict": "step", "boundary": 1.0, "control": "random_dir"}
    s = L4.score_response(curves, d, list(range(30)), lc, n_boot=200, theta0=th0)
    assert s["law"]["accuracy"] > 0.9 and s["law"]["step"] > 0.8 and s["verdict"]["law_like"]
    assert s["law"]["switch"] > 0.8 and s["law"]["wrong_way"] == 0.0
    assert s["random_dir"]["switch"] == 0.0                     # the unsteered readout never switches within a scene
    stuck = {**curves, "law": np.zeros_like(law)}                # never tips: no step
    assert not L4.score_response(stuck, d, list(range(30)), lc, n_boot=200, theta0=th0)["verdict"]["law_like"]
    assert L4.required_change(lc, 0.95, 1.05) == 1.0 and L4.required_change(lc, 1.15, 1.35) == 0.0
    md = L4.table({"law": "statics", "family": "f11", "theta": "x", "mode": "across", "pool": "roi_A.pre",
                   "predict": "step", "precondition": {}, "layers_passing": ["3"], "layer": "3", "positions": "roi_A",
                   "deltas": d.tolist(), "n_scenes": 30, "manifolds": [], "variants": {"frame_t": s}})
    assert "steps at theta + dtheta = 1" in md and "random_dir" in md


def test_new_law_blocks_config_and_label_values():
    from physlaw.config import load_config
    cfg = load_config("configs/eval/l4_causal.yaml")
    assert {L4.LAW_OF_FAMILY[f] for f in ("f6", "f9", "f11", "f12")} == {"friction", "oscillation", "statics",
                                                                          "angular_momentum"}
    for law in set(L4.LAW_OF_FAMILY.values()):
        lc = cfg["laws"][law]
        assert lc["readout"] in L4.READOUTS and lc["mode"] in ("ladder", "across")
    labels = {"physlaw": {"objects": {"A": {"mass": 2.0, "radius": 0.1}, "B": {"mass": 3.0, "radius": 1.2},
                                      "P": {"mass": 1.0, "radius": 0.05, "aux": True}},
                          "constants": {"impact_parameter": -0.6, "bar_half_length": 1.2, "overhang_x": 1.15,
                                        "inertia_factors": {"A": 0.81, "B": 0.5}}}}
    assert L4.label_value(labels, "log_mass_A") == pytest.approx(np.log(2.0))
    assert L4.label_value(labels, "log_radius_B") == pytest.approx(np.log(1.2))
    assert L4.label_value(labels, "impact_parameter_rel") == pytest.approx(0.5)
    assert L4.label_value(labels, "physlaw.constants.overhang_x") == 1.15
    assert L4.label_value(labels, "kappa_A") == 0.81
    # rolling: kappa varies per group, the law's slope is taken at the fitted scenes' median kappa_A
    lc = L4.resolve_law_cfg(dict(cfg["laws"]["rolling"]), np.array([0.5, 0.7, 0.9]))
    assert lc["kappa_ref"] == 0.7 and L4.law_value(lc) == pytest.approx(-1 / 1.7)


def test_sine_period_rotation_and_new_readouts():
    from physlaw.eval import laws
    t = np.arange(0.0, 1.2, 1 / 16)
    V = np.stack([0.8 * np.sin(2 * np.pi * t / 2.0 + 0.3), np.zeros_like(t), 0.1 * np.sin(4 * np.pi * t / 2.0)], 1)
    assert L4.sine_period(t, V) == pytest.approx(2.0, rel=0.02)                 # >= a quarter period seen
    assert L4.sine_period(t * 3.0, V) == pytest.approx(6.0, rel=0.02)           # time scale: the period scales
    assert L4.rotation_angle(t, np.tile([0.0, 0.5, 0.0], (len(t), 1))) == pytest.approx(0.5 * t[-1])
    rd = {k: L4.READOUTS[k][0] for k in L4.READOUTS}
    # friction: log of A's deceleration along its motion (mu g); B's is the non-target
    q = {"v": np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]]), "a": np.array([[-2.0, 0.0, 0.0], [-1.0, -1.0, 0.0]])}
    assert rd["log_decel_A"](None, q, {}) == pytest.approx(np.log(2.0))
    assert L4.NONTARGET["friction"](None, q, {}) == pytest.approx(np.log(np.sqrt(2.0)))
    # oscillation: A's post period
    assert rd["log_period_A"](None, {"T": np.array([1.8, 2.1])}, {}) == pytest.approx(np.log(1.8))
    # statics: the rotation over the tip window against laws.TIP_ANGLE
    assert rd["tipped"](None, {"phi": np.array([0.3])}, {}) == 1.0 and rd["tipped"](None, {"phi": np.array([0.0])}, {}) == 0.0
    # angular momentum: L_ratio of the decoded states with the labelled contact positions; spin x 1.2 moves it
    ctx = {"m": np.array([0.8, 1.6]), "I": np.array([0.02, 0.9]), "n": np.array([1.0, 0.0]), "c": np.array([0.0, 0.4])}
    pre = {"p": np.array([[-0.25, 0.4, 0.0], [0.15, 0.0, 0.0]]), "v": np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
           "w": np.zeros((2, 3))}
    post = laws.rigid_impulse(pre, ctx, 0.9)
    c = {**ctx, "p_event": pre["p"]}
    assert rd["L_ratio"]({"v": pre["v"]}, {"v": post["v"], "w": post["w"]}, c) == pytest.approx(1.0)
    assert abs(rd["L_ratio"]({"v": pre["v"]}, {"v": post["v"], "w": 1.2 * post["w"]}, c) - 1.0) > 0.01
    assert L4.NONTARGET["angular_momentum"](None, post, c) == pytest.approx(1.6, rel=1e-9)   # sum m v = m_A v_A


def _token_series(rng, n, t, te, make, ctx=None):
    """Per clip (V [T, 1, 3], W [T, 1, 3]) series and a linear token embedding of them (+ noise)."""
    P = rng.normal(size=(6, 24))
    emb = lambda V, W: np.concatenate([V, W], -1).reshape(len(V), -1) @ P + 0.002 * rng.normal(size=(len(V), 24))
    fr = {"t": [], "phase": [], "V": [], "W": [], "t_fc": []}
    real, fc, states = [], [], []
    ph = np.where(t < te, "pre", "post")
    for i in range(n):
        (V, W), (Vf, Wf) = make(i)
        fr["t"].append(t); fr["phase"].append(ph); fr["V"].append(V); fr["W"].append(W); fr["t_fc"].append(t[t >= te])
        real.append(emb(V, W)); fc.append(emb(Vf, Wf)[t >= te])
        states.append(({"v": V[0]}, {"v": V[-1]}, {"event_time_s": te, "clean_until_s": float(t[-1]), **(ctx or {})}))
    return fr, real, fc, states


def test_period_and_tip_readouts_from_token_frames():
    """Oscillation: the floor (real swings) reads A's own period, a forecast swinging 1.25x slower reads log 1.25 more.
    Statics: the floor's tip indicator matches the real motion; a forecast that tips everything reads 1."""
    rng = np.random.default_rng(1)
    t = np.arange(0.0, 3.0, 1 / 16)
    te = 0.9
    Tp = rng.uniform(1.5, 2.5, 30)
    sw = lambda T_, i: (np.stack([np.sin(2 * np.pi * t / T_ + i), 0 * t, 0 * t], 1)[:, None], np.zeros((len(t), 1, 3)))
    fr, real, fc, states = _token_series(rng, 30, t, te, lambda i: (sw(Tp[i], i), sw(1.25 * Tp[i], i)))
    grp = np.array([f"g{i}" for i in range(30)])
    rd = L4.frames_reader({"real": real, "fc": fc}, fr, grp, states, "log_period_A", 5, (1e-3, 1e-1, 10.0))
    f = L4.READOUTS["log_period_A"][0]
    floor = np.array([f(rd["pre"](i), rd["floor"](i), states[i][2]) for i in range(30)])
    fcst = np.array([f(rd["pre"](i), rd["post"](i, fc[i]), states[i][2]) for i in range(30)])
    assert np.median(np.abs(floor - np.log(Tp))) < 0.05
    assert np.median(fcst - floor) == pytest.approx(np.log(1.25), abs=0.05)
    tips = rng.random(30) < 0.5
    rot = lambda on: (np.zeros((len(t), 1, 3)), np.stack([0 * t, on * 1.5 * np.clip(t - te, 0, None), 0 * t], 1)[:, None])
    # the tip window is the clip's own tipping time scale (laws.incline.tip_window: K sqrt(a / g_clip))
    fr, real, fc, states = _token_series(rng, 30, t, te, lambda i: (rot(float(tips[i])), rot(1.0)),
                                         {"g_clip": 9.81, "half_length": 0.6})
    rd = L4.frames_reader({"real": real, "fc": fc}, fr, grp, states, "tipped", 5, (1e-3, 1e-1, 10.0))
    f = L4.READOUTS["tipped"][0]
    assert [f(None, rd["floor"](i), None) for i in range(30)] == [float(x) for x in tips]
    assert all(f(None, rd["post"](i, fc[i]), None) == 1.0 for i in range(30))


def test_friction_readout_window_ends_at_move_end():
    """F6: log_decel_A is fitted over [branch, move_end] (io.friction_windows via law_inputs' ctx), never over the
    rest after the stop: with move_end in ctx the floor and the forecast read A's own deceleration; ending the window
    at the clip end instead (no move_end) would mix in the rest period and read a smaller deceleration."""
    rng = np.random.default_rng(2)
    t = np.arange(0.0, 3.0, 1 / 16)
    te, n = 0.9, 30
    a, t_stop = rng.uniform(0.8, 1.6, n), rng.uniform(1.9, 2.3, n)
    slide = lambda i: (np.stack([a[i] * np.clip(t_stop[i] - t, 0, None), 0 * t, 0 * t], 1)[:, None],
                       np.zeros((len(t), 1, 3)))
    grp = np.array([f"g{i}" for i in range(n)])
    f = L4.READOUTS["log_decel_A"][0]
    read = {}
    for with_end in (True, False):
        fr, real, fc, states = _token_series(np.random.default_rng(3), n, t, te, lambda i: (slide(i), slide(i)))
        if with_end:
            states = [(p, q, {**c, "move_end": float(t_stop[i] - 0.1)}) for i, (p, q, c) in enumerate(states)]
        rd = L4.frames_reader({"real": real, "fc": fc}, fr, grp, states, "log_decel_A", 5, (1e-3, 1e-1, 10.0))
        read[with_end] = (np.array([f(rd["pre"](i), rd["floor"](i), states[i][2]) for i in range(n)]),
                          np.array([f(rd["pre"](i), rd["post"](i, fc[i]), states[i][2]) for i in range(n)]))
    for x in read[True]:
        assert np.median(np.abs(x - np.log(a))) < 0.05
    assert np.median(read[False][0] - np.log(a)) < -0.2           # the rest period would bias the fit low


def test_galilean_ladder_excludes_camera_boost_twins():
    """F4 camera-boost twins (cam_<level>_<e>: boost_kind camera, edit.boost = u as well) are L3's camera_vs_ball
    comparison, not rungs of L4's boost ladder: the configured galilean block keeps ball boosts (u0 included) only;
    labels from before camera boosts (no boost_kind) are ball boosts."""
    from physlaw.config import load_config
    lc = load_config("configs/eval/l4_causal.yaml")["laws"]["galilean"]
    twins = ["u0_e1", "usmall_e1", "ularge_e015", "cam_usmall_e1", "cam_ularge_e015", "old_usmall_e1"]
    labels = [{"physlaw": {"twin": t, "edit": {"boost": [0.0, 0.3]},
                           **({} if t.startswith("old") else {"boost_kind": "camera" if t.startswith("cam") else "ball"})}}
              for t in twins]
    assert [twins[i] for i in L4.law_rows(labels, lc)] == ["u0_e1", "usmall_e1", "ularge_e015", "old_usmall_e1"]
    assert L4.law_rows(labels, {}) == list(range(len(twins)))          # other laws: no boost filter


def test_invariance_scores_law_obeying_vs_leaking():
    """Zero-prediction laws (bug: their swap pairs had r* = 0 and were dropped): invariance ratio = |outcome change| /
    reference, plus the theta-readout-moved check. A law-obeying model moves its theta readout and keeps the outcome
    (ratio ~ 0); a model leaking theta into the outcome has a large ratio; a swap that moves nothing is not law-like
    (invariance would be vacuous)."""
    rng = np.random.default_rng(0)
    dth = rng.choice([-1.0, 1.0], 200) * rng.uniform(0.3, 1.0, 200)
    obey = L4.invariance_scores(0.01 * rng.normal(size=200), 0.9 * dth, dth, 0.5, "sd_real")
    leak = L4.invariance_scores(0.8 * dth, 0.9 * dth, dth, 0.5, "sd_real")
    inert = L4.invariance_scores(0.01 * rng.normal(size=200), 0.0 * dth, dth, 0.5, "sd_real")
    assert obey["invariance_ratio"] < 0.05 and obey["readout_moved_fraction"] == pytest.approx(0.9)
    assert obey["readout_moved"] and obey["law_like"]
    assert leak["invariance_ratio"] > 0.8 and not leak["invariant"] and not leak["law_like"]
    assert inert["invariant"] and inert["readout_moved"] is False and not inert["law_like"]
    vid = L4.invariance_scores(0.01 * rng.normal(size=200), None, dth, 0.5, "sd_real")     # generated videos
    assert vid["readout_moved"] is None and np.isfinite(vid["invariance_ratio"])
    assert all(L4.null_prediction({"predict": 0.0}) for _ in [0]) and not L4.null_prediction({"predict": 1.0})
    assert not L4.null_prediction({"predict": "saturation"})


@pytest.mark.parametrize("leak", [False, True, "inert"])
def test_galilean_swap_scores_invariance_end_to_end(tmp_path, monkeypatch, leak):
    """Galilean (law slope 0) through the whole routine on synthetic arrays: the swap moves the forecast's boost (the
    model's own theta readout moves) while eps stays put for a law-obeying model (invariance ratio ~ 0, reference =
    the e1 vs e015 twins' eps difference); a model whose eps follows the boost has a large ratio. Steering and swap
    verdicts (slope-0 rule): obeying -> pass, leaking -> fail, an inert model (the push changes nothing) ->
    uninformative."""
    from physlaw.config import load_config
    rng = np.random.default_rng(0)
    D, n_g = 10, 16
    M = rng.normal(size=(4, D))
    feat = lambda v: v.reshape(-1) @ M
    e_y = np.eye(D)[0]
    recs, labels, X, real_pre, real_post, base = [], [], [], [], [], {}
    for gi in range(n_g):
        g = f"g{gi}"
        off = rng.normal(size=D) * 2
        bs = {"u0": 0.0, "usmall": rng.uniform(0.15, 0.3), "ularge": rng.uniform(0.45, 0.7)}
        for en, e in (("e1", 1.0), ("e015", 0.15)):
            for un, b in bs.items():
                t = f"{un}_{en}"
                bv = np.array([0.0, b])
                vp = np.array([[1.0, 0.0], [0.0, 0.0]]) + bv
                vq = np.array([[(1 - e) / 2, 0.0], [(1 + e) / 2, 0.0]]) + bv
                recs.append({"group": g, "twin": t, "clip": f"{g}/{t}", "file": f"{g}/{t}.npz"})
                labels.append({"physlaw": {"objects": {"A": {"mass": 1.0}, "B": {"mass": 1.0}}, "contact_normal": [1.0, 0.0],
                                           "twin_group_id": g, "edit": {"boost": bv.tolist(), "e": e},
                                           "pre_velocity": {"A": vp[0].tolist(), "B": vp[1].tolist()},
                                           "post_velocity": {"A": vq[0].tolist(), "B": vq[1].tolist()}}})
                X.append(off + 2 * b * e_y + 0.02 * rng.normal(size=D))
                real_pre.append(feat(vp) + 0.01 * rng.normal(size=D))
                real_post.append(feat(vq) + 0.01 * rng.normal(size=D))
                base[(g, en, un)] = (e, b)
    X = np.array(X)
    A = X[:, None, None].astype(np.float32)
    monkeypatch.setattr(L4, "load_family", lambda c, d: (recs, labels, A, ["frame.pre"], ["3"]))
    real = {"frame.pre": np.array(real_pre), "frame.post": np.array(real_post)}
    monkeypatch.setattr(L4, "forecast_spaces", lambda *a, **k: {"final": (real, real)})
    monkeypatch.setattr(L4, "site_activations", lambda m, d, r, l, positions, *a: {"layers": ["3"], "3": X.astype(float)})
    cfg = load_config("configs/eval/l4_causal.yaml")
    cfg["probe"]["k_folds"] = 3
    cfg.update(null_draws=20, bootstrap={"n": 50}, localize={"enabled": False})

    def fake_interventions(m, mcfg, data_dir, scenes, plans, layer, wdir, positions, force, real_clips, log):
        """The forecast shows the boost implied by the site's component along the boost axis (activations carry
        2 x boost); ``leak``: the restitution follows the boost too (e + 2 dboost)."""
        for sc in scenes:
            g, e_name = sc.split("__")[0], "e1" if sc.endswith("1.0") else "e015"
            e, b0 = base[(g, e_name, "u0")]

            def post(db):
                db = 0.0 if leak == "inert" else db
                ee = float(np.clip(e + 2 * db if leak is True else e, 0, 2))
                return feat(np.array([[(1 - ee) / 2, b0 + db], [(1 + ee) / 2, b0 + db]]))
            rec = {"base|0|final": post(0.0)}
            for c, dd in plans[sc].items():
                for d, v in dd.items():
                    rec[f"{c}|{d:g}|final"] = post(float(v @ e_y) / 2)
            np.savez(wdir / f"{sc}.npz", **rec)
    monkeypatch.setattr(L4, "run_interventions", fake_interventions)
    import physlaw.models.registry as reg
    monkeypatch.setattr(reg, "load_adapter", lambda *a, **k: None)
    mcfg = type("M", (dict,), {"name": "fake"})()
    res = L4.run(tmp_path / "cache", tmp_path / "f4", cfg, mcfg, tmp_path / "work", log=lambda *a: None)
    assert res["law"] == "galilean" and res["positions"] == "context" and res["layer"] == "3"
    sw = res["variants"]["frame"]["pairs"]["swap"]
    assert sw["n_pairs"] > n_g and sw["reference_kind"] == "u0_e1 vs u0_e015"
    assert sw["reference"] == pytest.approx(0.85, abs=0.15)          # eps(e1) - eps(e015) read by the instrument
    vd = res["variants"]["frame"]["verdict"]
    if leak == "inert":
        assert not sw["readout_moved"] and sw["result"] == "inert swap (uninformative)" and not sw["law_like"]
        assert vd["result"] == "inert push (uninformative)" and vd["matches_law"] and not vd["law_like"]
        return
    assert sw["readout_moved"] and sw["readout_moved_fraction"] > 0.5
    assert res["variants"]["frame"]["theta_readout"]["by_condition"]["law"]["fraction_of_real"] > 0.5
    if leak:
        assert sw["invariance_ratio"] > 0.2 and not sw["law_like"] and sw["result"] == "fail"
        assert vd["result"] == "fail"
    else:
        assert sw["invariance_ratio"] < 0.1 and sw["law_like"] and sw["result"] == "pass"
        assert vd["result"] == "pass" and vd["law_like"]
    assert "invariance ratio" in L4.table(res)


@pytest.mark.parametrize("leak", [False, True])
def test_dit_zero_change_swap_reads_theta_from_the_swapped_block(tmp_path, monkeypatch, leak):
    """DiTs (bug: no theta readout from a generated video, so the zero-change swap could never pass): the model's own
    theta is read from its activations at the swapped block / SNR (R1 theta probe on the real clips' site activations,
    applied to the real site vector + the swap's shift), the outcome from the generated video. The tiny DiT stand-in
    shows the hooked block output pooled over the site tokens IS that vector. Obeying -> the swap moves theta and eps
    stays (pass); leaking (eps follows the boost) -> fail; the random swap does not move theta."""
    import torch
    from physlaw.config import load_config
    rng = np.random.default_rng(0)
    D, n_g = 10, 16
    e_y = np.eye(D)[0]
    recs, labels, X, eps_of, row_of = [], [], [], {}, {}
    for gi in range(n_g):
        g, off = f"g{gi}", rng.normal(size=D) * 2
        bs = {"u0": 0.0, "usmall": rng.uniform(0.15, 0.3), "ularge": rng.uniform(0.45, 0.7)}
        for en, e in (("e1", 1.0), ("e015", 0.15)):
            for un, b in bs.items():
                t = f"{un}_{en}"
                bv = np.array([0.0, b])
                vp = np.array([[1.0, 0.0], [0.0, 0.0]]) + bv
                vq = np.array([[(1 - e) / 2, 0.0], [(1 + e) / 2, 0.0]]) + bv
                row_of[(g, en, un)] = len(recs)
                recs.append({"group": g, "twin": t, "clip": f"{g}/{t}", "file": f"{g}/{t}.npz"})
                labels.append({"physlaw": {"objects": {"A": {"mass": 1.0}, "B": {"mass": 1.0}}, "contact_normal": [1.0, 0.0],
                                           "twin_group_id": g, "edit": {"boost": bv.tolist(), "e": e},
                                           "pre_velocity": {"A": vp[0].tolist(), "B": vp[1].tolist()},
                                           "post_velocity": {"A": vq[0].tolist(), "B": vq[1].tolist()}}})
                X.append(off + 2 * b * e_y + 0.02 * rng.normal(size=D))
                eps_of[f"{g}__{t}"] = e
    X = np.array(X)
    A = np.stack([X, X], 1)[:, None].astype(np.float32)                 # [n, P=1, (snr-8/L0, -1), D]
    monkeypatch.setattr(L4, "load_family", lambda c, d: (recs, labels, A, ["frame.pre"], ["snr-8/L0", "-1"]))
    cfg = load_config("configs/eval/l4_causal.yaml")
    cfg["probe"]["k_folds"] = 3
    cfg.update(null_draws=20, bootstrap={"n": 50}, localize={"enabled": False})
    m = _TinyDiT([_flow_t(-8.0)])
    n_tok, idx = 6, np.array([1, 4])
    checked = []

    def through_stand_in(x_in, v, snrs):
        """The shift injected by the DiT hook at the stand-in's block: pooled site output - input."""
        log = []
        with L4.steer_steps(m, "snr-8/L0", idx, v, snrs, log), torch.no_grad():
            h, _ = m._pass(torch.from_numpy(np.tile(x_in, (1, n_tok, 1))), torch.tensor(float(m.ts[0])))
        z = h[0].numpy()[idx].mean(0)
        assert np.allclose(z, x_in + v[-8.0]) and log[0]["curve"] == "snr-8"
        checked.append(1)
        return z - x_in

    def fake_interventions(m_, mcfg, data_dir, scenes, plans, layer, wdir, positions, force, real_clips, log, snrs=None):
        for sc in scenes:
            g, en = sc.split("__")[0], "e1" if sc.endswith("1.0") else "e015"
            e, x0 = (1.0 if en == "e1" else 0.15), X[row_of[(g, en, "u0")]]
            rows = {"base|0": {"measurable": True, "epsilon": e}}
            for c, dd in plans[sc].items():
                for d, v in dd.items():
                    db = float(through_stand_in(x0, v, snrs) @ e_y) / 2
                    rows[f"{c}|{d:g}"] = {"measurable": True, "epsilon": e + 2 * db if leak else e}
            for name in real_clips[sc]:
                rows[f"real|{name}"] = {"measurable": True, "epsilon": eps_of[name]}
            np.savez(wdir / f"{sc}.npz", x=np.zeros(1))
            (wdir / f"{sc}_readout.json").write_text(json.dumps(rows))
            (wdir / f"{sc}_steps.json").write_text(json.dumps({}))
    monkeypatch.setattr(L4, "run_interventions", fake_interventions)
    import physlaw.models.registry as reg
    monkeypatch.setattr(reg, "load_adapter", lambda *a, **k: None)
    mcfg = type("M", (dict,), {"name": "fake"})({"kind": "diffusion"})
    res = L4.run(tmp_path / "cache", tmp_path / "f4", cfg, mcfg, tmp_path / "work", log=lambda *a: None)
    assert checked and res["layer"] == "snr-8/L0" and res["site_theta"]["space"] == "snr-8/L0"
    v = res["variants"]["generated"]
    sw, rnd = v["pairs"]["swap"], v["pairs"]["swap_random"]
    assert sw["readout_moved"] is True and sw["readout_moved_fraction"] == pytest.approx(1.0, abs=0.25)
    assert not rnd["readout_moved"] and not rnd["law_like"]
    assert v["theta_readout"]["by_condition"]["law"]["fraction_of_real"] == pytest.approx(1.0, abs=0.25)
    if leak:
        assert sw["result"] == "fail" and v["verdict"]["result"] == "fail"
    else:
        assert sw["result"] == "pass" and sw["law_like"] and v["verdict"]["result"] == "pass"


def _tiny_pred2(make):
    """A tiny encoder with a 2-block predictor / decoder (an intermediate block exists)."""
    torch = pytest.importorskip("torch")
    if make == "vjepa":
        from transformers import VJEPA2Config, VJEPA2Model
        from physlaw.models.vjepa2 import VJEPA2Adapter
        torch.manual_seed(0)
        conf = VJEPA2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, mlp_ratio=2.0, crop_size=48,
                            frames_per_clip=8, patch_size=16, tubelet_size=2, pred_hidden_size=48,
                            pred_num_hidden_layers=2, pred_num_attention_heads=2, pred_num_mask_tokens=1)
        conf._attn_implementation = "sdpa"
        cfg = {"name": "tiny", "input": {"resolution": 48, "num_frames": 8, "fps": 24},
               "token_grid": {"patch": 16, "tubelet": 2}, "layers": [0, 1]}
        return torch, VJEPA2Adapter(cfg, VJEPA2Model(conf).eval(), [0.5] * 3, [0.25] * 3, "cpu")
    from physlaw.models.videomae2 import VideoMAEv2Adapter, _build
    torch.manual_seed(0)
    model = _build(torch)(enc_dim=32, enc_depth=2, enc_heads=2, dec_dim=16, dec_depth=2, dec_heads=2, patch=16,
                          tubelet=2).eval()
    cfg = {"name": "tiny", "input": {"resolution": 48, "num_frames": 8, "fps": 8},
           "token_grid": {"patch": 16, "tubelet": 2}, "layers": [0, 1]}
    return torch, VideoMAEv2Adapter(cfg, model, "cpu")


@pytest.mark.parametrize("make", ["vjepa", "videomae"])
def test_post_event_site_is_fitted_where_it_is_injected(make):
    """Energy steering for encoders (bug: the eps curve was fitted on post-event activations and injected into the
    pre-event context tokens, where every rung is identical). The post-event site is an intermediate predictor /
    decoder block: pred_real() returns that block's activations of the masked real post tokens, and with one mask
    group (every post token a target, the context = the pre tokens) they equal the activations the hook sees in
    forecast() at the same block and tokens, i.e. the curve's space and token set are the injection's. Steering
    there moves the captured site vector by exactly the shift and changes the forecast; the encoder site still sees
    only the context."""
    torch, m = _tiny_pred2(make)
    from physlaw.models.base import PRED_LAYER
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, (8, 48, 48, 3), dtype=np.uint8)
    x, k = m.to_tensor(frames), 4                           # token-frames 0, 1 context; 2, 3 post (9 tokens each)
    assert L4.site_layers(m, "post") == [PRED_LAYER.format(0)]            # intermediate blocks only (not the last)
    assert L4.site_layers(m, "context") == ["0", "1"]
    key = PRED_LAYER.format(0)
    pr = m.pred_real(x, k, n_masks=1, layers=[0])
    assert pr[key].shape[:3] == (2, 3, 3) and pr[-1].shape[:3] == (2, 3, 3)
    post_idx = np.arange(18, 36)
    cap = []
    with L4.capture(m, key, post_idx, cap), torch.no_grad():
        f0 = m.forecast(x, k)[-1]
    assert len(cap) == 1
    assert np.allclose(cap[0], pr[key].reshape(-1, pr[key].shape[-1]).float().mean(0).numpy(), atol=1e-4)
    delta = np.linspace(-1, 1, pr[key].shape[-1])
    cap2 = []
    with L4.steer(m, key, post_idx, delta), L4.capture(m, key, post_idx, cap2), torch.no_grad():
        f1 = m.forecast(x, k)[-1]
    assert np.allclose(cap2[0] - cap[0], delta, atol=1e-4) and not torch.allclose(f0, f1)
    cap3 = []                                             # pre-event site: the context-only encoder pass
    with L4.capture(m, "1", np.arange(18), cap3), torch.no_grad():
        m.forecast(x, k)
    edited = frames.copy()
    edited[k:] = 255 - edited[k:]
    cap4 = []
    with L4.capture(m, "1", np.arange(18), cap4), torch.no_grad():
        m.forecast(m.to_tensor(edited), k)
    assert np.allclose(cap3[0], cap4[0])
    assert L4.site_positions("frame.post") == "post" and L4.site_positions("frame.pre") == "context"
    assert L4.site_positions("roi_A.pre") == "roi_A"


# ---------------------------------------------------------------- DiTs: one curve per SNR space, chosen per step

class _TinyDiT:
    """DiT stand-in: two identity transformer blocks run by ``_pass(z_t, t, ...)`` (the adapters' signature), a
    rectified-flow schedule (t = s * n_train) and a sampler with a CFG-style second pass per step."""

    def __init__(self, ts, n_train=1000):
        import torch
        self.torch, self.ts, self.n_train = torch, ts, n_train
        self.tr = torch.nn.Module()
        self.tr.transformer_blocks = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])

    def _pass(self, z_t, t, embeds=None, keep=(), clean=0):
        for b in self.tr.transformer_blocks:
            z_t = b(z_t)
        return z_t, {}

    def sample(self, n_tok, D):
        out = []
        for t in self.ts:
            tt = self.torch.tensor(float(t))
            v, _ = self._pass(self.torch.zeros(1, n_tok, D), tt)
            self._pass(self.torch.zeros(1, n_tok, D), tt)          # unconditional pass: the same step
            out.append(v[0].numpy())
        return out


def _flow_t(db, n_train=1000):
    return n_train / (1.0 + 10 ** (db / 20)) if np.isfinite(db) else (n_train if db < 0 else 0.0)


def test_step_snr_db_and_nearest_snr():
    m = _TinyDiT([])
    for db in (-20.0, -3.5, 0.0, 8.0):
        assert L4.step_snr_db(m, _flow_t(db)) == pytest.approx(db, abs=1e-6)
    assert L4.step_snr_db(m, 1000) == -np.inf and L4.step_snr_db(m, 0) == np.inf
    ddpm = type("D", (), {"alphas_cumprod": np.array([0.999, 0.5, 0.0])})()
    assert L4.step_snr_db(ddpm, 1) == pytest.approx(0.0) and L4.step_snr_db(ddpm, 2) == -np.inf
    assert L4.step_snr_db(ddpm, 0) == pytest.approx(10 * np.log10(0.999 / 0.001))
    cached = [-20.0, -8.0, 0.0, 8.0]
    assert [L4.nearest_snr(d, cached) for d in (-np.inf, -15, -5, 3, 12, np.inf)] == [-20, -20, -8, 0, 8, 8]
    assert L4.snr_of("snr+8/L15") == 8.0 and L4.snr_of("snr-20/L3") == -20.0 and L4.snr_of("15") is None
    assert L4.snr_of("latent/snr+8") is None and L4.snr_name(-8.0) == "snr-8" and L4.snr_name(0) == "snr+0"


def test_dit_steering_picks_the_nearest_snr_curve_per_step_and_skips_failing_snrs():
    """Per sampling step the block output gets the shift fitted at the cached SNR nearest the step's SNR; a step
    whose nearest cached SNR failed the precondition (+8 here: no curve) gets none; the log records it per step."""
    import torch
    step_db = [-np.inf, -15.0, -5.0, 3.0, 12.0, np.inf]
    m = _TinyDiT([_flow_t(d) for d in step_db])
    D, n_tok, idx = 4, 6, np.array([1, 4])
    deltas = {-20.0: np.full(D, 1.0), -8.0: np.full(D, 2.0), 0.0: np.full(D, 3.0)}     # +8: failed, no curve
    log = []
    with L4.steer_steps(m, "snr-8/L0", idx, deltas, [-20.0, -8.0, 0.0, 8.0], log), torch.no_grad():
        out = m.sample(n_tok, D)
    want = [1.0, 1.0, 2.0, 3.0, 0.0, 0.0]
    for o, w in zip(out, want):
        assert np.allclose(o[idx], w) and np.allclose(np.delete(o, idx, 0), 0.0)
    assert [r["curve"] for r in log] == ["snr-20", "snr-20", "snr-8", "snr+0", "skipped", "skipped"]
    assert [r["step"] for r in log] == list(range(6)) and log[0]["snr_db"] == "-inf" and log[-1]["snr_db"] == "+inf"
    assert [r["nearest_snr_db"] for r in log] == [-20, -20, -8, 0, 8, 8]
    pure = L4.snr_schedule(step_db, [-20.0, -8.0, 0.0, 8.0], list(deltas), m.ts)          # pure schedule == hook
    assert [(r["curve"], r["nearest_snr_db"]) for r in log] == [(r["curve"], r["nearest_snr_db"]) for r in pure]
    json.dumps(log)                                                                       # recordable as is
    assert "_pass" not in vars(m) and all(np.allclose(o, 0) for o in m.sample(n_tok, D))  # hooks removed


def test_saturation_scores_all_nan_slices_without_warning():
    """All-NaN scenes / plateaus / steps: no RuntimeWarning, NaN where numpy's nan-reductions give NaN, the same
    values elsewhere."""
    import warnings
    a, c = _sat_curves()
    c = {k: np.array(v, float) for k, v in c.items()}
    c["law"][0] = np.nan                    # an unmeasured scene
    c["law"][:, 2] = np.nan                 # an unmeasured step
    c["passive"][:, -2:] = np.nan           # no plateau at all
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = L4.saturation_scores(c, a, list(range(10)), n_boot=200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p = np.nanmean(c["law"][:, -2:], 1)
        assert s["law"]["saturation"] == pytest.approx(np.nanmedian(p))
        assert s["law"]["rise"] == pytest.approx(np.nanmedian(p - c["law"][:, 0]))
        assert np.allclose(s["law"]["curve_median"], np.nanmedian(c["law"], 0), equal_nan=True)
    assert np.isnan(s["law"]["curve_median"][2]) and np.isnan(s["passive"]["saturation"])
    assert np.isnan(s["law"]["minus_passive"]["median"]) and s["law"]["minus_passive"]["ci95"] is None
    assert s["law"]["saturation"] == pytest.approx(1.0, abs=0.02) and s["verdict"]["differs_from_control"]


class _TinyLatentDiT:
    """Latent DiT stand-in (Wan-like rectified flow, one timestep for all tokens): latents [1, F, C, 2, 2], one token
    per latent pixel (patch 1, patch_t 1, D = C); ``_pass`` adds t / 1000 to every token (the block output depends on
    the timestep) and runs two identity blocks; the sampler replaces the context latents by the true context noised
    with a fixed draw at every step (as WanAdapter._sample) and returns the base pass's tokens per step."""

    def __init__(self, ts, F=3, C=3, kt=2):
        import torch
        self.torch, self.ts, self.n_train, self.lpt, self.F, self.C, self.kt = torch, ts, 1000, 1, F, C, kt
        self.token_grid = type("G", (), {"grid_hw": (2, 2)})()
        self.tr = torch.nn.Module()
        self.tr.transformer_blocks = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
        self.ctx_eps = torch.from_numpy(np.random.default_rng(7).normal(size=(1, F, C, 2, 2)))

    def n_context_tokens(self, k):
        return self.kt

    def _mixed(self, x, k):
        return x                                        # the stand-in's "video" is its clean latents

    def tokens(self, z, t):
        return z.permute(0, 1, 3, 4, 2).reshape(z.shape[0], -1, self.C) + float(t) / 1000

    def _pass(self, z_t, t, embeds=None, keep=(), clean=0):
        h = self.tokens(z_t, t)
        for b in self.tr.transformer_blocks:
            h = b(h)
        return h, {}

    def ctx_at(self, z0, t):
        s = float(t) / self.n_train
        return (1 - s) * z0[:, :self.kt] + s * self.ctx_eps[:, :self.kt]

    def sample(self, z0, seed=0):
        lat = self.torch.from_numpy(np.random.default_rng(100 + seed).normal(size=z0.shape))
        out = []
        for t in self.ts:
            lat[:, :self.kt] = self.ctx_at(z0, t)
            v, _ = self._pass(lat.clone(), self.torch.tensor(float(t)), keep=(0,))
            out.append((v[0].numpy(), lat.clone()))
            lat = 0.9 * lat                             # any update: the post latents evolve between steps
        return out



@pytest.mark.parametrize("random_ctl", [False, True])
def test_dit_patch_swaps_the_donors_live_roi_from_the_same_timestep(random_ctl):
    """DiT collision patch: at every sampling step a paired pass of the donor (same timestep, same context noise draw)
    supplies its live ROI activations; the base's pooled ROI becomes exactly the donor's pooled ROI AT THAT TIMESTEP
    (the stand-in's activations carry t, so a donor pass at any other step would miss), other tokens are untouched,
    and the hooks are gone afterwards. The random control moves the ROI by the same norm along a fixed direction."""
    import torch
    m = _TinyLatentDiT([999.0, 700.0, 400.0, 100.0])
    rng = np.random.default_rng(0)
    zb, zd = (torch.from_numpy(rng.normal(size=(1, 3, 3, 2, 2))) for _ in range(2))
    ib, idd = np.array([0, 5]), np.array([1, 4])
    with pytest.raises(ValueError, match="token-frame"):   # a donor ROI beyond the base's context is never truncated
        with L4.patch_live(m, "snr-8/L0", ib, np.array([1, 4, 9]), zb, zd, k=5, log=[]):
            pass
    assert "_pass" not in vars(m)
    log = []
    with L4.patch_live(m, "snr-8/L0", ib, idd, zb, zd, k=5, log=log, random_seed=3 if random_ctl else None), torch.no_grad():
        out = m.sample(zb)
    assert [r["t"] for r in log] == m.ts and all(r["donor_t"] == r["t"] for r in log) and len(log) == 4
    assert all(r["curve"] == "patch" for r in log) and json.dumps(log)
    plain = m.sample(zb)                                                  # no hooks: the unpatched base
    assert "_pass" not in vars(m)
    dirs = []
    for t, (o, lat), (o0, _) in zip(m.ts, out, plain):
        zdon = lat.clone()
        zdon[:, :m.kt] = m.ctx_at(zd, t)                                  # the donor's input at this very step
        don = m.tokens(zdon, t)[0].numpy()[[1, 4]].mean(0)               # its live pooled ROI at timestep t
        base = o0[ib].mean(0)
        assert np.allclose(np.delete(o, ib, 0), np.delete(o0, ib, 0))
        if random_ctl:
            sh = o[ib] - o0[ib]
            assert np.allclose(sh[0], sh[1]) and np.linalg.norm(sh[0]) == pytest.approx(np.linalg.norm(don - base))
            dirs.append(sh[0] / np.linalg.norm(sh[0]))
        else:
            assert np.allclose(o[ib].mean(0), don)                        # the donor's ROI at the same timestep
            other = m.tokens(zdon, m.ts[0] if t != m.ts[0] else m.ts[1])[0].numpy()[[1, 4]].mean(0)
            assert not np.allclose(o[ib].mean(0), other)                  # ... and not at another step's
    if random_ctl:
        assert all(np.allclose(d, dirs[0]) for d in dirs)


def test_patch_donors_share_the_contact_token_frame(tmp_path):
    """Patch donors: matched contact normal AND contact in the base's token-frame; the token-frame exclusions are
    counted (geometry exclusions are not); a cache without stage 4's `phases` is an error."""
    n = 8
    theta, scene = np.arange(n, dtype=float), np.array([f"s{i}" for i in range(n)])
    normals = [[1, 0]] * 6 + [[0, 1]] * 2                     # rows 6, 7: mismatched geometry
    states = [({}, {}, {"n": nn}) for nn in normals]
    ctf = {0: 3, 1: 3, 2: 4, 3: 3, 4: 5, 5: 3, 6: 3, 7: 4}
    rng = np.random.default_rng(0)
    d, ex = L4.patch_donors(0, list(range(n)), theta, scene, 10, rng, states, 0.94, ctf)
    assert sorted(d) == [1, 3, 5] and ex == 2                 # rows 2, 4: other token-frame; 6, 7: geometry
    with pytest.raises(ValueError, match="re-run stage 4"):
        L4.contact_token_frame("/nonexistent", {"file": "x.npz"})
    np.savez(tmp_path / "old.npz", pooled=np.zeros(1))
    with pytest.raises(ValueError, match="re-run stage 4"):
        L4.contact_token_frame(tmp_path, {"file": "old.npz"})
    np.savez(tmp_path / "c.npz", phases=np.array(["pre", "pre", "straddle", "post"]))
    assert L4.contact_token_frame(tmp_path, {"file": "c.npz"}) == 2


# ---------------------------------------------------------------- localization (where the law is used)

class _ShiftRunner:
    """Localization stand-in on arrays: the readout is base(i) + the first coordinate of every shift (a model whose
    law variable lives on axis 0 of the site); no token groups or components."""
    live = False

    def __init__(self, base):
        self.base, self.calls = base, 0

    def run(self, key, i, edits):
        self.calls += 1
        return {"r": float(self.base(i) + sum(float(e[3][0]) for e in edits if e[0] == "shift")), "theta": float("nan"),
                "site": {}}

    def groups(self, layer):
        return []

    def components(self, layer):
        return []


def test_localize_gate_reads_steering_swap_and_the_l2_kink(tmp_path):
    """Localization runs only for a model that passes: steering law-like, a passing swap test, or (energy / statics)
    an L2 / L2-rep kink beyond baselines; otherwise the reason is recorded."""
    fail = {"variants": {"frame": {"verdict": {"law_like": False},
                                   "pairs": {"swap": {"iia": 0.2, "p_sign_flip": 0.4, "error_ratio": 1.1},
                                             "swap_random": {"iia": 0.3}}}}}
    l2 = tmp_path / "l2.json"
    g = L4.localize_gate(fail, "energy", False, ["steering", "swap", "l2_kink"], l2)
    assert not g["gated_in"] and "no gate passed" in g["reason"] and not g["l2_json_found"]
    l2.write_text(json.dumps({"energy_kink": {"f2": {"distances": {"l2": {"verdict": "kink, not beyond every baseline"}}}},
                              "energy_kink_rep": {"f2": {"verdict": "representational kink beyond baselines"}}}))
    g = L4.localize_gate(fail, "energy", False, ["steering", "swap", "l2_kink"], l2)
    assert g["gated_in"] and g["passed"] == ["l2_kink"] and len(g["l2_verdicts"]) == 2
    assert not L4.localize_gate(fail, "momentum", False, ["steering", "swap", "l2_kink"], l2)["gated_in"]   # not its kink
    assert not L4.localize_gate(fail, "energy", False, ["steering", "swap"], l2)["gated_in"]      # gate list honoured
    ok = {"variants": {"frame": {"verdict": {"law_like": False},
                                 "pairs": {"swap": {"iia": 0.9, "p_sign_flip": 0.001, "error_ratio": 0.2},
                                           "swap_random": {"iia": 0.1}}}}}
    assert L4.localize_gate(ok, "momentum", False, ["swap"], l2)["passed"] == ["swap"]
    inv = {"variants": {"frame": {"pairs": {"swap": {"law_like": True}}}}}
    assert L4.localize_gate(inv, "friction", True, ["swap"], l2)["gated_in"]
    cfg = {"localize": {"max_groups": 5, "das": {"ks": [1]}, "dit": {"max_groups": 2, "das": {"steps": 5}}}}
    assert L4.loc_config(cfg)["max_groups"] == 5 and L4.loc_config(cfg)["das"]["ks"] == [1]
    assert L4.loc_config(cfg, True)["max_groups"] == 2 and L4.loc_config(cfg, True)["das"] == {
        **L4.LOC_DEFAULTS["das"], "ks": [1], "steps": 5}


class _ToyLawModel:
    """Tiny transformer stand-in with a PLANTED law (encoder-like adapter: ``model.encoder.layer`` blocks, each
    x + attention(x) + mlp(x); attention = 4 heads of width 8 concatenated into an identity output projection
    ``proj``). Tokens: 4 token-frames of 2 x 2; context = frames 0, 1 (tokens 0-7), A's ROI = tokens 0 and 4, B's 1 and
    5, contact 2 and 6, background 3 and 7; forecast() returns the post tokens 8-15. Inputs at A's ROI carry
    a = s u (dim 29), b = s w (dim 30) with a random sign s, the size (dim 31) and z (dim 28); theta = u w = a b is
    NOT linearly present before layer 1 (both signs per group). Layer 1, head 1 writes a b (``law``; else z) into dim 8
    at A's ROI; layer 2, head 2 MOVES A's dim 8 to every post token (dim 16); the readout is the post tokens' dim 16. So
    theta is used at layer 1 / head 1 / A's ROI / the 1-D subspace e_8, and nowhere after layer 2's move."""

    def __init__(self, law=True, seed=0):
        import types

        import torch
        nn = torch.nn
        self.torch = torch
        self.token_grid = types.SimpleNamespace(grid_hw=(2, 2), tubelet=1)
        roi, post = [0, 4], list(range(8, 16))
        g = torch.Generator().manual_seed(seed)

        class Attn(nn.Module):
            def __init__(self, role):
                super().__init__()
                self.role, self.num_attention_heads = role, 4
                self.proj = nn.Linear(32, 32)
                with torch.no_grad():
                    self.proj.weight.copy_(torch.eye(32))
                    self.proj.bias.zero_()
                self.V0 = nn.Parameter(torch.randn(4, 8, generator=g) * 0.5)

            def forward(self, x):
                out = torch.zeros_like(x)
                out[..., 0:8] = 0.1 * torch.tanh(x[..., 25:29] @ self.V0)
                if self.role == "plant":
                    v = x[..., 29] * x[..., 30] if law else x[..., 28]
                    out[:, roi, 8] = v[:, roi]
                else:
                    out[..., 9] = 0.1 * torch.tanh(x[..., 25])
                if self.role == "move":
                    out[:, post, 16] = x[:, roi, 8].mean(1)[:, None]
                else:
                    out[..., 17] = 0.1 * torch.tanh(x[..., 26])
                out[..., 24] = 0.1 * torch.tanh(x[..., 27])
                return self.proj(out)

        class Mlp(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1, self.fc2 = nn.Linear(32, 16), nn.Linear(16, 32)
                with torch.no_grad():
                    for lin in (self.fc1, self.fc2):
                        lin.weight.zero_()
                        lin.bias.zero_()
                    self.fc1.weight[:, :8] = torch.randn(16, 8, generator=g) * 0.3
                    self.fc2.weight[:8] = torch.randn(8, 16, generator=g) * 0.3

            def forward(self, x):
                return self.fc2(torch.tanh(self.fc1(x)))

        class Block(nn.Module):
            def __init__(self, role):
                super().__init__()
                self.attention, self.mlp = Attn(role), Mlp()

            def forward(self, x):
                return x + self.attention(x) + self.mlp(x)
        self.model = nn.Module()
        self.model.encoder = nn.Module()
        self.model.encoder.layer = nn.ModuleList([Block(r) for r in ("none", "plant", "move", "none")])

    def forecast(self, x, k):
        torch = self.torch                   # adapters call self.torch.no_grad(): attribution patching lifts it
        with torch.no_grad():
            h = x
            for b in self.model.encoder.layer:
                h = b(h)
        return {-1: h[0, 8:].reshape(2, 2, 2, h.shape[-1])}


def _toy_data(n_g=16, seed=1, law=True):
    """Inputs [n, 16, 32] (two clips per group: s = +1, -1, the same theta), theta, scene / group ids, size."""
    import torch
    rng = np.random.default_rng(seed)
    X, th, sc, size = [], [], [], []
    for gi in range(n_g):
        u, w, r, z = rng.uniform(0.5, 1.5), rng.uniform(0.5, 1.5), rng.normal(0, 0.3), rng.uniform(0.25, 2.25)
        for s in (1.0, -1.0):
            x = np.zeros((16, 32))
            x[:, 25:29] = 0.3 * rng.normal(size=(16, 4))
            for t in (0, 4):
                x[t, 29], x[t, 30], x[t, 31] = s * u, s * w, r
                if not law:
                    x[t, 28] = z
            X.append(torch.as_tensor(x[None], dtype=torch.float32))
            th.append(u * w)
            sc.append(f"g{gi:02d}")
            size.append(r)
    return X, np.array(th), np.array(sc), np.array(size)


TOY_GROUPS = {"site": [0, 4], "roi_A.pre": [0, 4], "roi_B.pre": [1, 5], "contact.pre": [2, 6],
              "background.pre": [3, 7], "pre": list(range(8)), "post": list(range(8, 16))}


def _toy_localize(tmp_path, law=True, **over):
    m = _ToyLawModel(law)
    X, th, sc, size = _toy_data(law=law)
    inputs = lambda i: {"x": X[i], "k": 2, "T": 4, "groups": {g: np.array(v) for g, v in TOY_GROUPS.items()}}
    readout = L4.LocReadout(space="final", value=lambda i, z: {"r": float(z[16]), "theta": float("nan"), "post_v": None})
    runner = L4.ModelRunner(m, inputs, readout, "roi_A", tmp_path / ("law" if law else "nolaw"), n_masks=1,
                            log=lambda *a: None)
    sites = {}
    for l in ("0", "1", "2", "3"):
        rows = []
        for x in X:
            cap = []
            with L4.capture(m, l, np.array([0, 4]), cap), m.torch.no_grad():
                m.forecast(x, 2)
            rows.append(cap[0])
        sites[l] = np.array(rows, float)
    lcfg = L4.loc_config({"localize": {"max_groups": 16, "top_k": 2, "n_control_components": 2, "bootstrap": 200,
                                       "das": {"ks": [1, 2], "steps": 200, "decoder_steps": 400, "max_model_pairs": 6},
                                       **over}})
    clip_of = {s: int(np.where(sc == s)[0][0]) for s in sorted(set(sc))}
    lc = {"mode": "across", "predict": 1.0, "tol": 0.25}
    res = L4.localize(runner, lc=lc, lcfg=lcfg, theta=th, theta_c=th, scene=sc, grp=sc, fit_rows=np.arange(len(th)),
                      clip_of=clip_of, sites=sites, size=size, min_dt=0.25 * th.std(), k_folds=4,
                      untrained={"1": {"score": 0.1, "score_ci95": [0.0, 0.2]}})
    return res, runner, m, X, sites, th


def test_localize_finds_the_planted_layer_tokens_head_and_subspace(tmp_path):
    """Planted law (:class:`_ToyLawModel`): step 1 peaks at layer 1 and only layer 1's IIA CI lies above every control's
    (random direction, shuffled targets, the size span, the untrained curve); step 2 picks A's ROI; step 3 ranks head 1
    first by attribution patching (first-order estimate = the exact patch here), exact patching and mean-ablation
    knock-out both single it out against random control components, and the agreement rule localizes it (and layer
    1); step 4's DAS (k = 1) transfers theta on held-out groups along the planted axis (the fitted curve's span)."""
    res, runner, m, X, sites, th = _toy_localize(tmp_path)
    L = res["layers"]
    assert L["peak"] == "1" and L["flagged"] == ["1"] and L["metric"] == "iia"
    b1 = L["by_layer"]["1"]
    assert b1["law"]["iia"] > 0.9 and b1["law"]["fraction_achieved"] == pytest.approx(1.0, abs=0.1)
    for c in ("random", "shuffled", "size", "untrained"):
        assert b1[c]["score"] < 0.6, c
    assert all(L["by_layer"][l]["law"]["iia"] < 0.2 for l in ("0", "2", "3"))      # before the write / after the move
    assert b1["off_distribution"]["by_condition"]["law"]["0"]["median"] > 0 and len(L["curves"]["law"]) == 4
    tok = res["tokens"]["1"]
    assert tok["best"] == "roi_A.pre" and tok["passing"] == ["roi_A.pre"]
    assert tok["by_group"]["pre"]["law"]["iia"] < 0.3                 # the pooled window dilutes the swap
    co = res["components"]["1"]
    assert co["attribution"]["available"] and co["ranked"][0] == "attn.h1"
    assert co["attribution"]["by_component"]["attn.h1"]["fraction_achieved"] == pytest.approx(1.0, abs=0.05)
    assert co["patch"]["attn.h1"]["iia"] > 0.9 and co["patch"]["attn.h1"]["passes"]
    assert co["knockout"]["attn.h1"]["effect"] > 0.8 and co["knockout"]["attn.h1"]["passes"]
    assert co["knockout"]["base"]["dependence"] == pytest.approx(1.0, abs=0.05)
    assert all(abs(co["knockout"][c]["effect"]) < 0.1 for c in co["controls"])
    v = res["verdict"]
    assert v["localized_components"] == ["1/attn.h1"] and v["localized_layers"] == ["1"]
    assert all(not a["localized"] for k_, a in v["agreement"].items() if k_ not in ("component:1/attn.h1", "layer:1"))
    das = res["das"]
    assert das["layer"] == "1" and das["objective"] == "model" and das["by_k"]["1"]["iia"] > 0.8
    assert das["k_needed"] == 1 and das["by_k"]["1"]["principal_cosines_to_curve"][0] > 0.9
    assert not das["more_dims_than_curve"]
    md = L4.loc_table(res)
    assert "attn.h1" in md and "DAS" in md
    assert any(runner.work.glob("layer_1_law_*.json"))                  # every run cached on disk (resumable)


def test_localize_without_a_law_localizes_nothing(tmp_path):
    """The same toy whose head 1 writes a variable unrelated to theta: no layer passes its controls (shuffled targets
    match the swap's IIA), and even when steps 2-4 are forced at the peak no component or layer is localized."""
    res, *_ = _toy_localize(tmp_path, law=False, run_unflagged=True)
    assert res["layers"]["flagged"] == [] and res["from_peak_only"]
    assert res["verdict"]["localized_layers"] == [] and res["verdict"]["localized_components"] == []
    co = next(iter(res["components"].values()))
    assert all(not co["knockout"][c].get("passes") for c in co["exact"])
    res0, *_ = _toy_localize(tmp_path / "b", law=False)
    assert "nothing to localize" in res0["steps_2_4"] and res0["verdict"]["localized_layers"] == []


def test_attribution_gradient_matches_exact_patching_on_tiny_vjepa(tmp_path):
    """Attribution patching on a real adapter (tiny random V-JEPA 2): the backward pass through forecast() (whose own
    torch.no_grad is lifted) gives, for a small per-token-frame change of a head's / the MLP's pooled output at an
    encoder block, the exact patch's readout change to first order; the hooks find every head and the MLP."""
    torch, m = _tiny_pred2("vjepa")
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, (8, 48, 48, 3), dtype=np.uint8)
    x = m.to_tensor(frames)
    w = rng.normal(size=32)
    idx = np.arange(18)                                 # every context token (2 token-frames of 3 x 3)
    inputs = lambda i: {"x": x, "k": 4, "T": 4, "groups": {"site": idx}}
    readout = L4.LocReadout(space="final", value=lambda i, z: {"r": float(np.tanh(z @ w / 8)), "theta": float("nan")})
    r = L4.ModelRunner(m, inputs, readout, "context", tmp_path, log=lambda *a: None)
    comps = r.components("1")
    assert comps == ["attn.h0", "attn.h1", "mlp"] and L4.block_components(m, "pred/L0")[-1] == "mlp"
    G = r.grad("1", 0, comps)
    A = r.comp_acts("1", [0], comps)
    for c in comps:
        a = np.asarray(A[c][0], float)
        assert np.isfinite(a[:2]).all() and np.isnan(a[2:]).all()          # site tokens live in token-frames 0, 1
        d = np.zeros_like(a)
        d[:2] = 1e-2 * rng.normal(size=a[:2].shape)
        exact = (r.run(f"p|{c}", 0, [("set", "1", c, a + d)])["r"] - r.run(f"m|{c}", 0, [("set", "1", c, a - d)])["r"]) / 2
        approx = float((np.nan_to_num(G[c]) * d).sum())     # first order vs the central difference of exact patches
        assert approx == pytest.approx(exact, rel=0.1, abs=1e-6), c
    assert m.torch is torch                             # the no_grad lift is undone


@pytest.mark.parametrize("make", ["vjepa", "videomae"])
def test_pred_real_capture_assembles_targets_in_clip_order(make):
    """Encoders' post-event sites: component outputs are captured in the model's own P-pred-real pass (several mask
    passes); reassembled by position (V-JEPA 2: position-sorted predictor; VideoMAE: visible then decode tokens), the
    block output equals pred_real's own PRED_LAYER activations, so heads / MLPs are read at the right tokens."""
    torch, m = _tiny_pred2(make)
    from physlaw.models.base import PRED_LAYER
    rng = np.random.default_rng(1)
    x = m.to_tensor(rng.integers(0, 255, (8, 48, 48, 3), dtype=np.uint8))
    got = L4.pred_real_capture(m, x, 4, 4, "pred/L0", ["block", "mlp", "attn.h1"], 3)
    pr = m.pred_real(x, 4, n_masks=3, layers=[0])[PRED_LAYER.format(0)]
    ref = pr.reshape(-1, pr.shape[-1]).float().numpy()
    assert got["block"].shape == ref.shape and np.allclose(got["block"], ref, atol=1e-4)
    assert np.isfinite(got["mlp"]).all() and got["attn.h1"].shape[0] == ref.shape[0]


def test_dit_edits_act_only_at_their_snr_steps():
    """DiT localization edits: a layer key "snr<dB>/L<b>" shifts the block output (and captures the site) only at the
    sampling steps whose nearest cached SNR is that SNR; the _pass wrapper is removed afterwards."""
    import torch
    step_db = [-np.inf, -15.0, -5.0, 3.0, 12.0]
    m = _TinyDiT([_flow_t(d) for d in step_db])
    m.token_grid = type("G", (), {"grid_hw": (1, 2)})()
    D, n_tok, idx = 4, 6, np.array([1, 4])
    caps = {"snr-8/L0": []}
    with L4.edit_hooks(m, [("shift", "snr-8/L0", "site", np.full(D, 2.0))], {"site": idx}, 2, 3, True,
                       [-20.0, -8.0, 0.0, 8.0], caps), torch.no_grad():
        out = m.sample(n_tok, D)
    want = [0.0, 0.0, 2.0, 0.0, 0.0]                    # only the -5 dB step's nearest cached SNR is -8
    for o, w_ in zip(out, want):
        assert np.allclose(o[idx], w_) and np.allclose(np.delete(o, idx, 0), 0.0)
    assert len(caps["snr-8/L0"]) == 2 and np.allclose(caps["snr-8/L0"][0], 2.0)    # both passes of that step
    assert "_pass" not in vars(m)


def test_knockout_scores_energy_saturation_and_invariance():
    """Knock-out re-scoring. Energy: a model saturating eps_hat at 1 has creation rate 0 and a flat slope above 1;
    knocking out the saturating component lets eps_hat follow eps past 1 (creation effect > 0, beyond a control
    knock-out). Zero-change law: a knock-out that makes the outcome depend on theta breaks the invariance."""
    rng = np.random.default_rng(0)
    eps = np.tile(np.round(np.arange(0.5, 1.51, 0.1), 2), 10)
    g = np.repeat(np.arange(10), 11)
    noise = lambda: 0.01 * rng.normal(size=len(eps))
    r = {"base": np.minimum(eps, 1.0) + noise(), "sat_head": eps + noise(), "ctl": np.minimum(eps, 1.0) + noise()}
    lc = {"predict": "saturation", "fit_max": 1.0}
    ko = L4.knockout_scores(r, eps, eps, g, lc, "ladder", ("ctl",), 0.1, 200)
    assert ko["base"]["creation_rate"] == 0 and abs(ko["base"]["slope_above_1"]) < 0.05
    assert ko["sat_head"]["creation_rate"] > 0.3 and ko["sat_head"]["slope_above_1"] == pytest.approx(1.0, abs=0.05)
    assert ko["sat_head"]["creation_effect"] > 0.3 and ko["sat_head"]["passes"] and "passes" not in ko["ctl"]
    assert ko["base"]["dependence"] == pytest.approx(1.0, abs=0.05)
    th = np.repeat(rng.normal(size=20), 2)
    gg = np.repeat(np.arange(20), 2)
    r0 = {"base": 0.01 * rng.normal(size=40), "leak": th + 0.01 * rng.normal(size=40), "ctl": 0.01 * rng.normal(size=40)}
    k0 = L4.knockout_scores(r0, th, th, gg, {"predict": 0.0}, "across", ("ctl",), n_boot=200, null=True)
    assert k0["leak"]["effect"] > 0.9 and k0["leak"]["passes"] and abs(k0["ctl"]["effect"]) < 0.05


def test_das_recovers_a_planted_2d_subspace_and_iia_grows_with_k(request):
    """DAS on synthetic activations: theta lives on a circular arc (1.8 pi) in a planted 2-D subspace U (no single
    direction carries it monotonically), plus distractor dimensions. With the frozen theta readout (early-stopped,
    weight-decayed MLP), interchanging k = 1 rotated dimension transfers theta only partly, k = 2 (the planted
    dimension) fully, and k = 4 adds nothing; the learned 2-D subspace is U (principal cosines ~ 1)."""
    import torch
    n0 = torch.get_num_threads()                       # pinned thread count, restored for the tests after this one
    request.addfinalizer(lambda: torch.set_num_threads(n0))
    torch.set_num_threads(4)
    rng = np.random.default_rng(0)
    n, D = 240, 12
    th = rng.uniform(0, 1, n)
    U = np.linalg.qr(rng.normal(size=(D, 2)))[0]
    phi = 1.8 * np.pi * th
    X = 2 * np.c_[np.cos(phi), np.sin(phi)] @ U.T + 0.02 * rng.normal(size=(n, D))
    P = np.linalg.qr(np.c_[U, rng.normal(size=(D, 3))])[0][:, 2:]     # distractors orthogonal to U
    X += rng.normal(size=(n, 3)) @ P.T
    tr, te = np.arange(180), np.arange(180, n)
    f = L4.theta_decoder(X[tr], th[tr], hidden=32, pcs=12, steps=3000, lr=0.01, weight_decay=1e-2)
    assert f.train_r2 > 0.95 and f.val_r2 > 0.95
    sc = np.arange(n).astype(str)
    prs = lambda rows: L4._all_pairs(rows, th, sc, "across", 0.2, 800, np.random.default_rng(1))
    iia, W = {}, {}
    for k in (1, 2, 4):
        W[k] = L4.das_fit(X, th, prs(tr), k, f, steps=400, lr=0.05)
        iia[k] = L4.das_hits(X, th, prs(te), W[k], f).mean()
    assert iia[1] < iia[2] - 0.08 and iia[2] > 0.95 and iia[4] == pytest.approx(iia[2], abs=0.05)
    assert min(L4.principal_cosines(W[2], U)) > 0.95
    assert W[4].shape == (D, 4) and np.allclose(W[4].T @ W[4], np.eye(4), atol=1e-8)


def test_dit_component_patch_uses_the_paired_pass_value():
    """DiT component patching: the donor's component value comes from the paired pass at the SAME timestep with the SAME
    noise draw (the base's z_t with the donor's context, noised like the base's), not from an independent donor run;
    the base's per-token-frame pooled MLP output at its ROI becomes exactly that value at every step, other tokens are
    untouched, gated steps run unpatched, and the runner routes "pset" edits through it."""
    import torch

    class Blk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = torch.nn.Linear(3, 3).double()
            with torch.no_grad():
                self.mlp.weight.copy_(torch.as_tensor(np.random.default_rng(3).normal(size=(3, 3))))

        def forward(self, h):
            return h + self.mlp(h)
    m = _TinyLatentDiT([999.0, 700.0, 400.0, 100.0])
    m.tr.transformer_blocks = torch.nn.ModuleList([Blk(), torch.nn.Identity()])
    rng = np.random.default_rng(0)
    zb, zd = (torch.from_numpy(rng.normal(size=(1, 3, 3, 2, 2))) for _ in range(2))
    ib, idd = np.array([0, 5]), np.array([1, 4])                  # one token per context token-frame (0, 1)
    with pytest.raises(ValueError, match="token-frame"):
        with L4.patch_live_component(m, "snr-8/L0", "mlp", ib, np.array([1, 9]), zb, zd, 5, []):
            pass
    log, gate = [], (lambda t: float(t) != 400.0)
    with L4.patch_live_component(m, "snr-8/L0", "mlp", ib, idd, zb, zd, 5, log, gate), torch.no_grad():
        out = m.sample(zb)
    with torch.no_grad():
        plain = m.sample(zb)
    assert "_pass" not in vars(m) and [r["t"] for r in log] == [999.0, 700.0, 100.0]
    mlp = m.tr.transformer_blocks[0].mlp
    for t, (o, lat), (o0, _) in zip(m.ts, out, plain):
        zdon = lat.clone()
        zdon[:, :m.kt] = m.ctx_at(zd, t)                          # the paired pass: base z_t, donor context, same noise
        with torch.no_grad():
            hd, hb = m.tokens(zdon, t)[0], m.tokens(lat, t)[0]
            md, mb = mlp(hd).numpy(), mlp(hb).numpy()
        assert np.allclose(np.delete(o, ib, 0), np.delete(o0, ib, 0))
        for tok_b, tok_d in zip(ib, idd):                         # one ROI token per frame: set to the donor's value
            want = hb[tok_b].numpy() + (md[tok_d] if t != 400.0 else mb[tok_b])
            assert np.allclose(o[tok_b], want)
    # runner routing: a "pset" edit installs the paired-pass component patch
    m.token_grid = type("G", (), {"grid_hw": (2, 2)})()
    m.forecast = lambda x, k, seed=0, return_video=False: ({-1: torch.as_tensor(np.stack([v for v, _ in m.sample(x)]))}, None)
    ins = {0: zb, 1: zd}
    sites = {0: ib, 1: idd}
    r = L4.ModelRunner(m, lambda i: {"x": ins[i], "k": 5, "T": 3, "groups": {"site": sites[i]}},
                       L4.LocReadout(space="final", value=lambda i, z: {"r": float(z.sum())}), "roi_A",
                       "_unused", diffusion=True, cached_snrs=[-8.0], log=lambda *a: None)
    res = r._forward(0, [("pset", "snr-8/L0", "mlp", 1)])
    res0 = r._forward(0, [])
    assert res["r"] != pytest.approx(res0["r"])
