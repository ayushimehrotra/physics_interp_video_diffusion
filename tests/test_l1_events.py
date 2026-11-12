"""L1 on single-object exchange events (F8 pad ladder, F7 bounce) and the per-token-frame kinds (gravity, rolling,
friction, glide inertia, oscillation, statics, angular momentum, the F9 swing): run() end to end on fake caches
(clip_states patched: these tests do not depend on the labels format), token_states + ratio_package per kind on
synthetic kinematics (legal -> S ~ 1, the law's null -> S ~ 0), and the ratio scores."""
from __future__ import annotations

import json

import numpy as np
import pytest

from physlaw.eval import l1_forecast_laws as L1
from physlaw.eval import laws

LADDER = [round(0.5 + 0.1 * i, 1) for i in range(11)]        # eps0.5 ... eps1.5
G = np.array([0.0, 0.0, -9.81])


def _fake_states(labels, ratio_dt=None):
    f = labels["fake"]
    pre, post = {"v": np.array(f["v_pre"]), "p": np.zeros((len(f["v_pre"]), 3))}, {"v": np.array(f["v_post"])}
    post["p"] = pre["p"]
    ctx = {k: (np.array(v) if isinstance(v, list) else v) for k, v in f["ctx"].items()}
    info = {"objects": f["objects"], "event": ctx["event"], "family": labels["physlaw"]["family"],
            "twin": labels["physlaw"]["twin"], "legal": f["legal"], "law": f["law"]}
    if ctx["event"] in L1.ENERGY_EVENTS:
        info["eps"] = f["eps"]
        info["energy_legal"] = f["eps"] <= 1
    return pre, post, ctx, info


def _write(root, clips: dict, family):
    data, cache = root / "data", root / "cache"
    cache.mkdir(parents=True)
    recs = []
    for c, fake in clips.items():
        g, t = c.split("/")
        (data / c).mkdir(parents=True)
        (data / c / "labels.json").write_text(json.dumps({"physlaw": {"family": family, "twin": t}, "fake": fake}))
        recs.append({"clip": c, "group": g, "twin": t, "family": family, "file": f"{c}.npz"})
    (cache / "_index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    return data, cache


def _run(data, cache, monkeypatch, real, fc, mf=None, states_fn=None):
    from physlaw.config import load_config
    clips = sorted(json.loads(x)["clip"] for x in (cache / "_index.jsonl").read_text().splitlines())
    np.savez(cache / "_maskfeat_v2.npz", clips=np.array(clips), **(mf or {}))
    monkeypatch.setattr(L1, "clip_states", states_fn or _fake_states)
    monkeypatch.setattr(L1, "_load_spaces", lambda *a, **k: {"final": {"real": real, "fc": fc}})
    cfg = load_config("configs/eval/l1_forecast_laws.yaml")
    cfg["fold_seeds_l1"] = 1
    cfg["bootstrap"] = {"n": 50}
    cfg["selection"] = {"n_splits": 4}
    return L1.run(cache, data, cfg, {}, log=lambda *a: None), clips


TK = np.arange(0.0, 3.0, 1 / 12)                              # token-frame times (12 token-frames / s)


def _bounce_v(t, t_rel, te, eps, vx):
    """Ball held until t_rel, falls (contact at te), rebounds at eps x the impact speed until the second contact
    t2 = te + 2 eps (te - t_rel); afterwards bounces on at eps. Returns (V [T, 1, 3], t2)."""
    g, T = 9.81, te - t_rel
    vz = np.where(t < t_rel, 0.0, -g * (t - t_rel))
    t2 = te + 2 * eps * T
    vz = np.where(t >= te, eps * g * T - g * (t - te), vz)
    vz = np.where(t >= t2, eps * eps * g * T - g * (t - t2), vz)
    V = np.zeros((len(t), 1, 3))
    V[:, 0, 0] = np.where(t < t_rel, 0.0, vx)
    V[:, 0, 2] = vz
    return V, t2


def _pad_p(t, t_rel, te, eps, h, x0, U=0.6, D=0.5):
    """Block held at height h until t_rel, descends to the floor (D s), runs to the pad (te), climbs to eps^2 h at
    te + U (the apex: clean_until_s), slides back. Positions P [T, 1, 3]."""
    z = np.where(t < t_rel, h, np.where(t < t_rel + D, h * np.cos(0.5 * np.pi * (t - t_rel) / D) ** 2, 0.0))
    z = np.where(t >= te, eps ** 2 * h * np.sin(0.5 * np.pi * np.clip(t - te, 0, 2 * U) / U) ** 2, z)
    P = np.zeros((len(t), 1, 3))
    P[:, 0, 0], P[:, 0, 2] = x0 + 2.0 * t, 1.0 + z
    return P


def _event_ladder_run(tmp_path, monkeypatch, event, e_model, n_groups=30, seed=1):
    """run() on a per-token-frame energy ladder (F7 bounce: velocities; F8 pad: positions), features linear in the
    quantity + noise, one forecast per group (the model's own eps, or static: the last context frame repeated)."""
    rng = np.random.default_rng(seed)
    clips, Q, oks, oko = {}, {}, {}, {}
    for g in range(n_groups):
        t_rel, te = 0.3, 0.3 + rng.uniform(0.4, 0.6)
        vx, h, x0 = rng.uniform(-1, 1), rng.uniform(0.8, 2.0), rng.uniform(-1, 1)
        for e in LADDER:
            if event == "bounce":
                X, t_end = _bounce_v(TK, t_rel, te, e, vx)
                ctx = {"event": "bounce", "g": G.tolist(), "event_time_s": te, "clean_until_s": t_end}
            else:
                X, t_end = _pad_p(TK, t_rel, te, e, h, x0), te + 0.6
                ctx = {"event": "pad", "g": G.tolist(), "event_time_s": te, "clean_until_s": t_end}
            c = f"g{g:03d}/eps{e}"
            clips[c] = {"v_pre": [[0.0, 0.0, 0.0]], "v_post": [[0.0, 0.0, 0.0]], "eps": e, "ctx": ctx,
                        "objects": ["A"], "legal": e <= 1, "law": event}
            Q[c] = X
            oks[c] = (TK >= t_rel - 1e-9) & (TK <= t_end + 1e-9)      # free: after the release, inside the clean window
            oko[c] = TK >= t_rel - 1e-9
            if e == 1.0:
                Q[f"g{g:03d}/model"] = (_bounce_v(TK, t_rel, te, e_model, vx)[0] if event == "bounce"
                                        else _pad_p(TK, t_rel, te, e_model, h, x0)) if e_model != "static" else None
    keys = sorted(clips)
    P = rng.normal(size=(3, 16))
    emb = lambda X: X.reshape(len(X), -1) @ P + 0.02 * rng.normal(size=(len(X), 16))
    k0 = {c: int(np.searchsorted(TK, clips[c]["ctx"]["event_time_s"])) for c in keys}
    real = [emb(Q[c]) for c in keys]
    fc = []
    for c, x in zip(keys, real):
        m = Q[c.split("/")[0] + "/model"]
        fc.append(np.repeat(x[k0[c] - 1][None], len(TK) - k0[c], 0) if m is None else emb(m[k0[c]:]))
    key = "V" if event == "bounce" else "P"
    z = [np.zeros_like(Q[c]) for c in keys]
    fr = {"t": [TK] * len(keys), "phase": [np.where(TK < clips[c]["ctx"]["event_time_s"], "pre", "post") for c in keys],
          "V": z, "W": z, "P": z, "ok": [oks[c] for c in keys], "t_fc": [TK[k0[c]:] for c in keys],
          "ok_fc": [oks[c][k0[c]:] for c in keys], "ok_open_fc": [oko[c][k0[c]:] for c in keys]}
    fr[key] = [Q[c] for c in keys]
    monkeypatch.setattr(L1, "load_token_frames", lambda *a: ({"final": {"real": real, "fc": fc}}, fr))
    d, c_ = _write(tmp_path, clips, "f7" if event == "bounce" else "f8")
    res, _ = _run(d, c_, monkeypatch, {}, {})
    return res


@pytest.mark.parametrize("event", ["pad", "bounce"])
@pytest.mark.parametrize("e_model", [1.4, 0.8, "static"])
def test_single_object_ladder_end_to_end(tmp_path, monkeypatch, event, e_model):
    """H3: the F7 bounce and the F8 pad are per-token-frame energy kinds (bounce: velocities extrapolated to the
    contact by line fits over the free token-frames on each side, held frames excluded, the post window ending at the
    second contact; pad: apex heights). One forecast per group (shared by its 11 rungs): the floor reads eps, a model
    at eps 1.4 creates energy, one at 0.8 does not, a static forecast is gated out; the H4 controls are scored."""
    res = _event_ladder_run(tmp_path, monkeypatch, event, e_model)
    assert res["laws"] == [event]
    v = res["variants"][f"{event}|final|frame"]
    e = v["energy"]
    assert e["status"] == "measured" and e["floor_check"]["slope"] == pytest.approx(1, abs=0.1)
    assert e["floor"]["creation_rate"] < 0.1 and e["floor"]["lower_tail_rate"] < 0.1
    assert set(e["observers"]) == set(L1.CONTROLS) and "verdict" in e
    assert e["observers"]["ctl_static"]["n"] == 0                       # the last context frame repeated: no event
    if e_model == "static":
        assert e["forecast"]["interaction_rate"] < 0.1 and e["forecast"]["creation_rate_sensitivity"] > 0.9   # M1
        return
    assert e["forecast"]["eps_median"] == pytest.approx(e_model, abs=0.08)
    assert e["forecast"]["pass_share"] > 0.9
    if e_model > 1:
        assert e["forecast"]["creation_rate"] > 0.8
    else:
        assert e["forecast"]["creation_rate"] < 0.1 and e["forecast"]["lower_tail_rate"] < 0.1
    md = L1.table(res)
    assert "creation rate" in md and "necessary, not sufficient" in md
    json.dumps(res, default=float)


def test_bounce_window_averages_vs_event_extrapolation():
    """The audit's bounce table (H3): window-averaged velocities with 0.3 s of held frames in the pre window inflate
    eps (0.6 -> 1.05, 1.0 -> 1.75, 1.2 -> 2.10) and a post window running to the second floor contact averages up and
    down to ~0; line fits on each side over the free frames, extrapolated to the contact (L1's bounce kind), return the
    true eps exactly."""
    t = np.arange(0.0, 4.0, 1 / 240)
    t_rel, te = 0.3, 0.7                                           # 0.3 s held, 0.4 s fall
    for eps, audit in ((0.6, 1.05), (1.0, 1.75), (1.2, 2.10)):
        V, t2 = _bounce_v(t, t_rel, te, eps, 0.0)
        vz = V[:, 0, 2]
        apex = te + eps * (te - t_rel)
        held = vz[t < te].mean()                                   # pre window from frame 0 (held frames in it)
        free = vz[(t >= t_rel) & (t < te)].mean()                  # pre window from the release
        assert -vz[(t >= te) & (t < apex)].mean() / free == pytest.approx(eps, abs=0.03)   # to the apex: right
        assert -vz[(t >= te) & (t < apex)].mean() / held == pytest.approx(audit, abs=0.05)  # held frames: inflated
        assert abs(vz[(t >= te) & (t < t2)].mean() / free) < 0.02                   # to the 2nd contact: ~0
        ph = np.where(t < te, "pre", "post")
        ok = (t >= t_rel) & (t < t2)
        k = L1.token_states("bounce", {"g": G}, t, {"v": V}, ph, t, {"v": V}, te, t2, ok, ph, ok)
        assert k["eps"] == pytest.approx(eps, abs=1e-6)
        k_bad = L1.token_states("bounce", {"g": G}, t, {"v": V}, ph, t, {"v": V}, te, t2, None, ph, None)
        assert abs(k_bad["eps"] - eps) > 0.1                       # held frames in the pre fit: biased


def test_f7_bounce_labels_give_the_true_eps():
    """On generated F7 bounce labels (ball held in the hand before the release), the TRUE row of the bounce kind (the
    labelled velocities through a V-JEPA-2-like token grid, free token-frames only) recovers each rung's eps."""
    from physlaw.models.base import token_spec
    from physlaw.tokens.project import token_frame_phase
    spec = token_spec({"input": {"resolution": 256}, "token_grid": {"patch": 16, "tubelet": 2}})
    n = 0
    for scene in _f7_labels("bounce", 1):
        for twin, lab in scene.items():
            pl = lab["physlaw"]
            k = int(pl["first_post_contact_frame"])
            frames = list(range(max(k - 64, 0), k + 64, 2))
            t, V, _, _, ok = L1.token_frame_targets(lab, frames, spec, ["A"])
            ph = np.array(token_frame_phase(frames, spec, k))
            kin = L1.token_states("bounce", {"g": G}, t, {"v": V}, ph, t, {"v": V}, float(pl["event_time_s"]),
                                  float(pl["clean_until_s"]), ok, ph, ok)
            if kin is None:
                continue
            assert kin["eps"] == pytest.approx(float(pl["epsilon"]), abs=1e-3), twin
            n += 1
    assert n >= 10


def test_negative_controls_catch_a_probe_that_fills_in(tmp_path):
    """H4: features carry only the time since release (and a per-set horizontal speed), never the fall itself. A
    probe trained on real legal falls maps time to v = −g tau: its conditional mean IS the law, so a forecast that
    knows nothing about this clip's fall scores g_ratio S ~ 1, and so do another set's forecast and another set's
    real post-event rows decoded by the same instrument: the verdict refuses the claim ("probe fills in")."""
    rng = np.random.default_rng(5)
    t = np.arange(0.0, 1.6, 1 / 12)
    te = 0.8
    clips, states, X, fc, V = [], {}, [], [], []
    for g in range(30):
        vx = rng.uniform(-1, 1)
        c = f"g{g:03d}/legal"
        clips.append(c)
        v = np.zeros((len(t), 1, 3))
        v[:, 0, 0], v[:, 0, 2] = vx, -9.81 * t
        V.append(v)
        feat = lambda tt: np.stack([tt, np.full(len(tt), vx), np.ones(len(tt))], 1) @ np.array(
            [[1.0, 0.3, -0.2, 0.5], [0.2, 1.0, 0.4, -0.3], [0.1, 0.1, 0.1, 0.1]]) + 0.01 * rng.normal(size=(len(tt), 4))
        X.append(feat(t))
        fc.append(feat(t[t >= te]))
        ctx = {"g": G, "event_time_s": te, "clean_until_s": 1.55, "m": np.array([1.0]), "event": "none"}
        states[c] = ({"v": v[0], "p": np.zeros((1, 3))}, {"v": v[-1]}, ctx,
                     {"legal": True, "objects": ["A"], "family": "f7", "twin": "legal", "event": "none"})
    groups = np.array([c.split("/")[0] for c in clips])
    z = [np.zeros_like(v) for v in V]
    fr = {"t": [t] * 30, "phase": [np.where(t < te, "pre", "post")] * 30, "V": V, "W": z, "P": z,
          "t_fc": [t[t >= te]] * 30}
    out = L1.kinematics_variants("gravity", states, clips, {"final": {"real": X, "fc": fc}}, fr, groups,
                                 L1.KIN_DEFAULTS, 5, 1, L1.DEFAULT_ALPHAS, 200, np.random.default_rng(0))
    r = out["final"]["ratio"]["g_ratio"]
    assert r["forecast"]["S"] > 0.9 and r["ctl_other_forecast"]["S"] > 0.9 and r["ctl_other_real"]["S"] > 0.9
    assert r["ctl_static"]["S"] < 0.2                     # the last context frame repeated does not fall
    assert not r["verdict"]["claim_allowed"] and r["verdict"]["verdict"].startswith("probe fills in")


def test_verdict_and_sensitivity():
    v = L1.verdict(0.9, {"ctl_static": float("nan"), "ctl_other_forecast": 0.1, "ctl_other_real": -0.4})
    assert v["claim_allowed"] and all(v["beats"].values())
    v = L1.verdict(0.9, {"ctl_static": 0.95, "ctl_other_forecast": 0.1, "ctl_other_real": -0.4}, mask_only=0.2)
    assert not v["claim_allowed"] and "ctl_static" in v["verdict"]
    v = L1.verdict(0.9, {c: 0.0 for c in L1.CONTROLS}, mask_only=0.95)
    assert not v["claim_allowed"] and v["verdict"].startswith("mask placement")
    v = L1.verdict(0.02, {c: 0.02 for c in L1.CONTROLS}, higher_better=False)      # creation rates tie: no claim
    assert not v["claim_allowed"]
    assert L1._sensitivity(0.8, 0.5) == pytest.approx(0.5 * 0.8 - 0.5)            # removed clips count as S = −1
    assert L1._sensitivity(0.1, 0.5, 1.0) == pytest.approx(0.55)                   # ... as creating energy


def test_m2_nonpositive_log_statistics_score_the_floor():
    """M2: in the log form a decoded T <= 0 (or inf / NaN where the clip's statistic is defined) has no logarithm and
    scores the floor −1 instead of leaving the median; S is clipped below at −1; undefined clips stay out."""
    assert laws.ratio_score(-0.3, 1.0, 0.5) != laws.ratio_score(-0.3, 1.0, 0.5)        # default: NaN
    assert laws.ratio_score(-0.3, 1.0, 0.5, floor=-1.0) == -1.0
    assert laws.ratio_score(1e-9, 1.0, 0.5, floor=-1.0) == -1.0                        # clipped
    g = np.arange(5)
    T = np.array([1.0, 1.0, -0.4, np.inf, np.nan])
    r = L1.ratio_scores({"forecast": T}, 1.0, np.full(5, 0.5), g, defined=[True, True, True, True, False])
    assert r["form"] == "log" and r["forecast"]["n_floored"] == 2
    assert r["forecast"]["S"] == pytest.approx(0.0)          # median of (1, 1, −1, −1): the worst forecasts count
    r = L1.ratio_scores({"forecast": np.array([1.0, np.nan])}, 1.0, np.zeros(2), [0, 1])
    assert r["form"] == "linear" and r["forecast"]["S"] == pytest.approx(0.0)        # (1 + (−1)) / 2


def test_n1_pass_through_is_gated_and_mass_free_checks_are_sanity_only():
    """N1: a pass-through forecast (balls keep their pre velocities) has eps = −1, which satisfies eps <= 1, but moves
    no momentum and never passes the interaction gate; the mass-free checks equal 1 for a momentum-edited outcome too
    (drift along n), so they are labelled sanity checks, never a law score."""
    from physlaw.data import collision_physics as cp
    st = cp.PreState(0.5, 2.0, np.array([2.0, 0.3]), np.array([0.0, 0.0]), np.array([1.0, 0.0]))
    pre = {"v": np.stack([st.v_a, st.v_b]), "p": np.array([[-0.1, 0.0], [0.1, 0.0]])}
    legal = pre["v"] + cp.legal_outcome(st, 1.0).reshape(2, 2)
    ctx = {"m": np.array([0.5, 2.0]), "n": st.n, "event": "collision"}
    states = {"g0/conserve_e1": (pre, {"v": legal, "p": pre["p"]}, ctx, {})}
    dec = {"floor": legal[None], "forecast": pre["v"][None]}
    gates = L1.collision_gates(states, ["g0/conserve_e1"], dec, L1.MOM_DEFAULTS)
    assert gates["floor"][0] and not gates["forecast"][0]
    assert L1._eps(pre["v"], pre["v"], ctx) == pytest.approx(-1.0)
    drift = legal - 0.3 * np.stack([st.n, st.n])                  # momentum created along n
    mf = laws.mass_free_momentum(pre, {"v": drift}, ctx)
    assert mf["antiparallel"] == pytest.approx(1) and mf["normal_A"] == pytest.approx(1)
    assert "cannot detect the drift edit" in L1.SANITY_NOTE


def test_f3_hidden_mass_has_no_momentum_score(tmp_path, monkeypatch):
    """H1: F3's ratios share their pre-event state (f3_twins: same pre velocities, masses by density), so one
    forecast serves all five ratios: L1 does not score S_mom on F3 (the json says why) but still scores energy."""
    from physlaw.data import collision_physics as cp
    from physlaw.data.twins import f3_twins
    tw = f3_twins(cp.PreState(1.0, 1.0, np.array([2.0, 0.2]), np.array([0.0, 0.0]), np.array([1.0, 0.0])), 1.0)
    gm = {round(np.sqrt(t.record["m_A"] * t.record["m_B"]), 9) for t in tw.values()}
    assert len(gm) == 1 and len({round(t.record["mass_ratio"], 2) for t in tw.values()}) == 5   # by density only
    rng = np.random.default_rng(0)
    clips = {}
    for g in range(20):
        n = np.array([1.0, 0.0])
        va = np.array([rng.uniform(1, 3), rng.uniform(-0.5, 0.5)])
        for name, t in f3_twins(cp.PreState(1.0, 1.0, va, np.zeros(2), n), 1.0).items():
            clips[f"g{g:03d}/{name}"] = {"v_pre": [list(va) + [0.0], [0.0, 0.0, 0.0]],
                                         "v_post": [list(t.v_a) + [0.0], list(t.v_b) + [0.0]], "eps": 1.0,
                                         "ctx": {"event": "collision", "m": [t.record["m_A"], t.record["m_B"]],
                                                 "n": [1.0, 0.0, 0.0]},
                                         "objects": ["A", "B"], "legal": True, "law": "energy"}
    keys = sorted(clips)
    W = rng.normal(size=(6, 12))
    emb = lambda v: np.array(v).reshape(-1) @ W + 0.01 * rng.normal(size=12)
    real = {"frame.pre": np.array([emb(clips[c]["v_pre"]) for c in keys]),
            "frame.post": np.array([emb(clips[c]["v_post"]) for c in keys])}
    fc = {"frame.post": np.array([emb(clips[c.split("/")[0] + "/ratio_1"]["v_post"]) for c in keys])}
    data, cache = _write(tmp_path, clips, "f3")

    def fake(labels, ratio_dt=None):
        pre, post, ctx, info = _fake_states(labels)
        pre["v"], post["v"] = pre["v"][:, :2], post["v"][:, :2]
        ctx["n"] = ctx["n"][:2]
        pre["p"] = post["p"] = np.array([[-0.1, 0.0], [0.1, 0.0]])
        return pre, post, ctx, {**info, "momentum_legal": True, "energy_legal": True}
    res, _ = _run(data, cache, monkeypatch, real, fc, states_fn=fake)
    v = res["variants"]["energy|final|frame"]
    assert v["momentum"]["status"] == "not scored" and "L2 / L4 family" in v["momentum"]["note"]
    assert "forecast" in v["energy"] and "S" not in v["momentum"]
    assert "F3" in L1.table(res)


def test_ratio_scores_one_for_legal_zero_for_null():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(20), 3)
    t_null = rng.uniform(0.3, 0.7, g.size)                  # a positive null ratio: log form
    r = L1.ratio_scores({"forecast": np.ones(g.size), "floor": t_null}, 1.0, t_null, g, n_boot=50, rng=rng)
    assert r["form"] == "log" and r["forecast"]["S"] == pytest.approx(1) and r["floor"]["S"] == pytest.approx(0)
    assert r["forecast"]["S_ci95"] == pytest.approx([1, 1])
    z = np.zeros(g.size)                                    # a null of 0 (no-gravity g_ratio, rolling order): linear
    r = L1.ratio_scores({"forecast": np.ones(g.size), "floor": z, "half": np.full(g.size, 0.5)}, 1.0, z, g)
    assert r["form"] == "linear" and (r["forecast"]["S"], r["floor"]["S"], r["half"]["S"]) == pytest.approx((1, 0, 0.5))


K_ROLL, R_ROLL, TE = np.array([0.5, 0.905]), np.array([0.4, 0.4]), 1.0
T_TOK = np.arange(16) * 0.25                                 # token-frame times; release at TE


def _rolling_clip(rng, legal=True, th=None):
    th = rng.uniform(0.2, 0.45) if th is None else th
    ctx = {"event": "none", "k": K_ROLL.tolist(), "g_mag": 9.81, "theta": th, "dt": 0.5, "r": R_ROLL.tolist(),
           "g": G.tolist(), "event_time_s": TE, "clean_until_s": 3.6, "m": [1.0, 1.0]}
    return {"v_pre": [[0.0, 0.0, 0.0]] * 2, "v_post": [[0.0, 0.0, 0.0]] * 2, "ctx": ctx, "objects": ["A", "B"],
            "legal": legal, "law": "rolling"}


def _roll_kin(th, a_of_k, spin=1.0):
    """Per token-frame V, W [T, 2, 3] of two cylinders released from rest at TE down the incline (x, z plane)."""
    u = np.array([np.cos(th), 0.0, -np.sin(th)])
    s = np.clip(T_TOK - TE, 0, None)[:, None] * a_of_k(9.81 * np.sin(th))[None]       # speed [T, 2]
    V = s[..., None] * u
    W = np.zeros_like(V)
    W[..., 1] = spin * s / R_ROLL
    return V, W


def test_rolling_order_and_slip_from_token_frames(tmp_path, monkeypatch):
    """run() decodes per-token-frame velocity AND angular velocity with one calibrated instrument each, fits lines over
    the windows: a law-following forecast scores order and slip S ~ 1 (as the floor and the labelled truth); one that
    ignores inertia (equal accelerations, no spin) scores ~ 0 (linear form: the null statistics are 0). The real rows
    include each group's violating twins ("tie": equal accelerations; "hollow_first": swapped), as F10 does: without it v_A and v_B would be
    collinear in every training row and no instrument could read them apart."""
    rng = np.random.default_rng(2)
    clips = {}
    for g in range(40):
        clips[f"g{g:03d}/legal"] = _rolling_clip(rng)
        for t in ("tie", "hollow_first"):
            clips[f"g{g:03d}/{t}"] = _rolling_clip(rng, False, clips[f"g{g:03d}/legal"]["ctx"]["theta"])
    keys = sorted(clips)
    P = rng.normal(size=(12, 32))
    emb = lambda V, W: np.concatenate([V.reshape(len(V), -1), W.reshape(len(W), -1)], 1) @ P + 0.01 * rng.normal(size=(len(V), 32))
    law = lambda g: g / (1 + K_ROLL)
    ignored = lambda g: np.full(2, g / 1.5)
    phase = np.where(T_TOK < TE, "pre", "post")
    out = {}
    for model, a_of_k, spin in (("law", law, 1.0), ("inertia_ignored", ignored, 0.0)):
        fr = {"t": [], "phase": [], "V": [], "W": [], "t_fc": []}
        real, fc = [], []
        for c in keys:
            th = clips[c]["ctx"]["theta"]
            V, W = _roll_kin(th, {"legal": law, "tie": ignored, "hollow_first": lambda g: law(g)[::-1]}[c.split("/")[1]])
            fr["t"].append(T_TOK); fr["phase"].append(phase); fr["V"].append(V); fr["W"].append(W)
            fr["t_fc"].append(T_TOK[4:])
            real.append(emb(V, W))
            Vf, Wf = _roll_kin(th, a_of_k, spin)
            fc.append(emb(Vf[4:], Wf[4:]))
        monkeypatch.setattr(L1, "load_token_frames", lambda *a, fr=fr, real=real, fc=fc: ({"final": {"real": real, "fc": fc}}, fr))
        d, c_ = _write(tmp_path / model, clips, "f10")
        res, _ = _run(d, c_, monkeypatch, {}, {})
        v = res["variants"]["rolling|final|frame"]
        out[model] = v["ratio"]
        assert v["instrument"] == "token_frames" and v["probe_r2"]["A"]["vw"] > 0.95     # one (v, w) instrument (M4)
        sv = v["selectivity"]["A"]["vw"]["scene"]  # partialled within (twin slot, token-frame index): the speed's time
        assert set(v["selectivity"]["A"]["vw"]) == {"scene", "object"}   # course no longer fits the control, so the
        assert sv["r2"] > 0.5 and sv["control_r2"] < 0.1 and sv["selectivity"] > sv["r2"] - 0.1   # quantity is selective
    for st in ("order", "slip_s", "slip_h"):
        assert out["law"][st]["form"] == "linear"
        assert out["law"][st]["true"]["S"] == pytest.approx(1, abs=0.02)
        assert out["law"][st]["floor"]["S"] > 0.9 and out["law"][st]["forecast"]["S"] > 0.9
        assert abs(out["inertia_ignored"][st]["forecast"]["S"]) < 0.15
    assert "Ratio laws" in L1.table(res)


def test_token_frames_need_the_stage4_pools(tmp_path):
    np.savez(tmp_path / "x.npz", pooled=np.zeros((1, 1, 2)), layers=np.array(["-1"]))
    with pytest.raises(RuntimeError, match="re-run stage 4"):
        L1.load_token_frames(tmp_path, [{"file": "x.npz"}], ["g/x"], {}, {"input": {"resolution": 32},
                             "token_grid": {"patch": 16, "tubelet": 2}}, {"g/x": "0"}, ["A"])


def test_gravity_kinematics_g_ratio_and_mass_ratio():
    """clip_kinematics + laws.gravity: a drop pair under g after a free-flight pre window scores g_ratio = 1 and
    mass_ratio = 1; the no-gravity null 0 and the Aristotelian mass null m_heavy / m_light."""
    t = np.arange(12) * 0.1
    V = np.zeros((12, 2, 3))
    V[..., 2] = -9.81 * t[:, None] + np.array([1.0, 0.5])
    ph = np.where(t < 0.55, "pre", "post")
    k = L1.clip_kinematics(t, V, ph, t, V, None, 0.55, np.inf, None, ph)
    ctx = {"g": G, "m": np.array([1.0, 3.0]), "dt": 0.0}
    st = laws.gravity.stats(k["pre"], k["post"], ctx)
    assert st["g_ratio"] == pytest.approx(1) and st["mass_ratio"] == pytest.approx(1)
    nul = L1.null_stats(laws.gravity, k["pre"], ctx)
    assert nul["g_ratio"] == pytest.approx(0) and nul["mass_ratio"] == pytest.approx(3)
    assert L1.clip_kinematics(t[:4], V[:4], ph[:4], t[:4], V[:4], None, 0.55, np.inf) is None


# --------------------------------------------------------------------------- the new per-token-frame kinds

TT = np.arange(0.0, 6.0, 0.125)                              # token-frame times
TE2 = 2.5
PH = np.where(TT < TE2, "pre", "post")


def test_ratio_kind_dispatch():
    assert [L1.ratio_kind(f, s) for f, s in (("f6", "slide"), ("f6", "pair"), ("f6", "glide"), ("f7", "drop"),
                                              ("f9", "pair"), ("f9", "swing"), ("f10", ""), ("f11", "support"),
                                              ("f12", "spin"))] == \
        ["friction", "friction", "inertia", "gravity", "oscillation", "swing", "rolling", "statics", "angular_momentum"]
    assert L1.ratio_kind("f1") is None and set(L1.RATIO_LAW.values()) <= set(L1.KINDS)


def _package(kind, clips, Q_true, Q_fc, ctxs, legal=None, pre_p=None):
    """ratio_package on synthetic token-frame quantities: floor = true, forecast = Q_fc (its post rows)."""
    states = {}
    for c, ctx in ctxs.items():
        n = Q_true[c][next(iter(Q_true[c]))].shape[1]
        pre = {"v": np.zeros((n, 3)), **({"p": pre_p[c]} if pre_p else {})}
        states[c] = (pre, {"v": np.zeros((n, 3))}, ctx, {"legal": True if legal is None else legal[c],
                                                         "objects": list("AB"[:n])})
    kin = lambda c, Q, ph: L1.token_states(kind, {**states[c][2], "objects": states[c][3]["objects"]}, TT, Q_true[c],
                                           PH, TT, Q, TE2, float(states[c][2].get("clean_until_s", 1e9)), None, ph)
    cl = list(ctxs)
    k = {"true": [kin(c, Q_true[c], PH) for c in cl], "floor": [kin(c, Q_true[c], PH) for c in cl],
         "forecast": [kin(c, Q_fc[c], None) for c in cl]}
    assert all(x is not None for v in k.values() for x in v)
    return L1.ratio_package(kind, states, cl, k, 0, np.random.default_rng(0))


def _v_of(speed, u):
    return speed[:, :, None] * np.asarray(u, float)


def _slide_v(v_e, a, u, kind="legal", k=1.0):
    """Token-frame velocities [T, n, 3] of slides decelerating at a (speed v_e at TE2) along u; after TE2: legal
    (deceleration k a until rest), reverse (-a through v = 0), viscous (v_e exp(-a tau / v_e))."""
    v_e, a = np.atleast_1d(v_e).astype(float), np.atleast_1d(a).astype(float)
    tau = (TT - TE2)[:, None]
    pre = v_e - a * tau
    post = {"legal": np.maximum(v_e - k * a * tau, 0), "reverse": v_e - a * tau,
            "viscous": v_e * np.exp(-a * np.clip(tau, 0, None) / v_e)}[kind]
    return _v_of(np.where(tau <= 0, pre, post), u)


def test_friction_kind_windows():
    """Coulomb slides (one deceleration for both masses, stop at TE2 + v_e / a): decel_const, mass_ratio = 1 and
    stop_hold = 0 (S 1); a forecast obeying the Stokes null (one drag constant: a ~ v / m) scores S ~ 0 on decel_const and
    mass_ratio and stop_hold > 0; the naive reverse forecast keeps decel_const (S 1) but fails stop_hold."""
    m, v_e, a, u = np.array([1.0, 3.0]), 1.5, 1.0, [0.8, 0.6, 0.0]
    legal = _slide_v([v_e, v_e], [a, a], u)
    lam = (a / v_e) * m.mean() / m                            # c fitted to the pre deceleration, a_i = c v_i / m_i
    tau = np.clip(TT - TE2, 0, None)[:, None]
    visc = np.where((TT <= TE2)[:, None, None], legal, _v_of(v_e * np.exp(-lam[None] * tau), u))
    ctx = {"m": m, "event_time_s": TE2, "t_stop_s": TE2 + v_e / a, "clean_until_s": TT[-1]}
    kin_t = L1.token_states("friction", ctx, TT, {"v": legal}, PH, TT, {"v": legal}, TE2, TT[-1], None, PH)
    assert kin_t["dt"] > 0 and np.allclose(kin_t["pre"]["a"] @ np.array(u), -a) and np.allclose(kin_t["post"]["v_stop"], 0)
    r = _package("friction", ["g0/legal"], {"g0/legal": {"v": legal}}, {"g0/legal": {"v": visc}}, {"g0/legal": ctx})
    for st in ("decel_const", "mass_ratio", "stop_hold"):
        assert r[st]["true"]["S"] == pytest.approx(1) and r[st]["floor"]["S"] == pytest.approx(1)
        assert r[st]["forecast"]["S"] < 0.5
    assert r["stop_hold"]["form"] == "linear" and r["decel_const"]["form"] == "log"
    rev = _slide_v([v_e, v_e], [a, a], u, "reverse")
    r = _package("friction", ["g0/legal"], {"g0/legal": {"v": legal}}, {"g0/legal": {"v": rev}}, {"g0/legal": ctx})
    assert r["decel_const"]["forecast"]["S"] == pytest.approx(1, abs=1e-6) and r["stop_hold"]["forecast"]["S"] < 0


def test_glide_inertia_unit_form():
    """Newton I on a glide: the same velocity scores 1; |v'| x 1.3 scores 0.7 (1 − |x − 1|); a stop scores 0."""
    v = np.tile([[[1.5, 0.5, 0.0]]], (len(TT), 1, 1))
    res = {}
    for name, f in (("same", 1.0), ("fast", 1.3), ("stop", 0.0)):
        fc = v * np.where(TT >= TE2, f, 1.0)[:, None, None]
        res[name] = _package("inertia", ["g0/legal"], {"g0/legal": {"v": v}}, {"g0/legal": {"v": fc}},
                             {"g0/legal": {"force_free": True, "event_time_s": TE2}})
    assert res["same"]["inertia_speed"]["forecast"]["S"] == pytest.approx(1)
    assert res["same"]["inertia_turn"]["forecast"]["S"] == pytest.approx(1)
    assert res["fast"]["inertia_speed"]["forecast"]["S"] == pytest.approx(0.7)
    assert res["stop"]["inertia_speed"]["forecast"]["S"] == pytest.approx(0)
    assert res["same"]["inertia_speed"]["form"] == "unit"


def _bobs(T_post):
    """Horizontal offsets / rates of two bobs (pivots at x = -1, 1) swinging with period 2 s, after TE2 with periods
    T_post (per bob)."""
    P, V = np.zeros((len(TT), 2, 3)), np.zeros((len(TT), 2, 3))
    for o, (x0, Tp) in enumerate(zip((-1.0, 1.0), T_post)):
        ph = np.where(TT < TE2, 2 * np.pi * (TT - 0.3) / 2.0, 2 * np.pi * (TE2 - 0.3) / 2.0 + 2 * np.pi * (TT - TE2) / Tp)
        w = np.where(TT < TE2, 2 * np.pi / 2.0, 2 * np.pi / Tp)
        P[:, o, 0] = x0 + 0.4 * np.sin(ph)
        V[:, o, 0] = 0.4 * w * np.cos(ph)
    return {"p": P, "v": V}


def test_oscillation_kind_periods_from_zero_crossings():
    """Periods from decoded offsets: legal pair S 1; a forecast obeying "heavier swings faster" (T ~ m^-1/2) S ~ 0."""
    m = np.array([1.0, 4.0])
    ctx = {"m": m, "pivots": {"A": [-1.0, 0.0, 5.0], "B": [1.0, 0.0, 5.0]}, "event_time_s": TE2}
    legal = _bobs((2.0, 2.0))
    kin = L1.token_states("oscillation", {**ctx, "objects": ["A", "B"]}, TT, legal, PH, TT, legal, TE2, 9.0, None, PH)
    assert np.allclose(kin["pre"]["T"], 2.0, rtol=0.02) and np.allclose(kin["post"]["T"], 2.0, rtol=0.02)
    r = _package("oscillation", ["g0/legal"], {"g0/legal": legal}, {"g0/legal": _bobs((2.0, 1.0))}, {"g0/legal": ctx})
    for st in ("period_ratio", "mass_period_ratio"):
        assert r[st]["form"] == "log" and r[st]["true"]["S"] == pytest.approx(1, abs=0.05)
    assert r["mass_period_ratio"]["forecast"]["S"] == pytest.approx(0, abs=0.1)


def test_statics_kind_tip_accuracy_and_alpha():
    """Tip iff x > 1: the legal motion scores tip_consistency 1 (accuracy) and alpha_ratio 1 (against the labelled
    clip's own windowed value); a forecast where nothing tips scores accuracy 0.5 over x = 0.75 / 1.25 and alpha 0."""
    ctx_of = lambda x: {"overhang_x": x, "aspect_ba": 0.5, "g_clip": 9.81, "half_length": 0.8, "event_time_s": TE2,
                        "tip_axis": [0.0, -1.0, 0.0]}
    al = 9.81 * float(__import__("physlaw.eval.laws", fromlist=["tip_rate"]).tip_rate(1.25, 0.5)) / 0.8
    W = lambda tips: np.stack([np.zeros(len(TT)), -al * np.clip(TT - TE2, 0, None) * tips, np.zeros(len(TT))], -1)[:, None]
    cl = ["g0/stay", "g1/tip"]
    Q = {"g0/stay": {"w": W(0.0)}, "g1/tip": {"w": W(1.0)}}
    r = _package("statics", cl, Q, {c: {"w": W(0.0)} for c in cl}, {"g0/stay": ctx_of(0.75), "g1/tip": ctx_of(1.25)})
    assert r["tip_consistency"]["agg"] == "mean" and r["tip_consistency"]["true"]["S"] == 1.0
    assert r["tip_consistency"]["forecast"]["S"] == pytest.approx(0.5)
    assert r["alpha_ratio"]["true"]["S"] == pytest.approx(1) and r["alpha_ratio"]["forecast"]["S"] == pytest.approx(0)
    kin = L1.token_states("statics", ctx_of(1.25), TT, Q["g1/tip"], PH, TT, Q["g1/tip"], TE2, 9.0, None, PH)
    w_tip = laws.tip_window(9.81, 0.8)          # 0.571 s: phi over EXACTLY the tip window (interpolated end), not
    assert kin["post"]["phi"][0] == pytest.approx(0.5 * al * w_tip ** 2, rel=0.01)   # the last token-frame in it
    assert laws.statics.stats(kin["pre"], kin["post"], ctx_of(1.25))["alpha_ratio"] == pytest.approx(1, rel=1e-6)


def test_angular_momentum_kind():
    """Puck hits a bar: the rigid impulse conserves L (L_ratio 1) and p (eta 0); a forecast that ignores the bar's
    spin (the law's null) scores L_ratio S ~ 0 while eta stays legal."""
    ctx = {"m": np.array([0.8, 1.6]), "I": np.array([0.02, 0.9]), "n": np.array([1.0, 0.0, 0.0]),
           "c": np.array([0.0, 0.4, 0.0]), "e": 0.9, "event_time_s": TE2}
    p0 = np.array([[-0.25, 0.4, 0.0], [0.15, 0.0, 0.0]])
    pre = {"p": p0, "v": np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), "w": np.zeros((2, 3))}
    post = laws.rigid_impulse(pre, ctx, 0.9)
    after = (TT >= TE2)[:, None, None]
    V = np.where(after, post["v"][None], pre["v"][None])
    W = np.where(after, post["w"][None], 0.0)
    r = _package("angular_momentum", ["g0/legal"], {"g0/legal": {"v": V, "w": W}},
                 {"g0/legal": {"v": V, "w": np.zeros_like(W)}}, {"g0/legal": ctx}, pre_p={"g0/legal": p0})
    assert r["L_ratio"]["true"]["S"] == pytest.approx(1, abs=1e-6)       # null L_ratio < 0 here: the linear form
    assert r["L_ratio"]["forecast"]["S"] == pytest.approx(0, abs=1e-6)
    assert r["eta"]["form"] == "unit" and r["eta"]["forecast"]["S"] == pytest.approx(1, abs=1e-6)


def test_swing_kind_apex_heights():
    """eps from decoded apex heights above the lowest decoded point (no g): sqrt(h_post / h_pre)."""
    for eps in (0.7, 1.0, 1.3):
        h = np.where(TT < TE2, 0.3 * np.cos(np.pi * TT / 2.0) ** 2, eps ** 2 * 0.3 * np.sin(np.pi * (TT - TE2) / 3.0) ** 2)
        P = np.zeros((len(TT), 1, 3))
        P[:, 0, 2] = 4.0 + h
        k = L1.token_states("swing", {"g": G}, TT, {"p": P}, PH, TT, {"p": P}, TE2, 5.5, None, PH)
        assert k["eps"] == pytest.approx(eps, rel=0.03) and k["rise"] == pytest.approx(eps ** 2 * 0.3, rel=0.05)


def test_friction_end_to_end(tmp_path, monkeypatch):
    """run() dispatches a per-token-frame kind (friction), ends the moving window before the legal stop and scores
    a law-following forecast S ~ 1 for decel_const."""
    rng = np.random.default_rng(3)
    clips, Vs = {}, {}
    for g in range(30):
        v0, a = rng.uniform(2.5, 3.5), rng.uniform(0.8, 1.2)
        u = np.array([np.cos(th := rng.uniform(0, 6.28)), np.sin(th), 0.0])
        for t, f in (("legal", 1.0), ("decel_x0.7", 0.7)):
            ctx = {"event": "none", "m": [1.0], "event_time_s": TE2, "t_stop_s": TE2 + v0 / a,
                   "t_stop_twin_s": TE2 + v0 / (a * f), "clean_until_s": 5.9}
            clips[f"g{g:03d}/{t}"] = {"v_pre": [list(u * v0)], "v_post": [list(u * v0)], "ctx": ctx, "objects": ["A"],
                                      "legal": t == "legal", "law": "friction"}
            Vs[f"g{g:03d}/{t}"] = _slide_v(v0, a, u, "legal", f)
    keys = sorted(clips)
    P = rng.normal(size=(3, 24))
    emb = lambda V: V.reshape(len(V), -1) @ P + 0.01 * rng.normal(size=(len(V), 24))
    k0 = int(np.searchsorted(TT, TE2))
    fr = {"t": [TT] * len(keys), "phase": [PH] * len(keys), "V": [Vs[c] for c in keys],
          "W": [np.zeros_like(Vs[c]) for c in keys], "P": [np.zeros_like(Vs[c]) for c in keys],
          "t_fc": [TT[k0:]] * len(keys)}
    real, fc = [emb(Vs[c]) for c in keys], [emb(Vs[c][k0:]) for c in keys]
    monkeypatch.setattr(L1, "load_token_frames", lambda *a: ({"final": {"real": real, "fc": fc}}, fr))
    d, c_ = _write(tmp_path, clips, "f6")
    res, _ = _run(d, c_, monkeypatch, {}, {})
    v = res["variants"]["friction|final|frame"]
    assert res["laws"] == ["friction"] and v["law"] == "friction"
    assert v["ratio"]["decel_const"]["true"]["S"] == pytest.approx(1, abs=0.01)
    assert v["ratio"]["decel_const"]["forecast"]["S"] > 0.8 and v["ratio"]["decel_const"]["floor"]["S"] > 0.8
    assert v["n_clips"]["windows_forecast"] == len(keys)
    json.dumps(res, default=float)


def _f7_labels(sub, n=2):
    """labels.json content (no frames) of every twin of the first n accepted closed-form F7 scenes of ``sub``."""
    from physlaw.config import load_config
    from physlaw.data import families as FAM
    from physlaw.data import generate as GEN
    cfg = load_config("configs/data/f7_freefall.yaml")
    fmt, mod, out = GEN.master_fmt(cfg), FAM.family_module("f7"), []
    for seed in range(80):
        sc = mod.sample(np.random.default_rng([seed, 0]), cfg, fmt, sub)
        if isinstance(sc, str) or sc.inframe_reason():
            continue
        out.append({name: json.loads(json.dumps(
            {"per_frame": GEN.per_frame_labels(sc, tw, sc.camera, None),
             "scene_constants": {"num_frames": len(tw.tracks[sc.bodies[0].name].p), "frame_rate": sc.fps},
             "physlaw": {"family": sc.family, "twin": name, **GEN.law_fields(sc, tw, sc.camera)}},
            default=GEN._json_default)) for name, tw in sc.twins.items()})
        if len(out) == n:
            return out
    raise AssertionError("sampler rejects almost everything")


@pytest.mark.parametrize("sub", ["drop", "projectile"])
def test_gravity_windows_use_free_in_frame_token_frames(sub):
    """The TRUE row of L1 gravity (labelled velocities through the token-frame windows of a V-JEPA-2-like grid: 24 fps,
    tubelet 2) gives g_ratio = 1 on legal F7 clips and 1.5 on gravity_x1.5: the windows hold only token-frames after
    the release, before the floor contact (last master frame <= clean_until_s) with every ball in frame. Drop balls
    rest in the hand before the release, so a pre window over every pre token-frame biases a_pre (g_ratio != 1 even
    on the truth); projectiles start off screen (decoded rows there carry no ball). The Aristotelian
    null (v_h' = 0, a' = 0) gives axis_ratio 0, so the projectile's axis_ratio is scorable (linear form)."""
    from physlaw.data.io import law_inputs
    from physlaw.models.base import token_spec
    from physlaw.tokens.project import token_frame_phase
    spec = token_spec({"input": {"resolution": 256}, "token_grid": {"patch": 16, "tubelet": 2}})
    n_bias = 0
    for scene in _f7_labels(sub):
        for twin, lab in scene.items():
            pl = lab["physlaw"]
            objs = [n for n, o in pl["objects"].items() if not o.get("aux")]
            k = int(pl["first_post_contact_frame"])
            frames = list(range(k - 64, k + 64, 2))
            t, V, _, _, ok = L1.token_frame_targets(lab, frames, spec, objs)
            ph = np.array(token_frame_phase(frames, spec, k))
            te, t_end = float(pl["event_time_s"]), float(pl.get("clean_until_s", np.inf))
            kin = L1.token_states("gravity", {}, t, {"v": V}, ph, t, {"v": V}, te, t_end, ok, ph, ok)
            pre, post, ctx = law_inputs(lab)
            st = laws.gravity.stats(kin["pre"], kin["post"], ctx)
            want = {"gravity_x15": 1.5, "heavy_x13": 1.15}.get(twin, 1.0)          # summed over the drop's balls
            assert st["g_ratio"] == pytest.approx(want, abs=1e-6), (sub, twin)
            assert ok[ph == "pre"].sum() >= L1.MIN_TOKEN_FRAMES and not ok.all()   # rest / off-screen frames dropped
            old = L1.token_states("gravity", {}, t, {"v": V}, ph, t, {"v": V}, te, t_end, None, ph, ok)
            n_bias += abs(laws.gravity.stats(old["pre"], old["post"], ctx)["g_ratio"] - want) > 0.05
            if sub == "projectile" and twin == "legal":
                nul = L1.null_stats(laws.gravity, {**pre, **kin["pre"]}, ctx)
                assert st["axis_ratio"] == pytest.approx(1.0, abs=1e-6) and nul["axis_ratio"] == 0.0
                S = L1.ratio_scores({"true": [st["axis_ratio"]]}, 1.0, [nul["axis_ratio"]], ["g"], 0,
                                    np.random.default_rng(0))
                assert S["form"] == "linear" and S["true"]["S"] == pytest.approx(1.0)
    # every pre token-frame: the drop's rest frames bias the truth itself (a projectile's off-screen labels stay
    # ballistic: there only the decoded rows, which see no ball, are biased)
    assert n_bias > 0 if sub == "drop" else True


def test_statics_g_clip_from_the_decoded_fall():
    """M4: where the block leaves the ledge inside the clean window, alpha_ratio's g_clip is the block's DECODED free
    fall (the line of v over the token-frames after t_separate_s), so alpha and g_clip share any common gain."""
    t_sep = TE2 + 0.8
    V = np.zeros((len(TT), 1, 3))
    V[:, 0, 2] = np.where(TT > t_sep, -7.0 * (TT - t_sep), 0.0)       # the clip's own g: 7.0 (not 9.81)
    W = np.zeros((len(TT), 1, 3))
    W[:, 0, 1] = -2.0 * np.clip(TT - TE2, 0, None)
    ctx = {"overhang_x": 1.25, "aspect_ba": 0.5, "g_clip": 9.81, "half_length": 0.8, "event_time_s": TE2,
           "tip_axis": [0.0, -1.0, 0.0], "t_separate_s": t_sep}
    k = L1.token_states("statics", ctx, TT, {"w": W, "v": V}, PH, TT, {"w": W, "v": V}, TE2, 5.5, None, PH)
    assert k["g_clip"] == pytest.approx(7.0, rel=1e-6)
    k2 = L1.token_states("statics", {**ctx, "t_separate_s": None}, TT, {"w": W, "v": V}, PH, TT, {"w": W}, TE2, 5.5,
                         None, PH)
    assert "g_clip" not in k2                                         # no separation: the labelled g_clip stays
