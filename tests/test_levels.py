"""L3, L4, output-stage and aggregation logic (synthetic inputs; no model, no data)."""
import json

import numpy as np
import pytest

from physlaw.analysis import aggregate
from physlaw.eval import l3_invariance as L3
from physlaw.eval import l4_causal as L4
from physlaw.eval import outputs as O


def _labels(va, vb, va1, vb1, n=(1.0, 0.0), ma=1.0, mb=2.0):
    return {"physlaw": {"pre_velocity": {"A": list(va), "B": list(vb)}, "post_velocity": {"A": list(va1), "B": list(vb1)},
                        "contact_normal": list(n), "objects": {"A": {"mass": ma, "radius": 0.4}, "B": {"mass": mb, "radius": 0.5}}}}


def test_galilean_stats_are_boost_invariant():
    from physlaw.eval.l0_states import event_inputs
    v = ((1.0, 0.2), (0.0, 0.0), (0.2, 0.2), (0.5, 0.0))
    u = np.array([0.7, -0.3])
    ev = [event_inputs(_labels(*vv)) for vv in (v, [np.add(x, u) for x in v])]
    st = L3.galilean_stats(np.stack([e[0] for e in ev]), np.stack([e[1] for e in ev]), [e[2] for e in ev])
    assert st["epsilon"][0] == pytest.approx(0.3) and st["epsilon"][1] == pytest.approx(0.3)
    assert st["theta_cm"][1] == pytest.approx(st["theta_cm"][0]) and st["eta"][1] == pytest.approx(st["eta"][0])


def test_get():
    assert L3.get({"a": {"b": 3}}, "a.b") == 3 and L3.get({"a": {}}, "a.b", 7) == 7


def test_crossfit_selects_the_informative_cell_and_predicts_unseen_groups():
    rng = np.random.default_rng(0)
    n = 120
    groups = np.repeat(np.arange(40), 3)
    y = rng.normal(size=n)
    good = np.c_[y + 0.05 * rng.normal(size=n), rng.normal(size=(n, 5))]
    cells = {"noise": rng.normal(size=(n, 6)), "good": good}
    fit = np.where(groups < 30)[0]
    pred, chosen = L3.crossfit(cells, y, groups, fit, k=5, seeds=1)
    assert set(chosen) == {"good"}
    assert np.isfinite(pred).all()
    assert np.corrcoef(pred[groups >= 30, 0], y[groups >= 30])[0, 1] > 0.95   # unseen groups: one probe on all fit rows


def test_steer_hook_adds_delta_to_chosen_tokens_only():
    torch = pytest.importorskip("torch")

    class Blk(torch.nn.Module):
        def forward(self, h):
            return (h * 1.0, None)

    class M:
        tr = type("T", (), {})()
    M.tr.transformer_blocks = torch.nn.ModuleList([Blk(), Blk()])
    h = torch.zeros(2, 5, 3)
    with L4.steer(M, "snr+8/L1", [1, 3], np.array([1.0, 2.0, 3.0])):
        out = M.tr.transformer_blocks[1](h)[0]
    assert torch.equal(out[:, [1, 3]], torch.tensor([1.0, 2.0, 3.0]).expand(2, 2, 3))
    assert out[:, [0, 2, 4]].abs().sum() == 0
    assert M.tr.transformer_blocks[1](h)[0].abs().sum() == 0        # hook removed
    assert L4.block_index("12") == 12 and L4.block_index("snr-20/L6") == 6


def test_line_at_and_readout_files():
    arr = np.stack([np.arange(10) * 2.0 + 1, np.zeros(10)], 1)
    assert O._line_at(arr, list(range(2, 8)), 0, 1.0) == pytest.approx([1.0, 0.0])
    assert O.readout_file("strict") == "_readout.jsonl" and O.readout_file("loose", real=True) == "_readout_real_loose.jsonl"
    s = O._stats([{"measurable": True, "epsilon": 0.5, "violations": ["curved_path"]}, {"measurable": False, "reason": "lost"}])
    assert s["measurable"] == 1 and s["median_epsilon"] == 0.5 and s["violations"] == {"curved_path": 1}
    assert s["drop_reasons"] == {"lost": 1}


def _write_results(root):
    d = root / "m1" / "f1"
    d.mkdir(parents=True)
    (d / "l0.json").write_text(json.dumps({"targets": {
        "A.pre.velocity": {"heldout": {"score": 0.5}, "mask_floor": {"r2": 0.1}, "r2_ci95": [0.4, 0.6],
                           "selectivity": {"object": {"selectivity": 0.4}}},
        "A.post.velocity": {"heldout": {"score": 0.7}, "mask_floor": {"r2": 0.3}, "r2_ci95": [0.6, 0.8]},
        "A.pre.log_mass": {"heldout": {"score": 0.9}, "mask_floor": {"r2": 0.0}}}}))
    (d / "l1.json").write_text(json.dumps({
        "selected": {"energy|frame": {"momentum": {"selected": "energy|final|frame", "score": 0.4},
                                      "floor_best": "energy|final|frame"},
                     "energy|roi": {"floor_best": "energy|final|roi"}},
        "variants": {"energy|final|frame": {"momentum": {"S": 0.35, "S_ci95": [0.1, 0.6], "eta": {"floor": 0.05}},
                                            "energy": {"status": "measured", "forecast": {"creation_rate": 0.2,
                                                       "creation_ci95": [0.1, 0.3]}, "floor": {"creation_rate": 0.05}}},
                     "energy|final|roi": {}}}))
    (d / "l2.json").write_text(json.dumps({
        "calibration": ["conserve_e1", "conserve_e015"], "primary_contrast": "mom_violate_e1 vs conserve_e1",
        "contrasts": {"mom_violate_e1 vs conserve_e1": {"m1": {"mse_obj": {
            "win_rate": 0.7, "win_rate_ci95": [0.6, 0.8], "vs_pixel": {"copy_last": {"baseline_win": 0.55}}}}}},
        "energy_kink": {"bounce": {"distances": {"mse_all": {"model": {"K": 0.3, "K_ci95": [0.1, 0.5]},
                                                             "baselines": {"copy_last": {"K": 0.05}}, "verdict": "kink"}}}},
        "energy_kink_rep": {"bounce": {"model": {"heldout_K_rep": 0.2, "K_rep_ci95": [0.0, 0.4]},
                                       "baselines": {"pixels": {"K_rep": 0.1}}, "verdict": "x"}}}))
    (d / "l2b.json").write_text(json.dumps({"contrasts": {"mom_violate_e1 vs conserve_e1": {
        "overall": {"score": 0.8, "auc_ci95": [0.7, 0.9]}, "kinematic": {"auc_mean": 0.6}},
        "conserve_e1 vs conserve_e015": {"control": True, "overall": {"score": 0.5}}}}))
    (d / "l4.json").write_text(json.dumps({"law": "momentum", "variants": {"frame": {
        "law_value": 1.0, "law": {"slope": 0.4, "slope_ci95": [0.2, 0.6], "wrong_way": 0.1},
        "random_dir": {"slope": 0.0}, "pairs": {"swap": {"iia": 0.7}}, "verdict": {"law_like": False}}}}))
    (d / "outputs_trackA.json").write_text(json.dumps({"momentum": {"legal": {"S_mom": 0.2, "S_mom_ci95": [0.0, 0.4]}},
                                                       "energy": {"tol": 0.1, "generated": {"share_le1": 0.8},
                                                                  "real_legal": {"share_le1": 0.9}}}))
    return d


def test_aggregate_one_primary_number_per_law_and_level(tmp_path):
    _write_results(tmp_path)
    rows = aggregate.run(tmp_path)["rows"]
    assert len(rows) == 1
    s = rows[0]["scores"]
    assert s["l0.velocity"]["score"] == pytest.approx(0.6) and s["l0.velocity"]["floor"] == pytest.approx(0.2)
    assert s["l0.velocity"]["ci95"] == pytest.approx([0.5, 0.7]) and s["l0.velocity"]["selectivity_object"] == 0.4
    assert s["l1.momentum"]["score"] == 0.35 and s["l1.momentum"]["floor"] == 0.0 and s["l1.momentum"]["ci95"] == [0.1, 0.6]
    assert s["l1.energy"]["score"] == 0.2 and s["l1.energy"]["floor"] == 0.05
    assert s["l2.momentum"]["score"] == 0.7 and s["l2.momentum"]["floor"] == 0.55
    assert s["l2.energy"]["score"] == 0.3 and s["l2.energy"]["floor"] == 0.05
    assert s["l2rep.energy"]["score"] == 0.2 and s["l2b.momentum"]["floor"] == 0.6
    assert s["l4.momentum"]["score"] == 0.4 and s["l4.momentum"]["iia"] == 0.7
    assert s["outA.momentum"]["score"] == 0.2 and s["outA.energy"]["floor"] == 0.9
    assert not any(k.startswith(("l3.", "outB.")) for k in s)          # not run: blank
    md = (tmp_path / "leaderboard.md").read_text()
    assert "| m1 |" in md and "+0.35 (+0.00) [+0.10, +0.60]" in md and "l1.momentum: S_mom" in md


def test_figures_draw_from_the_new_rows(tmp_path):
    pytest.importorskip("matplotlib")
    from physlaw.analysis import figures
    _write_results(tmp_path)
    d2 = tmp_path / "m2" / "f1"
    d2.mkdir(parents=True)
    for f in ("l0.json", "l1.json", "l2.json", "l2b.json", "outputs_trackA.json"):
        (d2 / f).write_text((tmp_path / "m1" / "f1" / f).read_text())
    rows = aggregate.run(tmp_path)["rows"]
    made = figures.draw(rows, tmp_path, tmp_path / "figures", log=lambda *a: None)
    assert {"fig1_l0_vs_l1", "fig2_l2_vs_pixels", "fig3_l2_vs_l2b", "fig5_internal_vs_output"} <= set(made)


def _side_clip(g=9.81, n_master=60, fps=48, vx=2.0, vz=3.0):
    """A ball (r = 0.3 m) on a parabola in the y = 0 plane, seen by a side camera 20 m away (50 px / m)."""
    from types import SimpleNamespace
    W, H, f, C = 832, 480, 1000.0, np.array([0.0, -20.0, 4.3])
    R = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    E = np.eye(4); E[:3, :3] = R; E[:3, 3] = -R @ C
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    t = np.arange(n_master) / fps
    P = np.stack([-3 + vx * t, 0 * t, 6 + vz * t - 0.5 * g * t ** 2], 1)
    uv = (P - C) @ R.T @ K.T
    uv = uv[:, :2] / uv[:, 2:]
    yy, xx = np.mgrid[:H, :W]
    frames = np.full((n_master, H, W, 3), 160, np.uint8)
    masks = np.zeros((n_master, H, W), np.uint8)
    for i, (u, v) in enumerate(uv):
        d = (xx - u) ** 2 + (yy - v) ** 2 <= 15 ** 2
        frames[i][d] = (200, 40, 40)
        masks[i][d] = 1
    labels = {"physlaw": {"family": "f7", "subfamily": "projectile", "first_post_contact_frame": 30,
                          "objects": {"A": {"mass": 1.0, "radius": 0.3, "segmentation_id": 1}},
                          "constants": {"g": 9.81, "fps": fps},
                          "camera": {"K": K.tolist(), "world_to_camera": E.tolist(), "resolution": [W, H]}},
              "per_frame": [{"A": {"position": p.tolist(), "pixel": q.tolist()}} for p, q in zip(P, uv)]}
    return SimpleNamespace(labels=labels, frames=frames, masks=masks)


def test_backproject_and_trajectory_readout_is_scale_free():
    c = _side_clip()
    pf = c.labels["per_frame"][7]["A"]
    w = O.backproject(np.array([pf["pixel"][1], pf["pixel"][0]]), c.labels, 480, 832, 20.0)
    assert np.allclose(w, pf["position"], atol=1e-6)
    r = O.readout_traj(c.frames[::3], c, 16.0)
    assert r["measurable"], r
    assert r["g_ratio"] == pytest.approx(1.0, abs=0.05) and r["axis_ratio"] == pytest.approx(1.0, abs=0.02)
    assert "g_post_over_true" not in r                # no true g enters: a whole arc at g x 1.5 is still legal
    fast = _side_clip(g=1.5 * 9.81)
    assert O.readout_traj(fast.frames[::3], fast, 16.0)["g_ratio"] == pytest.approx(1.0, abs=0.08)


def test_transfer_velocity_uses_the_image_plane():
    side = {"physlaw": {"pre_velocity": {"A": [1.0, 0.0, -2.0]},
                        "camera": {"world_to_camera": [[1, 0, 0, 0], [0, 0, -1, 4.3], [0, 1, 0, 20], [0, 0, 0, 1]]}}}
    top = {"physlaw": {"pre_velocity": {"A": [1.0, 2.0, 0.0]},
                       "camera": {"world_to_camera": [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 20.65], [0, 0, 0, 1]]}}}
    assert L3.clip_target(side, "velocity").tolist() == [1.0, -2.0]          # F7 side view: (v_x, v_z)
    assert L3.clip_target(top, "velocity").tolist() == [1.0, 2.0]            # F1 top-down: (v_x, v_y)
    assert L3.clip_target({"physlaw": {"pre_velocity": {"A": [3.0, 4.0]}}}, "velocity").tolist() == [3.0, 4.0]


def test_aggregate_new_laws(tmp_path):
    """F6 / F9 / F11 / F12 columns: L1 ratio laws (statics as an accuracy), L3 mass independence under its own law,
    output ratio laws (track A medians vs the real legal twins; track B vs real.legal)."""
    for fam, l1_ratio, gen, real in (
            ("f6", {"decel_const": {"forecast": {"S": 0.4, "S_ci95": [0.1, 0.7]}, "floor": {"S": 0.8}},
                    "inertia_speed": {"forecast": {"S": 0.6}, "floor": {"S": 0.9}},
                    "stop_hold": {"forecast": {"S": 0.2}, "floor": {"S": 0.7}}},
             {"median_decel_const": 0.9, "median_inertia_speed": 1.1, "median_stop_hold": 0.3},
             {"median_decel_const": 1.0, "median_stop_hold": 0.0}),
            ("f11", {"tip_consistency": {"forecast": {"accuracy": 0.7}, "floor": {"accuracy": 0.95}}},
             {"mean_tip_consistency": 0.6}, {"mean_tip_consistency": 1.0})):
        d = tmp_path / "m1" / fam
        d.mkdir(parents=True)
        (d / "l1.json").write_text(json.dumps({"selected": {"ratio|frame": {"floor_best": "v"}},
                                               "variants": {"v": {"ratio": l1_ratio}}}))
        (d / "outputs_trackA.json").write_text(json.dumps({"generated": gen, "real": {"legal": real}}))
        (d / "outputs_trackB.json").write_text(json.dumps({"gates": {"strict": {"generated": gen,
                                                                                "real": {"legal": real}}}}))
    (tmp_path / "m1" / "f6" / "l3.json").write_text(json.dumps({"mass_independence": {
        "law": "friction", "spaces": {"frame": {"score": {"S": 0.3, "sensitivity_log": 0.1}}}}}))
    rows = {r["family"]: r["scores"] for r in aggregate.run(tmp_path)["rows"]}
    f6, f11 = rows["f6"], rows["f11"]
    assert f6["l1.friction"]["score"] == 0.4 and f6["l1.friction"]["floor"] == 0.8
    assert f6["l1.momentum_inertia"]["score"] == 0.6 and f6["l3.friction"]["score"] == 0.3
    assert f6["outA.friction"]["score"] == 0.9 and f6["outA.friction"]["floor"] == 1.0
    assert f6["outB.momentum_inertia"]["score"] == 1.1 and f6["outB.momentum_inertia"]["floor"] is None
    assert f6["l1.friction_stop"]["score"] == 0.2 and f6["outA.friction_stop"]["floor"] == 0.0
    assert "law 0" in f6["outA.friction_stop"]["metric"]
    assert f11["l1.statics"]["score"] == 0.7 and f11["outB.statics"]["score"] == 0.6 and f11["outB.statics"]["floor"] == 1.0
    assert "l1.statics" in (tmp_path / "leaderboard.md").read_text()


def _f6_labels(sub, seeds=range(3)):
    """labels.json content (no frames) of every twin of closed-form F6 scenes of ``sub``."""
    from physlaw.config import load_config
    from physlaw.data import families as FAM
    from physlaw.data import generate as GEN
    cfg = load_config("configs/data/f6_friction.yaml")
    fmt, out = GEN.master_fmt(cfg), []
    for seed in seeds:
        sc = FAM.family_module("f6").sample(np.random.default_rng([seed, 0]), cfg, fmt, sub)
        out.append({name: json.loads(json.dumps(
            {"per_frame": GEN.per_frame_labels(sc, tw, sc.camera, None),
             "scene_constants": {"num_frames": len(tw.tracks["A"].p), "frame_rate": sc.fps},
             "physlaw": {"family": sc.family, "twin": name, **GEN.law_fields(sc, tw, sc.camera)}},
            default=GEN._json_default)) for name, tw in sc.twins.items()})
    return out


def test_f6_l0_targets_and_l3_decel():
    """F6 slide / pair labels: L0's deceleration target is the slide's own friction rate before the branch (= a,
    every twin) and the twin's after it (decel_x0.7: 0.7 a; decel_x1.3: 1.3 a); L3's per-object ``decel`` on a pair gives
    mass_ratio 1 (legal) and 1.3 (heavy_x13) under laws.friction."""
    from physlaw.eval.l0_states import targets
    slides = _f6_labels("slide")
    labs = [g[t] for g in slides for t in ("legal", "decel_x0.7", "decel_x1.3")]
    tg = targets(labs)
    a = np.array([l["physlaw"]["constants"]["a"] for l in labs])
    assert np.allclose(tg["A.pre.acceleration"], a, rtol=1e-6)
    assert np.allclose(tg["A.post.acceleration"], a * np.tile([1.0, 0.7, 1.3], len(slides)), rtol=1e-6, atol=1e-9)
    pairs = _f6_labels("pair")
    for name, want in (("legal", 1.0), ("heavy_x13", 1.3)):
        ls = [g[name] for g in pairs]
        dec = {o: np.array([L3.object_target(l, o, "decel", 0.5) for l in ls]) for o in "AB"}
        M = {o: np.array([l["physlaw"]["objects"][o]["mass"] for l in ls]) for o in "AB"}
        assert np.allclose(L3.mass_ratios(dec, M, np.arange(len(ls)), "friction"), want, rtol=1e-6)


def test_l0_mass_pre_event_and_uninformative_flags(tmp_path):
    """H1: L0 reports whether mass (F10: kappa) is decodable from PRE-event frames (R² CI > 0, selectivity > 0, above
    the untrained model); the leaderboard flags every mass-independence score of a model without it 'uninformative'
    (here: m1 decodes mass, blind does not; the untrained model sets the bar)."""
    from physlaw.eval.l0_states import mass_pre_event
    tgt = lambda r2, lo, sel: {"heldout": {"score": r2}, "r2_ci95": [lo, r2 + 0.1],
                               "selectivity": {"scene": {"selectivity": sel}, "object": {"selectivity": sel}}}
    mp = mass_pre_event({"A.pre.log_mass": tgt(0.6, 0.4, 0.3), "A.post.log_mass": tgt(0.9, 0.8, 0.5)},
                        {"targets": {"A.pre.log_mass": {"heldout": {"score": 0.7}}}})
    assert list(mp["targets"]) == ["A.pre.log_mass"] and mp["decodable"] is False      # below the untrained model
    assert mass_pre_event({"A.pre.log_mass": tgt(0.6, 0.4, 0.3)})["decodable"] is True
    assert mass_pre_event({"A.pre.log_mass": tgt(0.6, 0.4, -0.1)})["decodable"] is False  # reads identity only
    for model, (r2, sel) in {"m1": (0.6, 0.3), "blind": (0.05, -0.2), "untrained-vitl": (0.1, 0.05)}.items():
        d = tmp_path / model / "f6"
        d.mkdir(parents=True)
        (d / "l0.json").write_text(json.dumps({"targets": {"A.pre.log_mass": tgt(r2, r2 - 0.2, sel)}}))
        (d / "l3.json").write_text(json.dumps({"mass_independence": {
            "law": "friction", "spaces": {"frame": {"score": {"S": 0.9, "sensitivity_log": 0.1}}}}}))
        (d / "l1.json").write_text(json.dumps({"selected": {"friction|frame": {"floor_best": "v"}}, "variants": {"v": {
            "ratio": {"mass_ratio": {"forecast": {"S": 0.95}, "floor": {"S": 0.9},
                                     "verdict": {"claim_allowed": False}}}}}}))
    rows = {r["model"]: r["scores"] for r in aggregate.run(tmp_path)["rows"]}
    assert rows["m1"]["l0.mass_pre"]["decodable"] and "uninformative" not in rows["m1"]["l3.friction"]
    assert not rows["blind"]["l0.mass_pre"]["decodable"]
    assert "uninformative" in rows["blind"]["l3.friction"] and "uninformative" in rows["blind"]["l1.friction_mass"]
    assert rows["m1"]["l1.friction_mass"]["claim"] is False
    md = (tmp_path / "leaderboard.md").read_text()
    assert "(uninformative)" in md and "✗claim" in md and "NOT decodable" in md
