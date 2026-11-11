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


def test_restitution_and_scatter_angle_are_boost_invariant():
    lab = _labels((1.0, 0.2), (0.0, 0.0), (0.2, 0.2), (0.5, 0.0))
    u = np.array([0.7, -0.3])
    boosted = _labels(*(np.add(v, u) for v in ((1.0, 0.2), (0.0, 0.0), (0.2, 0.2), (0.5, 0.0))))
    assert L3.restitution(lab) == pytest.approx(0.3)
    assert L3.restitution(boosted) == pytest.approx(L3.restitution(lab))
    assert L3.scatter_angle(boosted) == pytest.approx(L3.scatter_angle(lab))


def test_get_and_within_slope():
    assert L3.get({"a": {"b": 3}}, "a.b") == 3 and L3.get({"a": {}}, "a.b", 7) == 7
    g = np.repeat(np.arange(20), 2)
    x = np.random.default_rng(0).normal(size=40) + 10 * g            # group offsets must not leak into the slope
    y = -0.5 * x + 3 * g
    assert L3._within_slope(y, x, g) == pytest.approx(-0.5)


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


def test_pred_change_follows_momentum():
    lab = _labels((1.0, 0.0), (0.0, 0.0), (0, 0), (0, 0))
    dB, dA = L4._pred_change(lab, q0=1.0, dlogq=np.log(2.0), e=1.0)
    # e = 1, v_n = 1: dv_B = 2 q / (1 + q) -> from 1 (q = 1) to 4/3 (q = 2); dv_A = -2 / (1 + q) -> from -1 to -2/3
    assert dB == pytest.approx(1 / 3) and dA == pytest.approx(1 / 3)
    assert L4._pred_change(lab, 1.0, 0.0, 1.0) == (0.0, 0.0)


def test_fit_direction_and_controls():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 10))
    w = np.zeros(10); w[3] = 2.0
    y = X @ w + 0.01 * rng.normal(size=80)
    d = L4.fit_direction(X, y, np.arange(80))
    assert abs(d["u"][3]) > 0.99 and d["gain"] == pytest.approx(2.0, rel=0.05)
    assert d["sd"] == pytest.approx((X @ d["u"]).std())
    c = L4.control_directions(X, d, {"random_dir": 2, "random_subspace": 1, "pc_matched": 1}, seed=0)
    assert [len(c[k]) for k in ("mass", "random_dir", "random_subspace", "pc_matched")] == [1, 2, 1, 1]
    for dirs in c.values():
        for u, s in dirs:
            assert np.linalg.norm(u) == pytest.approx(1.0) and s > 0
    assert c["random_dir"][0][1] == d["sd"]          # norm-matched shift


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
    s = O._stats([{"measurable": True, "e": 0.5, "violations": ["curved_path"]}, {"measurable": False, "reason": "lost"}], ("e",))
    assert s["measurable"] == 1 and s["median_e"] == 0.5 and s["violations"] == {"curved_path": 1}
    assert s["drop_reasons"] == {"lost": 1}


def test_aggregate_leaves_unrun_levels_blank(tmp_path):
    d = tmp_path / "m1" / "f1"
    d.mkdir(parents=True)
    (d / "l0.json").write_text(json.dumps({"targets": {"a": {"r2": 0.5}, "b": {"r2": 0.7}}}))
    (d / "l2.json").write_text(json.dumps({"motion_adjusted": {"contrasts": {"speed+copy_last": {
        "x vs y": {"mse_obj": {"adj_win_rate": 0.6, "raw_win_rate": 0.7}}}}}}))
    (d / "l2b.json").write_text(json.dumps({"contrasts": {"x vs y": {"auc": 0.9}}}))
    rows = aggregate.run(tmp_path)["rows"]
    assert len(rows) == 1 and rows[0]["l0"]["mean_r2"] == pytest.approx(0.6)
    assert rows[0]["encoded_unused"]["x vs y"] == pytest.approx(0.3)
    assert "l1" not in rows[0] and "l3" not in rows[0]
    md = (tmp_path / "leaderboard.md").read_text()
    assert "| m1 |" in md and "mean_r2 +0.60" in md


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


def test_backproject_and_trajectory_readout_recover_g():
    c = _side_clip()
    pf = c.labels["per_frame"][7]["A"]
    w = O.backproject(np.array([pf["pixel"][1], pf["pixel"][0]]), c.labels, 480, 832, 20.0)
    assert np.allclose(w, pf["position"], atol=1e-6)
    r = O.readout_traj(c.frames[::3], c, 16.0)
    assert r["measurable"], r
    assert r["g_ratio"] == pytest.approx(1.0, abs=0.05) and r["g_post_over_true"] == pytest.approx(1.0, abs=0.05)
    assert r["vx_ratio"] == pytest.approx(1.0, abs=0.02)
    from physlaw.eval.laws import gravity
    pre, post = ({k: np.asarray(v) for k, v in r[s].items()} for s in ("state_pre", "state_post"))
    assert gravity.residual(pre, post, r["ctx"]) < 0.05
    fast = _side_clip(g=1.5 * 9.81)          # a clip whose whole arc has g x 1.5: the readout sees it
    assert O.readout_traj(fast.frames[::3], fast, 16.0)["g_post_over_true"] == pytest.approx(1.5, abs=0.08)


def test_aggregate_reads_heldout_formats(tmp_path):
    d = tmp_path / "m2" / "f1"
    d.mkdir(parents=True)
    (d / "l1.json").write_text(json.dumps({"selected": {"P-real-mm|frame": {"family": {"score": 0.3}, "outcome": {"score": 0.1}},
                                                        "P-real-mm|roi": {"family": {"score": 0.9}}}}))
    (d / "l2.json").write_text(json.dumps({"motion_adjusted": {"summary": {"x vs y": {"score": 0.7, "covariates": "c"}}}}))
    (d / "l2b.json").write_text(json.dumps({"contrasts": {"x vs y": {"overall": {"score": 0.8}, "encoded_but_unused": 0.1}}}))
    r = aggregate.run(tmp_path)["rows"][0]
    assert r["l1"] == {"S_family.P-real-mm": 0.3, "S_outcome.P-real-mm": 0.1}       # frame pool only (leak-free)
    assert r["l2"]["x vs y"] == 0.7 and r["l2b"]["x vs y"] == 0.8 and r["encoded_unused"]["x vs y"] == 0.1


def test_track_a_summary_scores_outcome(tmp_path):
    """Stage 9 track A end to end on synthetic rows: law residuals and S_outcome (regression: L1's score helper)."""
    va, vb, n = [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]
    post = {"conserve_e1": ([0.0, 0.0], [0.5, 0.0]), "conserve_e015": ([0.4, 0.0], [0.3, 0.0])}   # m_A = 1, m_B = 2
    for g in ("g0", "g1"):
        for t, (pa, pb) in post.items():
            d = tmp_path / g / t
            d.mkdir(parents=True)
            lab = _labels(va, vb, pa, pb, n)
            lab["physlaw"]["family"] = "f1"
            lab["collision"] = {"contact_point_world": [0.0, 0.0, 0.0]}
            (d / "labels.json").write_text(json.dumps(lab))
    real = [{"group": g, "twin": t, "measurable": True, "v_post": [pa, pb]} for g in ("g0", "g1") for t, (pa, pb) in post.items()]
    gen = [{"group": g, "seed": 0, "measurable": True, "v_post": [[0.5, 0.0], [0.25, 0.0]]} for g in ("g0", "g1")]
    res = O.track_a_summary(gen, real, tmp_path, ["momentum"], n_boot=50)
    d = res["twins"]["conserve_e1"]
    lg = d["legacy"]                                   # r_p / E_out / S_outcome: legacy fields since 2026-11-11
    assert d["res_momentum_floor"] == pytest.approx(0.0) and lg["E_out_floor"] == pytest.approx(0.0)
    # generated: halfway between the null (no collision) and the elastic outcome -> E = 0.5, S_outcome = 0.5
    assert lg["E_out_generated"] == pytest.approx(0.5) and lg["S_outcome"] == pytest.approx(0.5)
    assert lg["S_outcome_ci95"] == pytest.approx([0.5, 0.5])
    assert "S_outcome" not in d and "momentum" in d


def test_transfer_velocity_uses_the_image_plane():
    side = {"physlaw": {"pre_velocity": {"A": [1.0, 0.0, -2.0]},
                        "camera": {"world_to_camera": [[1, 0, 0, 0], [0, 0, -1, 4.3], [0, 1, 0, 20], [0, 0, 0, 1]]}}}
    top = {"physlaw": {"pre_velocity": {"A": [1.0, 2.0, 0.0]},
                       "camera": {"world_to_camera": [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 20.65], [0, 0, 0, 1]]}}}
    assert L3.clip_target(side, "velocity").tolist() == [1.0, -2.0]          # F7 side view: (v_x, v_z)
    assert L3.clip_target(top, "velocity").tolist() == [1.0, 2.0]            # F1 top-down: (v_x, v_y)
    assert L3.clip_target({"physlaw": {"pre_velocity": {"A": [3.0, 4.0]}}}, "velocity").tolist() == [3.0, 4.0]


def test_l4_direction_defaults_to_f1_mass_beyond_size():
    from physlaw.config import load_config
    cfg = load_config("configs/eval/l4_causal.yaml")
    assert cfg["families"] == ["f3"] and cfg["direction_data"] == "f1"       # F3 hides mass before contact
    assert cfg["direction_target"] == "residual" and cfg["direction"] == "generative"
    assert L4._family_dir("f1").name == "f1"
