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
