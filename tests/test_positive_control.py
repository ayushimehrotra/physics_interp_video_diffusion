"""Positive controls (spec §4.4): masks, the training-set guard, the pass check, PC1's context encoding."""
from __future__ import annotations

import json

import numpy as np
import pytest

from physlaw.config import Cfg, load_config
from physlaw.models import positive_control as PC

CRIT = load_config("configs/models/positive_control.yaml")["pass_criterion"]


def test_future_mask_is_a_prefix():
    rng = np.random.default_rng(0)
    m = {"p_future": 1.0, "future_ctx": [8, 24], "tube_blocks": [[8, 0.15]], "aspect": [0.75, 1.5]}
    for _ in range(20):
        ctx, tgt = PC.sample_mask(rng, 32, 16, m)
        k = len(ctx) // 256
        assert 8 <= k <= 24 and len(ctx) == k * 256
        assert np.array_equal(ctx, np.arange(k * 256)) and np.array_equal(tgt, np.arange(k * 256, 32 * 256))


def test_tube_mask_partitions_tokens_identically_per_frame():
    rng = np.random.default_rng(1)
    m = {"p_future": 0.0, "future_ctx": [8, 24], "tube_blocks": [[8, 0.15], [2, 0.7]], "aspect": [0.75, 1.5]}
    for _ in range(20):
        ctx, tgt = PC.sample_mask(rng, 4, 16, m)
        assert len(ctx) and len(tgt) and not np.intersect1d(ctx, tgt).size
        assert np.array_equal(np.sort(np.r_[ctx, tgt]), np.arange(4 * 256))
        per_frame = [set(tgt[(tgt >= f * 256) & (tgt < (f + 1) * 256)] - f * 256) for f in range(4)]
        assert all(s == per_frame[0] for s in per_frame)


def _clip(root, fam, gid, twin, split="train", legal=None):
    (root / fam / gid / twin).mkdir(parents=True)
    (root / fam / gid / twin / "_COMPLETE").write_text("")
    g = root / fam / gid / "_GROUP_COMPLETE"
    rec = json.loads(g.read_text()) if g.exists() else {"group": gid, "split": split, "twins": [], "summaries": {}}
    rec["twins"].append(twin)
    rec["summaries"][twin] = {} if legal is None else {"legal": legal}
    g.write_text(json.dumps(rec))


def test_training_clips_legal_only_and_refuses_eval_groups(tmp_path):
    tr, ev = tmp_path / "pc1", tmp_path / "eval"
    _clip(tr, "f1", "f1_200000", "conserve_e1")
    _clip(tr, "f1", "f1_200000", "mom_violate_e1")
    _clip(tr, "f1", "f1_200001", "conserve_e015", split="val")
    _clip(tr, "f7", "f7_200000", "legal_a", legal=True)
    _clip(tr, "f7", "f7_200000", "gravity_x1.5", legal=False)
    (ev / "f1" / "f1_100000").mkdir(parents=True)
    dc = Cfg(root=str(tr), families=["f1", "f7"], exclude_eval=[str(ev / "f1")])
    got = sorted(f"{p.parent.name}/{p.name}" for p in PC.training_clips(dc, ["train"]))
    assert got == ["f1_200000/conserve_e1", "f7_200000/legal_a"]
    _clip(tr, "f1", "f1_100000", "conserve_e1")             # an evaluation group leaks into the training set
    with pytest.raises(ValueError, match="evaluation set"):
        PC.training_clips(dc, ["train"])


def _results(d, s_family, wins, control_ci=(0.4, 0.6), option="P-real-mm|frame"):
    d.mkdir(parents=True, exist_ok=True)
    (d / "l1.json").write_text(json.dumps({"selected": {option: {"family": {"score": s_family, "n_variants": 1}},
                                                        "P-real-mm|roi": {"family": {"score": 0.9}}}}))
    con = {c: {CRIT["l2_distance"]: {"adj_win_rate": w, "adj_win_rate_ci95": [w - 0.1, w + 0.1]}}
           for c, w in zip(CRIT["l2_contrasts"], wins)}
    con[CRIT["l2_control"]] = {CRIT["l2_distance"]: {"adj_win_rate": 0.5, "adj_win_rate_ci95": list(control_ci)}}
    (d / "l2.json").write_text(json.dumps({"motion_adjusted": {"contrasts": {CRIT["l2_covariates"]: con}}}))


def test_pass_check_verdicts(tmp_path):
    _results(tmp_path / "a", 0.6, [0.8, 0.75], option="P-pred-real|frame")
    r = PC.pass_check(tmp_path / "a", CRIT)
    assert r["verdict"] == "PASS" and r["l1"]["option"] == "P-pred-real|frame" and "PASS" in PC.format_pass(r)
    _results(tmp_path / "b", 0.3, [0.8, 0.75])                  # ROI pool's 0.9 is never used
    assert PC.pass_check(tmp_path / "b", CRIT)["verdict"] == "FAIL"
    _results(tmp_path / "c", 0.6, [0.8, 0.65])
    assert PC.pass_check(tmp_path / "c", CRIT)["verdict"] == "FAIL"
    _results(tmp_path / "d", 0.6, [0.8, 0.75], control_ci=(0.6, 0.8))
    assert PC.pass_check(tmp_path / "d", CRIT)["verdict"] == "INCONCLUSIVE"
    (tmp_path / "d" / "l1.json").unlink()
    assert PC.pass_check(tmp_path / "d", CRIT)["verdict"] == "MISSING"


def test_context_encoding_equals_encoding_the_prefix_alone():
    """Training's encode_context on the first k token-frames == forecast()'s encoder run on the prefix frames."""
    torch = pytest.importorskip("torch")
    cfg = load_config("configs/models/positive_control.yaml",
                      ["arch.depth=2", "arch.pred_depth=1", "input.num_frames=8", "input.resolution=64"])
    m = PC.build_model(cfg).eval()
    x = torch.randn(1, 8, 3, 64, 64)
    k, per = 2, 16
    with torch.no_grad():
        a = PC.encode_context(m.encoder, x, torch.arange(k * per))
        b = m(pixel_values_videos=x[:, :2 * k], skip_predictor=True).last_hidden_state
        p = PC.predict(m, x, torch.arange(k * per), torch.arange(k * per, 4 * per))
    assert torch.allclose(a, b, atol=1e-5)
    assert p.shape == (1, 2 * per, 384)
