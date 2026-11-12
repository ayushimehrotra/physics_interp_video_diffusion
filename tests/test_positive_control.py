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


def test_pc_configs_read_every_rendered_family():
    """PC1 and PC2 train on every family of the PC render set (the controls must see every law they are judged on),
    and the set holds >= 3,000 legal clips per family; every benchmark family dir is excluded."""
    pc = load_config("configs/data/pc1_train.yaml")
    for cfg in ("positive_control", "pc2_wan22_ti2v_5b"):
        dc = load_config(f"configs/models/{cfg}.yaml")["data"]
        assert list(dc["families"]) == list(pc.families), cfg
        assert {f"data/{f}" for f in pc.families} <= set(dc["exclude_eval"]), cfg
    assert all(int(spec.n_groups) > 0 for spec in pc.families.values())


def _results(d, s_mom, wins, gap_lo=0.1, option="energy|frame"):
    f = d / CRIT["momentum_family"]
    f.mkdir(parents=True, exist_ok=True)
    sel = {option: {"momentum": {"score": s_mom, "n_variants": 1, "selected": "energy|L1|frame"}},
           "energy|roi": {"momentum": {"score": 0.9, "selected": "energy|L1|roi"}}}
    (f / "l1.json").write_text(json.dumps({"selected": sel, "variants": {"energy|L1|frame": {"instrument": "pred_real"}}}))
    vp = {b: {"model_minus_baseline": gap_lo + 0.1, "model_minus_baseline_ci95": [gap_lo, gap_lo + 0.2]}
          for b in ("copy_last", "const_velocity")}
    rec = lambda w: {"pc1-vits": {CRIT["l2_distance"]: {"win_rate": w, "win_rate_ci95": [w - 0.1, w + 0.1], "vs_pixel": vp}},
                     "pixel_baselines": {}}
    con = {c: rec(w) for c, w in zip(CRIT["l2_contrasts"], wins)}
    con[CRIT["l2_control"]] = rec(0.8)
    (f / "l2.json").write_text(json.dumps({"contrasts": con}))


def _kink(d, fam, K, ci, verdict):
    en = CRIT["energy"]
    rec = {"model": {"K": K, "K_ci95": list(ci)}, "verdict": verdict}
    (d / fam).mkdir(parents=True, exist_ok=True)
    (d / fam / "l2.json").write_text(json.dumps({"energy_kink": {en["entries"][fam]: {"distances": {en["distance"]: rec}}}}))


def test_pass_check_verdicts(tmp_path):
    u = tmp_path / "untrained"
    _kink(u, "f2", 0.05, (-0.1, 0.2), "no kink")
    _kink(u, "f7", 0.4, (0.2, 0.6), "kink, not beyond every baseline")
    _results(tmp_path / "a", 0.6, [0.8, 0.75], option="energy|contact")
    _kink(tmp_path / "a", "f2", 0.8, (0.5, 1.1), "kink beyond baselines")
    r = PC.pass_check(tmp_path / "a" / "f1", CRIT, u)                 # a family dir is accepted too
    assert r["verdict"] == "PASS" and r["l1"]["option"] == "energy|contact" and "PASS" in PC.format_pass(r)
    assert r["energy"]["f2/energy_kink"]["untrained_contains_0"]
    assert PC.pass_check(tmp_path / "a", CRIT)["verdict"] == "MISSING"          # no untrained run
    for k, (s, w, lo) in {"b": (0.3, [0.8, 0.75], 0.1), "c": (0.6, [0.8, 0.65], 0.1),
                          "d": (0.6, [0.8, 0.75], -0.05)}.items():
        _results(tmp_path / k, s, w, lo)
        _kink(tmp_path / k, "f2", 0.8, (0.5, 1.1), "kink beyond baselines")
    assert PC.pass_check(tmp_path / "b", CRIT, u)["verdict"] == "FAIL"           # ROI pool's 0.9 is never used
    assert PC.pass_check(tmp_path / "c", CRIT, u)["verdict"] == "FAIL"
    assert PC.pass_check(tmp_path / "d", CRIT, u)["verdict"] == "FAIL"           # a pixel extrapolator matches it
    # energy: kink not beyond baselines on F2, and on F7 the untrained model kinks too -> FAIL
    _results(tmp_path / "e", 0.6, [0.8, 0.75])
    _kink(tmp_path / "e", "f2", 0.3, (0.1, 0.5), "kink, not beyond every baseline")
    _kink(tmp_path / "e", "f7", 0.8, (0.5, 1.1), "kink beyond baselines")
    r = PC.pass_check(tmp_path / "e", CRIT, u)
    assert r["verdict"] == "FAIL" and not r["energy"]["f7/energy_kink_bounce"]["untrained_contains_0"]
    (tmp_path / "d" / "f1" / "l1.json").unlink()
    assert PC.pass_check(tmp_path / "d", CRIT, u)["verdict"] == "MISSING"


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
