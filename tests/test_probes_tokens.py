"""Spec §8 unit tests: probe pipeline on synthetic linear data, token projection, frame sampling."""
from __future__ import annotations

import numpy as np
import pytest

from physlaw.models.base import TokenGridSpec, sample_frames
from physlaw.probes.ridge import auc_score, fit_probe, grouped_kfold, permutation_null
from physlaw.tokens.pool import pool, pool_weights
from physlaw.tokens.project import project_masks, token_frame_phase


# ------------------------------------------------------------------ probes

def synthetic(n=400, d=64, noise=0.3, seed=0, group_size=5):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=d); w /= np.linalg.norm(w)
    X = rng.normal(size=(n, d))
    y = X @ w + noise * rng.normal(size=n)
    groups = np.repeat(np.arange(n // group_size), group_size)
    return X, y, groups, w


def test_probe_recovers_known_direction():
    X, y, g, w = synthetic()
    res = fit_probe(X, y, g, fold_seeds=3)
    expected_r2 = 1 / (1 + 0.3 ** 2)
    assert res.score_mean == pytest.approx(expected_r2, abs=0.05)
    cos = abs(res.direction @ w) / np.linalg.norm(res.direction)
    assert cos > 0.98


def test_probe_on_noise_scores_near_zero():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 64)); y = rng.normal(size=300)
    g = np.repeat(np.arange(60), 5)
    assert fit_probe(X, y, g, fold_seeds=3).score_mean < 0.05


def test_groups_never_split_across_folds():
    g = np.repeat(np.arange(37), 5)
    for seed in range(5):
        folds = grouped_kfold(g, 5, seed)
        assert sorted(np.concatenate(folds)) == list(range(len(g)))
        for f in folds:
            others = np.setdiff1d(np.arange(len(g)), f)
            assert not set(g[f]) & set(g[others])


def test_auc_matches_definition():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 200); s = y + rng.normal(size=200)
    brute = np.mean([(a > b) + 0.5 * (a == b) for a in s[y == 1] for b in s[y == 0]])
    assert auc_score(y, s) == pytest.approx(brute, abs=1e-12)


def test_permutation_null_signal_vs_noise():
    X, y, g, _ = synthetic(n=200, d=16)
    rng = np.random.default_rng(3)
    grid = {"signal": X, "noise1": rng.normal(size=X.shape), "noise2": rng.normal(size=X.shape)}
    res = permutation_null(grid, y, g, n_permutations=40)
    assert res["best_cell"] == "signal" and res["p_corrected"] <= 1 / 41 + 1e-12
    res0 = permutation_null({"a": grid["noise1"], "b": grid["noise2"]}, y, g, n_permutations=40)
    assert res0["p_corrected"] > 0.05


# ------------------------------------------------------------------ frames / tokens

def test_sample_frames_exact_strides():
    assert sample_frames(master_fps=48, n_master=289, fps=8, n_frames=49) == list(range(0, 289, 6))
    f16 = sample_frames(master_fps=48, n_master=289, fps=16, n_frames=81)
    assert f16[-1] == 240 and set(np.diff(f16)) == {3}
    c = sample_frames(master_fps=48, n_master=289, fps=24, n_frames=64, contact_frame=120, center_on_contact=True)
    assert c[32] == 120 and set(np.diff(c)) == {2}
    with pytest.raises(ValueError):
        sample_frames(master_fps=48, n_master=289, fps=30, n_frames=10)


def _disk_masks(T=12, H=96, W=160, centers=((30, 40), (60, 120)), r=10):
    yy, xx = np.mgrid[:H, :W]
    m = np.zeros((T, H, W), np.uint8)
    for sid, (cy, cx) in enumerate(centers, start=1):
        m[:, (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] = sid
    return m


def test_projection_squash_center_and_radius():
    m = _disk_masks()
    spec = TokenGridSpec(input_hw=(64, 64), patch=8, tubelet=2)
    obj = project_masks(m, list(range(12)), spec)
    # squash maps (30, 40) in 96x160 to (20, 16) in 64x64 -> token (2.5, 2.0)
    assert obj["A"].center[0] == pytest.approx([30.5 * 64 / 96 / 8, 40.5 * 64 / 160 / 8], abs=0.05)
    assert obj["A"].roi[0].any() and not (obj["A"].roi[0] & obj["B"].roi[0]).any()
    assert obj["A"].occupancy.shape == (6, 8, 8)
    assert obj["A"].occupancy.max() <= 1.0 + 1e-6


def test_projection_letterbox_offsets():
    m = _disk_masks()
    spec = TokenGridSpec(input_hw=(64, 64), patch=8, tubelet=2, fit="letterbox")
    obj = project_masks(m, list(range(12)), spec)
    s = 64 / 160; oy = (64 - 96 * s) / 2
    assert obj["B"].center[0] == pytest.approx([(60.5 * s + oy) / 8, 120.5 * s / 8], abs=0.05)


def test_phase_and_pooling():
    m = _disk_masks()
    spec = TokenGridSpec(input_hw=(64, 64), patch=8, tubelet=2)
    frames = list(range(12))
    ph = token_frame_phase(frames, spec, first_post_contact_frame=5)
    assert ph == ["pre", "pre", "straddle", "post", "post", "post"]
    w = pool_weights(project_masks(m, frames, spec), ph)
    for k, v in w.items():
        assert v.sum() == pytest.approx(1.0) or v.sum() == 0
    acts = np.random.default_rng(0).normal(size=(6, 8, 8, 4)).astype(np.float32)
    p = pool(acts, w)
    assert np.allclose(p["frame.all"], acts.reshape(-1, 4).mean(0), atol=1e-5)
    assert not np.any(w["roi_A.pre"][3:])
