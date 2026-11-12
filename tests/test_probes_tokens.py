"""Spec §8 unit tests: token projection, frame sampling (probe tests: tests/test_probes.py)."""
from __future__ import annotations

import numpy as np
import pytest

from physlaw.models.base import TokenGridSpec, sample_frames
from physlaw.tokens.pool import pool, pool_weights
from physlaw.tokens.project import group_object_region, project_masks, token_frame_phase


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


def _moving_twin(dx_post, T=12, H=96, W=160, r=8, fpc=5):
    """A twin: ball A at (30, 40) until frame fpc - 1, then moving dx_post px per frame; ball B static; an aux rod
    (seg 3) everywhere on the bottom row (never part of the region)."""
    yy, xx = np.mgrid[:H, :W]
    m = np.zeros((T, H, W), np.uint8)
    m[:, H - 4:, :] = 3
    for t in range(T):
        cx = 40 + dx_post * max(t - fpc + 1, 0)
        m[t, (yy - 60) ** 2 + (xx - 120) ** 2 <= r ** 2] = 2
        m[t, (yy - 30) ** 2 + (xx - cx) ** 2 <= r ** 2] = 1
    labels = {"physlaw": {"first_post_contact_frame": fpc, "objects": {
        "A": {"segmentation_id": 1}, "B": {"segmentation_id": 2}, "rod": {"segmentation_id": 3, "aux": True}}}}
    return m, labels


@pytest.mark.parametrize("spec", [TokenGridSpec(input_hw=(64, 64), patch=8, tubelet=2), None])
def test_group_object_region_is_one_region_per_group(spec):
    """L2 "obj" surprise (stage 4, token grid) and the pixel baselines (stage 5, spec None): every twin of a group is
    scored on the SAME region, the union over the group's twins of the scored objects' masks over the clean window.
    Each twin's own post-event mask (the old region) differs between twins and reveals where its object went."""
    frames, fps = list(range(12)), 48.0
    twins = [_moving_twin(6), _moving_twin(-3)]
    t_end = 9 / fps                                      # clean window: post frames 5..9 (token-frames 3, 4)
    reg = group_object_region(twins, frames, spec, t_end, fps)
    assert np.array_equal(reg, group_object_region(twins[::-1], frames, spec, t_end, fps))   # order-free
    # every twin gets the identical weight map: what stage 4 / 5 hand to surprise() for twin 0 and for twin 1
    w = [np.broadcast_to(reg, (3, *reg.shape)) for _ in twins]
    assert np.array_equal(w[0], w[1])
    own = []
    for m, lab in twins:
        if spec is None:
            o = np.isin(m[5:10], [1, 2]).any(0)
        else:
            ob = project_masks(m, frames, spec)
            o = (ob["A"].roi | ob["B"].roi)[3:5].any(0)
        own.append(o)
        assert not (o & ~reg).any()                      # covers each twin's objects inside the window
    assert not np.array_equal(own[0], own[1])            # the per-twin regions would differ (the bug)
    assert np.array_equal(reg, own[0] | own[1])
    if spec is None:
        assert not reg[-4:].any()                        # aux bodies are never part of the region
        late = np.isin(twins[0][0][11], [1]) & ~np.isin(twins[0][0][5:10], [1, 2]).any(0)
        assert late.any() and not (late & reg).any()     # frames after the clean window do not enter


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


# ------------------------------------------------------------------ P-pred-real (spec §5.2), tiny random models

def _tiny_vjepa():
    torch = pytest.importorskip("torch")
    from transformers import VJEPA2Config, VJEPA2Model
    from physlaw.models.vjepa2 import VJEPA2Adapter
    torch.manual_seed(0)
    conf = VJEPA2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, mlp_ratio=2.0, crop_size=48,
                        frames_per_clip=8, patch_size=16, tubelet_size=2, pred_hidden_size=48,
                        pred_num_hidden_layers=1, pred_num_attention_heads=2, pred_num_mask_tokens=1)
    conf._attn_implementation = "sdpa"
    cfg = {"name": "tiny", "input": {"resolution": 48, "num_frames": 8, "fps": 24},
           "token_grid": {"patch": 16, "tubelet": 2}, "layers": []}
    return torch, VJEPA2Adapter(cfg, VJEPA2Model(conf).eval(), [0.5] * 3, [0.25] * 3, "cpu")


def _tiny_videomae():
    torch = pytest.importorskip("torch")
    from physlaw.models.videomae2 import VideoMAEv2Adapter, _build
    torch.manual_seed(0)
    model = _build(torch)(enc_dim=32, enc_depth=2, enc_heads=2, dec_dim=16, dec_depth=1, dec_heads=2,
                          patch=16, tubelet=2).eval()
    cfg = {"name": "tiny", "input": {"resolution": 48, "num_frames": 8, "fps": 8},
           "token_grid": {"patch": 16, "tubelet": 2}, "layers": []}
    return torch, VideoMAEv2Adapter(cfg, model, "cpu")


@pytest.mark.parametrize("make", [_tiny_vjepa, _tiny_videomae])
def test_pred_real_masks_only_its_targets(make):
    """Pass k must not see the pixels of the positions it predicts, and must see the other post tokens."""
    torch, m = make()
    from physlaw.models.base import spatial_mask_groups
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, (8, 48, 48, 3), dtype=np.uint8)
    k = 4                                            # context frames: token-frames 0, 1
    fin = -1
    base = m.pred_real(frames, k, n_masks=3)[fin]
    assert base.shape == (2, 3, 3, base.shape[-1])
    grp = spatial_mask_groups(9, 3)[0]
    edited = frames.copy()
    for s in grp:                                    # change group-0 patches in the post frames only
        r, c = divmod(int(s), 3)
        edited[k:, 16 * r:16 * r + 16, 16 * c:16 * c + 16] = 255 - edited[k:, 16 * r:16 * r + 16, 16 * c:16 * c + 16]
    new = m.pred_real(edited, k, n_masks=3)[fin]
    flat_b, flat_n = base.reshape(2, 9, -1), new.reshape(2, 9, -1)
    assert torch.allclose(flat_b[:, grp], flat_n[:, grp], atol=1e-5)
    other = np.setdiff1d(np.arange(9), grp)
    assert not torch.allclose(flat_b[:, other], flat_n[:, other], atol=1e-4)
    # the context-only forecast never sees post frames
    assert torch.equal(m.forecast(frames, k)[fin], m.forecast(edited, k)[fin])


@pytest.mark.parametrize("make", [_tiny_vjepa, _tiny_videomae])
def test_surprise_per_frame_record(make):
    """surprise()'s per_frame block (L2 clean window): one value per post token-frame whose weighted mean is the
    clip-level value, over all tokens and over the object mask (VideoMAE tube_*_obj: approximately, the object
    positions masked by a tube differ between draws)."""
    torch, m = make()
    frames = np.random.default_rng(0).integers(0, 255, (8, 48, 48, 3), dtype=np.uint8)
    obj = np.zeros((2, 3, 3), bool)
    obj[0, :2] = obj[1, 1] = True
    s = m.surprise(m.to_tensor(frames), 4, obj)
    pf = s.pop("per_frame")
    assert set(pf["surprise"]) == set(s) == set(pf["weight"]) and any(k.endswith("_obj") for k in s)
    for k, v in s.items():
        f, w = np.array(pf["surprise"][k], float), np.array(pf["weight"][k], float)
        ok = np.isfinite(f) & (w > 0)
        assert len(f) == 2 and ok.any()
        tol = 0.25 if k.startswith("tube_") and k.endswith("_obj") else 1e-5
        assert (w[ok] * f[ok]).sum() / w[ok].sum() == pytest.approx(v, rel=tol), k
    if "l1_obj" in s:
        assert pf["weight"]["l1_obj"] == [6.0, 3.0] and pf["weight"]["l1_all"] == [9.0, 9.0]


def test_vjepa_visible_encoder_matches_prefix_encoding():
    torch, m = _tiny_vjepa()
    x = m.to_tensor(np.random.default_rng(1).integers(0, 255, (8, 48, 48, 3), dtype=np.uint8))
    with torch.no_grad():
        ref = m.model(pixel_values_videos=x[:, :4], skip_predictor=True).last_hidden_state
        got = m._encode_visible(x, torch.arange(18))
    assert torch.allclose(ref, got, atol=1e-5)


def test_contact_pool_uses_pre_event_information_only():
    """contact.* weights depend only on the last pre-contact ROIs: moving the balls after contact changes
    roi_*.post but leaves contact.post untouched (no location leak), and the pool covers the contact region."""
    from types import SimpleNamespace
    from physlaw.tokens.pool import pool_weights
    T, gh, gw = 6, 16, 16
    phases = ["pre", "pre", "pre", "post", "post", "post"]
    def objs(shift):
        a = np.zeros((T, gh, gw), bool); b = np.zeros((T, gh, gw), bool)
        for t in range(T):
            d = 0 if t < 3 else shift * (t - 2)
            a[t, 8, max(0, 6 - d)] = True; b[t, 8, min(gw - 1, 9 + d)] = True
        return {"A": SimpleNamespace(roi=a), "B": SimpleNamespace(roi=b)}
    w1, w2 = pool_weights(objs(1), phases), pool_weights(objs(3), phases)
    assert not np.allclose(w1["roi_A.post"], w2["roi_A.post"])          # roi follows the true outcome
    assert np.allclose(w1["contact.post"], w2["contact.post"])          # contact does not
    assert w1["contact.post"].sum() == pytest.approx(1.0)
    assert (w1["contact.post"][3:, 8, 6] > 0).all() and (w1["contact.post"][:3] == 0).all()
