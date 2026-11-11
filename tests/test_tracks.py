"""Output-track assignment (spec §4.2 group 4 / §6, physlaw/models/track_a.py): config consistency, the context
frames each track A generator is conditioned on, VACE's mask, LTX's frame counts, stage 07's refusal rule.
Offline: no weights; pipelines are mocked (the PhysRVG clamp runs on a tiny random Wan)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from physlaw.config import load_config
from physlaw.models import track_a as T

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "configs" / "models"

ROSTER_A = {"cosmos_predict25_2b": "native", "physrvg": "native", "wan22_ti2v_5b_physrvg": "hack",
            "wan21_vace_1_3b": "variant", "wan21_vace_14b": "variant", "ltx_video_13b_distilled": "native"}
ROSTER_B = {"wan22_i2v_a14b", "wan21_i2v_14b", "cogvideox15_5b_i2v"}

# sample label: contact (first post-contact master frame) at 120, 48 fps master, 6 s clip
LABELS = {"scene_constants": {"frame_rate": 48, "num_frames": 289}, "physlaw": {"first_post_contact_frame": 120}}


def cfg(stem):
    return load_config(MODELS / f"{stem}.yaml")


def generator_stems():
    return sorted(p.stem for p in MODELS.glob("*.yaml") if T.is_generator(load_config(p)))


# ------------------------------------------------------------------ configs

def test_every_generator_states_track_and_mode_explicitly():
    stems = generator_stems()
    assert len(stems) >= 16
    for s in stems:
        raw = (MODELS / f"{s}.yaml").read_text()
        assert "\noutput_track:" in raw and "\ntrack_a_mode:" in raw, f"{s}: not explicit in its own file"
        T.track_fields(cfg(s))                                   # valid values


def test_track_rosters():
    tracks = {s: T.track_fields(cfg(s)) for s in generator_stems()}
    assert {s: m for s, (t, m) in tracks.items() if t == "A"} == ROSTER_A
    assert {s for s, (t, _) in tracks.items() if t == "B"} == ROSTER_B
    for s in ("cogvideox_2b", "cogvideox_5b", "videorepa_5b", "wan21_t2v_1_3b", "wan21_t2v_14b", "wan22_ti2v_5b"):
        assert tracks[s][0] == "none"
    ev = load_config(REPO / "configs" / "eval" / "outputs.yaml")
    assert set(ev["track_a"]["models"]) == set(ROSTER_A) and set(ev["track_b"]["models"]) == ROSTER_B
    assert "momentum" in ev                                       # other agents' section kept


def test_track_a_adapters_resolve():
    pytest.importorskip("torch")
    from physlaw.models import registry
    names = {s: registry.adapter_class(cfg(s)).__name__ for s in ROSTER_A}
    assert names == {"cosmos_predict25_2b": "CosmosPredictAdapter", "physrvg": "WanAdapter",
                     "wan22_ti2v_5b_physrvg": "WanAdapter", "wan21_vace_1_3b": "WanVACEAdapter",
                     "wan21_vace_14b": "WanVACEAdapter", "ltx_video_13b_distilled": "LTXConditionAdapter"}
    assert registry.adapter_class(cfg("cogvideox15_5b_i2v"), role="track_b").__name__ == "CogVideoX15I2VPipelineAdapter"
    assert registry.adapter_class(cfg("cogvideox15_5b_i2v")).__name__ == "CogVideoX15I2VAdapter"   # stage 4 unchanged


def test_new_configs_official_settings():
    v = cfg("wan21_vace_1_3b")
    assert v["revision"] == "ec4d2cb062b548996b179d493fdd05340de702a1"
    assert v["local_path"].endswith("/Wan2.1-VACE-1.3B-diffusers")
    o = v["official"]
    assert (o["height"], o["width"], o["num_frames"], o["fps"]) == (480, 832, 81, 16)
    assert (o["num_inference_steps"], o["guidance_scale"], o["flow_shift"]) == (50, 5.0, 16.0)
    v14 = cfg("wan21_vace_14b")
    assert v14["hf_id"] == "Wan-AI/Wan2.1-VACE-14B-diffusers" and v14["official"] == o
    lt = cfg("ltx_video_13b_distilled")["official"]
    assert lt["guidance_scale"] == 1.0 and lt["height"] % 32 == 0 and lt["width"] % 32 == 0


# ------------------------------------------------------------------ context frames (contact at master frame 120)

def test_physrvg_context_is_its_trained_5_frames():
    for s in ("physrvg", "wan22_ti2v_5b_physrvg"):
        c = cfg(s)
        assert T.context_master_frames(c, LABELS) == [105, 108, 111, 114, 117]       # 16 fps: every 3rd frame
        fr = T.model_frames(c, LABELS)
        e = T.context_end(c, LABELS)
        assert fr[e - 1] == 117 and fr[e] >= 120                  # ends at the last pre-contact window frame


def test_vace_context_is_every_pre_contact_frame():
    for s in ("wan21_vace_1_3b", "wan21_vace_14b"):
        assert T.context_master_frames(cfg(s), LABELS) == list(range(0, 120, 3))      # 40 frames, last 117


def test_ltx_context_49_frames_8k1():
    c = cfg("ltx_video_13b_distilled")
    ctx = T.context_master_frames(c, LABELS)
    assert ctx == list(range(22, 120, 2)) and len(ctx) == 49 and T.is_8k1(len(ctx))   # 24 fps, last 118
    assert T.is_8k1(c["official"]["num_frames"])
    assert c["official"]["fps"] == 24


def test_context_len_alignment_and_minimum():
    c = {"name": "x", "track_a": {"context": 49, "align": 8}}
    assert T.context_len(c, 60) == 49 and T.context_len(c, 30) == 25 and T.context_len(c, 8) == 1
    c = {"name": "x", "track_a": {"context": 5, "align": 4, "min_context": 5}}
    assert T.context_len(c, 24) == 5
    with pytest.raises(ValueError):
        T.context_len(c, 4)


def test_splice():
    real = np.zeros((20, 2, 2, 3), np.uint8)
    gen = np.full((49, 2, 2, 3), 7, np.uint8)
    v = T.splice(real, gen, 19, 49)
    assert v.shape[0] == 49 and (v[:19] == 0).all() and (v[19:] == 7).all()


# ------------------------------------------------------------------ VACE

def test_vace_mask_construction():
    ctx = np.random.default_rng(0).integers(0, 256, (40, 16, 32, 3), dtype=np.uint8)
    video, mask = T.vace_inputs(ctx, 81)
    clip = np.stack([np.asarray(f) for f in video])
    assert clip.shape == (81, 16, 32, 3) and len(mask) == 81
    assert np.array_equal(clip[:40], ctx) and (clip[40:] == 128).all()
    m = np.stack([np.asarray(x) for x in mask])
    assert m.shape == (81, 16, 32) and (m[:40] == 0).all() and (m[40:] == 255).all()
    pytest.importorskip("torch")
    from diffusers.video_processor import VideoProcessor
    vp = VideoProcessor(vae_scale_factor=8)                   # what WanVACEPipeline.preprocess_conditions does
    assert vp.get_default_height_width(video[0]) == (16, 32)
    v = vp.preprocess_video(video, 16, 32)
    mk = ((vp.preprocess_video(mask, 16, 32) + 1) / 2).clamp(0, 1)
    assert tuple(v.shape) == (1, 3, 81, 16, 32) and float(v[:, :, 40:].abs().max()) < 0.005   # grey fill ~ 0
    assert float(mk[:, :, :40].max()) == 0.0 and float(mk[:, :, 40:].min()) == 1.0


def test_vace_generate_mocked():
    pytest.importorskip("torch")
    from physlaw.models.wan_vace import WanVACEAdapter
    c = cfg("wan21_vace_1_3b")
    c["official"].update(height=16, width=32)
    seen = {}

    class Pipe:
        def __call__(self, video, mask, num_frames, **kw):
            seen.update(video=video, mask=mask, **kw)
            return type("O", (), {"frames": [np.full((num_frames, 16, 32, 3), 0.5, np.float32)]})()

    m = WanVACEAdapter(c, Pipe(), "cpu")
    x = np.random.default_rng(0).integers(0, 256, (81, 16, 32, 3), dtype=np.uint8)
    k = T.context_end(c, LABELS)
    v = m.generate(m.to_tensor(x), k, seed=0)
    assert k == 40 and v.shape == (81, 16, 32, 3) and (v == 128).all()
    assert len(seen["video"]) == 81 and np.array_equal(np.asarray(seen["video"][39]), x[39])
    assert sum(np.asarray(q).max() == 0 for q in seen["mask"]) == 40
    assert seen["num_inference_steps"] == 50 and seen["guidance_scale"] == 5.0


# ------------------------------------------------------------------ LTX

def test_ltx_generate_mocked():
    pytest.importorskip("torch")
    from physlaw.models.ltx import LTXConditionAdapter
    c = cfg("ltx_video_13b_distilled")
    c["official"].update(height=32, width=64)
    seen = {}

    class Pipe:
        def __call__(self, conditions, num_frames, **kw):
            seen.update(conditions=conditions, num_frames=num_frames, **kw)
            return type("O", (), {"frames": [np.full((num_frames, 32, 64, 3), 1.0, np.float32)]})()

    m = LTXConditionAdapter(c, Pipe(), "cpu")
    x = np.random.default_rng(0).integers(0, 255, (121, 32, 64, 3), dtype=np.uint8)
    k = T.context_end(c, LABELS)
    v = m.generate(m.to_tensor(x), k, seed=0)
    s = k - 49
    assert k == 60 and v.shape == (121, 32, 64, 3)
    assert np.array_equal(v[:s], x[:s]) and (v[s:] == 255).all()
    (cond,) = seen["conditions"]
    assert cond.frame_index == 0 and len(cond.video) == 49 and np.array_equal(np.asarray(cond.video[-1]), x[k - 1])
    assert T.is_8k1(seen["num_frames"]) and seen["frame_rate"] == 24 and seen["guidance_scale"] == 1.0
    assert seen["timesteps"] == [1000, 993, 987, 981, 975, 909, 725, 0.03]


def test_ltx_rejects_non_8k1():
    pytest.importorskip("torch")
    from physlaw.models.ltx import LTXConditionAdapter
    c = cfg("ltx_video_13b_distilled")
    c["official"]["num_frames"] = 120
    with pytest.raises(ValueError):
        LTXConditionAdapter(c, None, "cpu")


# ------------------------------------------------------------------ PhysRVG clamp (tiny random Wan2.2-style DiT)

def test_physrvg_clamp_holds_exactly_2_latents():
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    spec = importlib.util.spec_from_file_location("_tm", REPO / "tests" / "test_models.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    m = tm.tiny_wan(expand=True, guidance=None)
    m.cfg["track_a"] = {"context": 5, "align": 4, "min_context": 5}
    cleans = []
    orig = m._pass

    def spy(z_t, t, embeds, keep=(), clean=0):
        cleans.append((clean, z_t.shape[1]))
        return orig(z_t, t, embeds, keep=keep, clean=clean)

    m._pass = spy
    x = torch.rand(1, 3, 9, 32, 32) * 2 - 1
    v = m.generate(x, 7, seed=0)                 # context = window frames 2..6, real frames 0..1 in front
    assert v.shape == (9, 32, 32, 3)
    assert {c for c, _ in cleans} == {2} and {f for _, f in cleans} == {3}
    real = ((x[0, :, :2].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).numpy()
    assert np.array_equal(v[:2], real)


# ------------------------------------------------------------------ stage 07

def _stage07():
    spec = importlib.util.spec_from_file_location("_s07", REPO / "scripts" / "07_generate_outputs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stage07_refuses_track_a_without_hack():
    s07 = _stage07()
    assert s07.check_track(cfg("physrvg"), "a") == (True, "native")
    assert s07.check_track(cfg("wan22_ti2v_5b_physrvg"), "a") == (True, "hack")
    assert s07.check_track(cfg("wan21_vace_1_3b"), "a") == (True, "variant")
    for s in ("cogvideox_2b", "wan21_t2v_1_3b", "wan22_i2v_a14b", "cogvideox15_5b_i2v"):
        ok, msg = s07.check_track(cfg(s), "a")
        assert not ok and "--allow-hack" in msg
        assert s07.check_track(cfg(s), "a", allow_hack=True) == (True, "hack")
    assert s07.check_track(cfg("cogvideox15_5b_i2v"), "b") == (True, "hack")
    assert s07.check_track(cfg("wan22_i2v_a14b"), "b") == (True, "n/a")
    assert not s07.check_track(cfg("physrvg"), "b")[0]


# ------------------------------------------------------------------ real diffusers pipelines, tiny random weights

def test_vace_adapter_runs_real_pipeline_tiny():
    """The adapter's video / mask / settings go through diffusers' WanVACEPipeline unchanged (input checks,
    preprocessing, VACE context encoding, sampling, decode)."""
    torch = pytest.importorskip("torch")
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanVACEPipeline, WanVACETransformer3DModel
    from physlaw.models.wan_vace import WanVACEAdapter
    torch.manual_seed(0)
    vae = AutoencoderKLWan(base_dim=3, z_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
                           temperal_downsample=[False, True, True]).eval()
    tr = WanVACETransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12, in_channels=16,
                                   out_channels=16, text_dim=32, freq_dim=256, ffn_dim=32, num_layers=3,
                                   cross_attn_norm=True, qk_norm="rms_norm_across_heads", rope_max_seq_len=32,
                                   vace_layers=[0, 2], vace_in_channels=96).eval()
    sched = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=16.0)
    pipe = WanVACEPipeline(tokenizer=None, text_encoder=None, transformer=tr, vae=vae, scheduler=sched)
    pipe.set_progress_bar_config(disable=True)
    pe = torch.randn(1, 6, 32)
    pipe.encode_prompt = lambda *a, **k: (pe, torch.zeros_like(pe))
    c = cfg("wan21_vace_1_3b")
    c["official"].update(height=32, width=32, num_frames=9, num_inference_steps=2)
    m = WanVACEAdapter(c, pipe, "cpu")
    x = np.random.default_rng(0).integers(0, 256, (9, 32, 32, 3), dtype=np.uint8)
    v = m.generate(x, 6, seed=0)
    assert v.shape == (9, 32, 32, 3) and v.dtype == np.uint8


def test_ltx_adapter_runs_real_pipeline_tiny():
    """LTXConditionPipeline with the adapter's condition and the distilled settings (custom timesteps incl. 0.03,
    decode timestep / noise, guidance 1)."""
    torch = pytest.importorskip("torch")
    from diffusers import (AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler, LTXConditionPipeline,
                           LTXVideoTransformer3DModel)
    from physlaw.models.ltx import LTXConditionAdapter
    torch.manual_seed(0)
    tr = LTXVideoTransformer3DModel(in_channels=8, out_channels=8, patch_size=1, patch_size_t=1, num_attention_heads=4,
                                    attention_head_dim=8, cross_attention_dim=32, num_layers=1, caption_channels=32)
    vae = AutoencoderKLLTXVideo(in_channels=3, out_channels=3, latent_channels=8, block_out_channels=(8, 8, 8, 8),
                                decoder_block_out_channels=(8, 8, 8, 8), layers_per_block=(1, 1, 1, 1, 1),
                                decoder_layers_per_block=(1, 1, 1, 1, 1),
                                spatio_temporal_scaling=(True, True, False, False),
                                decoder_spatio_temporal_scaling=(True, True, False, False),
                                decoder_inject_noise=(False, False, False, False, False),
                                upsample_residual=(False, False, False, False), upsample_factor=(1, 1, 1, 1),
                                timestep_conditioning=False, patch_size=1, patch_size_t=1, encoder_causal=True,
                                decoder_causal=False)
    pipe = LTXConditionPipeline(scheduler=FlowMatchEulerDiscreteScheduler(), vae=vae, text_encoder=None,
                                tokenizer=None, transformer=tr)
    pipe.set_progress_bar_config(disable=True)
    pe, am = torch.randn(1, 8, 32), torch.ones(1, 8)
    pipe.encode_prompt = lambda *a, **k: (pe, am, torch.zeros_like(pe), am)
    c = cfg("ltx_video_13b_distilled")
    c["official"].update(height=32, width=32, num_frames=17)
    c["track_a"]["context"] = 9                   # tiny VAE: 4x temporal, so 8k+1 is also 4k+1
    m = LTXConditionAdapter(c, pipe, "cpu")
    x = np.random.default_rng(0).integers(0, 255, (17, 32, 32, 3), dtype=np.uint8)
    v = m.generate(x, 12, seed=0)                 # context = window frames 3..11
    assert v.shape == (17, 32, 32, 3) and np.array_equal(v[:3], x[:3])
