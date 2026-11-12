"""Model adapters on tiny random diffusers modules (CPU): DiT base (dit.py), Wan (wan.py), CogVideoX (cogvideox.py),
LoRA merging, weight pinning, the registry and the configs. No weights needed."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from physlaw.config import config_hash, load_config  # noqa: E402
from physlaw.models import dit, registry  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TXT_DIM = 32


# ------------------------------------------------------------------ tiny adapters

def _wan_cfg(expand: bool, guidance=5.0):
    return {"name": "tiny-wan", "kind": "diffusion", "dtype": "float32",
            "official": {"height": 32, "width": 32, "num_frames": 9, "fps": 8, "num_inference_steps": 4,
                         "guidance_scale": guidance, "scheduler": "UniPCMultistepScheduler", "negative_prompt": ""},
            "token_grid": {"patch": 2, "vae_spatial": 8, "vae_temporal": 4},
            "layers": [0, 1], "forecast_layers": [1], "encode_snr_db": [-8, 0, 8], "surprise": {"n_draws": 2},
            "noise_seed": 0, "prompt": "tiny"}


def _fake_text(self, text):
    g = torch.Generator().manual_seed(len(text))
    pe = torch.randn(1, 6, TXT_DIM, generator=g)
    return pe, (torch.zeros_like(pe) if (self.guidance or 0) > 1 else None)


def tiny_wan(expand=False, guidance=5.0):
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline, WanTransformer3DModel
    from physlaw.models.wan import WanAdapter
    torch.manual_seed(0)
    vae = AutoencoderKLWan(base_dim=3, z_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
                           temperal_downsample=[False, True, True]).eval()
    tr = WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12, in_channels=16,
                               out_channels=16, text_dim=TXT_DIM, freq_dim=256, ffn_dim=32, num_layers=2,
                               cross_attn_norm=True, qk_norm="rms_norm_across_heads", rope_max_seq_len=32).eval()
    sched = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0)
    pipe = WanPipeline(tokenizer=None, text_encoder=None, transformer=tr, vae=vae, scheduler=sched,
                       expand_timesteps=expand)
    cls = type("TinyWan", (WanAdapter,), {"_encode_text": _fake_text})
    return cls(_wan_cfg(expand, guidance), pipe, "cpu")


def tiny_cogvideox(rotary=False):
    from diffusers import (AutoencoderKLCogVideoX, CogVideoXDDIMScheduler, CogVideoXPipeline,
                           CogVideoXTransformer3DModel)
    from physlaw.models.cogvideox import CogVideoXAdapter
    torch.manual_seed(0)
    tr = CogVideoXTransformer3DModel(num_attention_heads=2, attention_head_dim=16, in_channels=4, out_channels=4,
                                     time_embed_dim=2, text_embed_dim=TXT_DIM, num_layers=2, sample_width=2,
                                     sample_height=2, sample_frames=9, patch_size=2, temporal_compression_ratio=4,
                                     max_text_seq_length=6, use_rotary_positional_embeddings=rotary).eval()
    vae = AutoencoderKLCogVideoX(in_channels=3, out_channels=3, down_block_types=("CogVideoXDownBlock3D",) * 4,
                                 up_block_types=("CogVideoXUpBlock3D",) * 4, block_out_channels=(8, 8, 8, 8),
                                 latent_channels=4, layers_per_block=1, norm_num_groups=2,
                                 temporal_compression_ratio=4).eval()
    pipe = CogVideoXPipeline(tokenizer=None, text_encoder=None, vae=vae, transformer=tr,
                             scheduler=CogVideoXDDIMScheduler(prediction_type="v_prediction",
                                                              rescale_betas_zero_snr=True, timestep_spacing="trailing"))
    cfg = {**_wan_cfg(False, 6.0), "name": "tiny-cog",
           "official": {"height": 32, "width": 32, "num_frames": 9, "fps": 8, "num_inference_steps": 3,
                        "guidance_scale": 6.0, "scheduler": "CogVideoXDDIMScheduler"}}
    cls = type("TinyCog", (CogVideoXAdapter,), {"_encode_text": _fake_text})
    return cls(cfg, pipe, "cpu")


def twins(T=9, k=5, seed=0):
    """Two uint8 clips [T, 32, 32, 3] sharing their first k frames."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 256, (T, 32, 32, 3), dtype=np.uint8)
    b = a.copy()
    b[k:] = rng.integers(0, 256, (T - k, 32, 32, 3), dtype=np.uint8)
    return a, b


# ------------------------------------------------------------------ DiT base

@pytest.mark.parametrize("make", [lambda: tiny_wan(False), lambda: tiny_wan(True), lambda: tiny_cogvideox(False),
                                  lambda: tiny_cogvideox(True)], ids=["wan21", "wan22-ti2v", "cog2b", "cog5b-rope"])
def test_dit_encode_surprise_forecast_twins(make):
    m = make()
    a, b = twins()
    k = 5
    xa, xb = m.to_tensor(a), m.to_tensor(b)
    ea = m.encode(xa)
    gh, gw = m.token_grid.grid_hw
    T = len(m.token_grid.token_frames(9))
    assert sorted(ea) == sorted([dit.key(db, l) for db in m.snrs for l in m.layers] +
                                [dit.latent_key(db) for db in m.snrs])
    for q, v in ea.items():
        C = m.vae.config.latent_channels if hasattr(m.vae.config, "latent_channels") else m.vae.config.z_dim
        assert v.shape[:3] == (T, gh, gw)
        if q.startswith("latent/"):
            assert v.shape[3] == C * m.p * m.p * m.lpt
    # the latent baseline equals encode()'s latent keys, without a transformer pass
    lb = m.latent_baseline(xa)
    assert all(torch.equal(lb[q], ea[q]) for q in lb)
    # twins: identical context latents and forecasts; surprise differs
    assert torch.equal(m.context_latents(xa, k), m.context_latents(xb, k))
    fa, va = m.forecast(xa, k, seed=0, return_video=True)
    fb, vb = m.forecast(xb, k, seed=0, return_video=True)
    kt = m.n_context_tokens(k)
    assert sorted(fa) == sorted(dit.key(db, l) for db in m.snrs for l in m.fc_layers)
    assert all(torch.equal(fa[q], fb[q]) and fa[q].shape[0] == T - kt for q in fa)
    assert np.array_equal(va, vb) and va.shape == (9, 32, 32, 3) and va.dtype == np.uint8
    obj = np.zeros((T - kt, gh, gw), bool)
    obj[:, :2] = True
    sa = m.surprise(xa, k, obj)
    sb = m.surprise(xb, k, None)
    pa, pb = sa.pop("per_frame"), sb.pop("per_frame")
    assert set(sa) == {"clean_mse_all", "noisy_mse_all", "clean_mse_obj", "noisy_mse_obj"}
    assert set(sb) == {"clean_mse_all", "noisy_mse_all"}
    assert sa["clean_mse_all"] != sb["clean_mse_all"] and np.isfinite(list(sa.values())).all()
    # per post token-frame: the weighted mean over frames is the clip-level value
    assert set(pa["surprise"]) == set(sa) and set(pb["surprise"]) == set(sb)
    for key, v in sa.items():
        f, w = np.array(pa["surprise"][key]), np.array(pa["weight"][key])
        assert len(f) == T - kt and (w > 0).all()
        assert (w * f).sum() / w.sum() == pytest.approx(v, rel=1e-5)
    assert pa["weight"]["clean_mse_obj"][0] == 2 * gw


def test_latent_tokens_layout():
    m = tiny_wan()
    z = torch.randn(1, 3, 16, 4, 6)                      # F, C, h, w
    t = m._latent_tokens(z)
    assert t.shape == (3, 2, 3, 16 * 4)
    f, i, j = 2, 1, 2
    patch = z[0, f, :, 2 * i:2 * i + 2, 2 * j:2 * j + 2]  # [C, 2, 2]
    assert torch.equal(t[f, i, j], patch.reshape(-1))
    # pooled over all tokens == the latent's spatial mean per (C, dy, dx)
    assert torch.allclose(t.reshape(-1, 64).mean(0).reshape(16, 2, 2), z[0].reshape(3, 16, 2, 2, 3, 2).mean((0, 2, 4)),
                          atol=1e-6)


def test_flow_snr_mapping():
    for db in (-20, -8, 0, 8):
        s = dit.flow_t_for_db(db) / 1000
        assert 10 * np.log10(((1 - s) / s) ** 2) == pytest.approx(db, abs=1e-9)


def test_wan_sampler_matches_pipeline():
    """Without context (k = 0) the sampler is the official WanPipeline, step for step (same initial latents)."""
    m = tiny_wan(False)
    x = m.to_tensor(twins()[0])
    _, lat = m._sample(x, 0, seed=3, capture=False)
    z0 = m.latents(x)
    init = m._noise(z0, 10_000 + 3).permute(0, 2, 1, 3, 4)
    ref = m.pipe(prompt_embeds=m.prompt_embeds, negative_prompt_embeds=m.neg_embeds, height=32, width=32,
                 num_frames=9, num_inference_steps=4, guidance_scale=5.0, latents=init, output_type="latent").frames
    assert torch.equal(lat.permute(0, 2, 1, 3, 4), ref)


def test_ti2v_clean_context_is_native_i2v():
    """expand_timesteps with one clean context latent = the official I2V mode (first frame clean, timestep 0)."""
    from diffusers import WanImageToVideoPipeline
    from PIL import Image
    m = tiny_wan(True)
    a = twins()[0]
    x = m.to_tensor(a)
    _, lat = m._sample(x, 1, seed=1, capture=False)
    p = m.pipe
    i2v = WanImageToVideoPipeline(tokenizer=None, text_encoder=None, image_encoder=None, image_processor=None,
                                  transformer=p.transformer, vae=p.vae, scheduler=p.scheduler, expand_timesteps=True)
    init = m._noise(m.latents(x), 10_000 + 1).permute(0, 2, 1, 3, 4)
    ref = i2v(image=Image.fromarray(a[0]), prompt_embeds=m.prompt_embeds, negative_prompt_embeds=m.neg_embeds,
              height=32, width=32, num_frames=9, num_inference_steps=4, guidance_scale=5.0, latents=init,
              output_type="latent").frames
    # only the image preprocessing differs (x / 255 * 2 - 1 vs x / 127.5 - 1): float rounding
    assert torch.allclose(lat.permute(0, 2, 1, 3, 4), ref, atol=1e-4)


def test_wan_i2v_generate_from_frame0():
    from diffusers import WanImageToVideoPipeline
    from physlaw.models.wan import WanI2VAdapter
    t = tiny_wan(True)
    p = t.pipe
    i2v = WanImageToVideoPipeline(tokenizer=None, text_encoder=None, image_encoder=None, image_processor=None,
                                  transformer=p.transformer, vae=p.vae, scheduler=p.scheduler, expand_timesteps=True)
    m = type("TinyI2V", (WanI2VAdapter,), {"_encode_text": _fake_text})(_wan_cfg(True), i2v, "cpu")
    a = twins()[0]
    v = m.generate(a, 1, seed=0)
    assert v.shape == (9, 32, 32, 3) and v.dtype == np.uint8
    assert np.array_equal(v, m.generate(m.to_tensor(a), 1, seed=0))     # uint8 frames or the input tensor
    assert m.generator and not m.internals


def test_capabilities_and_load_adapter_need():
    """Capability flags (base.Adapter) on every adapter class; load_adapter refuses a missing one before loading."""
    from physlaw.models.base import Adapter, ModelAdapter
    from physlaw.models.cogvideox import CogVideoX15I2VAdapter
    from physlaw.models.cogvideox_i2v import CogVideoX15I2VPipelineAdapter
    from physlaw.models.cosmos_predict import CosmosPredictAdapter
    from physlaw.models.ltx import LTXConditionAdapter
    from physlaw.models.videomae2 import VideoMAEv2Adapter
    from physlaw.models.vjepa2 import VJEPA2Adapter
    from physlaw.models.wan import WanAdapter, WanI2VAdapter
    from physlaw.models.wan_vace import WanVACEAdapter
    caps = lambda c: (c.internals, c.generator)
    assert caps(VJEPA2Adapter) == caps(VideoMAEv2Adapter) == (True, False)
    assert caps(WanAdapter) == caps(CogVideoX15I2VAdapter) == caps(CogVideoX15I2VPipelineAdapter) == (True, True)
    assert caps(WanI2VAdapter) == caps(CosmosPredictAdapter) == caps(WanVACEAdapter) == caps(LTXConditionAdapter) \
        == (False, True)
    assert not issubclass(CosmosPredictAdapter, ModelAdapter) and issubclass(CosmosPredictAdapter, Adapter)
    for name, need in (("cosmos_predict25_2b", "internals"), ("wan22_i2v_a14b", "internals"), ("vjepa2_vitl", "generator")):
        with pytest.raises(ValueError, match=f"has no {need}"):
            registry.load_adapter(load_config(REPO / "configs" / "models" / f"{name}.yaml"), need=need)
    with pytest.raises(KeyError, match="no adapter"):
        registry.adapter_class({"name": "x", "hf_id": "nobody/nothing"})


def test_videomae_base_loads_encoder_only(tmp_path):
    """A VideoMAEv2-Base-shaped model repo (config.json model_config + model.safetensors with ``model.*`` keys, no
    decoder) loads through the registry into the live VideoMAE-v2 adapter: encode() gives the block outputs, the
    adapter does not predict (stage 4 skips surprise / forecast), and a forecast is refused with a clear error."""
    from safetensors.torch import save_file
    from physlaw.models.videomae2 import VideoMAEv2Adapter, _build
    mc = {"embed_dim": 32, "depth": 2, "num_heads": 2, "mlp_ratio": 4, "patch_size": 16, "tubelet_size": 2,
          "qkv_bias": True, "init_values": 0.0, "use_learnable_pos_emb": False, "cos_attn": False}
    (tmp_path / "config.json").write_text(json.dumps({"model_config": mc}))
    torch.manual_seed(0)
    ref = _build(torch)(enc_dim=32, enc_depth=2, enc_heads=2, dec_depth=None, mlp_ratio=4, patch=16, tubelet=2)
    sd = {"model." + k[len("encoder."):]: v.contiguous() for k, v in ref.state_dict().items()}
    sd.update({"model.fc_norm.weight": torch.ones(32), "model.fc_norm.bias": torch.zeros(32)})
    save_file(sd, str(tmp_path / "model.safetensors"))
    cfg = load_config(REPO / "configs" / "models" / "videomae_v2_base.yaml",
                      [f"local_path={tmp_path}", "input.resolution=32", "input.num_frames=4", "layers=[0,1]"])
    assert registry.adapter_class(cfg) is VideoMAEv2Adapter
    with pytest.raises(RuntimeError, match="do not match"):        # the config pins the real weights
        registry.verify_weights(cfg)
    m = registry.load_adapter(cfg, device="cpu", verify=False, need="internals")
    assert m.internals and not m.predicts
    x = m.to_tensor(np.random.default_rng(0).integers(0, 256, (4, 32, 32, 3), dtype=np.uint8))
    a = m.encode(x)
    assert sorted(a) == [0, 1] and a[1].shape == (2, 2, 2, 32)
    with torch.no_grad():                                          # block 1 of the reference on all tokens
        h = ref.encoder.patch_embed(x) + m.pos_enc
        h = ref.encoder.blocks[1](ref.encoder.blocks[0](h))
    assert torch.allclose(a[1].reshape(-1, 32), h[0], atol=1e-5)
    with pytest.raises(RuntimeError, match="encoder-only"):
        m.forecast(x, 2)


@pytest.mark.parametrize("make", [lambda: tiny_wan(True), lambda: tiny_cogvideox(False)], ids=["wan22-ti2v", "cog2b"])
def test_stage4_diffusion_forecast_token_frame_pools(tmp_path, make):
    """Stage 4 stores the sampled forecast's frame pool per token-frame (fc_pooled_frame_t [T_post, K, D], rows from
    token-frame context_tokens on) beside fc_pooled, and L1's token-frame reader pairs every forecast key with the
    same key's real token-frame rows (pooled_frame_t) and the forecast rows' token times."""
    import importlib.util
    from physlaw.eval import l1_forecast_laws as L1
    spec = importlib.util.spec_from_file_location("stage4", REPO / "scripts" / "04_cache_activations.py")
    S4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S4)
    m = make()
    x, k = m.to_tensor(twins()[0]), 5
    kt, T, (gh, gw) = m.n_context_tokens(k), len(m.token_grid.token_frames(9)), m.token_grid.grid_hw
    phases = ["pre"] * kt + ["post"] * (T - kt)
    post = torch.tensor([p == "post" for p in phases], dtype=torch.float32).repeat_interleave(gh * gw)
    Wt = torch.stack([torch.ones(T * gh * gw), post])
    (tmp_path / "g").mkdir()
    new = S4.add_forecast(m, x, k, "g", None, "0", tmp_path, Wt / Wt.sum(1, keepdim=True), ["frame.all", "frame.post"],
                          phases, kt, {})
    acts = m.forecast(x, k, seed=0)
    fk = sorted(acts)
    F = new["fc_pooled_frame_t"].astype(np.float64)
    assert new["fc_keys"].tolist() == fk and F.shape == (T - kt, len(fk), acts[fk[0]].shape[-1])
    for j, q in enumerate(fk):
        assert np.allclose(F[:, j], acts[q].double().mean((1, 2)).numpy(), rtol=1e-2, atol=1e-3)
    assert np.allclose(new["fc_pooled"][1].astype(np.float64), F.mean(0), rtol=1e-2, atol=1e-3)   # frame.post
    assert (tmp_path / "g" / "gen_seed0.mp4").exists()
    # the cache record and L1's reader
    enc = m.encode(x)
    lk = sorted(q for q in enc if not q.startswith("latent/"))
    fr = [6 * i for i in range(9)]                                   # 48 fps master, 8 fps model
    np.savez(tmp_path / "g" / "t.npz", pooled_frame_t=S4.frame_pool_t([enc[q] for q in lk]), layers=np.array(lk),
             token_t_s=np.array(S4.token_times(fr, m.token_grid, 48.0)), frames=np.array(fr), phases=np.array(phases),
             **new)
    labels = {"g/t": {"per_frame": [{"A": {"velocity": [0.1 * f, 0.0, 0.0]}} for f in range(60)],
                      "scene_constants": {"frame_rate": 48}}}
    sp, frm = L1.load_token_frames(tmp_path, [{"file": "g/t.npz"}], ["g/t"], labels, m.cfg, None, ["A"])
    assert sorted(sp) == fk                                          # forecast_layers are among the encode layers
    for j, q in enumerate(fk):
        assert sp[q]["real"][0].shape == (T, F.shape[-1]) and np.allclose(sp[q]["fc"][0], F[:, j])
    assert np.allclose(frm["t_fc"][0], frm["t"][0][kt:]) and len(frm["t_fc"][0]) == T - kt


def test_wan_no_cfg_single_pass():
    m = tiny_wan(True, guidance=None)
    assert m.neg_embeds is None and not m._cfg_on()
    acts = m.forecast(m.to_tensor(twins()[0]), 5)
    assert acts


def test_cosmos_generate_alignment():
    """Video2World gets the context's frames, returns 93 frames from the last 5 context frames on: generate() puts
    the real frames before them in front, so the clip is aligned with input_frames()."""
    from physlaw.models.cosmos_predict import CosmosPredictAdapter
    cfg = load_config(REPO / "configs" / "models" / "cosmos_predict25_2b.yaml")
    cfg["official"].update(height=32, width=32)
    seen = {}

    class Pipe:
        vae = type("V", (), {"dtype": torch.float32})()

        def __call__(self, video, num_frames, **kw):
            seen.update(video=video, **kw)
            return type("O", (), {"frames": [np.full((num_frames, 32, 32, 3), 0.5, np.float32)]})()

    m = CosmosPredictAdapter(cfg, Pipe(), "cpu")
    a = np.random.default_rng(0).integers(0, 256, (93, 32, 32, 3), dtype=np.uint8)
    k = 41
    v = m.generate(m.to_tensor(a), k, seed=0)
    assert v.shape == (93, 32, 32, 3) and seen["video"].shape[2] == k
    assert np.array_equal(v[: k - 5], a[: k - 5]) and (v[k - 5:] == 128).all()
    assert seen["num_latent_conditional_frames"] == 2 and seen["guidance_scale"] == 7.0


# ------------------------------------------------------------------ LoRA / weights

class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.ModuleDict({"to_q": torch.nn.Linear(6, 4)}) for _ in range(2)])


def _lora(prefix, rank=2):
    g = torch.Generator().manual_seed(0)
    sd = {}
    for b in range(2):
        sd[f"{prefix}blocks.{b}.to_q.lora_A.weight"] = torch.randn(rank, 6, generator=g)
        sd[f"{prefix}blocks.{b}.to_q.lora_B.weight"] = torch.randn(4, rank, generator=g)
    return sd


@pytest.mark.parametrize("fmt", ["diffusers", "peft"])
def test_merge_lora(tmp_path, fmt):
    from safetensors.torch import save_file
    net = _Net()
    W0 = [blk["to_q"].weight.detach().clone() for blk in net.blocks]
    if fmt == "diffusers":
        sd = _lora("transformer.")
        path = tmp_path / "pytorch_lora_weights.safetensors"
        save_file(sd, str(path))
        want = 0.5
        n = dit.merge_lora(net, path, 0.5)
    else:
        sd = _lora("base_model.model.")
        save_file(sd, str(tmp_path / "adapter_model.safetensors"))
        (tmp_path / "adapter_config.json").write_text(json.dumps({"lora_alpha": 8, "r": 2}))
        want = 4.0
        n = dit.merge_lora(net, tmp_path, 1.0)             # a directory works too
    assert n == 2
    pre = "transformer." if fmt == "diffusers" else "base_model.model."
    for b, blk in enumerate(net.blocks):
        A, B = sd[f"{pre}blocks.{b}.to_q.lora_A.weight"], sd[f"{pre}blocks.{b}.to_q.lora_B.weight"]
        assert torch.allclose(blk["to_q"].weight, W0[b] + want * B @ A, atol=1e-5)


def test_merge_lora_peft_variants(tmp_path):
    """PEFT adapters beyond plain LoRA, merged as PEFT does: rsLoRA (alpha / sqrt(r)), alpha_pattern (per module),
    lora_bias (b += s b_B) and DoRA (W <- m / ||W + s B A||_row (W + s B A))."""
    from safetensors.torch import save_file
    sd = _lora("base_model.model.")
    g = torch.Generator().manual_seed(1)
    sd["base_model.model.blocks.0.to_q.lora_B.bias"] = torch.randn(4, generator=g)
    sd["base_model.model.blocks.1.to_q.lora_magnitude_vector"] = torch.rand(4, generator=g) + 0.5
    save_file(sd, str(tmp_path / "adapter_model.safetensors"))
    (tmp_path / "adapter_config.json").write_text(json.dumps({"lora_alpha": 8, "r": 2, "use_rslora": True,
                                                              "alpha_pattern": {"blocks.1.to_q": 2}}))
    net = _Net()
    W0 = [blk["to_q"].weight.detach().clone() for blk in net.blocks]
    b0 = net.blocks[0]["to_q"].bias.detach().clone()
    assert dit.merge_lora(net, tmp_path, 0.5) == 2
    BA = lambda b: sd[f"base_model.model.blocks.{b}.to_q.lora_B.weight"] @ sd[f"base_model.model.blocks.{b}.to_q.lora_A.weight"]
    s0, s1 = 0.5 * 8 / np.sqrt(2), 0.5 * 2 / np.sqrt(2)
    assert torch.allclose(net.blocks[0]["to_q"].weight, W0[0] + s0 * BA(0), atol=1e-5)
    assert torch.allclose(net.blocks[0]["to_q"].bias, b0 + s0 * sd["base_model.model.blocks.0.to_q.lora_B.bias"], atol=1e-5)
    Wn = W0[1] + s1 * BA(1)
    mag = sd["base_model.model.blocks.1.to_q.lora_magnitude_vector"]
    assert torch.allclose(net.blocks[1]["to_q"].weight, mag[:, None] * Wn / Wn.norm(dim=1, keepdim=True), atol=1e-5)
    assert torch.allclose(net.blocks[1]["to_q"].weight.norm(dim=1), mag, atol=1e-5)    # DoRA: row norms = m
    save_file({**sd, "base_model.model.blocks.0.to_q.alpha": torch.ones(1)}, str(tmp_path / "adapter_model.safetensors"))
    with pytest.raises(ValueError, match="unexpected LoRA key"):
        dit.merge_lora(_Net(), tmp_path)


def test_merge_lora_rejects_unknown_target(tmp_path):
    from safetensors.torch import save_file
    sd = _lora("transformer.")
    sd["transformer.nope.lora_A.weight"] = torch.zeros(2, 6)
    sd["transformer.nope.lora_B.weight"] = torch.zeros(4, 2)
    save_file(sd, str(tmp_path / "l.safetensors"))
    with pytest.raises(KeyError):
        dit.merge_lora(_Net(), tmp_path / "l.safetensors")


def test_load_dit_weights_strict(tmp_path):
    from safetensors.torch import save_file
    a, b = _Net(), _Net()
    save_file({k: v.contiguous() for k, v in a.state_dict().items()}, str(tmp_path / "w.safetensors"))
    assert dit.load_dit_weights(b, tmp_path / "w.safetensors") == 4
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
    save_file({"blocks.0.to_q.weight": torch.zeros(4, 6)}, str(tmp_path / "bad.safetensors"))
    with pytest.raises(KeyError):
        dit.load_dit_weights(b, tmp_path / "bad.safetensors")


def test_verify_weights(tmp_path):
    f = tmp_path / "m" / "transformer" / "w.safetensors"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"abc")
    good = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    cfg = {"local_path": str(tmp_path / "m"), "sha256": {"transformer/w.safetensors": good, "missing.bin": "0"}}
    assert registry.verify_weights(cfg) == 1                 # files not on disk are skipped
    assert (f.parent / "w.safetensors.sha256").read_text().split()[0] == good
    cfg["sha256"]["transformer/w.safetensors"] = "0" * 64
    with pytest.raises(RuntimeError):
        registry.verify_weights(cfg)


# ------------------------------------------------------------------ configs / registry

CONFIGS = sorted((REPO / "configs" / "models").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=[p.stem for p in CONFIGS])
def test_model_configs_dispatch_and_pin(path):
    cfg = load_config(path)
    if cfg.get("kind") in ("positive_control",) or cfg.get("hf_id") is None:
        pytest.skip("trained here, no Hub pin")
    cls = registry.adapter_class(cfg)
    assert cls.__name__.endswith("Adapter")
    assert cfg.get("revision"), f"{path.name}: no pinned revision"
    pins = dict(cfg.get("sha256") or {})
    for blk in ("checkpoint", "dit_weights", "lora"):
        if cfg.get(blk) and not (blk == "lora" and cfg.get("positive_control")):   # PC2: our own training output
            assert cfg[blk].get("sha256"), f"{path.name}: {blk} has no sha256"
    assert pins or cfg.get("checkpoint"), f"{path.name}: no sha256 pins"
    assert all(len(h) == 64 for h in pins.values())
    assert set(cfg.get("forecast_layers") or []) <= set(cfg.get("layers") or []), \
        f"{path.name}: forecast_layers must be encode layers (L1 / L3 read them on real rows, R1)"
    if cfg.get("kind") in ("diffusion", "generator"):
        off = cfg["official"]
        for k_ in ("height", "width", "num_frames", "fps", "num_inference_steps", "scheduler"):
            assert k_ in off, f"{path.name}: official.{k_} missing"
        from physlaw.models.base import token_spec
        ts = token_spec(cfg)
        assert ts.input_hw[0] % ts.patch == 0 and ts.input_hw[1] % ts.patch == 0


def _family_labels():
    """Minimal labels of every family / sub-family in configs/data (the fields the prompts may read): collision
    families with their two-sphere objects (materials as rendered: rubber, metal, F3's neutral)."""
    out = []
    for path in sorted((REPO / "configs" / "data").glob("f*.yaml")):
        c = load_config(path)
        for sub in list(c.get("subfamilies") or {}) or [""]:
            lab = {"physlaw": {"family": c.family, "subfamily": sub or None}}
            if c.family in dit.COLLISION_FAMILIES:
                for mats in (("rubber", "metal"), ("neutral", "neutral")):
                    out.append({**lab, "b_moving": True, "objects": {k: {"radius": r, "material": m} for k, r, m in
                                                                     zip("AB", (0.4, 0.5), mats)}})
            else:
                out.append(lab)
    return out


def test_prompts_for_every_family():
    """Every prompt template and the stage-4 clip_prompt work on every family's labels; F1-F4 keep the configured
    collision text, other families get their own neutral description (no balls colliding), a model card's own
    prompt (PhysRVG) is kept for every family."""
    labs = _family_labels()
    assert {lab["physlaw"]["family"] for lab in labs} >= {"f1", "f2", "f3", "f4", "f7", "f9", "f10", "f11", "f12"}
    t2v = load_config(REPO / "configs" / "models" / "wan21_t2v_1_3b.yaml")
    card = load_config(REPO / "configs" / "models" / "physrvg.yaml")
    for lab in labs:
        texts = [f(lab) for f in dit.PROMPTS.values()] + [dit.clip_prompt(t2v, lab), dit.clip_prompt(card, lab)]
        assert all(isinstance(t, str) and t for t in texts)
        assert texts[-1] == card["prompt"]
        if lab["physlaw"]["family"] in dit.COLLISION_FAMILIES:
            assert texts[-2] == t2v["prompt"] == dit.DEFAULT_PROMPT and "collide" in texts[0]
        else:
            assert texts[-2] == dit.family_scene(lab) and not any("collide" in t for t in texts[:-1])
    f1 = next(lab for lab in labs if lab["physlaw"]["family"] == "f1")      # rubber A, metal B
    assert dit.minimal_prompt(f1) == ("Overhead view of a plain flat grey floor with two balls on it: a smaller "
                                           "matte red rubber ball and a larger shiny grey metal ball. The two balls "
                                           "collide and then separate.")
    with pytest.raises(KeyError, match="no prompt description"):
        dit.minimal_prompt({"physlaw": {"family": "f99"}})


def test_pins_do_not_change_config_hash():
    cfg = load_config(REPO / "configs" / "models" / "cogvideox_2b.yaml")
    assert config_hash(cfg) == "72e33cd011b3"                 # the hash of every cache/cogvideox-2b record
    cfg2 = dict(cfg)
    cfg2.pop("sha256", None)
    assert config_hash(cfg2) == config_hash(cfg)
