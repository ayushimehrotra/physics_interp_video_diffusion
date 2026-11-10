"""CogVideoX adapters (spec §4.1, milestone 7): CogVideoX-2b, CogVideoX-5B (+ VideoREPA-5B, a LoRA on it) and
CogVideoX1.5-5B-I2V. Shared DiT machinery (encode / surprise / latent baseline / LoRA) lives in dit.py.

Official settings only (configs/models/cogvideox_2b.yaml, from the model card and the checkpoint's
scheduler_config.json): 720 x 480, 49 frames at 8 fps, CogVideoXDDIMScheduler (v-prediction,
zero-terminal-SNR, snr_shift 3, trailing), 50 steps, guidance 6, no dynamic CFG. The 6.0 s master
clip IS the model's native 6.0 s clip, so all 49 frames are used (every 6th master frame).

Text. T2V needs a prompt: one fixed scene description for every clip (``cfg.prompt``), negative ""
for CFG. Twins therefore differ only in their video.

Latents. The VAE is temporally causal: latent frame 0 = input frame 0, latent j >= 1 = frames
4j-3..4j; 8x spatial; transformer patch 2 -> token grid [13, 30, 45], 1920-d. The token grid is
TokenGridSpec(patch 16, tubelet 4, first_tubelet 1). The VAE's convolutions are causal but its
GroupNorms pool over time, so full-clip latents of pre-contact frames differ slightly between twins
(max |diff| 0.02 vs 1.7 after contact), enough to make sampling diverge. surprise() and forecast()
therefore take the CONTEXT latents from encoding the context frames alone (``context_latents``):
they depend only on pre-contact pixels, which twins share byte-for-byte.

encode(noise_level=dB): forward-noise the clean latents to the timestep whose SNR = alpha_bar /
(1 - alpha_bar) is closest to the requested SNR (fixed noise draw, identical for every clip), one
conditional transformer pass, block outputs at ``cfg.layers``. Keys: "snr{dB}/L{layer}".

surprise(): the training denoising loss on post-contact latents: x0-error weighted by 1/(1-alpha_bar)
(CogVideoX's own loss), averaged over the encode SNR timesteps and ``surprise.n_draws`` fixed noise
draws (identical across twins). ``clean_*``: context latents clean, as the spec asks (the timestep
embedding then mismatches the context: out of distribution, but the same for every twin);
``noisy_*``: whole clip noised with the same draw (context identical across twins anyway). Regions:
all post-context tokens, and ``obj`` (the balls' ROI tokens).

forecast() / generate(): official sampling with the context latents REPLACED at every step by the
true context latents noised to that step (fixed noise; the standard replacement method for
continuation with a T2V model). Activations of the conditional branch are captured at the sampling
steps whose timestep is nearest each encode SNR, for ``cfg.forecast_layers``: same keys as encode(),
so probes fit on encode() of real video apply to them. The final latents decode to the generated clip.
"""
from __future__ import annotations

import math

import numpy as np

from physlaw.models.dit import (DEFAULT_PROMPT, LOOKS, MINIMAL_LOOKS, DiTAdapter, apply_weight_overrides,  # noqa: F401
                                ddpm_t_for_db, key, minimal_prompt, scene_prompt, source, text_encoder_source)

_source = source   # old name, kept for scripts/experiments


def _load_pipe(pipe_cls, cfg, default_dtype: str, device: str):
    """Load every component shard by shard straight onto ``device`` (device_map). A CPU-side load followed by
    ``.to(device)`` holds T5 (9.5 GB) plus the transformer in host RAM at once: 13.7 GB peak for 2B, more
    than the current machine's 15 GB for 1.5-5B. Same weights and dtype either way. T5 is not loaded here:
    set_prompt() loads it on demand (text_encoder_source) and frees it after encoding."""
    import torch
    src, kw = source(cfg)
    dtype = getattr(torch, cfg.get("dtype", default_dtype))
    pipe = pipe_cls.from_pretrained(src, torch_dtype=dtype, device_map=device, text_encoder=None, **kw)
    apply_weight_overrides(pipe.transformer, cfg)
    return pipe


def _official_scheduler(pipe, cfg):
    """The CogVideo repo's inference/cli_demo.py: DDIM for 2B, CogVideoXDPMScheduler (trailing) with
    use_dynamic_cfg for the 5B models; VAE tiling as in the model-card examples when configured."""
    off = cfg["official"]
    if off["scheduler"] == "CogVideoXDPMScheduler" and pipe.scheduler.__class__.__name__ != off["scheduler"]:
        from diffusers import CogVideoXDPMScheduler
        pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    if cfg.get("vae_tiling"):
        pipe.vae.enable_tiling()
    if pipe.scheduler.__class__.__name__ != off["scheduler"]:
        raise RuntimeError(f"scheduler {pipe.scheduler.__class__.__name__} != official {off['scheduler']}")
    pipe.set_progress_bar_config(disable=True)


class CogVideoXAdapter(DiTAdapter):
    """CogVideoX T2V (2b; 5b with 3-D rotary embeddings): DDPM v-prediction, latents = VAE mode x scaling_factor."""

    def __init__(self, cfg, pipe, device):
        super().__init__(cfg, pipe, device)
        self.dynamic_cfg = bool(cfg["official"].get("use_dynamic_cfg", False))
        self.dpm = self.sched.__class__.__name__ == "CogVideoXDPMScheduler"
        self._rot = self._rotary() if self.tr.config.use_rotary_positional_embeddings else None

    def _rotary(self):
        """3-D rotary embeddings from the pipeline helper, for the official clip length (5B models)."""
        n_lat = (self.n_frames - 1) // 4 + 1
        return self.pipe._prepare_rotary_positional_embeddings(self.H, self.W, n_lat, self.device)

    @classmethod
    def load(cls, cfg, device: str = "cuda") -> "CogVideoXAdapter":
        from diffusers import CogVideoXPipeline
        pipe = _load_pipe(CogVideoXPipeline, cfg, "float16", device)
        _official_scheduler(pipe, cfg)
        return cls(cfg, pipe, device)

    # ------------------------------------------------------------------ family hooks (dit.py)

    def _init_schedule(self):
        self.alphas_cumprod = self.sched.alphas_cumprod.double()

    def _t_for_db(self, db):
        return ddpm_t_for_db(self.alphas_cumprod, db)

    def _encode_text(self, text: str):
        """T5 (9.5 GB) is freed after each encode instead of being parked in host RAM (15 GB on the current
        machine), and reloaded from disk straight to the GPU only for a prompt not yet encoded."""
        torch = self.torch
        if self.pipe.text_encoder is None:
            from transformers import T5EncoderModel
            src, kw = text_encoder_source(self.cfg)
            self.pipe.text_encoder = T5EncoderModel.from_pretrained(
                src, subfolder="text_encoder", torch_dtype=self.dtype, device_map=self.device, **kw)
        with torch.no_grad():
            pe, ne = self.pipe.encode_prompt(text, "", do_classifier_free_guidance=True, num_videos_per_prompt=1,
                                             max_sequence_length=226, device=self.device)
        self.pipe.text_encoder = None
        return pe.to(self.dtype), ne.to(self.dtype)

    def _vae_encode(self, x):
        with self.torch.no_grad():
            z = self.vae.encode(x).latent_dist.mode() * self.vae.config.scaling_factor
        return z.permute(0, 2, 1, 3, 4).contiguous()

    def _vae_decode(self, lat):
        return self.vae.decode(lat.permute(0, 2, 1, 3, 4) / self.vae.config.scaling_factor).sample

    def _add_noise(self, z0, eps, t):
        a = self.alphas_cumprod[t].to(z0.device)
        return (a.sqrt() * z0.double() + (1 - a).sqrt() * eps.double()).to(z0.dtype)

    def _x0(self, z_t, v, t):
        a = self.alphas_cumprod[t].to(z_t.device)
        return (a.sqrt() * z_t.double() - (1 - a).sqrt() * v.double())

    def _train_err(self, z_in, v, z0, eps, t):
        """CogVideoX's own loss: x0 error weighted by 1 / (1 - alpha_bar)."""
        w = 1.0 / max(1 - float(self.alphas_cumprod[t]), 1e-6)
        return w * (self._x0(z_in, v, t) - z0.double()) ** 2

    def _pass(self, z_t, t, embeds, keep=(), clean: int = 0):
        """One transformer pass; returns (v-prediction, {layer: video-token hidden [B, N, D]}). ``clean`` is
        ignored: one timestep for every token (the clean-context surprise is out of distribution for CogVideoX)."""
        torch = self.torch
        hs, hooks = {}, []
        for l in keep:
            hooks.append(self.tr.transformer_blocks[l].register_forward_hook(
                lambda mod, inp, out, l=l: hs.__setitem__(l, out[0])))
        try:
            z_in, extra = self._condition(z_t)
            with torch.no_grad():
                v = self.tr(hidden_states=z_in, encoder_hidden_states=embeds,
                            timestep=torch.full((z_t.shape[0],), t, device=z_t.device, dtype=torch.long),
                            return_dict=False, **extra)[0]
        finally:
            for h in hooks:
                h.remove()
        return v, hs

    def _condition(self, z_t):
        """Model input and extra kwargs (T2V: the 5B's rotary embeddings, none for 2b)."""
        return z_t, ({"image_rotary_emb": self._rot} if self._rot is not None else {})

    # ------------------------------------------------------------------ sampling (forecast / generate)

    def _guide_grad(self, lat, t, z0, kl):
        """Reconstruction guidance (Ho et al. 2022, 'Video Diffusion Models', Sec. 3.1): gradient w.r.t. the
        noisy latents of || x0_hat(context) - x0(context) ||^2, from one conditional pass WITH grad
        (gradient checkpointing). Only the generated part of the gradient is used by the caller."""
        torch = self.torch
        z = lat.detach().clone().requires_grad_(True)
        z_in, extra = self._condition(z)
        with torch.enable_grad():
            v = self.tr(hidden_states=self.sched.scale_model_input(z_in, t), encoder_hidden_states=self.prompt_embeds,
                        timestep=torch.full((1,), t, device=z.device, dtype=torch.long), return_dict=False, **extra)[0]
            a = self.alphas_cumprod[t].to(z.device).float()
            x0 = a.sqrt() * z.float() - (1 - a).sqrt() * v.float()
            loss = ((x0[:, :kl] - z0[:, :kl].float()) ** 2).sum()
            g, = torch.autograd.grad(loss, z)
        return g.float(), float(loss)

    def _sample(self, x, context_end_frame: int, seed: int, capture: bool = True, mode: str = "replace",
                guide_w: float = 0.0, hold_strength: float = 1.0, guide_tmin: int = 0):
        """Official DDIM/DPM sampling with the context latents replaced at every step. Returns (captured future
        acts, latents). Training-free ways to make a T2V model continue its context (2026-11-05):
          mode="replace" (default, stage 4): replacement only.
          mode="guided": + reconstruction guidance on the generated part: x0_b <- x0_b - (guide_w * sqrt(a_t) / 2)
                         * grad_{z_b} ||x0_hat_a - x0_a||^2 (Ho et al. 2022), applied through the v-prediction,
                         only at timesteps t >= guide_tmin (late, low-noise steps amplify it into texture noise).
          mode="hold":   start from the clip with every post-context frame a copy of the LAST context frame
                         (no future information), noised to the step at which a fraction hold_strength of the
                         schedule remains (SDEdit); replacement as usual."""
        torch = self.torch
        z0 = self._mixed(x, context_end_frame)
        kt = self.n_context_tokens(context_end_frame)
        kl = kt * self.lpt
        self.sched.set_timesteps(self.steps, device=z0.device)
        ts = [int(t) for t in self.sched.timesteps]
        i0 = 0
        if mode == "hold":
            xh = x.clone()
            xh[:, :, context_end_frame:] = x[:, :, context_end_frame - 1:context_end_frame]
            with torch.no_grad():
                zh = (self.vae.encode(xh).latent_dist.mode() * self.vae.config.scaling_factor).permute(0, 2, 1, 3, 4)
            zh = self._pad(zh.contiguous()) if hasattr(self, "_pad") else zh.contiguous()
            zh[:, :kl] = z0[:, :kl]
            i0 = int(round((1 - hold_strength) * self.steps))
        self.guide_log = []
        if mode == "guided":
            self.tr.enable_gradient_checkpointing()
        cap_step = {}
        for db in self.snrs:                                             # step nearest each encode SNR
            i = int(np.argmin([abs(t - self.t_for_snr[db]) for t in ts]))
            cap_step.setdefault(i, []).append(db)
        ctx_eps = self._noise(z0, self.noise_seed + 7)                   # fixed: identical across twins
        lat = self._noise(z0, 10_000 + seed) * self.sched.init_noise_sigma
        if mode == "hold":
            lat = self._add_noise(zh, self._noise(z0, 10_000 + seed), ts[i0])
        embeds = torch.cat([self.neg_embeds, self.prompt_embeds])
        acts = {}
        old_x0 = None
        gen = torch.Generator(device=z0.device).manual_seed(20_000 + seed)   # DPM step noise: same for twins
        for i, t in enumerate(ts):
            if i < i0:
                continue
            lat[:, :kl] = self._add_noise(z0[:, :kl], ctx_eps[:, :kl], t)
            inp = self.sched.scale_model_input(torch.cat([lat, lat]), t)
            keep = self.fc_layers if (capture and i in cap_step) else ()
            v, hs = self._pass(inp, t, embeds, keep=keep)
            v = v.float()
            g = self.guidance
            if self.dynamic_cfg:     # exactly the pipeline's formula (it uses the timestep value)
                g = 1 + self.guidance * ((1 - math.cos(math.pi * ((self.steps - t) / self.steps) ** 5.0)) / 2)
            v = v[:1] + g * (v[1:] - v[:1])
            if mode == "guided" and guide_w > 0 and t >= guide_tmin:
                grad, loss = self._guide_grad(lat, t, z0, kl)
                a = float(self.alphas_cumprod[t])
                # x0_b -= (w sqrt(a) / 2) grad_b  <=>  v_b += (w sqrt(a) / 2) grad_b / sqrt(1 - a)
                dv = (guide_w * math.sqrt(a) / 2) * grad[:, kl:] / max(math.sqrt(1 - a), 1e-4)
                self.guide_log.append({"step": i, "t": t, "ctx_loss": loss,
                                       "dv_over_v": float(dv.norm() / v[:, kl:].norm().clamp_min(1e-8))})
                v[:, kl:] = v[:, kl:] + dv.to(v.dtype)
            for l, h in hs.items():
                for db in cap_step[i]:
                    acts[key(db, l)] = self._grid(h[1], z0.shape[1] // self.lpt)[kt:]   # conditional branch, future only
            if self.dpm:
                lat, old_x0 = self.sched.step(v, old_x0, t, ts[i - 1] if i > 0 else None, lat, generator=gen,
                                              return_dict=False)
                lat = lat.to(z0.dtype)
            else:
                lat = self.sched.step(v, t, lat, return_dict=False)[0].to(z0.dtype)
        lat[:, :kl] = z0[:, :kl]
        if mode == "guided":
            self.tr.disable_gradient_checkpointing()
        return acts, lat


class CogVideoX15I2VAdapter(CogVideoXAdapter):
    """CogVideoX1.5-5B-I2V (configs/models/cogvideox15_5b_i2v.yaml): 1360 x 768, 81 frames at 16 fps.

    Differences from 2B, all following diffusers' CogVideoXImageToVideoPipeline at the pinned commit:
      * 21 latent frames, temporal patch 2: the pipeline samples 22 latents and DROPS the first before
        decoding. Here that extra leading latent is set to a copy of the first real latent everywhere
        (context replacement, clean context, encode), so token-frame 0 = input frame 0 and token-frame
        j >= 1 = input frames 8j-7..8j: TokenGridSpec(patch 16, tubelet 8, first_tubelet 1).
      * the transformer input concatenates "image latents" on channels: the first frame's latent
        (VAE mode / scaling_factor, since vae.config.invert_scale_latents) followed by zeros;
        the first frame is the clip's own frame 0, identical across twins.
      * ofs embedding = 2.0, 3-D rotary embeddings from the pipeline helper; no SNR shift.
    """

    @classmethod
    def load(cls, cfg, device: str = "cuda") -> "CogVideoX15I2VAdapter":
        from diffusers import CogVideoXImageToVideoPipeline
        pipe = _load_pipe(CogVideoXImageToVideoPipeline, cfg, "bfloat16", device)
        _official_scheduler(pipe, cfg)
        return cls(cfg, pipe, device)

    def __init__(self, cfg, pipe, device):
        super().__init__(cfg, pipe, device)
        self._img_lat = None

    def _rotary(self):
        n_lat = (self.n_frames - 1) // 4 + 1
        self.n_lat_padded = n_lat + (self.lpt - n_lat % self.lpt) % self.lpt
        return self.pipe._prepare_rotary_positional_embeddings(self.H, self.W, self.n_lat_padded, self.device)

    def _enc(self, x):
        with self.torch.no_grad():
            return self.vae.encode(x).latent_dist.mode()

    def _pad(self, z):
        """Prepend a copy of the first latent (the pipeline's extra, dropped, leading latent)."""
        return self.torch.cat([z[:, :1], z], 1) if (z.shape[1] % self.lpt) else z

    def latents(self, x):
        sig = (x.data_ptr(), tuple(x.shape))
        if self._lat_cache is not None and self._lat_cache[0] == sig:
            return self._lat_cache[1]
        z = (self._enc(x) * self.vae.config.scaling_factor).permute(0, 2, 1, 3, 4).contiguous()
        z = self._pad(z)
        # image condition from this clip's frame 0 (shared by twins)
        im = (self._enc(x[:, :, :1]) / self.vae.config.scaling_factor).permute(0, 2, 1, 3, 4)
        self._img_lat = self.torch.cat([im, self.torch.zeros_like(z[:, 1:])], 1)
        self._lat_cache = (sig, z)
        return z

    def context_latents(self, x, k: int):
        sig = (x.data_ptr(), tuple(x.shape), k)
        if getattr(self, "_ctx_cache", None) is not None and self._ctx_cache[0] == sig:
            return self._ctx_cache[1]
        z = (self._enc(x[:, :, :k]) * self.vae.config.scaling_factor).permute(0, 2, 1, 3, 4).contiguous()
        z = self.torch.cat([z[:, :1], z], 1)          # same leading duplicate as latents()
        self._ctx_cache = (sig, z)
        return z

    def _condition(self, z_t):
        img = self._img_lat.to(z_t.dtype).expand(z_t.shape[0], -1, -1, -1, -1)
        ofs = z_t.new_full((1,), 2.0)
        return self.torch.cat([z_t, img], 2), {"ofs": ofs, "image_rotary_emb": self._rot}

    def decode(self, lat) -> np.ndarray:
        return super().decode(lat[:, 1:])
