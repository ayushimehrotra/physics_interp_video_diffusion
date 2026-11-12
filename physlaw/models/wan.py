"""Wan adapters (spec §4.2): Wan2.1-T2V-1.3B / 14B, Wan2.2-TI2V-5B and PhysRVG (RL-tuned TI2V-5B), plus the
image-to-video models of output track B (Wan2.1-I2V-14B, Wan2.2-I2V-A14B). One module: they share the VAE family,
the UMT5 text encoder, the flow-matching objective and the UniPC sampler. Shared DiT code: dit.py.

Schedule (rectified flow, as trained): z_t = (1 - s) z0 + s eps at timestep 1000 s; the transformer predicts the
velocity eps - z0, so surprise() is that MSE (Wan's training loss). encode SNR dB -> s = 1 / (1 + 10^(dB / 20)).
Latents: VAE mode, normalised (z - latents_mean) / latents_std as the pipelines do; VAE in fp32 (model cards),
transformer in bf16. Wan2.1 VAE: 8x spatial, 16 ch; Wan2.2 VAE: 16x spatial (its 2x2 patchify inside), 48 ch;
both 4x temporal causal (latent 0 = frame 0). Transformer patch (1, 2, 2). The VAE's norms are per frame, but
context latents are still taken from the context frames alone (dit.py rule).

Context conditioning (forecast() / generate(), and the clean-context surprise):
  * Wan2.1 T2V (one timestep per clip): as for CogVideoX, the context latents are replaced at every sampling step
    by the true context latents noised to that step's sigma (fixed noise); the clean-context surprise is out of
    distribution (the same for every twin).
  * Wan2.2 TI2V-5B and PhysRVG (``expand_timesteps``: one timestep per token): the models' own conditioning. The
    TI2V image-to-video pipeline holds the first latent frame clean at timestep 0; PhysRVG's video-to-video
    pipeline holds the first 2 (5 frames). Here every context latent frame is held clean at timestep 0, for
    sampling and for the clean-context surprise (in distribution).
  * Output track A (generate() with a ``track_a`` block, configs/models/physrvg.yaml): exactly PhysRVG's trained
    context, 2 latent frames = 5 frames ending at the last pre-contact frame (``track_a_mode: native``); the base
    Wan2.2-TI2V-5B at PhysRVG's settings gets the same clamp, flagged ``track_a_mode: hack`` (pair comparison).
Sampling follows the diffusers pipelines step for step (sequential cond / uncond passes, CFG in the transformer's
dtype, UniPC with the official flow_shift); activations of the conditional pass are captured at the steps nearest
each encode SNR (keys as encode()).

WanI2VAdapter (output track B, spec §6.3): generate() runs the official WanImageToVideoPipeline from frame 0 of the
clip with the current prompt (set_prompt(minimal_prompt(labels))); no internals.
"""
from __future__ import annotations

from physlaw.models.dit import (DiTAdapter, apply_weight_overrides, flow_t_for_db, key, source,
                                text_encoder_source)


def _load_pipe(pipe_cls, cfg, device):
    """VAE in fp32, the rest in cfg.dtype, straight to the GPU; no text encoder (loaded per prompt, then freed);
    UniPC with the official flow_shift (the model cards set it this way); dit_weights / lora applied."""
    import torch
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
    src, kw = source(cfg)
    vae = AutoencoderKLWan.from_pretrained(src, subfolder="vae", torch_dtype=torch.float32, device_map=device, **kw)
    pipe = pipe_cls.from_pretrained(src, vae=vae, text_encoder=None, torch_dtype=getattr(torch, cfg.get("dtype", "bfloat16")),
                                    device_map=device, **kw)
    off = cfg["official"]
    if off.get("flow_shift") is not None:
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=float(off["flow_shift"]))
    if pipe.scheduler.__class__.__name__ != off["scheduler"]:
        raise RuntimeError(f"scheduler {pipe.scheduler.__class__.__name__} != official {off['scheduler']}")
    for t in (pipe.transformer, getattr(pipe, "transformer_2", None)):
        if t is not None:
            apply_weight_overrides(t, cfg)
    pipe.set_progress_bar_config(disable=True)
    return pipe


class WanAdapter(DiTAdapter):

    @classmethod
    def load(cls, cfg, device: str = "cuda") -> "WanAdapter":
        from diffusers import WanPipeline
        return cls(cfg, _load_pipe(WanPipeline, cfg, device), device)

    def __init__(self, cfg, pipe, device):
        self.expand = bool(getattr(pipe.config, "expand_timesteps", False))
        super().__init__(cfg, pipe, device)

    # ------------------------------------------------------------------ family hooks (dit.py)

    def _init_schedule(self):
        torch = self.torch
        c = self.vae.config
        self.lat_mean = torch.tensor(c.latents_mean).view(1, c.z_dim, 1, 1, 1).to(self.device, torch.float32)
        self.lat_inv_std = 1.0 / torch.tensor(c.latents_std).view(1, c.z_dim, 1, 1, 1).to(self.device, torch.float32)
        self.n_train = int(self.sched.config.num_train_timesteps)

    def _t_for_db(self, db):
        return flow_t_for_db(db, self.n_train)

    def _cfg_on(self) -> bool:
        return self.guidance is not None and self.guidance > 1.0

    def _encode_text(self, text: str):
        torch = self.torch
        if self.pipe.text_encoder is None:
            from transformers import UMT5EncoderModel
            src, kw = text_encoder_source(self.cfg)
            self.pipe.text_encoder = UMT5EncoderModel.from_pretrained(
                src, subfolder="text_encoder", torch_dtype=self.dtype, device_map=self.device, **kw)
        with torch.no_grad():
            pe, ne = self.pipe.encode_prompt(text, self.negative_prompt, do_classifier_free_guidance=self._cfg_on(),
                                             num_videos_per_prompt=1, max_sequence_length=512, device=self.device)
        self.pipe.text_encoder = None
        return pe.to(self.dtype), (ne.to(self.dtype) if ne is not None else None)

    def _vae_encode(self, x):
        with self.torch.no_grad():
            z = self.vae.encode(x.to(self.vae.dtype)).latent_dist.mode().float()
        return ((z - self.lat_mean) * self.lat_inv_std).permute(0, 2, 1, 3, 4).contiguous()

    def _vae_decode(self, lat):
        z = lat.permute(0, 2, 1, 3, 4).to(self.vae.dtype) / self.lat_inv_std + self.lat_mean
        return self.vae.decode(z, return_dict=False)[0]

    def _add_noise(self, z0, eps, t):
        s = float(t) / self.n_train
        return ((1 - s) * z0.double() + s * eps.double()).to(z0.dtype)

    def _train_err(self, z_in, pred, z0, eps, t):
        """Flow-matching loss: squared error of the predicted velocity against eps - z0."""
        return (pred.double() - (eps.double() - z0.double())) ** 2

    def _pass(self, z_t, t, embeds, keep=(), clean: int = 0):
        """One transformer pass on [B, F, C, h, w]; returns (velocity [B, F, C, h, w] in the transformer's dtype,
        {layer: block output [B, N, D]}). ``clean``: leading latent frames given timestep 0 (expand_timesteps
        models only; T2V models get one timestep for all tokens)."""
        torch = self.torch
        hs, hooks = {}, []
        for l in keep:
            hooks.append(self.tr.blocks[l].register_forward_hook(lambda mod, inp, out, l=l: hs.__setitem__(l, out)))
        try:
            x = z_t.permute(0, 2, 1, 3, 4).to(self.dtype)
            B, _, F_, h, w = x.shape
            if self.expand:
                ts = torch.full((F_, h // self.p, w // self.p), float(t), device=x.device)
                ts[:clean] = 0
                ts = ts.flatten()[None].expand(B, -1)
            else:
                ts = torch.full((B,), float(t), device=x.device)
            with torch.no_grad():
                v = self.tr(hidden_states=x, timestep=ts, encoder_hidden_states=embeds, return_dict=False)[0]
        finally:
            for hk in hooks:
                hk.remove()
        return v.permute(0, 2, 1, 3, 4), hs

    def _sample(self, x, context_end_frame: int, seed: int, capture: bool = True):
        """Official UniPC sampling from fixed noise (seed), conditioned on the context latents (see the module
        docstring). Returns (captured future activations, final latents with the true context)."""
        torch = self.torch
        z0 = self._mixed(x, context_end_frame)
        kt = self.n_context_tokens(context_end_frame)
        kl = kt * self.lpt
        self.sched.set_timesteps(self.steps, device=z0.device)
        ts = self.sched.timesteps
        cap = self._capture_steps(ts) if capture else {}
        ctx_eps = self._noise(z0, self.noise_seed + 7)                 # fixed: identical across twins
        lat = self._noise(z0, 10_000 + seed)
        acts, T = {}, z0.shape[1] // self.lpt
        for i, t in enumerate(ts):
            if self.expand:
                zin = lat.clone()
                zin[:, :kl] = z0[:, :kl]
            else:
                s = float(self.sched.sigmas[i])
                lat[:, :kl] = ((1 - s) * z0[:, :kl].double() + s * ctx_eps[:, :kl].double()).to(lat.dtype)
                zin = lat
            clean = kl if self.expand else 0
            v, hs = self._pass(zin, t, self.prompt_embeds, keep=self.fc_layers if i in cap else (), clean=clean)
            if self._cfg_on():
                vu, _ = self._pass(zin, t, self.neg_embeds, clean=clean)
                v = vu + self.guidance * (v - vu)
            for l, h in hs.items():
                for db in cap[i]:
                    acts[key(db, l)] = self._grid(h[0], T)[kt:]
            with torch.no_grad():
                lat = self.sched.step(v, t, lat, return_dict=False)[0]
        lat[:, :kl] = z0[:, :kl]
        return acts, lat

    def generate(self, video, context_end_frame: int, seed: int, **sample_kw):
        """Track A. With a ``track_a`` block (PhysRVG, and Wan2.2-TI2V-5B at its settings as the flagged hack): the
        model gets exactly ``track_a.context`` frames (PhysRVG's trained 5 = 2 latent frames, held clean at timestep
        0) ending at the last pre-contact frame (window index context_end_frame - 1), and samples its own
        num_frames clip from the first of them; the real window frames before the context are put in front
        (track_a.splice), so the clip stays aligned with input_frames(). Without the block: the stage-4 path
        (every whole pre-contact latent frame as context)."""
        if not self.cfg.get("track_a"):
            return super().generate(video, context_end_frame, seed, **sample_kw)
        from physlaw.models.track_a import context_len, splice
        torch = self.torch
        x = self._x(video)
        n = context_len(self.cfg, context_end_frame)
        s = context_end_frame - n
        win = x[:, :, s: s + self.n_frames]
        if win.shape[2] < self.n_frames:     # frames after the context only set the latent shape (never sampled from)
            pad = win[:, :, -1:].expand(-1, -1, self.n_frames - win.shape[2], -1, -1)
            win = torch.cat([win, pad], 2)
        win = win.contiguous()
        self._lat_cache = self._ctx_cache = None
        _, lat = self._sample(win, n, seed, capture=False, **sample_kw)
        self._lat_cache = self._ctx_cache = None
        real = ((x[0, :, :s].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()
        return splice(real, self.decode(lat), s, self.n_frames)


class WanI2VAdapter(WanAdapter):
    """Output track B generator (frame 0 + prompt -> clip): the official WanImageToVideoPipeline call. No internals:
    the I2V transformers take image conditioning (CLIP image embeds / extra latent channels) that WanAdapter's
    encode / forecast / surprise do not supply, so stages that need internals refuse this class."""

    internals = False

    @classmethod
    def load(cls, cfg, device: str = "cuda") -> "WanI2VAdapter":
        from diffusers import WanImageToVideoPipeline
        return cls(cfg, _load_pipe(WanImageToVideoPipeline, cfg, device), device)

    def generate(self, video, context_end_frame: int = 1, seed: int = 0):
        """Clip from frame 0 of ``video`` (uint8 [T, H, W, 3] or the [1, 3, T, H, W] tensor of to_tensor). The
        frame is fitted to height x width like every other input; noise from torch.Generator('cuda', seed)."""
        import numpy as np
        from PIL import Image
        torch = self.torch
        if torch.is_tensor(video):
            f0 = ((video[0, :, 0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
        else:
            from physlaw.models.base import fit_frames
            f0 = fit_frames(np.asarray(video[:1]), (self.H, self.W), self.token_grid.fit)[0]
        out = self.pipe(image=Image.fromarray(f0), prompt_embeds=self.prompt_embeds,
                        negative_prompt_embeds=self.neg_embeds, height=self.H, width=self.W,
                        num_frames=self.n_frames, num_inference_steps=self.steps,
                        guidance_scale=self.guidance if self._cfg_on() else 1.0,
                        generator=torch.Generator(self.device).manual_seed(seed), output_type="np").frames[0]
        return (np.clip(out, 0, 1) * 255).round().astype(np.uint8)
