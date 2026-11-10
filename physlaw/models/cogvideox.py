"""CogVideoX adapter (spec §4.1, milestone 7): encode (noise levels) / surprise / forecast / generate.

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
from typing import Any

import numpy as np

from physlaw.models.base import ModelAdapter, TokenGridSpec, fit_frames, sample_frames

DEFAULT_PROMPT = ("A static camera looks straight down at a flat grey floor. Two balls roll across the floor, "
                  "collide with each other and bounce apart.")


LOOKS = {"rubber": "matte red rubber ball", "metal": "polished silver metal ball with soft reflections"}


def scene_prompt(labels: dict) -> str:
    """Long, literal scene description in the style of CogVideoX's training captions (the model cards
    recommend long, detailed prompts). Uses PRE-contact facts only (materials, sizes, which ball moves),
    so every twin of a group gets the same prompt. Adapted from the old project's scene_prompt."""
    o = labels["objects"]
    small, large = sorted(("A", "B"), key=lambda k: o[k]["radius"])
    name = {small: f"smaller {LOOKS[o[small]['material']]}", large: f"larger {LOOKS[o[large]['material']]}"}
    motion = (f"The {name['A']} and the {name['B']} roll toward each other across the floor at a steady pace"
              if labels.get("b_moving") else
              f"The {name['A']} rolls steadily across the floor toward the {name['B']}, which sits still")
    return (f"A still, top-down view looking straight down at a plain, smooth, evenly lit light-grey floor that fills "
            f"the whole frame. There are exactly two balls on the floor: a {name[small]} and a {name[large]}, each "
            f"casting a small soft shadow. {motion}. The two balls meet and collide once, a single clean impact, and "
            f"then both balls roll away from each other in straight lines, keeping their size, colour and shape, "
            f"sliding smoothly without spinning or bouncing off the floor. Nothing else is in the scene; no people, "
            f"no text, no other objects. The camera is completely fixed: no zoom, no pan, no cuts, no change of "
            f"lighting. Realistic physics, clean 3D render, sharp focus, natural motion.")


MINIMAL_LOOKS = {"rubber": "matte red rubber ball", "metal": "shiny grey metal ball"}


def minimal_prompt(labels: dict) -> str:
    """The user's minimal prompt (2026-11-05): name the two balls, say they collide and separate, nothing
    else -- no motion, outcome, camera or quality wording; the model decides the collision. First
    measurable collisions came from this template on Wan2.2-I2V-A14B. Pre-contact facts only."""
    o = labels["objects"]
    small, large = sorted(("A", "B"), key=lambda k: o[k]["radius"])
    return (f"Overhead view of a plain flat grey floor with two balls on it: a smaller "
            f"{MINIMAL_LOOKS[o[small]['material']]} and a larger {MINIMAL_LOOKS[o[large]['material']]}. "
            f"The two balls collide and then separate.")


def _source(cfg) -> tuple[str, dict]:
    """Weights location and from_pretrained kwargs: the plain-file local copy, else the Hub at the pinned revision."""
    if cfg.get("local_path"):
        return cfg["local_path"], {}
    return cfg["hf_id"], {"revision": cfg.get("revision")}


def _load_pipe(pipe_cls, cfg, default_dtype: str, device: str):
    """Load every component shard by shard straight onto ``device`` (device_map). A CPU-side load followed by
    ``.to(device)`` holds T5 (9.5 GB) plus the transformer in host RAM at once: 13.7 GB peak for 2B, more
    than the current machine's 15 GB for 1.5-5B. Same weights and dtype either way."""
    import torch
    src, kw = _source(cfg)
    dtype = getattr(torch, cfg.get("dtype", default_dtype))
    return pipe_cls.from_pretrained(src, torch_dtype=dtype, device_map=device, **kw)


def _official_scheduler(pipe, cfg):
    """The CogVideo repo's inference/cli_demo.py: DDIM for 2B, CogVideoXDPMScheduler (trailing) with
    use_dynamic_cfg for the 5B models; VAE tiling as in the model-card examples when configured."""
    off = cfg["official"]
    if off["scheduler"] == "CogVideoXDPMScheduler" and pipe.scheduler.__class__.__name__ != off["scheduler"]:
        from diffusers import CogVideoXDPMScheduler
        pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    if cfg.get("vae_tiling"):
        pipe.vae.enable_tiling()


def key(db: float, layer: int) -> str:
    return f"snr{int(db):+d}/L{layer}"


class CogVideoXAdapter(ModelAdapter):

    def __init__(self, cfg, pipe, device):
        super().__init__(cfg)
        import torch
        self.torch, self.pipe, self.device = torch, pipe, device
        self.tr, self.vae, self.sched = pipe.transformer, pipe.vae, pipe.scheduler
        off = cfg["official"]
        self.H, self.W = int(off["height"]), int(off["width"])
        self.n_frames, self.fps = int(off["num_frames"]), int(off["fps"])
        self.steps, self.guidance = int(off["num_inference_steps"]), float(off["guidance_scale"])
        self.dynamic_cfg = bool(off.get("use_dynamic_cfg", False))
        self.dpm = self.sched.__class__.__name__ == "CogVideoXDPMScheduler"
        tg = cfg["token_grid"]
        vs, vt, p = int(tg["vae_spatial"]), int(tg["vae_temporal"]), int(tg["patch"])
        self.lpt = int(tg.get("patch_t") or 1)          # latent frames per token-frame
        # token-frame 0 = input frame 0 (causal VAE; for 1.5 its pair is the duplicated first latent)
        self.token_grid = TokenGridSpec(input_hw=(self.H, self.W), patch=vs * p, tubelet=vt * self.lpt,
                                        first_tubelet=1, fit=cfg.get("input", {}).get("fit", "squash"))
        self.dtype = self.tr.dtype
        self.snrs = [float(s) for s in cfg["encode_snr_db"]]
        self.fc_layers = [int(l) for l in cfg.get("forecast_layers", cfg["layers"])]
        ac = self.sched.alphas_cumprod.double()
        self.alphas_cumprod = ac
        logsnr = torch.log(ac / (1 - ac).clamp_min(1e-20))
        self.t_for_snr = {db: int(torch.argmin((logsnr - db / 10 * math.log(10)).abs())) for db in self.snrs}
        s = cfg.get("surprise", {})
        self.n_draws = int(s.get("n_draws", 2))
        self.noise_seed = int(cfg.get("noise_seed", 0))
        self._lat_cache = None
        self._src = _source(cfg)
        self._emb = {}                                   # prompt text -> (prompt_embeds, neg_embeds)
        self.set_prompt(cfg.get("prompt", DEFAULT_PROMPT))

    def set_prompt(self, text: str):
        """Replace the conditioning prompt (per-clip prompts, e.g. minimal_prompt); negative stays "".
        T5 (9.5 GB) is freed after each encode instead of being parked in host RAM (15 GB on the current
        machine), and reloaded from disk straight to the GPU only for a prompt not yet encoded."""
        torch = self.torch
        if text not in self._emb:
            if self.pipe.text_encoder is None:
                from transformers import T5EncoderModel
                src, kw = self._src
                self.pipe.text_encoder = T5EncoderModel.from_pretrained(
                    src, subfolder="text_encoder", torch_dtype=self.dtype, device_map=self.device, **kw)
            with torch.no_grad():
                pe, ne = self.pipe.encode_prompt(text, "", do_classifier_free_guidance=True, num_videos_per_prompt=1,
                                                 max_sequence_length=226, device=self.device)
            self._emb[text] = (pe.to(self.dtype), ne.to(self.dtype))
            self.pipe.text_encoder = None
            torch.cuda.empty_cache()
        self.prompt_embeds, self.neg_embeds = self._emb[text]
        self._prompt_text = text

    @classmethod
    def load(cls, cfg, device: str = "cuda") -> "CogVideoXAdapter":
        from diffusers import CogVideoXPipeline
        pipe = _load_pipe(CogVideoXPipeline, cfg, "float16", device)
        _official_scheduler(pipe, cfg)
        if pipe.scheduler.__class__.__name__ != cfg["official"]["scheduler"]:
            raise RuntimeError(f"scheduler {pipe.scheduler.__class__.__name__} != official {cfg['official']['scheduler']}")
        pipe.set_progress_bar_config(disable=True)
        return cls(cfg, pipe, device)

    # ------------------------------------------------------------------ geometry / input

    def input_frames(self, labels: dict) -> list[int]:
        sc = labels["scene_constants"]
        return sample_frames(master_fps=sc["frame_rate"], n_master=sc["num_frames"], fps=self.fps,
                             n_frames=self.n_frames, contact_frame=labels["physlaw"]["first_post_contact_frame"],
                             center_on_contact=bool(self.cfg.get("input", {}).get("center_on_contact", False)))

    def context_frames(self, labels: dict) -> int:
        """Leading input frames covered by token-frames (latent frames) that are entirely pre-contact."""
        fr = self.input_frames(labels)
        fpc = labels["physlaw"]["first_post_contact_frame"]
        k = 0
        for idx in self.token_grid.token_frames(len(fr)):
            if max(fr[i] for i in idx) < fpc:
                k = idx[-1] + 1
            else:
                break
        return k

    def n_context_tokens(self, k: int) -> int:
        """Token-frames (latent frames) entirely inside the first k input frames."""
        return sum(1 for idx in self.token_grid.token_frames(self.n_frames) if idx[-1] < k)

    def to_tensor(self, frames_uint8: np.ndarray):
        torch = self.torch
        x = fit_frames(frames_uint8, (self.H, self.W), self.token_grid.fit)
        t = torch.from_numpy(x).to(self.device).permute(3, 0, 1, 2)[None].float() / 127.5 - 1.0
        return t.to(self.dtype)                                          # [1, 3, T, H, W]

    def _grid(self, tok, T):
        gh, gw = self.token_grid.grid_hw
        return tok.reshape(T, gh, gw, tok.shape[-1])

    # ------------------------------------------------------------------ latents / passes

    def latents(self, x):
        """Clean latents [1, F, C, h, w] (VAE mode x scaling factor); memoised for the last input."""
        torch = self.torch
        sig = (x.data_ptr(), tuple(x.shape))
        if self._lat_cache is not None and self._lat_cache[0] == sig:
            return self._lat_cache[1]
        with torch.no_grad():
            z = self.vae.encode(x).latent_dist.mode() * self.vae.config.scaling_factor
        z = z.permute(0, 2, 1, 3, 4).contiguous()
        self._lat_cache = (sig, z)
        return z

    def context_latents(self, x, k: int):
        """Latents of the first k input frames encoded ALONE (k = 1 + 4j): identical across twins."""
        torch = self.torch
        sig = (x.data_ptr(), tuple(x.shape), k)
        if getattr(self, "_ctx_cache", None) is not None and self._ctx_cache[0] == sig:
            return self._ctx_cache[1]
        with torch.no_grad():
            z = self.vae.encode(x[:, :, :k]).latent_dist.mode() * self.vae.config.scaling_factor
        z = z.permute(0, 2, 1, 3, 4).contiguous()
        self._ctx_cache = (sig, z)
        return z

    def _mixed(self, x, k):
        """[context latents from the context alone, future latents from the full clip]."""
        z = self.latents(x).clone()
        zc = self.context_latents(x, k)
        z[:, : zc.shape[1]] = zc
        return z

    def _noise(self, z, seed):
        g = self.torch.Generator(device="cpu").manual_seed(seed)
        return self.torch.randn(z.shape, generator=g, dtype=self.torch.float32).to(z.device, z.dtype)

    def _add_noise(self, z0, eps, t):
        a = self.alphas_cumprod[t].to(z0.device)
        return (a.sqrt() * z0.double() + (1 - a).sqrt() * eps.double()).to(z0.dtype)

    def _pass(self, z_t, t, embeds, keep=()):
        """One transformer pass; returns (v-prediction, {layer: video-token hidden [B, N, D]})."""
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
        """Model input and extra kwargs (T2V: none)."""
        return z_t, {}

    def _x0(self, z_t, v, t):
        a = self.alphas_cumprod[t].to(z_t.device)
        return (a.sqrt() * z_t.double() - (1 - a).sqrt() * v.double())

    def _token_err(self, err):
        """Latent error [1, F, C, h, w] -> per-token mean [F, h/p, w/p]."""
        p, q = int(self.cfg["token_grid"]["patch"]), self.lpt
        e = err[0].mean(1)                                               # [F, h, w]
        F_, h, w = e.shape
        return e.reshape(F_ // q, q, h // p, p, w // p, p).mean((1, 3, 5))

    # ------------------------------------------------------------------ spec API

    def encode(self, video, noise_level=None) -> dict[str, Any]:
        torch = self.torch
        x = video if torch.is_tensor(video) else self.to_tensor(video)
        z0 = self.latents(x)
        eps = self._noise(z0, self.noise_seed)
        out = {}
        for db in (self.snrs if noise_level is None else [float(noise_level)]):
            t = self.t_for_snr[db]
            _, hs = self._pass(self._add_noise(z0, eps, t), t, self.prompt_embeds, keep=self.layers)
            for l, h in hs.items():
                out[key(db, l)] = self._grid(h[0], z0.shape[1] // self.lpt)
        return out

    def surprise(self, video, context_end_frame: int, object_masks=None) -> dict[str, float]:
        torch = self.torch
        x = video if torch.is_tensor(video) else self.to_tensor(video)
        z0 = self._mixed(x, context_end_frame)
        kt = self.n_context_tokens(context_end_frame)
        kl = kt * self.lpt
        obj = None
        if object_masks is not None:
            obj = torch.as_tensor(np.asarray(object_masks), device=z0.device).bool()
        acc = {}
        for d in range(self.n_draws):
            eps = self._noise(z0, self.noise_seed + 1000 + d)
            for db in self.snrs:
                t = self.t_for_snr[db]
                a = float(self.alphas_cumprod[t])
                w = 1.0 / max(1 - a, 1e-6)
                zt = self._add_noise(z0, eps, t)
                for variant in ("clean", "noisy"):
                    zin = zt.clone()
                    if variant == "clean":
                        zin[:, :kl] = z0[:, :kl]
                    v, _ = self._pass(zin, t, self.prompt_embeds)
                    e = self._token_err(w * (self._x0(zin, v, t) - z0.double()) ** 2)[kt:]
                    acc.setdefault(f"{variant}_mse_all", []).append(float(e.mean()))
                    if obj is not None and obj.any():
                        acc.setdefault(f"{variant}_mse_obj", []).append(float(e[obj].mean()))
        return {k_: float(np.mean(v_)) for k_, v_ in acc.items()}

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

    def decode(self, lat) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            v = self.vae.decode(lat.permute(0, 2, 1, 3, 4) / self.vae.config.scaling_factor).sample
        v = ((v[0].float().clamp(-1, 1) + 1) * 127.5).round().byte()     # [3, T, H, W]
        return v.permute(1, 2, 3, 0).cpu().numpy()

    def forecast(self, video, context_end_frame: int, seed: int = 0, return_video: bool = False):
        torch = self.torch
        x = video if torch.is_tensor(video) else self.to_tensor(video)
        acts, lat = self._sample(x, context_end_frame, seed)
        if return_video:
            return acts, self.decode(lat)
        return acts

    def generate(self, video, context_end_frame: int, seed: int, **sample_kw):
        """sample_kw: mode / guide_w / hold_strength (see _sample); default = the stage-4 replacement sampler."""
        torch = self.torch
        x = video if torch.is_tensor(video) else self.to_tensor(video)
        _, lat = self._sample(x, context_end_frame, seed, capture=False, **sample_kw)
        return self.decode(lat)


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
        if pipe.scheduler.__class__.__name__ != cfg["official"]["scheduler"]:
            raise RuntimeError(f"scheduler {pipe.scheduler.__class__.__name__} != official {cfg['official']['scheduler']}")
        pipe.set_progress_bar_config(disable=True)
        return cls(cfg, pipe, device)

    def __init__(self, cfg, pipe, device):
        super().__init__(cfg, pipe, device)
        n_lat = (self.n_frames - 1) // 4 + 1
        self.n_lat_padded = n_lat + (self.lpt - n_lat % self.lpt) % self.lpt
        self._rot = pipe._prepare_rotary_positional_embeddings(self.H, self.W, self.n_lat_padded, device)
        self._img_lat = None

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
