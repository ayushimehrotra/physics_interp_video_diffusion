"""Shared machinery of the diffusion-transformer adapters (cogvideox.py, wan.py, cosmos_predict.py).

Geometry: the official clip (height x width, num_frames at fps; every master/fps-th master frame) and a causal VAE:
token-frame 0 = input frame 0, token-frame j >= 1 = the next ``vae_temporal x patch_t`` frames (base.token_spec).

Rules carried over from CogVideoX-2b (docs/STATUS.md, milestone 7) and applied to every DiT:
  * context latents come from VAE-encoding the context frames ALONE (``context_latents``): twins share them exactly.
  * encode(noise_level=dB): forward-noise the clean latents to the timestep whose SNR is closest to dB (one fixed
    noise draw, identical for every clip), one conditional pass; block outputs at ``cfg.layers``, keys "snr{dB}/L{l}".
    It also returns the noised-input-latent baseline (spec §5.1): "latent/snr{dB}" = the noised VAE latent the model
    was fed, as one vector per token (the token's latent patch, C x patch_t x patch x patch values), so stage 4 pools
    it exactly like the activations. ``latent_baseline()`` computes only these keys (no transformer pass).
  * surprise(): the model's own training loss on post-context latents over the encode SNR timesteps x
    ``surprise.n_draws`` fixed noise draws (identical across twins); ``clean_*`` with clean context latents,
    ``noisy_*`` with the whole clip noised. Regions: all post-context tokens and ``obj`` (the balls' ROI tokens).

Latents are [1, F, C, h, w] (frames first) throughout; each family converts at its VAE and transformer boundary and
supplies the schedule (``_t_for_db``, ``_add_noise``, ``_train_err``), the VAE (``_vae_encode``, ``_vae_decode``), the
text encoder (``_encode_text``), the transformer pass (``_pass``) and the sampler behind forecast() / generate().

Weights. ``lora: {path, scale}`` merges a LoRA into the transformer at load (``merge_lora``); ``dit_weights: {path}``
replaces the transformer's weights key-for-key first (a fully fine-tuned DiT shipped as one state dict).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from physlaw.models.base import ModelAdapter, fit_frames, sample_frames, token_spec

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


PROMPTS = {"scene": scene_prompt, "minimal": minimal_prompt}


def key(db: float, layer: int) -> str:
    return f"snr{int(db):+d}/L{layer}"


def latent_key(db: float) -> str:
    return f"latent/snr{int(db):+d}"


def source(cfg) -> tuple[str, dict]:
    """Weights location and from_pretrained kwargs: the plain-file local copy, else the Hub at the pinned revision."""
    if cfg.get("local_path"):
        return cfg["local_path"], {}
    return cfg["hf_id"], {"revision": cfg.get("revision")}


def text_encoder_source(cfg) -> tuple[str, dict]:
    """Where the text encoder lives: ``text_encoder.path`` when the model's own copy is not on disk (a verified
    identical copy shipped with a sibling model, see the config), else the model's own repo."""
    te = cfg.get("text_encoder") or {}
    if te.get("path"):
        return te["path"], {}
    return source(cfg)


# --------------------------------------------------------------------------- weights

def _lora_pairs(sd: dict) -> dict[str, tuple[Any, Any]]:
    """{module name: (A, B)} from a diffusers ('transformer.<m>.lora_A.weight') or PEFT
    ('base_model.model.<m>.lora_A.weight') LoRA state dict."""
    out = {}
    for k in sd:
        if not k.endswith(".lora_A.weight"):
            continue
        m = k[: -len(".lora_A.weight")]
        for pre in ("base_model.model.", "transformer."):
            if m.startswith(pre):
                m = m[len(pre):]
                break
        out[m] = (sd[k], sd[k.replace(".lora_A.weight", ".lora_B.weight")])
    if len(out) * 2 != len(sd):
        raise ValueError(f"LoRA file has {len(sd)} tensors, {len(out)} A/B pairs: unexpected keys")
    return out


def lora_scale(path: str | Path, scale: float = 1.0) -> float:
    """Total factor on B @ A: ``scale`` x (lora_alpha / r) when a PEFT adapter_config.json sits beside the weights
    (PEFT's own scaling), else ``scale`` alone (a diffusers LoRA file without alpha: alpha = r)."""
    p = Path(path)
    c = (p if p.is_dir() else p.parent) / "adapter_config.json"
    if not c.exists():
        return float(scale)
    c = json.loads(c.read_text())
    if c.get("use_dora") or c.get("use_rslora") or c.get("rank_pattern") or c.get("alpha_pattern"):
        raise NotImplementedError(f"{c}: only plain LoRA is supported")
    return float(scale) * float(c["lora_alpha"]) / float(c["r"])


def merge_lora(module, path: str | Path, scale: float = 1.0) -> int:
    """Merge a LoRA into ``module``'s Linear weights in place: W += (s * B @ A) cast to W's dtype, the product in
    fp32 (as PEFT's merge / diffusers' fuse_lora do). ``path``: a .safetensors file, or a directory holding
    adapter_model.safetensors (PEFT) or pytorch_lora_weights.safetensors (diffusers). ``s`` from :func:`lora_scale`.
    Returns the number of merged layers; every LoRA pair must match a module of the right shape."""
    import torch
    from safetensors.torch import load_file
    p = Path(path)
    f = next((p / n for n in ("adapter_model.safetensors", "pytorch_lora_weights.safetensors") if (p / n).exists()),
             p) if p.is_dir() else p
    s = lora_scale(p, scale)
    dev = next(module.parameters()).device
    pairs = _lora_pairs(load_file(str(f), device=str(dev)))
    mods = dict(module.named_modules())
    missing = [m for m in pairs if m not in mods]
    if missing:
        raise KeyError(f"{len(missing)} LoRA targets not in the model, e.g. {missing[:3]}")
    with torch.no_grad():
        for m, (A, B) in pairs.items():
            W = mods[m].weight
            delta = (B.float() @ A.float()) * s
            if delta.shape != W.shape:
                raise ValueError(f"{m}: LoRA delta {tuple(delta.shape)} != weight {tuple(W.shape)}")
            W += delta.to(W.dtype)
    return len(pairs)


def load_dit_weights(module, path: str | Path) -> int:
    """Replace every parameter/buffer of ``module`` from a full state dict (strict: same keys and shapes), tensor by
    tensor straight onto the module's device (no second full copy in host RAM)."""
    import torch
    from safetensors import safe_open
    sd = dict(module.state_dict())
    with safe_open(str(path), "pt", device=str(next(module.parameters()).device)) as f:
        keys = set(f.keys())
        if keys != set(sd):
            raise KeyError(f"{path}: {len(keys ^ set(sd))} keys differ from the model, e.g. {sorted(keys ^ set(sd))[:3]}")
        with torch.no_grad():
            for k in keys:
                t = f.get_tensor(k)
                if t.shape != sd[k].shape:
                    raise ValueError(f"{k}: {tuple(t.shape)} != {tuple(sd[k].shape)}")
                sd[k].copy_(t)
    return len(keys)


def apply_weight_overrides(transformer, cfg) -> None:
    """``dit_weights`` then ``lora`` from the config (see the module docstring)."""
    if cfg.get("dit_weights"):
        load_dit_weights(transformer, cfg["dit_weights"]["path"])
    if cfg.get("lora"):
        merge_lora(transformer, cfg["lora"]["path"], float(cfg["lora"].get("scale", 1.0)))


# --------------------------------------------------------------------------- adapter

class DiTAdapter(ModelAdapter):
    """Common part of the diffusion adapters; see the module docstring for the hooks a family provides."""

    def __init__(self, cfg, pipe, device):
        super().__init__(cfg)
        import torch
        self.torch, self.pipe, self.device = torch, pipe, device
        self.tr, self.vae, self.sched = pipe.transformer, pipe.vae, pipe.scheduler
        off = cfg["official"]
        self.H, self.W = int(off["height"]), int(off["width"])
        self.n_frames, self.fps = int(off["num_frames"]), int(off["fps"])
        self.steps = int(off["num_inference_steps"])
        self.guidance = float(off["guidance_scale"]) if off.get("guidance_scale") is not None else None
        self.lpt = int(cfg["token_grid"].get("patch_t") or 1)          # latent frames per token-frame
        self.p = int(cfg["token_grid"]["patch"])
        self.token_grid = token_spec(cfg)
        self.dtype = self.tr.dtype
        self.snrs = [float(s) for s in cfg.get("encode_snr_db", [])]          # generator-only configs: none
        self.fc_layers = [int(l) for l in cfg.get("forecast_layers", cfg.get("layers", []))]
        self._init_schedule()
        self.t_for_snr = {db: self._t_for_db(db) for db in self.snrs}
        self.n_draws = int(cfg.get("surprise", {}).get("n_draws", 2))
        self.noise_seed = int(cfg.get("noise_seed", 0))
        self._lat_cache = self._ctx_cache = None
        self._emb = {}                                   # prompt text -> (prompt_embeds, neg_embeds)
        self.negative_prompt = off.get("negative_prompt", "")
        self.set_prompt(cfg.get("prompt", DEFAULT_PROMPT))

    # ------------------------------------------------------------------ family hooks
    def _init_schedule(self): ...

    def _t_for_db(self, db: float): raise NotImplementedError

    def _add_noise(self, z0, eps, t): raise NotImplementedError

    def _train_err(self, z_in, pred, z0, eps, t): raise NotImplementedError

    def _vae_encode(self, x): raise NotImplementedError

    def _vae_decode(self, lat): raise NotImplementedError

    def _encode_text(self, text: str): raise NotImplementedError

    def _pass(self, z_t, t, embeds, keep=(), clean: int = 0): raise NotImplementedError

    def _sample(self, x, context_end_frame: int, seed: int, capture: bool = True, **kw): raise NotImplementedError

    # ------------------------------------------------------------------ text

    def set_prompt(self, text: str):
        """Replace the conditioning prompt (per-clip prompts, e.g. minimal_prompt). Embeddings are cached per text;
        the text encoder is loaded only for an unseen prompt and freed right after (``_encode_text``)."""
        if text not in self._emb:
            self._emb[text] = self._encode_text(text)
            self.torch.cuda.empty_cache()
        self.prompt_embeds, self.neg_embeds = self._emb[text]
        self._prompt_text = text

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
        """uint8 [T, H, W, 3] -> [1, 3, T, H, W] in [-1, 1], in the VAE's dtype."""
        torch = self.torch
        x = fit_frames(frames_uint8, (self.H, self.W), self.token_grid.fit)
        t = torch.from_numpy(x).to(self.device).permute(3, 0, 1, 2)[None].float() / 127.5 - 1.0
        return t.to(self.vae.dtype)

    def _grid(self, tok, T):
        gh, gw = self.token_grid.grid_hw
        return tok.reshape(T, gh, gw, tok.shape[-1])

    def _x(self, video):
        return video if self.torch.is_tensor(video) else self.to_tensor(video)

    # ------------------------------------------------------------------ latents

    def latents(self, x):
        """Clean model-space latents [1, F, C, h, w] of the whole clip; memoised for the last input."""
        sig = (x.data_ptr(), tuple(x.shape))
        if self._lat_cache is not None and self._lat_cache[0] == sig:
            return self._lat_cache[1]
        z = self._vae_encode(x)
        self._lat_cache = (sig, z)
        return z

    def context_latents(self, x, k: int):
        """Latents of the first k input frames encoded ALONE (k = 1 + 4j): identical across twins."""
        sig = (x.data_ptr(), tuple(x.shape), k)
        if self._ctx_cache is not None and self._ctx_cache[0] == sig:
            return self._ctx_cache[1]
        z = self._vae_encode(x[:, :, :k])
        self._ctx_cache = (sig, z)
        return z

    def _mixed(self, x, k):
        """[context latents from the context alone, future latents from the full clip]."""
        z = self.latents(x).clone()
        if k == 0:                                       # no context: plain T2V sampling
            return z
        zc = self.context_latents(x, k)
        z[:, : zc.shape[1]] = zc
        return z

    def _noise(self, z, seed):
        g = self.torch.Generator(device="cpu").manual_seed(seed)
        return self.torch.randn(z.shape, generator=g, dtype=self.torch.float32).to(z.device, z.dtype)

    def _token_err(self, err):
        """Latent error [1, F, C, h, w] -> per-token mean [F / patch_t, h / p, w / p]."""
        p, q = self.p, self.lpt
        e = err[0].mean(1)                                               # [F, h, w]
        F_, h, w = e.shape
        return e.reshape(F_ // q, q, h // p, p, w // p, p).mean((1, 3, 5))

    def _latent_tokens(self, z):
        """Latent [1, F, C, h, w] -> one vector per token [F / patch_t, h / p, w / p, patch_t * C * p * p]."""
        p, q = self.p, self.lpt
        _, F_, C, h, w = z.shape
        t = z[0].float().reshape(F_ // q, q, C, h // p, p, w // p, p)
        return t.permute(0, 3, 5, 1, 2, 4, 6).reshape(F_ // q, h // p, w // p, q * C * p * p)

    # ------------------------------------------------------------------ spec API

    def latent_baseline(self, video) -> dict[str, Any]:
        """Noised-input-latent baseline (spec §5.1): the latents encode() feeds the transformer, per token. No
        transformer pass, so stage 4 can add these keys to an existing cache cheaply."""
        x = self._x(video)
        z0 = self.latents(x)
        eps = self._noise(z0, self.noise_seed)
        return {latent_key(db): self._latent_tokens(self._add_noise(z0, eps, self.t_for_snr[db])) for db in self.snrs}

    def encode(self, video, noise_level=None) -> dict[str, Any]:
        x = self._x(video)
        z0 = self.latents(x)
        eps = self._noise(z0, self.noise_seed)
        out = {}
        for db in (self.snrs if noise_level is None else [float(noise_level)]):
            t = self.t_for_snr[db]
            zt = self._add_noise(z0, eps, t)
            _, hs = self._pass(zt, t, self.prompt_embeds, keep=self.layers)
            for l, h in hs.items():
                out[key(db, l)] = self._grid(h[0], z0.shape[1] // self.lpt)
            out[latent_key(db)] = self._latent_tokens(zt)
        return out

    def surprise(self, video, context_end_frame: int, object_masks=None) -> dict[str, float]:
        torch = self.torch
        x = self._x(video)
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
                zt = self._add_noise(z0, eps, t)
                for variant in ("clean", "noisy"):
                    zin = zt.clone()
                    if variant == "clean":
                        zin[:, :kl] = z0[:, :kl]
                    v, _ = self._pass(zin, t, self.prompt_embeds, clean=kl if variant == "clean" else 0)
                    e = self._token_err(self._train_err(zin, v, z0, eps, t))[kt:]
                    acc.setdefault(f"{variant}_mse_all", []).append(float(e.mean()))
                    if obj is not None and obj.any():
                        acc.setdefault(f"{variant}_mse_obj", []).append(float(e[obj].mean()))
        return {k_: float(np.mean(v_)) for k_, v_ in acc.items()}

    def _capture_steps(self, ts) -> dict[int, list[float]]:
        """Sampling step index nearest each encode SNR's timestep -> those SNRs (forecast keys match encode's)."""
        cap = {}
        for db in self.snrs:
            i = int(np.argmin([abs(float(t) - float(self.t_for_snr[db])) for t in ts]))
            cap.setdefault(i, []).append(db)
        return cap

    def decode(self, lat) -> np.ndarray:
        """[1, F, C, h, w] model latents -> uint8 [T, H, W, 3]."""
        with self.torch.no_grad():
            v = self._vae_decode(lat)
        v = ((v[0].float().clamp(-1, 1) + 1) * 127.5).round().byte()     # [3, T, H, W]
        return v.permute(1, 2, 3, 0).cpu().numpy()

    def forecast(self, video, context_end_frame: int, seed: int = 0, return_video: bool = False):
        acts, lat = self._sample(self._x(video), context_end_frame, seed)
        if return_video:
            return acts, self.decode(lat)
        return acts

    def generate(self, video, context_end_frame: int, seed: int, **sample_kw):
        _, lat = self._sample(self._x(video), context_end_frame, seed, capture=False, **sample_kw)
        return self.decode(lat)


def flow_t_for_db(db: float, num_train_timesteps: int = 1000) -> float:
    """Rectified flow z_t = (1 - s) z0 + s eps: SNR = ((1 - s) / s)^2, timestep = s * num_train_timesteps."""
    s = 1.0 / (1.0 + 10 ** (db / 20))
    return s * num_train_timesteps


def ddpm_t_for_db(alphas_cumprod, db: float) -> int:
    """DDPM z_t = sqrt(a) z0 + sqrt(1 - a) eps: the integer timestep whose SNR a / (1 - a) is closest to dB."""
    import torch
    logsnr = torch.log(alphas_cumprod / (1 - alphas_cumprod).clamp_min(1e-20))
    return int(torch.argmin((logsnr - db / 10 * math.log(10)).abs()))
