"""Shared machinery of the diffusion-transformer adapters (cogvideox.py, wan.py, cosmos_predict.py).

Geometry: the official clip (height x width, num_frames at fps; every master/fps-th master frame) and a causal VAE:
token-frame 0 = input frame 0, token-frame j >= 1 = the next ``vae_temporal x patch_t`` frames (base.token_spec).

Rules carried over from CogVideoX-2b (docs/RESUME.md, milestone 7) and applied to every DiT:
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
import re
from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from physlaw.models.base import ModelAdapter, fit_frames, frame_means, sample_frames, token_spec

DEFAULT_PROMPT = ("A static camera looks straight down at a flat grey floor. Two balls roll across the floor, "
                  "collide with each other and bounce apart.")

# Collision families (F1-F4): two balls, labels.objects A / B with radius and material. Every other family gets a
# neutral description of what is in the frame before the event (FAMILY_SCENES; never the outcome), so no prompt
# reads a label field a family does not have.
COLLISION_FAMILIES = ("f1", "f2", "f3", "f4")
FAMILY_SCENES = {   # family -> text, or {subfamily: text, "": the family's fallback}
    "f6": "Objects slide across a flat surface.",
    "f7": {"projectile": "Side view of a ball flying through the air above a flat floor.",
           "drop": "Side view of two balls held in the air above a flat floor and released together.",
           "bounce": "Side view of a ball held in the air above a flat floor and released.",
           "": "Side view of balls moving through the air above a flat floor."},
    "f8": "Side view of a block on a ramp that runs down to a flat floor with a strip of a different surface and up "
          "a second ramp. The block is released and slides down the ramp.",
    "f9": {"pair": "Side view of two pendulums hanging side by side and swinging together.",
           "swing": "Side view of a pendulum released from one side, swinging down over a strip below its lowest "
                    "point.",
           "": "Side view of a swinging pendulum."},
    "f10": "View across an incline of two cylinders, one solid and one hollow, held side by side and released "
           "together.",
    "f11": "Side view of a block resting on the edge of a shelf, overhanging it, held up by a thin prop that drops "
           "away.",
    "f12": "Overhead view of a smooth table: a puck slides towards a bar lying at rest and hits it.",
}
SCENE_TAIL = ("Nothing else is in the scene; no people, no text, no other objects. The camera is completely fixed: no "
              "zoom, no pan, no cuts, no change of lighting. Realistic physics, clean 3D render, sharp focus, natural "
              "motion.")

LOOKS = {"rubber": "matte red rubber ball", "metal": "polished silver metal ball with soft reflections",
         "neutral": "matte dark grey ball"}


def family_of(labels: dict) -> tuple[str, str]:
    pl = labels.get("physlaw", {})
    return str(pl.get("family", "")), str(pl.get("subfamily") or "")


def family_scene(labels: dict) -> str:
    """The neutral pre-event description of a non-collision family (sub-family text where one exists)."""
    fam, sub = family_of(labels)
    if fam not in FAMILY_SCENES:
        raise KeyError(f"no prompt description for family {fam!r} (physlaw.models.dit.FAMILY_SCENES)")
    d = FAMILY_SCENES[fam]
    return d if isinstance(d, str) else d.get(sub, d[""])


def scene_prompt(labels: dict) -> str:
    """Long, literal scene description in the style of CogVideoX's training captions (the model cards
    recommend long, detailed prompts). Uses PRE-contact facts only (materials, sizes, which ball moves),
    so every twin of a group gets the same prompt. Adapted from the old project's scene_prompt.
    Non-collision families: the family's description + the same fixed-camera / render tail."""
    if family_of(labels)[0] not in COLLISION_FAMILIES:
        return f"{family_scene(labels)} {SCENE_TAIL}"
    o = labels["objects"]
    small, large = sorted(("A", "B"), key=lambda k: o[k]["radius"])
    look = lambda k: LOOKS.get(o[k].get("material"), "ball")
    name = {small: f"smaller {look(small)}", large: f"larger {look(large)}"}
    motion = (f"The {name['A']} and the {name['B']} roll toward each other across the floor at a steady pace"
              if labels.get("b_moving") else
              f"The {name['A']} rolls steadily across the floor toward the {name['B']}, which sits still")
    return (f"A still, top-down view looking straight down at a plain, smooth, evenly lit light-grey floor that fills "
            f"the whole frame. There are exactly two balls on the floor: a {name[small]} and a {name[large]}, each "
            f"casting a small soft shadow. {motion}. The two balls meet and collide once, a single clean impact, and "
            f"then both balls roll away from each other in straight lines, keeping their size, colour and shape, "
            f"sliding smoothly without spinning or bouncing off the floor. {SCENE_TAIL}")


MINIMAL_LOOKS = {"rubber": "matte red rubber ball", "metal": "shiny grey metal ball", "neutral": "matte dark grey ball"}


def minimal_prompt(labels: dict) -> str:
    """The user's minimal prompt (2026-11-05): name the two balls, say they collide and separate, nothing
    else -- no motion, outcome, camera or quality wording; the model decides the collision. First
    measurable collisions came from this template on Wan2.2-I2V-A14B. Pre-contact facts only.
    Non-collision families: the family's neutral description (FAMILY_SCENES)."""
    if family_of(labels)[0] not in COLLISION_FAMILIES:
        return family_scene(labels)
    o = labels["objects"]
    small, large = sorted(("A", "B"), key=lambda k: o[k]["radius"])
    look = lambda k: MINIMAL_LOOKS.get(o[k].get("material"), "ball")
    return (f"Overhead view of a plain flat grey floor with two balls on it: a smaller {look(small)} and a larger "
            f"{look(large)}. The two balls collide and then separate.")


def clip_prompt(cfg, labels: dict) -> str:
    """Stage-4 (encode / surprise / forecast), stage-7 track A without a template and PC2 training text for one clip:
    the config's fixed ``prompt`` on the collision families. Elsewhere the ball-collision description
    (DEFAULT_PROMPT, the text the T2V configs pin) is replaced by the family's neutral description; any other
    configured prompt is a model card's own (PhysRVG: "The video shows rigid body motion.") and kept for every family."""
    fixed = cfg.get("prompt", DEFAULT_PROMPT)
    if fixed != DEFAULT_PROMPT or family_of(labels)[0] in COLLISION_FAMILIES:
        return fixed
    return family_scene(labels)


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

LORA_SUFFIXES = {".lora_A.weight": "A", ".lora_B.weight": "B", ".lora_B.bias": "bias",
                 ".lora_magnitude_vector.weight": "m", ".lora_magnitude_vector": "m"}


def _lora_parts(sd: dict) -> dict[str, dict[str, Any]]:
    """{module name: {"A", "B"[, "bias" (PEFT lora_bias), "m" (DoRA magnitude)]}} from a diffusers
    ('transformer.<m>.lora_A.weight') or PEFT ('base_model.model.<m>.lora_A.weight') LoRA state dict."""
    out = {}
    for k, v in sd.items():
        suf = next((x for x in LORA_SUFFIXES if k.endswith(x)), None)
        if suf is None:
            raise ValueError(f"unexpected LoRA key {k!r} (known suffixes: {list(LORA_SUFFIXES)})")
        m = k[: -len(suf)]
        for pre in ("base_model.model.", "transformer."):
            if m.startswith(pre):
                m = m[len(pre):]
                break
        out.setdefault(m, {})[LORA_SUFFIXES[suf]] = v
    bad = [m for m, d in out.items() if not {"A", "B"} <= set(d)]
    if bad:
        raise ValueError(f"{len(bad)} LoRA modules without an A/B pair, e.g. {bad[:3]}")
    return out


def lora_config(path: str | Path) -> dict:
    """The PEFT adapter_config.json beside the weights ({} for a diffusers LoRA file without one)."""
    p = Path(path)
    c = (p if p.is_dir() else p.parent) / "adapter_config.json"
    return json.loads(c.read_text()) if c.exists() else {}


def lora_scale(conf: dict, module: str, r: int, scale: float = 1.0) -> float:
    """Factor on B @ A for one module, as PEFT computes it: ``scale`` x alpha / r (rsLoRA: alpha / sqrt(r)); r is the
    module's own rank (A's rows, so ``rank_pattern`` needs no lookup), alpha from ``alpha_pattern`` (PEFT's match:
    the pattern equals the module name or a dotted suffix of it) else ``lora_alpha``. No config (a diffusers file
    without alpha): alpha = r."""
    alpha = conf.get("lora_alpha", r)
    for pat, a in (conf.get("alpha_pattern") or {}).items():
        if re.match(rf"(.*\.)?{pat}$", module):
            alpha = a
            break
    return float(scale) * float(alpha) / (math.sqrt(r) if conf.get("use_rslora") else r)


def merge_lora(module, path: str | Path, scale: float = 1.0) -> int:
    """Merge a LoRA into ``module``'s Linear weights in place, as PEFT's merge does, the arithmetic in fp32 and the
    result cast to W's dtype: W += s B @ A (s from :func:`lora_scale`); DoRA (a magnitude vector m): W <- m / ||W +
    s B @ A||_row (W + s B @ A); lora_bias: b += s b_B. ``path``: a .safetensors file, or a directory holding
    adapter_model.safetensors (PEFT) or pytorch_lora_weights.safetensors (diffusers). Returns the number of merged
    layers; every LoRA module must match a module of the right shape."""
    import torch
    from safetensors.torch import load_file
    p = Path(path)
    f = next((p / n for n in ("adapter_model.safetensors", "pytorch_lora_weights.safetensors") if (p / n).exists()),
             p) if p.is_dir() else p
    conf = lora_config(p)
    dev = next(module.parameters()).device
    parts = _lora_parts(load_file(str(f), device=str(dev)))
    mods = dict(module.named_modules())
    missing = [m for m in parts if m not in mods]
    if missing:
        raise KeyError(f"{len(missing)} LoRA targets not in the model, e.g. {missing[:3]}")
    with torch.no_grad():
        for m, d in parts.items():
            W = mods[m].weight
            s = lora_scale(conf, m, d["A"].shape[0], scale)
            delta = (d["B"].float() @ d["A"].float()) * s
            if delta.shape != W.shape:
                raise ValueError(f"{m}: LoRA delta {tuple(delta.shape)} != weight {tuple(W.shape)}")
            if "m" in d:
                Wn = W.float() + delta
                W.copy_((d["m"].float().reshape(-1, 1) / torch.linalg.norm(Wn, dim=1, keepdim=True) * Wn).to(W.dtype))
            else:
                W += delta.to(W.dtype)
            if "bias" in d:
                mods[m].bias += (d["bias"].float() * s).to(mods[m].bias.dtype)
    return len(parts)


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

    generator = True

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
    def _init_schedule(self):
        """Optional: family state the hooks below need (called before the encode SNRs are mapped to timesteps)."""

    @abstractmethod
    def _t_for_db(self, db: float):
        """Training timestep whose SNR is closest to ``db``."""

    @abstractmethod
    def _add_noise(self, z0, eps, t):
        """Forward-noised latents at timestep t (the training schedule)."""

    @abstractmethod
    def _train_err(self, z_in, pred, z0, eps, t):
        """The model's training loss per latent element [1, F, C, h, w]."""

    @abstractmethod
    def _vae_encode(self, x):
        """[1, 3, T, H, W] in [-1, 1] -> model-space latents [1, F, C, h, w]."""

    @abstractmethod
    def _vae_decode(self, lat):
        """Model-space latents [1, F, C, h, w] -> pixels [1, 3, T, H, W] in [-1, 1]."""

    @abstractmethod
    def _encode_text(self, text: str):
        """(prompt_embeds, negative_embeds or None)."""

    @abstractmethod
    def _pass(self, z_t, t, embeds, keep=(), clean: int = 0):
        """One transformer pass: (prediction [B, F, C, h, w], {layer: block output [B, N, D]} for ``keep``)."""

    @abstractmethod
    def _sample(self, x, context_end_frame: int, seed: int, capture: bool = True, **kw):
        """The official sampler conditioned on the context: ({fc key: [T_post, gh, gw, D]}, final latents)."""

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
        acc, pf = {}, {}
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
                    pf.setdefault(f"{variant}_mse_all", []).append(frame_means(e))
                    if obj is not None and obj.any():
                        acc.setdefault(f"{variant}_mse_obj", []).append(float(e[obj].mean()))
                        pf.setdefault(f"{variant}_mse_obj", []).append(frame_means(e, obj))
        res = {k_: float(np.mean(v_)) for k_, v_ in acc.items()}
        # per post token-frame: mean over draws x SNRs (the weights are the same in every draw)
        res["per_frame"] = {"surprise": {k_: np.nanmean([d[0] for d in v_], 0).tolist() for k_, v_ in pf.items()},
                            "weight": {k_: v_[0][1] for k_, v_ in pf.items()}}
        return res

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
