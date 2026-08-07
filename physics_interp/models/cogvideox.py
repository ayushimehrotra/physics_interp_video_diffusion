"""CogVideoX adapter (2b and 5b).

Architecture notes that drive the site map
------------------------------------------
``CogVideoXTransformer3DModel`` is an MMDiT-style stack. ``patch_embed``
concatenates text and video tokens; the transformer then splits them and each
``CogVideoXBlock`` carries **two** residual streams, returning them as a tuple
``(video_hidden_states, text_encoder_hidden_states)``. Inside a block::

    norm_h, norm_e, gate_msa, e_gate_msa = norm1(h, e, temb)
    attn_h, attn_e                       = attn1(norm_h, norm_e)
    h = h + gate_msa   * attn_h
    e = e + e_gate_msa * attn_e
    norm_h, norm_e, gate_ff, e_gate_ff   = norm2(h, e, temb)
    ff_out = ff(cat([norm_e, norm_h], dim=1))          # text FIRST
    h = h + gate_ff   * ff_out[:, text_len:]
    e = e + e_gate_ff * ff_out[:, :text_len]
    return h, e

Two consequences the site map encodes explicitly:

* ``attn_out`` / ``ff_out`` are taken **before** the AdaLN gate and the residual
  add, so editing them is scaled by the model's own timestep-dependent gate.
  Edit ``resid_post`` instead if you want an unscaled additive push.
* ``ff_out`` is a single tensor with **text tokens first**, then video — the
  opposite grouping from the block outputs. ``TokenLayout(kind="text+video")``
  records this so region masks land on the right tokens.

Video tokens are flattened frame-major: ``idx = f*(H*W) + y*W + x`` over the
patchified latent grid. For the default 49x480x720 request that is
13 x 30 x 45 = 17550 video tokens plus 226 text tokens, at dim 1920.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .base import ModelAdapter, SiteSpec, TokenLayout

__all__ = ["CogVideoXAdapter"]


class CogVideoXAdapter(ModelAdapter):
    key = "cogvideox-2b"
    default_repo = "zai-org/CogVideoX-2b"
    # CogVideoX-2b was trained in fp16; bf16 visibly degrades it. (The 5b
    # checkpoint is the opposite — see REGISTRY in models/__init__.py.)
    native_dtype = "float16"
    calls_per_step = 1  # one denoiser call per denoising step, CFG batched

    # ---- loading ---------------------------------------------------------- #

    @classmethod
    def load(cls, cfg, device: torch.device) -> "CogVideoXAdapter":
        from diffusers import CogVideoXPipeline

        repo = cfg.repo or cls.default_repo
        dtype_name = cls.native_dtype if cfg.dtype == "auto" else cfg.dtype
        dtype = getattr(torch, dtype_name)

        pipe = CogVideoXPipeline.from_pretrained(
            repo, torch_dtype=dtype, cache_dir=cfg.cache_dir
        )

        # Offload modes are mutually exclusive and both move the model
        # themselves, so only call .to(device) when neither is active.
        if cfg.cpu_offload:
            pipe.enable_sequential_cpu_offload(device=device)
        elif cfg.model_cpu_offload:
            pipe.enable_model_cpu_offload(device=device)
        else:
            pipe.to(device)

        if cfg.vae_tiling:
            pipe.vae.enable_tiling()
        if cfg.vae_slicing:
            pipe.vae.enable_slicing()
        pipe.set_progress_bar_config(disable=None)

        if cfg.compile:
            # Guarded because compilation inlines submodules and can make hook
            # site paths unresolvable — leave it off for interp runs.
            pipe.transformer = torch.compile(pipe.transformer)

        return cls(pipe=pipe, repo=repo, device=device, dtype=dtype)

    # ---- structure -------------------------------------------------------- #

    @property
    def denoiser(self) -> nn.Module:
        return self.pipe.transformer

    @property
    def blocks(self) -> nn.ModuleList:
        return self.pipe.transformer.transformer_blocks

    @property
    def hidden_dim(self) -> int:
        c = self.denoiser.config
        return int(c.num_attention_heads * c.attention_head_dim)

    @property
    def text_len(self) -> int:
        return int(getattr(self.denoiser.config, "max_text_seq_length", 226))

    def latent_layout(self, num_frames: int, height: int, width: int) -> TokenLayout:
        vae_cfg = self.pipe.vae.config
        spatial = 2 ** (len(vae_cfg.block_out_channels) - 1)               # 8
        temporal = int(getattr(vae_cfg, "temporal_compression_ratio", 4))  # 4
        patch = int(getattr(self.denoiser.config, "patch_size", 2))
        patch_t = getattr(self.denoiser.config, "patch_size_t", None)

        # Matches CogVideoXPipeline.prepare_latents.
        latent_frames = (int(num_frames) - 1) // temporal + 1
        if patch_t:
            # CogVideoX 1.5 pads the latent frames up to a multiple of
            # patch_size_t, then patchifies time as well.
            patch_t = int(patch_t)
            latent_frames += (-latent_frames) % patch_t
            grid_frames = latent_frames // patch_t
        else:
            grid_frames = latent_frames

        lat_h, lat_w = int(height) // spatial, int(width) // spatial
        if lat_h % patch or lat_w % patch:
            raise ValueError(
                f"{height}x{width} does not patchify cleanly: latent grid "
                f"{lat_h}x{lat_w} is not divisible by patch_size={patch}. "
                f"Use multiples of {spatial * patch}."
            )
        return TokenLayout(
            kind="video",
            frames=grid_frames,
            height=lat_h // patch,
            width=lat_w // patch,
            text_len=0,
        )

    # ---- site map --------------------------------------------------------- #

    def build_sites(self, layout: TokenLayout) -> dict[str, SiteSpec]:
        video = layout  # kind="video", text_len=0
        text = TokenLayout(
            kind="text", frames=layout.frames, height=layout.height,
            width=layout.width, text_len=self.text_len,
        )
        joint = TokenLayout(
            kind="text+video", frames=layout.frames, height=layout.height,
            width=layout.width, text_len=self.text_len,
        )

        sites: dict[str, SiteSpec] = {}

        def add(spec: SiteSpec) -> None:
            sites[spec.name] = spec

        for i in range(self.num_blocks):
            path = f"transformer_blocks.{i}"

            # --- residual streams (the main object of study) --------------- #
            add(SiteSpec(f"block.{i}.resid_pre", path, "input", "hidden_states", video, True, i))
            add(SiteSpec(f"block.{i}.resid_post", path, "output", 0, video, True, i))
            add(SiteSpec(f"block.{i}.text_resid_pre", path, "input", "encoder_hidden_states", text, True, i))
            add(SiteSpec(f"block.{i}.text_resid_post", path, "output", 1, text, True, i))

            # --- sublayer contributions (pre-gate, pre-residual-add) ------- #
            add(SiteSpec(f"block.{i}.attn_out", f"{path}.attn1", "output", 0, video, True, i))
            add(SiteSpec(f"block.{i}.text_attn_out", f"{path}.attn1", "output", 1, text, True, i))
            # ff sees [text, video] concatenated — note the ordering.
            add(SiteSpec(f"block.{i}.ff_out", f"{path}.ff", "output", None, joint, True, i))

        # --- embedding and head ------------------------------------------- #
        # patch_embed returns text and video concatenated; the transformer
        # splits them immediately afterwards.
        add(SiteSpec("embed.out", "patch_embed", "output", None, joint, True, None))
        # As of diffusers 0.35 norm_final runs on the video stream only (the
        # text half has already been split off). If a future version changes
        # that, HookManager._verify_layout raises rather than silently
        # mis-indexing tokens.
        add(SiteSpec("final.norm", "norm_final", "output", None, video, True, None))
        add(SiteSpec("final.resid", "norm_out", "output", None, video, True, None))
        return sites

    # ---- sampling --------------------------------------------------------- #

    def generate(self, *, prompt: str, generation, generator, **kwargs) -> Any:
        g = generation
        call: dict[str, Any] = dict(
            prompt=prompt,
            negative_prompt=g.negative_prompt,
            num_frames=g.num_frames,
            height=g.height,
            width=g.width,
            num_inference_steps=g.num_inference_steps,
            guidance_scale=g.guidance_scale,
            num_videos_per_prompt=1,
            generator=generator,
            output_type="pil",
        )
        call.update(kwargs)
        return self.pipe(**call)

    # ---- inversion primitives --------------------------------------------- #
    #
    # Two axis orders are in play and mixing them up produces a latent that is
    # the right *size* and complete nonsense, which no shape check would catch:
    #   VAE:         [B, C, F, H, W]
    #   transformer: [B, F, C, H, W]

    @torch.no_grad()
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """``float [T, H, W, 3]`` in [0, 1] -> scaled latent ``[1, F, C, h, w]``."""
        x = video.to(device=self.device, dtype=self.dtype)
        x = x * 2.0 - 1.0                                   # VAE convention
        x = x.permute(3, 0, 1, 2).unsqueeze(0)              # [1, C, T, H, W]

        self._reset_vae_cache()
        dist = self.pipe.vae.encode(x).latent_dist
        # The distribution mode, not a sample: inversion is meant to be a
        # deterministic function of the clip, and drawing from the posterior
        # would make every probe feature carry an extra dose of VAE noise.
        latents = dist.mode() * self.pipe.vae.config.scaling_factor
        return latents.permute(0, 2, 1, 3, 4)               # -> [1, F, C, h, w]

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`encode_video`, as ``float [T, H, W, 3]`` in [0, 1]."""
        z = latents.to(device=self.device, dtype=self.dtype)
        z = z.permute(0, 2, 1, 3, 4) / self.pipe.vae.config.scaling_factor

        self._reset_vae_cache()
        frames = self.pipe.vae.decode(z).sample                    # [1, C, T, H, W]
        frames = (frames / 2.0 + 0.5).clamp(0, 1)
        return frames[0].permute(1, 2, 3, 0).float()               # [T, H, W, 3]

    def _reset_vae_cache(self) -> None:
        """Clear the causal VAE's inter-chunk state.

        CogVideoX's VAE is causal over time and caches the tail of the previous
        chunk. Left over from an earlier call, that cache silently conditions
        the first frames of the next clip on a *different scene* — the kind of
        leak that shows up as a mysteriously well-performing probe.
        """
        for name in ("clear_cache", "_clear_fake_context_parallel_cache"):
            fn = getattr(self.pipe.vae, name, None)
            if callable(fn):
                fn()
                return

    @torch.no_grad()
    def embed_prompt(self, prompt: str) -> torch.Tensor:
        embeds, _ = self.pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=None,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            device=self.device,
            dtype=self.dtype,
        )
        return embeds

    @torch.no_grad()
    def denoise(
        self, latents: torch.Tensor, timestep: torch.Tensor, prompt_embeds: torch.Tensor
    ) -> torch.Tensor:
        h = latents.to(dtype=self.dtype)
        rotary = self._rotary_embeddings(h)
        out = self.denoiser(
            hidden_states=h,
            encoder_hidden_states=prompt_embeds.to(dtype=self.dtype),
            timestep=timestep.expand(h.shape[0]).to(device=self.device),
            image_rotary_emb=rotary,
            return_dict=False,
        )[0]
        return out

    def _rotary_embeddings(self, latents: torch.Tensor):
        """RoPE tables, or None for checkpoints that do not use them.

        CogVideoX-2b has ``use_rotary_positional_embeddings=False`` while 5b has
        it on, so this cannot be decided once for the architecture. Cached by
        latent shape since it is a pure function of the grid.
        """
        if not getattr(self.denoiser.config, "use_rotary_positional_embeddings", False):
            return None

        spatial = 2 ** (len(self.pipe.vae.config.block_out_channels) - 1)
        _, frames, _, lat_h, lat_w = latents.shape
        key = (frames, lat_h, lat_w)
        if getattr(self, "_rope_key", None) != key:
            self._rope_key = key
            self._rope = self.pipe._prepare_rotary_positional_embeddings(
                height=lat_h * spatial,
                width=lat_w * spatial,
                num_frames=frames,
                device=self.device,
            )
        return self._rope
