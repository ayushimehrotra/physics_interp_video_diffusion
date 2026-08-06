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
