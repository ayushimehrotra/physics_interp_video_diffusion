"""LTX-Video 0.9.8 13B distilled (spec §4.2 group 4, output track A, ``track_a_mode: native``): multi-frame
conditioning is part of LTX-Video's training (conditioning items: images or video segments at target frame
indices). generate() only.

Pipeline: diffusers ``LTXConditionPipeline`` (diffusers >= 0.34; physics_venv has 0.35.1) with one
``LTXVideoCondition(video=<context frames>, frame_index=0)``: the context is the last 49 pre-contact frames of the
input window at 24 fps (8k + 1 frames, ``track_a``), and the pipeline returns num_frames (8k + 1) frames starting
with them. Settings (configs/models/ltx_video_13b_distilled.yaml): the distilled first pass of the model card /
diffusers docs (7 distilled timesteps, guidance 1, decode timestep 0.05, decode noise 0.025, conditioning noise 0)
at the final resolution, single pass: the card's optional latent-upsampler refinement pass is not run (its 0.9.8
diffusers upsampler repo is not public).
"""
from __future__ import annotations

import numpy as np

from physlaw.models.dit import source
from physlaw.models.track_a import ContinuationAdapter, is_8k1, to_uint8


class LTXConditionAdapter(ContinuationAdapter):

    def __init__(self, cfg, pipe, device):
        super().__init__(cfg, pipe, device)
        if not is_8k1(self.n_frames):
            raise ValueError(f"LTX-Video needs 8k+1 frames, got {self.n_frames}")
        if self.H % 32 or self.W % 32:
            raise ValueError(f"LTX-Video needs sizes divisible by 32, got {self.H} x {self.W}")

    @classmethod
    def load(cls, cfg, device: str = "cuda") -> "LTXConditionAdapter":
        import torch
        from diffusers import LTXConditionPipeline
        src, kw = source(cfg)
        pipe = LTXConditionPipeline.from_pretrained(src, torch_dtype=getattr(torch, cfg.get("dtype", "bfloat16")),
                                                    device_map=device, **kw)
        if pipe.scheduler.__class__.__name__ != cfg["official"]["scheduler"]:
            raise RuntimeError(f"scheduler {pipe.scheduler.__class__.__name__} != official {cfg['official']['scheduler']}")
        if cfg.get("vae_tiling"):
            pipe.vae.enable_tiling()
        pipe.set_progress_bar_config(disable=True)
        return cls(cfg, pipe, device)

    def condition(self, context: np.ndarray):
        from PIL import Image
        from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
        if not is_8k1(len(context)):
            raise ValueError(f"LTX condition video needs 8k+1 frames, got {len(context)}")
        return LTXVideoCondition(video=[Image.fromarray(f) for f in context], frame_index=0)

    def _continue(self, context: np.ndarray, seed: int) -> np.ndarray:
        import torch
        off = self.cfg["official"]
        out = self.pipe(conditions=[self.condition(context)], prompt=self._prompt_text,
                        negative_prompt=self.negative_prompt, height=self.H, width=self.W, num_frames=self.n_frames,
                        frame_rate=self.fps, timesteps=list(off["timesteps"]),
                        guidance_scale=float(off["guidance_scale"]), guidance_rescale=float(off["guidance_rescale"]),
                        decode_timestep=float(off["decode_timestep"]), decode_noise_scale=float(off["decode_noise_scale"]),
                        image_cond_noise_scale=float(off["image_cond_noise_scale"]),
                        generator=torch.Generator(device="cpu").manual_seed(seed), output_type="np").frames[0]
        return to_uint8(out)
