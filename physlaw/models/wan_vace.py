"""Wan2.1-VACE-1.3B / 14B (spec §4.2 group 4, output track A, ``track_a_mode: variant``): the official
all-in-one-editing sibling of Wan2.1-T2V, used as the T2V models' track A route. generate() only.

Pipeline: diffusers ``WanVACEPipeline`` (in diffusers >= 0.34; physics_venv has 0.35.1) with VACE's "firstclip"
video extension (ali-vilab/VACE, FrameRefExpandAnnotator): ``video`` = the real context frames, then grey frames;
``mask`` = 0 on the context (known), 255 on the frames to generate (track_a.vace_inputs). The context is every
pre-contact frame of the input window (``track_a.context: all``). Official settings in
configs/models/wan21_vace_1_3b.yaml / wan21_vace_14b.yaml (github.com/Wan-Video/Wan2.1 generate.py VACE defaults);
VAE fp32, the rest bf16, as the diffusers model card loads it.
"""
from __future__ import annotations

import numpy as np

from physlaw.models.dit import source
from physlaw.models.track_a import ContinuationAdapter, to_uint8, vace_inputs


class WanVACEAdapter(ContinuationAdapter):

    @classmethod
    def load(cls, cfg, device: str = "cuda") -> "WanVACEAdapter":
        import torch
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanVACEPipeline
        src, kw = source(cfg)
        vae = AutoencoderKLWan.from_pretrained(src, subfolder="vae", torch_dtype=torch.float32, device_map=device, **kw)
        pipe = WanVACEPipeline.from_pretrained(src, vae=vae, torch_dtype=getattr(torch, cfg.get("dtype", "bfloat16")),
                                               device_map=device, **kw)
        off = cfg["official"]
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=float(off["flow_shift"]))
        if pipe.scheduler.__class__.__name__ != off["scheduler"]:
            raise RuntimeError(f"scheduler {pipe.scheduler.__class__.__name__} != official {off['scheduler']}")
        pipe.set_progress_bar_config(disable=True)
        return cls(cfg, pipe, device)

    def _continue(self, context: np.ndarray, seed: int) -> np.ndarray:
        import torch
        off = self.cfg["official"]
        video, mask = vace_inputs(context, self.n_frames)
        out = self.pipe(video=video, mask=mask, prompt=self._prompt_text, negative_prompt=self.negative_prompt,
                        height=self.H, width=self.W, num_frames=self.n_frames,
                        num_inference_steps=int(off["num_inference_steps"]),
                        guidance_scale=float(off["guidance_scale"]),
                        generator=torch.Generator(device="cpu").manual_seed(seed), output_type="np").frames[0]
        return to_uint8(out)
