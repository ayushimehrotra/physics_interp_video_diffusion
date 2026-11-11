"""Cosmos-Predict2.5-2B Video2World (spec §4.2 group 4, output track A): generate() only.

Pipeline: diffusers ``Cosmos2_5_PredictBasePipeline`` at the official diffusers revision of nvidia/Cosmos-Predict2.5-2B
(``diffusers/base/post-trained``), called as NVIDIA's cosmos-predict2.5 repo does in scripts/diffusers_inference.py.
That script's environment (diffusers >= 0.39 for the 2.5 transformer, transformers 4.57, cosmos-guardrail) does not
fit physics_venv (diffusers 0.35.1, which keeps every other model's outputs bit-identical), so this adapter runs in
its own venv, /root/envs/cosmos_venv (scripts/00_setup_envs.sh, PHYSLAW_COSMOS=1). The pipeline's guardrail stays on,
as shipped (the NVIDIA Open Model License forbids bypassing it); it needs the gated Cosmos-Guardrail1 and
Llama-Guard-3-8B weights.

Video2World conditions on the LAST 5 frames (2 latent frames) of the video it is given and returns num_frames
frames starting with those 5. Here it is given the real pre-contact context (the clip's first k frames, k from
context_frames), so the returned clip is aligned with input_frames(): real frames 0..k-6, then the pipeline's 93
frames (its first 5 reproduce frames k-5..k-1), cut to num_frames. Official settings in
configs/models/cosmos_predict25_2b.yaml.
"""
from __future__ import annotations

import numpy as np

from physlaw.models.base import ModelAdapter, fit_frames, sample_frames, token_spec
from physlaw.models.dit import DEFAULT_PROMPT, source


def _diffusers(min_version: str):
    import diffusers
    from packaging.version import Version
    if Version(diffusers.__version__) < Version(min_version):
        raise RuntimeError(f"Cosmos-Predict2.5 needs diffusers >= {min_version} (got {diffusers.__version__}): run it "
                           "with /root/envs/cosmos_venv/bin/python (scripts/00_setup_envs.sh, PHYSLAW_COSMOS=1)")
    return diffusers


class CosmosPredictAdapter(ModelAdapter):

    def __init__(self, cfg, pipe, device):
        super().__init__(cfg)
        import torch
        self.torch, self.pipe, self.device = torch, pipe, device
        off = cfg["official"]
        self.H, self.W = int(off["height"]), int(off["width"])
        self.n_frames, self.fps = int(off["num_frames"]), int(off["fps"])
        self.n_cond = int(off["num_latent_conditional_frames"])
        self.token_grid = token_spec(cfg)
        self.set_prompt(cfg.get("prompt", DEFAULT_PROMPT))

    @classmethod
    def load(cls, cfg, device: str = "cuda") -> "CosmosPredictAdapter":
        import torch
        d = _diffusers(cfg.get("min_diffusers", "0.39.0"))
        src, kw = source(cfg)
        pipe = d.Cosmos2_5_PredictBasePipeline.from_pretrained(src, torch_dtype=getattr(torch, cfg.get("dtype", "bfloat16")),
                                                               device_map=device, **kw)
        pipe.set_progress_bar_config(disable=True)
        return cls(cfg, pipe, device)

    def set_prompt(self, text: str):
        self._prompt_text = text

    def input_frames(self, labels: dict) -> list[int]:
        sc = labels["scene_constants"]
        return sample_frames(master_fps=sc["frame_rate"], n_master=sc["num_frames"], fps=self.fps,
                             n_frames=self.n_frames, contact_frame=labels["physlaw"]["first_post_contact_frame"],
                             center_on_contact=bool(self.cfg.get("input", {}).get("center_on_contact", False)))

    def context_frames(self, labels: dict) -> int:
        """Leading input frames in whole causal latent frames (1 + 4j) that are entirely pre-contact."""
        fr = self.input_frames(labels)
        fpc = labels["physlaw"]["first_post_contact_frame"]
        k = 0
        for idx in self.token_grid.token_frames(len(fr)):
            if max(fr[i] for i in idx) < fpc:
                k = idx[-1] + 1
            else:
                break
        return k

    def to_tensor(self, frames_uint8: np.ndarray):
        x = fit_frames(frames_uint8, (self.H, self.W), self.token_grid.fit)
        t = self.torch.from_numpy(x).to(self.device).permute(3, 0, 1, 2)[None].float() / 127.5 - 1.0
        return t                                                          # [1, 3, T, H, W] in [-1, 1]

    def generate(self, video, context_end_frame: int, seed: int):
        torch = self.torch
        off = self.cfg["official"]
        x = video if torch.is_tensor(video) else self.to_tensor(video)
        n_in = 4 * (self.n_cond - 1) + 1
        k = context_end_frame
        if k < n_in:
            raise ValueError(f"context has {k} frames; Video2World needs {n_in}")
        out = self.pipe(video=x[:, :, :k].to(self.pipe.vae.dtype), prompt=self._prompt_text,
                        negative_prompt=off.get("negative_prompt"), height=self.H, width=self.W,
                        num_frames=self.n_frames, num_inference_steps=int(off["num_inference_steps"]),
                        guidance_scale=float(off["guidance_scale"]), num_latent_conditional_frames=self.n_cond,
                        conditional_frame_timestep=float(off["conditional_frame_timestep"]),
                        generator=torch.Generator(device="cpu").manual_seed(seed), output_type="np").frames[0]
        gen = (np.clip(out, 0, 1) * 255).round().astype(np.uint8)
        real = ((x[0, :, : k - n_in].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()
        return np.concatenate([real, gen])[: self.n_frames]

    def encode(self, video, noise_level=None):
        raise NotImplementedError(f"{self.name}: output track A generator only (spec §4.2 group 4)")

    forecast = surprise = encode
