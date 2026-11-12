"""CogVideoX1.5-5B-I2V as an output track B generator (spec §6.3): frame 0 + prompt -> clip through the official
diffusers ``CogVideoXImageToVideoPipeline`` call (no latent replacement). Stage 4 keeps using
physlaw.models.cogvideox.CogVideoX15I2VAdapter; stage 07 loads this class through the config's
``track_b_adapter``; its internals (encode / forecast / surprise) are that stage-4 adapter's own. Settings: configs/models/cogvideox15_5b_i2v.yaml (1360 x 768, 81 frames at 16 fps, 50 steps,
guidance 6, CogVideoXDPMScheduler trailing, dynamic CFG).
"""
from __future__ import annotations

import numpy as np

from physlaw.models.base import fit_frames
from physlaw.models.cogvideox import CogVideoX15I2VAdapter


class CogVideoX15I2VPipelineAdapter(CogVideoX15I2VAdapter):

    def generate(self, video, context_end_frame: int = 1, seed: int = 0):
        """Clip from frame 0 of ``video`` (uint8 [T, H, W, 3] or the [1, 3, T, H, W] tensor of to_tensor), fitted to
        height x width like every other input; noise from torch.Generator('cuda', seed)."""
        from PIL import Image
        torch = self.torch
        if torch.is_tensor(video):
            f0 = ((video[0, :, 0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
        else:
            f0 = fit_frames(np.asarray(video[:1]), (self.H, self.W), self.token_grid.fit)[0]
        out = self.pipe(image=Image.fromarray(f0), prompt_embeds=self.prompt_embeds,
                        negative_prompt_embeds=self.neg_embeds, height=self.H, width=self.W,
                        num_frames=self.n_frames, num_inference_steps=self.steps, guidance_scale=self.guidance,
                        use_dynamic_cfg=bool(self.dynamic_cfg),
                        generator=torch.Generator(self.device).manual_seed(seed), output_type="np").frames[0]
        return (np.clip(out, 0, 1) * 255).round().astype(np.uint8)
