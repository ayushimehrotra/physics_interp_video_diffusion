"""CogVideoX1.5-5B-I2V adapter: the generatively competent model for steering.

CogVideoX-2b cannot show a steered clip (§7), so pixel-level steering needs a model
that produces a plausible two-ball collision. This one is image-conditioned, and the
differences from the 2b/5b adapter are all in what the denoiser is fed:

* **Image conditioning on channels.** ``in_channels = 32``: the 16-channel noisy
  latent is concatenated with a 16-channel image latent that holds the first frame's
  VAE latent at latent frame 0 and zeros elsewhere, plus ``ofs = 2.0``, exactly as
  ``CogVideoXImageToVideoPipeline`` builds it. The image latent is scaled by
  ``1 / scaling_factor`` (``invert_scale_latents``), the opposite of the video latent
  — that is the checkpoint's training convention, not a bug here.
* **Temporal patching.** ``patch_size_t = 2``: a token covers two latent frames. Use
  ``num_frames = 45`` (12 latents, no padding). At 49 frames the pipeline pads one
  latent at the front and the token at contact straddles latents 5|6, i.e. pixel
  frames 17-24 across the frame-20 contact. At 45, token-frames 0-2 are pixel frames
  0-20 (pre) and 3-5 are 21-44 (post), with no straddling token.
* **Resolution.** The checkpoint wants min(H, W) = 768, so clips are upsampled
  720x480 -> 1152x768 (exactly 1.6x) on the way into the VAE. Masks still tile the
  48x72 token grid at their native resolution (10 px per token).
* **T5 on CPU.** The text encoder is 9.5 GB in bf16 and is used once per prompt;
  keeping it off the GPU leaves room for 20736-token CFG batches. Embeds are cached.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .cogvideox import CogVideoXAdapter

__all__ = ["CogVideoX15I2VAdapter"]


class CogVideoX15I2VAdapter(CogVideoXAdapter):
    key = "cogvideox1.5-5b-i2v"
    default_repo = "/workspace/models/CogVideoX1.5-5B-I2V"
    native_dtype = "bfloat16"
    work_hw = (768, 1152)
    clip_prefix = True   # 49-frame sets are read as frames 0-44

    @classmethod
    def _pipeline_class(cls):
        from diffusers import CogVideoXImageToVideoPipeline
        return CogVideoXImageToVideoPipeline

    @classmethod
    def load(cls, cfg, device: torch.device) -> "CogVideoX15I2VAdapter":
        if cfg.cpu_offload or cfg.model_cpu_offload:
            return super().load(cfg, device)
        # Load to CPU, then move everything except T5.
        adapter = super().load(type(cfg)(**{**cfg.__dict__, "model_cpu_offload": False,
                                           "cpu_offload": False}), torch.device("cpu"))
        pipe = adapter.pipe
        pipe.transformer.to(device)
        pipe.vae.to(device)
        adapter.device = device
        adapter._embed_cache = {}
        adapter._image_latent = None
        return adapter

    # ---- layout ------------------------------------------------------------ #

    @property
    def patch_t(self) -> int:
        return int(self.denoiser.config.patch_size_t or 1)

    def pixel_frame_to_cell(self, frame: int) -> int:
        """Token-frame holding pixel frame ``frame`` (latent frame // patch_size_t)."""
        from ..video import pixel_frame_to_latent
        return pixel_frame_to_latent(frame) // self.patch_t

    # ---- conditioning ------------------------------------------------------ #

    def _to_work_res(self, video: torch.Tensor) -> torch.Tensor:
        """``[T, H, W, 3]`` -> ``[T, h, w, 3]`` at the checkpoint's working resolution."""
        if tuple(video.shape[1:3]) == self.work_hw:
            return video
        x = video.permute(0, 3, 1, 2).float()
        x = F.interpolate(x, size=self.work_hw, mode="bicubic", align_corners=False, antialias=True)
        return x.clamp(0, 1).permute(0, 2, 3, 1)

    @torch.no_grad()
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode the clip and, from its first frame, set the image conditioning."""
        x = self._to_work_res(video).to(device=self.device, dtype=self.dtype)
        x = (x * 2.0 - 1.0).permute(3, 0, 1, 2).unsqueeze(0)       # [1, C, T, H, W]
        self._reset_vae_cache()
        mode = self.pipe.vae.encode(x).latent_dist.mode()          # [1, C, F, h, w]
        if mode.shape[2] % self.patch_t:
            raise ValueError(
                f"{video.shape[0]} frames give {mode.shape[2]} latent frames, not a multiple of "
                f"patch_size_t={self.patch_t}; use 45 (see module docstring)")
        sf = self.pipe.vae.config.scaling_factor
        # The causal VAE encodes frame 0 alone, so this is the single-image encoding.
        self.set_image_latent(mode[:, :, :1].permute(0, 2, 1, 3, 4))
        return (mode * sf).permute(0, 2, 1, 3, 4)                  # [1, F, C, h, w]

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> None:
        """Set the conditioning from one ``[H, W, 3]`` frame in [0, 1]."""
        x = self._to_work_res(image[None]).to(device=self.device, dtype=self.dtype)
        x = (x * 2.0 - 1.0).permute(3, 0, 1, 2).unsqueeze(0)
        self._reset_vae_cache()
        mode = self.pipe.vae.encode(x).latent_dist.mode()
        self.set_image_latent(mode.permute(0, 2, 1, 3, 4))

    def set_image_latent(self, unscaled: torch.Tensor) -> None:
        cfg = self.pipe.vae.config
        sf = cfg.scaling_factor
        self._image_latent = unscaled / sf if getattr(cfg, "invert_scale_latents", False) else unscaled * sf

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        z = latents.to(device=self.device, dtype=self.dtype)
        z = z.permute(0, 2, 1, 3, 4) / self.pipe.vae.config.scaling_factor
        self._reset_vae_cache()
        frames = self.pipe.vae.decode(z).sample
        frames = (frames / 2.0 + 0.5).clamp(0, 1)
        return frames[0].permute(1, 2, 3, 0).float()

    @torch.no_grad()
    def embed_prompt(self, prompt: str) -> torch.Tensor:
        if prompt not in self._embed_cache:
            embeds, _ = self.pipe.encode_prompt(
                prompt=prompt, negative_prompt=None, do_classifier_free_guidance=False,
                num_videos_per_prompt=1, device=torch.device("cpu"), dtype=self.dtype)
            self._embed_cache[prompt] = embeds
        return self._embed_cache[prompt].to(self.device)

    @torch.no_grad()
    def denoise(self, latents: torch.Tensor, timestep: torch.Tensor,
                prompt_embeds: torch.Tensor) -> torch.Tensor:
        if self._image_latent is None:
            raise RuntimeError("no image conditioning: call encode_video or encode_image first")
        h = latents.to(dtype=self.dtype)
        b, f = h.shape[:2]
        img = self._image_latent.to(device=h.device, dtype=self.dtype)
        cond = torch.cat([img, img.new_zeros((1, f - img.shape[1]) + tuple(img.shape[2:]))], dim=1)
        h_in = torch.cat([h, cond.expand(b, -1, -1, -1, -1)], dim=2)
        return self.denoiser(
            hidden_states=h_in,
            encoder_hidden_states=prompt_embeds.to(dtype=self.dtype),
            timestep=timestep.expand(b).to(device=self.device),
            ofs=h.new_full((1,), 2.0),
            image_rotary_emb=self._rotary_embeddings(h),
            return_dict=False,
        )[0]
