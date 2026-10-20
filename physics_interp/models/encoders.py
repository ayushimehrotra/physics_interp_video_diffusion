"""Video *encoder* adapters: VideoMAE and V-JEPA2.

Why these sit beside the diffusion adapters
-------------------------------------------
An encoder answers the same probing question with two fewer axes. A diffusion
model's activation is indexed by (block, denoising step) and is recovered along
an *inverted* trajectory whose fidelity is itself a parameter; an encoder's is
indexed by block alone and comes from a single forward pass on the real pixels.
That makes an encoder the cheap control: if a physical variable does not decode
here, the diffusion questions about it are moot, and if it does, the diffusion
result has something to be compared against.

Everything above ``models/`` is reused unchanged — the hook substrate, the
region x phase pooling, the probe grid, the permutation nulls. Only the five
*diffusion* primitives on :class:`ModelAdapter` (schedule algebra, denoise,
encode/decode latents, prompt embedding) have no meaning for an encoder, and
they raise rather than pretend to work.

Geometry, measured rather than assumed
--------------------------------------
A wrong token grid is the dangerous kind of bug: it does not crash, it silently
misaligns every pooling mask and every phase split. All of this was read off the
running models.

=================  ==========  ==========  ====  ==============  ========
model              frames in   grid        dim   block output    n_blocks
=================  ==========  ==========  ====  ==============  ========
VideoMAE base      16 (fixed)  8 x 14x14   768   bare Tensor     12
V-JEPA2 ViT-L      48 (any)    24 x 16x16  1024  1-tuple         24
=================  ==========  ==========  ====  ==============  ========

The differing block-output type is not cosmetic. ``HookManager._make_output_hook``
takes a tuple element when ``SiteSpec.index`` is an int and the whole output when
it is ``None`` — and an int index against a module returning a bare tensor makes
the site *silently skipped*, which would show up much later as a missing site in
a stacked capture. Hence ``block_index`` is declared per subclass.

Both models normalise with ImageNet statistics, confirmed from their own
processors.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from .base import ModelAdapter, SiteSpec, TokenLayout

__all__ = ["EncoderAdapter", "VideoMAEAdapter", "VJEPA2Adapter", "tubelet_phase_split"]


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def tubelet_phase_split(frame_indices: Sequence[int], contact_frame: float,
                        tubelet: int) -> int:
    """Number of purely pre-contact temporal cells, or raise if one straddles.

    The whole pre/post pooling rests on no temporal cell containing frames from
    both sides of the collision. For CogVideoX that fell out of the causal VAE's
    grouping and is asserted in ``video.pixel_frame_to_latent``; here it is a
    property of *which pixel frames the encoder was fed*, so the subclasses pick
    that subsample deliberately and this function checks the result mechanically
    rather than trusting the arithmetic.
    """
    n_cells = len(frame_indices) // tubelet
    if n_cells * tubelet != len(frame_indices):
        raise ValueError(
            f"{len(frame_indices)} frames do not divide into tubelets of {tubelet}")

    cells = []
    for c in range(n_cells):
        group = frame_indices[c * tubelet:(c + 1) * tubelet]
        is_pre = [f <= contact_frame for f in group]
        if any(is_pre) and not all(is_pre):
            raise ValueError(
                f"temporal cell {c} spans frames {group}, which straddle contact at "
                f"{contact_frame}. Pre- and post-collision pooling would mix. "
                "Change the frame subsample for this encoder.")
        cells.append(all(is_pre))

    n_pre = sum(cells)
    if cells != [True] * n_pre + [False] * (n_cells - n_pre):
        raise ValueError(
            f"pre/post cells are not contiguous: {cells}. The frame subsample is "
            "not monotone in time.")
    return n_pre


class EncoderAdapter(ModelAdapter):
    """Common machinery for a plain ViT-style video encoder.

    Subclasses declare the geometry and the two call conventions that differ
    between checkpoints; everything else is shared.
    """

    #: Square pixel resolution the model consumes.
    input_size: int = 224
    #: Temporal patch depth, and spatial patch size.
    tubelet: int = 2
    patch: int = 16
    #: How many pixel frames to feed. Fixed by the position embedding for
    #: VideoMAE; free for V-JEPA2, where it is a cost/coverage choice.
    num_input_frames: int = 16
    #: forward() keyword carrying the video tensor.
    video_kwarg: str = "pixel_values"
    #: SiteSpec.index for a transformer block's output. See the module docstring
    #: — an int here against a bare-tensor output silently disables the site.
    block_index: int | None = None
    #: Dotted path to the ModuleList of blocks, relative to the model.
    blocks_path: str = "encoder.layer"
    #: Axis order the forward expects. VideoMAE v1 and V-JEPA2 take
    #: ``[B, T, C, H, W]``; VideoMAE v2 takes ``[B, C, T, H, W]``. Getting this
    #: wrong does not raise — 3 channels and 16 frames both survive a transpose
    #: into a differently-shaped-but-valid tensor only by luck, and where it does
    #: raise it does so far from the cause. Declared per subclass.
    channels_first: bool = False
    #: Fallback when the checkpoint's config exposes no geometry (v2's is empty).
    hidden_size: int | None = None

    native_dtype = "float32"
    calls_per_step = 1

    # ---- loading ---------------------------------------------------------- #

    @classmethod
    def _load_hf(cls, repo: str, dtype: torch.dtype, cache_dir):
        raise NotImplementedError

    @classmethod
    def load(cls, cfg, device: torch.device) -> "EncoderAdapter":
        repo = getattr(cfg, "repo", None) or cls.default_repo
        name = cls.native_dtype if getattr(cfg, "dtype", "auto") == "auto" else cfg.dtype
        dtype = getattr(torch, name)
        model = cls._load_hf(repo, dtype, getattr(cfg, "cache_dir", None))
        model.eval().to(device=device, dtype=dtype)
        return cls(pipe=model, repo=repo, device=device, dtype=dtype)

    # ---- structure -------------------------------------------------------- #

    @property
    def model(self) -> nn.Module:
        return self.pipe

    @property
    def denoiser(self) -> nn.Module:
        """The module the hooks are installed on.

        Named for the diffusion case it was introduced for; for an encoder it is
        simply the whole model. Kept rather than renamed so HookManager, which
        registers its step counter here, needs no encoder-specific branch.
        """
        return self.pipe

    @property
    def blocks(self) -> nn.ModuleList:
        return self.get_module(self.blocks_path)

    @property
    def hidden_dim(self) -> int:
        if self.hidden_size is not None:
            return int(self.hidden_size)
        return int(self.denoiser.config.hidden_size)

    def frame_indices(self, num_frames: int) -> list[int]:
        """Which pixel frames of a ``num_frames``-long clip to feed the model.

        Chosen so that no tubelet straddles the collision — see
        :func:`tubelet_phase_split`, which re-checks it per scene.
        """
        raise NotImplementedError

    def latent_layout(self, num_frames: int, height: int, width: int) -> TokenLayout:
        """Token grid. Fixed by the model's own input geometry, not the request.

        ``num_frames``/``height``/``width`` describe the *dataset* clip; an
        encoder resamples and resizes it to its own fixed input, so they do not
        enter the grid. The signature is the base class's.
        """
        cells = self.num_input_frames // self.tubelet
        side = self.input_size // self.patch
        return TokenLayout(kind="video", frames=cells, height=side, width=side,
                           text_len=0)

    def build_sites(self, layout: TokenLayout) -> dict[str, SiteSpec]:
        sites: dict[str, SiteSpec] = {}
        for i in range(self.num_blocks):
            path = f"{self.blocks_path}.{i}"
            sites[f"block.{i}.resid_post"] = SiteSpec(
                f"block.{i}.resid_post", path, "output", self.block_index,
                layout, True, i)
            sites[f"block.{i}.resid_pre"] = SiteSpec(
                f"block.{i}.resid_pre", path, "input", 0, layout, True, i)
        # The patch embedding, before any block has run. This is the encoder's
        # analogue of the VAE-latent control in the diffusion pipeline: a linear
        # readout that already works here says the quantity was in the pixels,
        # not built by the transformer.
        sites["embed.out"] = SiteSpec(
            "embed.out", f"{self.blocks_path}.0", "input", 0, layout, True, -1)
        return sites

    # ---- the one thing an encoder does ------------------------------------ #

    def preprocess(self, frames: np.ndarray) -> torch.Tensor:
        """``uint8 [T, H, W, 3]`` -> normalised ``[1, T, 3, S, S]``.

        Resized, never cropped. A centre crop of a 3:2 clip would cut the frame
        edges, and the generator only guarantees the spheres stay inside the
        *full* frame — up to 0.2% of frames already have a sphere near the
        border. Squashing 720x480 to a square is a fixed affine distortion of
        the geometry, which a linear probe absorbs; losing an object is not
        recoverable at all.
        """
        x = torch.as_tensor(np.asarray(frames))
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        x = x.permute(0, 3, 1, 2).to(device=self.device, dtype=torch.float32)
        x = torch.nn.functional.interpolate(
            x, size=(self.input_size, self.input_size), mode="bilinear",
            align_corners=False, antialias=True)
        mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std                                   # [T, 3, S, S]
        x = x.unsqueeze(0)                                     # [1, T, 3, S, S]
        if self.channels_first:
            x = x.permute(0, 2, 1, 3, 4)                       # -> [1, 3, T, S, S]
        return x.to(self.dtype)

    def preprocess_masks(self, masks: np.ndarray) -> np.ndarray:
        """``uint8 [T, H, W]`` instance ids -> ``[T, S, S]``, resized to match.

        The same geometric transform :meth:`preprocess` applies to the pixels,
        because a pooling mask that is not in the model's own frame addresses the
        wrong tokens. **Nearest neighbour, not bilinear**: these are instance ids,
        and interpolating between sphere A (1) and sphere B (2) would invent a
        value of 1.5 that belongs to neither and is silently dropped by the
        ``masks == seg_id`` test, eroding both masks along their shared edge.
        """
        m = torch.as_tensor(np.asarray(masks)).float()[:, None]      # [T, 1, H, W]
        m = torch.nn.functional.interpolate(
            m, size=(self.input_size, self.input_size), mode="nearest")
        return m[:, 0].round().to(torch.uint8).numpy()

    @torch.no_grad()
    def encode(self, frames: np.ndarray) -> None:
        """One forward pass. Returns nothing — the hooks do the recording."""
        self.denoiser(**{self.video_kwarg: self.preprocess(frames)})

    # ---- diffusion primitives: not applicable ----------------------------- #

    def _no(self, what: str):
        raise NotImplementedError(
            f"{type(self).__name__} is an encoder: {what} has no meaning for it. "
            "Use runner.encode_scene() rather than invert()/capture_scene().")

    @property
    def scheduler(self):
        self._no("a noise scheduler")

    def generate(self, **kw) -> Any:
        self._no("sampling")

    def encode_video(self, video):
        self._no("a diffusion latent")

    def decode_latents(self, latents):
        self._no("latent decoding")

    def embed_prompt(self, prompt: str):
        self._no("prompt embedding")

    def denoise(self, latents, timestep, prompt_embeds):
        self._no("denoising")

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "family": "encoder",
                "input_frames": self.num_input_frames, "input_size": self.input_size}


class VideoMAEAdapter(EncoderAdapter):
    """VideoMAE base — masked *pixel* reconstruction, ViT-B.

    16 input frames is not a choice: the position embedding is built for exactly
    that many and a 48-frame batch raises a shape error. So a 49-frame clip is
    subsampled, and the stride/offset are picked to keep the collision on a
    tubelet boundary: frames 3, 6, ... 48 put the last pre-contact frame (18) at
    subsample index 5 and the first post-contact frame (21) at index 6, so
    tubelet 2 = {15, 18} is pre and tubelet 3 = {21, 24} is post. Offsets 0-2 all
    land the boundary mid-tubelet instead.
    """

    key = "videomae-base"
    default_repo = "MCG-NJU/videomae-base"
    input_size = 224
    num_input_frames = 16
    video_kwarg = "pixel_values"
    block_index = None            # returns a bare Tensor

    @classmethod
    def _load_hf(cls, repo, dtype, cache_dir):
        from transformers import VideoMAEModel

        return VideoMAEModel.from_pretrained(repo, torch_dtype=dtype,
                                             cache_dir=cache_dir)

    def frame_indices(self, num_frames: int) -> list[int]:
        if num_frames != 49:
            raise ValueError(
                f"the frame subsample below is derived for a 49-frame clip with "
                f"contact at frame 20; got {num_frames}. Re-derive it before use.")
        return list(range(3, 49, 3))


class VideoMAEv2Adapter(EncoderAdapter):
    """VideoMAE **v2** — the checkpoint Joseph et al. actually probe.

    Same token geometry as v1 (8 x 14 x 14 = 1568 at dim 768, 12 blocks) and the
    same bare-tensor block output, but three things differ and none of them are
    guessable from the family name — all were read off the running model:

    * blocks live at ``model.blocks``, not ``encoder.layer``;
    * the forward takes ``[B, C, T, H, W]``, the transpose of v1's
      ``[B, T, C, H, W]``;
    * ``config`` exposes no geometry fields at all, so hidden size and the grid
      are declared here rather than read.

    Loading needs ``trust_remote_code`` plus ``timm`` and ``easydict``, which the
    remote modeling file imports.
    """

    key = "videomaev2-base"
    default_repo = "OpenGVLab/VideoMAEv2-Base"
    input_size = 224
    num_input_frames = 16
    video_kwarg = "pixel_values"
    block_index = None            # bare Tensor, like v1
    blocks_path = "model.blocks"
    channels_first = True
    hidden_size = 768

    @classmethod
    def _load_hf(cls, repo, dtype, cache_dir):
        from transformers import AutoModel

        return AutoModel.from_pretrained(
            repo, trust_remote_code=True, cache_dir=cache_dir)

    def frame_indices(self, num_frames: int) -> list[int]:
        # Same derivation as v1: 16 frames at stride 3 from offset 3 keeps the
        # collision on a tubelet boundary. See VideoMAEAdapter.
        if num_frames != 49:
            raise ValueError(
                f"the frame subsample is derived for a 49-frame clip with contact "
                f"at frame 20; got {num_frames}. Re-derive it before use.")
        return list(range(3, 49, 3))


class VJEPA2Adapter(EncoderAdapter):
    """V-JEPA2 ViT-L — *latent*-space prediction, the other training objective.

    Unlike VideoMAE this accepts any frame count (verified at 16/32/48/64), so
    the clip is used almost whole: frames 1..48, which is 24 tubelets. Dropping
    frame 0 rather than frame 48 is what puts the boundary in the right place —
    the last pre-contact frame (20) lands at subsample index 19, so tubelet 9 =
    {19, 20} is pre and tubelet 10 = {21, 22} is post. Keeping frame 0 instead
    would put frames 20 and 21 in one tubelet.
    """

    key = "vjepa2-vitl"
    default_repo = "facebook/vjepa2-vitl-fpc64-256"
    input_size = 256
    num_input_frames = 48
    video_kwarg = "pixel_values_videos"
    block_index = 0               # returns a 1-tuple

    @classmethod
    def _load_hf(cls, repo, dtype, cache_dir):
        from transformers import AutoModel

        return AutoModel.from_pretrained(repo, torch_dtype=dtype,
                                         cache_dir=cache_dir)

    def frame_indices(self, num_frames: int) -> list[int]:
        if num_frames != 49:
            raise ValueError(
                f"the frame subsample below is derived for a 49-frame clip with "
                f"contact at frame 20; got {num_frames}. Re-derive it before use.")
        return list(range(1, 49))


class VideoMAEv2GiantAdapter(VideoMAEv2Adapter):
    """VideoMAE **v2-giant** — the checkpoint Joseph et al. actually probe.

    Their App C.1 notes that smaller VideoMAE-v2 variants "fail to exhibit
    reliable emergence", and §19 attributed our VideoMAE-vs-V-JEPA2 split to
    exactly that. This is the 1B-parameter ViT-g, so it is the direct test.

    Three geometry fields differ from v2-base and none is readable from the
    config, which nests everything under ``model_config`` rather than exposing
    ``hidden_size``:

    * ``embed_dim`` 1408, not 768;
    * ``depth`` 40 blocks, not 12;
    * ``patch_size`` **14**, not 16 — so the spatial grid is 16 x 16 = 256 and a
      clip is 8 x 256 = **2048** tokens, against v1/v2-base's 8 x 14 x 14 = 1568.

    Everything else — the ``model.blocks`` path, the channels-first forward, the
    16-frame stride-3 subsample that keeps the collision off a tubelet boundary —
    is inherited unchanged.
    """

    key = "videomaev2-giant"
    default_repo = "OpenGVLab/VideoMAEv2-giant"
    patch = 14
    hidden_size = 1408
