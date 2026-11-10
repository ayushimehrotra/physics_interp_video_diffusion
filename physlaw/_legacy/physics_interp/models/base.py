"""Model adapter interface.

Every video diffusion model has a different diffusers pipeline class and a
different internal module layout. Everything above this file (the hook layer,
the runner, and whatever analysis you build on them) is written against
*canonical site names* and a *latent layout*, never against a concrete module
path. Adding a new model means writing one adapter — nothing else changes.

Canonical site naming
---------------------
Sites are dotted names with a numeric block index, e.g.::

    block.12.resid_pre        video-token residual stream entering block 12
    block.12.resid_post       video-token residual stream leaving block 12
    block.12.attn_out         attention contribution (pre-residual-add)
    block.12.ff_out           MLP contribution (pre-residual-add)
    block.12.text_resid_post  text-token residual stream leaving block 12
    final.resid               denoiser output stream before the un-patchify head

Patterns like ``block.*.resid_post`` or ``block.[0-9].attn_out`` select groups.

Token layout
------------
A site's tensor is ``[batch, tokens, dim]``. ``TokenLayout`` records what the
token axis means, so a hook can map a token index to a (frame, y, x) position in
the latent video grid. This is the piece that makes spatial-physics questions
("does this direction track where the ball *is*?") answerable, so it is part of
the interface rather than something each analysis re-derives.
"""

from __future__ import annotations

import fnmatch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn as nn

__all__ = ["TokenLayout", "SiteSpec", "ModelAdapter"]


@dataclass(frozen=True)
class TokenLayout:
    """Describes the token axis of a site tensor.

    A site tensor is ``[batch, tokens, dim]``. Tokens are laid out as an optional
    block of ``text_len`` text tokens followed (or preceded) by
    ``frames * height * width`` video tokens in frame-major order.

    Attributes
    ----------
    kind : "video" | "text" | "text+video" | "video+text"
        Which token groups the tensor contains and in what order.
    frames, height, width : int
        Latent-grid dimensions of the video tokens, *after* patchification.
    text_len : int
        Number of text tokens (0 when ``kind == "video"``).
    """

    kind: str
    frames: int
    height: int
    width: int
    text_len: int = 0

    @property
    def num_video_tokens(self) -> int:
        return self.frames * self.height * self.width

    @property
    def num_tokens(self) -> int:
        n = self.num_video_tokens if "video" in self.kind else 0
        return n + (self.text_len if "text" in self.kind else 0)

    def video_slice(self) -> slice:
        """Slice selecting the video tokens along the token axis."""
        if self.kind == "video":
            return slice(0, self.num_video_tokens)
        if self.kind == "text":
            return slice(0, 0)
        if self.kind == "text+video":
            return slice(self.text_len, self.text_len + self.num_video_tokens)
        if self.kind == "video+text":
            return slice(0, self.num_video_tokens)
        raise ValueError(f"unknown token layout kind: {self.kind}")

    def text_slice(self) -> slice:
        """Slice selecting the text tokens along the token axis."""
        if self.kind == "text":
            return slice(0, self.text_len)
        if self.kind == "video":
            return slice(0, 0)
        if self.kind == "text+video":
            return slice(0, self.text_len)
        if self.kind == "video+text":
            return slice(self.num_video_tokens, self.num_video_tokens + self.text_len)
        raise ValueError(f"unknown token layout kind: {self.kind}")

    def as_grid(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape the video tokens of ``[b, tokens, d]`` to ``[b, f, h, w, d]``."""
        vid = x[:, self.video_slice(), :]
        b, n, d = vid.shape
        if n != self.num_video_tokens:
            raise ValueError(
                f"expected {self.num_video_tokens} video tokens "
                f"({self.frames}x{self.height}x{self.width}), got {n}. "
                "The adapter's latent layout disagrees with the model."
            )
        return vid.reshape(b, self.frames, self.height, self.width, d)

    def grid_indices(self, frames=None, y=None, x=None) -> torch.Tensor:
        """Token indices for a (frames, y, x) box, as absolute positions.

        Each argument is ``None`` (everything) or a ``(start, end)`` pair in
        latent-grid units. Returned indices are valid for the full token axis,
        i.e. already offset past any leading text tokens.
        """
        def _rng(spec, n):
            if spec is None:
                return torch.arange(n)
            start, end = spec
            start = 0 if start is None else int(start)
            end = n if end is None else int(end)
            if not (0 <= start < end <= n):
                raise ValueError(f"region slice {spec} out of bounds for size {n}")
            return torch.arange(start, end)

        fi = _rng(frames, self.frames)
        yi = _rng(y, self.height)
        xi = _rng(x, self.width)
        # frame-major flattening: idx = f*(h*w) + y*w + x
        flat = (
            fi.view(-1, 1, 1) * (self.height * self.width)
            + yi.view(1, -1, 1) * self.width
            + xi.view(1, 1, -1)
        ).reshape(-1)
        return flat + self.video_slice().start


@dataclass(frozen=True)
class SiteSpec:
    """Where and how to tap one canonical site.

    Attributes
    ----------
    name : str
        Canonical site name, e.g. ``"block.12.resid_post"``.
    module_path : str
        Dotted path to the module *within the denoiser*, e.g.
        ``"transformer_blocks.12"``.
    hook : "output" | "input"
        Tap the module's output or its input.
    index : int | str | None
        Which value to tap. For ``hook="output"``: an int selects a tuple
        element, ``None`` means the output is a bare tensor. For
        ``hook="input"``: an int selects a positional arg, a str selects a
        keyword arg by name.
    layout : TokenLayout
        Token-axis semantics of the tapped tensor.
    writable : bool
        Whether a hook's return value is fed back into the model. False marks a
        tap the model does not consume downstream — observation only.
    block : int | None
        Depth index, used only for ordering site lists.
    """

    name: str
    module_path: str
    hook: str = "output"
    index: int | str | None = None
    layout: TokenLayout = None  # type: ignore[assignment]
    writable: bool = True
    block: int | None = None


class ModelAdapter(ABC):
    """Uniform handle on a loaded video diffusion model.

    Subclasses own four responsibilities:

    1. Loading the pipeline (``load``) with the checkpoint's correct dtype.
    2. Exposing the denoiser and its per-block tap points (``build_sites``).
    3. Running the sampler (``generate``) and reporting how the batch axis is
       organised under classifier-free guidance (``cfg_batch_order``).
    4. The three primitives inversion needs — ``encode_video``,
       ``embed_prompt`` and ``denoise`` — so that the trajectory recovery in
       :meth:`physics_interp.runner.VideoDiffusionRunner.invert` is written
       once against any backbone.

    The *schedule algebra* that inversion rests on lives here rather than in
    each subclass, because it is a property of the noise parameterisation
    (epsilon / v / flow) and not of the architecture. See :meth:`alpha_sigma`.
    """

    #: Registry key.
    key: str = ""
    #: Default HF repo id.
    default_repo: str = ""
    #: dtype the checkpoint was trained in; "auto" in a config resolves to this.
    native_dtype: str = "float16"
    #: Denoiser forward passes per denoising step. 1 when CFG is batched into a
    #: single call (the usual case); 2 when the pipeline runs cond and uncond
    #: separately. Hooks divide the call counter by this to get the step index.
    calls_per_step: int = 1

    def __init__(self, pipe, repo: str, device: torch.device, dtype: torch.dtype):
        self.pipe = pipe
        self.repo = repo
        self.device = device
        self.dtype = dtype
        # Keyed by layout: resolution changes the latent grid, and a stale site
        # map would carry the old grid into every spatial calculation.
        self._sites: dict[TokenLayout, dict[str, SiteSpec]] = {}

    # ---- loading ---------------------------------------------------------- #

    @classmethod
    @abstractmethod
    def load(cls, cfg, device: torch.device) -> "ModelAdapter":
        """Instantiate the diffusers pipeline and wrap it."""

    # ---- structure -------------------------------------------------------- #

    @property
    @abstractmethod
    def denoiser(self) -> nn.Module:
        """The transformer/UNet whose internals we hook."""

    @property
    @abstractmethod
    def blocks(self) -> nn.ModuleList:
        """The list of transformer blocks, in depth order."""

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    @property
    def hidden_dim(self) -> int:
        return int(self.denoiser.config.num_attention_heads * self.denoiser.config.attention_head_dim)

    @abstractmethod
    def latent_layout(self, num_frames: int, height: int, width: int) -> TokenLayout:
        """Latent token grid for a given output resolution, in tokens."""

    @abstractmethod
    def build_sites(self, layout: TokenLayout) -> dict[str, SiteSpec]:
        """All canonical sites for this model at the given latent layout."""

    def sites(self, layout: TokenLayout) -> dict[str, SiteSpec]:
        if layout not in self._sites:
            self._sites[layout] = self.build_sites(layout)
        return self._sites[layout]

    def site_names(self, layout: TokenLayout) -> list[str]:
        return sorted(self.sites(layout))

    def resolve_sites(
        self, patterns: str | Iterable[str], layout: TokenLayout
    ) -> list[SiteSpec]:
        """Expand glob patterns into concrete SiteSpecs, in depth order."""
        if isinstance(patterns, str):
            patterns = [patterns]
        patterns = list(patterns)
        all_sites = self.sites(layout)
        chosen: dict[str, SiteSpec] = {}
        for pat in patterns:
            matched = fnmatch.filter(all_sites.keys(), pat)
            if not matched:
                raise ValueError(
                    f"site pattern {pat!r} matched nothing. "
                    f"available: {sorted(all_sites)[:8]}... "
                    f"({len(all_sites)} sites total)"
                )
            for m in matched:
                chosen[m] = all_sites[m]
        return sorted(
            chosen.values(), key=lambda s: (s.block if s.block is not None else 10**6, s.name)
        )

    def get_module(self, module_path: str) -> nn.Module:
        """Resolve a dotted path relative to the denoiser."""
        mod: nn.Module = self.denoiser
        if module_path:
            for part in module_path.split("."):
                mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
        return mod

    # ---- sampling --------------------------------------------------------- #

    @abstractmethod
    def generate(self, *, prompt: str, generation, generator, **kwargs) -> Any:
        """Run the sampler. Returns an object with ``.frames[0]`` (list of PIL)."""

    # ---- inversion primitives --------------------------------------------- #

    @property
    def scheduler(self):
        """The pipeline's noise scheduler."""
        return self.pipe.scheduler

    @abstractmethod
    def encode_video(self, video: "torch.Tensor") -> torch.Tensor:
        """Encode ``float [T, H, W, 3]`` in [0, 1] to a scaled clean latent.

        Returns the latent in the layout the denoiser consumes, already
        multiplied by the VAE's scaling factor — i.e. the ``Z_1`` that
        inversion integrates backwards from.
        """

    @abstractmethod
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`encode_video`, as ``float [T, H, W, 3]`` in [0, 1].

        Only needed to verify that a recovered trajectory still reconstructs
        its source clip — the reconstruction check that tells you the inversion
        did not diverge.
        """

    @abstractmethod
    def embed_prompt(self, prompt: str) -> torch.Tensor:
        """Text embeddings for a single unbatched prompt, ``[1, tokens, dim]``."""

    @abstractmethod
    def denoise(
        self, latents: torch.Tensor, timestep: torch.Tensor, prompt_embeds: torch.Tensor
    ) -> torch.Tensor:
        """One raw denoiser forward. No guidance, no scheduler step.

        This is the call whose internals the hooks observe. Whatever the model
        parameterises (epsilon, v, or flow velocity) is returned verbatim;
        :meth:`split_prediction` is what turns it into ``(x0, eps)``.
        """

    # ---- schedule algebra -------------------------------------------------- #

    def alpha_sigma(self, timestep) -> tuple[float, float]:
        """Coefficients ``(alpha, sigma)`` with ``x_t = alpha * x_0 + sigma * eps``.

        This single form covers both families we care about, which is what lets
        one inversion loop serve all of them:

        * **DDPM/DDIM** (CogVideoX): ``alpha = sqrt(a_bar_t)``,
          ``sigma = sqrt(1 - a_bar_t)``.
        * **Flow matching** (LTX, Wan): ``x_t = (1 - s) x_0 + s * eps``, so
          ``alpha = 1 - s`` and ``sigma = s``.

        Expressing the update as a re-projection onto ``(alpha, sigma)`` at the
        next noise level — rather than as a hand-rolled per-scheduler formula —
        also makes the Euler step *exact* for rectified flow, where the ODE
        solution between two noise levels is precisely this interpolation.
        """
        sched = self.scheduler
        if hasattr(sched, "alphas_cumprod"):
            idx = int(timestep)
            a_bar = float(sched.alphas_cumprod[idx])
            return a_bar**0.5, (1.0 - a_bar) ** 0.5
        if hasattr(sched, "sigmas"):
            # Flow-matching schedulers index sigmas by step, not by timestep, so
            # locate the entry matching this timestep rather than assuming order.
            ts = getattr(sched, "timesteps", None)
            if ts is None:
                raise RuntimeError("flow scheduler exposes sigmas but no timesteps")
            pos = int((ts - float(timestep)).abs().argmin())
            s = float(sched.sigmas[pos])
            return 1.0 - s, s
        raise RuntimeError(
            f"{type(sched).__name__} exposes neither alphas_cumprod nor sigmas; "
            "alpha_sigma() cannot infer its noise parameterisation."
        )

    def split_prediction(
        self, model_out: torch.Tensor, sample: torch.Tensor, alpha: float, sigma: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert a raw denoiser output into ``(x0_pred, eps_pred)``.

        Reads ``prediction_type`` off the scheduler config rather than assuming
        one, because it genuinely differs between checkpoints of the *same*
        architecture — so hardcoding it would silently produce a wrong
        trajectory rather than an error.
        """
        kind = getattr(self.scheduler.config, "prediction_type", "epsilon")
        if kind == "epsilon":
            eps = model_out
            x0 = (sample - sigma * eps) / max(alpha, 1e-8)
        elif kind == "v_prediction":
            x0 = alpha * sample - sigma * model_out
            eps = alpha * model_out + sigma * sample
        elif kind == "sample":
            x0 = model_out
            eps = (sample - alpha * x0) / max(sigma, 1e-8)
        elif kind in ("flow_prediction", "flow"):
            # Rectified flow: the model predicts v = eps - x0.
            x0 = sample - sigma * model_out
            eps = sample + alpha * model_out
        else:
            raise NotImplementedError(f"unsupported prediction_type {kind!r}")
        return x0, eps

    def cfg_batch_order(self, guidance_scale: float) -> list[str]:
        """Names of the batch-axis groups during denoising, in order.

        With classifier-free guidance the denoiser sees a doubled batch; almost
        all diffusers pipelines order it ``[uncond, cond]``. Override if a model
        does something else.
        """
        return ["uncond", "cond"] if guidance_scale > 1.0 else ["cond"]

    def branch_labels(self, batch_size: int, guidance_scale: float) -> list[str]:
        """One CFG branch label per batch row.

        Hooks use this so a tap can address ``cond`` and ``uncond`` separately —
        or take their difference, which is the guidance direction itself and has
        no analogue in single-forward-pass models.
        """
        order = self.cfg_batch_order(guidance_scale)
        if batch_size % len(order) != 0:
            return ["cond"] * batch_size  # unexpected layout; don't guess
        per = batch_size // len(order)
        return [b for b in order for _ in range(per)]

    # ---- misc ------------------------------------------------------------- #

    def describe(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "repo": self.repo,
            "dtype": str(self.dtype).replace("torch.", ""),
            "device": str(self.device),
            "num_blocks": self.num_blocks,
            "hidden_dim": self.hidden_dim,
        }
