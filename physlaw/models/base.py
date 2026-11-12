"""Adapter interfaces (spec §4.1) and the model-agnostic input geometry.

:class:`Adapter` is the common part (config, input window, capability flags); :class:`ModelAdapter` adds the internals
interface (encode / forecast / surprise) that stage 4 and every probe level read. Models with internals (vjepa2.py,
videomae2.py, untrained.py, positive_control.py, the DiTs in dit.py / cogvideox.py / wan.py) subclass ModelAdapter;
the pipeline-only output generators (cosmos_predict.py, track_a.ContinuationAdapter) subclass Adapter.

Frame sampling is fixed here, not in each adapter, so every model's view of a
clip is recorded identically: a model with native rate ``fps`` gets every
``master_fps / fps``-th master frame (must divide exactly; the 48 fps master
makes 8, 16, 24 fps exact), never a repeated frame.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TokenGridSpec:
    """How master pixels map to a model's token grid.

    ``input_hw``   the frame size the model is fed (after fit)
    ``patch``      spatial pixels per token side, in ``input_hw`` pixels
                   (for latent models: vae_spatial * patch)
    ``tubelet``    input frames per token-frame (for latent models: vae temporal
                   compression * temporal patch; first latent frame of a causal VAE
                   covers 1 frame — set ``first_tubelet`` = 1 then)
    ``fit``        "squash" (resize the whole frame, aspect changes; the old
                   project's choice) or "letterbox" (pad to aspect, then resize)
    """
    input_hw: tuple[int, int]
    patch: int
    tubelet: int
    fit: str = "squash"
    first_tubelet: int | None = None

    @property
    def grid_hw(self) -> tuple[int, int]:
        return self.input_hw[0] // self.patch, self.input_hw[1] // self.patch

    def token_frames(self, n_input_frames: int) -> list[list[int]]:
        """Input-frame indices (0-based, into the sampled clip) covered by each token-frame."""
        out, i = [], 0
        if self.first_tubelet:
            out.append(list(range(min(self.first_tubelet, n_input_frames))))
            i = self.first_tubelet
        while i < n_input_frames:
            out.append(list(range(i, min(i + self.tubelet, n_input_frames))))
            i += self.tubelet
        return out


def frame_means(err, mask=None) -> tuple[list[float], list[float]]:
    """Per token-frame mean of a per-token error ``err`` [T, ...] (numpy or torch; NaN = not scored) over ``mask``
    (bool, same shape; None = every position). Returns (values [T], NaN where empty; weights [T] = positions averaged),
    so the weighted mean of the values is the clip-level mean over every scored position."""
    to_np = lambda a: np.asarray(a.detach().float().cpu() if hasattr(a, "detach") else a, float)
    e = to_np(err).reshape(len(err), -1)
    m = np.isfinite(e) if mask is None else (to_np(mask).reshape(e.shape) > 0) & np.isfinite(e)
    w = m.sum(1).astype(float)
    with np.errstate(invalid="ignore"):
        v = np.where(w > 0, np.where(m, e, 0.0).sum(1) / np.maximum(w, 1), np.nan)
    return v.tolist(), w.tolist()


def add_frame_means(res: dict, name: str, err, mask=None) -> None:
    """res["per_frame"]["surprise" / "weight"][name] = frame_means(err, mask)."""
    v, w = frame_means(err, mask)
    pf = res.setdefault("per_frame", {"surprise": {}, "weight": {}})
    pf["surprise"][name], pf["weight"][name] = v, w


def sample_frames(*, master_fps: int, n_master: int, fps: int, n_frames: int,
                  contact_frame: int | None = None, center_on_contact: bool = False) -> list[int]:
    """Master-frame indices a model sees: every (master_fps/fps)-th frame, n_frames long.

    With ``center_on_contact`` the window is placed so ``contact_frame`` sits at the
    window's centre (clamped to the clip); otherwise it starts at frame 0.
    """
    if master_fps % fps:
        raise ValueError(f"model fps {fps} does not divide master fps {master_fps}: "
                         "frames would be unevenly spaced")
    step = master_fps // fps
    span = (n_frames - 1) * step + 1
    if span > n_master:
        raise ValueError(f"{n_frames} frames at {fps} fps need {span} master frames, clip has {n_master}")
    start = 0
    if center_on_contact:
        if contact_frame is None:
            raise ValueError("center_on_contact needs contact_frame")
        start = int(np.clip(contact_frame - (n_frames // 2) * step, 0, n_master - span))
    return [start + k * step for k in range(n_frames)]


def fit_frames(frames: np.ndarray, out_hw: tuple[int, int], fit: str = "squash") -> np.ndarray:
    """uint8 [T, H, W, C] -> uint8 [T, h, w, C] by the given fit (bilinear)."""
    from PIL import Image
    h, w = out_hw
    T, H, W = frames.shape[:3]
    if fit == "letterbox":
        scale = min(h / H, w / W)
        nh, nw = int(round(H * scale)), int(round(W * scale))
        out = np.zeros((T, h, w, frames.shape[3]), np.uint8)
        y0, x0 = (h - nh) // 2, (w - nw) // 2
        for t in range(T):
            out[t, y0:y0 + nh, x0:x0 + nw] = np.asarray(Image.fromarray(frames[t]).resize((nw, nh), Image.BILINEAR))
        return out
    if fit != "squash":
        raise ValueError(fit)
    return np.stack([np.asarray(Image.fromarray(f).resize((w, h), Image.BILINEAR)) for f in frames])


class Adapter(ABC):
    """Common part of every adapter. Capabilities are class attributes, checked by
    ``physlaw.models.registry.load_adapter(cfg, need=...)`` before a model is loaded:

    ``internals``  encode / forecast / surprise (:class:`ModelAdapter`): stage 4 and every probe level;
    ``generator``  ``generate(video, context_end_frame, seed) -> uint8 [T, H, W, 3]``: stage 07 output tracks;
    ``predicts``   (set per instance by adapters whose weights decide it) forecast / surprise are available, not
                   just encode() (False: encoder-only VideoMAE-v2 base; stage 4 caches encode() alone, L0 only).
    """

    internals: bool = False
    generator: bool = False
    predicts: bool = False
    name: str
    layers: list[int]
    token_grid: TokenGridSpec
    cfg: dict[str, Any]

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.name = cfg["name"]
        self.layers = list(cfg.get("layers", []))

    @classmethod
    @abstractmethod
    def load(cls, cfg: dict[str, Any], device: str = "cuda") -> "Adapter": ...

    @abstractmethod
    def input_frames(self, labels: dict) -> list[int]:
        """Master-frame indices this model is fed for a clip (via :func:`sample_frames`)."""


class ModelAdapter(Adapter):
    """Spec §4.1: an adapter with internals. Tensors are torch tensors; numpy is accepted for inputs."""

    internals = predicts = True

    @abstractmethod
    def encode(self, video, noise_level=None) -> dict[int, Any]:
        """Activations per cached layer, [T_tok, H_tok, W_tok, D].
        Diffusion: noise_level is an SNR in dB; VAE-encode, forward-noise, one DiT pass."""

    @abstractmethod
    def forecast(self, video, context_end_frame: int) -> dict[int, Any]:
        """The model's own prediction of future token-frames from frames < context_end_frame."""

    @abstractmethod
    def surprise(self, video, context_end_frame: int, object_masks) -> dict[str, Any]:
        """Zero-fit surprise on the post-context segment, restricted to object regions.
        Returns {metric: float} (several distances: l1, mse, cosine...) and ``per_frame``: {"surprise": {metric:
        [T_post]}, "weight": {metric: [T_post]}}, the metric per post-context token-frame (:func:`frame_means`), so
        L2 can restrict to the clean window (stage 4 adds the token-frame times)."""


def token_spec(cfg) -> TokenGridSpec:
    """A model's token grid from its config, without loading the model.
    Encoders: input.resolution (square), token_grid.patch / tubelet.
    Diffusion (``official`` block): official height x width; patch = vae_spatial x patch; tubelet =
    vae_temporal x patch_t; first token-frame = 1 input frame (causal VAE). Matches the adapters."""
    fit = cfg.get("input", {}).get("fit", "squash")
    tg = cfg["token_grid"]
    if "official" in cfg:
        off = cfg["official"]
        return TokenGridSpec((int(off["height"]), int(off["width"])), int(tg["vae_spatial"]) * int(tg["patch"]),
                             int(tg["vae_temporal"]) * int(tg.get("patch_t") or 1), fit, first_tubelet=1)
    R = int(cfg["input"]["resolution"])
    return TokenGridSpec((R, R), int(tg["patch"]), int(tg["tubelet"]), fit)


PRED_LAYER = "pred/L{}"   # key of predictor (V-JEPA 2 / PC1) or pixel-decoder (VideoMAE) block i (pred_real, L4 sites)


def spatial_mask_groups(n_spatial: int, n_masks: int, seed: int = 0) -> list[np.ndarray]:
    """Partition of the spatial token positions into ``n_masks`` random groups (seeded, so identical for
    every clip). P-pred-real (spec §5.2) masks one group in every post-event token-frame per pass."""
    perm = np.random.default_rng(seed).permutation(n_spatial)
    return [np.sort(perm[k::n_masks]) for k in range(n_masks)]
