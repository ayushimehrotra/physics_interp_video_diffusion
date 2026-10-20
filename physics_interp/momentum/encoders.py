"""Frozen video encoders as capture targets: VideoMAE-base, V-JEPA2-ViT-L, VideoMAEv2-giant.

This replaces the diffusion-inversion capture for the register's protocols.
An encoder is run once per clip, the residual stream is tapped at the patch
embedding (``embed.out``) and after every block (``block.<i>.resid_post``), and
each tap is pooled on-device over ``{A, B, all} x {pre, post, all}`` using the
simulator's segmentation masks mapped onto the token grid.

Frame sampling is explicit and recorded. Every encoder takes a fixed number of
frames; the clip has 49 and contact is after frame 20. A spec like
``"10:41:2"`` (frames 10, 12, ..., 40 — 16 frames) keeps every tubelet on one
side of contact so the phase split is exact. A spec with ``x2`` repeats each
frame (``"5:37:1x2"`` is 64 frames for V-JEPA2). A tubelet that straddles
contact is refused unless ``allow_straddle`` is set.

Truncation (§43/§47): ``truncate=k`` shows the encoder the first ``k`` frames of
motion and holds frame ``k - 1`` for the rest of the clip.

Nothing here is verified against the other box's capture code: the register
says only that the three sets' captures are "identical in pools / frames /
phase split". Treat frame specs as a documented assumption.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

__all__ = ["ENCODERS", "EncoderSpec", "parse_frames", "FrameLayout", "Encoder", "pool_weights"]

CONTACT_FRAME = 20          # last pre-contact rendered frame


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    hf_id: str
    n_frames: int
    resolution: int
    patch: int
    tubelet: int
    trust_remote_code: bool = False
    default_frames: str = "10:41:2"
    notes: str = ""


ENCODERS = {
    "videomae-base": EncoderSpec(
        "videomae-base", "MCG-NJU/videomae-base", 16, 224, 16, 2,
        notes="12 blocks, 8x14x14 = 1568 tokens, dim 768"),
    "vjepa2-vitl": EncoderSpec(
        "vjepa2-vitl", "facebook/vjepa2-vitl-fpc64-256", 64, 256, 16, 2,
        default_frames="5:37:1x2",
        notes="24 blocks, 32x16x16 = 8192 tokens, dim 1024; frames 5..36 each shown twice"),
    "videomaev2-giant": EncoderSpec(
        "videomaev2-giant", "OpenGVLab/VideoMAEv2-giant", 16, 224, 14, 2, trust_remote_code=True,
        notes="40 blocks, 8x16x16 = 2048 tokens, dim 1408"),
}

_SPEC_RE = re.compile(r"^(\d+):(\d+):(\d+)(?:x(\d+))?$")


def parse_frames(spec: str, n_frames: int) -> list[int]:
    m = _SPEC_RE.match(spec)
    if not m:
        raise ValueError(f"frame spec {spec!r}; want start:stop:step[xrepeat], e.g. 10:41:2 or 5:37:1x2")
    start, stop, step, rep = (int(m.group(i)) if m.group(i) else 1 for i in (1, 2, 3, 4))
    frames = [f for f in range(start, stop, step) for _ in range(rep)]
    if len(frames) != n_frames:
        raise ValueError(f"spec {spec!r} gives {len(frames)} frames, the encoder takes {n_frames}")
    return frames


@dataclass
class FrameLayout:
    frames: list[int]           # clip frame index per encoder frame
    tubelet: int
    grid_h: int
    grid_w: int

    @property
    def n_tubelets(self) -> int:
        return len(self.frames) // self.tubelet

    def tubelet_frames(self, j: int) -> list[int]:
        return self.frames[j * self.tubelet:(j + 1) * self.tubelet]

    def phase_of_tubelet(self, j: int, allow_straddle: bool) -> str:
        fs = self.tubelet_frames(j)
        if max(fs) <= CONTACT_FRAME:
            return "pre"
        if min(fs) > CONTACT_FRAME:
            return "post"
        if allow_straddle:
            return "post"
        raise ValueError(f"tubelet {j} covers frames {fs}, straddling contact (frame {CONTACT_FRAME}); "
                         "choose a frame spec whose tubelets fall on one side, or --allow-straddle")

    @property
    def n_tokens(self) -> int:
        return self.n_tubelets * self.grid_h * self.grid_w


def pool_weights(masks: np.ndarray, layout: FrameLayout, *, allow_straddle: bool = False,
                 truncate: int | None = None) -> tuple[np.ndarray, list[str]]:
    """``([9, n_tokens] float32, names)`` for ``{A, B, all} x {pre, post, all}``.

    ``masks`` is the clip's ``uint8 [49, H, W]`` instance mask (0 floor, 1 A, 2 B),
    at any resolution; it is resized by nearest neighbour to the token grid's
    pixel extent and reduced to occupancy fractions per patch. Rows are
    normalised to sum to 1; an empty region is an all-zero row.
    """
    T, H, W = masks.shape
    gh, gw = layout.grid_h, layout.grid_w
    ys = (np.arange(H) * gh // H)
    xs = (np.arange(W) * gw // W)
    occ = {r: np.zeros((layout.n_tubelets, gh, gw), np.float32) for r in ("A", "B", "all")}
    phases = [layout.phase_of_tubelet(j, allow_straddle) for j in range(layout.n_tubelets)]
    for j in range(layout.n_tubelets):
        for f in layout.tubelet_frames(j):
            if truncate is not None:
                f = min(f, truncate - 1)
            m = masks[f]
            for r, sid in (("A", 1), ("B", 2)):
                hit = (m == sid).astype(np.float32)
                cell = np.zeros((gh, gw), np.float32)
                np.add.at(cell, (ys[:, None], xs[None, :]), hit)
                occ[r][j] += cell
        occ["all"][j] = 1.0
    rows, names = [], []
    for r in ("A", "B", "all"):
        for p in ("pre", "post", "all"):
            w = occ[r].copy()
            for j in range(layout.n_tubelets):
                if p != "all" and phases[j] != p:
                    w[j] = 0
            w = w.reshape(-1)
            tot = w.sum()
            rows.append(w / tot if tot > 0 else w)
            names.append(f"{r}.{p}")
    return np.stack(rows), names


class Encoder:
    """A loaded HF video encoder with residual-stream taps."""

    def __init__(self, spec: EncoderSpec, device: str = "cuda", hf_id: str | None = None, dtype=None):
        import torch
        from transformers import AutoModel
        self.spec = spec
        self.device = device
        self.torch = torch
        self.model = AutoModel.from_pretrained(hf_id or spec.hf_id, trust_remote_code=spec.trust_remote_code,
                                               torch_dtype=dtype or torch.float16).to(device).eval()
        self.blocks = self._find_blocks(self.model)
        self.mean, self.std = self._norm_stats(hf_id or spec.hf_id)
        self.sites = ["embed.out"] + [f"block.{i}.resid_post" for i in range(len(self.blocks))]
        self._store: dict[str, Any] = {}
        self._handles = []
        self._attach()

    # ---- model plumbing --------------------------------------------------- #

    @staticmethod
    def _find_blocks(model):
        import torch.nn as nn
        best = None
        for name, mod in model.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) >= 6 and len({type(m) for m in mod}) == 1:
                if "predictor" in name:
                    continue
                if best is None or len(mod) > len(best[1]):
                    best = (name, mod)
        if best is None:
            raise RuntimeError("could not locate the transformer block list in the model")
        return best[1]

    def _norm_stats(self, hf_id):
        try:
            from transformers import AutoImageProcessor
            proc = AutoImageProcessor.from_pretrained(hf_id, trust_remote_code=self.spec.trust_remote_code)
            return np.array(proc.image_mean, np.float32), np.array(proc.image_std, np.float32)
        except Exception:
            return np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)

    def _attach(self):
        def pre_hook(mod, args, kwargs):
            x = args[0] if args else kwargs.get("hidden_states")
            self._store["embed.out"] = x.detach()
        self._handles.append(self.blocks[0].register_forward_pre_hook(pre_hook, with_kwargs=True))
        for i, blk in enumerate(self.blocks):
            def hook(mod, args, out, i=i):
                y = out[0] if isinstance(out, (tuple, list)) else out
                self._store[f"block.{i}.resid_post"] = y.detach()
            self._handles.append(blk.register_forward_hook(hook))

    def close(self):
        for h in self._handles:
            h.remove()

    # ---- preprocessing ---------------------------------------------------- #

    def layout(self, frames: list[int]) -> FrameLayout:
        g = self.spec.resolution // self.spec.patch
        return FrameLayout(frames, self.spec.tubelet, g, g)

    def preprocess(self, rgb: np.ndarray, frames: Sequence[int], truncate: int | None):
        """``uint8 [49, H, W, 3]`` -> ``[1, T, 3, R, R]`` normalised tensor (whole frame squeezed, no crop)."""
        from PIL import Image
        R = self.spec.resolution
        out = np.empty((len(frames), R, R, 3), np.float32)
        for k, f in enumerate(frames):
            if truncate is not None:
                f = min(f, truncate - 1)
            im = Image.fromarray(rgb[f]).resize((R, R), Image.BILINEAR)
            out[k] = (np.asarray(im, np.float32) / 255.0 - self.mean) / self.std
        t = self.torch.from_numpy(out).permute(0, 3, 1, 2)[None]          # [1, T, 3, R, R]
        return t.to(self.device, dtype=next(self.model.parameters()).dtype)

    # ---- capture ---------------------------------------------------------- #

    def capture(self, rgb: np.ndarray, masks: np.ndarray, frames: list[int], *, truncate: int | None = None,
                allow_straddle: bool = False) -> tuple[dict[str, np.ndarray], list[str], FrameLayout]:
        layout = self.layout(frames)
        W, names = pool_weights(masks, layout, allow_straddle=allow_straddle, truncate=truncate)
        x = self.preprocess(rgb, frames, truncate)
        self._store.clear()
        with self.torch.no_grad():
            self.model(pixel_values_videos=x) if "vjepa" in self.spec.hf_id.lower() else self.model(pixel_values=x)
        Wt = self.torch.from_numpy(W).to(self.device)
        pooled = {}
        for site in self.sites:
            h = self._store[site]
            if h.ndim != 3:
                raise RuntimeError(f"{site}: expected [B, N, D], got {tuple(h.shape)}")
            n = h.shape[1]
            if n == layout.n_tokens + 1:          # a CLS token in front
                h = h[:, 1:]
            elif n != layout.n_tokens:
                raise RuntimeError(f"{site}: {n} tokens, layout expects {layout.n_tokens} "
                                   f"({layout.n_tubelets}x{layout.grid_h}x{layout.grid_w})")
            pooled[site] = (Wt @ h[0].float()).to(self.torch.float16).cpu().numpy()
        return pooled, names, layout

    # ---- V-JEPA2 surprise (§59) ------------------------------------------- #

    def surprise(self, rgb: np.ndarray, masks: np.ndarray, frames: list[int], *, allow_straddle: bool = False
                 ) -> dict[str, float]:
        """Prediction error of V-JEPA2's own predictor for the post-contact tokens given the pre-contact ones.

        Context = every token of every pre-contact tubelet; target = every token of
        every post-contact tubelet. Reports L1, MSE and 1 - cos between predicted
        and encoded target tokens, over all target tokens and over ball tokens only.
        Relies on ``VJEPA2Model.forward(pixel_values_videos, context_mask=, target_mask=)``;
        if the installed transformers exposes a different predictor API this raises
        with the available output fields.
        """
        if "vjepa" not in self.spec.hf_id.lower():
            raise RuntimeError("surprise needs V-JEPA2 (an encoder with a predictor)")
        torch = self.torch
        layout = self.layout(frames)
        per_tok = layout.grid_h * layout.grid_w
        phases = [layout.phase_of_tubelet(j, allow_straddle) for j in range(layout.n_tubelets)]
        ctx = [j * per_tok + t for j, ph in enumerate(phases) if ph == "pre" for t in range(per_tok)]
        tgt = [j * per_tok + t for j, ph in enumerate(phases) if ph == "post" for t in range(per_tok)]
        W, names = pool_weights(masks, layout, allow_straddle=allow_straddle)
        ball = (W[names.index("A.post")] + W[names.index("B.post")]) > 0
        ball_t = torch.from_numpy(ball[np.array(tgt)]).to(self.device)
        x = self.preprocess(rgb, frames, None)
        with torch.no_grad():
            out = self.model(pixel_values_videos=x,
                             context_mask=[torch.tensor(ctx, device=self.device)[None]],
                             target_mask=[torch.tensor(tgt, device=self.device)[None]])
        po = getattr(out, "predictor_output", None)
        if po is None or not hasattr(po, "last_hidden_state") or not hasattr(po, "target_hidden_state"):
            raise RuntimeError(f"unexpected V-JEPA2 output fields: {list(out.keys()) if hasattr(out, 'keys') else type(out)}")
        pred = po.last_hidden_state[0].float()
        true = po.target_hidden_state[0].float()
        res = {}
        for tag, m in (("all", torch.ones(len(tgt), dtype=torch.bool, device=self.device)), ("ball", ball_t)):
            if m.sum() == 0:
                continue
            p_, t_ = pred[m], true[m]
            res[f"l1_{tag}"] = float((p_ - t_).abs().mean())
            res[f"mse_{tag}"] = float(((p_ - t_) ** 2).mean())
            res[f"cos_{tag}"] = float(1 - torch.nn.functional.cosine_similarity(p_, t_, dim=1).mean())
        res["ball_token_fraction"] = float(ball_t.float().mean())
        return res
