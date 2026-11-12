"""Output track A (spec §6.2): which generators may continue the REAL pre-contact context, and with which frames.

Every generator config states (spec §4.2, scripts/07_generate_outputs.py):
  ``output_track``   A (continues several real context frames; scored against the real scene's physics),
                     B (generates from frame 0 + prompt; scored on self-consistency) or none.
  ``track_a_mode``   native  - the model's own trained multi-frame conditioning (Cosmos Video2World, PhysRVG v2v,
                               LTX-Video multi-frame conditions);
                     variant - an official sibling checkpoint of a text-only model that conditions on frames
                               (Wan2.1-VACE for Wan2.1-T2V);
                     hack    - conditioning the model was not trained for (latent replacement, or Wan2.2-TI2V-5B
                               held on PhysRVG's 2 context latents); runs only flagged, or with --allow-hack;
                     n/a     - no track A path at all (the I2V track B models).
A model conditioned on one frame cannot be track A.

``track_a`` block (models whose context is not simply "every whole pre-contact latent frame"):
  ``context``  frames handed to the model, ending at the LAST pre-contact frame of the model's input window
               (an int, or "all" = every pre-contact frame of the window);
  ``align``    the context length is rounded down to 1 + align * j frames (whole causal latent frames: 4 for
               the Wan VAE, 8 for LTX's); default 1.
The generated clip is aligned with the model's input window (physlaw.eval.outputs.Geometry): real frames before
the context, then the model's output from the first context frame on (its first frames reproduce the context),
cut to num_frames (``splice``).

Torch-free helpers (tested offline in tests/test_tracks.py) plus ``ContinuationAdapter``, the base of the
pipeline-only track A adapters (wan_vace.py, ltx.py).
"""
from __future__ import annotations

from abc import abstractmethod

import numpy as np

from physlaw.models.base import Adapter, fit_frames, sample_frames, token_spec

TRACKS = ("A", "B", "none")
MODES = ("native", "variant", "hack", "n/a")


def track_fields(cfg) -> tuple[str, str]:
    """(output_track, track_a_mode) of a generator config; raises if either is missing or invalid."""
    t, m = cfg.get("output_track"), cfg.get("track_a_mode")
    if t not in TRACKS or m not in MODES:
        raise ValueError(f"{cfg.get('name')}: output_track {t!r} / track_a_mode {m!r} must be one of {TRACKS} / "
                         f"{MODES} (configs/models/*.yaml)")
    if t == "A" and m == "n/a":
        raise ValueError(f"{cfg.get('name')}: track A needs a track_a_mode")
    return t, m


def is_generator(cfg) -> bool:
    """Diffusion / generator configs (an ``official`` sampling block); encoders and PC1 are not."""
    return "official" in cfg


def model_frames(cfg, labels) -> list[int]:
    """Master frames of the model's input window (as the adapters' input_frames and eval Geometry)."""
    sc, off = labels["scene_constants"], cfg["official"]
    return sample_frames(master_fps=sc["frame_rate"], n_master=sc["num_frames"], fps=int(off["fps"]),
                         n_frames=int(off["num_frames"]), contact_frame=labels["physlaw"]["first_post_contact_frame"],
                         center_on_contact=bool(cfg.get("input", {}).get("center_on_contact", False)))


def n_pre_contact(frames: list[int], contact_frame: int) -> int:
    """Leading window frames strictly before the first post-contact master frame."""
    n = 0
    for f in frames:
        if f >= contact_frame:
            break
        n += 1
    return n


def context_len(cfg, end: int) -> int:
    """Frames given to the model when the context ends at window index ``end`` (exclusive)."""
    ta = cfg["track_a"]
    n = end if ta["context"] == "all" else min(int(ta["context"]), end)
    a = int(ta.get("align", 1))
    n = 1 + ((n - 1) // a) * a if n >= 1 else 0
    if n < int(ta.get("min_context", 1)) or n < 1:
        raise ValueError(f"{cfg.get('name')}: only {end} pre-contact frames in the window; context {ta['context']}")
    return n


def context_end(cfg, labels) -> int:
    """Window index after the last pre-contact frame: the k stage 07 passes to generate()."""
    return n_pre_contact(model_frames(cfg, labels), labels["physlaw"]["first_post_contact_frame"])


def context_bounds(cfg, labels) -> tuple[int, int]:
    e = context_end(cfg, labels)
    return e - context_len(cfg, e), e


def context_master_frames(cfg, labels) -> list[int]:
    """Master frames the model is conditioned on (written into every track A output record)."""
    s, e = context_bounds(cfg, labels)
    return model_frames(cfg, labels)[s:e]


def splice(real: np.ndarray, gen: np.ndarray, start: int, n_frames: int) -> np.ndarray:
    """Real window frames [0, start) then the model's frames (which begin at the first context frame)."""
    out = np.concatenate([np.asarray(real[:start]), np.asarray(gen)])
    if len(out) < n_frames:
        raise ValueError(f"spliced clip has {len(out)} frames < {n_frames}")
    return out[:n_frames]


def is_8k1(n: int) -> bool:
    return n >= 1 and (n - 1) % 8 == 0


def vace_inputs(context: np.ndarray, n_frames: int, grey: int = 128):
    """VACE "firstclip" extension (ali-vilab/VACE vace/annotators/frameref.py FrameRefExpandAnnotator): the known
    clip, then frames filled with grey (VACE's REF_COLOR is 127.5; 128 as diffusers' WanVACEPipeline example does,
    since the pipeline takes the frames as images: 0.004 off 0 in [-1, 1]); mask 0 on known frames, 255 on frames
    to generate. Returns (video, mask): T PIL RGB / T PIL 'L' images, as WanVACEPipeline(video=..., mask=...) takes
    them (it checks len(video) == len(mask) and reads the size from video[0], so per-frame numpy arrays would not
    do: a [H, W, 3] array is read as a [B, H, W] batch)."""
    from PIL import Image
    context = np.asarray(context, dtype=np.uint8)
    k, H, W = context.shape[:3]
    if k >= n_frames:
        raise ValueError(f"context {k} >= num_frames {n_frames}")
    fill = Image.new("RGB", (W, H), (grey, grey, grey))
    known, todo = Image.new("L", (W, H), 0), Image.new("L", (W, H), 255)
    return ([Image.fromarray(f) for f in context] + [fill] * (n_frames - k),
            [known] * k + [todo] * (n_frames - k))


class ContinuationAdapter(Adapter):
    """Pipeline-only track A generator: input window, context bounds and splicing; no internals (stage 4 does not
    run these models). Subclasses implement ``load`` and ``_continue``."""

    generator = True

    def __init__(self, cfg, pipe, device):
        super().__init__(cfg)
        self.pipe, self.device = pipe, device
        off = cfg["official"]
        self.H, self.W = int(off["height"]), int(off["width"])
        self.n_frames, self.fps = int(off["num_frames"]), int(off["fps"])
        self.token_grid = token_spec(cfg)
        self.negative_prompt = off.get("negative_prompt", "")
        self._prompt_text = cfg.get("prompt", "")

    def set_prompt(self, text: str):
        self._prompt_text = text

    def input_frames(self, labels: dict) -> list[int]:
        return model_frames(self.cfg, labels)

    def context_frames(self, labels: dict) -> int:
        return context_end(self.cfg, labels)

    def to_tensor(self, frames_uint8: np.ndarray) -> np.ndarray:
        return fit_frames(np.asarray(frames_uint8), (self.H, self.W), self.token_grid.fit)

    @abstractmethod
    def _continue(self, context: np.ndarray, seed: int) -> np.ndarray:
        """uint8 context frames [k, H, W, 3] -> the pipeline's uint8 clip, starting at the first context frame."""

    def generate(self, video, context_end_frame: int, seed: int) -> np.ndarray:
        x = np.asarray(video)
        if x.shape[1:3] != (self.H, self.W):
            x = self.to_tensor(x)
        n = context_len(self.cfg, context_end_frame)
        s = context_end_frame - n
        return splice(x, self._continue(x[s:context_end_frame], seed), s, self.n_frames)


def to_uint8(frames01) -> np.ndarray:
    return (np.clip(np.asarray(frames01), 0, 1) * 255).round().astype(np.uint8)
