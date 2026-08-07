"""Video and frame IO, plus the pixel -> latent-grid mask machinery.

Two directions live here:

* **Output** (``save_video`` / ``save_frames`` / ``write_json``) — writing what a
  generation produced.
* **Input** (``load_frames`` / ``load_segmentation`` / ``masks_to_latent_grid``)
  — reading a *rendered* clip back in so it can be pushed through the model's
  encoder and inverted. This is the direction the probing work needs, and it is
  the reason this module knows about the latent grid at all: an activation is
  indexed by latent cell, a ground-truth segmentation mask is indexed by pixel,
  and something has to reconcile the two.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

__all__ = [
    "save_video",
    "save_frames",
    "write_json",
    "load_frames",
    "load_segmentation",
    "pixel_frame_to_latent",
    "masks_to_latent_grid",
]


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #


def save_video(frames: Sequence[Any], path: str | Path, fps: int = 8) -> Path:
    """Write a list of PIL frames to an mp4.

    Uses imageio's bundled ffmpeg, so no system ffmpeg install is required —
    which matters because this has to run unattended on someone else's box.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    import imageio.v3 as iio

    arr = np.stack([np.asarray(f.convert("RGB")) for f in frames])
    # yuv420p + even dimensions keeps the file playable in browsers and QuickTime.
    h, w = arr.shape[1:3]
    if h % 2 or w % 2:
        arr = arr[:, : h - (h % 2), : w - (w % 2)]
    iio.imwrite(str(path), arr, fps=fps, codec="libx264", pixelformat="yuv420p")
    return path


def save_frames(frames: Sequence[Any], directory: str | Path, stem: str = "frame") -> Path:
    """Write frames as zero-padded PNGs, for frame-level analysis."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    width = max(4, len(str(len(frames) - 1)))
    for i, frame in enumerate(frames):
        frame.save(directory / f"{stem}_{i:0{width}d}.png")
    return directory


def write_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path


# --------------------------------------------------------------------------- #
# input
# --------------------------------------------------------------------------- #


def load_frames(scene_dir: str | Path, num_frames: int | None = None) -> np.ndarray:
    """Load a scene's RGB frames as ``uint8 [T, H, W, 3]``.

    Prefers ``frames/frame_XXXX.png`` over ``video.mp4``. The PNGs are the
    renderer's exact output; the mp4 is h264 at CRF 17. That is visually
    lossless but not numerically lossless, and every compression artifact
    becomes signal the probe could latch onto. Since the PNGs are already on
    disk there is no reason to inherit the codec's noise — the mp4 is a
    fallback for scenes shipped without frames.
    """
    scene_dir = Path(scene_dir)
    frame_paths = sorted((scene_dir / "frames").glob("frame_*.png"))

    if frame_paths:
        import imageio.v3 as iio

        frames = np.stack([np.asarray(iio.imread(p))[..., :3] for p in frame_paths])
    else:
        mp4 = scene_dir / "video.mp4"
        if not mp4.exists():
            raise FileNotFoundError(f"no frames/ and no video.mp4 in {scene_dir}")
        import imageio.v3 as iio

        frames = np.asarray(iio.imread(mp4, plugin="pyav"))[..., :3]

    if num_frames is not None and len(frames) != num_frames:
        raise ValueError(
            f"{scene_dir} has {len(frames)} frames, expected {num_frames}. "
            "A truncated clip would silently misalign every per-frame label."
        )
    return frames


def load_segmentation(scene_dir: str | Path) -> np.ndarray:
    """Load a scene's instance masks as ``uint8 [T, H, W]`` holding raw ids.

    Values are the generator's ``SEG_ID``: 0 floor, 1 sphere A, 2 sphere B.
    """
    scene_dir = Path(scene_dir)
    paths = sorted((scene_dir / "segmentation").glob("segmentation_*.png"))
    if not paths:
        raise FileNotFoundError(f"no segmentation/ in {scene_dir}")

    import imageio.v3 as iio

    masks = np.stack([np.asarray(iio.imread(p)) for p in paths])
    if masks.ndim == 4:  # some writers round-trip a trailing channel axis
        masks = masks[..., 0]
    return masks


def pixel_frame_to_latent(frame: int, temporal_compression: int = 4) -> int:
    """Latent frame index containing pixel frame ``frame``.

    CogVideoX's causal VAE encodes the first frame on its own and then every
    subsequent group of ``temporal_compression`` frames into one latent frame,
    which is why ``num_frames -> (num_frames - 1) // 4 + 1``. So the mapping is
    *not* a plain ``frame // 4``::

        pixel frame 0        -> latent 0
        pixel frames 1..4    -> latent 1
        pixel frames 17..20  -> latent 5
        pixel frames 21..24  -> latent 6

    That grouping is what makes the dataset's frame-20 contact convenient:
    contact is measured at 20.03-20.07, i.e. just *after* rendered frame 20,
    which is the last frame of latent 5. Latent frames 0-5 are therefore purely
    pre-collision and 6-12 purely post-collision, with no straddling cell.
    """
    if frame < 0:
        raise ValueError(f"frame must be non-negative, got {frame}")
    return 0 if frame == 0 else (frame - 1) // temporal_compression + 1


def masks_to_latent_grid(
    masks: np.ndarray,
    layout,
    *,
    ids: Sequence[int] = (1, 2),
    coverage: float = 0.0,
    temporal_compression: int = 4,
) -> np.ndarray:
    """Reduce pixel instance masks to per-latent-cell occupancy.

    Returns ``float32 [n_ids, frames, height, width]`` matching ``layout``'s
    latent grid, where each entry is the fraction of that cell's pixels covered
    by the object (or a bool-valued 0/1 array when ``coverage > 0``).

    Two separate downsamplings are involved and conflating them is an easy way
    to silently misalign every masked pooling:

    * **Space** — the VAE downsamples by 8 and the patch embedder by a further
      2, so one latent cell is a 16x16 pixel block. Occupancy is the mean over
      that block, which degrades gracefully for small objects instead of the
      all-or-nothing behaviour of nearest-neighbour sampling.
    * **Time** — the causal VAE's first-frame-alone grouping, via
      :func:`pixel_frame_to_latent`. A cell is occupied to the degree the
      object covers it in *any* contributing pixel frame, so a fast object
      leaves a short streak rather than vanishing between latent frames.

    ``coverage`` binarises the result at that threshold; the default of 0.0
    keeps soft occupancy, which makes a better pooling weight for a small
    object that never fully fills a 16x16 cell.
    """
    if masks.ndim != 3:
        raise ValueError(f"expected masks [T, H, W], got shape {masks.shape}")

    t_pix, h_pix, w_pix = masks.shape
    fy, fx = h_pix // layout.height, w_pix // layout.width
    if layout.height * fy != h_pix or layout.width * fx != w_pix:
        raise ValueError(
            f"{h_pix}x{w_pix} pixels do not tile the {layout.height}x{layout.width} "
            f"latent grid evenly. Masks and layout disagree about resolution."
        )

    out = np.zeros((len(ids), layout.frames, layout.height, layout.width), np.float32)
    counts = np.zeros(layout.frames, np.int64)

    for t in range(t_pix):
        lat_t = pixel_frame_to_latent(t, temporal_compression)
        if lat_t >= layout.frames:
            # More pixel frames than the layout accounts for: refuse rather than
            # fold the tail into the last latent frame, which would corrupt the
            # post-collision phase most of all.
            raise ValueError(
                f"pixel frame {t} maps to latent frame {lat_t}, beyond the "
                f"layout's {layout.frames}. Check num_frames vs the clip length."
            )
        counts[lat_t] += 1
        for k, seg_id in enumerate(ids):
            block = (masks[t] == seg_id).reshape(layout.height, fy, layout.width, fx)
            out[k, lat_t] += block.mean(axis=(1, 3))

    out /= np.maximum(counts, 1)[None, :, None, None]
    if coverage > 0:
        out = (out >= coverage).astype(np.float32)
    return out
