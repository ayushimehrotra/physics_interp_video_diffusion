"""Video and frame output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

__all__ = ["save_video", "save_frames", "write_json"]


def save_video(frames: Sequence[Any], path: str | Path, fps: int = 8) -> Path:
    """Write a list of PIL frames to an mp4.

    Uses imageio's bundled ffmpeg, so no system ffmpeg install is required —
    which matters because this has to run unattended on someone else's box.
    """
    import numpy as np

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
