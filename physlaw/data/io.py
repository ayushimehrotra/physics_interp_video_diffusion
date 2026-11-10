"""Reading rendered clips (any venv with numpy + imageio[ffmpeg])."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Clip:
    path: Path
    labels: dict
    frames: np.ndarray | None = None     # uint8 [T, H, W, 3], master frames
    masks: np.ndarray | None = None      # uint8 [T, H, W], 0 floor, 1 A, 2 B

    @property
    def clip_id(self) -> str:
        pl = self.labels["physlaw"]
        return f"{pl['twin_group_id']}/{pl['twin']}"

    @property
    def pl(self) -> dict:
        return self.labels["physlaw"]


def load_clip(path: str | Path, frames: bool = True, masks: bool = True) -> Clip:
    path = Path(path)
    labels = json.loads((path / "labels.json").read_text())
    c = Clip(path, labels)
    if frames:
        import imageio.v3 as iio
        c.frames = iio.imread(path / "video.mp4", plugin="FFMPEG")
    if masks:
        c.masks = np.load(path / "masks.npz")["instance"]
    return c


def list_clips(data_dir: str | Path, twins: list[str] | None = None) -> list[Path]:
    """Every completed twin clip under a family dir, ordered by group then twin index."""
    out = []
    for g in sorted(Path(data_dir).glob("*/_GROUP_COMPLETE")):
        rec = json.loads(g.read_text())
        for t in rec["twins"]:
            if (twins is None or t in twins) and (g.parent / t / "_COMPLETE").exists():
                out.append(g.parent / t)
    return out
