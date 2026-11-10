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


def law_inputs(labels: dict) -> tuple[dict, dict, dict]:
    """``(pre, post, ctx)`` for ``physlaw.eval.laws.<law>.residual(pre, post, ctx)``, from any labels file.

    pre / post: {"objects": [names], "p": [n, 3], "v": [n, 3], "w": [n, 3], "t_s": float} at the last
    shared frame and the first post-event frame (labels-2 ``pre_state`` / ``post_state``; rebuilt from
    the per-frame block for the frozen labels-1 F1 render). ctx: ``physlaw.constants`` (g, fps, masses,
    sizes, and each family's constants; documented in schema/labels.schema.json) plus event_time_s,
    family, subfamily, twin, legal, violated_law, violation_magnitude.
    """
    pl = labels["physlaw"]
    names = list(pl["objects"])

    def state(st):
        return {"objects": st["objects"], "t_s": st["t_s"], "frame": st["frame"],
                **{k: np.asarray(st[k], dtype=np.float64) for k in ("p", "v", "w")}}

    if "pre_state" in pl:
        pre, post = state(pl["pre_state"]), state(pl["post_state"])
        ctx = dict(pl["constants"])
        ctx.update(event_time_s=pl["event_time_s"], subfamily=pl.get("subfamily", ""))
    else:                                    # labels-1 (F1, 2026-11-05): same quantities, older layout
        k = pl["first_post_contact_frame"]
        fps = labels["scene_constants"]["frame_rate"]

        def frame(f):
            fr = labels["per_frame"][f]
            return {"objects": names, "t_s": f / fps, "frame": f,
                    "p": np.array([fr[n]["position"] for n in names], dtype=np.float64),
                    "v": np.array([fr[n]["velocity"] for n in names], dtype=np.float64),
                    "w": np.array([fr[n].get("angular_velocity", [0.0, 0.0, 0.0]) for n in names], dtype=np.float64)}

        pre, post = frame(k - 1), frame(k)
        ctx = {"g": 0.0, "fps": fps, "e": pl["e_target"], "e_sim": pl["e_sim"],
               "contact_normal": list(pl["contact_normal"]) + [0.0], "boost_u": [0.0, 0.0, 0.0],
               "masses": {n: pl["objects"][n]["mass"] for n in names},
               "sizes": {n: pl["objects"][n]["radius"] for n in names},
               "event_time_s": pl["contact_frame_actual"] / fps, "subfamily": "",
               "legal": pl["twin"].startswith("conserve"),
               "violated_law": {"mom_violate_e1": "momentum", "mom_violate_e015": "momentum",
                                "energy_create": "energy"}.get(pl["twin"])}
    ctx.update(family=pl["family"], twin=pl["twin"])
    for key in ("legal", "violated_law", "violation_magnitude"):
        if key in pl:
            ctx[key] = pl[key]
    return pre, post, ctx
