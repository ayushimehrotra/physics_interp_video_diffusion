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


def law_inputs(labels: dict, dt: float | None = None) -> tuple[dict, dict, dict]:
    """``(pre, post, ctx)`` for ``physlaw.eval.laws.<law>.residual(pre, post, ctx)``, from any labels file.

    pre / post: {"objects": [names], "p": [n, 3], "v": [n, 3], "w": [n, 3], "t_s"} (world frame, z up).
      ``dt=None``: the instants just before and just after the event (labels-2 ``event_state``; for
      collisions and bounces). ``dt`` (s): pre at the event, post the labelled frame nearest to
      event + dt (gravity, rolling: post "dt after the branch"); ctx["dt"] is the exact gap.
      ``dt="bottom"`` (f8): post at the instant the block reaches the floor run-out (ramp energy).
      The frozen labels-1 F1 render has constant velocities around contact, so its frames serve.
    ctx: the eval laws' vocabulary -- g (vector [0, 0, -|g|]), g_mag, dt, m [n], r [n] (radius or
      half-extent), k [n] (I / (m r^2)), n (contact normal, collision families), theta, h, e (twin's
      restitution), e_range (0, 1), a_ref (= g_mag) -- plus every ``physlaw.constants`` entry under
      its own name (schema/labels.schema.json), event_time_s, family, subfamily, twin, legal,
      violated_law, violation_magnitude; f4 also ref = (pre, post) of the unboosted twin with the same e.
    """
    pl = labels["physlaw"]
    names = list(pl["objects"])
    fps = labels["scene_constants"]["frame_rate"]

    def as_state(st):
        return {"objects": names, "t_s": st["t_s"], **{k: np.asarray(st[k], dtype=np.float64) for k in ("p", "v", "w")}}

    def frame(f):
        fr = labels["per_frame"][f]
        return {"objects": names, "t_s": f / fps,
                **{k: np.array([fr[n].get(key, [0.0, 0.0, 0.0]) for n in names], dtype=np.float64)
                   for k, key in (("p", "position"), ("v", "velocity"), ("w", "angular_velocity"))}}

    k = pl["first_post_contact_frame"]
    if "event_state" in pl:
        pre, post = as_state(pl["event_state"]["pre"]), as_state(pl["event_state"]["post"])
        c = dict(pl["constants"])
        te = pl["event_time_s"]
    else:                                    # labels-1 (F1, 2026-11-05): constant velocity on both sides
        te = pl["contact_frame_actual"] / fps
        pre, post = frame(k - 1), frame(k)
        c = {"g": 0.0, "fps": fps, "e": pl["e_target"], "e_sim": pl["e_sim"],
             "contact_normal": list(pl["contact_normal"]) + [0.0],
             "masses": {n: pl["objects"][n]["mass"] for n in names},
             "sizes": {n: pl["objects"][n]["radius"] for n in names}}
        for st in (pre, post):               # positions at the contact instant
            st["p"] = st["p"] + st["v"] * (te - st["t_s"])
            st["t_s"] = te
        c["legal"] = pl["twin"].startswith("conserve")
        c["violated_law"] = {"mom_violate_e1": "momentum", "mom_violate_e015": "momentum",
                             "energy_create": "energy"}.get(pl["twin"])
    if dt == "bottom":                       # constant speed on the run-out: step back to the instant
        t_b = float(c["t_bottom_s"])
        post = frame(int(np.ceil(t_b * fps)))
        post["p"] = post["p"] - post["v"] * (post["t_s"] - t_b)
        post["t_s"] = t_b
    elif dt is not None:
        f = min(max(int(round((te + dt) * fps)), k), len(labels["per_frame"]) - 1)
        post = frame(f)
    g_mag = float(c.get("g", 0.0))
    ctx = dict(c)
    inertia = {n: pl["objects"][n].get("inertia") for n in names}
    ctx.update(g=np.array([0.0, 0.0, -g_mag]), g_mag=g_mag, dt=float(post["t_s"] - pre["t_s"]),
               m=np.array([c["masses"][n] for n in names]), r=np.array([c["sizes"][n] for n in names]),
               e_range=(0.0, 1.0), a_ref=g_mag, event_time_s=te, family=pl["family"], twin=pl["twin"],
               subfamily=pl.get("subfamily", ""))
    if all(v is not None for v in inertia.values()):
        ctx["k"] = np.array([inertia[n] / (c["masses"][n] * c["sizes"][n] ** 2) for n in names])
    if "contact_normal" in c:
        ctx["n"] = np.asarray(c["contact_normal"], dtype=np.float64)
        ctx["e_ref"] = c.get("e")
    if "ref_event_state" in pl:              # f4: the unboosted twin's (pre, post) for the Galilean law
        ctx["ref"] = (as_state(pl["ref_event_state"]["pre"]), as_state(pl["ref_event_state"]["post"]))
    for src in ("theta_rad", "theta1_rad"):
        if src in c:
            ctx.setdefault("theta", c[src])
    if "slope_angles_rad" in c:              # f8: this twin's path (first slope)
        ctx["theta"] = c["slope_angles_rad"][0]
    for key in ("legal", "violated_law", "violation_magnitude"):
        if key in pl:
            ctx[key] = pl[key]
    return pre, post, ctx
