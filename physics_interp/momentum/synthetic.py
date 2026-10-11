"""Synthetic clips and a synthetic encoder, so every protocol runs without a GPU.

:func:`sample_scene` draws a pre-contact state with the generator's ranges and
builds labels in the same schema as ``momentum_dataset/generate_scene.py``
(the subset the protocols read). :func:`make_conservation_set`,
:func:`make_three_arm_set` and :func:`make_ladder_set` assemble the four live
sets' structure — honest/violating twins at pinned restitution, sub/anchor/super
arms, and 11-arm ladders — using the *same* edit functions the renderer uses.

:class:`SyntheticEncoder` maps a clip to pooled "activations" per site. It is a
caricature built to reproduce the register's qualitative pool dependence, not
a model of any real encoder: each object's pool sees that object's kinematics,
the frame-wide pool sees their *sum* (superposition, Part 2 §2), the pre pools
see pre-contact state and the post pools post-contact state, and depth trades
noise for a mild nonlinearity. Its only job is to make the scripts testable and
to give the user a smoke test that runs in seconds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from . import physics
from .captures import write_capture

NUM_FRAMES = 49
FRAME_RATE = 8
CONTACT_FRAME = 20
PRE_FRAME, POST_FRAME = 18, 22
RADIUS_RANGE = (0.30, 0.60)
MATERIALS = {"metal": 6.0, "rubber": 1.5}
SPEED_RANGE = (0.35, 1.55)

LADDER_LEGAL = (0.15, 0.2, 1 / 3, 0.5, 2 / 3, 0.8, 1.0)
LADDER_IMPOSSIBLE = (1.25, 1.5, 2.0, 3.0, 5.0)
ENERGY_LADDER_ARMS = (0.2, 1 / 3, 0.5, 2 / 3, 0.8, 1.0) + LADDER_IMPOSSIBLE   # 11 arms, anchor e = 1


def arm_tag(e: float) -> str:
    return f"e{e:.4g}"


def sample_scene(rng: np.random.Generator, restitution: float) -> dict:
    """A pre-contact state + honest post-contact state, as a labels dict."""
    r_a, r_b = rng.uniform(*RADIUS_RANGE, size=2)
    mat_a, mat_b = rng.choice(list(MATERIALS), size=2)
    m_a = MATERIALS[mat_a] * 4 / 3 * np.pi * r_a ** 3
    m_b = MATERIALS[mat_b] * 4 / 3 * np.pi * r_b ** 3
    speed_a = rng.uniform(*SPEED_RANGE)
    speed_b = rng.uniform(*SPEED_RANGE) if rng.uniform() < 0.7 else 0.0
    th_a, th_b = rng.uniform(0, 2 * np.pi, size=2)
    v_a = speed_a * np.array([np.cos(th_a), np.sin(th_a)])
    v_b = speed_b * np.array([np.cos(th_b), np.sin(th_b)])
    v_rel = v_a - v_b
    if np.linalg.norm(v_rel) < 1e-3:
        return sample_scene(rng, restitution)
    u = v_rel / np.linalg.norm(v_rel)
    w = physics.perp(u)
    b = rng.uniform(-0.9, 0.9) * (r_a + r_b)
    along = np.sqrt((r_a + r_b) ** 2 - b ** 2)
    n = (along * u + b * w) / (r_a + r_b)
    pre = physics.PreState(m_a, m_b, v_a, v_b, n)
    pre.check()
    contact = rng.uniform(-1.2, 1.2, size=2)
    pos_a_c, pos_b_c = contact - r_a * n, contact + r_b * n
    t_c = CONTACT_FRAME / FRAME_RATE
    pos_a0, pos_b0 = pos_a_c - v_a * t_c, pos_b_c - v_b * t_c
    labels = {
        "schema_version": 1, "seed": None, "restitution": float(restitution),
        "objects": {
            "A": {"radius": float(r_a), "material": str(mat_a), "mass": float(m_a)},
            "B": {"radius": float(r_b), "material": str(mat_b), "mass": float(m_b)},
        },
        "mass_ratio": float(m_a / m_b),
        "b_moving": bool(speed_b > 0),
        "collision": {
            "target_frame": CONTACT_FRAME, "actual_frame": CONTACT_FRAME + 0.05,
            "contact_normal_analytic": [float(n[0]), float(n[1]), 0.0],
            "pre": {"frame": PRE_FRAME, "velocity_A": [*map(float, v_a), 0.0],
                    "velocity_B": [*map(float, v_b), 0.0]},
        },
        "_start": {"A": pos_a0.tolist(), "B": pos_b0.tolist()},
    }
    post_a, post_b = pre.v_a + physics.legal_outcome(pre, restitution)[:2], \
        pre.v_b + physics.legal_outcome(pre, restitution)[2:]
    return finish_labels(labels, pre, post_a, post_b, arm="honest", violates=False,
                         restitution_realised=restitution, edit=None)


def finish_labels(labels: dict, pre: physics.PreState, post_a, post_b, *, arm: str,
                  violates: bool, restitution_realised: float, edit: dict | None) -> dict:
    """Fill the post-contact block, per-frame positions and the register's extra fields."""
    labels = {**labels, "collision": {**labels["collision"]}}
    post_a, post_b = np.asarray(post_a, float), np.asarray(post_b, float)
    p_pre = pre.m_a * pre.v_a + pre.m_b * pre.v_b
    p_post = pre.m_a * post_a + pre.m_b * post_b
    ke_pre = 0.5 * pre.m_a * pre.v_a @ pre.v_a + 0.5 * pre.m_b * pre.v_b @ pre.v_b
    ke_post = 0.5 * pre.m_a * post_a @ post_a + 0.5 * pre.m_b * post_b @ post_b
    labels["collision"].update({
        "post": {"frame": POST_FRAME, "velocity_A": [*map(float, post_a), 0.0],
                 "velocity_B": [*map(float, post_b), 0.0]},
        "momentum_rel_error": float(np.linalg.norm(p_post - p_pre) / np.linalg.norm(p_pre)),
        "kinetic_energy_ratio": float(ke_post / ke_pre),
        "restitution_measured": float(physics.implied_restitution(
            np.concatenate([post_a - pre.v_a, post_b - pre.v_b]), pre)),
    })
    labels["collision"]["pre"].update({"total_momentum": [*map(float, p_pre), 0.0],
                                       "kinetic_energy": float(ke_pre)})
    labels["collision"]["post"].update({"total_momentum": [*map(float, p_post), 0.0],
                                        "kinetic_energy": float(ke_post)})
    labels["violates_momentum"] = bool(violates)
    labels["momentum_conserved"] = not violates
    labels["arm"] = arm
    labels["restitution_realised"] = float(restitution_realised)
    labels["edit"] = edit
    # per-frame positions: straight lines, kink at the contact frame
    frames = []
    t_c = CONTACT_FRAME / FRAME_RATE
    for f in range(NUM_FRAMES):
        t = f / FRAME_RATE
        entry = {"frame": f}
        for tag, v0, v1 in (("A", pre.v_a, post_a), ("B", pre.v_b, post_b)):
            p0 = np.asarray(labels["_start"][tag])
            pos = p0 + v0 * t if t <= t_c else p0 + v0 * t_c + v1 * (t - t_c)
            entry[tag] = {"position": [*map(float, pos), 0.0],
                          "velocity": [*map(float, v0 if t <= t_c else v1), 0.0]}
        frames.append(entry)
    labels["per_frame"] = frames
    return labels


def _honest_post(labels):
    col = labels["collision"]
    return np.asarray(col["post"]["velocity_A"][:2]), np.asarray(col["post"]["velocity_B"][:2])


def violating_twin(honest: dict, rng: np.random.Generator,
                   split_range=(0.10, 0.35)) -> dict:
    """The split edit applied to an honest clip (Part 1, "How a violation is made")."""
    pre = physics.pre_state_from_labels(honest)
    pa, pb = _honest_post(honest)
    ds = rng.uniform(*split_range) * rng.choice([-1.0, 1.0])
    pa2, pb2, rec = physics.split_edit(pre, pa, pb, ds)
    return finish_labels(honest, pre, pa2, pb2, arm="violating", violates=True,
                         restitution_realised=honest["restitution_realised"],
                         edit={"kind": "split", **rec})


def restitution_arm(honest: dict, e_new: float) -> dict:
    pre = physics.pre_state_from_labels(honest)
    pa, pb = _honest_post(honest)
    pa2, pb2, rec = physics.restitution_edit(pre, pa, pb, honest["restitution_realised"], e_new)
    return finish_labels(honest, pre, pa2, pb2, arm=arm_tag(e_new), violates=False,
                         restitution_realised=e_new, edit={"kind": "restitution", **rec})


def make_conservation_set(n_seeds: int = 150, seed0: int = 101, rng_seed: int = 0) -> list[dict]:
    """75 elastic + 75 inelastic seeds, each an honest clip and its violating twin."""
    rng = np.random.default_rng(rng_seed)
    clips = []
    for i in range(n_seeds):
        e = 1.0 if i % 2 == 0 else 0.15
        honest = sample_scene(rng, e)
        honest["seed"] = seed0 + i
        honest["seed_group"] = seed0 + i
        clips.append(honest)
        clips.append(violating_twin(honest, rng))
    return clips


def make_three_arm_set(n_seeds: int = 150, seed0: int = 101, rng_seed: int = 0,
                       sub: float = -0.2, sup: float = 0.3) -> list[dict]:
    """energy_voe_450's structure: anchor e plus sub (e + sub) and super (e + sup) arms."""
    rng = np.random.default_rng(rng_seed)
    clips = []
    for i in range(n_seeds):
        e = float(rng.uniform(0.25, 0.7))
        honest = sample_scene(rng, e)
        honest["seed"] = honest["seed_group"] = seed0 + i
        honest["arm"] = "anchor"
        clips += [honest, restitution_arm(honest, e + sub), restitution_arm(honest, e + sup)]
        clips[-2]["arm"], clips[-1]["arm"] = "sub", "super"
    return clips


def make_ladder_set(n_seeds: int = 20, seed0: int = 9500, rng_seed: int = 0,
                    arms: Sequence[float] = ENERGY_LADDER_ARMS, anchor_e: float = 1.0,
                    with_violating: bool = False, pinned: Sequence[float] | None = None) -> list[dict]:
    """energy_ladder_v1 (anchor e = 1, 11 arms) or, with ``pinned``/``with_violating``,
    scene_ladder_v1 (every conservation seed's own ladder plus its violating twin)."""
    rng = np.random.default_rng(rng_seed)
    clips = []
    for i in range(n_seeds):
        e_sim = anchor_e if pinned is None else pinned[i % len(pinned)]
        honest = sample_scene(rng, e_sim)
        honest["seed"] = honest["seed_group"] = seed0 + i
        for e in arms:
            if abs(e - e_sim) < 1e-9:
                clips.append({**honest, "arm": "honest"})
            else:
                clips.append(restitution_arm(honest, e))
        if with_violating:
            clips.append(violating_twin(honest, rng))
    return clips


# --------------------------------------------------------------------------- #
# the synthetic encoder
# --------------------------------------------------------------------------- #

POOLS = [f"{r}.{p}" for r in ("A", "B", "all") for p in ("pre", "post", "all")]


class SyntheticEncoder:
    """Pooled fake activations with the register's pool algebra baked in.

    Per object and phase a 12-feature kinematic descriptor (positions at a few
    frames, velocity, radius, material); each pool is a random linear map of
    its descriptor passed through ``tanh`` at a depth-dependent gain, plus
    isotropic noise that shrinks with depth. The frame-wide pool is fed the
    *sum* of the two objects' descriptors.
    """

    def __init__(self, n_blocks: int = 6, dim: int = 96, seed: int = 0, noise0: float = 1.5,
                 noise1: float = 0.15):
        self.rng = np.random.default_rng(seed)
        self.dim = dim
        self.sites = ["embed.out"] + [f"block.{i}.resid_post" for i in range(n_blocks)]
        self.W = {s: self.rng.normal(size=(12, dim)) / np.sqrt(12) for s in self.sites}
        self.noise = np.linspace(noise0, noise1, len(self.sites))
        self.gain = np.linspace(0.3, 1.0, len(self.sites))

    @staticmethod
    def descriptor(labels: dict, tag: str, phase: str) -> np.ndarray:
        frames = labels["per_frame"]
        if phase == "pre":
            fs = (2, 10, 18)
        elif phase == "post":
            fs = (24, 32, 40)
        else:
            fs = (2, 18, 40)
        pos = np.concatenate([frames[f][tag]["position"][:2] for f in fs])
        vel = np.asarray(frames[fs[1]][tag]["velocity"][:2])
        obj = labels["objects"][tag]
        return np.concatenate([pos, vel, [obj["radius"], 1.0 if obj["material"] == "metal" else 0.0,
                                          obj["radius"] ** 3, 1.0]])

    def encode(self, labels: dict, clip_rng: np.random.Generator) -> dict[str, np.ndarray]:
        out = {}
        for k, site in enumerate(self.sites):
            rows = []
            for pool in POOLS:
                region, phase = pool.split(".")
                if region == "all":
                    d = self.descriptor(labels, "A", phase) + self.descriptor(labels, "B", phase)
                else:
                    d = self.descriptor(labels, region, phase)
                h = np.tanh(self.gain[k] * (d @ self.W[site]))
                rows.append(h + self.noise[k] * clip_rng.normal(size=self.dim))
            out[site] = np.stack(rows)
        return out


def write_synthetic_captures(clips: list[dict], root: Path, *, encoder: SyntheticEncoder | None = None,
                             model: str = "synthetic", seed: int = 0) -> Path:
    """Encode every clip and write the capture directory ``root``."""
    root = Path(root)
    enc = encoder or SyntheticEncoder()
    rng = np.random.default_rng(seed)
    for i, labels in enumerate(clips):
        clip_id = f"seed{labels['seed_group']:05d}_{labels['arm']}"
        acts = enc.encode(labels, rng)
        meta = {"labels": labels, "pools": POOLS, "sites": enc.sites, "model": model,
                "frames": list(range(NUM_FRAMES)), "phase_split_frame": CONTACT_FRAME,
                "truncate": None, "clip_id": clip_id}
        write_capture(root / f"{clip_id}.npz", acts, meta)
    return root
