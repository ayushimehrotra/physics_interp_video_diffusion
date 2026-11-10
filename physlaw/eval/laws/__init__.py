"""Physical laws as residuals, legal families and nulls (spec §1, §2.3, §5.2). Pure numpy, torch-free:
the data, eval and outputs code all import it.

Every law is a :class:`Law` with the same interface::

    NAME: str
    residual(pre, post, ctx) -> float                  normalised violation; 0 = the law holds exactly
    family_distance(post, pre, ctx) -> (d, param)      laws with a legal family only (else None)
    null_outcome(pre, ctx) -> dict                     the §5.2 null (no-event outcome)

Use: ``from physlaw.eval.laws import momentum; momentum.residual(pre, post, ctx)``, or ``LAWS[name]``.
Laws that share their algebra share a module (collision.py, freefall.py, incline.py); each one is
exported here under its spec name.

States. ``pre`` / ``post`` are dicts of per-object arrays, objects in a fixed order (A, B, ...):
``v`` [n_obj, d] velocity, optionally ``p`` [n_obj, d] position and ``w`` angular velocity
(scalar per object in the plane: [n_obj] or [n_obj, 1]). ``pre`` is the state just before the
event, ``post`` the state just after it (collisions, bounce) or ``ctx["dt"]`` seconds after the
branch (gravity, ramp, pendulum, rolling). ``ctx`` holds what the scene fixes: masses ``m``,
radii ``r``, contact normal ``n`` (unit, A -> B) and point ``c``, gravity vector ``g``, ``h``,
``e_range`` and so on (see each law). :func:`from_labels` builds all three from a ``labels.json``.

Normalisation (one rule for every family distance). A distance is measured in the law's own
metric (stated per law) and divided by the distance of the law's §5.2 NULL to the legal family.
So d = 0 on the family, and d = 1 for the null by construction, for every law and every scene.
This makes S_family = 1 - (d_forecast - d_floor) / (d_null - d_floor) comparable across laws, with
d_null = 1. Residuals are scale-free ratios; each law's docstring gives its denominator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Law:
    NAME: str
    residual: Callable[[dict, dict, dict], float]
    null_outcome: Callable[[dict, dict], dict]
    family_distance: Callable[[dict, dict, dict], tuple[float, float | None]] | None = None
    doc: str = ""


def arr(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def cross2(a, b) -> np.ndarray:
    """z-component of a x b for planar vectors (works row-wise)."""
    a, b = arr(a), arr(b)
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


from physlaw.eval.laws.collision import angular_momentum, com, energy, galilean, momentum  # noqa: E402
from physlaw.eval.laws.freefall import axis_independence, bounce, gravity, mass_independence, pendulum  # noqa: E402
from physlaw.eval.laws.incline import ramp_energy, rolling  # noqa: E402

LAWS: dict[str, Law] = {l.NAME: l for l in (momentum, energy, com, angular_momentum, galilean, gravity,
                                             axis_independence, bounce, mass_independence, pendulum,
                                             ramp_energy, rolling)}


def from_labels(labels: dict) -> tuple[dict, dict, dict]:
    """(pre, post, ctx) of a collision clip (F1, F2, F3, F4) from its ``labels.json``.

    pre:  v = pre-contact velocities [2, 2], p = centres at the contact instant [2, 2]
    post: v = the twin's post-contact velocities (``physlaw.post_velocity``), p = the same centres
    ctx:  m, r, n (A -> B), c (contact point), e_range (0, 1), point (0, 0): the fixed point for
          angular momentum (the world origin on the floor's axis; any fixed point works, see
          ``angular_momentum``)."""
    pl = labels["physlaw"]
    fam = pl.get("family", "f1")
    if fam not in ("f1", "f2", "f3", "f4"):
        raise NotImplementedError(f"from_labels: family {fam} (collision families only so far)")
    m = arr([pl["objects"][o]["mass"] for o in "AB"])
    r = arr([pl["objects"][o]["radius"] for o in "AB"])
    n = arr(pl["contact_normal"])[:2]
    n = n / np.linalg.norm(n)
    c = arr(labels["collision"]["contact_point_world"])[:2]
    p = np.stack([c - r[0] * n, c + r[1] * n])
    pre = {"v": arr([pl["pre_velocity"][o][:2] for o in "AB"]), "p": p}
    post = {"v": arr([pl["post_velocity"][o][:2] for o in "AB"]), "p": p.copy()}
    ctx = {"m": m, "r": r, "n": n, "c": c, "e_range": (0.0, 1.0), "point": np.zeros(2)}
    return pre, post, ctx


__all__ = ["Law", "LAWS", "from_labels", *LAWS]
