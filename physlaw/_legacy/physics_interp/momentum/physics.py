"""Two-body collision algebra shared by every momentum protocol.

Conventions
-----------
* Everything is planar. Labels carry 3-vectors with ``z == 0`` exactly; this
  module drops ``z`` on the way in and never puts it back.
* ``n`` is the unit contact normal **from A's centre to B's centre** at contact
  (the generator's ``contact_normal_analytic``). With this sign the approach
  speed ``(v_A - v_B) . n`` is positive for every real collision.
* The *outcome* of a collision is the 4-vector ``o = (dv_A, dv_B)`` of velocity
  changes, ``dv = v_post - v_pre``. The outcome space is R^4 with the plain
  Euclidean metric; every geometric statement in the register (the legal line,
  "89% off the line", the ``|m_A - m_B| / (sqrt2 sqrt(m_A^2 + m_B^2))`` closed
  form) is a statement in this space with this metric.

The valid-collision line (register Part 2 §4)
----------------------------------------------
Fix the pre-contact state. A frictionless collision transfers an impulse ``J``
along ``n``::

    dv_A = -(J / m_A) n,    dv_B = +(J / m_B) n,    J = (1 + e) mu (v_rel . n)

with ``mu = m_A m_B / (m_A + m_B)`` and ``v_rel = v_A - v_B``. So every legal
outcome is a multiple of one direction ``b_normal = (-n/m_A, +n/m_B)``: the
legal family is a line through the origin of outcome space, parameterised by
``e``, and ``e in [0, 1]`` is a segment of it. The three ways to step off it
each have a name:

    b_tangential = (-t/m_A, +t/m_B)   an impulse along ``t``, the in-plane
                                      perpendicular to ``n`` — needs friction
    b_drift_n    = (n, n)             both balls shifted together along ``n``
    b_drift_t    = (t, t)             ... and along ``t``

The two drift directions are exactly the directions that change total
momentum: ``dp = m_A dv_A + m_B dv_B = (m_A + m_B) * drift``. The impulse
directions change it by zero. The basis is *not* orthogonal (``b_normal`` and
``b_drift_n`` have cosine ``(m_A - m_B) / (sqrt2 sqrt(m_A^2 + m_B^2))``), which is
why a common drift is "already 45% along the legal line" (§57h), and why the
decomposition below solves a 4x4 system rather than projecting.

The two dataset edits
---------------------
* :func:`split_edit` — the conservation set's violation. Write the two balls'
  speed changes along ``n`` as ``a = |dv_A . n|`` and ``b = |dv_B . n|``. The
  total ``a + b`` is fixed by restitution; the split ``b / a = m_A / m_B`` is
  fixed by conservation. The edit keeps ``a + b`` and moves the split, which is
  the same as adding a common drift ``-delta * n`` to both post velocities.
  Separation speed, and therefore realised restitution, is untouched.
* :func:`restitution_edit` — the ladder sets' arms. Rescale both deflections by
  ``(1 + e_new) / (1 + e_sim)``. Because ``J`` is linear in ``(1 + e)`` this is
  exact: the edited outcome is the legal outcome at ``e_new``, momentum is
  conserved to round-off, and ``e_new > 1`` is an impossible, energy-creating
  arm. ``e_new == e_sim`` is the identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

__all__ = [
    "PreState",
    "perp",
    "legal_outcome",
    "legal_direction",
    "outcome_basis",
    "decompose",
    "implied_restitution",
    "delta_p",
    "distance_to_legal_line",
    "membership_score",
    "drift_legal_cosine",
    "split_edit",
    "restitution_edit",
    "pre_state_from_labels",
    "outcome_from_labels",
    "clip_restitution",
    "clip_seed",
    "clip_arm",
]


def _xy(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return v[:2].copy()


def perp(n: np.ndarray) -> np.ndarray:
    """Left perpendicular of a planar unit vector."""
    return np.array([-n[1], n[0]], dtype=np.float64)


@dataclass(frozen=True)
class PreState:
    """Everything fixed before contact."""

    m_a: float
    m_b: float
    v_a: np.ndarray   # [2]
    v_b: np.ndarray   # [2]
    n: np.ndarray     # [2], unit, A -> B

    @property
    def mu(self) -> float:
        return self.m_a * self.m_b / (self.m_a + self.m_b)

    @property
    def approach_speed(self) -> float:
        return float(np.dot(self.v_a - self.v_b, self.n))

    def check(self) -> None:
        if abs(np.linalg.norm(self.n) - 1.0) > 1e-6:
            raise ValueError(f"contact normal is not unit length: |n| = {np.linalg.norm(self.n)}")
        if self.approach_speed <= 0:
            raise ValueError(
                f"approach speed {self.approach_speed:.3g} <= 0: with n pointing A -> B the "
                "balls must be closing. Check the sign convention of the normal."
            )


# --------------------------------------------------------------------------- #
# the legal line and its complement
# --------------------------------------------------------------------------- #


def legal_direction(pre: PreState) -> np.ndarray:
    """``b_normal = (-n/m_A, +n/m_B)`` — the one legal direction (unnormalised)."""
    return np.concatenate([-pre.n / pre.m_a, pre.n / pre.m_b])


def legal_outcome(pre: PreState, e: float) -> np.ndarray:
    """The legal outcome ``(dv_A, dv_B)`` at restitution ``e``."""
    J = (1.0 + e) * pre.mu * pre.approach_speed
    return J * legal_direction(pre)


def outcome_basis(pre: PreState) -> np.ndarray:
    """Columns ``[b_normal, b_tangential, b_drift_n, b_drift_t]`` as a ``[4, 4]`` matrix.

    ``b_normal`` and ``b_tangential`` are in *impulse* units (a coefficient of 1
    is an impulse of one momentum unit); the drifts are in velocity units.
    """
    t = perp(pre.n)
    return np.stack([
        np.concatenate([-pre.n / pre.m_a, pre.n / pre.m_b]),
        np.concatenate([-t / pre.m_a, t / pre.m_b]),
        np.concatenate([pre.n, pre.n]),
        np.concatenate([t, t]),
    ], axis=1)


def decompose(outcome: np.ndarray, pre: PreState) -> dict[str, float]:
    """Express an outcome in the named basis.

    Returns the impulse coefficients ``J_normal``, ``J_tangential`` (momentum
    units), the drift velocity ``(drift_n, drift_t)``, the total momentum change
    ``dp`` it implies, and the two register fractions:

    * ``tangential_fraction`` — ``|J_t| / sqrt(J_n^2 + J_t^2)``, the share of the
      impulse that smooth spheres forbid (§47).
    * ``relative_drift`` — ``|dp|`` divided by the magnitude of the impulse
      actually transferred, ``sqrt(J_n^2 + J_t^2)``. A legal outcome scores 0;
      a pure drift scores inf.
    """
    o = np.asarray(outcome, dtype=np.float64).reshape(4)
    coef = np.linalg.solve(outcome_basis(pre), o)
    J_n, J_t, d_n, d_t = (float(c) for c in coef)
    dp = (pre.m_a + pre.m_b) * (d_n * pre.n + d_t * perp(pre.n))
    J_mag = float(np.hypot(J_n, J_t))
    return {
        "J_normal": J_n,
        "J_tangential": J_t,
        "drift_n": d_n,
        "drift_t": d_t,
        "dp": dp,
        "dp_norm": float(np.linalg.norm(dp)),
        "tangential_fraction": float(abs(J_t) / J_mag) if J_mag > 0 else float("nan"),
        "relative_drift": float(np.linalg.norm(dp) / J_mag) if J_mag > 0 else float("inf"),
    }


def implied_restitution(outcome: np.ndarray, pre: PreState) -> float:
    """Restitution an outcome implies: ``-(v_rel_post . n) / (v_rel_pre . n)``."""
    o = np.asarray(outcome, dtype=np.float64).reshape(4)
    rel_post = pre.v_a + o[:2] - pre.v_b - o[2:]
    return float(-np.dot(rel_post, pre.n) / pre.approach_speed)


def delta_p(outcome: np.ndarray, pre: PreState) -> np.ndarray:
    """Total momentum change ``m_A dv_A + m_B dv_B``; exactly zero on a legal outcome."""
    o = np.asarray(outcome, dtype=np.float64).reshape(4)
    return pre.m_a * o[:2] + pre.m_b * o[2:]


def distance_to_legal_line(outcome: np.ndarray, pre: PreState, *,
                           segment: tuple[float, float] | None = None) -> float:
    """Euclidean distance from an outcome to the legal line (or segment) in R^4.

    With ``segment=None`` the whole line is used, so uncertainty about ``e`` is
    projected away entirely (§49). With ``segment=(e_lo, e_hi)`` the nearest
    point is clamped to that range of restitutions.
    """
    o = np.asarray(outcome, dtype=np.float64).reshape(4)
    b = legal_direction(pre)
    bb = float(b @ b)
    J = float(o @ b) / bb
    if segment is not None:
        scale = pre.mu * pre.approach_speed
        J = float(np.clip(J, (1 + segment[0]) * scale, (1 + segment[1]) * scale))
    return float(np.linalg.norm(o - J * b))


def membership_score(outcome: np.ndarray, pre: PreState, **kw) -> float:
    """§49's statistic: distance to the legal line divided by the total change ``|o|``.

    Scale-free, label-free. 0 for any legal outcome; 1 for an outcome entirely
    orthogonal to the legal direction.
    """
    o = np.asarray(outcome, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(o))
    return distance_to_legal_line(o, pre, **kw) / norm if norm > 0 else float("nan")


def drift_legal_cosine(m_a: float, m_b: float) -> float:
    """``|cos|`` between a common drift along ``n`` and the legal direction.

    Closed form ``|m_A - m_B| / (sqrt2 sqrt(m_A^2 + m_B^2))`` (§57h): the physics
    baseline a membership statistic has to be read against.
    """
    return float(abs(m_a - m_b) / (np.sqrt(2.0) * np.hypot(m_a, m_b)))


# --------------------------------------------------------------------------- #
# the two edits
# --------------------------------------------------------------------------- #


def split_edit(pre: PreState, post_a: np.ndarray, post_b: np.ndarray,
               delta_split: float) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """The conservation set's violation: keep ``a + b``, move the split.

    ``post_a``/``post_b`` are the *honest* post-contact velocities. Let
    ``a = -(dv_A . n)`` and ``b = (dv_B . n)`` be the speed changes along ``n``
    (both positive for a real collision). The honest split fraction is
    ``s = b / (a + b) = m_A / (m_A + m_B)``. The edit sets ``s' = s + delta_split``
    with ``a' + b' = a + b``, i.e. ``b' = b - delta`` and ``a' = a + delta`` with
    ``delta = delta_split * (a + b)``. Both balls receive the same extra
    velocity ``-delta n``, so this is a common drift along the normal, and the
    separation speed — hence realised restitution — is unchanged.

    Returns the edited velocities and a record of what was done.
    """
    pa, pb = _xy(post_a), _xy(post_b)
    dv_a, dv_b = pa - pre.v_a, pb - pre.v_b
    a = float(-np.dot(dv_a, pre.n))
    b = float(np.dot(dv_b, pre.n))
    total = a + b
    if total <= 0:
        raise ValueError("post velocities do not describe a collision along n (a + b <= 0)")
    delta = delta_split * total
    shift = -delta * pre.n
    pa2, pb2 = pa + shift, pb + shift
    rec = {
        "a": a, "b": b, "total_kick": total,
        "split_honest": b / total, "split_edited": (b - delta) / total,
        "delta_split": float(delta_split), "delta_speed": float(delta),
        "dp": ((pre.m_a + pre.m_b) * shift).tolist(),
    }
    return pa2, pb2, rec


def restitution_edit(pre: PreState, post_a: np.ndarray, post_b: np.ndarray,
                     e_sim: float, e_new: float) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """A ladder arm: rescale both deflections by ``(1 + e_new) / (1 + e_sim)``.

    Exact because the impulse is linear in ``(1 + e)``. Momentum is conserved
    to round-off for every ``e_new``; kinetic energy follows the closed form
    ``dKE = -(1/2) mu (1 - e^2) (v_rel . n)^2``, which is negative for
    ``e < 1``, zero at ``e = 1`` and *positive* — impossible — for ``e > 1``.
    """
    pa, pb = _xy(post_a), _xy(post_b)
    factor = (1.0 + e_new) / (1.0 + e_sim)
    dv_a, dv_b = (pa - pre.v_a) * factor, (pb - pre.v_b) * factor
    pa2, pb2 = pre.v_a + dv_a, pre.v_b + dv_b
    u = pre.approach_speed
    rec = {
        "e_sim": float(e_sim), "e_new": float(e_new), "factor": float(factor),
        "dke_closed_form": float(-0.5 * pre.mu * (1.0 - e_new ** 2) * u ** 2),
        "impossible": bool(e_new > 1.0 + 1e-12),
    }
    return pa2, pb2, rec


# --------------------------------------------------------------------------- #
# label accessors
# --------------------------------------------------------------------------- #


def pre_state_from_labels(labels: Mapping[str, Any]) -> PreState:
    """Build the pre-contact state from a scene's ``labels.json``."""
    col = labels["collision"]
    n = _xy(col.get("contact_normal_analytic") or col["contact_normal"])
    n /= np.linalg.norm(n)
    pre = PreState(
        m_a=float(labels["objects"]["A"]["mass"]),
        m_b=float(labels["objects"]["B"]["mass"]),
        v_a=_xy(col["pre"]["velocity_A"]),
        v_b=_xy(col["pre"]["velocity_B"]),
        n=n,
    )
    if pre.approach_speed < 0:
        # The analytic normal always points A -> B along the approach; if a label
        # carries only the solver's event normal its sign is not guaranteed.
        pre = PreState(pre.m_a, pre.m_b, pre.v_a, pre.v_b, -n)
    pre.check()
    return pre


def outcome_from_labels(labels: Mapping[str, Any]) -> np.ndarray:
    """``(dv_A, dv_B)`` read from the pre/post velocities in the labels."""
    col = labels["collision"]
    return np.concatenate([
        _xy(col["post"]["velocity_A"]) - _xy(col["pre"]["velocity_A"]),
        _xy(col["post"]["velocity_B"]) - _xy(col["pre"]["velocity_B"]),
    ])


def clip_restitution(labels: Mapping[str, Any]) -> float:
    """The restitution a clip actually shows.

    The label trap (register, pipeline audit 2026-09-23): on edited sets
    ``labels["restitution"]`` is the value the *simulator* ran at, and an edited
    arm keeps it unchanged. The field that tracks the edit is
    ``restitution_realised``; this helper prefers it, falls back to the
    solver's ``restitution_measured`` for honest clips, and only then to the
    sampled value — and refuses to fall back when the clip is marked as edited.
    """
    if "restitution_realised" in labels and labels["restitution_realised"] is not None:
        return float(labels["restitution_realised"])
    edit = labels.get("edit") or {}
    if edit.get("kind") == "restitution":
        raise KeyError(
            f"clip {labels.get('scene_id')} is a restitution arm but has no "
            "restitution_realised field; labels['restitution'] would be the pre-edit value"
        )
    measured = labels.get("collision", {}).get("restitution_measured")
    if measured is not None:
        return float(measured)
    return float(labels["restitution"])


def clip_seed(labels: Mapping[str, Any]) -> int:
    """The physics seed: twins and ladder arms of one scene share it."""
    return int(labels.get("seed_group", labels["seed"]))


def clip_arm(labels: Mapping[str, Any]) -> str:
    """``'honest'``, ``'violating'`` or a restitution arm tag like ``'e1.25'``."""
    if "arm" in labels:
        return str(labels["arm"])
    return "violating" if labels.get("violates_momentum", False) else "honest"
