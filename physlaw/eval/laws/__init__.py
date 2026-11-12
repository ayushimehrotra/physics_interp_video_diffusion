"""Physical laws as dimensionless statistics (spec §5.0, v3). Pure numpy, torch-free: the data, eval and
outputs code all import it.

Nine laws, each a :class:`Law`::

    NAME: str
    stats(pre, post, ctx) -> dict[str, float]      dimensionless statistics (NaN where undefined)
    legal: dict[str, (op, value)]                  per statistic: ("eq", v) the law fixes it at v,
                                                   ("le", v) the law bounds it above (one-sided)
    null_outcome(pre, ctx) -> dict                 the §5.2 no-event null (a post-like state)
    family_distance(post, pre, ctx) -> (d, param)  momentum only (S_family, secondary)

    momentum         (collision.py)  F1-F4, F6 glide   eta, antiparallel, normal_A, normal_B; inertia_speed, inertia_turn
    energy           (collision.py)  F1-F4, F7 bounce, F8 pad, F9 swing   epsilon (generalised restitution),
                                     rho_star = eps^2, creation
    gravity          (freefall.py)   F7                g_ratio, axis_ratio, mass_ratio
    friction         (freefall.py)   F6 slide / pair   decel_const, stop_hold, mass_ratio
    oscillation      (freefall.py)   F9 pair           period_ratio, mass_period_ratio
    rolling          (incline.py)    F10               order, slip_s, slip_h (|w| r / |v|)
    statics          (incline.py)    F11               tip_consistency, alpha_ratio
    galilean         (collision.py)  F4                d_epsilon, d_theta
    angular_momentum (collision.py)  F12               L_ratio

Every statistic is a ratio of one kinematic quantity measured twice in the same clip, so it is unchanged by a
length scale, a time scale (velocities / tau, accelerations / tau^2) and a common gain on the measured vectors
(``tests/test_laws.py`` checks all three). No statistic uses the scene's g or any metric constant: where a law needs
g it is measured in the clip (statics: ctx["g_clip"], a free fall in the same clip), and the swing epsilon compares
apex heights only. Exceptions to the gain invariance, documented per statistic: rolling slip and angular_momentum
L_ratio compare a measured rotation with a measured velocity, statics alpha_ratio an angular acceleration with a
measured g_clip / half-length, so a gain on the measured vectors alone is not common to both terms. Never score
KE'/KE (not boost-invariant; diluted by CoM motion).

States. ``pre`` / ``post`` are dicts of per-object arrays, objects in a fixed order (A, B, ...): ``v`` [n_obj, d]
velocity, optionally ``p`` positions and ``a`` accelerations [n_obj, d]. ``pre`` is the state just before the event,
``post`` just after it (or ``ctx["dt"]`` seconds later). ``ctx`` holds what the scene fixes: ``event`` (collision |
bounce | pad | swing, for :func:`epsilon`), masses ``m``, contact normal ``n`` (unit, A -> B), gravity vector ``g``,
inertia factors ``k``, ``dt``, ``force_free`` (F6 glide), ``overhang_x`` / ``aspect_ba`` / ``g_clip`` /
``half_length`` (F11), ``p_bottom`` (F9 swing), ``c`` (F12 contact point) ... (see each law); only the direction of
``g`` is ever used. ``physlaw.data.io.law_inputs`` builds all three from any labels file; :func:`from_labels` gives
the 2-D collision form.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Law:
    NAME: str
    stats: Callable[[dict, dict, dict], dict]
    legal: dict
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
    """num / den, NaN when den is zero or not finite (a statistic that the state cannot define)."""
    num, den = float(num), float(den)
    return num / den if den != 0 and np.isfinite(den) else float("nan")


S_FLOOR = -1.0     # the lowest score a forecast can get (L1 clips scores to [S_FLOOR, 1])


def ratio_score(t_fc, t_legal, t_null, floor: float | None = None):
    """S = 1 - |log t_fc - log t_legal| / |log t_null - log t_legal| (spec §5.0.4, gravity / rolling): 1 at the legal
    value, 0 at the null's, negative beyond. Vectorised; NaN wherever t_legal or t_null is non-positive or not finite,
    or t_null == t_legal. ``floor`` None: also NaN where t_fc is non-positive or not finite. ``floor`` given (L1, audit
    M2): such a t_fc has no logarithm and is the WORST outcome (e.g. a decoded deceleration ratio of the wrong sign),
    so it scores ``floor``, and every score is clipped below at ``floor``; dropping it as NaN would remove the worst
    forecasts from a median. For ``rolling.order`` (already a log ratio, legal 1, null 0) pass t = a_s / a_h instead,
    or use the equivalent 1 - |order - 1|."""
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = [np.log(np.where(np.asarray(x, np.float64) > 0, x, np.nan)) for x in (t_fc, t_legal, t_null)]
        den = np.abs(lg[2] - lg[1])
        den = np.where(den > 0, den, np.nan)
        s = 1.0 - np.abs(lg[0] - lg[1]) / den
        if floor is not None:
            bad = ~np.isfinite(lg[0]) & np.isfinite(den)        # t_fc <= 0, inf or NaN where the score is defined
            s = np.where(bad, floor, np.maximum(s, floor))
    return float(s) if np.ndim(s) == 0 else s


from physlaw.eval.laws.collision import angular_momentum, energy, epsilon, galilean, momentum, rho_star  # noqa: E402
from physlaw.eval.laws.collision import mass_blind_outcome, mass_free_momentum, momentum_exchange  # noqa: E402
from physlaw.eval.laws.collision import angular_momentum_z, rigid_impulse  # noqa: E402
from physlaw.eval.laws.freefall import friction, gravity, oscillation  # noqa: E402
from physlaw.eval.laws.incline import rolling, statics, tip_factor, tip_rate, tip_window  # noqa: E402

LAWS: dict[str, Law] = {l.NAME: l for l in (momentum, energy, gravity, friction, oscillation, rolling, statics, galilean,
                                            angular_momentum)}


def from_labels(labels: dict) -> tuple[dict, dict, dict]:
    """(pre, post, ctx) of a collision clip (F1-F4) from its ``labels.json``, in the floor plane (2-D).

    pre:  v = pre-contact velocities [2, 2] (``physlaw.pre_velocity``), p = centres at contact [2, 2]
    post: v = the twin's post-contact velocities (``physlaw.post_velocity``), p = the same centres
    ctx:  event "collision", m, r, n (A -> B), c (contact point), e_range (0, 1).
    With per-frame labels the centres and n come from ``io.law_inputs`` (centres extrapolated to the contact
    instant, n = (p_B - p_A) / |p_B - p_A|; the labelled ``contact_normal`` of F1 is up to 0.66 deg off);
    otherwise n is the labelled normal and the centres sit at c -/+ r n."""
    pl = labels["physlaw"]
    fam = pl.get("family", "f1")
    if fam not in ("f1", "f2", "f3", "f4"):
        raise ValueError(f"from_labels: family {fam} is not a collision family (f1-f4)")
    m = arr([pl["objects"][o]["mass"] for o in "AB"])
    r = arr([pl["objects"][o]["radius"] for o in "AB"])
    if "per_frame" in labels:
        from physlaw.data.io import law_inputs
        pre3, _, ctx3 = law_inputs(labels)
        p, n = pre3["p"][:, :2], ctx3["n"][:2] / np.linalg.norm(ctx3["n"][:2])
        c = p[0] + r[0] * n
    else:
        n = arr(pl["contact_normal"])[:2]
        n = n / np.linalg.norm(n)
        c = arr(labels.get("collision", {}).get("contact_point_world", [0.0, 0.0]))[:2]
        p = np.stack([c - r[0] * n, c + r[1] * n])
    pre = {"v": arr([pl["pre_velocity"][o][:2] for o in "AB"]), "p": p}
    post = {"v": arr([pl["post_velocity"][o][:2] for o in "AB"]), "p": p.copy()}
    ctx = {"event": "collision", "m": m, "r": r, "n": n, "c": c, "e_range": (0.0, 1.0)}
    return pre, post, ctx


__all__ = ["Law", "LAWS", "S_FLOOR", "from_labels", "epsilon", "rho_star", "ratio_score", "momentum_exchange",
           "mass_blind_outcome", "mass_free_momentum", "angular_momentum_z", "rigid_impulse", "tip_factor", "tip_rate", "tip_window", *LAWS]
