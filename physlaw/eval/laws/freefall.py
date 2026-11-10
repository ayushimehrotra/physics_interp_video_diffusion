"""Gravity-driven laws (F7 free fall / projectiles / bounces / drop pairs, F9 pendulum).

``ctx["g"]`` is the gravity vector (same dimension as v; e.g. [0, 0, -9.81]); ``ctx["dt"]`` the time
from the branch (``pre``) to the ``post`` state. "Vertical" means along g.
"""
from __future__ import annotations

import numpy as np

from physlaw.eval.laws import Law, arr, safe_ratio
from physlaw.eval.laws.collision import pass_through


def _unique(post: dict, legal: dict, null: dict, keys=("v",), scale: dict | None = None) -> float:
    """Euclidean distance of ``post`` to the single legal outcome over ``keys`` (each divided by
    ``scale[key]``), relative to the null's distance."""
    scale = scale or {}
    dist = lambda x: np.sqrt(sum((((arr(x[k]) - arr(legal[k])) / scale.get(k, 1.0)) ** 2).sum() for k in keys))
    return safe_ratio(dist(post), dist(null))


def _vertical(g):
    g = arr(g)
    return g / np.linalg.norm(g)


# --------------------------------------------------------------------------- constant g (parabola)

def _parabola(pre, ctx) -> dict:
    g, dt = arr(ctx["g"]), float(ctx["dt"])
    out = {"v": arr(pre["v"]) + g * dt}
    if "p" in pre:
        out["p"] = arr(pre["p"]) + arr(pre["v"]) * dt + 0.5 * g * dt ** 2
    return out


def _no_gravity(pre, ctx) -> dict:
    """§5.2 null for projectiles and drops: constant velocity after the branch."""
    dt = float(ctx["dt"])
    out = {"v": arr(pre["v"]).copy()}
    if "p" in pre:
        out["p"] = arr(pre["p"]) + arr(pre["v"]) * dt
    return out


def _gravity_family(post, pre, ctx) -> tuple[float, None]:
    """Distance to the standard-g parabola after ``dt``: velocities, plus positions (divided by dt,
    so in velocity units) when both states carry ``p``; relative to the no-gravity null's distance
    (|g| dt for velocities). The family is a single member: no hidden parameter."""
    keys = ("v", "p") if "p" in pre and "p" in post else ("v",)
    return _unique(post, _parabola(pre, ctx), _no_gravity(pre, ctx), keys, {"p": float(ctx["dt"])}), None


gravity = Law("gravity", lambda pre, post, ctx: _gravity_family(post, pre, ctx)[0], _no_gravity, _gravity_family,
              "constant standard g after the branch (parabola)")


def _axis_residual(pre, post, ctx) -> float:
    """|v_h' - v_h| / |v_h| over all objects, v_h the component perpendicular to g: the horizontal
    motion must not change at the branch (axis independence)."""
    z = _vertical(ctx["g"])
    h = lambda v: arr(v) - (arr(v) @ z)[..., None] * z
    return safe_ratio(np.linalg.norm(h(post["v"]) - h(pre["v"])), np.linalg.norm(h(pre["v"])))


axis_independence = Law("axis_independence", _axis_residual, _no_gravity, None,
                        "horizontal velocity independent of the vertical motion")


def _mass_residual(pre, post, ctx) -> float:
    """RMS over object pairs of |a_i - a_j| / a_ref with a_i = (v_i' - v_i) / dt (for two objects
    |a_A - a_B| / a_ref); a_ref = ctx["a_ref"] or |g|. Objects of different mass must accelerate
    equally (drop pairs; F6 slides with a_ref = mu g)."""
    a = (arr(post["v"]) - arr(pre["v"])) / float(ctx["dt"])
    spread = np.sqrt(2.0 / (len(a) - 1) * ((a - a.mean(0)) ** 2).sum())        # RMS pairwise difference
    return safe_ratio(spread, float(ctx.get("a_ref", np.linalg.norm(arr(ctx.get("g", 0.0))))))


mass_independence = Law("mass_independence", _mass_residual, _no_gravity, None,
                        "acceleration independent of mass")


# --------------------------------------------------------------------------- floor bounce

def _bounce_legal(pre, ctx, e) -> np.ndarray:
    z = _vertical(ctx["g"])
    v = arr(pre["v"])
    vz = (v @ z)[..., None] * z
    return v - vz - e * vz


def _bounce_family(post, pre, ctx) -> tuple[float, float]:
    """Legal family: horizontal velocity kept, vertical reversed and scaled by e_floor in e_range.
    Euclidean velocity distance to the nearest member, relative to the no-bounce null (keeps falling:
    distance (1 + e_lo) |v_z|). Returns (d, e_hat); e_hat clamped to e_range. The rebound height
    ratio is e_hat^2 (``height_ratio``)."""
    e_lo, e_hi = ctx.get("e_range", (0.0, 1.0))
    z = _vertical(ctx["g"])
    v0, v1 = arr(pre["v"]), arr(post["v"])
    e_hat = float(np.clip(-(v1 @ z).sum() / (v0 @ z).sum(), e_lo, e_hi))
    d = np.linalg.norm(v1 - _bounce_legal(pre, ctx, e_hat))
    return safe_ratio(d, np.linalg.norm(arr(pre["v"]) - _bounce_legal(pre, ctx, e_lo))), e_hat


def height_ratio(pre: dict, post: dict, ctx: dict) -> float:
    """Rebound height / drop height = (v_z' / v_z)^2 (e^2 for a legal bounce; > 1 creates energy)."""
    z = _vertical(ctx["g"])
    return float(((arr(post["v"]) @ z) ** 2).sum() / ((arr(pre["v"]) @ z) ** 2).sum())


bounce = Law("bounce", lambda pre, post, ctx: _bounce_family(post, pre, ctx)[0],
             pass_through, _bounce_family,
             "floor bounce: rebound at e_floor in [0, 1], height ratio e^2, no energy creation")


# --------------------------------------------------------------------------- pendulum (small angle)

def _pendulum_legal(pre, ctx) -> dict:
    w0 = np.sqrt(float(ctx["g_mag"]) / float(ctx["L"]))
    th, w, t = arr(pre["p"]), arr(pre["w"]), float(ctx["dt"])
    return {"p": th * np.cos(w0 * t) + w / w0 * np.sin(w0 * t), "w": -th * w0 * np.sin(w0 * t) + w * np.cos(w0 * t)}


def _frozen(pre, ctx) -> dict:
    """§5.2 null for the pendulum: the bob stays where it is."""
    return {"p": arr(pre["p"]).copy(), "w": np.zeros_like(arr(pre["w"]))}


def _pendulum_family(post, pre, ctx) -> tuple[float, None]:
    """State after dt vs the small-angle solution with period 2 pi sqrt(L / g) (ctx L, g_mag, dt).
    ``p`` = angle (rad), ``w`` = angular velocity; metric on (angle, w / omega0), relative to the
    frozen-bob null. Unique family (period independent of mass and amplitude)."""
    w0 = np.sqrt(float(ctx["g_mag"]) / float(ctx["L"]))
    return _unique(post, _pendulum_legal(pre, ctx), _frozen(pre, ctx), ("p", "w"), {"w": w0}), None


pendulum = Law("pendulum", lambda pre, post, ctx: _pendulum_family(post, pre, ctx)[0], _frozen, _pendulum_family,
               "small-angle period 2 pi sqrt(L/g), independent of mass")
