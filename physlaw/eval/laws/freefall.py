"""Acceleration laws. Each statistic is a ratio of one kinematic quantity measured twice in the clip.

  gravity      F7 projectile / drop: constant g through the branch, horizontal motion independent of the vertical,
               acceleration independent of mass
  friction     F6 slide / pair: kinetic (Coulomb) friction gives a constant deceleration along the motion, independent
               of speed (decel_const; viscous drag a ~ v gives < 1), a stop that holds (stop_hold) and independence of
               mass (mass_ratio)
  oscillation  F9 pendulum: the period does not change within the clip (period_ratio) and does not depend on the
               bob's mass (mass_period_ratio)

``ctx["g"]`` is the gravity vector (only its direction is used; e.g. [0, 0, -9.81]). ``pre`` / ``post`` carry
accelerations ``a`` [n_obj, d] (``io.law_inputs`` fits them to the positions on each side of the event) and, for
the pendulum, measured periods ``T`` [n_obj].
"""
from __future__ import annotations

import numpy as np

from physlaw.eval.laws import Law, arr, safe_ratio


def _ghat(ctx, d: int) -> np.ndarray:
    g = arr(ctx["g"])[:d]
    return g / np.linalg.norm(g)


def _g_ratio(pre, post, ctx) -> float:
    if "a" not in pre or "a" not in post or "g" not in ctx:
        return float("nan")
    a0, a1 = arr(pre["a"]), arr(post["a"])
    z = _ghat(ctx, a0.shape[-1])
    return safe_ratio((a1 @ z).sum(), (a0 @ z).sum())


def _axis_ratio(pre, post, ctx) -> float:
    if "g" not in ctx:
        return float("nan")
    v0, v1 = arr(pre["v"]), arr(post["v"])
    z = _ghat(ctx, v0.shape[-1])
    h = lambda v: v - (v @ z)[..., None] * z
    return safe_ratio(np.linalg.norm(h(v1)), np.linalg.norm(h(v0)))


def _heavy_light(ctx) -> tuple[int, int] | None:
    if "m" not in ctx:
        return None
    m = arr(ctx["m"]).reshape(-1)
    if len(m) < 2 or m.max() == m.min():
        return None
    return int(np.argmax(m)), int(np.argmin(m))


def _mass_ratio(pre, post, ctx) -> float:
    """a_heavy / a_light after the event, components along g_hat. NaN without two distinct masses."""
    hl = _heavy_light(ctx)
    if "a" not in post or hl is None or "g" not in ctx:
        return float("nan")
    a = arr(post["a"])
    comp = a @ _ghat(ctx, a.shape[-1])
    return safe_ratio(comp[hl[0]], comp[hl[1]])


def _gravity_stats(pre, post, ctx) -> dict:
    """g_ratio = (a_post . g_hat) / (a_pre . g_hat) (summed over objects; NaN when the object was held before the
    event), axis_ratio = |v_h'| / |v_h| (horizontal = perpendicular to g, over all objects; NaN for a vertical drop),
    mass_ratio = a_heavy / a_light (see :func:`_mass_ratio`)."""
    return {"g_ratio": _g_ratio(pre, post, ctx), "axis_ratio": _axis_ratio(pre, post, ctx),
            "mass_ratio": _mass_ratio(pre, post, ctx)}


def _aristotelian(pre, ctx) -> dict:
    """§5.2 null, Aristotle's fall: no acceleration after the branch (a = 0: constant fall, g_ratio 0) and the
    horizontal (violent) motion ceases (v_h' = 0: the horizontal motion is NOT independent of the vertical,
    axis_ratio 0); mass_ratio has no kinematic null here (0 / 0; L1 substitutes Aristotle's a ~ m, m_heavy / m_light).
    A null that kept the horizontal velocity would sit at the legal axis_ratio 1 and leave that statistic unscorable."""
    v = arr(pre["v"])
    z = _ghat(ctx, v.shape[-1])
    v1 = (v @ z)[..., None] * z
    out = {"v": v1, "a": np.zeros_like(v)}
    if "p" in pre:
        out["p"] = arr(pre["p"]) + v1 * float(ctx.get("dt", 0.0))
    return out


gravity = Law("gravity", _gravity_stats, {"g_ratio": ("eq", 1.0), "axis_ratio": ("eq", 1.0), "mass_ratio": ("eq", 1.0)},
              _aristotelian, None, "constant g through the branch, axis independence, mass independence")


# --------------------------------------------------------------------------- friction (F6)

def along_motion(st: dict) -> np.ndarray:
    """Per object, the acceleration component along its own velocity, a . v_hat (NaN at rest or without a)."""
    if "a" not in st:
        return np.full(len(np.atleast_2d(arr(st["v"]))), np.nan)
    a, v = np.atleast_2d(arr(st["a"])), np.atleast_2d(arr(st["v"]))
    s = np.linalg.norm(v, axis=-1)
    return np.where(s > 0, (a * v).sum(-1) / np.where(s > 0, s, 1.0), np.nan)


def _speed_hold(pre, post) -> float:
    """max over objects of |v_stop| / |v_pre| (NaN without post["v_stop"] or a moving pre state)."""
    if "v_stop" not in post:
        return float("nan")
    vs, v0 = np.atleast_2d(arr(post["v_stop"])), np.atleast_2d(arr(pre["v"]))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.linalg.norm(vs, axis=-1) / np.linalg.norm(v0, axis=-1)
    r = r[np.isfinite(r)]
    return float(r.max()) if len(r) else float("nan")


def _friction_stats(pre, post, ctx) -> dict:
    """``pre`` = the slide before the branch (v at the branch, a fitted before it); ``post`` = after the branch while
    the block still moves (a fitted over [branch, legal stop - margin], v there) plus ``v_stop`` [n, d] = the mean
    velocity (line fit of the positions) over the stop window [legal stop + margin, clean end]
    (``io.law_inputs``; ``io.friction_windows``).
      decel_const = sum_i a_i . v_hat_i (post) / the same (pre)       Coulomb 1; viscous drag a ~ v < 1; none 0
      stop_hold   = max_i |v_stop_i| / |v_pre_i|                       Coulomb 0 (stops and stays); reverse, viscous
                                                                       creep, no friction > 0
      mass_ratio  = (a . v_hat)_heavy / (a . v_hat)_light on post      1 (NaN without two distinct masses)
    Each is a ratio of one measured quantity at two times: no length / time scale or common gain enters."""
    e, l = along_motion(pre), along_motion(post)
    hl = _heavy_light(ctx)
    return {"decel_const": safe_ratio(np.nansum(l) if np.isfinite(l).any() else np.nan,
                                      np.nansum(e) if np.isfinite(e).any() else np.nan),
            "stop_hold": _speed_hold(pre, post),
            "mass_ratio": safe_ratio(l[hl[0]], l[hl[1]]) if hl is not None else float("nan")}


def _viscous(pre, ctx) -> dict:
    """§5.2 null: Stokes drag instead of Coulomb friction, F_i = -c v_i with ONE drag constant c (fitted to the pre
    decelerations, c = mean_i m_i |a_i . v_hat_i| / |v_i|): v_i' = v_i exp(-c dt / m_i), a_i' = -(c / m_i) v_i';
    v_stop = v_i exp(-lam_i tau_i) at the Coulomb stop time tau_i = |v_i| / |a_i . v_hat_i| (e^-1 for one body: drag
    matched at the branch never stops). decel_const = exp(-c dt / m) < 1, stop_hold = e^-1 (one body),
    mass_ratio = m_light / m_heavy (the lighter body is slowed more)."""
    v = np.atleast_2d(arr(pre["v"]))
    m = arr(ctx["m"]).reshape(-1) if "m" in ctx else np.ones(len(v))
    sp = np.linalg.norm(v, axis=-1)
    dec = np.abs(along_motion(pre))
    c = float(np.nanmean(m * dec / sp))
    lam = (c / m)[:, None]
    dt = float(ctx.get("dt", 0.0))
    v1 = v * np.exp(-lam * dt)
    return {"v": v1, "a": -lam * v1, "v_stop": v * np.exp(-lam * (sp / dec)[:, None])}


friction = Law("friction", _friction_stats, {"decel_const": ("eq", 1.0), "stop_hold": ("eq", 0.0),
                                             "mass_ratio": ("eq", 1.0)}, _viscous, None,
               "kinetic friction: constant deceleration along the motion until rest, then rest; independent of speed "
               "and of mass")


# --------------------------------------------------------------------------- oscillation (F9)

def _osc_stats(pre, post, ctx) -> dict:
    """``pre["T"]`` / ``post["T"]`` [n_obj]: periods measured before / after the branch (``io.law_inputs``: twice
    a half period between successive passes through the lowest point).
      period_ratio      = geometric mean over objects of T_post / T_pre     1 (a pendulum keeps its period)
      mass_period_ratio = T_post[heavy] / T_post[light]                      1 (NaN without two distinct masses)"""
    nan = float("nan")
    t0 = arr(pre["T"]).reshape(-1) if "T" in pre else np.full(1, nan)
    t1 = arr(post["T"]).reshape(-1) if "T" in post else np.full(1, nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(t1 / t0)
    ok = np.isfinite(r)
    hl = _heavy_light(ctx)
    return {"period_ratio": float(np.exp(r[ok].mean())) if ok.any() else nan,
            "mass_period_ratio": safe_ratio(t1[hl[0]], t1[hl[1]]) if hl is not None and len(t1) > 1 else nan}


def _heavier_faster(pre, ctx) -> dict:
    """§5.2 null (Aristotle: heavier bobs swing faster): after the branch T_i' = T_i sqrt(m_min / m_i)."""
    out = {k: (arr(x).copy() if not isinstance(x, (str, list)) else x) for k, x in pre.items()}
    if "T" in pre and "m" in ctx:
        m = arr(ctx["m"]).reshape(-1)
        out["T"] = arr(pre["T"]).reshape(-1) * np.sqrt(m.min() / m)
    return out


oscillation = Law("oscillation", _osc_stats, {"period_ratio": ("eq", 1.0), "mass_period_ratio": ("eq", 1.0)},
                  _heavier_faster, None, "pendulum period constant within the clip and independent of the bob's mass")
