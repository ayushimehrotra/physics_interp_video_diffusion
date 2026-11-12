"""Rigid bodies on supports: rolling (F10) and static equilibrium / torque balance (F11, :data:`statics`).

Rolling order (F10): on one incline, a_i = g sin(theta) / (1 + k_i) with I_i = k_i m_i r_i^2 (solid cylinder
k = 1/2, tube 1), so log(a_s / a_h) = log((1 + k_h) / (1 + k_s)) whatever g, theta, the length or the time scale.
Rolling without slipping ties rotation to translation: |w| r = |v| (both sides length / time, so the ratio is
scale-free).
"""
from __future__ import annotations

import numpy as np

from physlaw.eval.laws import Law, arr, safe_ratio


def _speed(v) -> np.ndarray:
    return np.linalg.norm(np.atleast_2d(arr(v)), axis=-1)


def _along(pre, post, ctx) -> np.ndarray:
    """Acceleration along the incline per object: (|v'| - |v|) / dt, or, when ctx["dt"] is 0 / absent and ``post``
    carries ``a``, the rate of change of speed a . v_hat' (|a| at rest)."""
    dt = float(ctx.get("dt", 0.0) or 0.0)
    if dt > 0:
        return (_speed(post["v"]) - _speed(pre["v"])) / dt
    if "a" not in post:
        return np.full(len(_speed(post["v"])), np.nan)
    a, v = np.atleast_2d(arr(post["a"])), np.atleast_2d(arr(post["v"]))
    s = np.linalg.norm(v, axis=-1)
    return np.where(s > 0, (a * v).sum(-1) / np.where(s > 0, s, 1.0), np.linalg.norm(a, axis=-1))


def _slip(post, ctx) -> np.ndarray:
    """|w| r / |v| per object on the post state (w angular velocity, r = ctx["r"]); NaN without w or at rest."""
    v = np.atleast_2d(arr(post["v"]))
    if "w" not in post or "r" not in ctx:
        return np.full(len(v), np.nan)
    w = arr(post["w"]).reshape(len(v), -1)
    sp = np.linalg.norm(v, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sp > 0, np.linalg.norm(w, axis=-1) * arr(ctx["r"]).reshape(-1) / sp, np.nan)


def _rolling_stats(pre, post, ctx) -> dict:
    """order = log(a_s / a_h) / log((1 + k_h) / (1 + k_s)), s / h = the objects with the smallest / largest k
    (ctx["k"]). 1 = the law, 0 = inertia ignored (equal accelerations), < 0 = the hollow one wins.
    slip_s, slip_h = |w| r / |v| of those two objects after the event (1 = rolling without slipping, 0 = no
    rotation). With dt = 0 (event instant, both at rest) order uses post["a"] and slip is NaN: pass dt > 0."""
    a, k = _along(pre, post, ctx), arr(ctx["k"])
    s, h = int(np.argmin(k)), int(np.argmax(k))
    slip = _slip(post, ctx)
    order = (safe_ratio(np.log(a[s] / a[h]), np.log((1 + k[h]) / (1 + k[s]))) if a[s] > 0 and a[h] > 0
             else float("nan"))
    return {"order": order, "slip_s": float(slip[s]), "slip_h": float(slip[h])}


def _inertia_ignored(pre, ctx) -> dict:
    """§5.2 null ("inertia ignored"): every object gains the same speed, g sin(theta) dt, along its motion (along the
    first axis from rest), without rotating; order = 0, slip = 0."""
    v = np.atleast_2d(arr(pre["v"]))
    s = np.linalg.norm(v, axis=-1, keepdims=True)
    u = np.where(s > 0, v / np.where(s > 0, s, 1.0), np.eye(v.shape[-1])[0])
    dv = float(ctx["g_mag"]) * np.sin(float(ctx["theta"])) * float(ctx["dt"])
    return {"v": v + dv * u, "w": np.zeros_like(v)}


rolling = Law("rolling", _rolling_stats, {"order": ("eq", 1.0), "slip_s": ("eq", 1.0), "slip_h": ("eq", 1.0)},
              _inertia_ignored, None, "rolling order a_s / a_h = (1 + k_h) / (1 + k_s); rolling without slipping")



# --------------------------------------------------------------------------- statics: torque balance (F11)

def tip_factor(x, beta) -> np.ndarray:
    """|a_cm| / g of a uniform block (half-length a, half-height b = beta a) released at rest on a frictionless
    ledge edge with overhang x (the centre of mass (x - 1) a beyond the edge), when it tips: the edge reaction is
    vertical, so the CoM starts straight down with |a_cm| / g = d^2 / (d^2 + (1 + beta^2) / 3), d = |x - 1|
    (lengths in units of a; Lagrangian of (angle, contact coordinate), see families.f11_support). The same
    magnitude at d = |x - 1| on both sides of x = 1 (the violating tip at x < 1 mirrors the lever)."""
    d2 = (np.asarray(x, float) - 1.0) ** 2
    return d2 / (d2 + (1.0 + np.asarray(beta, float) ** 2) / 3.0)


def tip_rate(x, beta) -> np.ndarray:
    """alpha a / g of the same block at the release: its initial angular acceleration alpha = (g / a) d / (d^2 +
    (1 + beta^2) / 3), d = |x - 1| (so |a_cm| = alpha d a = g tip_factor)."""
    d = np.abs(np.asarray(x, float) - 1.0)
    return d / (d * d + (1.0 + np.asarray(beta, float) ** 2) / 3.0)


TIP_ANGLE = 0.05         # rad: tipped = the block turned by more than this over the tip window (stays turn 0)
TIP_WINDOW_K = 2.0       # tip window = this many tipping time scales sqrt(a / g_clip) after the release


def tip_window(g_clip, half_length, t_separate=None, fallback=None) -> float:
    """Length of the tip window after the release, in the clip's own time unit (so tip_consistency is time-scale
    invariant): TIP_WINDOW_K sqrt(a / g_clip), the tipping time scale (a = the half-length; g_clip measured in the
    clip; phi there = (K^2 / 2) d / (d^2 + (1 + beta^2) / 3) from the legal start, > TIP_ANGLE at every grid overhang),
    or up to ``t_separate`` (time from the release until the block leaves the ledge), whichever is first. Without a
    finite g_clip: ``fallback`` (the twin-shared time from the release to the clean window's end; NaN if None)."""
    g, a = float(g_clip) if g_clip is not None else float("nan"), float(half_length)
    w = TIP_WINDOW_K * np.sqrt(a / g) if np.isfinite(g) and g > 0 and a > 0 else (
        float("nan") if fallback is None else float(fallback))
    return float(min(w, t_separate)) if t_separate is not None and np.isfinite(w) else float(w)


def _rotation(pre, post, ctx) -> float:
    """The block's rotation angle (rad) over the window: post["phi"] (io.law_inputs: the angle between the
    orientations at the release and the end of the tip window, :func:`tip_window`), else the trapezoid
    1/2 (|w_pre| + |w_post|) dt."""
    if "phi" in post:
        return float(np.abs(arr(post["phi"]).reshape(-1)[0]))
    dt = float(ctx.get("dt", 0.0) or 0.0)
    if dt > 0 and "w" in post:
        w = lambda st: float(np.linalg.norm(np.atleast_2d(arr(st["w"]))[0])) if "w" in st else 0.0
        return 0.5 * (w(pre) + w(post)) * dt
    return float("nan")


def _statics_stats(pre, post, ctx) -> dict:
    """Block = object 0. Every term is measured in the clip, so both statistics are length- and time-scale free:
      tip_consistency = 1 if (rotation over the window > TIP_ANGLE) == (x > 1) else 0     legal 1
                        (the rotation is in radians; :func:`_rotation`)
      alpha_ratio     = alpha / alpha_legal for x > 1 (NaN for x <= 1), alpha = |post["alpha"]| the block's angular
                        acceleration right after the release, alpha_legal = g_clip tip_rate(x, beta) / a with
                        g_clip = ctx["g_clip"] (a free fall measured in the same clip: the block after it leaves the
                        ledge, or the prop; NaN when the clip has none -> alpha_ratio NaN, only tip_consistency is
                        scored) and a = ctx["half_length"] in the same length unit.      legal 1; the null (stays) 0
    ctx: overhang_x, aspect_ba (b / a), g_clip, half_length. alpha_ratio is not gain-invariant (g_clip and the
    half-length are measured differently from the angular acceleration)."""
    nan = float("nan")
    x, beta = float(ctx["overhang_x"]), float(ctx["aspect_ba"])
    phi = _rotation(pre, post, ctx)
    tip = float((phi > TIP_ANGLE) == (x > 1.0)) if np.isfinite(phi) else nan
    ratio = nan
    if x > 1.0 and "alpha" in post:
        alpha = float(np.linalg.norm(np.atleast_2d(arr(post["alpha"]))[0]))
        g_clip, a = float(ctx.get("g_clip", nan)), float(ctx.get("half_length", nan))
        ratio = safe_ratio(alpha, g_clip * float(tip_rate(x, beta)) / a) if np.isfinite(g_clip * a) and a > 0 else nan
    return {"tip_consistency": tip, "alpha_ratio": ratio}


def _stays(pre, ctx) -> dict:
    """§5.2 null (no event): the block stays where it was (a = 0, v = 0, no rotation)."""
    v = np.atleast_2d(arr(pre["v"]))
    return {"v": np.zeros_like(v), "a": np.zeros_like(v), "w": np.zeros_like(v), "alpha": np.zeros_like(v),
            "phi": np.zeros(len(v)), **({"p": arr(pre["p"]).copy()} if "p" in pre else {})}


statics = Law("statics", _statics_stats, {"tip_consistency": ("eq", 1.0), "alpha_ratio": ("eq", 1.0)}, _stays, None,
              "torque balance: a block tips off a ledge iff its centre of mass is beyond the edge, at the closed-form rate")
