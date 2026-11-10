"""Incline laws: frictionless ramp energy exchange (F8) and rolling order (F10).

Speeds are |v| per object; ``ctx["g_mag"]`` is |g|.
"""
from __future__ import annotations

import numpy as np

from physlaw.eval.laws import Law, arr, safe_ratio
from physlaw.eval.laws.collision import pass_through


def _speed(v) -> np.ndarray:
    return np.linalg.norm(np.atleast_2d(arr(v)), axis=-1)


# --------------------------------------------------------------------------- ramp energy

def _ramp_family(post, pre, ctx) -> tuple[float, None]:
    """Bottom speed sqrt(|v_entry|^2 + 2 g h) (sqrt(2 g h) from rest), ctx h = height dropped,
    whatever the path (slope, kink). d = |s' - s_legal| / |s_legal - s_entry|: relative to the null,
    which holds the ramp-entry speed. Unique family (the speed-scaled-with-slope twin is off it)."""
    s0, s1 = _speed(pre["v"]), _speed(post["v"])
    legal = np.sqrt(s0 ** 2 + 2 * float(ctx["g_mag"]) * float(ctx["h"]))
    return safe_ratio(np.linalg.norm(s1 - legal), np.linalg.norm(legal - s0)), None


ramp_energy = Law("ramp_energy", lambda pre, post, ctx: _ramp_family(post, pre, ctx)[0], pass_through, _ramp_family,
                  "frictionless ramp: bottom speed sqrt(2 g h), independent of path")


# --------------------------------------------------------------------------- rolling

def _rolling_acc(ctx) -> np.ndarray:
    """a_i = g sin(theta) / (1 + k_i) along the incline, I_i = k_i m_i r_i^2 (solid cylinder 1/2,
    hollow 1)."""
    return float(ctx["g_mag"]) * np.sin(float(ctx["theta"])) / (1 + arr(ctx["k"]))


def _rolling_residual(pre, post, ctx) -> float:
    """Order violation: max(0, 1 - (a_s - a_h) / (a_s^legal - a_h^legal)), a = (|v'| - |v|) / dt,
    s / h = the objects with the smallest / largest k. 0 if the solid one pulls ahead at least as
    the law says, 1 if both arrive together, > 1 if the hollow one wins."""
    a = (_speed(post["v"]) - _speed(pre["v"])) / float(ctx["dt"])
    k = arr(ctx["k"])
    s, h = int(np.argmin(k)), int(np.argmax(k))
    leg = _rolling_acc(ctx)
    return float(max(0.0, 1 - (a[s] - a[h]) / (leg[s] - leg[h])))


def _rolling_family(post, pre, ctx) -> tuple[float, None]:
    """Speeds after dt vs |v| + a_legal dt per object, relative to the constant-speed null."""
    s0, s1 = _speed(pre["v"]), _speed(post["v"])
    legal = s0 + _rolling_acc(ctx) * float(ctx["dt"])
    return safe_ratio(np.linalg.norm(s1 - legal), np.linalg.norm(legal - s0)), None


rolling = Law("rolling", _rolling_residual, pass_through, _rolling_family,
              "rolling order: lower rotational inertia reaches the bottom first")
