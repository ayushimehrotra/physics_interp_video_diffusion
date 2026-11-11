"""Two-body collision laws (F1, F2, F3, F4): momentum (with the restitution family), energy,
centre-of-mass velocity, angular momentum about a fixed point, Galilean invariance.

Outcome space and the legal family follow ``physlaw.data.collision_physics``: an outcome is
o = (dv_A, dv_B) in R^4 (plain Euclidean velocity metric), and the frictionless legal outcomes for
a fixed pre-contact state are the segment o(e) = (1 + e) mu u_n (-n/m_A, n/m_B), e in ``e_range``.
The §5.2 null (balls pass through, o = 0, i.e. e = -1) is off the segment.
"""
from __future__ import annotations

import numpy as np

from physlaw.data import collision_physics as cp
from physlaw.eval.laws import Law, arr, cross2, safe_ratio


def _state(pre: dict, ctx: dict) -> cp.PreState:
    v = arr(pre["v"])
    return cp.PreState(float(ctx["m"][0]), float(ctx["m"][1]), v[0, :2], v[1, :2], arr(ctx["n"])[:2])


def _outcome(pre: dict, post: dict) -> np.ndarray:
    return (arr(post["v"])[:, :2] - arr(pre["v"])[:, :2]).reshape(4)


def pass_through(pre: dict, ctx: dict) -> dict:
    """§5.2 null for collisions: no contact, every object keeps its pre-contact velocity."""
    return {k: arr(v).copy() for k, v in pre.items()}


# --------------------------------------------------------------------------- momentum

def _momentum_residual(pre, post, ctx) -> float:
    """|sum m v' - sum m v| / |sum m v| (= labels ``summary.momentum_residual``)."""
    m = arr(ctx["m"])
    p0 = m @ arr(pre["v"])
    return safe_ratio(np.linalg.norm(m @ arr(post["v"]) - p0), np.linalg.norm(p0))


def _momentum_family(post, pre, ctx) -> tuple[float, float]:
    """Distance of the outcome to the legal segment {o(e): e in e_range} in R^4, divided by the
    null's distance to it (|o(e_lo)|, since the null is o = 0). Returns (d, e_hat), e_hat the
    restitution of the nearest family member (clamped to e_range)."""
    s = _state(pre, ctx)
    e_lo, e_hi = ctx.get("e_range", (0.0, 1.0))
    o = _outcome(pre, post)
    b = cp.legal_direction(s)
    scale = s.mu * s.approach_speed
    J = float(np.clip(o @ b / (b @ b), (1 + e_lo) * scale, (1 + e_hi) * scale))
    d = np.linalg.norm(o - J * b)
    return safe_ratio(d, np.linalg.norm(cp.legal_outcome(s, e_lo))), J / scale - 1.0


momentum = Law("momentum", _momentum_residual, pass_through, _momentum_family,
               "linear momentum; legal family = frictionless outcomes o(e), hidden e")


def momentum_exchange(pre: dict, post: dict, ctx: dict) -> dict:
    """Momentum-only diagnostics of an outcome (the L1 S_mom ingredients), blind to restitution, energy
    and impulse direction.

    Any outcome splits uniquely into an INTERACTION part (a change of relative velocity: an equal and
    opposite impulse, which conserves momentum whatever its size or direction) and a CENTRE-OF-MASS part
    (dv_cm = dp / M, exactly the momentum created). With q_i = m_i dv_i the momentum change of ball i:

        num = |q_A + q_B|            momentum created, |dp|
        den = |q_A| + |q_B|          momentum moved around, D
        eta = num / den in [0, 1]    0: what A loses B gains; 1: no transfer (one ball changes, or both
                                     are pushed the same way). Bounded by the triangle inequality.
        d_min = 2 mu u_n             D of the weakest legal collision (e = 0); the interaction gate's unit

    The no-event null has num = den = 0, so eta is undefined there: momentum on its own is satisfied by
    nothing happening, and eta is only read on outcomes that pass a gate ``den >= kappa * d_min``. eta is
    invariant to rescaling the outcome, so uniform probe shrinkage does not move it."""
    m = arr(ctx["m"])
    q = m[:, None] * (arr(post["v"])[:, :2] - arr(pre["v"])[:, :2])
    num = float(np.linalg.norm(q.sum(0)))
    den = float(np.linalg.norm(q, axis=-1).sum())
    s = _state(pre, ctx)
    return {"num": num, "den": den, "eta": safe_ratio(num, den), "d_min": 2.0 * s.mu * s.approach_speed}


def mass_free_momentum(pre: dict, post: dict, ctx: dict | None = None) -> dict:
    """Momentum checks that need NO masses (generated video, hidden mass): for any positive masses,
    m_A dv_A = -m_B dv_B forces dv_A and dv_B to be antiparallel, and a smooth central contact puts both
    along the contact normal n.

        antiparallel = -cos(dv_A, dv_B)          1 = exactly opposite (conservation possible), -1 = same direction
        normal_A/B   = |dv . n| / |dv|           1 = along the contact normal (needs ctx["n"])
        implied_ratio = |dv_B| / |dv_A|          the m_A / m_B momentum would need (compare with the true ratio)
        log_ratio_err = log(implied_ratio) - log(m_A/m_B)   (needs ctx["m"]); 0 for a conserving outcome
    NaN where a ball does not move."""
    dv = arr(post["v"])[:, :2] - arr(pre["v"])[:, :2]
    na, nb = np.linalg.norm(dv, axis=-1)
    out = {"antiparallel": safe_ratio(-float(dv[0] @ dv[1]), na * nb),
           "implied_ratio": safe_ratio(nb, na)}
    if ctx is not None and "n" in ctx:
        n = arr(ctx["n"])[:2] / np.linalg.norm(arr(ctx["n"])[:2])
        out["normal_A"], out["normal_B"] = safe_ratio(abs(float(dv[0] @ n)), na), safe_ratio(abs(float(dv[1] @ n)), nb)
    if ctx is not None and "m" in ctx and np.isfinite(out["implied_ratio"]) and out["implied_ratio"] > 0:
        m = arr(ctx["m"])
        out["log_ratio_err"] = float(np.log(out["implied_ratio"]) - np.log(m[0] / m[1]))
    return out


def mass_blind_outcome(pre: dict, post: dict, ctx: dict) -> dict:
    """The mass-blind reference for S_mom: keep ``post``'s change of relative velocity but split it
    EQUALLY between the balls ("equal and opposite velocity changes", masses ignored). Its momentum
    imbalance is eta = |m_A - m_B| / (m_A + m_B): 0 at equal masses, which is why equal-mass scenes
    cannot show momentum understanding."""
    v0 = arr(pre["v"])[:, :2]
    dv = arr(post["v"])[:, :2] - v0
    h = (dv[0] - dv[1]) / 2.0
    return {"v": v0 + np.stack([h, -h])}


# --------------------------------------------------------------------------- energy

def _ke(m, v) -> float:
    return float(0.5 * (arr(m) * (arr(v) ** 2).sum(-1)).sum())


def _energy_residual(pre, post, ctx) -> float:
    """Energy creation: max(0, KE' - KE) / KE (KE lost in an inelastic contact is legal)."""
    k0 = _ke(ctx["m"], pre["v"])
    return safe_ratio(max(0.0, _ke(ctx["m"], post["v"]) - k0), k0)


energy = Law("energy", _energy_residual, pass_through, None, "no kinetic-energy creation")


# --------------------------------------------------------------------------- centre of mass

def _com_residual(pre, post, ctx) -> float:
    """|v_cm' - v_cm| / (sum m |v| / M). With velocity-only states this is the momentum change
    again (M dv_cm = dp); only the denominator differs: the mass-weighted mean speed, which stays
    away from 0 when the total momentum is small (where r_p blows up)."""
    m = arr(ctx["m"])
    v0, v1 = arr(pre["v"]), arr(post["v"])
    scale = (m * np.linalg.norm(v0, axis=-1)).sum() / m.sum()
    return safe_ratio(np.linalg.norm(m @ v1 - m @ v0) / m.sum(), scale)


com = Law("com", _com_residual, pass_through, None, "centre-of-mass velocity constant through contact")


# --------------------------------------------------------------------------- angular momentum

def _angmom(m, p, v, o) -> np.ndarray:
    r = arr(p) - arr(o)
    L = cross2(r, v) if r.shape[-1] == 2 else np.cross(r, arr(v))
    return (arr(m)[:, None] * np.atleast_2d(L.T).T).sum(0)


def _angmom_residual(pre, post, ctx) -> float:
    """|L' - L| / sum m |p - o| |v| about the fixed point o = ctx["point"] (default origin), with
    L = sum m (p - o) x v. ``post["p"]`` defaults to ``pre["p"]`` (contact is instantaneous).
    Frictionless spheres conserve L about any fixed point; the F1 momentum edit (a common drift
    delta along n) changes it by M (r_cm - o) x delta, so it is invisible about the contact point
    and visible about a point off the line through r_cm along n (hence the origin by default)."""
    o = ctx.get("point", np.zeros(arr(pre["p"]).shape[-1]))
    m, p0 = arr(ctx["m"]), arr(pre["p"])
    p1 = arr(post.get("p", p0))
    dL = _angmom(m, p1, post["v"], o) - _angmom(m, p0, pre["v"], o)
    scale = (m * np.linalg.norm(p0 - arr(o), axis=-1) * np.linalg.norm(arr(pre["v"]), axis=-1)).sum()
    return safe_ratio(np.linalg.norm(dL), scale)


angular_momentum = Law("angular_momentum", _angmom_residual, pass_through, None,
                       "angular momentum about a fixed point, conserved by frictionless contact")


# --------------------------------------------------------------------------- Galilean invariance

def invariants(pre: dict, post: dict, ctx: dict) -> tuple[float, float]:
    """Boost-invariant summaries of a collision: realised restitution
    e = -((v_A' - v_B') . n) / ((v_A - v_B) . n), and A's scattering angle in the centre-of-mass
    frame (signed, radians)."""
    m, v0, v1, n = arr(ctx["m"]), arr(pre["v"]), arr(post["v"]), arr(ctx["n"])
    e = -float((v1[0] - v1[1]) @ n) / float((v0[0] - v0[1]) @ n)
    ua, ub = v0[0] - m @ v0 / m.sum(), v1[0] - m @ v1 / m.sum()
    return e, float(np.arctan2(cross2(ua, ub), ua @ ub))


def _galilean_residual(pre, post, ctx) -> float:
    """|e - e_ref| + |theta - theta_ref| / pi, against the unboosted clip's invariants
    (ctx["e_ref"], ctx["theta_ref"]; or ctx["ref"] = (pre, post) of that clip). Both summaries
    are already boost-invariant, so any nonzero residual is a broken Galilean invariance."""
    if "ref" in ctx:
        e_ref, th_ref = invariants(*ctx["ref"], ctx)
    else:
        e_ref, th_ref = ctx["e_ref"], ctx["theta_ref"]
    e, th = invariants(pre, post, ctx)
    dth = (th - th_ref + np.pi) % (2 * np.pi) - np.pi
    return float(abs(e - e_ref) + abs(dth) / np.pi)


galilean = Law("galilean", _galilean_residual, pass_through, None,
               "restitution and CoM-frame scattering angle invariant under a common boost")
