"""Exchange-event laws: momentum (F1-F4; inertia on F6 glides), energy as generalised restitution (F1-F4, F7 bounce,
F8 pad, F9 swing), Galilean invariance (F4) and angular momentum (F12).

Momentum's secondary legal family follows ``physlaw.data.collision_physics``: an outcome is o = (dv_A, dv_B) in
R^4 and the frictionless legal outcomes for a fixed pre-contact state are the segment
o(e) = (1 + e) mu u_n (-n/m_A, n/m_B), e in ``e_range``. The no-event null (o = 0, e = -1) is off the segment.
"""
from __future__ import annotations

import numpy as np

from physlaw.data import collision_physics as cp
from physlaw.eval.laws import Law, arr, cross2, safe_ratio

EVENTS = ("collision", "bounce", "pad", "swing")


def _state(pre: dict, ctx: dict) -> cp.PreState:
    v = arr(pre["v"])
    return cp.PreState(float(ctx["m"][0]), float(ctx["m"][1]), v[0, :2], v[1, :2], arr(ctx["n"])[:2])


def _outcome(pre: dict, post: dict) -> np.ndarray:
    return (arr(post["v"])[:, :2] - arr(pre["v"])[:, :2]).reshape(4)


def _unit(x, d: int) -> np.ndarray:
    x = arr(x)[:d]
    return x / np.linalg.norm(x)


def pass_through(pre: dict, ctx: dict) -> dict:
    """§5.2 no-event null: every object keeps its pre-event state (balls pass through, no bounce, pad ignored)."""
    return {k: (arr(v).copy() if not isinstance(v, (str, list)) else v) for k, v in pre.items()}


# --------------------------------------------------------------------------- energy: generalised restitution

def epsilon(pre: dict, post: dict, ctx: dict) -> float:
    """Generalised restitution of the exchange event, by ``ctx["event"]`` (inferred "collision" when absent and ctx
    has a normal ``n``):

      collision  -(v_A' - v_B') . n / ((v_A - v_B) . n)      n = ctx["n"] (A -> B), mass-free, 2-D or 3-D
      bounce     -(v' . g_hat) / (v . g_hat)                 g = ctx["g"] (floor = infinite mass); summed over objects
      pad        |v'| / |v|                                  speed after / before the dissipation pad (F8)
      swing      sqrt(h' / h), h = (p - p_bottom) . (-g_hat)   the apex heights above the lowest point of the
                 swing before / after the damper (F9; summed over objects). pre / post are the apex states
                 (v = 0, so h is the whole energy height and no g enters: ``io.law_inputs`` gives them for swing
                 events). Needs ctx["p_bottom"]; only the direction of ctx["g"] is used (default -z)

    eps <= 1 is legal; > 1 creates energy. With momentum conserved, KE - KE' = 1/2 mu u_n^2 (1 - eps^2), so
    eps^2 is the internal (CoM-frame, normal) energy ratio: boost-invariant and mass-free. Vectors of different
    dimension are compared on their common leading components."""
    kind = ctx.get("event")
    if kind is None:
        if "n" not in ctx:
            raise ValueError("epsilon: ctx has no 'event' and no contact normal 'n'")
        kind = "collision"
    v0, v1 = arr(pre["v"]), arr(post["v"])
    if kind == "collision":
        d = min(v0.shape[-1], arr(ctx["n"]).shape[-1])
        n = _unit(ctx["n"], d)
        return safe_ratio(-float((v1[0, :d] - v1[1, :d]) @ n), float((v0[0, :d] - v0[1, :d]) @ n))
    if kind == "bounce":
        d = min(v0.shape[-1], arr(ctx["g"]).shape[-1])
        z = _unit(ctx["g"], d)
        return safe_ratio(-float((v1[..., :d] @ z).sum()), float((v0[..., :d] @ z).sum()))
    if kind == "pad":
        return safe_ratio(np.linalg.norm(v1), np.linalg.norm(v0))
    if kind == "swing":
        g = arr(ctx.get("g", [0.0, 0.0, -1.0]))
        d = min(arr(pre["p"]).shape[-1], g.shape[-1])
        up, pb = -_unit(g, d), arr(ctx["p_bottom"])[..., :d]
        h = lambda st: float(((arr(st["p"])[..., :d] - pb) @ up).sum())
        r = safe_ratio(h(post), h(pre))
        return float(np.sqrt(r)) if r >= 0 else float("nan")
    raise ValueError(f"epsilon: event {kind!r} not in {EVENTS}")


def rho_star(pre: dict, post: dict, ctx: dict) -> float:
    """eps^2: the fraction of the exchanged (internal) energy the event returns."""
    return epsilon(pre, post, ctx) ** 2


def _energy_stats(pre, post, ctx) -> dict:
    e = epsilon(pre, post, ctx)
    return {"epsilon": e, "rho_star": e * e, "creation": max(0.0, e - 1.0) if np.isfinite(e) else float("nan")}


energy = Law("energy", _energy_stats, {"epsilon": ("le", 1.0)}, pass_through, None,
             "no energy creation at the exchange event: generalised restitution eps <= 1 (one-sided)")


# --------------------------------------------------------------------------- momentum

def momentum_exchange(pre: dict, post: dict, ctx: dict) -> dict:
    """Momentum-only diagnostics of an outcome (the L1 S_mom ingredients), blind to restitution, energy
    and impulse direction. With q_i = m_i dv_i:

        num = |q_A + q_B|            momentum created, |dp|
        den = |q_A| + |q_B|          momentum moved around, D
        eta = num / den in [0, 1]    0: what A loses B gains; 1: no transfer
        d_min = 2 mu u_n             D of the weakest legal collision (e = 0); the interaction gate's unit

    The no-event null has num = den = 0 (eta undefined), so eta is read only on outcomes that pass a gate
    ``den >= kappa * d_min``. eta is invariant to rescaling the outcome."""
    m = arr(ctx["m"])
    q = m[:, None] * (arr(post["v"])[:, :2] - arr(pre["v"])[:, :2])
    num = float(np.linalg.norm(q.sum(0)))
    den = float(np.linalg.norm(q, axis=-1).sum())
    s = _state(pre, ctx)
    return {"num": num, "den": den, "eta": safe_ratio(num, den), "d_min": 2.0 * s.mu * s.approach_speed}


def mass_free_momentum(pre: dict, post: dict, ctx: dict | None = None) -> dict:
    """Momentum checks that need NO masses: for any positive masses, m_A dv_A = -m_B dv_B forces dv_A and dv_B
    antiparallel, and a smooth central contact puts both along the contact normal n.

        antiparallel = -cos(dv_A, dv_B)          1 = exactly opposite, -1 = same direction
        normal_A/B   = |dv . n| / |dv|           1 = along the contact normal (needs ctx["n"])
        implied_ratio = |dv_B| / |dv_A|          the m_A / m_B momentum would need
        log_ratio_err = log(implied_ratio) - log(m_A/m_B)   (needs ctx["m"])
    NaN where a ball does not move."""
    dv = arr(post["v"])[:, :2] - arr(pre["v"])[:, :2]
    na, nb = np.linalg.norm(dv, axis=-1)
    out = {"antiparallel": safe_ratio(-float(dv[0] @ dv[1]), na * nb), "implied_ratio": safe_ratio(nb, na)}
    if ctx is not None and "n" in ctx:
        n = _unit(ctx["n"], 2)
        out["normal_A"], out["normal_B"] = safe_ratio(abs(float(dv[0] @ n)), na), safe_ratio(abs(float(dv[1] @ n)), nb)
    if ctx is not None and "m" in ctx and np.isfinite(out["implied_ratio"]) and out["implied_ratio"] > 0:
        m = arr(ctx["m"])
        out["log_ratio_err"] = float(np.log(out["implied_ratio"]) - np.log(m[0] / m[1]))
    return out


def mass_blind_outcome(pre: dict, post: dict, ctx: dict) -> dict:
    """The mass-blind reference for S_mom: keep ``post``'s change of relative velocity but split it EQUALLY between
    the balls. Its eta = |m_A - m_B| / (m_A + m_B)."""
    v0 = arr(pre["v"])[:, :2]
    dv = arr(post["v"])[:, :2] - v0
    h = (dv[0] - dv[1]) / 2.0
    return {"v": v0 + np.stack([h, -h])}


def inertia(pre: dict, post: dict) -> dict:
    """Newton's first law on a force-free segment (F6 glide): inertia_speed = |V'| / |V| (Frobenius norm over the
    objects: |v'| / |v| for one body), inertia_turn = the largest angle(v_i, v_i') / pi. Legal 1 and 0."""
    v0, v1 = arr(pre["v"]).reshape(-1, arr(pre["v"]).shape[-1]), arr(post["v"]).reshape(-1, arr(post["v"]).shape[-1])
    n0, n1 = np.linalg.norm(v0, axis=-1), np.linalg.norm(v1, axis=-1)
    cr = np.cross(np.pad(v0, ((0, 0), (0, 3 - v0.shape[-1]))), np.pad(v1, ((0, 0), (0, 3 - v1.shape[-1]))))
    ang = np.where(n0 * n1 > 0, np.arctan2(np.linalg.norm(cr, axis=-1), (v0 * v1).sum(-1)), np.nan)
    return {"inertia_speed": safe_ratio(np.linalg.norm(v1), np.linalg.norm(v0)),
            "inertia_turn": float(np.max(ang)) / np.pi}


def _momentum_stats(pre, post, ctx) -> dict:
    """Two-body exchange (F1-F4): eta needs ctx["m"] (else NaN); antiparallel needs nothing; normal_A/B need ctx["n"]
    (else NaN). Force-free segment (ctx["force_free"], F6 glide): inertia_speed, inertia_turn (:func:`inertia`).
    Statistics of the other kind are NaN."""
    nan = float("nan")
    out = {"eta": nan, "antiparallel": nan, "normal_A": nan, "normal_B": nan, "inertia_speed": nan,
           "inertia_turn": nan}
    if ctx.get("force_free"):
        out.update(inertia(pre, post))
        return out
    if len(arr(pre["v"])) < 2:
        return out
    mf = mass_free_momentum(pre, post, ctx)
    if "m" in ctx:
        m = arr(ctx["m"])
        q = m[:, None] * (arr(post["v"])[:, :2] - arr(pre["v"])[:, :2])
        out["eta"] = safe_ratio(np.linalg.norm(q.sum(0)), np.linalg.norm(q, axis=-1).sum())
    out.update(antiparallel=mf["antiparallel"], normal_A=mf.get("normal_A", nan), normal_B=mf.get("normal_B", nan))
    return out


def _momentum_null(pre: dict, ctx: dict) -> dict:
    """§5.2 nulls: no exchange (:func:`pass_through`) for collisions; for a force-free segment the Aristotelian
    null, motion dies without a force (v' = 0: inertia_speed 0, inertia_turn undefined)."""
    if ctx.get("force_free"):
        out = pass_through(pre, ctx)
        out["v"] = np.zeros_like(arr(pre["v"]))
        return out
    return pass_through(pre, ctx)


def _momentum_family(post, pre, ctx) -> tuple[float, float]:
    """Distance of the outcome to the legal segment {o(e): e in e_range} in R^4, divided by the null's distance to
    it (|o(e_lo)|, the null being o = 0). Returns (d, e_hat), e_hat the restitution of the nearest member."""
    s = _state(pre, ctx)
    e_lo, e_hi = ctx.get("e_range", (0.0, 1.0))
    o = _outcome(pre, post)
    b = cp.legal_direction(s)
    scale = s.mu * s.approach_speed
    J = float(np.clip(o @ b / (b @ b), (1 + e_lo) * scale, (1 + e_hi) * scale))
    d = np.linalg.norm(o - J * b)
    return safe_ratio(d, np.linalg.norm(cp.legal_outcome(s, e_lo))), J / scale - 1.0


momentum = Law("momentum", _momentum_stats,
               {"eta": ("eq", 0.0), "antiparallel": ("eq", 1.0), "normal_A": ("eq", 1.0), "normal_B": ("eq", 1.0),
                "inertia_speed": ("eq", 1.0), "inertia_turn": ("eq", 0.0)},
               _momentum_null, _momentum_family,
               "linear momentum: eta = |q_A + q_B| / (|q_A| + |q_B|) = 0; mass-free: dv antiparallel, along n; "
               "Newton I (no force: |v'| / |v| = 1, no turn)")


# --------------------------------------------------------------------------- Galilean invariance

def invariants(pre: dict, post: dict, ctx: dict) -> tuple[float, float]:
    """Boost-invariant summaries of a collision: eps (:func:`epsilon`, collision) and A's signed scattering angle
    in the centre-of-mass frame (radians; needs ctx["m"])."""
    m, v0, v1 = arr(ctx["m"]), arr(pre["v"])[:, :2], arr(post["v"])[:, :2]
    e = epsilon(pre, post, {**ctx, "event": "collision"})
    ua, ub = v0[0] - m @ v0 / m.sum(), v1[0] - m @ v1 / m.sum()
    return e, float(np.arctan2(cross2(ua, ub), ua @ ub))


def _galilean_stats(pre, post, ctx) -> dict:
    """d_epsilon = |eps - eps_ref|, d_theta = |wrap(theta_cm - theta_ref)| / pi against the unboosted clip
    (ctx["ref"] = (pre, post) of that clip, or ctx["e_ref"], ctx["theta_ref"])."""
    if "ref" in ctx:
        e_ref, th_ref = invariants(*ctx["ref"], ctx)
    else:
        e_ref, th_ref = float(ctx["e_ref"]), float(ctx["theta_ref"])
    e, th = invariants(pre, post, ctx)
    dth = (th - th_ref + np.pi) % (2 * np.pi) - np.pi
    return {"d_epsilon": abs(e - e_ref), "d_theta": abs(dth) / np.pi}


galilean = Law("galilean", _galilean_stats, {"d_epsilon": ("eq", 0.0), "d_theta": ("eq", 0.0)}, pass_through, None,
               "restitution and CoM-frame scattering angle invariant under a common boost")


# --------------------------------------------------------------------------- angular momentum (F12)

def _spin(st: dict, n: int) -> np.ndarray:
    """z angular velocity per object: w [n, 3] (z component), [n, 1] / [n] (already z), 0 when absent."""
    if "w" not in st:
        return np.zeros(n)
    w = arr(st["w"])
    return w.reshape(n, -1)[:, -1]


def inertias(ctx: dict) -> np.ndarray:
    """I_i about the z axis through each centroid: ctx["I"], else k_i m_i r_i^2 (ctx["k"], ["m"], ["r"])."""
    if "I" in ctx:
        return arr(ctx["I"]).reshape(-1)
    return arr(ctx["k"]).reshape(-1) * arr(ctx["m"]).reshape(-1) * arr(ctx["r"]).reshape(-1) ** 2


def angular_momentum_z(st: dict, ctx: dict) -> float:
    """Planar angular momentum about the state's centre of mass: sum m (r - R) x (v - V) + sum I w (z component).
    Origin-free and, with linear momentum conserved, the same about any fixed point."""
    m = arr(ctx["m"]).reshape(-1)
    p, v = arr(st["p"])[:, :2], arr(st["v"])[:, :2]
    R, V = m @ p / m.sum(), m @ v / m.sum()
    return float((m * cross2(p - R, v - V)).sum() + (inertias(ctx) * _spin(st, len(m))).sum())


def rigid_impulse(pre: dict, ctx: dict, e: float) -> dict:
    """Frictionless planar impulse between two rigid bodies at contact point ctx["c"] (or "contact_point") with
    normal n (A -> B) and restitution e: J = (1 + e) u_n / (1/m_A + 1/m_B + (r_A x n)^2 / I_A + (r_B x n)^2 / I_B),
    u_n the approach speed of the contact points. Returns the post state {"v" [2, d], "w" [2, 3], "p"}."""
    m, I = arr(ctx["m"]).reshape(-1), inertias(ctx)
    n = _unit(ctx["n"], 2)
    c = arr(ctx["c"] if "c" in ctx else ctx["contact_point"])[:2]
    v0 = arr(pre["v"])
    p, v, w = arr(pre["p"])[:, :2], v0[:, :2], _spin(pre, 2)
    r = c - p
    rn = cross2(r, n)
    perp = lambda x: np.array([-x[1], x[0]])
    vc = v + w[:, None] * np.stack([perp(r[0]), perp(r[1])])
    u = float((vc[0] - vc[1]) @ n)
    J = (1.0 + e) * u / (1 / m[0] + 1 / m[1] + rn[0] ** 2 / I[0] + rn[1] ** 2 / I[1])
    dv = np.stack([-J * n / m[0], J * n / m[1]])
    v1 = v0.copy()
    v1[:, :2] = v + dv
    w1 = np.zeros((2, 3))
    w1[:, 2] = w + np.array([-rn[0] * J / I[0], rn[1] * J / I[1]])
    return {"v": v1, "w": w1, **({"p": arr(pre["p"]).copy()} if "p" in pre else {})}


def _angular_stats(pre, post, ctx) -> dict:
    """L_ratio = L_z after / before about the system centre of mass (positions of both states at the contact
    instant; needs ctx m and the inertias). Legal 1. Not gain-invariant: a gain on the measured velocities alone
    is not common to the measured spins (as rolling's slip)."""
    return {"L_ratio": safe_ratio(angular_momentum_z(post, ctx), angular_momentum_z(pre, ctx))}


def _spin_ignored(pre, ctx) -> dict:
    """§5.2 null ("rotation ignored"): the legal impulse (restitution ctx["e"], default 1) with every spin left
    unchanged; L_ratio = 1 - I_B w_B' / L."""
    out = rigid_impulse(pre, ctx, float(ctx.get("e", 1.0)))
    n = len(arr(ctx["m"]).reshape(-1))
    out["w"] = np.zeros((n, 3))
    out["w"][:, 2] = _spin(pre, n)
    return out


angular_momentum = Law("angular_momentum", _angular_stats, {"L_ratio": ("eq", 1.0)}, _spin_ignored, None,
                       "angular momentum about the centre of mass (orbital + spin) unchanged by a contact")
