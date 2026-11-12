"""F8: Galileo's ramp (spec §5.0.2), plus ramp constants F6 and F10 reuse.

Side view. A frictionless cube is held at rest on ramp A with its centre h above its flush-on-the-floor height,
released, and descends (pre-event: the clip shows its own g and scale). It runs onto the flat floor and crosses a
PAD of length L_pad (a visibly different strip, rendered identically in every twin). Event = the block's centre
entering the pad (between frames, event_kind "pad"). On the pad the tangential acceleration is constant and takes the
speed from s to eps*s; the block then climbs ramp C (angle theta_C) to an apex at centre height eps^2 h and slides
back (the pad acts the same on every crossing; all of that is after the clean window).

Twins: one per rung of ``twins.epsilon_ladder`` (default 0.5, 0.55, ..., 1.5), named ``eps<e>``:
eps < 1 dissipative pad and eps = 1 frictionless pad are legal; eps > 1 creates energy
(violated_law "energy", violation_magnitude eps). One pre_segment: every twin is byte-identical before
the event. The old twins (slope_b, kink, x1.25, path dependence, the k<f> time-rescaled speed ladder) are
gone: each was the legal motion with gravity k^2 g, i.e. a playback-speed change.

Per twin: epsilon = eps; clean_until_s = the apex time on ramp C; event_state pre = the block entering
the pad, post = the block leaving it (exact states, post.t_s = t_pad_exit_s); stop_dose NaN (|v'| = eps s
is minimal at eps = 0, outside the ladder).

Geometry (2026-11-12): the ramp SURFACE is straight parts joined by concave circular fillets (radius 4 x the block's
half-diagonal); the block rides it in exact two-corner contact (:class:`Contact`: flush on straight parts, its face a
chord of the surface where it bridges a fillet, so no corner ever enters the surface), and turns while it bridges.
Motion (:func:`motion`): one degree of freedom with the rotation in the energy, constant acceleration on straight
parts (exact), DOP853 at rtol 1e-10 around fillets, in both directions. Ramps render as one swept mesh each
(``Static.profile``).
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

import numpy as np

from physlaw.data.families import (G, LOOKS, Body, Scene, Static, Track, Twin, eps_ladder, eps_name, event_time,
                                   framed_side_camera, frame_times, pick_look, quat_y, stop_dose,
                                   visible_box)

SUBFAMILIES = {"ramp": 250}
FILLET = 0.6               # fillet radius of the block-centre path (m)
RAMP_THICK = 2.0           # ramp slabs extend this far below their surface (into the floor)
RAMP_DEPTH = 0.75          # half-depth (y) of the ramp slabs
FILLET_DIAG = 4.0          # F8 fillet radius / the block's half-diagonal (surface)
FRAME_FILL = 0.85         # F8: release .. eps_max apex fill this much of the binding image dimension


# --------------------------------------------------------------------------- #
# path: straight parts joined by concave circular fillets
# --------------------------------------------------------------------------- #


def _d(phi):   # unit tangent of a path descending to +x at angle phi
    return np.array([math.cos(phi), 0.0, -math.sin(phi)])


def _n(phi):   # upward unit normal
    return np.array([math.sin(phi), 0.0, math.cos(phi)])


@dataclass
class Path:
    """Centre path of a body touching a ramp surface at distance ``offset`` (its half-size).

    Straight part i descends at ``angles[i]`` (negative: climbing) over ``lengths[i]`` from its corner
    vertex to the next (the last: to its end); every corner is a concave fillet of radius ``fillet``.
    Arc length s = 0 at ``start``; beyond either end the end parts continue straight."""
    start: np.ndarray
    angles: list
    lengths: list
    offset: float
    fillet: float = FILLET
    prims: list = field(default_factory=list)   # ("line", P, phi, L) | ("arc", C, phi0, phi1, rho)

    def __post_init__(self):
        if any(a1 > a0 for a0, a1 in zip(self.angles[:-1], self.angles[1:])):
            raise ValueError("convex corner")
        tl = [0.0] + [self.fillet * math.tan(0.5 * (a0 - a1)) for a0, a1 in zip(self.angles[:-1], self.angles[1:])]
        tl.append(0.0)
        P = np.asarray(self.start, float)
        for i, phi in enumerate(self.angles):
            L = self.lengths[i] - tl[i] - tl[i + 1]
            if L < -1e-12:
                raise ValueError("fillets overlap")
            self.prims.append(("line", P, phi, L))
            P = P + _d(phi) * L
            if tl[i + 1] > 0:
                a1 = self.angles[i + 1]
                C = P + self.fillet * _n(phi)
                self.prims.append(("arc", C, phi, a1, self.fillet))
                P = C - self.fillet * _n(a1)
        self.bounds = np.cumsum([0.0] + [self.length_of(p) for p in self.prims])

    @staticmethod
    def length_of(p):
        return p[3] if p[0] == "line" else p[4] * (p[2] - p[3])

    def line_start(self, i: int) -> float:
        """Arc length where straight part i starts."""
        return float(self.bounds[[j for j, p in enumerate(self.prims) if p[0] == "line"][i]])

    def locate(self, s: float):
        i = int(np.clip(np.searchsorted(self.bounds, s, side="right") - 1, 0, len(self.prims) - 1))
        return i, s - self.bounds[i]

    def phi(self, s: float) -> float:
        i, u = self.locate(s)
        p = self.prims[i]
        return p[2] if p[0] == "line" else p[2] - u / p[4]

    def point(self, s: float) -> np.ndarray:
        i, u = self.locate(s)
        p = self.prims[i]
        if p[0] == "line":
            return p[1] + _d(p[2]) * u
        return p[1] - p[4] * _n(p[2] - u / p[4])

    def curvature(self, s: float) -> float:
        i, _ = self.locate(s)
        p = self.prims[i]
        return 0.0 if p[0] == "line" else 1.0 / p[4]

    def points(self, u) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised :meth:`point` and :meth:`phi`: ([n, 3], [n]); beyond either end the end parts continue."""
        u = np.atleast_1d(np.asarray(u, float))
        i = np.clip(np.searchsorted(self.bounds, u, side="right") - 1, 0, len(self.prims) - 1)
        w = u - self.bounds[i]
        P, phi = np.zeros((u.size, 3)), np.zeros(u.size)
        for j, p in enumerate(self.prims):
            m = i == j
            if not m.any():
                continue
            if p[0] == "line":
                phi[m] = p[2]
                P[m] = p[1] + np.outer(w[m], _d(p[2]))
            else:
                a = p[2] - w[m] / p[4]
                phi[m] = a
                P[m] = p[1] - p[4] * np.stack([np.sin(a), np.zeros_like(a), np.cos(a)], -1)
        return P, phi

    def profile(self, u0: float, u1: float, per_arc: int = 256) -> list:
        """Surface polyline (x, z) from arc length u0 to u1: the line ends plus ``per_arc`` chords per fillet."""
        us = {float(u0), float(u1)} | {float(b) for b in self.bounds if u0 < b < u1}
        for j, p in enumerate(self.prims):
            if p[0] == "arc":
                lo, hi = max(self.bounds[j], u0), min(self.bounds[j + 1], u1)
                if hi > lo:
                    us |= set(np.linspace(lo, hi, per_arc + 1).tolist())
        P, _ = self.points(sorted(us))
        return [(float(x), float(z)) for x, z in P[:, [0, 2]]]


# --------------------------------------------------------------------------- #
# a flat-faced block on the concave surface: exact two-corner contact
# --------------------------------------------------------------------------- #


class Contact:
    """A cube of half-size ``s`` on a concave ``surface`` (a Path with offset 0), its bottom face a chord of the
    surface: rear bottom corner at surface arc length u, front corner at the surface point 2s further in straight
    distance. On a straight part the face lies flush (centre at s along the normal, orientation = the slope); on and
    near a fillet both corners touch the surface and the face bridges it, so no corner ever enters the surface (a
    flat face cannot follow a concave curve). Configuration = u: centre c(u), orientation theta(u) (the chord's
    descent angle), with their first and second u-derivatives in closed form (:meth:`geo`). Fillet zones
    (u in [arc start - 2s, arc end]) are split at the four instants a corner enters or leaves the arc (``breaks``),
    where the second derivatives jump."""

    def __init__(self, surface: Path, s: float):
        self.surf, self.s = surface, float(s)
        self.k = (2.0 * self.s) ** 2 / 6.0                      # I / m of a cube about a face-parallel axis
        self.zones, self.breaks, self.pre_cache = [], set(), {}
        for j, p in enumerate(surface.prims):
            if p[0] != "arc":
                continue
            a0, a1 = float(surface.bounds[j]), float(surface.bounds[j + 1])
            lo = a0 - 2.0 * self.s
            from scipy.optimize import brentq
            u_fl = brentq(lambda x: self._front(x) - a1, lo, a1, xtol=1e-14)      # front corner leaves the arc
            self.zones.append((lo, a1))
            self.breaks |= {lo, a0, float(u_fl), a1}

    def _at(self, u: float):
        """Surface point, unit tangent and second derivative at arc length u (scalar; (x, z) pairs)."""
        sf = self.surf
        i = min(max(bisect.bisect_right(sf.bounds, u) - 1, 0), len(sf.prims) - 1)
        p = sf.prims[i]
        w = u - sf.bounds[i]
        if p[0] == "line":
            ph = p[2]
            return (p[1][0] + w * math.cos(ph), p[1][2] - w * math.sin(ph)), (math.cos(ph), -math.sin(ph)), (0.0, 0.0)
        C, rho = p[1], p[4]
        a = p[2] - w / rho
        P = (C[0] - rho * math.sin(a), C[2] - rho * math.cos(a))
        return P, (math.cos(a), -math.sin(a)), ((C[0] - P[0]) / rho ** 2, (C[2] - P[1]) / rho ** 2)

    def _front(self, u: float) -> float:
        """Arc length of the front corner: |S(uf) - S(u)| = 2s, uf >= u + 2s (Newton)."""
        P, _, _ = self._at(u)
        uf = u + 2.0 * self.s                              # (no warm start: results must not depend on call order)
        for _ in range(50):
            Q, tq, _ = self._at(uf)
            dx, dz = Q[0] - P[0], Q[1] - P[1]
            step = (dx * dx + dz * dz - 4.0 * self.s ** 2) / (2.0 * (dx * tq[0] + dz * tq[1]))
            uf -= step
            if abs(step) < 1e-14:
                break
        return uf

    def in_zone(self, u: float) -> bool:
        return any(lo <= u <= hi for lo, hi in self.zones)

    def geo(self, u: float):
        """(c, c', c'', theta, theta', theta'') at u: c* are (x, z) pairs, derivatives with respect to u."""
        s = self.s
        P, tr, Pr = self._at(u)
        if not self.in_zone(u):                                 # flush on a straight part
            ph = math.atan2(-tr[1], tr[0])
            n = (math.sin(ph), math.cos(ph))
            return (P[0] + s * (tr[0] + n[0]), P[1] + s * (tr[1] + n[1])), tr, (0.0, 0.0), ph, 0.0, 0.0
        uf = self._front(u)
        Q, tf, Qf = self._at(uf)
        D = (Q[0] - P[0], Q[1] - P[1])
        dot = lambda a, b: a[0] * b[0] + a[1] * b[1]
        f1 = dot(D, tr) / dot(D, tf)                            # uf'
        D1 = (tf[0] * f1 - tr[0], tf[1] * f1 - tr[1])
        f2 = (-dot(D1, D1) - dot(D, Qf) * f1 * f1 + dot(D, Pr)) / dot(D, tf)
        D2 = (Qf[0] * f1 * f1 + tf[0] * f2 - Pr[0], Qf[1] * f1 * f1 + tf[1] * f2 - Pr[1])
        th = math.atan2(-D[1], D[0])
        e, ep = (math.cos(th), -math.sin(th)), (-math.sin(th), -math.cos(th))
        t1, t2 = dot(D1, ep) / (2 * s), dot(D2, ep) / (2 * s)
        c = (0.5 * (P[0] + Q[0]) + s * math.sin(th), 0.5 * (P[1] + Q[1]) + s * math.cos(th))
        c1 = tuple(0.5 * (tr[i] + tf[i] * f1) + s * t1 * e[i] for i in range(2))
        c2 = tuple(0.5 * (Pr[i] + Qf[i] * f1 * f1 + tf[i] * f2) + s * t2 * e[i] + s * t1 * t1 * ep[i] for i in range(2))
        return c, c1, c2, th, t1, t2

    def state(self, u) -> dict:
        """c [n, 3], dc/du [n, 3], theta [n], dtheta/du [n] at u (array)."""
        g = [self.geo(float(x)) for x in np.atleast_1d(np.asarray(u, float))]
        xz3 = lambda a: np.array([[q[0], 0.0, q[1]] for q in a]).reshape(-1, 3)
        return {"c": xz3([x[0] for x in g]), "dc": xz3([x[1] for x in g]),
                "th": np.array([x[3] for x in g]), "dth": np.array([x[4] for x in g])}

    def u_at_height(self, z: float, lo: float, hi: float) -> float:
        """Rear-corner u in [lo, hi] at which the centre is at height z (centre height monotone there)."""
        from scipy.optimize import brentq
        return float(brentq(lambda x: self.geo(x)[0][1] - z, lo, hi, xtol=1e-13))


def _first_hit(x0: float, v: float, a: float, target: float) -> float:
    """Smallest dt > 0 with x0 + v dt + a dt^2 / 2 = target (inf if never)."""
    c = x0 - target
    if a == 0.0:
        dt = -c / v if v != 0.0 else math.inf
        return dt if dt > 0.0 else math.inf
    disc = v * v - 2.0 * a * c
    if disc < 0.0:
        return math.inf
    q = -(v + math.copysign(math.sqrt(disc), v if v != 0.0 else 1.0))
    roots = [q / a] + ([2.0 * c / q] if q != 0.0 else [])
    pos = [r for r in roots if r > 0.0]
    return min(pos) if pos else math.inf


def motion(ct: Contact, tau: np.ndarray, tau_end: float, pad=None):
    """Configuration u (rear-corner arc length) and its rate at times ``tau`` after a release from rest at u = 0.

    Frictionless rigid block, one degree of freedom: L = m/2 M(u) u'^2 - m g c_z(u), M = |c'|^2 + k theta'^2
    (k = I / m), so u'' = -(g c_z' + M' u'^2 / 2) / M and m/2 M u'^2 + m g c_z (translational + rotational + potential
    energy) is conserved. Flush on a straight part M = 1, theta' = 0: constant acceleration g sin(phi), exact. Around a
    fillet: DOP853 at rtol 1e-10 on the closed-form geometry. ``pad = (u_lo, u_hi, eps)`` (on the flat, block flush): constant
    acceleration set at each entry so the exit speed is eps x the entry speed (either direction). Around a fillet
    the zones are split at :attr:`Contact.breaks` (second derivatives jump there). Returns
    (u, u', events), events (kind, tau, u, u') with kind in pad_in | pad_out | apex (u' = 0)."""
    from scipy.integrate import solve_ivp

    bps = sorted(set(ct.breaks) | (set(map(float, pad[:2])) if pad else set()))
    edges = [-math.inf] + bps + [math.inf]
    kinds = []
    for j in range(len(edges) - 1):
        mid = edges[j + 1] - 1.0 if j == 0 else edges[j] + 1.0 if j == len(edges) - 2 else 0.5 * (edges[j] + edges[j + 1])
        if pad and pad[0] <= mid <= pad[1]:
            kinds.append(("pad",))
        elif ct.in_zone(mid):
            kinds.append(("ode",))
        else:
            kinds.append(("const", G * math.sin(ct.geo(mid)[3])))
    recs, events = [], []
    t, u, v = 0.0, 0.0, 0.0
    key = (tuple(pad[:2]), tau_end) if pad else None
    if key in ct.pre_cache:                     # the motion up to the first pad entry is the same for every eps
        recs, events, t, u, v = ct.pre_cache[key]
        recs, events = list(recs), list(events)
    while t < tau_end:
        fwd = v > 0 or (v == 0 and kinds[bisect.bisect_right(edges, u) - 1][0] == "const"
                        and kinds[bisect.bisect_right(edges, u) - 1][1] > 0)
        j = bisect.bisect_right(edges, u) - 1 if fwd else bisect.bisect_left(edges, u) - 1
        lo, hi, kind = edges[j], edges[j + 1], kinds[j]
        if kind[0] == "pad" and key is not None and key not in ct.pre_cache:
            ct.pre_cache[key] = (list(recs), list(events), t, u, v)
        if kind[0] == "ode":
            def rhs(_, y):
                _, c1, c2, _, t1, t2 = ct.geo(y[0])
                M = c1[0] * c1[0] + c1[1] * c1[1] + ct.k * t1 * t1
                M1 = 2.0 * (c1[0] * c2[0] + c1[1] * c2[1] + ct.k * t1 * t2)
                return [y[1], -(G * c1[1] + 0.5 * M1 * y[1] * y[1]) / M]
            ev_hi = lambda _, y: y[0] - hi
            ev_lo = lambda _, y: y[0] - lo
            ev_v = lambda _, y: y[1]
            ev_hi.terminal, ev_hi.direction, ev_lo.terminal, ev_lo.direction = True, 1, True, -1
            sol = solve_ivp(rhs, (t, tau_end), [u, v], method="DOP853", rtol=1e-10, atol=1e-11,
                            events=[ev_hi, ev_lo, ev_v], dense_output=True)
            recs.append((t, "ode", sol))
            for te_, ye in zip(sol.t_events[2], sol.y_events[2]):
                events.append(("apex", float(te_), float(ye[0]), 0.0))
            if sol.status != 1:
                break
            hit_hi = len(sol.t_events[0]) > 0
            t = float((sol.t_events[0] if hit_hi else sol.t_events[1])[0])
            v = float((sol.y_events[0] if hit_hi else sol.y_events[1])[0][1])
            u = hi if hit_hi else lo
            continue
        if kind[0] == "pad":
            eps = pad[2]
            a = (eps * eps - 1.0) * v * v / (2.0 * (hi - lo)) * (1.0 if v > 0 else -1.0)
            events.append(("pad_in", t, u, v))
        else:
            a = kind[1]
        dt_hi = _first_hit(u, v, a, hi) if hi < math.inf else math.inf
        dt_lo = _first_hit(u, v, a, lo) if lo > -math.inf else math.inf
        dt = min(dt_hi, dt_lo)
        if v * a < 0 and -v / a < min(dt, tau_end - t):
            events.append(("apex", t - v / a, u - v * v / (2 * a), 0.0))
        recs.append((t, "const", (u, v, a)))
        if dt > tau_end - t:
            break
        v = pad[2] * v if kind[0] == "pad" else v + a * dt           # pad: exit speed exactly eps x entry speed
        t, u = t + dt, (hi if dt == dt_hi else lo)
        if kind[0] == "pad":
            events.append(("pad_out", t, u, v))
    starts = np.array([r[0] for r in recs])
    tau = np.asarray(tau, float)
    U, V = np.zeros(tau.size), np.zeros(tau.size)
    for i, x in enumerate(tau):
        if x <= 0:
            continue
        t0, kind, data = recs[int(np.searchsorted(starts, x, side="right") - 1)]
        if kind == "ode":
            U[i], V[i] = data.sol(x)
        else:
            u0, v0, a = data
            dt = x - t0
            U[i], V[i] = u0 + v0 * dt + 0.5 * a * dt * dt, v0 + a * dt
    return U, V, sorted(events, key=lambda e: e[1])


def track_on_path(ct: Contact, t, t0, side: float, pad=None):
    """Track of the block (released at t0 from u = 0); mirrored in x when side < 0. Returns (track, events with
    absolute times)."""
    t = np.asarray(t, float)
    u, ud, events = motion(ct, t - t0, float(t.max()) - t0 + 1e-6, pad)
    st = ct.state(u)
    p, v = st["c"].copy(), st["dc"] * ud[:, None]
    th, wy = st["th"], st["dth"] * ud
    q = np.stack([np.cos(th / 2), np.zeros_like(th), np.sin(th / 2), np.zeros_like(th)], -1)
    w = np.stack([np.zeros_like(wy), wy, np.zeros_like(wy)], -1)
    if side < 0:
        p[:, 0] *= -1
        v[:, 0] *= -1
        q[:, 2] *= -1
        w[:, 1] *= -1
    return Track(p, v, q, w), [(k, t0 + tt, uu, vv) for k, tt, uu, vv in events]


def mirror_static(st: Static, side: float) -> Static:
    if side > 0:
        return st
    q = st.quaternion
    return Static(st.name, st.half, (-st.position[0], st.position[1], st.position[2]),
                  (q[0], q[1], -q[2], q[3]), st.look, st.friction)


# --------------------------------------------------------------------------- #
# F8 sampler
# --------------------------------------------------------------------------- #


def sample(rng, cfg, fmt, sub="ramp"):
    """Geometry (2026-11-12): the SURFACE is primary -- ramp A, flat, ramp C joined by concave fillets of radius
    ``FILLET_DIAG`` x the block's half-diagonal -- and the block rides it in exact two-corner contact
    (:class:`Contact`: flush on straight parts, bridging the fillets, never entering them), with its rotation in the
    energy (:func:`motion`). h = the centre's drop from the release to its flush-on-the-floor height. Every length
    scales with h (block half-size 0.38-0.45 h, pad 0.7-0.9 h, the block flush over the whole pad). Framing: the
    camera is fitted per group (``framed_side_camera``) to the release point .. the top of the block at the highest
    rung's apex, filling ``FRAME_FILL`` of the binding image dimension; draws with a block under
    ``acceptance.min_block_px`` (70) px across at 832 px are rejected."""
    ladder = eps_ladder(cfg, "epsilon_ladder")
    T, fps = fmt["num_frames"], fmt["fps"]
    t = frame_times(fmt)
    t_end = (T - 1) / fps
    te = event_time(rng, fmt, aligned=False)
    side = float(rng.choice([-1.0, 1.0]))
    h = float(rng.uniform(1.0, 2.6))
    s_half = h * float(rng.uniform(0.38, 0.45))
    R_f = FILLET_DIAG * math.sqrt(2.0) * s_half             # surface fillet radius
    th_a, th_c = math.radians(rng.uniform(36, 50)), math.radians(rng.uniform(45, 60))
    L_pad = h * float(rng.uniform(0.7, 0.9))
    d_in, d_out = s_half + h * float(rng.uniform(0.05, 0.12)), s_half + h * float(rng.uniform(0.05, 0.12))
    tl_a, tl_c = R_f * math.tan(th_a / 2), R_f * math.tan(th_c / 2)
    flat = tl_a + d_in + L_pad + d_out + tl_c
    # surface line of ramp A from the rear corner at the release (u = 0) to the floor: centre height s + h there
    L_a = (h + s_half * (1.0 - math.cos(th_a) + math.sin(th_a))) / math.sin(th_a)
    if L_a - tl_a < 2.0 * s_half + 0.03 * h:              # the release flush on ramp A's straight part
        return "release_on_the_fillet"
    e_max = max(ladder)
    x0 = -(L_a * math.cos(th_a) + 0.5 * flat) + float(rng.uniform(-0.4, 0.4))
    rise = (e_max ** 2 * h + 6.0 * h) / math.sin(th_c) + 4.0 * tl_c   # generous: cut at the frame top below
    try:
        surf = Path(np.array([x0, 0.0, L_a * math.sin(th_a)]), [th_a, 0.0, -th_c], [L_a, flat, rise], 0.0, R_f)
    except ValueError as exc:
        return str(exc).replace(" ", "_")
    ct = Contact(surf, s_half)
    s_pad = surf.line_start(1) + d_in                      # pad on the floor from surface arc length s_pad
    pad_u = (s_pad - s_half, s_pad - s_half + L_pad)       # rear-corner u while the CENTRE is over the pad
    # camera: the block's silhouette (FAM.outside: bounding sphere x 1.05) from the release to the highest apex
    rb = s_half * math.sqrt(3.0) * 1.05
    c0 = ct.state(0.0)["c"][0]
    u_top = ct.u_at_height(s_half + e_max ** 2 * h, pad_u[1] + 2 * s_half, surf.bounds[-1])
    c1 = ct.state(u_top)["c"][0]
    x_lo, x_hi = c0[0] - rb, c1[0] + rb
    if side < 0:
        x_lo, x_hi = -x_hi, -x_lo
    cam = framed_side_camera(fmt, x_lo, x_hi, s_half - rb, c1[2] + rb, fill=FRAME_FILL)
    depth = -cam.position[1]
    block_px = 2 * s_half * cam.px_per_m(depth) * 832.0 / cam.resolution[0]
    min_px = float(((cfg.get("acceptance") if hasattr(cfg, "get") else None) or {}).get("min_block_px", 70.0))
    if block_px < min_px:
        return f"block_{block_px:.0f}px_below_{min_px:.0f}"
    zt = cam.position[2] + visible_box(cam, depth)[1]
    if surf.points(surf.bounds[-1])[0][0, 2] < zt + 1.0:
        return "ramp_c_ends_in_frame"
    _, _, ev = motion(ct, np.zeros(0), 30.0, (*pad_u, 1.0))
    t_rel = te - next(e[1] for e in ev if e[0] == "pad_in")
    if t_rel < 0.4:
        return "descent_too_long"
    s_in = math.sqrt(2 * G * h)
    sgn = np.array([side, 1.0, 1.0])
    x_pad = sorted(float(side * surf.points(x)[0][0, 0]) for x in (s_pad, s_pad + L_pad))
    look_ramp, look = dict(LOOKS["ramp"]), pick_look(rng)
    block = Body("A", "cube", s_half, float(rng.uniform(0.5, 2.0)), dict(LOOKS[look]), 1)
    # the ramps: one swept mesh each, from the floor out through the top of the frame (ramp A also behind the start)
    u_a = -(zt + 1.0 - L_a * math.sin(th_a)) / math.sin(th_a)
    u_c = surf.bounds[4] + (zt + 1.0 - float(surf.points(surf.bounds[4])[0][0, 2])) / math.sin(th_c)
    statics = []
    for name, u0, u1 in (("ramp_a", u_a, surf.bounds[2]), ("ramp_c", surf.bounds[3], u_c)):
        prof = [(side * x, z) for x, z in surf.profile(u0, u1)]
        statics.append(Static(name, (0.0, RAMP_DEPTH, 0.5 * RAMP_THICK), (0.0, 0.0, 0.0), look=dict(look_ramp),
                              profile=sorted(prof)))
    statics.append(Static("pad", (0.5 * L_pad, RAMP_DEPTH, 0.002), (0.5 * (x_pad[0] + x_pad[1]), 0.0, -0.0015),
                          (1.0, 0.0, 0.0, 0.0), dict(LOOKS["pad"])))

    def state(tt, u, ud):
        st = ct.state(u)
        return {"t_s": float(tt), "objects": ["A"], "p": [(st["c"][0] * sgn).tolist()],
                "v": [(st["dc"][0] * ud * sgn).tolist()], "w": [[0.0, 0.0, 0.0]]}

    twins, posts = {}, {}
    for eps in ladder:
        name = eps_name(eps)
        tr, events = track_on_path(ct, t, t_rel, side, (*pad_u, eps))
        _, t_in, u_in, v_in = next(e for e in events if e[0] == "pad_in")
        _, t_out, u_out, v_out = next(e for e in events if e[0] == "pad_out")
        apex = next((e for e in events if e[0] == "apex" and e[1] > t_out), None)
        if apex is None or apex[1] > t_end - 0.1:
            return f"{name}_apex_after_clip_end"
        if apex[2] <= pad_u[1]:
            return f"{name}_turns_before_ramp_c"
        legal = eps <= 1.0
        twins[name] = Twin(
            name, {"A": tr}, legal=legal, violated_law=None if legal else "energy",
            violation_magnitude=0.0 if legal else eps, flags={"energy_physical": legal},
            record={"kind": "pad", "epsilon": eps, "pad_entry_speed": v_in, "pad_exit_speed": v_out,
                    "pad_accel": (v_out ** 2 - v_in ** 2) / (2 * L_pad), "t_pad_exit_s": t_out,
                    "t_apex_s": apex[1], "apex_height": float(ct.state(apex[2])["c"][0, 2] - s_half)},
            epsilon=eps, clean_until_s=apex[1],
            event_state={"pre": state(t_in, u_in, v_in), "post": state(t_out, u_out, v_out)})
        posts[eps] = {"A": twins[name].event_state["post"]["v"][0]}
    sc = Scene("f8", "ramp", [block], statics, cam, G, te, "pad", twins, fps,
               constants={"g": G, "h": h, "theta_a_rad": th_a, "theta_c_rad": th_c, "slope_angles_rad": [th_a, -th_c],
                          "half_size": s_half, "fillet_radius": R_f, "mu": 0.0, "z_floor": s_half,
                          "inertia_over_mass": ct.k, "release_time_s": t_rel, "pad_length": L_pad,
                          "pad_x_range": x_pad, "pad_entry_speed": s_in, "direction": side},
               inframe={"start": 0, "stop": T},       # each twin: frame 0 to its clean_until_s
               stop_dose=stop_dose(posts),
               # Bullet check down the straight part of ramp A, until the block's FRONT corner reaches fillet A
               # (Bullet's ramp there is chorded boxes: the box would bump on the chord corners)
               sim_init={"twin": eps_name(ladder[0]), "t0": t_rel,
                         "until": t_rel + math.sqrt(2 * (surf.bounds[1] - 2 * s_half) / (G * math.sin(th_a)))},
               params={"look": look, "start": (c0 * sgn).tolist(), "pad_u": list(pad_u), "d_in": d_in, "d_out": d_out,
                       "fillet_radius": R_f, "block_px_832": block_px})
    return sc


def acceptance(view) -> list:
    """Energy E / m = (|v|^2 + k |w|^2) / 2 + g z (k = I / m: the block turns while it bridges a fillet). Descent:
    E = g (h + z_floor); pad entry speed sqrt(2gh); after the pad up to the apex E = g (eps^2 h + z_floor) with eps
    the twin's epsilon; pad exit speed eps x entry speed; apex height eps^2 h; legal iff eps <= 1."""
    from physlaw.data.families import close
    g, k, err = view.constants["g"], view.k, []
    h, zf, rec = view.constants["h"], view.constants["z_floor"], view.record
    kr = float(view.constants.get("inertia_over_mass", 0.0))
    p, v = view.p["A"], view.v["A"]
    w = view.w.get("A", np.zeros_like(v))
    t = np.arange(len(p)) / view.fps
    v2 = (v ** 2).sum(1) + kr * (w ** 2).sum(1)
    held = t <= view.constants["release_time_s"]
    if np.abs(v[held]).max() != 0.0 or np.abs(p[held] - p[0]).max() != 0.0:
        err.append("block moves before the release")
    if not close(p[0, 2] - zf, h, scale=h):
        err.append("release height != h")
    pre = ~held & (np.arange(len(p)) < k)
    if not close(v2[pre], 2 * g * (zf + h - p[pre, 2]), scale=2 * g * h):
        err.append("(|v|^2 + k |w|^2) != 2 g dz on the descent")
    s_in = math.sqrt(2 * g * h)
    if not close(rec["pad_entry_speed"], s_in, scale=s_in):
        err.append(f"pad entry speed {rec['pad_entry_speed']:.9g} != sqrt(2gh)")
    climb = (t > rec["t_pad_exit_s"]) & (t <= rec["t_apex_s"])
    if not climb.any():
        return err + ["no frame between the pad and the apex"]
    e2 = (v2[climb] + 2 * g * (p[climb, 2] - zf)) / (2 * g * h)      # v2 includes k |w|^2
    eps = math.sqrt(float(np.mean(e2)))
    if not close(e2, np.full_like(e2, rec["epsilon"] ** 2), scale=1.0):
        err.append(f"energy after the pad != eps^2 m g h (eps = {rec['epsilon']})")
    if not close(rec["pad_exit_speed"], rec["epsilon"] * rec["pad_entry_speed"], scale=s_in):
        err.append("pad exit speed != eps x entry speed")
    if not close(rec["apex_height"], rec["epsilon"] ** 2 * h, scale=h):
        err.append(f"apex height {rec['apex_height']:.9g} != eps^2 h")
    if view.violated_law:
        if view.violated_law != "energy" or eps <= 1.0 or not close(eps, view.magnitude, scale=1.0):
            err.append(f"violation {view.violated_law} x{view.magnitude} but realised eps {eps:.9g}")
    elif eps > 1.0 + 1e-9:
        err.append(f"legal twin creates energy: realised eps {eps:.9g}")
    return err
