"""F8: ramp energy exchange (spec §2.3 F8), plus the ramp geometry F6 and F10 reuse.

A frictionless cube (no rotational energy while sliding) is held at rest on a ramp at
centre height h above its floor-level centre height, released at the event frame,
and slides down onto the floor. Ramps are static rotated boxes. Every corner of the
path, including the foot of the ramp, is a circular fillet (approximated by short
boxes when rendered): a sharp concave corner would remove the normal velocity
component and make the legal bottom speed path dependent.

Twins (one ramp geometry per ``pre_segment``):

  legal_slope_a   straight ramp at theta_1                  bottom speed sqrt(2 g h)
  energy_x125     ramp A, speed x1.25 along the whole path  (energy created)
  legal_slope_b   straight ramp at theta_2, same h          same bottom speed, other time
  path_dependent  ramp B, speed x sqrt(sin th2 / sin th1)   (speed depends on the path)
  legal_kink      two slopes (steep then shallow), same h   same bottom speed

Deviation from the spec (recorded in docs/RESUME.md): the spec asks for a shared flat
ledge identical across ALL twins before the branch. A block at rest on a flat
frictionless ledge never starts to slide, and the three ramp geometries are visible
from frame 0 whatever the ledge does, so twins are byte-identical before the release
only within one geometry (labels: ``pre_segment``). The violating twins share the
geometry (and so every pre-event frame) of the legal twin they are contrasted with.

The motion along the path is the ODE  s'' = c g sin(phi(s))  (c = 1 sliding,
1/(1+I/(m r^2)) rolling), integrated per primitive with DOP853 at rtol 1e-12; the
speed it returns matches sqrt(2 c g dz) to ~1e-10, which the acceptance tests check.
A violating twin with factor k is the legal motion in rescaled time,
s_k(t) = s(t_r + k (t - t_r)), so its speed is k times the legal speed at every point.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from physlaw.data.families import (G, LOOKS, Body, Scene, Static, Track, Twin, event_time, frame_times,
                                   pick_look, quat_y, side_camera, visible_box)

SUBFAMILIES = {"ramp": 250}
FILLET = 0.6               # fillet radius of the block-centre path (m)
RAMP_THICK = 2.0           # ramp slabs extend this far below their surface (into the floor)
RAMP_DEPTH = 0.75          # half-depth (y) of the ramp slabs
MARGIN = 0.15


# --------------------------------------------------------------------------- #
# path: lines joined by concave circular fillets, ending in a floor run-out
# --------------------------------------------------------------------------- #


def _d(phi):   # unit tangent of a path descending to +x at angle phi
    return np.array([math.cos(phi), 0.0, -math.sin(phi)])


def _n(phi):   # upward unit normal
    return np.array([math.sin(phi), 0.0, math.cos(phi)])


@dataclass
class Path:
    """Centre path of a body touching a ramp surface at distance ``offset`` (its half-size)."""
    start: np.ndarray
    angles: list                  # descending slope angles of the straight parts, the last is 0 (floor)
    drops: list                   # centre height lost on each sloped part (sum = h)
    offset: float
    fillet: float = FILLET
    prims: list = field(default_factory=list)   # ("line", P, phi, L) | ("arc", C, phi0, phi1, rho)

    def __post_init__(self):
        V = [np.asarray(self.start, float)]
        for phi, dz in zip(self.angles[:-1], self.drops):
            V.append(V[-1] + _d(phi) * dz / math.sin(phi))
        tl = [0.0] + [self.fillet * math.tan(0.5 * (a0 - a1)) for a0, a1 in zip(self.angles[:-1], self.angles[1:])]
        P = V[0]
        for i, phi in enumerate(self.angles):
            if i == len(self.angles) - 1:
                self.prims.append(("line", P, phi, 60.0))
                break
            L = np.linalg.norm(V[i + 1] - V[i]) - tl[i] - tl[i + 1]
            if L < -1e-12:
                raise ValueError("fillets overlap")
            self.prims.append(("line", P, phi, L))
            P = P + _d(phi) * L
            a1 = self.angles[i + 1]
            C = P + self.fillet * _n(phi)
            self.prims.append(("arc", C, phi, a1, self.fillet))
            P = C - self.fillet * _n(a1)
        self.bounds = np.cumsum([0.0] + [self.length_of(p) for p in self.prims])

    @staticmethod
    def length_of(p):
        return p[3] if p[0] == "line" else p[4] * (p[2] - p[3])

    @property
    def bottom_s(self) -> float:
        """Arc length at the start of the floor run-out."""
        return float(self.bounds[-2])

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

    def statics(self, prefix: str, look: dict, chords: int = 6) -> list:
        """Static boxes whose top faces trace the surface (centre path shifted down by ``offset``)."""
        out = []
        surf = []                                  # (a, b, phi) surface chords
        for j, p in enumerate(self.prims[:-1]):    # the run-out is the floor itself
            if p[0] == "line":
                a = p[1] - self.offset * _n(p[2])
                if j == 0:                         # extend behind the start so the body sits fully on it
                    a = a - _d(p[2]) * (self.offset * 1.5 + 0.3)
                surf.append((a, p[1] - self.offset * _n(p[2]) + _d(p[2]) * p[3], p[2]))
            else:
                C, f0, f1, rho = p[1], p[2], p[3], p[4]
                fs = np.linspace(f0, f1, chords + 1)
                R = rho + self.offset
                for u, w in zip(fs[:-1], fs[1:]):
                    surf.append((C - R * _n(u), C - R * _n(w), 0.5 * (u + w)))
        for i, (a, b, phi) in enumerate(surf):
            L = float(np.linalg.norm(b - a))
            mid = 0.5 * (a + b) - 0.5 * RAMP_THICK * _n(phi)
            out.append(Static(f"{prefix}_{i}", (0.5 * L + 0.01, RAMP_DEPTH, 0.5 * RAMP_THICK), tuple(mid),
                              quat_y(phi), dict(look)))
        return out


def slide(path: Path, t: np.ndarray, t0: float, c: float = 1.0, k: float = 1.0, t_extra: float = 0.0):
    """Arc length, speed and phi along ``path`` at times t, released from rest at t0.

    ``k`` rescales time (violating twins); ``c`` scales gravity along the path (rolling).
    Returns (s, sdot, phi, t_bottom, v_bottom): when the run-out starts, and the speed there.
    """
    from scipy.integrate import solve_ivp

    t = np.asarray(t, float)
    tau_end = k * (float(t.max()) - t0) + t_extra + 1e-6
    seg_t, seg_sol, v_bottom = [0.0], [], math.nan
    y = np.array([0.0, 0.0])
    tau = 0.0
    for i, p in enumerate(path.prims):
        s_end = path.bounds[i + 1]
        if p[0] == "line":
            seg_sol.append(("line", tau, y.copy(), c * G * math.sin(p[2])))
            a = c * G * math.sin(p[2])
            if i == len(path.prims) - 1:
                v_bottom = float(y[1])
                break
            L = s_end - y[0]
            dt = (-y[1] + math.sqrt(y[1] ** 2 + 2 * a * L)) / a if a > 0 else L / y[1]
            tau += dt
            y = np.array([s_end, y[1] + a * dt])
        else:
            f0, rho = p[2], p[4]
            s0 = path.bounds[i]

            def rhs(_, yy, f0=f0, rho=rho, s0=s0):
                return [yy[1], c * G * math.sin(f0 - (yy[0] - s0) / rho)]

            ev = lambda _, yy, s_end=s_end: yy[0] - s_end
            ev.terminal, ev.direction = True, 1
            sol = solve_ivp(rhs, (tau, tau + 100.0), y, method="DOP853", rtol=1e-12, atol=1e-12,
                            events=ev, dense_output=True)
            seg_sol.append(("arc", tau, sol))
            tau = float(sol.t_events[0][0])
            y = np.array([s_end, float(sol.y_events[0][0][1])])
        seg_t.append(tau)
        if tau > tau_end + 10:
            break
    t_bottom_tau = seg_t[len(path.prims) - 1] if len(seg_t) >= len(path.prims) else math.inf

    def at(tau):
        j = int(np.clip(np.searchsorted(seg_t, tau, side="right") - 1, 0, len(seg_sol) - 1))
        rec = seg_sol[j]
        if rec[0] == "line":
            _, t_s, y0, a = rec
            dt = tau - t_s
            return y0[0] + y0[1] * dt + 0.5 * a * dt * dt, y0[1] + a * dt
        yy = rec[2].sol(tau)
        return float(yy[0]), float(yy[1])

    s = np.zeros(t.size)
    sd = np.zeros(t.size)
    for j, ti in enumerate(t):
        if ti <= t0:
            continue
        s[j], sd[j] = at(k * (ti - t0))
        sd[j] *= k
    return s, sd, np.array([path.phi(x) for x in s]), t0 + t_bottom_tau / k, k * v_bottom


def track_on_path(path: Path, t, t0, side: float, c=1.0, k=1.0, roll_radius: float | None = None):
    """Track of a body sliding (or rolling, with ``roll_radius``) along ``path``; mirrored when side < 0.

    Returns (track, t_bottom, v_bottom)."""
    s, sd, phi, tb, vb = slide(path, t, t0, c=c, k=k)
    p = np.array([path.point(x) for x in s])
    d = np.stack([np.cos(phi), np.zeros_like(phi), -np.sin(phi)], -1)
    v = d * sd[:, None]
    kappa = np.array([path.curvature(x) for x in s])
    if roll_radius is None:                       # sliding: the body turns with the surface
        ang, wy = phi, -kappa * sd
    else:                                         # rolling without slipping: angle = s / r (+ tilt)
        ang, wy = s / roll_radius, sd / roll_radius
    q = np.stack([np.cos(ang / 2), np.zeros_like(ang), np.sin(ang / 2), np.zeros_like(ang)], -1)
    w = np.stack([np.zeros_like(wy), wy, np.zeros_like(wy)], -1)
    if side < 0:
        p[:, 0] *= -1
        v[:, 0] *= -1
        q[:, 2] *= -1
        w[:, 1] *= -1
    return Track(p, v, q, w), tb, vb


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
    cam = side_camera(fmt)
    hw, hh = visible_box(cam, 20.0)
    zt = cam.position[2] + hh
    T, fps = fmt["num_frames"], fmt["fps"]
    t = frame_times(fmt)
    te = event_time(rng, fmt, aligned=True)
    kr = int(round(te * fps))
    side = float(rng.choice([-1.0, 1.0]))
    s_half = float(rng.uniform(0.25, 0.35))
    h = float(rng.uniform(1.2, 3.0))
    d2r = math.radians
    th1 = d2r(rng.uniform(25, 50))
    th2 = th1 + float(rng.choice([-1, 1])) * d2r(rng.uniform(12, 22))
    if not d2r(18) <= th2 <= d2r(62):
        th2 = th1 - (th2 - th1)
    tk1 = d2r(rng.uniform(45, 62))
    tk2 = d2r(rng.uniform(15, math.degrees(tk1) - 20))
    frac = float(rng.uniform(0.4, 0.6))
    x0 = -hw + 1.0 + s_half
    start = np.array([x0, 0.0, s_half + h])
    if start[2] + s_half * 1.5 > zt - MARGIN:
        return "top_above_frame"
    try:
        paths = {"a": Path(start, [th1, 0.0], [h], s_half), "b": Path(start, [th2, 0.0], [h], s_half),
                 "k": Path(start, [tk1, tk2, 0.0], [frac * h, (1 - frac) * h], s_half)}
    except ValueError as exc:
        return str(exc).replace(" ", "_")
    look_ramp, look = dict(LOOKS["ramp"]), pick_look(rng)
    block = Body("A", "cube", s_half, float(rng.uniform(0.5, 2.0)), dict(LOOKS[look]), 1)
    statics = {g: [mirror_static(st, side) for st in P.statics(f"ramp_{g}", look_ramp)] for g, P in paths.items()}
    k_path = math.sqrt(math.sin(th2) / math.sin(th1))
    if abs(k_path - 1) < 0.08:
        return "path_dependence_too_small"
    v_bottom = math.sqrt(2 * G * h)
    spec = (("legal_slope_a", "a", 1.0, None), ("energy_x125", "a", 1.25, "energy"),
            ("legal_slope_b", "b", 1.0, None), ("path_dependent", "b", k_path, "path_independence"),
            ("legal_kink", "k", 1.0, None))
    twins = {}
    for name, g, k, law in spec:
        tr, tb, vb = track_on_path(paths[g], t, te, side, k=k)
        # after the run-out starts the body must stay in view for 0.25 s
        fb = int(math.ceil(tb * fps))
        if fb + int(0.25 * fps) >= T:
            return f"{name}_reaches_bottom_after_clip_end"
        x_late = abs(tr.p[fb + int(0.25 * fps), 0])
        if x_late + s_half * 1.5 > hw - MARGIN:
            return f"{name}_runout_leaves_frame"
        angs = paths[g].angles
        twins[name] = Twin(name, {"A": tr}, pre_segment=g, legal=law is None, violated_law=law,
                           violation_magnitude=k if law else 0.0, statics=[st.name for st in statics[g]],
                           flags={"energy_physical": k <= 1.0},
                           constants={"slope_angles_rad": [float(a) for a in angs[:-1]], "path": g,
                                      "t_bottom_s": float(tb), "v_bottom_legal": v_bottom},
                           record={"kind": "release", "speed_factor": k, "t_bottom_s": float(tb),
                                   "v_bottom": vb})
    sc = Scene("f8", "ramp", [block], sum(statics.values(), []), cam, G, te, "release", twins, fps,
               constants={"g": G, "h": h, "theta1_rad": th1, "theta2_rad": th2, "kink_angles_rad": [tk1, tk2],
                          "kink_drop_fraction": frac, "half_size": s_half, "fillet_radius": FILLET,
                          "mu": 0.0, "z_floor": s_half, "release_frame": kr, "release_time_s": te,
                          "v_bottom_legal": v_bottom, "direction": side},
               inframe={"start": 0, "stop": min(int(math.ceil(min(tw.constants["t_bottom_s"] for tw in twins.values()) * fps)), T)},
               # Bullet check down the straight part of ramp A (2 mm there); over the chorded
               # fillet the box bumps on the chord corners (5-15 cm), which says nothing about the geometry.
               sim_init={"twin": "legal_slope_a", "t0": te,
                         "until": te + math.sqrt(2 * paths["a"].bounds[1] / (G * math.sin(th1)))},
               params={"look": look, "start": start.tolist()})
    return sc


def acceptance(view) -> list:
    """Energy along the whole path: |v|^2 = k^2 2 g (z_release - z), bottom speed = k sqrt(2 g h)."""
    from physlaw.data.families import close
    g, k, err = view.constants["g"], view.k, []
    p, v = view.p["A"], view.v["A"]
    h = view.constants["h"]
    factor = view.record.get("speed_factor", 1.0)
    want_f = view.magnitude if view.violated_law else 1.0
    if not close(factor, want_f, scale=1.0):
        err.append(f"speed factor {factor} != {want_f}")
    dz = p[k - 1, 2] - p[k:, 2]
    v2 = (v[k:] ** 2).sum(1)
    if not close(v2, factor ** 2 * 2 * g * dz, scale=2 * g * h):
        err.append("|v|^2 != k^2 2 g dz along the path")
    if not close(view.record["v_bottom"], factor * math.sqrt(2 * g * h), scale=math.sqrt(2 * g * h)):
        err.append(f"bottom speed {view.record['v_bottom']:.9g} != {factor} sqrt(2gh)")
    if not close(p[k - 1, 2] - view.constants["z_floor"], h, scale=h):
        err.append("release height != h")
    if np.abs(v[:k]).max() != 0.0:
        err.append("block moves before the release")
    return err
