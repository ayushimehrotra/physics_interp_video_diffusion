"""F9: pendulum (spec §2.3 F9), closed form. Side view, gravity -9.81 z.

No hinge constraint is needed (Kubric's PyBullet bodies refuse every constraint): the bob (a sphere) and the rod
(a thin box, an ``aux`` body: massless in the physics, labelled ``aux``) are keyframed from the exact pendulum
motion. The rod is massless and the bob a uniform sphere rigidly fixed to it, so the pivot inertia is
m (L^2 + 2 r^2 / 5) and w0^2 = g L / (L^2 + 2 r^2 / 5) for every bob mass; kappa = 1 + 2 r^2 / (5 L^2) = I_pivot /
(m L^2). Large-angle motion is exact (Jacobi elliptic functions): with apex amplitude A at t_a,
sin(theta / 2) = sin(A / 2) sn(K - w0 (t - t_a) | m), m = sin^2(A / 2), period 4 K(m) / w0.

  pair   two pendulums side by side (same length, amplitude and phase; bobs of different density, the heavier
         is metal), swinging from the first frame; the branch is on a frame at a common apex (theta' = 0, so a
         period change keeps theta and theta' continuous). Twins:
           legal           unchanged
           period_x<k>     both periods x k after the branch (``twins.period_factors``, default 0.8, 1.25):
                           theta_k(t) = theta(t_e + (t - t_e) / k), same amplitude;  violated_law "oscillation"
           heavy_x<k>      the heavier bob's period x k (``twins.heavy_period_factor``, default 0.8): mass changes
                           the period; violated_law "oscillation" (statistic mass_period_ratio)
         magnitude = the period factor. A faster swing at the same amplitude creates energy (flagged).
  swing  the energy ladder (spec §5.0.2). One pendulum held at amplitude A0 and released; at the bottom of the
         swing it crosses a damper zone |theta| <= delta (a pad strip under the bottom, rendered identically in every
         twin). Event = the bob entering the damper (between frames, event_kind "swing"). Inside the zone a
         constant torque does work (eps^2 - 1) m g h0 (h0 = L (1 - cos A0), the apex height above the lowest
         point), so the next apex is at eps^2 h0: eps = sqrt(h_apex_after / h_apex_before). Twins eps<e> for e in
         ``twins.swing_ladder`` (default 0.5, 0.55, ..., 1.5); e <= 1 legal, e > 1 violated_law "energy" (magnitude e).
         Later damper crossings take ``twins.swing_eps_later`` (default 0.9), after the clean window, which ends at
         the next damper entry (clean_until_s). event_state pre / post = the damper entry / exit (exact); the
         record's apex_states pre / post = the release apex and the first apex after the damper (v = 0; the energy
         law reads these, so no g enters); stop_dose = the eps at which the exit speed vanishes when inside the
         ladder, else NaN.

Constants: g, pendulum_length, bob_radius, amplitude_rad, pivots {bob: [x, y, z]}, bobs, omega0, period_legal_s,
kappa, p_bottom (swing: the lowest bob position), damper_half_angle_rad (swing).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import ellipj, ellipk, ellipkinc

from physlaw.data.families import (G, LOOKS, Body, Scene, Static, Track, Twin, eps_ladder, eps_name, event_time,
                                   framed_side_camera, frame_times, pick_look, quat_y_array, set_clean_until)

SUBFAMILIES = {"pair": 150, "swing": 150}
# Framing: the camera is fitted per group (``families.framed_side_camera``) to the pendulums' whole range of motion
# (every twin; swing: up to the largest rung's apex) plus the support beam and the damper, filling FILL of the binding
# image dimension; the bob is large relative to the rod (BOB_FRAC) so it is >= MIN_BOB_FRAC of the image width.
PIVOT_Z = 3.0             # pivot height (m); no floor in view
LENGTH = (0.8, 1.2)       # rod length L (pivot to bob centre)
BOB_FRAC = (0.15, 0.2)    # bob radius / L
FILL = 0.85
MIN_BOB_FRAC = 70.0 / 832.0   # bob diameter / image width (70 px at the master's 832)
ROD_HALF = 0.02           # rod half-thickness
ROD_DENSITY = 1e-3        # the rod is massless in the physics; a token density for the labels
BEAM_TOP = 0.05 + ROD_HALF + 0.05    # support beam top above the pivot
DAMPER_DROP = 0.083       # damper bottom below the bob's lowest point
PERIOD_FACTORS = (0.8, 1.25)
HEAVY_FACTOR = 0.8
EPS_LATER = 0.9


# --------------------------------------------------------------------------- #
# exact pendulum motion
# --------------------------------------------------------------------------- #


def omega0(L: float, r: float) -> float:
    return math.sqrt(G * L / (L * L + 0.4 * r * r))


def kappa(L: float, r: float) -> float:
    return 1.0 + 0.4 * r * r / (L * L)


def period(amp: float, w0: float) -> float:
    return 4.0 * float(ellipk(math.sin(abs(amp) / 2) ** 2)) / w0


def free_swing(t, t_apex: float, amp: float, w0: float):
    """(theta, theta') of the free pendulum whose apex is theta = amp at t_apex (any t)."""
    k = math.sin(abs(amp) / 2)
    m = k * k
    sn, cn, _, _ = ellipj(float(ellipk(m)) - w0 * (np.asarray(t, float) - t_apex), m)
    s = math.copysign(1.0, amp)
    return s * 2.0 * np.arcsin(k * sn), -s * 2.0 * k * w0 * cn


def time_to_angle(amp: float, ang: float, w0: float) -> float:
    """Time from an apex of amplitude |amp| down to |theta| = ang (0 <= ang < |amp|)."""
    k = math.sin(abs(amp) / 2)
    m = k * k
    return (float(ellipk(m)) - float(ellipkinc(math.asin(math.sin(ang / 2) / k), m))) / w0


def pose(theta, thetad, pivot, L, r):
    """Bob and rod tracks for angle theta from the downward vertical (positive towards +x)."""
    th, td = np.asarray(theta, float), np.asarray(thetad, float)
    e = np.stack([np.sin(th), np.zeros_like(th), -np.cos(th)], -1)
    et = np.stack([np.cos(th), np.zeros_like(th), np.sin(th)], -1)
    q = quat_y_array(-th)
    w = np.stack([np.zeros_like(td), -td, np.zeros_like(td)], -1)
    Lc = 0.5 * (L - r)
    piv = np.asarray(pivot, float)
    return (Track(piv + L * e, L * td[:, None] * et, q, w),
            Track(piv + Lc * e, Lc * td[:, None] * et, q.copy(), w.copy()))


def swing_motion(t, t_rel: float, amp0: float, delta: float, w0: float, eps_of, L: float, r: float):
    """theta, theta' at times t of a pendulum held at amp0 until t_rel, then swinging through a damper zone
    |theta| <= delta; crossing i takes the energy above the bottom from E to eps_of(i)^2 E. Returns
    (theta, theta', crossings) with crossings [(t_in, t_out, theta'_in, theta'_out, t_next_apex)], or a reason."""
    t = np.asarray(t, float)
    t_end = float(t.max())
    segs = [("hold", -math.inf, t_rel, amp0)]
    cross, t_a, A, i, t_prev = [], t_rel, amp0, 0, t_rel
    while True:
        t_in = t_a + time_to_angle(A, delta, w0)
        segs.append(("free", t_prev, min(t_in, t_end + 1.0), (t_a, A)))
        if t_in > t_end:
            break
        dirn = -math.copysign(1.0, A)
        eps = float(eps_of(i))
        c = (eps * eps - 1.0) * w0 * w0 * (1.0 - math.cos(A)) / (2.0 * delta) * dirn
        _, td_in = free_swing([t_in], t_a, A, w0)
        hit = lambda _, y: y[0] - dirn * delta
        hit.terminal, hit.direction = True, dirn
        stall = lambda _, y: y[1]
        stall.terminal = True
        sol = solve_ivp(lambda _, y: [y[1], -w0 * w0 * math.sin(y[0]) + c], (t_in, t_in + 20.0),
                        [-dirn * delta, float(td_in[0])], method="DOP853", rtol=1e-12, atol=1e-12,
                        events=[hit, stall], dense_output=True)
        if not len(sol.t_events[0]) or (len(sol.t_events[1]) and sol.t_events[1][0] < sol.t_events[0][0]):
            return f"stalls_in_damper_crossing_{i}"
        t_out, td_out = float(sol.t_events[0][0]), float(sol.y_events[0][0][1])
        segs.append(("damper", t_in, t_out, sol))
        cosA = 1.0 - eps * eps * (1.0 - math.cos(A))
        if cosA <= -0.5:
            return f"swings_over_120deg_crossing_{i}"
        A = dirn * math.acos(cosA)
        t_a = t_out + time_to_angle(A, delta, w0)
        cross.append((t_in, t_out, float(td_in[0]), td_out, t_a))
        t_prev = t_out
        if t_out > t_end:
            break
        i += 1
    th, td = np.zeros(t.size), np.zeros(t.size)
    for kind, lo, hi, data in segs:
        m = (t > lo) & (t <= hi) if kind != "hold" else (t <= hi)
        if not m.any():
            continue
        if kind == "hold":
            th[m], td[m] = data, 0.0
        elif kind == "free":
            th[m], td[m] = free_swing(t[m], data[0], data[1], w0)
        else:
            th[m], td[m] = data.sol(t[m])
    return th, td, cross


# --------------------------------------------------------------------------- #
# samplers
# --------------------------------------------------------------------------- #


def sample(rng, cfg, fmt, sub):
    return {"pair": _pair, "swing": _swing}[sub](rng, cfg, fmt)


def _twins_cfg(cfg) -> dict:
    return (cfg.get("twins") if hasattr(cfg, "get") else None) or {}


def _frame(fmt, x_lo, x_hi, z_lo, z_hi, r):
    """The group's camera (the box filling FILL of the view), or a rejection reason when the bob would be small."""
    cam = framed_side_camera(fmt, x_lo, x_hi, z_lo, z_hi, FILL)
    depth = -cam.position[1]
    if 2.0 * r * cam.px_per_m(depth) < MIN_BOB_FRAC * cam.resolution[0]:
        return "bob_too_small_in_view"
    return cam


def _rod(name, L, r, seg):
    return Body(name, "box", 0.5 * (L - r), ROD_DENSITY, dict(LOOKS["neutral"]), seg,
                half=(ROD_HALF, ROD_HALF, 0.5 * (L - r)), aux=True)


def _pair(rng, cfg, fmt):
    tc = _twins_cfg(cfg)
    factors = [float(x) for x in (tc.get("period_factors") or PERIOD_FACTORS)]
    kh = float(tc.get("heavy_period_factor") or HEAVY_FACTOR)
    zp = PIVOT_Z
    T, fps = fmt["num_frames"], fmt["fps"]
    t = frame_times(fmt)
    te = event_time(rng, fmt, aligned=True)
    k = int(round(te * fps)) + 1
    L = float(rng.uniform(*LENGTH))
    r = L * float(rng.uniform(*BOB_FRAC))
    amp = math.radians(rng.uniform(20.0, 30.0)) * float(rng.choice([-1.0, 1.0]))
    w0 = omega0(L, r)
    Tp = period(amp, w0)
    reach = L * math.sin(abs(amp)) + r
    sep = 2.0 * reach + float(rng.uniform(0.1, 0.3))     # bob surfaces >= that apart in any phase
    beam_half = sep / 2 + 0.3
    # both bobs swing at amplitude amp in every twin (the period edits keep it): their range plus the beam
    x_half = max(sep / 2 + reach, beam_half)
    cam = _frame(fmt, -x_half, x_half, zp - L - r, zp + BEAM_TOP, r)
    if isinstance(cam, str):
        return cam
    heavy = str(rng.choice(["A", "B"]))
    rho_lo = float(np.exp(rng.uniform(np.log(0.5), np.log(1.5))))
    ratio = float(np.exp(rng.uniform(np.log(2.0), np.log(5.0))))
    light_look = pick_look(rng, exclude=("metal",))
    looks = {n: ("metal" if n == heavy else light_look) for n in "AB"}
    pivots = {"A": np.array([-sep / 2, 0.0, zp]), "B": np.array([sep / 2, 0.0, zp])}
    bodies = [Body(n, "sphere", r, rho_lo * (ratio if n == heavy else 1.0), dict(LOOKS[looks[n]]), i + 1)
              for i, n in enumerate("AB")]
    bodies += [_rod("R" + n, L, r, i + 3) for i, n in enumerate("AB")]
    th_l, td_l = free_swing(t, te, amp, w0)
    pre = t <= te + 1e-12

    def tracks(fac):
        out = {}
        for n in "AB":
            f = fac.get(n, 1.0)
            th_f, td_f = free_swing(t, te, amp, w0 / f)
            th, td = np.where(pre, th_l, th_f), np.where(pre, td_l, td_f)
            out[n], out["R" + n] = pose(th, td, pivots[n], L, r)
        return out

    spec = [("legal", {}, None, 0.0)]
    spec += [(f"period_x{f:g}", {"A": f, "B": f}, "oscillation", f) for f in factors]
    spec.append((f"heavy_x{kh:g}", {heavy: kh}, "oscillation", kh))
    # exact event states: the branch is at a common apex (theta = amp, theta' = 0 on both sides; the finite-difference
    # limits of generate.event_limits assume a piecewise-constant acceleration, which a pendulum does not have)
    apex = {}
    for n in "AB":
        bob, rod = pose(np.array([amp]), np.array([0.0]), pivots[n], L, r)
        apex[n], apex["R" + n] = bob, rod
    names = ["A", "B", "RA", "RB"]
    st_apex = {"t_s": te, "objects": names, "p": [apex[n].p[0].tolist() for n in names],
               "v": [apex[n].v[0].tolist() for n in names], "w": [apex[n].w[0].tolist() for n in names]}
    twins = {}
    for name, fac, law, mag in spec:
        twins[name] = Twin(name, tracks(fac), legal=law is None, violated_law=law, violation_magnitude=mag,
                           flags={"energy_physical": all(f >= 1.0 for f in fac.values())},
                           record={"kind": "branch", "period_factor": {n: fac.get(n, 1.0) for n in "AB"},
                                   "period_post_s": {n: Tp * fac.get(n, 1.0) for n in "AB"}, "heavier": heavy},
                           event_state={"pre": st_apex, "post": st_apex})
    beam = Static("beam", (beam_half, 0.12, 0.05), (0.0, 0.0, zp + 0.05 + ROD_HALF), look=dict(LOOKS["ramp"]))
    sc = Scene("f9", "pair", bodies, [beam], cam, G, te, "branch", twins, fps,
                 constants={"g": G, "pendulum_length": L, "bob_radius": r, "amplitude_rad": abs(amp),
                            "pivots": {n: pivots[n].tolist() for n in "AB"}, "bobs": ["A", "B"], "omega0": w0,
                            "period_legal_s": Tp, "kappa": kappa(L, r), "heavier": heavy,
                            "mass_ratio_heavy_over_light": ratio},
                 look={"floor": False}, inframe={"start": 0, "stop": T},
                 params={"looks": looks, "density_light": rho_lo, "separation": sep, "phase_sign": math.copysign(1, amp)})
    return set_clean_until(sc) or sc      # no secondary event: the bobs swing in frame to the clip end


def _swing(rng, cfg, fmt):
    tc = _twins_cfg(cfg)
    ladder = eps_ladder(cfg, "swing_ladder")
    e_later = float(tc.get("swing_eps_later") or EPS_LATER)
    zp = PIVOT_Z
    T, fps = fmt["num_frames"], fmt["fps"]
    t = frame_times(fmt)
    t_end = float(t[-1])
    te = event_time(rng, fmt, aligned=False)
    L = float(rng.uniform(*LENGTH))
    r = L * float(rng.uniform(*BOB_FRAC))
    amp0 = math.radians(rng.uniform(25.0, 35.0)) * float(rng.choice([-1.0, 1.0]))
    delta = abs(amp0) * float(rng.uniform(0.2, 0.3))
    w0 = omega0(L, r)
    t_rel = te - time_to_angle(amp0, delta, w0)
    if t_rel < 0.5:
        return "release_too_early"
    x0 = float(rng.uniform(-0.6, 0.6))
    pivot = np.array([x0, 0.0, zp])
    z_bottom = zp - L
    a_max = math.acos(max(-1.0, 1.0 - max(ladder) ** 2 * (1.0 - math.cos(amp0))))
    # framing: the release apex on one side, the largest rung's first apex on the other (later crossings take
    # eps_later < 1: smaller), the beam above the pivot and the damper below the bottom
    sgn = math.copysign(1.0, amp0)
    reach_rel = L * math.sin(abs(amp0)) + r
    reach_max = L * math.sin(min(a_max, math.pi / 2)) + r
    x_rel, x_far = x0 + sgn * max(reach_rel, 0.4), x0 - sgn * max(reach_max, 0.4)
    cam = _frame(fmt, min(x_rel, x_far), max(x_rel, x_far), zp - L - r - DAMPER_DROP,
                 max(zp + BEAM_TOP, zp - L * math.cos(a_max) + r), r)
    if isinstance(cam, str):
        return cam
    look = pick_look(rng)
    bob = Body("A", "sphere", r, float(rng.uniform(0.8, 3.0)), dict(LOOKS[look]), 1)
    rod = _rod("R", L, r, 2)
    h0 = L * (1.0 - math.cos(amp0))
    names = ["A", "R"]
    twins = {}
    for eps in ladder:
        name = eps_name(eps)
        res = swing_motion(t, t_rel, amp0, delta, w0, lambda i, e=eps: e if i == 0 else e_later, L, r)
        if isinstance(res, str):
            return f"{name}_{res}"
        th, td, cross = res
        t_in, t_out, td_in, td_out, t_apex = cross[0]
        if abs(t_in - te) > 1e-9:
            return "event_time_mismatch"
        clean = cross[1][0] if len(cross) > 1 else t_end
        if t_apex > t_end - 0.1:
            return f"{name}_apex_after_clip_end"
        bob_tr, rod_tr = pose(th, td, pivot, L, r)

        def st(tt, tht, tdt):
            b, rr = pose(np.array([tht]), np.array([tdt]), pivot, L, r)
            return {"t_s": float(tt), "objects": names, "p": [b.p[0].tolist(), rr.p[0].tolist()],
                    "v": [b.v[0].tolist(), rr.v[0].tolist()], "w": [b.w[0].tolist(), rr.w[0].tolist()]}

        dirn = -math.copysign(1.0, amp0)
        amp1 = dirn * math.acos(1.0 - eps * eps * (1.0 - math.cos(amp0)))
        legal = eps <= 1.0
        twins[name] = Twin(
            name, {"A": bob_tr, "R": rod_tr}, legal=legal, violated_law=None if legal else "energy",
            violation_magnitude=0.0 if legal else eps, flags={"energy_physical": legal},
            record={"kind": "swing", "epsilon": eps, "t_damper_exit_s": t_out, "t_apex_after_s": t_apex,
                    "apex_height_before": h0, "apex_height_after": eps * eps * h0,
                    "damper_entry_speed": abs(td_in) * L, "damper_exit_speed": abs(td_out) * L,
                    "eps_later": e_later, "damper_crossings_s": [[c[0], c[1]] for c in cross],
                    # the law's states (v = 0): the release apex and the first apex after the damper
                    "apex_states": {"pre": st(t_rel, amp0, 0.0), "post": st(t_apex, amp1, 0.0)}},
            epsilon=eps, clean_until_s=min(clean, t_end),
            event_state={"pre": st(t_in, -dirn * delta, td_in), "post": st(t_out, dirn * delta, td_out)})
    # exit speed^2 ~ eps^2 (1 - cos A0) - (1 - cos delta): zero at eps* = sqrt((1 - cos delta) / (1 - cos A0))
    e_star = math.sqrt((1.0 - math.cos(delta)) / (1.0 - math.cos(amp0)))
    dose = e_star if min(ladder) < e_star < max(ladder) else math.nan
    chord = L * math.sin(delta) + 0.5 * r
    damper = Static("damper", (chord, 0.25, 0.04), (x0, 0.0, z_bottom - r - 0.043), look=dict(LOOKS["pad"]))
    beam = Static("beam", (0.4, 0.12, 0.05), (x0, 0.0, zp + 0.05 + ROD_HALF), look=dict(LOOKS["ramp"]))
    return Scene("f9", "swing", [bob, rod], [damper, beam], cam, G, te, "swing", twins, fps,
                 constants={"g": G, "pendulum_length": L, "bob_radius": r, "amplitude_rad": abs(amp0),
                            "pivots": {"A": pivot.tolist()}, "bobs": ["A"], "omega0": w0,
                            "period_legal_s": period(amp0, w0), "kappa": kappa(L, r),
                            "p_bottom": [x0, 0.0, z_bottom], "damper_half_angle_rad": delta,
                            "release_time_s": t_rel, "apex_height": h0},
                 stop_dose={"A": dose}, look={"floor": False}, inframe={"start": 0, "stop": T},
                 params={"look": look, "release_sign": math.copysign(1, amp0)})


# --------------------------------------------------------------------------- #
# acceptance (spec §2.4)
# --------------------------------------------------------------------------- #


def _angles(view, n):
    """(theta, theta', |p - pivot|) per frame of bob n."""
    piv = np.asarray(view.constants["pivots"][n], float)
    rel = view.p[n] - piv
    th = np.arctan2(rel[:, 0], -rel[:, 2])
    et = np.stack([np.cos(th), np.zeros_like(th), np.sin(th)], -1)
    L = np.linalg.norm(rel, axis=1)
    return th, (view.v[n] * et).sum(1) / L, L


def acceptance(view) -> list:
    """Bob on a circle of radius L about its pivot. Energy height above the bottom, in units of L,
    e = kappa L theta'^2 / (2 g) + 1 - cos theta: pair: = 1 - cos A on every frame, with theta' x period factor k
    after the branch (legal: k = 1); swing: held, then = 1 - cos A0 before the damper and eps^2 (1 - cos A0) after
    it up to the clean window's end, realised eps vs the label (legal iff <= 1)."""
    from physlaw.data.families import close
    c, k, err = view.constants, view.k, []
    g, L, kap = c["g"], c["pendulum_length"], c["kappa"]
    e0 = 1.0 - math.cos(c["amplitude_rad"])
    t = np.arange(len(view.p["A"])) / view.fps
    for n in c["bobs"]:
        th, td, rad = _angles(view, n)
        if not close(rad, np.full_like(rad, L), scale=L):
            err.append(f"{n}: bob not on its circle")
        if view.sub == "pair":
            f = view.record["period_factor"][n] if view.violated_law else 1.0
            fac = np.where(np.arange(len(th)) >= k, f, 1.0)
            e = kap * L * (fac * td) ** 2 / (2 * g) + 1.0 - np.cos(th)
            if not close(e, np.full_like(e, e0), scale=1.0):
                err.append(f"{n}: energy height (period factor {f}) not constant")
            if not view.violated_law and view.record["period_factor"][n] != 1.0:
                err.append(f"{n}: legal twin with a period factor")
            continue
        rec = view.record
        held = t <= c["release_time_s"]
        if np.abs(view.v[n][held]).max() != 0.0:
            err.append("bob moves before the release")
        e = kap * L * td ** 2 / (2 * g) + 1.0 - np.cos(th)
        before = ~held & (t < view.t_event)
        cr = rec["damper_crossings_s"]
        after = (t > rec["t_damper_exit_s"]) & (t < (cr[1][0] if len(cr) > 1 else t[-1] + 1.0))
        if not close(e[before], np.full(before.sum(), e0), scale=1.0):
            err.append("energy not conserved before the damper")
        if not after.any():
            err.append("no frame after the damper")
            continue
        eps = math.sqrt(float(np.mean(e[after])) / e0)
        if not close(e[after], np.full(after.sum(), rec["epsilon"] ** 2 * e0), scale=1.0):
            err.append(f"energy after the damper != eps^2 x before (eps = {rec['epsilon']})")
        if view.violated_law:
            if view.violated_law != "energy" or eps <= 1.0 or not close(eps, view.magnitude, scale=1.0):
                err.append(f"violation {view.violated_law} x{view.magnitude} but realised eps {eps:.9g}")
        elif eps > 1.0 + 1e-9:
            err.append(f"legal twin creates energy: realised eps {eps:.9g}")
    return err
