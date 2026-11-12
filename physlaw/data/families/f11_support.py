"""F11: static equilibrium / torque balance (law ``statics``). Side view, gravity -9.81 z, no floor.

A uniform block (half-length a along x, half-height b = beta a) rests on a thin shelf whose edge is at x_e;
the block overhangs it by x a, i.e. its centre of mass is (x - 1) a beyond the edge: the law boundary (CoM over
the edge) is x = 1, x < 1 is stable, x > 1 tips. A thin prop under the overhanging part holds the block in every
group; at the event (a frame) the prop drops away (3 g, the same in every twin). Groups take x on the grid
``twins.overhang_grid`` (default 0.55, 0.65, ..., 1.45: symmetric about 1, so an across-group kink at x = 1 is
testable); two twins per group, sharing every pre-event frame:

  stay   the block stays at rest                         legal iff x < 1
  tip    the block tips off the edge                     legal iff x > 1
violated_law "statics", violation_magnitude |x - 1|.

Tipping is closed form (DOP853, rtol 1e-12) for a frictionless edge: Lagrangian in (phi, u), phi the rotation
about +y (outer end down), u the edge's coordinate along the block's bottom face from the face's centre, CoM
c = O - u e_x + b e_z (O the edge, e_x = (cos phi, -sin phi), e_z = (sin phi, cos phi) in x-z):
    (b^2 + u^2 + k^2) phi'' - b u'' + 2 u u' phi' = g (-u cos phi + b sin phi),    k^2 = (a^2 + b^2) / 3
    -b phi'' + u'' - u phi'^2 + g sin phi = 0
until the edge's normal reaction vanishes or the edge reaches the block's end, then free flight (parabola, constant
spin). The reaction is vertical at the release, so the CoM starts straight down with |a_cm| = g d^2 / (d^2 + k^2),
d = |x - 1| a (``laws.tip_factor``). The violating tip at x < 1 mirrors the lever (generalised force
g (|u| cos phi + b sin phi)) until the CoM has passed over the edge, then follows the same frictionless dynamics:
its initial |a_cm| equals the legal one at the mirrored overhang 2 - x (pointing up), so the tip kinematics are
smooth in |x - 1| across the boundary. Draws in which the falling block would cut the shelf or the prop are
rejected. clean_until_s per twin: the last instant the block is wholly in frame (stay: the clip end). The record's t_physical_from_s is where the tip becomes frictionless physics
(the release for legal tips; the handover or the separation for the mirrored lever).
event_state pre / post are exact and carry the CoM acceleration ``a`` and the angular acceleration ``alpha`` (post:
the initial a_cm and alpha). The prop's descent (constants.prop_acc_over_g = 3 g) and the block's free flight after
the separation are the clip's own g references (``io.law_inputs`` ctx["g_clip"]; the statics law uses no metric g).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp

from physlaw.data.families import (G, LOOKS, Body, Scene, Static, Track, Twin, event_time, frame_times, outside,
                                   pick_look, quat_y_array, side_camera, visible_box)

SUBFAMILIES = {"support": 220}
OVERHANG_GRID = tuple(round(0.55 + 0.1 * i, 2) for i in range(10))
CAM_DIST = 14.0
SHELF_T = 0.12              # shelf thickness
PROP_ACC = 3.0 * G          # the prop drops faster than anything the block does
PROP_HALF = 0.05
MARGIN = 0.15


# --------------------------------------------------------------------------- #
# tipping dynamics
# --------------------------------------------------------------------------- #


def _rhs(y, b, k2, fake):
    ph, u, w, ud = y
    lever = abs(u) if fake else -u
    Q = G * (lever * math.cos(ph) + b * math.sin(ph))
    M = np.array([[b * b + u * u + k2, -b], [-b, 1.0]])
    return np.linalg.solve(M, [Q - 2.0 * u * ud * w, u * w * w - G * math.sin(ph)])


def _normal(y, b, k2, fake):
    """Normal reaction of the edge on the bottom face, per unit mass (along e_z)."""
    ph, u, w, ud = y
    pdd, _ = _rhs(y, b, k2, fake)
    return -b * w * w + 2.0 * ud * w + u * pdd + G * math.cos(ph)


def _frame_vectors(ph):
    ph = np.asarray(ph, float)
    ex = np.stack([np.cos(ph), -np.sin(ph)], -1)
    ez = np.stack([np.sin(ph), np.cos(ph)], -1)
    return ex, ez


def tip_motion(tau, a: float, b: float, x: float):
    """Block pose relative to the edge O at times tau >= 0 after the release (x-z plane, side +1).

    Returns (c [n, 2] CoM - O, cdot [n, 2], phi [n], phidot [n], info) with info: t_sep, phase ends, a0 (initial
    CoM acceleration [2]), alpha0 (initial angular acceleration phi''), fake (x < 1)."""
    k2 = (a * a + b * b) / 3.0
    fake = x < 1.0
    y0 = [0.0, -(x - 1.0) * a, 0.0, 0.0]
    phases, t0 = [], 0.0
    for is_fake in ((True, False) if fake else (False,)):
        f = lambda _, y, fk=is_fake: [y[2], y[3], *_rhs(y, b, k2, fk)]
        e_n = lambda _, y, fk=is_fake: _normal(y, b, k2, fk)
        e_n.terminal, e_n.direction = True, -1
        e_end = lambda _, y: abs(y[1]) - a
        e_end.terminal = True
        e_over = lambda _, y: -y[1] * math.cos(y[0]) + b * math.sin(y[0])      # CoM x relative to the edge
        e_over.terminal, e_over.direction = True, 1
        sol = solve_ivp(f, (t0, t0 + 10.0), y0, method="DOP853", rtol=1e-12, atol=1e-12,
                        events=[e_n, e_end] + ([e_over] if is_fake else []), dense_output=True)
        phases.append((t0, float(sol.t[-1]), sol))
        if is_fake and not (len(sol.t_events) > 2 and len(sol.t_events[2])):
            break                                      # left the edge before the CoM passed over it
        t0, y0 = float(sol.t[-1]), sol.y[:, -1].tolist()
    t_sep = phases[-1][1]
    ph_s, u_s, w_s, ud_s = phases[-1][2].y[:, -1]
    ex, ez = _frame_vectors(ph_s)
    c_s = -u_s * ex + b * ez
    v_s = (b * w_s - ud_s) * ex + u_s * w_s * ez
    tau = np.asarray(tau, float)
    C, V, PH, W = np.zeros((tau.size, 2)), np.zeros((tau.size, 2)), np.zeros(tau.size), np.zeros(tau.size)
    for i, s in enumerate(tau):
        if s <= t_sep:
            sol = next(p[2] for p in phases if p[0] <= s <= p[1] + 1e-12)
            ph, u, w, ud = sol.sol(max(s, 0.0))
            e1, e2 = _frame_vectors(ph)
            C[i], V[i], PH[i], W[i] = -u * e1 + b * e2, (b * w - ud) * e1 + u * w * e2, ph, w
        else:
            d = s - t_sep
            C[i] = c_s + v_s * d + np.array([0.0, -0.5 * G * d * d])
            V[i] = v_s + np.array([0.0, -G * d])
            PH[i], W[i] = ph_s + w_s * d, w_s
    pdd, udd = _rhs([0.0, -(x - 1.0) * a, 0.0, 0.0], b, k2, fake)
    a0 = (b * pdd - udd) * np.array([1.0, 0.0]) + (-(x - 1.0) * a) * pdd * np.array([0.0, 1.0])
    return C, V, PH, W, {"t_sep": t_sep, "phase_ends": [p[1] for p in phases], "a0": a0, "alpha0": float(pdd), "fake": fake,
                         "handover": phases[0][1] if fake and len(phases) > 1 else None}


def _block_corners(c, ph, a, b):
    ex, ez = _frame_vectors(ph)
    return np.stack([c + sx * a * ex + sz * b * ez for sx in (-1, 1) for sz in (-1, 1)], -2)


def _inside(pts, c, ph, a, b, tol=1e-6):
    """pts [..., 2] strictly inside the block rectangle (centre c [n, 2], angle ph [n])."""
    ex, ez = _frame_vectors(ph)
    q = pts - c[:, None, :]
    return ((np.abs((q * ex[:, None, :]).sum(-1)) < a - tol) & (np.abs((q * ez[:, None, :]).sum(-1)) < b - tol)).any(-1)


# --------------------------------------------------------------------------- #
# sampler
# --------------------------------------------------------------------------- #


def sample(rng, cfg, fmt, sub="support"):
    tw_cfg = (cfg.get("twins") if hasattr(cfg, "get") else None) or {}
    grid = [float(v) for v in (tw_cfg.get("overhang_grid") or OVERHANG_GRID)]
    cam = side_camera(fmt, zc=4.3, dist=CAM_DIST)
    hw, hh = visible_box(cam, CAM_DIST)
    zb = 4.3 - hh
    T, fps = fmt["num_frames"], fmt["fps"]
    t = frame_times(fmt)
    t_end = float(t[-1])
    te = event_time(rng, fmt, aligned=True)
    k = int(round(te * fps)) + 1
    x = float(grid[int(rng.integers(len(grid)))])
    a = float(rng.uniform(0.6, 0.9))
    beta = float(rng.uniform(0.45, 0.65))
    b, depth = beta * a, 0.3
    side = float(rng.choice([-1.0, 1.0]))
    z_top = 4.3 + float(rng.uniform(0.6, 1.6))
    x_e = float(rng.uniform(-2.0, -0.5))
    d = (x - 1.0) * a
    c0 = np.array([x_e + d, z_top + b])
    if x_e - (2.0 - x) * a < -hw + MARGIN or z_top + 2 * b > 4.3 + hh - MARGIN:
        return "block_outside_frame"
    tau = t[k:] - te
    C, V, PH, W, info = tip_motion(tau, a, b, x)
    O = np.array([x_e, z_top])
    Cw = C + O
    # the falling block must not cut the shelf (edge corners inside it, or its corners inside the shelf slab)
    shelf_pts = np.array([[x_e, z_top], [x_e, z_top - SHELF_T]])
    corners = _block_corners(Cw, PH, a, b)
    cut = _inside(np.broadcast_to(shelf_pts, (len(tau), 2, 2)), Cw, PH, a, b) | (
        ((corners[..., 0] < x_e - 1e-6) & (corners[..., 1] < z_top - 1e-6) & (corners[..., 1] > z_top - SHELF_T)).any(-1))
    if cut.any():
        return f"tip_cuts_shelf_at_{float(tau[int(np.argmax(cut))]):.3f}s"
    # prop: top at z_top - PROP_ACC tau^2 / 2 under x_p, half-width PROP_HALF
    x_p = x_e + 0.6 * x * a
    zp = z_top - 0.5 * PROP_ACC * tau ** 2
    prop_pts = np.stack([np.stack([np.full_like(zp, x_p - PROP_HALF), zp], -1),
                         np.stack([np.full_like(zp, x_p + PROP_HALF), zp], -1)], 1)
    in_prop = ((np.abs(corners[..., 0] - x_p) < PROP_HALF) & (corners[..., 1] < zp[:, None])).any(-1)
    if (_inside(prop_pts, Cw, PH, a, b) | in_prop).any():
        return "tip_hits_prop"
    look = pick_look(rng)
    block = Body("A", "box", a, float(rng.uniform(0.5, 2.0)), dict(LOOKS[look]), 1, half=(a, depth, b), spin_axis=1)
    Hp = 0.5 * (z_top - zb + 1.0)
    prop = Body("P", "box", PROP_HALF, 1.0, dict(LOOKS["neutral"]), 2, half=(PROP_HALF, 0.2, Hp), aux=True)

    def world(cxz, vxz, ph, w):
        n = len(ph)
        p = np.stack([cxz[:, 0], np.zeros(n), cxz[:, 1]], -1)
        v = np.stack([vxz[:, 0], np.zeros(n), vxz[:, 1]], -1)
        q, wv = quat_y_array(ph), np.stack([np.zeros(n), w, np.zeros(n)], -1)
        if side < 0:
            p[:, 0] *= -1
            v[:, 0] *= -1
            q[:, 2] *= -1
            wv[:, 1] *= -1
        return Track(p, v, q, wv)

    still = world(np.tile(c0, (T, 1)), np.zeros((T, 2)), np.zeros(T), np.zeros(T))
    ptau = np.clip(t - te, 0.0, None)
    prop_tr = world(np.stack([np.full(T, x_p), z_top - Hp - 0.5 * PROP_ACC * ptau ** 2], -1),
                    np.stack([np.zeros(T), -PROP_ACC * ptau], -1), np.zeros(T), np.zeros(T))
    tip_tr = still.copy()
    tipped = world(Cw, V, PH, W)
    for key in ("p", "v", "q", "w"):
        getattr(tip_tr, key)[k:] = getattr(tipped, key)
    sgn = np.array([side, 1.0, 1.0])

    def st3(cxz, vxz, axz, wy, prop_a, alpha_y=0.0):
        return {"t_s": te, "objects": ["A", "P"],
                "p": [(np.array([cxz[0], 0.0, cxz[1]]) * sgn).tolist(), prop_tr.p[k - 1].tolist()],
                "v": [(np.array([vxz[0], 0.0, vxz[1]]) * sgn).tolist(), [0.0, 0.0, 0.0]],
                "w": [[0.0, side * wy, 0.0], [0.0, 0.0, 0.0]],
                "a": [(np.array([axz[0], 0.0, axz[1]]) * sgn).tolist(), [0.0, 0.0, prop_a]],
                "alpha": [[0.0, side * alpha_y, 0.0], [0.0, 0.0, 0.0]]}

    pre_state = st3(c0, np.zeros(2), np.zeros(2), 0.0, 0.0)
    twins = {}
    for name, tr, a_post, al_post in (("stay", still, np.zeros(2), 0.0), ("tip", tip_tr, info["a0"], info["alpha0"])):
        legal = (name == "tip") == (x > 1.0)
        out = outside(cam, block, tr.p[k:], 4.0)
        stop = k + (int(np.argmax(out)) if out.any() else T - k)
        if (stop - k) / fps < 0.5:
            return f"{name}_leaves_frame_within_0.5s"
        twins[name] = Twin(
            name, {"A": tr, "P": prop_tr}, legal=legal, violated_law=None if legal else "statics",
            violation_magnitude=0.0 if legal else abs(x - 1.0),
            flags={"energy_physical": legal or name == "stay"},
            record={"kind": "release", "tipped": name == "tip", "mirrored_lever": name == "tip" and info["fake"],
                    **({"t_separate_s": te + info["t_sep"],
                        # from here on the motion is the frictionless one (energy conserved)
                        "t_physical_from_s": te + (info["t_sep"] if info["handover"] is None else info["handover"])
                        if info["fake"] else te} if name == "tip" else {})},
            clean_until_s=min((stop - 1) / fps, t_end),
            event_state={"pre": pre_state, "post": st3(c0, np.zeros(2), a_post, 0.0, -PROP_ACC, al_post)})
    shelf = Static("shelf", (6.0, 0.6, SHELF_T / 2), (side * (x_e - 6.0), 0.0, z_top - SHELF_T / 2),
                   look=dict(LOOKS["ramp"]))
    return Scene("f11", "support", [block, prop], [shelf], cam, G, te, "release", twins, fps,
                 constants={"g": G, "overhang_x": x, "aspect_ba": beta, "half_length": a, "half_height": b,
                            "edge_point": [side * x_e, 0.0, z_top], "direction": side,
                            "inertia_over_mass": (a * a + b * b) / 3.0, "release_time_s": te,
                            "prop_acc_over_g": PROP_ACC / G,
                            "tip_axis": [0.0, side, 0.0], "a0_legal_over_g": float((x - 1) ** 2 / ((x - 1) ** 2 + (1 + beta ** 2) / 3))},
                 look={"floor": False}, inframe={"start": 0, "stop": T},
                 params={"look": look, "prop_x": side * x_p})


# --------------------------------------------------------------------------- #
# acceptance (spec §2.4)
# --------------------------------------------------------------------------- #


def acceptance(view) -> list:
    """Block still before the release; stay twins still throughout; tip twins move. tipped == (x > 1) iff legal.
    Tip twins: |a_cm| / g right after the release = tip_factor(x, beta) (the event_state's exact value is checked
    in tests); mechanical energy (translation + spin about y + m g z) conserved after the release (legal) or after
    the handover (mirrored lever); free flight after the separation: a = -g z, constant spin."""
    from physlaw.data.families import close
    from physlaw.eval.laws.incline import tip_factor
    c, k, fps, err = view.constants, view.k, view.fps, []
    g, x = c["g"], c["overhang_x"]
    p, v, w = view.p["A"], view.v["A"], view.w["A"]
    if np.abs(v[:k]).max() != 0.0 or np.abs(p[:k] - p[0]).max() != 0.0:
        err.append("block moves before the release")
    moved = bool(np.abs(v[k:]).max() > 0.0)
    if moved != bool(view.record.get("tipped")):
        err.append("record.tipped does not match the motion")
    if view.legal != (moved == (x > 1.0)):
        err.append(f"tipped={moved} at overhang x={x} labelled legal={view.legal}")
    if not moved:
        return err
    t = np.arange(len(p)) / fps
    kc2 = c["inertia_over_mass"]
    E = 0.5 * (v ** 2).sum(1) + 0.5 * kc2 * w[:, 1] ** 2 + g * p[:, 2]
    t0 = view.record["t_physical_from_s"]
    seg = t >= t0 - 1e-9
    ref = E[k - 1] if t0 <= view.t_event + 1e-9 else E[seg][0]
    if not close(E[seg], np.full(seg.sum(), ref), scale=g * c["half_length"]):
        err.append("mechanical energy not conserved while tipping / falling")
    fl = np.nonzero(t > view.record["t_separate_s"] + 1e-9)[0]
    fl = fl[fl < len(p) - 1]
    if len(fl):
        acc = (v[fl + 1] - v[fl]) * fps
        if not close(acc, np.tile([0.0, 0.0, -g], (len(fl), 1)), scale=g) or not close(w[fl + 1], w[fl], scale=1.0):
            err.append("free flight after the separation is not ballistic")
    a0 = 2.0 * (p[k, 2] - p[k - 1, 2]) * fps * fps          # vertical start from rest over one frame (O(dt) check)
    want = g * float(tip_factor(x, c["aspect_ba"]))
    if abs(abs(a0) - want) > 0.05 * want + 0.05:
        err.append(f"initial |a_cm| {abs(a0):.4g} != g tip_factor {want:.4g}")
    return err
