"""F6: friction geometry and mass-independent slides (spec §2.3 F6; slides from v1 F5).

  flat     one cube slides at constant speed over frictionless floor and crosses onto a
           friction band (event = its centre crossing the band edge, between frames);
           it then decelerates at a = mu g until it stops. Top-down camera.
           Twins: legal_mu1, legal_mu2 (two legal friction coefficients, same look).
  incline  one cube held at rest on a long incline at theta, released at the event frame;
           a = g (sin th - mu cos th) with mu < tan th, so mu = tan th - a / (g cos th).
           Side camera, no floor in view (the slab runs out of the frame).
           Twins: legal_mu1, legal_mu2.
  slides   two cubes, same size and initial speed, different density (visibly different
           materials: the heavier is metal), cross the band together.
           Twins: legal_mu1, legal_mu2, heavy_x13 (the heavier cube decelerates 1.3x
           faster at mu1; mass independence). Energy is only lost, never created.

Closed form throughout; static friction is taken equal to kinetic friction, so a cube
that stops stays stopped.
"""
from __future__ import annotations

import math

import numpy as np

from physlaw.data.families import (G, LOOKS, Body, Scene, Static, Track, Twin, event_time, frame_times,
                                   pick_look, quat_y, side_camera, top_camera, visible_box)
from physlaw.data.families.f8_ramp import RAMP_DEPTH, RAMP_THICK, mirror_static

SUBFAMILIES = {"flat": 300, "incline": 300, "slides": 200}
MARGIN = 0.15
HALF = (0.30, 0.42)


def sample(rng, cfg, fmt, sub):
    return {"flat": _flat, "incline": _incline, "slides": _slides}[sub](rng, cfg, fmt)


def _quat_z(psi):
    return (math.cos(psi / 2), 0.0, 0.0, math.sin(psi / 2))


def _decel_track(t, te, p_e, d, v0, a, half, psi):
    """Constant velocity v0*d through p_e at te, then deceleration a until rest."""
    T = t.size
    dt = t - te
    s = np.where(dt <= 0, v0 * dt, 0.0)
    sd = np.where(dt <= 0, v0, 0.0)
    post = dt > 0
    t_stop = v0 / a
    tt = np.minimum(dt[post], t_stop)
    s[post] = v0 * tt - 0.5 * a * tt * tt
    sd[post] = np.maximum(v0 - a * dt[post], 0.0)
    p = np.asarray(p_e, float) + s[:, None] * d
    p[:, 2] = half
    v = sd[:, None] * d
    return Track(p, v, np.tile(_quat_z(psi), (T, 1)), np.zeros((T, 3))), te + t_stop


def _check_window(te, v0, a, half, t_stop, t_end):
    """PyBullet check once the cube is wholly on the band: Bullet applies band friction as soon as any
    contact touches it, while the closed form switches it on at the centre crossing."""
    disc = v0 * v0 - 2 * a * half * 1.05
    if disc <= 0:
        return {}
    return {"t0": te + (v0 - math.sqrt(disc)) / a + 1e-3, "until": min(t_stop + 0.3, t_end)}


def _band(psi, p_e, d, look):
    """A thin band covering the half-plane ahead of the band edge (through p_e, normal d)."""
    L = 30.0
    centre = np.asarray(p_e, float) + d * L
    return Static("band", (L, L, 0.002), (float(centre[0]), float(centre[1]), 0.0005), _quat_z(psi), dict(look))


def _planar_layout(rng, fmt, cam, v0, max_stop, extra_lateral=0.0):
    hw, hh = visible_box(cam, cam.position[2])
    psi = float(rng.uniform(0, 2 * math.pi))
    d = np.array([math.cos(psi), math.sin(psi), 0.0])
    # place the event point so the stop point lands inside the view
    p_e = np.array([rng.uniform(-hw + 2, hw - 2), rng.uniform(-hh + 1.5, hh - 1.5), 0.0]) - d * 0.5 * max_stop
    return psi, d, p_e


def _flat(rng, cfg, fmt):
    cam = top_camera(fmt)
    T, fps = fmt["num_frames"], fmt["fps"]
    t = frame_times(fmt)
    te = event_time(rng, fmt, aligned=False)
    half = float(rng.uniform(*HALF))
    v0 = float(rng.uniform(1.5, 3.0))
    mu1 = float(rng.uniform(0.15, 0.35))
    mu2 = float(min(mu1 * rng.uniform(1.6, 2.6), 0.9))
    psi, d, p_e = _planar_layout(rng, fmt, cam, v0, v0 * v0 / (2 * mu1 * G))
    look = pick_look(rng)
    cube = Body("A", "cube", half, float(rng.uniform(0.5, 3.0)), dict(LOOKS[look]), 1)
    twins, stops = {}, {}
    for name, mu in (("legal_mu1", mu1), ("legal_mu2", mu2)):
        tr, ts = _decel_track(t, te, p_e, d, v0, mu * G, half, psi)
        stops[name] = ts
        twins[name] = Twin(name, {"A": tr}, constants={"mu": mu, "a": mu * G, "t_stop_s": ts},
                           flags={"energy_physical": True}, record={"kind": "band", "mu": mu, "a": mu * G})
    k = int(math.floor(te * fps)) + 1
    sc = Scene("f6", "flat", [cube], [_band(psi, p_e, d, LOOKS["band"])], cam, G, te, "band", twins, fps,
               constants={"g": G, "v0": v0, "half_size": half, "direction_rad": psi, "band_edge_point": p_e.tolist(),
                          "band_normal": d.tolist()},
               inframe={"start": max(0, k - fps), "stop": min(T, int(math.ceil(max(stops.values()) * fps)) + 12)},
               sim_init={"twin": "legal_mu1", **_check_window(te, v0, mu1 * G, half, stops["legal_mu1"], t[-1])},
               params={"look": look})
    return sc


def _slides(rng, cfg, fmt):
    cam = top_camera(fmt)
    T, fps = fmt["num_frames"], fmt["fps"]
    t = frame_times(fmt)
    te = event_time(rng, fmt, aligned=False)
    half = float(rng.uniform(*HALF))
    v0 = float(rng.uniform(1.5, 3.0))
    mu1 = float(rng.uniform(0.15, 0.30))
    mu2 = float(min(mu1 * rng.uniform(1.6, 2.4), 0.9))
    psi, d, p_e = _planar_layout(rng, fmt, cam, v0, v0 * v0 / (2 * mu1 * G))
    lat = np.array([-d[1], d[0], 0.0])
    sep = float(rng.uniform(1.4, 2.4))
    rho_lo = float(np.exp(rng.uniform(np.log(0.5), np.log(1.5))))
    ratio = float(np.exp(rng.uniform(np.log(2.0), np.log(5.0))))
    heavy = str(rng.choice(["A", "B"]))
    identical = cfg.get("slide_look", "distinct") == "identical"
    light_look = pick_look(rng, exclude=("metal",))
    looks = {n: (light_look if identical or n != heavy else "metal") for n in "AB"}
    bodies = [Body(n, "cube", half, rho_lo * (ratio if n == heavy else 1.0), dict(LOOKS[looks[n]]), i + 1)
              for i, n in enumerate("AB")]
    pe = {"A": p_e + lat * sep / 2, "B": p_e - lat * sep / 2}
    twins, stops = {}, []
    for name, mu, mult in (("legal_mu1", mu1, 1.0), ("legal_mu2", mu2, 1.0), ("heavy_x13", mu1, 1.3)):
        tracks = {}
        for n in "AB":
            a = mu * G * (mult if n == heavy else 1.0)
            tracks[n], ts = _decel_track(t, te, pe[n], d, v0, a, half, psi)
            stops.append(ts)
        law = None if mult == 1.0 else "mass_independence"
        twins[name] = Twin(name, tracks, legal=law is None, violated_law=law, violation_magnitude=mult if law else 0.0,
                           constants={"mu": mu, "a": mu * G}, flags={"energy_physical": True},
                           record={"kind": "band", "mu": mu, "decel_heavy": mu * G * mult, "decel_light": mu * G,
                                   "heavier": heavy})
    k = int(math.floor(te * fps)) + 1
    sc = Scene("f6", "slides", bodies, [_band(psi, p_e, d, LOOKS["band"])], cam, G, te, "band", twins, fps,
               constants={"g": G, "v0": v0, "half_size": half, "direction_rad": psi, "band_edge_point": p_e.tolist(),
                          "band_normal": d.tolist(), "heavier": heavy, "mass_ratio_heavy_over_light": ratio,
                          "separation": sep},
               inframe={"start": max(0, k - fps), "stop": min(T, int(math.ceil(max(stops) * fps)) + 12)},
               sim_init={"twin": "legal_mu1", **_check_window(te, v0, mu1 * G, half, max(stops), t[-1])},
               params={"looks": looks, "density_light": rho_lo})
    return sc


def _incline(rng, cfg, fmt):
    cam = side_camera(fmt)
    hw, hh = visible_box(cam, 20.0)
    zt = cam.position[2] + hh
    T, fps = fmt["num_frames"], fmt["fps"]
    t = frame_times(fmt)
    te = event_time(rng, fmt, aligned=True)
    kr = int(round(te * fps))
    side = float(rng.choice([-1.0, 1.0]))
    th = math.radians(rng.uniform(20, 40))
    tan = math.tan(th)
    mu1 = float(rng.uniform(0.05, max(tan - 0.25, 0.06)))
    mu2 = float(rng.uniform(mu1 + 0.1, tan - 0.05)) if tan - 0.05 > mu1 + 0.1 else None
    if mu2 is None:
        return "no_second_mu_fits"
    half = float(rng.uniform(*HALF))
    d = np.array([math.cos(th), 0.0, -math.sin(th)])
    n = np.array([math.sin(th), 0.0, math.cos(th)])
    start = np.array([-hw + 1.2 + half, 0.0, zt - MARGIN - 1.6 * half - float(rng.uniform(0.0, 1.5))])
    # slab under the whole path, far past the frame
    L = 40.0
    a_top = start - half * n - d * (half * 1.5 + 0.3)
    mid = a_top + d * L / 2 - 0.5 * RAMP_THICK * n
    slab = mirror_static(Static("slab", (L / 2, RAMP_DEPTH, RAMP_THICK / 2), tuple(mid), quat_y(th),
                                dict(LOOKS["ramp"])), side)
    look = pick_look(rng)
    cube = Body("A", "cube", half, float(rng.uniform(0.5, 3.0)), dict(LOOKS[look]), 1)
    q = quat_y(th * side)
    twins = {}
    for name, mu in (("legal_mu1", mu1), ("legal_mu2", mu2)):
        a = G * (math.sin(th) - mu * math.cos(th))
        dt = np.clip(t - te, 0.0, None)
        s = 0.5 * a * dt * dt
        p = start + s[:, None] * d
        v = (a * dt)[:, None] * d
        if side < 0:
            p[:, 0] *= -1
            v[:, 0] *= -1
        tr = Track(p, v, np.tile(q, (T, 1)), np.zeros((T, 3)))
        twins[name] = Twin(name, {"A": tr}, constants={"mu": mu, "a": a}, flags={"energy_physical": True},
                           record={"kind": "release", "mu": mu, "a": a})
    if side < 0:
        start = start * [-1, 1, 1]
    sc = Scene("f6", "incline", [cube], [slab], cam, G, te, "release", twins, fps,
               constants={"g": G, "theta_rad": th, "half_size": half, "release_frame": kr, "release_time_s": te,
                          "direction": side, "start": start.tolist()},
               look={"floor": False},
               inframe={"start": 0, "stop": min(T, kr + 1 + int(1.0 * fps))},
               sim_init={"twin": "legal_mu1", "t0": te, "until": min(te + 1.0, t[-1])},
               params={"look": look})
    return sc


def acceptance(view) -> list:
    """Deceleration while sliding on the band = mu g (heavier x1.3 in heavy_x13); incline a = g(sin - mu cos)."""
    from physlaw.data.families import close
    g, k, fps, err = view.constants["g"], view.k, view.fps, []
    if view.sub == "incline":
        th, mu = view.constants["theta_rad"], view.constants["mu"]
        a_want = g * (math.sin(th) - mu * math.cos(th))
        a = np.linalg.norm(view.accel("A", k, len(view.v["A"]) - 1), axis=1)
        if not close(a, a_want, scale=g):
            err.append(f"incline accel {a.min():.9g}..{a.max():.9g} != {a_want:.9g}")
        if not close(math.tan(th) - a_want / (g * math.cos(th)), mu, scale=1.0):
            err.append("mu != tan th - a / (g cos th)")
        return err
    for n in view.p:
        sp = np.linalg.norm(view.v[n], axis=1)
        frames = [f for f in range(k, len(sp) - 1) if sp[f + 1] > 0]
        a = (sp[frames] - sp[[f + 1 for f in frames]]) * fps
        mult = view.magnitude if (view.violated_law and n == view.record.get("heavier")) else 1.0
        want = view.constants["mu"] * g * mult
        if not frames or not close(a, want, scale=g):
            err.append(f"{n}: decel {a.min() if frames else None} != {want:.9g}")
        if not close(sp[:k - 1], view.constants["v0"], scale=1.0):
            err.append(f"{n}: speed not constant before the band")
    return err
