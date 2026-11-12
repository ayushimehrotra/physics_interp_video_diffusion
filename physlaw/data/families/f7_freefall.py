"""F7: free fall, projectiles and floor bounces (spec §2.3 F7). Gravity -9.81 z, side view.

Sub-families (one event per group, in the master event window):

  projectile  a ball thrown in from off-screen; branch mid-flight (on a frame).
              legal       closed-form parabola, floor bounces at e_floor = 0.8
              legal_e04   same flight, floor bounces at e_floor = 0.4 (second legal twin)
              gravity_x15 g scaled x1.5 after the branch           (violated_law "gravity": constant g)
              vx_x06      horizontal velocity x0.6 at the branch   (violated_law "gravity": axis independence)
  drop        two balls, identical size and look, different density, held at rest and
              released together; branch mid-fall.
              legal, legal_e04 (as above), heavy_x13: the heavier ball's acceleration
              x1.3 after the branch (violated_law "gravity": mass independence; creates energy -> flagged)
              projectile / drop clean_until_s per twin: the first floor contact (exact) or the last frame a
              ball is wholly in frame, whichever comes first (``families.set_clean_until``).
  bounce      one ball held at rest and released (the fall shows g and the scale); event = first
              floor contact (event_kind "bounce"). Twins: the ladder eps<e> (config
              ``twins.bounce_ladder``, default 0.5, 0.55, ..., 1.5): first-bounce restitution e,
              rebound apex e^2 x the drop height; e <= 1 legal, e > 1 violated_law "energy"
              (magnitude e). Later bounces at 0.8 (outside the clean window, which ends at the
              second floor contact: ``clean_until_s``). The camera is fitted per group so the floor
              contact .. the e = 1.5 apex fill 85% of the image height; the ball radius is 0.17-0.22 x the
              drop height (>= 50 px across at 832 x 480). stop_dose NaN (v' = e |v| is smallest at e = 0, off the ladder).

Everything is closed form (``families.ballistic``); see the package docstring for why.
The PyBullet check stops before the first floor contact: Bullet's restitution is not
a usable reference (at 960 Hz the ball does not bounce at all; at 240 Hz the rebound
is off by up to 0.3 m).
"""
from __future__ import annotations

import math

import numpy as np

from physlaw.data.families import (G, LOOKS, Body, Scene, Track, Twin, ballistic, energy, eps_ladder, eps_name,
                                   event_time, framed_side_camera, frame_times, pick_look, set_clean_until,
                                   side_camera, splice, stop_dose, visible_box)

SUBFAMILIES = {"projectile": 300, "drop": 200, "bounce": 200}
E_HI, E_LO = 0.8, 0.4
RADIUS = (0.30, 0.45)
# bounce: the camera is fitted per group to the floor contact .. the highest rung's apex, and the ball scales with the
# drop height H (radius / H), so the ball is >= BOUNCE_MIN_BALL_FRAC of the image width at every drop height
BOUNCE_RADIUS_FRAC = (0.17, 0.22)
BOUNCE_FILL = 0.85
BOUNCE_MIN_BALL_FRAC = 50.0 / 832.0
MARGIN = 0.15            # m between a silhouette and the image border


def _ball(name, r, rho, look, seg):
    return Body(name, "sphere", float(r), float(rho), dict(LOOKS[look]), seg)


def _flags(scene, tw, k):
    E0 = energy(scene, tw, k - 1)
    Es = [energy(scene, tw, f) for f in range(k, tw.tracks["A"].p.shape[0])]
    return {"energy_physical": bool(max(Es) <= E0 * (1 + 1e-9) + 1e-9)}


def _parabola(t, t0, p0, v0, g=G):
    dt = (np.asarray(t, float) - t0)[:, None]
    p = np.asarray(p0, float) + np.asarray(v0, float) * dt - 0.5 * g * dt ** 2 * np.array([0, 0, 1.0])
    v = np.asarray(v0, float) + 0 * dt - g * dt * np.array([0, 0, 1.0])
    return p, v


def sample(rng, cfg, fmt, sub):
    return {"projectile": _projectile, "drop": _drop, "bounce": _bounce}[sub](rng, cfg, fmt)


def _frame(fmt):
    cam = side_camera(fmt)
    hw, hh = visible_box(cam, 20.0)
    return cam, hw, cam.position[2] - hh, cam.position[2] + hh   # x half-width, z bottom, z top


def _projectile(rng, cfg, fmt):
    cam, hw, zb, zt = _frame(fmt)
    T, fps = fmt["num_frames"], fmt["fps"]
    t = frame_times(fmt)
    tb = event_time(rng, fmt, aligned=True)
    k = int(round(tb * fps)) + 1
    r = float(rng.uniform(*RADIUS))
    s = float(rng.choice([-1.0, 1.0]))
    # Sampled at the moment the ball crosses into view (centre one radius outside the side
    # edge), then flown forward to the branch: a ball airborne for the ~1 s before the branch
    # cannot have come from inside the frame without rising out of the floor.
    dt_in = float(rng.uniform(0.7, 1.4))
    z_in = float(rng.uniform(1.0, 5.0))
    tau = dt_in + 0.65                                 # airborne from entry until well after the branch
    vz_lo = (0.5 * G * tau ** 2 - (z_in - r)) / tau
    vz_hi = math.sqrt(max(2 * G * (zt - MARGIN - r - z_in), 0.0))
    if vz_lo >= vz_hi:
        return "no_arc_fits"
    p_in = np.array([-s * (hw + r), 0.0, z_in])
    v_in = np.array([s * rng.uniform(3.5, 6.5), 0.0, rng.uniform(vz_lo, vz_hi)])
    (pb,), (vb,) = _parabola([tb], tb - dt_in, p_in, v_in)
    pre_p, pre_v = _parabola(t[:k], tb, pb, vb)
    inside = np.abs(pre_p[:, 0]) <= hw + r
    if (pre_p[inside, 2] - r < 0.02).any():
        return "enters_through_floor"
    if (pre_p[:, 2] + r > zt - MARGIN).any():
        return "apex_above_frame"
    full_in = np.where(np.abs(pre_p[:, 0]) + r <= hw - MARGIN)[0]
    if full_in.size == 0 or (k - full_in[0]) < int(0.5 * fps):
        return "too_little_flight_before_branch"
    look = pick_look(rng)
    ball = _ball("A", r, float(rng.uniform(0.8, 3.0)), look, 1)
    pre = Track(np.vstack([pre_p, np.zeros((T - k, 3))]), np.vstack([pre_v, np.zeros((T - k, 3))]),
                np.tile([1.0, 0, 0, 0], (T, 1)), np.zeros((T, 3)))

    def post(v0, g, e):
        p, v, bounces = ballistic(t[k:], tb, pb, v0, g, r, e)
        return splice(pre, p, v, k), bounces

    twins, land = {}, {}
    for name, v0, g, e, law, mag in (("legal", vb, G, E_HI, None, 0.0), ("legal_e04", vb, G, E_LO, None, 0.0),
                                     ("gravity_x15", vb, 1.5 * G, E_HI, "gravity", 1.5),
                                     ("vx_x06", vb * [0.6, 1, 1], G, E_HI, "gravity", 0.6)):
        tr, b = post(v0, g, e)
        land[name] = b[0] if b else math.inf
        twins[name] = Twin(name, {"A": tr}, legal=law is None, violated_law=law, violation_magnitude=mag,
                           constants={"e_floor": e},
                           record={"kind": "branch", "g_post": g, "v_branch": vb.tolist(), "v_post": list(map(float, v0)),
                                   "e_floor_post": e, "first_landing_s": land[name]})
    if min(land.values()) - tb < 0.4:
        return "lands_too_soon_after_branch"
    sc = Scene("f7", "projectile", [ball], [], cam, G, tb, "branch", twins, fps,
               constants={"g": G, "radius": r, "e_floor": E_HI, "e_floor_lo": E_LO, "z_floor": r},
               inframe={"start": k - int(0.5 * fps), "stop": k + int(0.4 * fps)},
               sim_init={"twin": "legal", "t0": float(t[full_in[0]]), "until": land["legal"] - 0.5 / fps},
               params={"branch_position": pb.tolist(), "branch_velocity": vb.tolist(), "direction": s, "look": look})
    for tw in twins.values():
        tw.flags = _flags(sc, tw, k)
    return set_clean_until(sc, land) or sc


def _held_fall(t, t_rel, p0, r, e_seq, g=G):
    """Track at rest at p0 until t_rel, then free fall with floor bounces."""
    T = t.size
    tr = Track.still(T, p0)
    m = t >= t_rel
    p, v, b = ballistic(t[m], t_rel, p0, np.zeros(3), g, r, e_seq)
    tr.p[m], tr.v[m] = p, v
    return tr, b


def _drop(rng, cfg, fmt):
    cam, hw, zb, zt = _frame(fmt)
    T, fps = fmt["num_frames"], fmt["fps"]
    t = frame_times(fmt)
    tb = event_time(rng, fmt, aligned=True)
    kb = int(round(tb * fps))
    k = kb + 1
    kr = kb - int(round(rng.uniform(0.30, 0.45) * fps))
    tr_ = kr / fps
    r = float(rng.uniform(*RADIUS))
    z0 = float(rng.uniform(5.0, zt - MARGIN - r))
    if math.sqrt(2 * (z0 - r) / G) < (tb - tr_) + 0.3:
        return "lands_too_soon_after_branch"
    d, xc = float(rng.uniform(2.0, 4.5)), float(rng.uniform(-3.0, 3.0))
    rho_lo = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
    ratio = float(np.exp(rng.uniform(np.log(2.0), np.log(5.0))))
    heavy = str(rng.choice(["A", "B"]))
    look = pick_look(rng)
    bodies = [_ball(n, r, rho_lo * (ratio if n == heavy else 1.0), look, i + 1) for i, n in enumerate("AB")]
    p0 = {"A": np.array([xc - d / 2, 0.0, z0]), "B": np.array([xc + d / 2, 0.0, z0])}
    twins, land = {}, {}
    for name, e in (("legal", E_HI), ("legal_e04", E_LO)):
        falls = {n: _held_fall(t, tr_, p0[n], r, e) for n in "AB"}
        tracks = {n: f[0] for n, f in falls.items()}
        land[name] = min((f[1][0] for f in falls.values() if f[1]), default=math.inf)
        twins[name] = Twin(name, tracks, constants={"e_floor": e}, record={"kind": "release", "e_floor_post": e})
    # heavier ball: g x1.3 from the branch on, starting from its legal state at the branch
    tracks = {n: twins["legal"].tracks[n].copy() for n in "AB"}
    h = twins["legal"].tracks[heavy]
    p, v, b_heavy = ballistic(t[k:], tb, h.p[kb], h.v[kb], 1.3 * G, r, E_HI)
    tracks[heavy] = splice(h, p, v, k)
    land["heavy_x13"] = min(b_heavy[:1] + [land["legal"]])
    twins["heavy_x13"] = Twin("heavy_x13", tracks, legal=False, violated_law="gravity",
                              violation_magnitude=1.3, constants={"e_floor": E_HI},
                              record={"kind": "branch", "body": heavy, "g_post": 1.3 * G, "e_floor_post": E_HI})
    sc = Scene("f7", "drop", bodies, [], cam, G, tb, "branch", twins, fps,
               constants={"g": G, "radius": r, "e_floor": E_HI, "e_floor_lo": E_LO, "z_floor": r,
                          "release_time_s": tr_, "release_frame": kr, "drop_height": z0 - r, "heavier": heavy,
                          "mass_ratio_heavy_over_light": ratio},
               inframe={"start": kr - 4, "stop": k + int(0.4 * fps)},
               sim_init={"twin": "legal", "t0": tr_, "until": tr_ + math.sqrt(2 * (z0 - r) / G) - 0.5 / fps},
               params={"release_height": z0, "separation": d, "x_centre": xc, "look": look,
                       "density_light": rho_lo, "density_ratio": ratio})
    for tw in twins.values():
        tw.flags = _flags(sc, tw, k)
    return set_clean_until(sc, land) or sc


def _bounce(rng, cfg, fmt):
    ladder = eps_ladder(cfg, "bounce_ladder")
    T, fps = fmt["num_frames"], fmt["fps"]
    t = frame_times(fmt)
    t_end = (T - 1) / fps
    tc = event_time(rng, fmt, aligned=False)
    kr = int(round((tc - rng.uniform(0.55, 0.85)) * fps))
    tr_ = kr / fps
    tf = tc - tr_
    H = 0.5 * G * tf ** 2
    r = H * float(rng.uniform(*BOUNCE_RADIUS_FRAC))
    z0 = r + H
    # framing: the floor contact to the top of the highest rung's apex fills BOUNCE_FILL of the image height
    z_top = 2.0 * r + max(max(ladder), 1.0) ** 2 * H
    cam = framed_side_camera(fmt, -1.0, 1.0, 0.0, z_top, BOUNCE_FILL)
    hw, _ = visible_box(cam, -cam.position[1])
    if 2.0 * r * cam.px_per_m(-cam.position[1]) < BOUNCE_MIN_BALL_FRAC * cam.resolution[0]:
        return "ball_too_small_in_view"
    x0 = float(rng.uniform(-0.5, 0.5)) * (hw - r)
    look = pick_look(rng)
    ball = _ball("A", r, float(rng.uniform(0.8, 3.0)), look, 1)
    p0 = np.array([x0, 0.0, z0])
    twins, posts = {}, {}
    for e in ladder:
        name = eps_name(e)
        tr, b = _held_fall(t, tr_, p0, r, lambda i, e=e: e if i == 0 else E_HI)
        assert abs(b[0] - tc) < 1e-9, (b[0], tc)
        legal = e <= 1.0
        twins[name] = Twin(name, {"A": tr}, legal=legal, violated_law=None if legal else "energy",
                           violation_magnitude=0.0 if legal else e,
                           constants={"e_floor": e, "e_floor_later": E_HI},
                           record={"kind": "bounce", "epsilon": e, "e_first_bounce": e,
                                   "height_ratio_first_bounce": e * e, "apex_height": e * e * H,
                                   "e_later_bounces": E_HI, "bounce_times_s": b[:4]},
                           epsilon=e, clean_until_s=min(b[1], t_end) if len(b) > 1 else t_end)
        posts[e] = {"A": [0.0, 0.0, e * G * tf]}
    sc = Scene("f7", "bounce", [ball], [], cam, G, tc, "bounce", twins, fps,
               constants={"g": G, "radius": r, "z_floor": r, "release_time_s": tr_, "release_frame": kr,
                          "drop_height": H},
               inframe={"start": kr, "stop": T},      # each twin: up to its clean_until_s (second bounce)
               stop_dose=stop_dose(posts),
               sim_init={"twin": eps_name(ladder[0]), "t0": tr_, "until": tc - 0.5 / fps},
               params={"release_height": z0, "x": x0, "look": look})
    for tw in twins.values():
        tw.flags = _flags(sc, tw, sc.first_post)
    return sc


# --------------------------------------------------------------------------- #
# acceptance (spec §2.4)
# --------------------------------------------------------------------------- #


def _flight_frames(view, n, f0, f1):
    """Frame pairs in [f0, f1) with no floor contact between them (v_z does not jump upward)."""
    a = view.accel(n, f0, f1)[:, 2]
    return [f0 + i for i in range(len(a)) if a[i] < 0]


def acceptance(view) -> list:
    from physlaw.data.families import close
    g, k, fps, err = view.constants["g"], view.k, view.fps, []
    gz = np.array([0.0, 0.0, -g])

    def check_accel(n, frames, factor, what):
        if not frames:
            err.append(f"{what}: no flight frames")
            return
        a = np.array([view.accel(n, f, f + 1)[0] for f in frames])
        if not close(a, np.tile(gz * factor, (len(frames), 1)), scale=g):
            err.append(f"{what}: accel {a[:, 2].min():.6g}..{a[:, 2].max():.6g} != {-g * factor:.6g}")
        # positions integrate the labelled velocities exactly under constant acceleration
        p, v = view.p[n], view.v[n]
        dp = np.array([p[f + 1] - p[f] for f in frames]) * fps
        vm = np.array([(v[f] + v[f + 1]) / 2 for f in frames])
        if not close(dp, vm, scale=max(1.0, float(np.abs(vm).max()))):
            err.append(f"{what}: positions do not integrate the velocities")

    names = list(view.p)
    if view.sub == "bounce":
        n = names[0]
        kr = view.constants["release_frame"]
        check_accel(n, _flight_frames(view, n, kr, k - 1), 1.0, "pre-bounce fall")
        vb = view.v[n][k - 1, 2] - g * (view.t_event - (k - 1) / fps)
        va = view.v[n][k, 2] + g * (k / fps - view.t_event)
        e = -va / vb
        if not close(e, view.record["epsilon"], scale=1.0):
            err.append(f"realised restitution {e:.9g} != epsilon {view.record['epsilon']:.9g}")
        # rebound flight up to the second contact: |v|^2 + 2 g dz = e^2 2 g H
        H, zf = view.constants["drop_height"], view.constants["z_floor"]
        t = np.arange(len(view.p[n])) / fps
        b = view.record["bounce_times_s"]
        fl = (t > b[0]) & (t < (b[1] if len(b) > 1 else np.inf))
        e2 = ((view.v[n][fl] ** 2).sum(1) + 2 * g * (view.p[n][fl, 2] - zf)) / (2 * g * H)
        if not close(e2, np.full_like(e2, e * e), scale=1.0):
            err.append("rebound energy != e^2 m g H")
        if view.violated_law:
            if view.violated_law != "energy" or e <= 1.0 or not close(e, view.magnitude, scale=1.0):
                err.append(f"violation {view.violated_law} x{view.magnitude} but realised e {e:.9g}")
        elif e > 1.0 + 1e-9:
            err.append(f"legal twin creates energy: realised e {e:.9g}")
        return err
    land = view.record.get("first_landing_s") if view.sub == "projectile" else None
    for n in names:
        if view.sub == "projectile":
            f_land = int(math.floor(land * fps))
            pre = _flight_frames(view, n, max(0, k - int(0.5 * fps)), k - 1)
            post = list(range(k, f_land))
        else:
            kr = view.constants["release_frame"]
            pre = list(range(kr + 1, k - 1))
            post = _flight_frames(view, n, k, min(k + int(0.4 * fps), len(view.v[n]) - 1))
            t_land = view.constants["release_time_s"] + math.sqrt(2 * view.constants["drop_height"] / g)
            post = [f for f in post if view.v[n][f + 1, 2] < view.v[n][f, 2]] if n == view.record.get("body") else \
                [f for f in post if (f + 1) / fps < t_land - 1e-9]
        check_accel(n, pre, 1.0, f"{n} pre-event")
        # post-event g: the projectile's (gravity_x15: x1.5), the drop's edited ball (heavy_x13: the heavier, x1.3)
        g_fac = view.record.get("g_post", g) / g if view.sub == "projectile" or n == view.record.get("body") else 1.0
        check_accel(n, post, g_fac, f"{n} post-event")
        if g_fac != 1.0 and not (view.violated_law == "gravity" and close(g_fac, view.magnitude, scale=1.0)):
            err.append(f"{n}: post-event g x{g_fac:.6g} but violation {view.violated_law} x{view.magnitude}")
        if view.sub == "projectile":   # vx_x06: horizontal velocity x0.6 (the gravity law's axis independence)
            r = view.v[n][k, 0] / view.v[n][k - 1, 0]
            r_want = view.magnitude if view.violated_law == "gravity" and g_fac == 1.0 else 1.0
            if not close(r, r_want, scale=1.0):
                err.append(f"vx ratio {r:.9g} != {r_want}")
    if view.sub == "drop" and view.legal and not close(view.p["A"][:, 2], view.p["B"][:, 2], scale=1.0):
        err.append("legal drop pair does not fall equally")
    return err
