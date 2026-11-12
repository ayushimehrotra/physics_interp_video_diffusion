"""F6: Coulomb friction on one uniform surface, mass-independent slides, force-free glides (spec §2.3 F6).

  slide  one block is ALREADY sliding at frame 0 on a uniform textured floor (the same surface everywhere and in every
         twin; no band, nothing marks the branch) and decelerates at a = mu g (Coulomb), so the pre-event frames show
         the clip's own friction rate. Event = an invisible branch on a frame mid-slide (event_kind "branch"); the legal
         stop comes 0.9-1.6 s later and 2.1-2.4 s before the clip end, so reverse slides back >= ~150 px (median ~200)
         at 832 px. Oblique camera fitted to every twin's whole path (the block >= MIN_BLOCK_PX wide at 832 px), so
         every twin stays in frame to the clip end. (No no_friction twin since 2026-11-12, user: its long glide set the
         camera span and shrank every other twin's effect.)
         Twins (identical before the branch):
           legal        the same deceleration until v = 0, then at rest (static friction holds it);
           reverse      the same acceleration continued through v = 0: the block slides back (the naive
                        constant-acceleration extrapolation; friction cannot do it), magnitude 1;
           viscous      drag ~ v from the branch, matched to the legal deceleration there: v = v_e exp(-lam tau),
                        lam = a / v_e; creeps on, no clean stop, magnitude exp(-1) (speed left at the legal stop);
           decel_x<k>   deceleration x k from the branch (``twins.slide_decel_ladder``, default 0.7, 0.85, 1.15,
                        1.3): an equality law, L2 contrasts (the pair x0.85 + x1.15 is speed-matched), magnitude k.
  pair   mass independence: two blocks of the same size and visibly different material / density (the heavier is
         metal) side by side on the same surface, the same speed and deceleration at frame 0; a steeper camera
         (45-55 deg) and a lateral gap that keep the two silhouettes apart (no occlusion). The branch is in the first
         half of the event window (frames 112-120) and the legal stop 0.5-0.6 s before the clip end, so most of the
         slide is after the branch: heavy_x13 stops ~35-50 px (median ~43) short of the legal stop at 832 px.
         Twins: legal (both stop together), heavy_x13 (the heavier decelerates x1.3 from the branch).
  glide  Newton's first law (folded into the momentum law): one puck slides on frictionless ice (top-down, nothing
         else in view); a branch on a frame mid-glide, no visible event. Twins: legal (same velocity), speed_x<s> for s
         in ``twins.glide_speed_ladder`` (default 0.7, 0.8, 0.9, 1.1, 1.2, 1.3: |v'| = s |v|, same direction;
         violated_law "momentum", magnitude s; L2 contrasts, not a kink) and turn_<deg> (``twins.glide_turn_deg``,
         default 20: same speed, direction turned; magnitude = deg / 180 = inertia_turn). clean_until_s per twin = the
         last instant the puck is wholly in frame.

Every slide / pair twin violates law "friction" (laws.friction: decel_const, stop_hold, mass_ratio) except legal;
reverse also creates energy (flags.energy_physical False). glide speed_x>1 creates kinetic energy with no force
(flags.energy_physical False: these twins break two laws, Newton I and energy; L2 reports the slower and faster halves
of each speed-matched pair separately). Labels: constants t_stop_s (the legal stop, shared by the
group), v0 (speed at t = 0), a (legal deceleration), mu = a / g, direction, start {body: p(0)}; per twin t_stop_twin_s
(when the twin comes to rest, if it does) and an exact event_state (pre / post accelerations at the branch).
clean_until_s = the clip end (nothing leaves the frame, no secondary event). Closed form throughout; static friction
equals kinetic friction, so a block that stops stays stopped. PyBullet check: the legal twin from frame 0 on a slab
with friction mu.
"""
from __future__ import annotations

import math

import numpy as np

from physlaw.data.families import (G, LOOKS, Body, Scene, Static, Track, Twin, event_time, frame_times, outside,
                                   pick_look, set_clean_until, top_camera, visible_box)
from physlaw.data.families.f10_rolling import oblique_camera

SUBFAMILIES = {"slide": 300, "pair": 200, "glide": 150}
GLIDE_LADDER = (0.7, 0.8, 0.9, 1.1, 1.2, 1.3)
DECEL_LADDER = (0.7, 0.85, 1.15, 1.3)
HEAVY_FACTOR = 1.3
HALF = (0.35, 0.55)            # block half-size (m)
SPAN_BLOCKS = (5.5, 8.0)       # length of the union of the twins' paths / block width
# legal stop -> clip end. slide: long, so reverse slides back >= ~150 px at 832 px; pair: short (and the branch early),
# so the heavier block's x1.3 deceleration stops it ~40 px short of the legal stop (both 2026-11-12)
POST_STOP_S = {"slide": (2.1, 2.4), "pair": (0.5, 0.6)}
PITCH_DEG = (30.0, 42.0)       # camera depression angle (slide)
PAIR_PITCH_DEG = (45.0, 55.0)  # pair: steeper, so the near block does not hide the far one
PAIR_GAP = (1.3, 1.6)          # pair separation / the least that keeps the silhouettes apart, 2 h (1 + cot pitch)
FOCAL_MM = 100.0
FILL = 0.96                    # the paths (bounding discs) fill this share of the image width
MIN_BLOCK_PX = 70.0
# the surface: concrete-like two-scale world-space noise (generate.surface_texture), identical in every twin
FLOOR = {"color": (0.62, 0.58, 0.52), "metallic": 0.0, "roughness": 0.8, "specular": 0.2, "ior": 1.45,
         "texture": {"scale": 22.0, "contrast": 0.45}}


def sample(rng, cfg, fmt, sub):
    return {"slide": _slide, "pair": _pair, "glide": _glide}[sub](rng, cfg, fmt)


def _twin_cfg(cfg) -> dict:
    return (cfg.get("twins") if hasattr(cfg, "get") else None) or {}


# --------------------------------------------------------------------------- closed form

def slide_motion(t, te: float, v0: float, a: float, kind: str = "decel", k: float = 1.0):
    """(s, u) distance along the motion and signed speed at times t of a block sliding at v0 at t = 0 and decelerating
    at a; from the branch te on: ``decel`` deceleration k a until rest (k = 1: legal), ``reverse`` the acceleration -a
    through v = 0, ``viscous`` u = u_e exp(-lam tau) with lam = a / u_e (the legal deceleration at the branch),
    Before te every kind is the same expression (bit-identical pre-event frames)."""
    t = np.asarray(t, float)
    s, u = v0 * t - 0.5 * a * t * t, v0 - a * t
    if kind == "reverse":
        return s, u
    se, ue = v0 * te - 0.5 * a * te * te, v0 - a * te
    tau = np.clip(t - te, 0.0, None)
    if kind == "decel":
        ak = k * a
        tt = np.minimum(tau, ue / ak)
        sp, up = se + ue * tt - 0.5 * ak * tt * tt, np.maximum(ue - ak * tau, 0.0)
    elif kind == "viscous":
        lam = a / ue
        sp, up = se + ue / lam * (1.0 - np.exp(-lam * tau)), ue * np.exp(-lam * tau)
    else:
        raise ValueError(kind)
    post = t > te
    return np.where(post, sp, s), np.where(post, up, u)


def _track(t, start, d, s, u) -> Track:
    T = len(t)
    return Track(np.asarray(start, float) + s[:, None] * d, u[:, None] * d, np.tile((1.0, 0.0, 0.0, 0.0), (T, 1)),
                 np.zeros((T, 3)))


def _post_accel(kind: str, k: float, a: float) -> float:
    """Acceleration along the motion just after the branch."""
    return {"decel": -k * a, "reverse": -a, "viscous": -a}[kind]


def _fit_camera(fmt, lo, hi, radius: float, pitch: float) -> object:
    """Oblique camera (yaw 0: the motion runs along image x at constant depth) looking at the centre of the box
    [lo, hi] (world points), at the distance where every point's bounding disc (``radius``) fits FILL of the image."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    c = 0.5 * (lo + hi)
    W, H = fmt["resolution"]
    mx, my = 0.5 * (1 - FILL) * W, 0.5 * (1 - FILL) * H

    def fits(dist):
        cam = oblique_camera(fmt, c, dist, 0.0, pitch, FOCAL_MM)
        uvz = cam.project(corners)
        r = radius * cam.K()[0, 0] / uvz[:, 2]
        return bool(((uvz[:, 0] - r >= mx) & (uvz[:, 0] + r <= W - mx) & (uvz[:, 1] - r >= my)
                     & (uvz[:, 1] + r <= H - my)).all())
    a_, b_ = 0.5, 500.0
    for _ in range(60):
        m = 0.5 * (a_ + b_)
        a_, b_ = (a_, m) if fits(m) else (m, b_)
    return oblique_camera(fmt, c, b_, 0.0, pitch, FOCAL_MM)


def _block_px(cam, p, half) -> float:
    """Apparent width (px) of a block of half-size ``half`` at the farthest of the positions p [n, 3]."""
    depth = cam.project(np.asarray(p, float))[:, 2].max()
    return 2.0 * half * cam.px_per_m(depth)


def _slide_scene(rng, cfg, fmt, sub: str):
    T, fps = fmt["num_frames"], fmt["fps"]
    t = frame_times(fmt)
    if sub == "pair":                  # branch in the first half of the event window: the most slide after it
        lo, hi = fmt["event_frame_range"]
        te = int(rng.integers(lo, (lo + hi) // 2 + 1)) / fps
    else:
        te = event_time(rng, fmt, aligned=True)
    t_stop = t[-1] - float(rng.uniform(*POST_STOP_S[sub]))
    half = float(rng.uniform(*HALF))
    sgn = float(rng.choice([-1.0, 1.0]))
    d = np.array([sgn, 0.0, 0.0])
    names = ["A"] if sub == "slide" else ["A", "B"]
    gap = float(rng.uniform(*PAIR_GAP))
    # twins: (name, kind, deceleration factor per body, violated_law, magnitude)
    if sub == "slide":
        ladder = sorted(float(k) for k in (_twin_cfg(cfg).get("slide_decel_ladder") or DECEL_LADDER))
        spec = [("legal", "decel", {"A": 1.0}, None, 0.0), ("reverse", "reverse", {"A": 1.0}, "friction", 1.0),
                ("viscous", "viscous", {"A": 1.0}, "friction", math.exp(-1.0))]
        spec += [(f"decel_x{k:g}", "decel", {"A": k}, "friction", k) for k in ladder]
        heavy = None
    else:
        heavy = str(rng.choice(names))
        spec = [("legal", "decel", {"A": 1.0, "B": 1.0}, None, 0.0),
                ("heavy_x13", "decel", {n: (HEAVY_FACTOR if n == heavy else 1.0) for n in names}, "friction",
                 HEAVY_FACTOR)]
    # unit deceleration first: every length scales with a for fixed times, so a follows from the span
    motions = {(kind, k) for _, kind, ks, _, _ in spec for k in ks.values()}
    span1 = max(float(np.max(slide_motion(t, te, t_stop, 1.0, kind, k)[0])) for kind, k in motions)
    a = float(rng.uniform(*SPAN_BLOCKS)) * 2.0 * half / span1
    pitch = float(rng.uniform(*(PITCH_DEG if sub == "slide" else PAIR_PITCH_DEG)))
    sep = 2.0 * half * (1.0 + 1.0 / math.tan(math.radians(pitch))) * gap
    ys = {"A": 0.0} if sub == "slide" else {"A": sep / 2, "B": -sep / 2}
    v0 = a * t_stop
    x0 = -sgn * 0.5 * a * span1
    start = {n: np.array([x0, ys[n], half]) for n in names}
    tracks = {name: {n: _track(t, start[n], d, *slide_motion(t, te, v0, a, kind, ks[n])) for n in names}
              for name, kind, ks, _, _ in spec}
    allp = np.concatenate([tr.p for trs in tracks.values() for tr in trs.values()])
    cam = _fit_camera(fmt, allp.min(0) - [0, 0, half], allp.max(0) + [0, 0, half], half * math.sqrt(3.0), pitch)
    px = _block_px(cam, allp, half)
    if px < MIN_BLOCK_PX:
        return f"block_{px:.0f}px"
    if sub == "pair":                  # image rows depend on (y, z) only (yaw 0): the silhouettes must not overlap
        cube = np.array([[x, y, z] for x in (-half, half) for y in (-half, half) for z in (-half, half)])
        rows = {n: cam.project(start[n] + cube)[:, 1] for n in names}
        if rows["A"].max() + 4.0 > rows["B"].min():
            return "pair_occlusion"
    if sub == "slide":
        look = pick_look(rng)
        looks = {"A": look}
        bodies = [Body("A", "cube", half, float(rng.uniform(0.5, 3.0)), dict(LOOKS[look]), 1)]
        ratio = 1.0
    else:
        rho_lo = float(np.exp(rng.uniform(np.log(0.5), np.log(1.5))))
        ratio = float(np.exp(rng.uniform(np.log(2.0), np.log(5.0))))
        light = pick_look(rng, exclude=("metal",))
        looks = {n: ("metal" if n == heavy else light) for n in names}
        bodies = [Body(n, "cube", half, rho_lo * (ratio if n == heavy else 1.0), dict(LOOKS[looks[n]]), i + 1)
                  for i, n in enumerate(names)]
    k_e = int(round(te * fps))
    twins = {}
    for name, kind, ks, law, mag in spec:
        pe = [tracks[name][n].p[k_e].tolist() for n in names]
        ve = [tracks[name][n].v[k_e].tolist() for n in names]
        st = lambda accs: {"t_s": te, "objects": names, "p": pe, "v": ve, "w": [[0.0] * 3] * len(names),
                           "a": [(x * d).tolist() for x in accs]}
        ue = v0 - a * te
        stops = [te + ue / (ks[n] * a) for n in names if kind == "decel"]
        twins[name] = Twin(name, tracks[name], legal=law is None, violated_law=law, violation_magnitude=mag,
                           event_state={"pre": st([-a] * len(names)), "post": st([_post_accel(kind, ks[n], a)
                                                                                   for n in names])},
                           constants={"mu": a / G, **({"t_stop_twin_s": min(stops)} if stops else {})},
                           flags={"energy_physical": kind != "reverse"},
                           record={"kind": kind, "decel_factor": dict(ks), "heavier": heavy,
                                   **({"lambda": a / ue} if kind == "viscous" else {})})
    floor = Static("slab", (40.0, 40.0, 0.25), (0.0, 0.0, -0.25), (1.0, 0.0, 0.0, 0.0), dict(FLOOR))
    sc = Scene("f6", sub, bodies, [floor], cam, G, te, "branch", twins, fps,
               constants={"g": G, "v0": v0, "a": a, "mu": a / G, "t_stop_s": t_stop, "half_size": half,
                          "direction": d.tolist(), "start": {n: start[n].tolist() for n in names},
                          **({"heavier": heavy, "mass_ratio_heavy_over_light": ratio, "separation": sep}
                             if heavy else {})},
               look={"floor": False}, inframe={"start": 0, "stop": T},
               sim_init={"twin": "legal", "t0": 0.0, "until": min(t_stop + 0.5, float(t[-1]))},
               params={"looks": looks, "pitch_deg": pitch, "block_px": px})
    return set_clean_until(sc) or sc


def _slide(rng, cfg, fmt):
    return _slide_scene(rng, cfg, fmt, "slide")


def _pair(rng, cfg, fmt):
    return _slide_scene(rng, cfg, fmt, "pair")


# --------------------------------------------------------------------------- glide (Newton I)

def _glide(rng, cfg, fmt):
    tw_cfg = _twin_cfg(cfg)
    ladder = sorted(float(s) for s in (tw_cfg.get("glide_speed_ladder") or GLIDE_LADDER))
    turn = float(tw_cfg.get("glide_turn_deg") or 20.0)
    cam = top_camera(fmt)
    T, fps = fmt["num_frames"], fmt["fps"]
    t = frame_times(fmt)
    te = event_time(rng, fmt, aligned=True)
    k = int(round(te * fps)) + 1
    hw, hh = visible_box(cam, cam.position[2])
    r, hth = float(rng.uniform(0.25, 0.35)), 0.06
    v0 = float(rng.uniform(1.0, 1.8))
    psi = float(rng.uniform(0, 2 * math.pi))
    d = np.array([math.cos(psi), math.sin(psi), 0.0])
    z = hth + 0.0025                                   # resting on the ice sheet (0.5 mm proud of the floor)
    p_e = np.array([rng.uniform(-hw + 3.0, hw - 3.0), rng.uniform(-hh + 2.2, hh - 2.2), z]) - d * 0.3
    look = pick_look(rng)
    puck = Body("A", "cylinder", r, float(rng.uniform(0.5, 3.0)), dict(LOOKS[look]), 1, length=hth)
    q = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)        # disc axis (body y) -> world z
    tau = (t - te)[:, None]

    def track(v_post):
        v = np.where(tau <= 0, v0 * d, v_post)
        return Track(p_e + v * tau, v, np.tile(q, (T, 1)), np.zeros((T, 3)))

    c, s_ = math.cos(math.radians(turn)), math.sin(math.radians(turn))
    sgn = float(rng.choice([-1.0, 1.0]))
    d_turn = np.array([c * d[0] - sgn * s_ * d[1], sgn * s_ * d[0] + c * d[1], 0.0])
    spec = [("legal", v0 * d, None, 0.0, {"kind": "branch"})]
    spec += [(f"speed_x{x:g}", x * v0 * d, "momentum", x, {"kind": "branch", "speed_factor": x}) for x in ladder]
    spec.append((f"turn_{turn:g}", v0 * d_turn, "momentum", turn / 180.0,
                 {"kind": "branch", "turn_deg": sgn * turn, "speed_factor": 1.0}))
    twins = {}
    for name, v_post, law, mag, rec in spec:
        tr = track(v_post)
        out = outside(cam, puck, tr.p[k:], 4.0)
        stop = k + (int(np.argmax(out)) if out.any() else T - k)
        if (stop - k) / fps < 1.0:
            return f"{name}_leaves_frame_within_1s"
        twins[name] = Twin(name, {"A": tr}, legal=law is None, violated_law=law, violation_magnitude=mag,
                           flags={"energy_physical": float(np.linalg.norm(v_post)) <= v0 * (1 + 1e-12)},
                           record={**rec, "v_pre": (v0 * d).tolist(), "v_post": list(map(float, v_post))},
                           clean_until_s=(stop - 1) / fps)
    ice = Static("ice", (30.0, 30.0, 0.002), (0.0, 0.0, 0.0005), (1.0, 0.0, 0.0, 0.0), dict(LOOKS["ice"]))
    return Scene("f6", "glide", [puck], [ice], cam, G, te, "branch", twins, fps,
                 constants={"g": G, "v0": v0, "direction_rad": psi, "mu": 0.0, "branch_position": p_e.tolist(),
                            "radius": r, "half_thickness": hth},
                 inframe={"start": k - fps, "stop": T}, params={"look": look, "turn_sign": sgn})


# --------------------------------------------------------------------------- acceptance

def acceptance(view) -> list:
    """glide: constant velocity on each side of the branch, |v'| / |v| and the turn as recorded (legal: unchanged).
    slide / pair: positions and velocities equal the closed form of the twin's kind (legal: Coulomb, whatever the
    record says) to 1e-6, and from finite differences of the per-frame speeds: deceleration a = mu g before the branch
    along the constant direction; after it legal / decel_x<k> / heavy_x13 decelerate at k a until rest and then stay
    at rest (legal: rest from t_stop_s on, >= 0.5 s before the clip end), reverse keeps -a through v = 0 (the speed
    along the direction turns negative), viscous decays by exp(-lam / fps) per frame with lam u_e = a."""
    from physlaw.data.families import close
    k, fps, err = view.k, view.fps, []
    if view.sub == "glide":
        v, p = view.v["A"], view.p["A"]
        if not (close(v[:k], np.tile(v[0], (k, 1)), scale=1.0) and close(v[k:], np.tile(v[k], (len(v) - k, 1)), scale=1.0)):
            err.append("velocity not constant on a side of the branch")
        dp = np.delete(np.diff(p, axis=0) * fps - 0.5 * (v[1:] + v[:-1]), k - 1, axis=0)
        if not close(dp, np.zeros_like(dp), scale=1.0):          # every frame pair but the one across the branch
            err.append("positions do not integrate the velocities")
        s, ang = np.linalg.norm(v[k]) / np.linalg.norm(v[0]), math.degrees(
            math.atan2(float(v[0, 0] * v[k, 1] - v[0, 1] * v[k, 0]), float(v[0] @ v[k])))
        want_s = view.record.get("speed_factor", 1.0) if view.violated_law else 1.0
        want_a = view.record.get("turn_deg", 0.0) if view.violated_law else 0.0
        if not close(s, want_s, scale=1.0) or not close(ang, want_a, scale=180.0):
            err.append(f"|v'|/|v| = {s:.9g}, turn {ang:.6g} deg != {want_s}, {want_a}")
        if view.violated_law and close(s, 1.0, scale=1.0) and close(ang, 0.0, scale=180.0):
            err.append("violating twin keeps its velocity")
        return err
    c = view.constants
    g, a, v0, te, t_stop = c["g"], c["a"], c["v0"], view.t_event, c["t_stop_s"]
    d = np.asarray(c["direction"], float)
    kind = "decel" if view.legal else view.record["kind"]
    T = len(next(iter(view.v.values())))
    t = np.arange(T) / fps
    if not close(a, c["mu"] * g, scale=g):
        err.append(f"a {a:.9g} != mu g")
    if not t_stop + 0.5 <= t[-1] + 1e-9 or not te < t_stop:
        err.append(f"legal stop at {t_stop:.4g} s: not after the branch and >= 0.5 s before the clip end")
    for n in view.p:
        kf = 1.0 if view.legal else float((view.record.get("decel_factor") or {}).get(n, 1.0))
        s, u = slide_motion(t, te, v0, a, kind, kf)
        start = np.asarray(c["start"][n], float)
        if not close(view.p[n], start + s[:, None] * d, scale=1.0) or not close(view.v[n], u[:, None] * d, scale=1.0):
            err.append(f"{n}: not the closed-form {kind} x{kf:g} slide")
        ua = view.v[n] @ d                                             # signed speed along the motion
        if not close(view.v[n] - ua[:, None] * d, np.zeros_like(view.v[n]), scale=1.0):
            err.append(f"{n}: velocity leaves the line of motion")
        acc = -np.diff(ua) * fps                                       # deceleration over frame pairs (f, f + 1)
        if not close(acc[:k - 1], a, scale=g):
            err.append(f"{n}: pre-branch deceleration != mu g")
        post, up = acc[k:], ua[k:]
        if kind == "decel":
            moving = up[1:] > 0
            if not close(post[moving], kf * a, scale=g) or np.any(up < -1e-12):
                err.append(f"{n}: post-branch deceleration != {kf:g} mu g")
            rest = np.nonzero(up <= 0)[0]
            if len(rest) and not close(up[rest[0]:], 0.0, scale=1.0):
                err.append(f"{n}: does not stay at rest")
            if view.legal and not (len(rest) and t[k + rest[0]] <= t_stop + 1.0 / fps + 1e-9
                                   and close(ua[t >= t_stop - 1e-9], 0.0, scale=1.0)):
                err.append(f"{n}: legal slide not at rest from t_stop_s")
        elif kind == "reverse":
            if not close(post, a, scale=g) or not up[-1] < 0:
                err.append(f"{n}: reverse does not keep -a through v = 0")
        elif kind == "viscous":
            lam = float(view.record["lambda"])
            if not close(up[1:] / up[:-1], math.exp(-lam / fps), scale=1.0) or not close(lam * (v0 - a * te), a, scale=g):
                err.append(f"{n}: viscous speed does not decay as exp(-lam t) from the legal deceleration")
    if view.legal != (view.violated_law is None):
        err.append("legal / violated_law disagree")
    elif view.violated_law is not None:
        mags = {"reverse": 1.0, "viscous": math.exp(-1.0)}
        want = mags.get(kind, max((view.record.get("decel_factor") or {"A": 1.0}).values()))
        if view.violated_law != "friction" or not close(view.magnitude, want, scale=1.0):
            err.append(f"violated_law / magnitude {view.violated_law} {view.magnitude} != friction {want}")
        if kind == "decel" and all(f == 1.0 for f in (view.record.get("decel_factor") or {}).values()):
            err.append("violating twin is the legal slide")
    return err
