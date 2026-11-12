"""F12: conservation of angular momentum (law ``angular_momentum``). Top-down view of a frictionless table.

A puck (a disc: mass m_A, radius r, no spin) slides at speed v and hits a free bar (a box: half-length l, half-width
w, mass m_B, I_B = m_B (l^2 + w^2) / 3 about its centre) at rest, on the bar's long face at impact parameter b
(along the bar, from its centre). The puck comes in obliquely, 20-40 deg off the face normal towards the nearer bar
end: the frictionless contact keeps that tangential velocity, so the puck slides off past the end and clears the
disc the spinning bar sweeps (head-on, it lingers in front of the face and the bar sweeps through it again). Event = the contact (between frames, event_kind "contact"). Closed-form
frictionless impulse along the face normal n (A -> B) at the contact point c (``laws.rigid_impulse``):
J = (1 + e) v_n / (1/m_A + 1/m_B + (r_B x n)^2 / I_B); the puck gets -J n / m_A, the bar J n / m_B and spin
(r_B x n) J / I_B. Linear momentum and the angular momentum about the centre of mass (orbital + I w) are conserved.
The bar carries a high-contrast quadrant marker (``look.marked_bodies``) so its rotation is visible.

Twins (one pre_segment):
  legal          e = ``twins.e_hi`` (default 0.9)
  legal_e<lo>    e = ``twins.e_lo`` (default 0.4): a second legal outcome with less motion
  spin_x<k>      the legal (e_hi) outcome with the bar's spin x k (``twins.spin_ladder``, default 0.7, 0.8, 0.9,
                 1.1, 1.2, 1.3), linear velocities untouched: linear momentum conserved, L_ratio =
                 1 + (k - 1) I_B w_B' / L; violated_law "angular_momentum", magnitude k (an equality law: L2 reports
                 contrasts, not a kink). Energy grows for k > 1 when that exceeds the e_hi loss (flagged
                 energy_physical False: such a twin breaks two laws, angular momentum and energy).
  spin_ke_x<k>   KE-conserving angular-momentum violators (``twins.spin_ke_ladder``, default 0.8, 1.2; the primary
                 F12 contrast since 2026-11-12, audit M3): the bar's spin x k as spin_x<k>, and the energy that adds
                 (k > 1) or removes (k < 1) is taken from / given to the RADIAL part of the relative translational
                 velocity (along r_A - r_B at the contact; v_A', v_B' = V +- (m_B, m_A) / M u'). The CoM velocity V
                 (linear momentum), the orbital angular momentum (r x u has no radial part) and the total KE (= the
                 legal twin's, so energy_physical True) are unchanged; only L changes: L_ratio = 1 + (k - 1) I_B w_B' / L
                 exactly as spin_x<k>. Draws whose radial relative KE cannot pay for k > 1 are rejected.
Physically clean for the whole clip: draws in which any twin's puck and bar would touch again before the last
frame are rejected, and the group is placed so the bar stays wholly in frame from one second before the contact to
the clip end in every twin. clean_until_s per twin: the last frame before the puck leaves the frame (the clip end if
it never does; it stays in frame at least MIN_CLEAN_S after the contact).
Constants: e (this twin's legal-world restitution; spin twins e_hi), contact_point, contact_normal,
impact_parameter, inertias {A, B}, bar_half_length, bar_half_width, v0, approach_angle_rad.
"""
from __future__ import annotations

import math

import numpy as np

from physlaw.data.families import (G, LOOKS, Body, Scene, Static, Track, Twin, bounding_radius, event_time,
                                   frame_times, outside, pick_look, top_camera, visible_box)

SUBFAMILIES = {"spin": 200}
SPIN_LADDER = (0.7, 0.8, 0.9, 1.1, 1.2, 1.3)
SPIN_KE_LADDER = (0.8, 1.2)
E_HI, E_LO = 0.9, 0.4
CAM_HEIGHT = 14.0
PUCK_HALF_T = 0.06
BAR_HALF_T = 0.08
MIN_CLEAN_S = 0.6
SPEED = (1.2, 2.0)         # puck speed (m/s): the bar's drift over the clip must fit the frame
APPROACH_DEG = (20.0, 40.0)   # puck path vs the face normal, towards the nearer bar end (no second contact in the clip)
PLACE_MARGIN_PX = 4.0


def _quat_z(psi):
    return np.array([math.cos(psi / 2), 0.0, 0.0, math.sin(psi / 2)])


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _recontact(pA, pB, psi, r, l, w) -> np.ndarray:
    """[n] bool: the puck's disc (centre pA, radius r) overlaps the bar rectangle (centre pB, angle psi)."""
    d = pA[:, :2] - pB[:, :2]
    c, s = np.cos(psi), np.sin(psi)
    lx, ly = d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c
    qx, qy = np.clip(lx, -l, l), np.clip(ly, -w, w)
    return np.hypot(lx - qx, ly - qy) < r - 1e-6


def sample(rng, cfg, fmt, sub="spin"):
    from physlaw.eval.laws.collision import angular_momentum_z, rigid_impulse
    tw_cfg = (cfg.get("twins") if hasattr(cfg, "get") else None) or {}
    ladder = sorted(float(x) for x in (tw_cfg.get("spin_ladder") or SPIN_LADDER))
    ke_ladder = sorted(float(x) for x in (tw_cfg.get("spin_ke_ladder") or SPIN_KE_LADDER))
    e_hi, e_lo = float(tw_cfg.get("e_hi") or E_HI), float(tw_cfg.get("e_lo") or E_LO)
    cam = top_camera(fmt, height=CAM_HEIGHT)
    hw, hh = visible_box(cam, CAM_HEIGHT)
    T, fps = fmt["num_frames"], fmt["fps"]
    t = frame_times(fmt)
    te = event_time(rng, fmt, aligned=False)
    k = int(math.floor(te * fps + 1e-9)) + 1
    r = float(rng.uniform(0.22, 0.30))
    l, w = float(rng.uniform(1.0, 1.4)), float(rng.uniform(0.12, 0.18))
    v0 = float(rng.uniform(*SPEED))
    psi_u = float(rng.uniform(0, 2 * math.pi))
    u = np.array([math.cos(psi_u), math.sin(psi_u)])
    up = np.array([-u[1], u[0]])
    bimp = float(rng.uniform(0.35, 0.8)) * l * float(rng.choice([-1.0, 1.0]))
    # oblique approach, its tangential part towards the nearer bar end: the frictionless contact keeps it, so the puck
    # slides off past that end and clears the disc the spinning bar sweeps (head-on, it lingers and gets swept again)
    alpha = math.radians(float(rng.uniform(*APPROACH_DEG))) * math.copysign(1.0, bimp)
    a_in = math.cos(alpha) * u + math.sin(alpha) * up
    # geometry relative to the bar's centre; the whole group is placed in the frame once every twin's motion is known
    pA_c = -(w + r) * u + bimp * up
    c = -w * u + bimp * up
    look = pick_look(rng, exclude=("metal",))
    bar_look = pick_look(rng, exclude=(look,))
    puck = Body("A", "cylinder", r, float(rng.uniform(1.0, 3.0)), dict(LOOKS[look]), 1, length=PUCK_HALF_T)
    bar = Body("B", "box", l, 1.0, dict(LOOKS[bar_look]), 2, half=(l, w, BAR_HALF_T), spin_axis=2)
    bar.density = puck.mass / float(rng.uniform(0.3, 1.5)) / bar.volume          # m_A / m_B in [0.3, 1.5]
    m = np.array([puck.mass, bar.mass])
    I = np.array([puck.inertia, bar.inertia])
    psi_b = psi_u + math.pi / 2                                                  # bar long axis along u_perp
    zA, zB = PUCK_HALF_T + 0.0005, BAR_HALF_T + 0.0005
    q_puck = np.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0])          # disc axis -> world z
    tau = (t - te)[:, None]
    before = tau[:, 0] <= 0

    def tracks(post, cB):
        vA, vB, wB = post["v"][0], post["v"][1], float(post["w"][1, 2])
        pa0 = cB + pA_c
        pa = np.where(before[:, None], pa0 + v0 * a_in * tau, pa0 + vA * tau)
        pb = np.where(before[:, None], np.tile(cB, (T, 1)), cB + vB * tau)
        va = np.where(before[:, None], np.tile(v0 * a_in, (T, 1)), np.tile(vA, (T, 1)))
        vb = np.where(before[:, None], 0.0, np.tile(vB, (T, 1)))
        psi = np.where(before, psi_b, psi_b + wB * tau[:, 0])
        ww = np.zeros((T, 3))
        ww[~before, 2] = wB
        A = Track(np.c_[pa, np.full(T, zA)], np.c_[va, np.zeros(T)], np.tile(q_puck, (T, 1)), np.zeros((T, 3)))
        B = Track(np.c_[pb, np.full(T, zB)], np.c_[vb, np.zeros(T)], np.stack([_quat_z(x) for x in psi]), ww)
        return {"A": A, "B": B}, psi

    pre0 = {"p": np.stack([pA_c, np.zeros(2)]), "v": np.stack([v0 * a_in, np.zeros(2)]), "w": np.zeros((2, 3))}
    ctx0 = {"m": m, "I": I, "n": u, "c": c}
    legal_hi = rigid_impulse(pre0, ctx0, e_hi)
    spec = [("legal", legal_hi, e_hi, None, 1.0)]
    spec.append((f"legal_e{e_lo:g}".replace(".", ""), rigid_impulse(pre0, ctx0, e_lo), e_lo, None, 1.0))
    for kf in ladder:
        post = {"v": legal_hi["v"].copy(), "w": legal_hi["w"].copy(), "p": legal_hi["p"].copy()}
        post["w"][1, 2] *= kf
        spec.append((f"spin_x{kf:g}", post, e_hi, "angular_momentum", kf))
    for kf in ke_ladder:
        post = spin_ke_post(legal_hi, pA_c, m, I[1], kf)
        if post is None:
            return f"spin_ke_x{kf:g}_infeasible"
        spec.append((f"spin_ke_x{kf:g}", post, e_hi, "angular_momentum", kf))
    # Physically clean for the whole clip: the puck and the bar never touch again before the last frame (the bar's
    # constant spin would otherwise sweep it through the puck, which the frictionless closed form does not resolve).
    rel = {}
    for name, post, *_ in spec:
        trs, psi = tracks(post, np.zeros(2))
        if _recontact(trs["A"].p[k:], trs["B"].p[k:], psi[k:], r, l, w).any():
            return f"{name}_recontact_in_clip"
        rel[name] = trs
    # Placement: the bar wholly in frame over the whole in-frame span (one second before the contact to the clip end)
    # in every twin, the puck from that second up to MIN_CLEAN_S after the contact (it may leave the frame later).
    f0, f_puck = k - fps, k + int(math.ceil(MIN_CLEAN_S * fps))
    ppm = cam.px_per_m(CAM_HEIGHT)
    lo, hi = np.full(2, -np.inf), np.full(2, np.inf)
    for trs in rel.values():
        for b, p in ((bar, trs["B"].p[f0:]), (puck, trs["A"].p[f0:f_puck])):
            rad = bounding_radius(b) * 1.05 * CAM_HEIGHT / (CAM_HEIGHT - BAR_HALF_T) + (PLACE_MARGIN_PX + 1.0) / ppm
            lo = np.maximum(lo, (-np.array([hw, hh]) + rad - p[:, :2]).max(0))
            hi = np.minimum(hi, (np.array([hw, hh]) - rad - p[:, :2]).min(0))
    if (lo > hi).any():
        return "motion_wider_than_frame"
    cB = np.array([rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1])])
    c = c + cB
    pre = {"p": np.stack([cB + pA_c, cB]), "v": pre0["v"], "w": pre0["w"]}
    ctx = {"m": m, "I": I, "n": u, "c": c}
    L0 = angular_momentum_z(pre, ctx)
    ke = lambda st: 0.5 * float((m * (st["v"][:, :2] ** 2).sum(-1)).sum() + (I * st["w"][:, 2] ** 2).sum())
    twins = {}
    for name, post, e, law, kf in spec:
        trs, psi = tracks(post, cB)
        if outside(cam, bar, trs["B"].p[f0:]).any():
            return f"{name}_bar_leaves_frame"
        out = outside(cam, puck, trs["A"].p[k:])                 # the only secondary event left: the puck leaving
        stop = k + (int(np.argmax(out)) if out.any() else T - k)
        if (stop - k) / fps < MIN_CLEAN_S:
            return f"{name}_leaves_frame_within_{MIN_CLEAN_S}s"
        if _recontact(trs["A"].p[:k], trs["B"].p[:k], psi[:k], r, l, w).any():
            return "puck_overlaps_bar_before_contact"
        twins[name] = Twin(
            name, trs, legal=law is None, violated_law=law, violation_magnitude=kf if law else 0.0,
            flags={"energy_physical": ke(post) <= ke(pre) * (1 + 1e-12), "momentum_conserved": True},
            constants={"e": e},
            record={"kind": "contact", "e": e, "spin_factor": kf, "w_legal": float(legal_hi["w"][1, 2]),
                    "v_post": post["v"].tolist(), "w_bar": float(post["w"][1, 2]),
                    "L_ratio": angular_momentum_z(post, ctx) / L0, "ke_post": ke(post), "ke_pre": ke(pre),
                    "ke_legal": ke(legal_hi), "ke_conserving": name.startswith("spin_ke_")},
            clean_until_s=(stop - 1) / fps)
    table = Static("table", (30.0, 30.0, 0.002), (0.0, 0.0, -0.0015), look=dict(LOOKS["ice"]))
    return Scene("f12", "spin", [puck, bar], [table], cam, G, te, "contact", twins, fps,
                 constants={"g": G, "contact_point": [float(c[0]), float(c[1]), zA],
                            "contact_normal": [float(u[0]), float(u[1]), 0.0], "impact_parameter": bimp,
                            "inertias": {"A": float(I[0]), "B": float(I[1])}, "bar_half_length": l,
                            "bar_half_width": w, "v0": v0, "approach_angle_rad": alpha, "e_hi": e_hi, "e_lo": e_lo},
                 look={"marked_bodies": True}, inframe={"start": f0, "stop": T},
                 params={"looks": {"A": look, "B": bar_look}, "bar_angle_rad": psi_b, "bar_centre": cB.tolist()})


def spin_ke_post(legal: dict, r_rel, m, I_bar: float, k: float) -> dict | None:
    """The spin_ke_x<k> post state from the legal one: bar spin x k, the radial part (along r_rel = r_A - r_B at the
    contact) of the relative velocity u = v_A - v_B rescaled so the total KE equals the legal KE; V and the
    tangential part of u (hence the orbital angular momentum) unchanged. None if the radial KE cannot pay for k."""
    v, w = np.asarray(legal["v"], float), float(legal["w"][1, 2])
    M, mu = float(m.sum()), float(m[0] * m[1] / m.sum())
    V = (m[0] * v[0] + m[1] * v[1]) / M
    rh = np.asarray(r_rel, float)[:2] / np.linalg.norm(np.asarray(r_rel, float)[:2])
    u = v[0] - v[1]
    ur = float(u @ rh)
    ur2 = ur * ur - I_bar * w * w * (k * k - 1.0) / mu
    if ur2 <= 0.0:
        return None
    u2 = u + (math.copysign(math.sqrt(ur2), ur) - ur) * rh
    post = {"v": np.stack([V + m[1] / M * u2, V - m[0] / M * u2]), "w": np.asarray(legal["w"], float).copy(),
            "p": np.asarray(legal["p"], float).copy()}
    post["w"][1, 2] = k * w
    return post


# --------------------------------------------------------------------------- #
# acceptance (spec §2.4)
# --------------------------------------------------------------------------- #


def acceptance(view) -> list:
    """Constant velocities and spins on each side of the contact, puck never spins; linear momentum conserved in
    every twin; legal twins: L about the CoM conserved and the contact-point restitution = e; spin twins: the bar's
    spin = k x the legal one (spin_x<k>: with the legal linear velocities; spin_ke_x<k>: KE = the legal KE)."""
    from physlaw.data.families import close
    from physlaw.eval.laws.collision import angular_momentum_z
    c, k, fps, err = view.constants, view.k, view.fps, []
    te = view.t_event
    m = np.array([view.masses["A"], view.masses["B"]])
    I = np.array([c["inertias"]["A"], c["inertias"]["B"]])
    for n in "AB":
        v = view.v[n]
        if not (close(v[:k], np.tile(v[0], (k, 1)), scale=1.0) and close(v[k:], np.tile(v[k], (len(v) - k, 1)), scale=1.0)):
            err.append(f"{n}: velocity not constant on a side of the contact")
        dp = np.delete(np.diff(view.p[n], axis=0) * fps - 0.5 * (v[1:] + v[:-1]), k - 1, axis=0)
        if not close(dp, np.zeros_like(dp), scale=1.0):          # every frame pair but the one across the contact
            err.append(f"{n}: positions do not integrate the velocities")
    if np.abs(view.w["A"]).max() != 0.0 or np.abs(view.w["B"][:k]).max() != 0.0:
        err.append("puck spins, or the bar spins before the contact")
    at = lambda f: {"p": np.stack([view.p[n][f] + view.v[n][f] * (te - f / fps) for n in "AB"]),
                    "v": np.stack([view.v[n][f] for n in "AB"]), "w": np.stack([view.w[n][f] for n in "AB"])}
    pre, post = at(k - 1), at(k)
    ctx = {"m": m, "I": I}
    if not close(m @ post["v"], m @ pre["v"], scale=float(np.linalg.norm(m @ pre["v"]))):
        err.append("linear momentum not conserved")
    L_ratio = angular_momentum_z(post, ctx) / angular_momentum_z(pre, ctx)
    rec = view.record
    if view.legal:
        if not close(L_ratio, 1.0, scale=1.0):
            err.append(f"legal twin: L_ratio {L_ratio:.9g} != 1")
        n = np.asarray(c["contact_normal"], float)[:2]
        cp = np.asarray(c["contact_point"], float)[:2]
        rB = cp - pre["p"][1, :2]
        perp = np.array([-rB[1], rB[0]])
        rel = lambda st: float((st["v"][0, :2] - st["v"][1, :2] - st["w"][1, 2] * perp) @ n)
        if not close(-rel(post) / rel(pre), c["e"], scale=1.0):
            err.append(f"restitution {-rel(post) / rel(pre):.9g} != e {c['e']}")
    else:
        if view.violated_law != "angular_momentum" or not close(view.w["B"][k, 2], rec["spin_factor"] * rec["w_legal"],
                                                                 scale=abs(rec["w_legal"])):
            err.append("spin twin: bar spin != k x legal")
        if not close(L_ratio, rec["L_ratio"], scale=1.0) or close(L_ratio, 1.0, scale=1e3):
            err.append(f"spin twin: L_ratio {L_ratio:.9g}")
        ke = lambda st: 0.5 * float(m @ (st["v"][:, :2] ** 2).sum(-1) + I @ st["w"][:, 2] ** 2)
        if rec.get("ke_conserving") and not close(ke(post), rec["ke_legal"], scale=rec["ke_legal"]):
            err.append(f"spin_ke twin: KE {ke(post):.9g} != legal KE {rec['ke_legal']:.9g}")
    return err
