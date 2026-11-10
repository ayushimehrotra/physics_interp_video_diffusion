"""F10: rolling (spec §2.3 F10, P2). A solid cylinder and a hollow one (a tube, inner radius
0.9 r), same radius and mass, held side by side on one incline and released together at the
event frame. Rolling without slipping: a = g sin(th) / (1 + I / (m r^2)), with I/(m r^2) =
1/2 (solid) and (1 + 0.9^2)/2 = 0.905 (tube), so the solid arrives first.

  legal         both at their own rolling acceleration
  hollow_first  accelerations swapped                     (rolling; hollow gains energy)
  tie           the tube rolls at the solid's acceleration (rolling; both arrive together)

Kubric has no cylinder primitive: the bodies are generated .obj meshes (``mesh_obj``),
textured with an object-space checker so the rotation is visible. The camera looks
obliquely along the cylinder axes so both lanes and the end faces (disc vs ring) show.
The slab runs out of the frame (no floor), as in F6 incline.
"""
from __future__ import annotations

import math

import numpy as np

from physlaw.data.families import (G, LOOKS, Body, Camera, Scene, Static, Track, Twin, event_time, frame_times,
                                   pick_look, quat_y, visible_box)
from physlaw.data.families.f8_ramp import RAMP_THICK, mirror_static

SUBFAMILIES = {"rolling": 150}
TUBE_INNER = 0.9
LANE = 1.3            # |y| of each lane centre


def oblique_camera(fmt, target, dist=20.0, yaw_deg=-30.0, pitch_deg=45.0) -> Camera:
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    off = dist * np.array([math.sin(yaw) * math.cos(pitch), -math.cos(yaw) * math.cos(pitch), math.sin(pitch)])
    return Camera(tuple(np.asarray(target) + off), tuple(target), (0.0, 0.0, 1.0),
                  sensor_w_mm=fmt["sensor_width_mm"], resolution=tuple(fmt["resolution"]), aspect=fmt["aspect"])


def k_of(b: Body) -> float:
    return b.inertia / (b.mass * b.size ** 2)


def sample(rng, cfg, fmt, sub="rolling"):
    T, fps = fmt["num_frames"], fmt["fps"]
    t = frame_times(fmt)
    te = event_time(rng, fmt, aligned=True)
    kr = int(round(te * fps))
    th = math.radians(rng.uniform(12, 25))
    side = float(rng.choice([-1.0, 1.0]))
    r = float(rng.uniform(0.35, 0.5))
    half_len = float(rng.uniform(0.3, 0.4))
    target = (0.0, 0.0, 3.0)
    cam = oblique_camera(fmt, target, yaw_deg=-30.0 * side)
    hw, hh = visible_box(cam, 20.0)
    d = np.array([math.cos(th), 0.0, -math.sin(th)])
    n = np.array([math.sin(th), 0.0, math.cos(th)])
    start_x = -hw + 2.0 + float(rng.uniform(0.0, 1.0))
    start = np.array([start_x, 0.0, target[2] + hh - 1.2 - r - float(rng.uniform(0.0, 1.0))])
    looks = (pick_look(rng, exclude=("metal",)),) * 2
    rho = float(rng.uniform(0.6, 2.0))
    solid = Body("A", "cylinder", r, rho, dict(LOOKS[looks[0]]), 1, length=half_len)
    tube = Body("B", "tube", r, 1.0, dict(LOOKS[looks[1]]), 2, length=half_len)
    tube.density = solid.mass / (tube.volume)                          # equal masses: only I differs
    L = 40.0
    top = start - r * n - d * (r * 1.5 + 0.3)
    slab = mirror_static(Static("slab", (L / 2, LANE + half_len + 0.3, RAMP_THICK / 2),
                                tuple(top + d * L / 2 - 0.5 * RAMP_THICK * n), quat_y(th), dict(LOOKS["ramp"]),
                                friction=1.0), side)
    a_legal = {b.name: G * math.sin(th) / (1 + k_of(b)) for b in (solid, tube)}
    twins = {}
    for name, acc, law, mag in (
            ("legal", a_legal, None, 0.0),
            ("hollow_first", {"A": a_legal["B"], "B": a_legal["A"]}, "rolling", a_legal["A"] / a_legal["B"]),
            ("tie", {"A": a_legal["A"], "B": a_legal["A"]}, "rolling", a_legal["A"] / a_legal["B"])):
        tracks = {}
        for b, y in ((solid, -LANE), (tube, LANE)):
            dt = np.clip(t - te, 0.0, None)
            s = 0.5 * acc[b.name] * dt * dt
            sd = acc[b.name] * dt
            p = start + s[:, None] * d + np.array([0.0, y, 0.0])
            v = sd[:, None] * d
            ang = s / r
            q = np.stack([np.cos(ang / 2), np.zeros_like(ang), np.sin(ang / 2), np.zeros_like(ang)], -1)
            w = np.stack([np.zeros_like(sd), sd / r, np.zeros_like(sd)], -1)
            if side < 0:
                p[:, 0] *= -1
                v[:, 0] *= -1
                q[:, 2] *= -1
                w[:, 1] *= -1
            tracks[b.name] = Track(p, v, q, w)
        twins[name] = Twin(name, tracks, legal=law is None, violated_law=law, violation_magnitude=mag,
                           flags={"energy_physical": name != "hollow_first" and name != "tie"},
                           record={"kind": "release", "accel": dict(acc)})
    for tw in twins.values():          # the two lanes must not overlap on screen
        uv = {n: cam.project(tw.tracks[n].p) for n in "AB"}
        gap = np.linalg.norm(uv["A"][:, :2] - uv["B"][:, :2], axis=1)
        reach = sum(cam.K()[0, 0] * math.hypot(r, half_len) / uv[n][:, 2] for n in "AB")
        if (gap < 1.05 * reach).any():
            return "lanes_overlap_in_view"
    sc = Scene("f10", "rolling", [solid, tube], [slab], cam, G, te, "release", twins, fps,
               constants={"g": G, "theta_rad": th, "inertia_factors": {"A": k_of(solid), "B": k_of(tube)},
                          "accel_legal": a_legal, "release_frame": kr, "release_time_s": te, "direction": side,
                          "tube_inner_ratio": TUBE_INNER, "half_length": half_len},
               look={"floor": False, "checker_bodies": True},
               inframe={"start": 0, "stop": min(T, kr + 1 + int(0.8 * fps))},
               sim_init={"twin": "legal", "t0": te, "until": min(te + 0.8, t[-1])},
               params={"looks": looks, "start": start.tolist()})
    return sc


def mesh_obj(body: Body, path, segments: int = 48) -> None:
    """Write a cylinder (or tube) .obj along the y axis, centred on the origin, in Blender axes."""
    r, h = body.size, body.length
    ri = r * TUBE_INNER if body.shape == "tube" else 0.0
    a = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    V, F = [], []

    def ring(rad, y):
        base = len(V)
        V.extend((rad * math.cos(x), y, rad * math.sin(x)) for x in a)
        return base

    o0, o1 = ring(r, -h), ring(r, h)
    for i in range(segments):
        j = (i + 1) % segments
        F.append((o0 + i, o0 + j, o1 + j, o1 + i))
    if ri > 0:
        i0, i1 = ring(ri, -h), ring(ri, h)
        for i in range(segments):
            j = (i + 1) % segments
            F.append((i0 + j, i0 + i, i1 + i, i1 + j))
            F.append((o0 + j, o0 + i, i0 + i, i0 + j))      # end rings
            F.append((o1 + i, o1 + j, i1 + j, i1 + i))
    else:
        c0, c1 = len(V), len(V) + 1
        V.extend([(0.0, -h, 0.0), (0.0, h, 0.0)])
        for i in range(segments):
            j = (i + 1) % segments
            F.append((c0, o0 + j, o0 + i))
            F.append((c1, o1 + i, o1 + j))
    with open(path, "w") as f:
        f.writelines(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in V)
        f.writelines("f " + " ".join(str(i + 1) for i in face) + "\n" for face in F)


def acceptance(view) -> list:
    """Rolling: a = g sin th / (1 + I/(m r^2)) per body (the violating twins' accelerations as recorded),
    and rolling without slipping: w = v / r."""
    from physlaw.data.families import close
    g, k, err = view.constants["g"], view.k, []
    th = view.constants["theta_rad"]
    acc = view.record["accel"]
    for n in view.p:
        legal_a = g * math.sin(th) / (1 + view.constants["inertia_factors"][n])
        a = np.linalg.norm(view.accel(n, k, len(view.v[n]) - 1), axis=1)
        if not close(a, acc[n], scale=g):
            err.append(f"{n}: accel {a.mean():.9g} != {acc[n]:.9g}")
        if view.legal and not close(acc[n], legal_a, scale=g):
            err.append(f"{n}: legal twin off the rolling law")
        if np.abs(view.v[n][:k]).max() != 0.0:
            err.append(f"{n} moves before the release")
        w = view.w[n][:, 1] * view.constants.get("direction", 1.0) if view.w else None
        if w is not None and not close(np.abs(w) * view.constants["sizes"][n], np.linalg.norm(view.v[n], axis=1), scale=g):
            err.append(f"{n}: slips (|w| r != |v|)")
    if not view.legal and close(acc["B"], g * math.sin(th) / (1 + view.constants["inertia_factors"]["B"]), scale=g):
        err.append("violating twin leaves the tube on its legal acceleration")
    return err
