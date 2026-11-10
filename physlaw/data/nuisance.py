"""Nuisance variants (spec §2.3): camera pose, lighting, floor and object textures randomised,
physics identical. Pure numpy; ``generate.py`` applies the look to the Kubric scene.

A nuisance variant re-renders every twin of a group with one draw from :func:`sample`,
into ``<group>_nuisance/``. Its labels carry ``physlaw.nuisance = 1``,
``twin_group_id = <group>_nuisance``, ``base_group_id = <group>`` and the base group's
split, so a model never trains on one and tests on the other. Bodies that look
identical in the base scene (F3, F7 drop pairs) stay identical to each other.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from physlaw.data.families import Camera


def selected(seed: int, frac: float) -> bool:
    """Deterministic per seed, independent of the physics stream."""
    u = int(hashlib.sha256(f"nuisance:{seed}".encode()).hexdigest()[:12], 16) / float(16 ** 12)
    return u < frac


def sample(rng: np.random.Generator, body_names: list, identical: bool) -> dict:
    hue = lambda: _hsv(rng.uniform(0, 1), rng.uniform(0.35, 0.9), rng.uniform(0.35, 0.85))
    shared = hue()
    return {
        "camera": {"yaw_deg": float(rng.uniform(-15, 15)), "pitch_deg": float(rng.uniform(-8, 8)),
                   "roll_deg": float(rng.uniform(-6, 6)), "dist_scale": float(rng.uniform(0.97, 1.12))},
        "sun": {"azimuth_deg": float(rng.uniform(0, 360)), "elevation_deg": float(rng.uniform(35, 75)),
                "intensity": float(rng.uniform(2.0, 4.5))},
        "ambient": float(rng.uniform(0.02, 0.10)),
        "floor": {"color": _hsv(rng.uniform(0, 1), rng.uniform(0.0, 0.35), rng.uniform(0.25, 0.6)),
                  "checker": bool(rng.uniform() < 0.6), "checker_scale": float(rng.uniform(1.0, 6.0)),
                  "checker_contrast": float(rng.uniform(0.1, 0.35))},
        "bodies": {n: (shared if identical else hue()) for n in body_names},
    }


def _hsv(h, s, v):
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    return [float(x) for x in [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]]


def perturb_camera(cam: Camera, look: dict) -> Camera:
    """Orbit the camera about its look-at point (yaw about world z, pitch about its right axis),
    scale the distance and roll about the view axis."""
    c = look["camera"]
    C, T = np.asarray(cam.position, float), np.asarray(cam.look_at, float)
    off = (C - T) * c["dist_scale"]
    off = _rot(np.array([0.0, 0.0, 1.0]), math.radians(c["yaw_deg"])) @ off
    R = cam.rotation()
    off = _rot(R[0], math.radians(c["pitch_deg"])) @ off
    up = np.asarray(cam.up, float)
    if abs(np.dot(R[2], up / np.linalg.norm(up))) > 0.999:
        up = -R[1]                                   # top-down camera: image-up
    up = _rot(R[2], math.radians(c["roll_deg"])) @ up
    return Camera(tuple(T + off), tuple(T), tuple(up), cam.focal_mm, cam.sensor_w_mm, cam.resolution, cam.aspect)


def sun_position(look: dict, dist: float = 15.0) -> tuple:
    s = look["sun"]
    az, el = math.radians(s["azimuth_deg"]), math.radians(s["elevation_deg"])
    return (dist * math.cos(el) * math.cos(az), dist * math.cos(el) * math.sin(az), dist * math.sin(el))


def _rot(axis, ang):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K
