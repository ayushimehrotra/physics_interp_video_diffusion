#!/usr/bin/env python3
"""A cube sliding down an inclined plane. §32 — the set that separates mu from a.

    python3 generate_incline_scene.py --seed 0 --output-dir out/scene_00000

WHY THIS SET EXISTS
-------------------
Every friction result before this one is an ACCELERATION result wearing a
different label. On a level floor `a = mu*g`, and with g fixed at 1.0 the two are
the same number: measured on `slide_set_v2`, corr(mu, decel) = 0.99976 and
predicting mu from the deceleration with no fitting at all gives R2 = 0.995. There
is no scene in that set where a probe could tell them apart, so "friction is
decodable at 0.93/0.97" (§30) says only that the cube's deceleration is decodable
— which Joseph et al. §5.2 already report for acceleration.

An incline breaks the identity. Released from rest on a slope of angle theta:

    a = g * (sin(theta) - mu * cos(theta))        so    mu = (sin(theta) - a/g) / cos(theta)

theta is *visible* — it is the orientation of the floor in the image — and it is
sampled independently of mu (measured corr = -0.004). So mu is no longer a linear
functional of the speed trace: recovering it needs a visual quantity and a
dynamic one, combined nonlinearly. Measured over 80k draws, **corr(mu, a) =
-0.654** instead of 0.9998, so a linear readout of acceleration alone can reach at
most R2 ~ 0.43 on mu. Anything above that is the model doing something else.

This also makes §26/§30's format claim FALSIFIABLE here. That claim says linear
functionals of the represented factors are directly readable while products and
quotients are not. mu is now a quotient-and-difference of two visible quantities,
so the prediction is that the DIRECT probe on mu should now fall short of
composing decoded `a` with decoded `theta` — the opposite of what happened on the
flat floor, where direct ~ composed because mu was a linear functional.

Controls fall out for free: `a` is kinematic and should decode; `theta` is a large
visual feature and should decode; cube size cannot enter `a` at all.

DESIGN
------
* **Three unknowns in the initial condition: start, launch speed, and `a`.**
  This is the §32b fix and it is the reason the set was re-rendered. v1 pinned the
  start AND released from rest, so position at frame k was exactly
  `s0 + 0.5*a*t^2` with `s0` a constant every scene shared — which makes the cube's
  location in ANY single frame a complete readout of `a`, and with theta visible in
  the same frame, of mu. Measured on the v1 render: mu from the cube's segmentation
  centroid alone reached **R2 = 0.877 at frame 21**, 0.988 with the mask area, and
  0.997 from the world position — against the 0.459 the set exists to impose.
  Jittering the start adds one unknown (two frames now needed, i.e. a velocity);
  a random launch speed adds a second (three frames needed, i.e. an ACCELERATION).
  Measured on the new sampler over 20 subsamples of 150: worst single frame
  **0.069 mean / 0.175 max**, worst frame pair 0.48, frame triple 0.997. The minimum sufficient
  statistic is now (a, theta) and nothing simpler, which is exactly the claim the
  set is supposed to test.
* **theta in (28, 42) degrees.** A cube tips when tan(theta) > base/height = 1,
  i.e. beyond 45 degrees; 42 leaves margin. The low end keeps mu < tan(theta) - 0.08
  for every draw, so the cube always slides and nothing is rejected.
* **mu in (0.15, 0.40), log-uniform, independent of theta.**
* **g = 0.45 and camera at 18.** Both are frame-budget choices, not physics. On a
  slope the path moves vertically as well as horizontally and the vertical extent
  binds, which the flat sets never had to consider. The launch speed and the start
  jitter both consume runway (up to 0.60 * 6.125 = 3.7 units and 2.5 units), so g
  came down from 0.7 and the camera went back from 14 to pay for them. Cost: the
  smallest cube renders ~28 px across instead of ~36. Verified rejection-free over
  80k draws at exactly these values; every neighbouring setting that rejects even
  0.02% of draws was discarded, because a rejection that depends on travel depends
  on `a` and therefore on mu (§24/§25).
* **The start is anchored to the runway, then jittered.** `s0 = -runway(theta, h)
  + JITTER/2 + U(-JITTER/2, +JITTER/2)`, where `runway` is the largest |s| whose
  cube stays inside both frame axes. Anchoring to theta and h rather than to a
  constant buys the room the launch speed needs, and it is safe because theta and h
  are drawn independently of mu — unlike v1's first attempt, which anchored to
  `travel` and therefore to `a`. What must stay true is that no function of frame 0
  predicts mu, and that is measured, not assumed.
* **Banded floor, inherited from §25**, bands running across the slope. Identical
  in every scene, so they cannot carry mu; without them a translating cube and a
  panning camera produce identical pixels.
* **Rotation damped, not frozen.** §24's `localInertiaDiagonal = 1e6` sends the
  sim NaN here as it did in §31 (reported as a cube that never moves). Angular
  damping touches only the rotational DOF, leaving `a` exact.

THE SAMPLER REJECTS NOTHING. Re-verified over 80k draws at the post-fix constants:
mu < tan(theta) - 0.08, travel > 0.8 units, and the whole path inside both frame
axes with margin, in 100%. Acceptance never depends on mu, so mu cannot pick up a
correlation with theta or size through the back door (§24/§25). The launch speed
also retires the `travel < 0.8` edge outright: v0*6.125 alone clears it.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import tempfile
import traceback
from pathlib import Path

import numpy as np

logger = logging.getLogger("generate_incline_scene")

NUM_FRAMES = 49
FRAME_RATE = 8
STEP_RATE = 240
RESOLUTION = (720, 480)

ARMS = ("latent_friction", "visible_friction")

GRAVITY = 0.45                     # a frame-budget choice; see the docstring
THETA_RANGE_DEG = (28.0, 42.0)
MU_RANGE = (0.15, 0.40)
HALF_EXTENT_RANGE = (0.25, 0.40)
DENSITY = 1.0
MU_SLIDE_MARGIN = 0.08             # mu must stay this far below tan(theta)

# The two nuisance draws that break the single-frame readout. Neither touches the
# physics: Coulomb friction is speed-independent, so `a` is unchanged by v0, and a
# different start is a different place to stand. See derive() and §32b.
START_JITTER = 2.5                 # span of the uniform jitter on the start, world units
LAUNCH_SPEED_RANGE = (0.0, 0.60)   # down-slope speed at frame 0, world units / s

CAMERA_DISTANCE = 18.0
CAMERA_HEIGHT = 1.5                # MUST be above the slope plane, and only just.
                                   # At 0.0 the camera sits IN the plane: every point is
                                   # edge-on, the render is a black wedge, and the validator
                                   # passed it (floor luma 0.6 of 255) because it measured
                                   # the mu-driven SWING, never whether the floor was lit.
                                   # At 5.0 the surface is lit but fills 89% of the frame and
                                   # the slope/background boundary — the only cue for theta —
                                   # leaves the image. 1.5 keeps both: lit bands AND a
                                   # diagonal horizon. Checked by looking at the pixels.
                                   # sits IN the plane, so every point is edge-on and the
                                   # render is a black wedge with no visible bands (checked
                                   # by looking at the pixels, not the luma summary — the
                                   # validator passed it at floor luma 0.6 of 255).
LOOK_AT_Z = 0.0
FOCAL_LENGTH = 50.0
SENSOR_WIDTH = 36.0
HALF_WIDTH = 0.5 * CAMERA_DISTANCE * SENSOR_WIDTH / FOCAL_LENGTH
HALF_HEIGHT = 0.5 * CAMERA_DISTANCE * (SENSOR_WIDTH * RESOLUTION[1] / RESOLUTION[0]) / FOCAL_LENGTH
FRAME_MARGIN = 0.5

FLOOR_ALBEDO = 0.42
FLOOR_ALBEDO_RANGE = (0.42, 0.78)
FLOOR_SURFACE = {"metallic": 0.0, "roughness": 0.8, "specular": 0.2}
CUBE_ALBEDO = 0.20
CUBE_SURFACE = {"metallic": 0.0, "roughness": 0.6, "specular": 0.4, "ior": 1.45}
SLOPE_EXTENT = 40.0                # along-slope half-length; runs out of frame
FLOOR_THICKNESS = 1.0
FLOOR_BAND_WIDTH = 1.0
FLOOR_BAND_CONTRAST = 1.09

# No event happens in this set, so the split is a FIXED frame, identical in every
# scene. A per-scene boundary would be an information channel (§30's rule).
PHASE_SPLIT_FRAME = 20

SEG_ID = {"cube": 1, "floor": 2}

ANGULAR_DAMPING = 8.0
ACCEL_REL_TOLERANCE = 0.08
MAX_ANGULAR_SPEED = 0.60
MAX_ORIENTATION_DRIFT_DEG = 3.0    # measured against the SLOPE, not the world
MAX_OFF_SLOPE = 0.03
MAX_LATERAL_SPEED = 0.02
MAX_ATTEMPTS = 20
SAMPLES_PER_PIXEL = 32             # the slope fills most of the frame here, so 64 spp
                                   # costs ~850 s/scene against the flat set's ~190. Halved;
                                   # the pilot's appearance-leak numbers confirm it is clean
CLIP_SECONDS = (NUM_FRAMES - 1) / FRAME_RATE


def albedo_for_mu(mu: float) -> float:
    lo, hi = MU_RANGE
    frac = (np.log(mu) - np.log(lo)) / (np.log(hi) - np.log(lo))
    a_lo, a_hi = FLOOR_ALBEDO_RANGE
    return float(a_lo + frac * (a_hi - a_lo))


def sample_params(rng: np.random.Generator, arm: str = "latent_friction") -> dict:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    mu = float(np.exp(rng.uniform(*np.log(MU_RANGE))))
    theta = float(np.radians(rng.uniform(*THETA_RANGE_DEG)))
    return {
        "arm": arm,
        "mu": mu,
        "theta": theta,
        "theta_deg": float(np.degrees(theta)),
        "floor_albedo": FLOOR_ALBEDO if arm == "latent_friction" else albedo_for_mu(mu),
        "half_extent": float(rng.uniform(*HALF_EXTENT_RANGE)),
        "y_offset": float(rng.uniform(-0.4, 0.4)),
        # Drawn here, with mu and theta, so they are visibly independent of both.
        "start_jitter": float(rng.uniform(-0.5 * START_JITTER, 0.5 * START_JITTER)),
        "launch_speed": float(rng.uniform(*LAUNCH_SPEED_RANGE)),
    }


def runway(theta: float, half_extent: float) -> float:
    """Largest |s| whose cube still sits inside both frame axes, with margin.

    Depends on theta and the cube size only. Both are drawn independently of mu,
    so anchoring the start to this leaks nothing about the target — the thing v1
    got wrong was anchoring to `travel`, which is `a`, which is mu given theta.
    """
    h = half_extent
    return float(min(
        (HALF_WIDTH - h - FRAME_MARGIN - h * np.sin(theta)) / np.cos(theta),
        (HALF_HEIGHT - h - FRAME_MARGIN - h * np.cos(theta)) / np.sin(theta),
    ))


def derive(p: dict) -> dict:
    """Closed form for a block launched down an incline at v0."""
    th, mu, h = p["theta"], p["mu"], p["half_extent"]
    v0 = p["launch_speed"]
    a = GRAVITY * (np.sin(th) - mu * np.cos(th))
    travel = v0 * CLIP_SECONDS + 0.5 * a * CLIP_SECONDS ** 2
    # THE START POSITION IS FIXED, and this is not cosmetic.
    #
    # The first version centred each path on the origin (s0 = -travel/2), which
    # makes the start position a deterministic function of the travel distance and
    # therefore of the acceleration: measured corr(s0, a) = -0.9998, and `a`
    # recovered from the start position alone scored R2 = 0.9987. Since theta is
    # also visible in frame 0 (it is the slope's orientation), mu was recoverable
    # from THE FIRST FRAME ALONE at R2 = 1.0000 — no motion required, the entire
    # target given away by where the cube was placed.
    #
    # That is the identity trap arriving through the RENDERING rather than the
    # physics, and it voids the set: a probe could score perfectly without the
    # model representing any dynamics at all. Pinning s0 makes frame 0 carry
    # nothing about `a`; the information then exists only in the motion, which is
    # the whole point. Verified feasible over 80k draws: both the start and the
    # end stay inside both frame axes in 100% of draws.
    # THE FIX (§32b). Pinning s0 removed the cue from frame 0 and, by making the
    # start a constant every scene shared, handed it to every LATER frame instead:
    # s(k) - s0 = 0.5*a*t^2 exactly, so one snapshot gives `a` and, with the
    # visible theta, mu. Measured on the v1 render: R2 = 0.877 on mu from the cube
    # centroid at frame 21 alone, 0.988 adding the mask area. The check that was
    # supposed to catch this only ever looked at frame 0, where the set was clean.
    #
    # Two independent nuisance draws close it. The jitter makes one position
    # insufficient (two frames -> a velocity); the launch speed makes two
    # insufficient (three frames -> an acceleration, which is the quantity the set
    # is about). Anchor at the top of the runway so the longer paths still fit.
    s0 = -runway(th, h) + 0.5 * START_JITTER + p["start_jitter"]
    return {
        "accel": float(a),
        "travel": float(travel),
        "s_start": float(s0),
        "s_end": float(s0 + travel),
        "launch_speed": float(v0),
        "runway": runway(th, h),
        "mass": float(DENSITY * (2 * h) ** 3),
        "speed_final": float(v0 + a * CLIP_SECONDS),
        "downhill": [float(np.cos(th)), 0.0, float(-np.sin(th))],
        "normal": [float(np.sin(th)), 0.0, float(np.cos(th))],
    }


def surface_point(p: dict, g: dict, s: float) -> np.ndarray:
    """Cube-centre position at along-slope coordinate `s`."""
    u = np.array(g["downhill"]); n = np.array(g["normal"])
    return s * u + p["half_extent"] * n + np.array([0.0, p["y_offset"], 0.0])


def check_frame0(p: dict, g: dict) -> str:
    """Feasibility. Expected NEVER to fire — verified over 80k draws."""
    if p["mu"] >= np.tan(p["theta"]) - MU_SLIDE_MARGIN:
        return "would_not_slide"
    if g["travel"] < 0.8:
        return "too_little_travel"
    h = p["half_extent"]
    for s in (g["s_start"], g["s_end"]):
        pos = surface_point(p, g, s)
        if abs(pos[0]) + h + FRAME_MARGIN > HALF_WIDTH:
            return "out_of_frame_x"
        if abs(pos[2]) + h + FRAME_MARGIN > HALF_HEIGHT:
            return "out_of_frame_z"
    return ""


def build_and_simulate(p: dict, g: dict, scratch_dir: str, render: bool):
    import kubric as kb
    from kubric.simulator import PyBullet
    import pybullet as pb

    th = p["theta"]
    scene = kb.Scene(
        frame_start=0, frame_end=NUM_FRAMES - 1, frame_rate=FRAME_RATE,
        step_rate=STEP_RATE, resolution=RESOLUTION,
        gravity=(0.0, 0.0, -GRAVITY),
        background=kb.Color(0.16, 0.17, 0.20, 1.0),
        ambient_illumination=kb.Color(0.05, 0.05, 0.05, 1.0),
    )
    simulator = PyBullet(scene, scratch_dir)
    pb.setPhysicsEngineParameter(splitImpulsePenetrationThreshold=0.0,
                                 physicsClientId=simulator.physics_client)
    renderer = None
    if render:
        from kubric.renderer import Blender
        renderer = Blender(scene, scratch_dir, use_denoising=True,
                           adaptive_sampling=True, samples_per_pixel=SAMPLES_PER_PIXEL)
        renderer.blender_scene.view_settings.view_transform = "Standard"

    camera = kb.PerspectiveCamera(
        name="camera", focal_length=FOCAL_LENGTH, sensor_width=SENSOR_WIDTH,
        position=(0.0, -CAMERA_DISTANCE, CAMERA_HEIGHT),
        look_at=(0.0, 0.0, LOOK_AT_Z),
    )
    scene += camera
    scene.camera = camera
    scene += kb.DirectionalLight(name="sun", position=(-6.0, -10.0, 12.0),
                                 look_at=(0.0, 0.0, 0.0), intensity=3.2)
    scene += kb.DirectionalLight(name="fill", position=(8.0, -6.0, 6.0),
                                 look_at=(0.0, 0.0, 0.0), intensity=0.8)

    # Rotation about +y by theta takes local +z onto the slope normal
    # (sin th, 0, cos th) and local +x onto the downhill direction.
    quat = (float(np.cos(th / 2)), 0.0, float(np.sin(th / 2)), 0.0)
    u = np.array(g["downhill"]); n = np.array(g["normal"])

    floor = None
    n_bands = int(np.ceil(2 * SLOPE_EXTENT / FLOOR_BAND_WIDTH))
    for i in range(n_bands):
        s0 = -SLOPE_EXTENT + i * FLOOR_BAND_WIDTH
        s1 = min(s0 + FLOOR_BAND_WIDTH, SLOPE_EXTENT)
        albedo = p["floor_albedo"] * (FLOOR_BAND_CONTRAST if i % 2 else 1.0)
        centre = 0.5 * (s0 + s1) * u - 0.5 * FLOOR_THICKNESS * n
        band = kb.Cube(
            name=f"slope_{i:03d}",
            scale=(0.5 * (s1 - s0), SLOPE_EXTENT, 0.5 * FLOOR_THICKNESS),
            position=tuple(float(v) for v in centre), quaternion=quat,
            material=kb.PrincipledBSDFMaterial(
                color=kb.Color(albedo, albedo, albedo, 1.0), **FLOOR_SURFACE),
            static=True, friction=p["mu"], restitution=0.0,
            background=True, segmentation_id=SEG_ID["floor"],
        )
        scene += band
        if floor is None:
            floor = band

    h = p["half_extent"]
    cube = kb.Cube(
        name="cube", scale=(h, h, h),
        position=tuple(float(v) for v in surface_point(p, g, g["s_start"])),
        quaternion=quat,
        # Launched down-slope at v0, not released from rest (§32b). Kinetic friction
        # is speed-independent, so `a` is exactly what it was; only the trajectory's
        # offset changes, which is the whole point.
        velocity=tuple(float(v) for v in g["launch_speed"] * u),
        mass=g["mass"],
        friction=1.0, restitution=0.0,
        material=kb.PrincipledBSDFMaterial(
            color=kb.Color(CUBE_ALBEDO, CUBE_ALBEDO, CUBE_ALBEDO, 1.0), **CUBE_SURFACE),
        segmentation_id=SEG_ID["cube"],
    )
    scene += cube
    pb.changeDynamics(cube.linked_objects[simulator], -1,
                      linearDamping=0.0, angularDamping=ANGULAR_DAMPING,
                      rollingFriction=0.0, spinningFriction=0.0,
                      physicsClientId=simulator.physics_client)

    animation, collisions = simulator.run(frame_start=0, frame_end=NUM_FRAMES - 1)
    return scene, cube, floor, camera, simulator, renderer, animation


def check_simulation(p, g, animation, cube):
    """Everything is measured ALONG THE SLOPE, not along world x."""
    anim = animation[cube]
    pos = np.array(anim["position"])
    vel = np.array(anim["velocity"])
    ang = np.array(anim["angular_velocity"])
    u = np.array(g["downhill"]); n = np.array(g["normal"])

    s = pos @ u                                   # along-slope coordinate
    off = pos @ n - p["half_extent"]              # height above the surface
    v_s = vel @ u                                 # along-slope speed
    meas = {"max_off_slope": float(np.abs(off).max()),
            "max_ang": float(np.abs(ang).max()),
            "s": s.tolist(), "v_s": v_s.tolist()}

    if np.abs(off).max() > MAX_OFF_SLOPE:
        return "left_the_slope", meas
    if np.abs(vel[:, 1]).max() > MAX_LATERAL_SPEED:
        return "veered_sideways", meas
    if np.abs(ang).max() > MAX_ANGULAR_SPEED:
        return "tumbled", meas

    # Orientation drift measured against the slope: the cube is BUILT rotated, so
    # the world-frame quaternion is not near identity and §25's test would fire on
    # every scene.
    quat = np.array(anim["quaternion"])
    q0 = quat[0] / np.linalg.norm(quat[0])
    dots = np.abs(quat @ q0) / np.linalg.norm(quat, axis=1)
    drift = float(np.degrees(2 * np.arccos(np.clip(dots, 0, 1))).max())
    meas["drift_vs_slope"] = drift
    if drift > MAX_ORIENTATION_DRIFT_DEG:
        return "orientation_drift", meas

    t = np.arange(NUM_FRAMES) / FRAME_RATE
    moving = np.abs(v_s) > 1e-4
    if moving.sum() < 10:
        return "did_not_slide", meas
    slope_fit, intercept = np.polyfit(t[moving], v_s[moving], 1)
    accel = float(slope_fit)
    meas["accel_measured"] = accel
    # The fitted intercept is the launch speed. Checked because a mismatch would
    # mean the cube did not receive the velocity it was given, which is exactly the
    # failure mode that would silently restore the single-frame leak (§32b).
    meas["launch_speed_measured"] = float(intercept)
    v0 = g["launch_speed"]
    meas["launch_speed_error"] = float(abs(intercept - v0))
    if abs(intercept - v0) > 0.02 + 0.05 * v0:
        return "launch_speed_mismatch", meas
    rel = abs(accel - g["accel"]) / abs(g["accel"])
    meas["accel_rel_error"] = float(rel)
    if rel > ACCEL_REL_TOLERANCE:
        return "accel_mismatch", meas
    meas["travel_measured"] = float(s[-1] - s[0])
    return "", meas


def build_labels(p, g, meas, seed, scene_id, render_device):
    th = p["theta"]
    v_s = np.array(meas["v_s"])
    a_meas = meas["accel_measured"]
    return {
        "scene_id": scene_id,
        "seed": int(seed),
        "scenario": "incline",
        "arm": p["arm"],
        "momentum_conserved": True,
        "friction": {
            "gravity": GRAVITY,
            "mu": p["mu"],
            "log_mu": float(np.log(p["mu"])),
            "theta": float(th),
            "theta_deg": float(np.degrees(th)),
            "sin_theta": float(np.sin(th)),
            "cos_theta": float(np.cos(th)),
            "tan_theta": float(np.tan(th)),
            "accel": g["accel"],
            "accel_measured": a_meas,
            "accel_rel_error": meas["accel_rel_error"],
            # mu recovered from what a camera can see, using the SAME identity a
            # probe would have to implement. Equals p["mu"] up to solver error;
            # stored so the oracle can be built without re-deriving it.
            "mu_from_kinematics": float((np.sin(th) - a_meas / GRAVITY) / np.cos(th)),
            "travel": g["travel"],
            "travel_measured": meas["travel_measured"],
            "speed_final": float(v_s[-1]),
            "floor_albedo": p["floor_albedo"],
            # §32b's two nuisance draws, stored so the validator can confirm they
            # are independent of mu and so the leak checks can be re-run offline.
            "launch_speed": g["launch_speed"],
            "launch_speed_measured": meas["launch_speed_measured"],
            "start_s": g["s_start"],
            "start_jitter": p["start_jitter"],
            "runway": g["runway"],
        },
        # Nominal, fixed, identical in every scene — see PHASE_SPLIT_FRAME.
        "collision": {
            "actual_frame": float(PHASE_SPLIT_FRAME),
            "actual_frame_rounded": PHASE_SPLIT_FRAME,
            "nominal": True,
        },
        "objects": {
            "A": {"kind": "cube", "half_extent": p["half_extent"],
                  "size": float(2 * p["half_extent"]), "mass": g["mass"],
                  "density": DENSITY, "radius": p["half_extent"],
                  "speed_initial": g["launch_speed"],
                  "segmentation_id": SEG_ID["cube"]},
            "B": {"kind": "slope", "mu": p["mu"], "theta": float(th),
                  "albedo": p["floor_albedo"], "segmentation_id": SEG_ID["floor"]},
        },
        "per_frame": [{"frame": int(i), "s": float(meas["s"][i]),
                       "v_s": float(v_s[i])} for i in range(NUM_FRAMES)],
        "scene_constants": {
            "num_frames": NUM_FRAMES, "frame_rate": FRAME_RATE,
            "resolution": list(RESOLUTION), "gravity": GRAVITY,
            "camera_distance": CAMERA_DISTANCE, "camera_height": CAMERA_HEIGHT,
            "half_width": HALF_WIDTH, "half_height": HALF_HEIGHT,
            "theta_range_deg": list(THETA_RANGE_DEG), "mu_range": list(MU_RANGE),
            "half_extent_range": list(HALF_EXTENT_RANGE),
            "start_jitter": START_JITTER,
            "launch_speed_range": list(LAUNCH_SPEED_RANGE),
            "phase_split_frame": PHASE_SPLIT_FRAME,
            "seg_id": SEG_ID, "render_device": render_device,
            "floor_band_width": FLOOR_BAND_WIDTH,
            "floor_band_contrast": FLOOR_BAND_CONTRAST,
        },
        "sampling_attempts": {"rejections": []},
    }


def configure_gpu(renderer) -> str:
    if not os.environ.get("KUBRIC_USE_GPU"):
        return "cpu (KUBRIC_USE_GPU not set)"
    try:
        import bpy
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "OPTIX"
        prefs.get_devices()
        names = []
        for d in prefs.devices:
            d.use = d.type != "CPU"
            if d.use:
                names.append(d.name)
        bpy.context.scene.cycles.device = "GPU"
        return f"OPTIX: {', '.join(names)}" if names else "cpu (no OPTIX devices)"
    except Exception as exc:                                       # noqa: BLE001
        return f"cpu (OPTIX setup failed: {exc})"


def _trial(p, g, scratch_dir, render):
    built = build_and_simulate(p, g, scratch_dir, render)
    scene, cube, floor, camera, simulator, renderer, animation = built
    reason, meas = check_simulation(p, g, animation, cube)
    return reason, meas, built


def generate(seed: int, out_dir: Path, scene_id: str, render: bool = True,
             arm: str = "latent_friction") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rejections = []
    scratch = tempfile.mkdtemp(prefix="incline_")
    try:
        for attempt in range(MAX_ATTEMPTS):
            rng = np.random.default_rng([seed, attempt])
            p = sample_params(rng, arm=arm)
            g = derive(p)
            why = check_frame0(p, g)
            if why:
                rejections.append({"attempt": attempt, "reason": why})
                continue
            reason, meas, built = _trial(p, g, scratch, render)
            if reason:
                rejections.append({"attempt": attempt, "reason": reason})
                del built
                gc.collect()
                continue
            scene, cube, floor, camera, simulator, renderer, animation = built
            device = configure_gpu(renderer) if render else "no render"
            labels = build_labels(p, g, meas, seed, scene_id, device)
            labels["sampling_attempts"]["rejections"] = rejections
            if render:
                write_outputs(out_dir, renderer.render())
            (out_dir / "labels.json").write_text(json.dumps(labels, indent=2))
            (out_dir / "_COMPLETE").write_text("")
            return labels
        raise RuntimeError(f"{scene_id}: {MAX_ATTEMPTS} attempts all rejected: "
                           f"{[r['reason'] for r in rejections]}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        gc.collect()


def write_outputs(out_dir: Path, frames):
    import imageio.v2 as imageio
    rgb = frames["rgba"][..., :3]
    seg = frames["segmentation"][..., 0].astype(np.uint8)
    (out_dir / "frames").mkdir(exist_ok=True)
    (out_dir / "segmentation").mkdir(exist_ok=True)
    for i in range(rgb.shape[0]):
        imageio.imwrite(out_dir / "frames" / f"frame_{i:04d}.png", rgb[i])
        # MUST be segmentation_%04d.png — video.load_segmentation globs exactly that.
        imageio.imwrite(out_dir / "segmentation" / f"segmentation_{i:04d}.png", seg[i])
    write_mp4(out_dir / "video.mp4", rgb)


def write_mp4(path: Path, rgb):
    try:
        import imageio.v2 as imageio
        with imageio.get_writer(path, fps=FRAME_RATE, codec="libx264",
                                output_params=["-movflags", "+faststart"]) as w:
            for f in rgb:
                w.append_data(f)
    except Exception as exc:                                       # noqa: BLE001
        logger.warning("mp4 writing failed (%s); frames are still on disk", exc)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--scene-id", default=None)
    ap.add_argument("--arm", default="latent_friction", choices=ARMS)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    scene_id = args.scene_id or args.output_dir.name
    try:
        lab = generate(args.seed, args.output_dir, scene_id,
                       render=not args.no_render, arm=args.arm)
    except Exception:                                              # noqa: BLE001
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "_FAILED").write_text(traceback.format_exc())
        logger.error("scene %s FAILED\n%s", scene_id, traceback.format_exc())
        return 1
    f = lab["friction"]
    logger.info("%s: theta=%.1f deg mu=%.3f -> a=%.4f (measured %.4f, %.2f%%), "
                "mu_from_kinematics=%.3f, travel=%.2f", scene_id, f["theta_deg"],
                f["mu"], f["accel"], f["accel_measured"],
                100 * f["accel_rel_error"], f["mu_from_kinematics"], f["travel_measured"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
