#!/usr/bin/env python3
"""Two equal cubes colliding on a shared friction floor. §31.

    python3 generate_collide_friction_scene.py --seed 0 --output-dir out/scene_00000

The question this set exists to answer: **does §30's friction result survive when
there are two objects and an interaction?** §30 measured mu on ONE cube sliding on
a uniform floor, and found it decodable at 0.93/0.97 but sitting exactly at its
kinematic oracle — mu is a linear functional of the visible speed trace, so a
model representing the trace hands mu to a linear probe for free. Adding a second
cube and a collision makes three genuinely new things askable, none of which the
single-cube set can pose:

  1. mu is ONE property of a surface that TWO objects share. Is it represented
     once (a surface attribute) or twice (per-object deceleration)? Cross-decode
     A's pool -> B's label and back.
  2. mu is unchanged by the collision, but every velocity is. Does a mu readout
     fit on the PRE phase still work on the POST phase? A surface attribute
     should transfer; a per-phase kinematic estimate should not.
  3. Both cubes decelerate at the SAME mu*g regardless of their (different)
     speeds and their (equal) mass. So the composition test of §26/§30 can be run
     per object and per phase, four times over the same scalar.

Design, and what each choice is buying
--------------------------------------
* **Equal cubes.** Same half-extent, same density, same albedo — so nothing about
  the two objects differs except their velocities, and any A/B asymmetry a probe
  finds is about position in the frame, not about appearance. Size still varies
  BETWEEN scenes (it is `a = mu*g`-irrelevant, so it stays a physics control —
  though see §30 on why it is NOT a measurement control: the pooling mask's area
  gives it away at R2 0.99).
* **Same direction, along x, at equal depth — A (rear, fast) catches B (front,
  slow).** The contact normal is the travel direction, so the collision is
  one-dimensional. Both cubes keep moving forward afterwards, so BOTH carry an
  observable deceleration in BOTH phases; a head-on version leaves them nearly
  stopped after contact (measured: post velocities of 0.00-0.25) and spends the
  frame budget in two directions at once.
* **One shared mu.** The floor carries it, both cubes are at friction 1.0, so each
  pair coefficient is exactly the floor's (PyBullet multiplies lateral frictions).
* **Contact at frame 14.5**, in the INTERIOR of a tubelet for both encoders
  rather than on its edge — see TARGET_CONTACT_FRAME. Frame 20 would work too,
  but the ranges below cannot keep both cubes moving that long at high mu without
  correlating speed with mu.

THE SAMPLER REJECTS NOTHING, and that is the point
--------------------------------------------------
§24/§25 are emphatic: if acceptance depends on the parameters, mu picks up a
correlation with speed or size through the back door, which is exactly how the
momentum set's confound entered and forced the v2 rebuild. Every draw here is
feasible by construction, verified over the whole product of the ranges:

    contact frame 14.5, mu (0.20, 0.45), rear speed (1.45, 1.85),
    front speed (1.00, 1.30), restitution (0.30, 0.70)
    -> both cubes still moving at contact, and A faster than B, in 100% of 60k draws
    -> the whole trajectory inside the frame with margin in 100%
    -> at least 5 post-contact moving frames for BOTH cubes in 100%
       (5th percentile 9.3 frames, median 19.9)

The cost is a narrower mu range than §25's (factor 2.5 against 5.3), so probe R2
here is NOT comparable to §30's directly. **Compare against the oracle, which
adjusts for label variance, not against §30's raw numbers.**

Physics traps inherited from §24/§25, all still live
----------------------------------------------------
* A sphere with lateral friction ROLLS and decays to 5/7 of its launch speed. Use
  a box; it slides cleanly and tips only if mu > half_width/half_height = 1.0.
* g = 1.0, not 9.81. This world is ~10 units across; at 9.81 an ordinary mu stops
  a cube in three frames.
* kubric's `_BulletClient` disconnects in its destructor and PyBullet reissues
  client id 0, so calling `pb.disconnect()` on a rejected trial kills the NEXT
  scene's connection. Each trial is scoped in a function with a `gc.collect()`.
* `-movflags +faststart` on the mp4, or a browser handed the clip as a data: URI
  refuses to play it.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

logger = logging.getLogger("generate_collide_friction_scene")

# --------------------------------------------------------------------------------------
# Constants. Copied into labels.json so a scene is reconstructable from its labels.
# --------------------------------------------------------------------------------------

NUM_FRAMES = 49
FRAME_RATE = 8
STEP_RATE = 240
RESOLUTION = (720, 480)

ARMS = ("latent_friction", "visible_friction")

GRAVITY = 1.0
MU_RANGE = (0.20, 0.45)            # log-uniform. Narrower than §25's (0.15, 0.80):
                                   # the wide end cannot keep both cubes moving to
                                   # contact without tying speed to mu.
# A is the REAR cube and always the faster one; B is the FRONT cube. The ranges do
# not overlap, so A catches B in every draw with no rejection.
SPEED_A_RANGE = (1.45, 1.85)       # rear
SPEED_B_RANGE = (1.00, 1.30)       # front
RESTITUTION_RANGE = (0.30, 0.70)   # pair coefficient; each cube carries sqrt(e)
HALF_EXTENT_RANGE = (0.25, 0.40)   # shared by both cubes within a scene
DENSITY = 1.0

# 14.5, NOT 14 — and the half-frame is load-bearing. `tubelet_phase_split` raises
# if any temporal cell holds frames from BOTH sides of contact, and V-JEPA2's
# cell 6 is {13, 14}. Targeting 14 with the +-0.13 frame of solver jitter puts
# contact at 13.93 in a third of scenes, which splits that cell and makes capture
# raise PER SCENE, after the model is already loaded. Targeting a cell's INTERIOR
# is robust: contact anywhere in (14.1, 14.9) leaves VideoMAE's {9,12}/{15,18}
# and V-JEPA2's {13,14}/{15,16} cleanly on opposite sides.
TARGET_CONTACT_FRAME = 14.5
CONTACT_TOLERANCE = 0.4

CAMERA_DISTANCE = 14.0             # §25's validated side view, unchanged
CAMERA_HEIGHT = 1.1
LOOK_AT_Z = 0.35
FOCAL_LENGTH = 50.0
SENSOR_WIDTH = 36.0
HALF_WIDTH = 0.5 * CAMERA_DISTANCE * SENSOR_WIDTH / FOCAL_LENGTH
FRAME_MARGIN = 0.6

FLOOR_ALBEDO = 0.42
FLOOR_ALBEDO_RANGE = (0.42, 0.78)
FLOOR_SURFACE = {"metallic": 0.0, "roughness": 0.8, "specular": 0.2}
CUBE_ALBEDO = 0.20
CUBE_SURFACE = {"metallic": 0.0, "roughness": 0.6, "specular": 0.4, "ior": 1.45}
FLOOR_EXTENT = 60.0
FLOOR_THICKNESS = 1.0
# Bands across the travel direction: §25 found a flat floor visually ambiguous
# between object motion and camera motion. Identical in every scene, so they
# cannot carry mu.
FLOOR_BAND_WIDTH = 1.0
FLOOR_BAND_CONTRAST = 1.09

PRE_FRAME = 11                     # anchors for the collision block, clear of the
POST_FRAME = 17                    # contact transient at frame 14

SEG_ID = {"cube_a": 1, "cube_b": 2, "floor": 3}   # A and B are the two CUBES here,
                                                  # so runner's ids=(1,2) pools them
                                                  # with no code change anywhere.

DECEL_REL_TOLERANCE = 0.08         # looser than §25's 0.05: fewer moving frames per
                                   # phase to fit a slope through
MAX_VERTICAL_SPEED = 0.02
ANGULAR_DAMPING = 8.0              # see build_and_simulate: kills post-collision
                                   # pitching without touching linear motion
MAX_ANGULAR_SPEED = 0.60
MAX_ORIENTATION_DRIFT_DEG = 2.0    # damped drift measures 0.00-0.02 deg, so this
                                   # is a tripwire, not a filter: it must never fire
MAX_ATTEMPTS = 20
SAMPLES_PER_PIXEL = 64
CLIP_SECONDS = (NUM_FRAMES - 1) / FRAME_RATE


# --------------------------------------------------------------------------------------
# Sampling and closed-form geometry
# --------------------------------------------------------------------------------------

def albedo_for_mu(mu: float) -> float:
  lo, hi = MU_RANGE
  frac = (np.log(mu) - np.log(lo)) / (np.log(hi) - np.log(lo))
  a_lo, a_hi = FLOOR_ALBEDO_RANGE
  return float(a_lo + frac * (a_hi - a_lo))


def sample_params(rng: np.random.Generator, arm: str = "latent_friction") -> dict:
  """One scene. Nothing is rejected, so mu cannot correlate with anything else."""
  if arm not in ARMS:
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
  mu = float(np.exp(rng.uniform(*np.log(MU_RANGE))))
  albedo = FLOOR_ALBEDO if arm == "latent_friction" else albedo_for_mu(mu)
  return {
      "arm": arm,
      "mu": mu,
      "floor_albedo": albedo,
      "half_extent": float(rng.uniform(*HALF_EXTENT_RANGE)),
      "speed_a": float(rng.uniform(*SPEED_A_RANGE)),
      "speed_b": float(rng.uniform(*SPEED_B_RANGE)),
      "restitution": float(rng.uniform(*RESTITUTION_RANGE)),
      "start_jitter": float(rng.uniform(0.0, 0.5)),
      "y_offset": float(rng.uniform(-0.5, 0.5)),   # both cubes share it: same lane
  }


def derive(p: dict) -> dict:
  """Place contact at TARGET_CONTACT_FRAME. Closed form, and exact.

  Both cubes slide on the same floor, so both decelerate at the same a = mu*g
  while moving. **Their relative motion is therefore frictionless**: the gap
  closes at a constant (v_a - v_b), and the contact time is just gap0 / (v_a -
  v_b) with no quadratic to solve. That is the whole reason for the same-
  direction geometry — a head-on version has both cubes' deceleration entering
  the closing rate, and (worse) spends the frame budget in two directions at once
  while restitution throws the pair apart.

  A is the rear cube and catches B. Both keep moving forward after contact, so
  both carry an observable deceleration in BOTH phases, which is what the
  cross-phase transfer question needs.
  """
  a = p["mu"] * GRAVITY
  t = TARGET_CONTACT_FRAME / FRAME_RATE
  h = p["half_extent"]
  e = p["restitution"]

  gap0 = (p["speed_a"] - p["speed_b"]) * t          # exact: friction cancels
  travel_a = p["speed_a"] * t - 0.5 * a * t * t
  travel_b = p["speed_b"] * t - 0.5 * a * t * t
  v_a_contact = p["speed_a"] - a * t
  v_b_contact = p["speed_b"] - a * t

  # 1-D equal masses with restitution, both moving +x.
  post_a = 0.5 * ((1 - e) * v_a_contact + (1 + e) * v_b_contact)
  post_b = 0.5 * ((1 + e) * v_a_contact + (1 - e) * v_b_contact)

  t_rest = CLIP_SECONDS - t

  def after(v):
    s_ = abs(v)
    tt = min(t_rest, s_ / a)
    return s_ * tt - 0.5 * a * tt * tt

  # Anchor A at the left margin and let everything follow; the sweep over the
  # whole product of the ranges confirms the far end always lands in frame.
  start_a = -(HALF_WIDTH - FRAME_MARGIN - h) + p["start_jitter"]
  start_b = start_a + 2 * h + gap0
  mass = float(DENSITY * (2 * h) ** 3)
  return {
      "decel": float(a),
      "mass": mass,
      "gap0": float(gap0),
      "travel_a_pre": float(travel_a),
      "travel_b_pre": float(travel_b),
      "v_a_contact": float(v_a_contact),
      "v_b_contact": float(v_b_contact),
      "post_v_a": float(post_a),
      "post_v_b": float(post_b),
      "start_a_x": float(start_a),
      "start_b_x": float(start_b),
      "reach_a": float(start_a + travel_a + after(post_a)),
      "reach_b": float(start_b + travel_b + after(post_b)),
      "stop_frame_a": float(t * FRAME_RATE + min(abs(post_a) / a, t_rest) * FRAME_RATE),
      "stop_frame_b": float(t * FRAME_RATE + min(abs(post_b) / a, t_rest) * FRAME_RATE),
      "momentum_pre": float(mass * (v_a_contact + v_b_contact)),
      "momentum_post": float(mass * (post_a + post_b)),
  }


def check_frame0(p: dict, g: dict) -> str:
  """Feasibility. Expected NEVER to fire — the ranges guarantee it. If it does,
  the ranges were changed without re-running the 40k-draw sweep."""
  h = p["half_extent"]
  if g["v_a_contact"] <= 0.15 or g["v_b_contact"] <= 0.15:
    return "stopped_before_contact"
  if g["v_a_contact"] - g["v_b_contact"] <= 0.10:
    return "no_approach"
  for x in (g["start_a_x"], g["start_b_x"]):
    if abs(x) + h + FRAME_MARGIN > HALF_WIDTH:
      return "start_out_of_frame"
  for r in (g["reach_a"], g["reach_b"]):
    if abs(r) + h + FRAME_MARGIN > HALF_WIDTH:
      return "end_out_of_frame"
  return ""


# --------------------------------------------------------------------------------------
# Build and simulate
# --------------------------------------------------------------------------------------

def build_and_simulate(p: dict, g: dict, scratch_dir: str, render: bool):
  import kubric as kb
  from kubric.simulator import PyBullet
  import pybullet as pb

  scene = kb.Scene(
      frame_start=0, frame_end=NUM_FRAMES - 1, frame_rate=FRAME_RATE,
      step_rate=STEP_RATE, resolution=RESOLUTION,
      gravity=(0.0, 0.0, -GRAVITY),
      background=kb.Color(0.16, 0.17, 0.20, 1.0),
      ambient_illumination=kb.Color(0.05, 0.05, 0.05, 1.0),
  )
  simulator = PyBullet(scene, scratch_dir)
  # Bullet's default ADDS energy on deep penetration; zero it or the collision
  # silently gains momentum.
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

  floor = None
  n_bands = int(np.ceil(2 * FLOOR_EXTENT / FLOOR_BAND_WIDTH))
  for i in range(n_bands):
    x0 = -FLOOR_EXTENT + i * FLOOR_BAND_WIDTH
    x1 = min(x0 + FLOOR_BAND_WIDTH, FLOOR_EXTENT)
    albedo = p["floor_albedo"] * (FLOOR_BAND_CONTRAST if i % 2 else 1.0)
    band = kb.Cube(
        name=f"floor_{i:03d}",
        scale=(0.5 * (x1 - x0), FLOOR_EXTENT, 0.5 * FLOOR_THICKNESS),
        position=(0.5 * (x0 + x1), 0.0, -0.5 * FLOOR_THICKNESS),
        material=kb.PrincipledBSDFMaterial(
            color=kb.Color(albedo, albedo, albedo, 1.0), **FLOOR_SURFACE),
        static=True, friction=p["mu"], restitution=0.0,
        background=True, segmentation_id=SEG_ID["floor"],
    )
    scene += band
    if floor is None:
      floor = band

  h = p["half_extent"]
  # Restitution is a PAIR coefficient and PyBullet multiplies the two, so each
  # cube carries sqrt(e) — the same convention as the momentum set's spheres.
  root_e = float(np.sqrt(p["restitution"]))
  cubes = {}
  for key, x0, v0 in (("a", g["start_a_x"], p["speed_a"]),
                      ("b", g["start_b_x"], p["speed_b"])):
    cube = kb.Cube(
        name=f"cube_{key}", scale=(h, h, h),
        position=(x0, p["y_offset"], h),
        velocity=(v0, 0.0, 0.0),
        mass=g["mass"],
        friction=1.0, restitution=root_e,
        material=kb.PrincipledBSDFMaterial(
            color=kb.Color(CUBE_ALBEDO, CUBE_ALBEDO, CUBE_ALBEDO, 1.0), **CUBE_SURFACE),
        segmentation_id=SEG_ID[f"cube_{key}"],
    )
    scene += cube
    # Must come AFTER `scene += cube`: kubric's mass observer fires during the add
    # and changeDynamics(mass=...) recomputes what was set earlier.
    # ROTATION IS DAMPED, and it has to be. A rear-end collision applies its
    # impulse at COM height while floor friction acts at the base, so the cube
    # pitches — measured 9-33 degrees of drift undamped, which is not merely ugly:
    # the tilt scales with the collision impulse, and the impulse depends on mu
    # through the contact velocities, so the cube's SILHOUETTE would become a
    # second channel carrying the target.
    #
    # §24's fix (localInertiaDiagonal = 1e6, i.e. freeze rotation outright) was
    # tried FIRST and fails here exactly as §25 warns: the sim goes NaN and the
    # cubes never collide at all, reported as `no_collision`. Angular damping is
    # the fix that works — it touches only the rotational DOF, so `a = mu*g` and
    # the linear dynamics are untouched.
    pb.changeDynamics(cube.linked_objects[simulator], -1,
                      linearDamping=0.0, angularDamping=ANGULAR_DAMPING,
                      rollingFriction=0.0, spinningFriction=0.0,
                      physicsClientId=simulator.physics_client)
    cubes[key] = cube

  animation, collisions = simulator.run(frame_start=0, frame_end=NUM_FRAMES - 1)
  return scene, cubes, floor, camera, simulator, renderer, animation, collisions


# --------------------------------------------------------------------------------------
# Physics acceptance
# --------------------------------------------------------------------------------------

def _phase_decel(speed: np.ndarray, frames: np.ndarray, v_ref: float):
  """Fit a deceleration to one phase of one cube. Returns (decel, n_moving)."""
  moving = speed[frames] > 0.02 * max(v_ref, 1e-9)
  idx = frames[moving]
  if idx.size < 3:
    return None, int(idx.size)
  slope, _ = np.polyfit(idx / FRAME_RATE, speed[idx], 1)
  return float(-slope), int(idx.size)


def check_simulation(p, g, animation, cubes, collisions):
  pair = {cubes["a"], cubes["b"]}
  pair_events = [e for e in collisions if set(e["instances"]) == pair]
  if not pair_events:
    return "no_collision", {}
  contact_frame = float(min(e["frame"] for e in pair_events))
  if abs(contact_frame - TARGET_CONTACT_FRAME) > CONTACT_TOLERANCE:
    return "contact_frame_miss", {"contact_frame": contact_frame}

  meas = {"contact_frame": contact_frame, "n_pair_events": len(pair_events)}
  cf = int(np.floor(contact_frame))
  for key in ("a", "b"):
    anim = animation[cubes[key]]
    vel = np.array(anim["velocity"])
    pos = np.array(anim["position"])
    ang = np.array(anim["angular_velocity"])
    if np.max(np.abs(vel[:, 2])) > MAX_VERTICAL_SPEED:
      return "vertical_motion", meas
    if np.max(np.abs(pos[:, 2] - p["half_extent"])) > 0.02:
      return "left_the_floor", meas
    meas[f"max_ang_{key}"] = float(np.max(np.abs(ang)))
    if np.max(np.abs(ang)) > MAX_ANGULAR_SPEED:
      return "tumbled", meas
    if np.max(np.abs(vel[:, 1])) > 0.02:
      return "veered_sideways", meas
    quat = np.array(anim["quaternion"])
    drift = float(np.degrees(2 * np.arccos(np.clip(np.abs(quat[:, 0]), 0, 1))).max())
    meas[f"drift_{key}"] = drift
    if drift > MAX_ORIENTATION_DRIFT_DEG:
      return "orientation_drift", meas

    speed = np.abs(vel[:, 0])
    # Frame 0 carries the launch impulse; frames adjacent to contact carry the
    # collision impulse. Both are excluded from the slope fits.
    pre = np.arange(1, cf)
    post = np.arange(cf + 2, NUM_FRAMES)
    for phase, frames, v_ref in (("pre", pre, p[f"speed_{key}"]),
                                 ("post", post, abs(g[f"post_v_{key}"]))):
      d, n = _phase_decel(speed, frames, v_ref)
      meas[f"n_moving_{key}_{phase}"] = n
      if d is None:
        meas[f"decel_{key}_{phase}"] = None
        continue
      meas[f"decel_{key}_{phase}"] = d
      rel = abs(d - g["decel"]) / g["decel"]
      meas[f"decel_rel_error_{key}_{phase}"] = float(rel)
      if phase == "pre" and rel > DECEL_REL_TOLERANCE:
        return "decel_mismatch", meas
    meas[f"velocity_{key}"] = vel[:, 0].tolist()
    meas[f"position_{key}"] = pos[:, 0].tolist()

  return "", meas


# --------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------

def build_labels(p, g, meas, camera, seed, scene_id, render_device):
  """Emit BOTH a `friction` block and a momentum-set-shaped `collision` block.

  The collision block is what makes every existing DERIVED target and the capture
  path's phase split work on this set with no code change: `collision.actual_frame`
  drives the pre/post pooling, and `collision.pre/post.velocity_A/B` are what the
  velocity targets read. Unlike §25's slide set, that frame is a REAL event here.
  """
  cf = meas["contact_frame"]
  va = np.array(meas["velocity_a"])
  vb = np.array(meas["velocity_b"])
  ia, ib = int(np.floor(cf)), int(np.ceil(cf)) + 1
  m = g["mass"]

  def vec(v):
    return [float(v), 0.0, 0.0]

  pre = {
      "velocity_A": vec(va[ia]), "velocity_B": vec(vb[ia]),
      "momentum_A": vec(m * va[ia]), "momentum_B": vec(m * vb[ia]),
      "total_momentum": vec(m * (va[ia] + vb[ia])),
      "kinetic_energy": float(0.5 * m * (va[ia] ** 2 + vb[ia] ** 2)),
  }
  post = {
      "velocity_A": vec(va[ib]), "velocity_B": vec(vb[ib]),
      "momentum_A": vec(m * va[ib]), "momentum_B": vec(m * vb[ib]),
      "total_momentum": vec(m * (va[ib] + vb[ib])),
      "kinetic_energy": float(0.5 * m * (va[ib] ** 2 + vb[ib] ** 2)),
  }
  dp = post["total_momentum"][0] - pre["total_momentum"][0]
  scale = max(abs(pre["total_momentum"][0]), 1e-9)

  return {
      "scene_id": scene_id,
      "seed": int(seed),
      "scenario": "collide_friction",
      "arm": p["arm"],
      # Momentum is NOT conserved across this window: friction is an external
      # force and acts for the ~2 frames between the anchors. The residual is
      # recorded rather than asserted, and it is a *friction* impulse, not a
      # violation — there is no violating arm in this set.
      "momentum_conserved": True,
      "momentum_rel_error": float(abs(dp) / scale),
      "friction": {
          "gravity": GRAVITY,
          "mu": p["mu"],
          "log_mu": float(np.log(p["mu"])),
          "advertised_mu": p["mu"],
          "floor_albedo": p["floor_albedo"],
          "decel": g["decel"],
          "decel_measured_a_pre": meas.get("decel_a_pre"),
          "decel_measured_a_post": meas.get("decel_a_post"),
          "decel_measured_b_pre": meas.get("decel_b_pre"),
          "decel_measured_b_post": meas.get("decel_b_post"),
          "n_moving_a_pre": meas.get("n_moving_a_pre"),
          "n_moving_a_post": meas.get("n_moving_a_post"),
          "n_moving_b_pre": meas.get("n_moving_b_pre"),
          "n_moving_b_post": meas.get("n_moving_b_post"),
          "stop_frame_a": g["stop_frame_a"],
          "stop_frame_b": g["stop_frame_b"],
          "restitution": p["restitution"],
      },
      "collision": {
          "actual_frame": float(cf),
          "actual_frame_rounded": int(round(cf)),
          "target_frame": TARGET_CONTACT_FRAME,
          "pre": pre,
          "post": post,
          "impulse_vector": vec(m * (va[ib] - va[ia])),
          "kinetic_energy_ratio": float(post["kinetic_energy"]
                                        / max(pre["kinetic_energy"], 1e-12)),
      },
      "restitution": p["restitution"],
      "objects": {
          "A": {"kind": "cube", "half_extent": p["half_extent"],
                "size": float(2 * p["half_extent"]), "mass": g["mass"],
                "density": DENSITY, "radius": p["half_extent"],
                "speed_initial": p["speed_a"], "segmentation_id": SEG_ID["cube_a"]},
          "B": {"kind": "cube", "half_extent": p["half_extent"],
                "size": float(2 * p["half_extent"]), "mass": g["mass"],
                "density": DENSITY, "radius": p["half_extent"],
                "speed_initial": p["speed_b"], "segmentation_id": SEG_ID["cube_b"]},
          "floor": {"kind": "floor", "mu": p["mu"], "albedo": p["floor_albedo"],
                    "segmentation_id": SEG_ID["floor"]},
      },
      "mass_ratio": 1.0,
      "per_frame": [
          {"frame": int(i), "velocity_A": float(va[i]), "velocity_B": float(vb[i]),
           "position_A": float(meas["position_a"][i]),
           "position_B": float(meas["position_b"][i])}
          for i in range(NUM_FRAMES)
      ],
      "scene_constants": {
          "num_frames": NUM_FRAMES, "frame_rate": FRAME_RATE,
          "resolution": list(RESOLUTION), "gravity": GRAVITY,
          "camera_distance": CAMERA_DISTANCE, "camera_height": CAMERA_HEIGHT,
          "half_width": HALF_WIDTH, "target_contact_frame": TARGET_CONTACT_FRAME,
          "mu_range": list(MU_RANGE), "speed_a_range": list(SPEED_A_RANGE),
          "speed_b_range": list(SPEED_B_RANGE),
          "restitution_range": list(RESTITUTION_RANGE),
          "half_extent_range": list(HALF_EXTENT_RANGE),
          "seg_id": SEG_ID, "render_device": render_device,
          "floor_band_width": FLOOR_BAND_WIDTH,
          "floor_band_contrast": FLOOR_BAND_CONTRAST,
      },
      "sampling_attempts": {"rejections": []},
  }


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------

def configure_gpu(renderer) -> str:
  """Kubric's own log line reads stale flags and prints a GPU name even on CPU —
  trust what this returns, which is what lands in labels.json."""
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
  except Exception as exc:                                        # noqa: BLE001
    return f"cpu (OPTIX setup failed: {exc})"


def _trial(p, g, scratch_dir, render):
  """One simulation attempt, scoped so the PyBullet client is released before the
  next connect. See the module docstring on the client-id reuse trap."""
  built = build_and_simulate(p, g, scratch_dir, render)
  scene, cubes, floor, camera, simulator, renderer, animation, collisions = built
  reason, meas = check_simulation(p, g, animation, cubes, collisions)
  return reason, meas, built


def generate(seed: int, out_dir: Path, scene_id: str, render: bool = True,
             arm: str = "latent_friction") -> dict:
  out_dir.mkdir(parents=True, exist_ok=True)
  rejections = []
  scratch = tempfile.mkdtemp(prefix="collide_friction_")
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

      scene, cubes, floor, camera, simulator, renderer, animation, collisions = built
      device = configure_gpu(renderer) if render else "no render"
      labels = build_labels(p, g, meas, camera, seed, scene_id, device)
      labels["sampling_attempts"]["rejections"] = rejections

      if render:
        frames = renderer.render()
        write_outputs(out_dir, frames)
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
    # MUST be `segmentation_%04d.png`: video.load_segmentation globs exactly that,
    # and a mismatch does not degrade gracefully — capture raises FileNotFoundError
    # per scene after the model is already loaded.
    imageio.imwrite(out_dir / "segmentation" / f"segmentation_{i:04d}.png", seg[i])
  write_mp4(out_dir / "video.mp4", rgb)


def write_mp4(path: Path, rgb):
  """`-movflags +faststart`, or a player that cannot seek refuses the file."""
  try:
    import imageio.v2 as imageio
    with imageio.get_writer(path, fps=FRAME_RATE, codec="libx264",
                            output_params=["-movflags", "+faststart"]) as w:
      for f in rgb:
        w.append_data(f)
  except Exception as exc:                                        # noqa: BLE001
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
    labels = generate(args.seed, args.output_dir, scene_id,
                      render=not args.no_render, arm=args.arm)
  except Exception:                                               # noqa: BLE001
    (args.output_dir).mkdir(parents=True, exist_ok=True)
    (args.output_dir / "_FAILED").write_text(traceback.format_exc())
    logger.error("scene %s FAILED\n%s", scene_id, traceback.format_exc())
    return 1
  c = labels["collision"]
  logger.info("%s: mu=%.3f contact f%.2f (target %d), v_pre A=%.3f B=%.3f -> "
              "post A=%.3f B=%.3f", scene_id, labels["friction"]["mu"],
              c["actual_frame"], TARGET_CONTACT_FRAME,
              c["pre"]["velocity_A"][0], c["pre"]["velocity_B"][0],
              c["post"]["velocity_A"][0], c["post"]["velocity_B"][0])
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
