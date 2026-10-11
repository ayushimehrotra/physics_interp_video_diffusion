#!/usr/bin/env python3
"""Generate one two-body collision scene for momentum probing.

    python3 generate_scene.py --seed 0 --output-dir out/scene_00000

Designed to run unmodified inside `kubricdockerhub/kubruntu` (see README.md).
Depends only on kubric, pybullet, numpy and imageio.

Physics design (see README.md "Design decisions" for the reasoning):
  * gravity is zero and both sphere CENTERS sit at the same height, so every
    contact normal is horizontal and the motion is exactly planar. Nothing
    else touches the spheres, so linear momentum is conserved to solver
    precision and that is the correctness check the pilot validates.
  * all friction terms are zero, so the spheres never spin up: angular
    velocity is identically zero in every label. That is expected, not a bug.
  * the initial separation is solved analytically so first contact lands on
    frame TARGET_CONTACT_FRAME. The simulation is then run WITHOUT rendering
    to check where contact actually landed; scenes that miss by more than
    CONTACT_TOLERANCE frames are rejected and resampled before a single
    pixel is rendered.

Edited aftermaths (the register's conservation / ladder sets). The collision is
always simulated honestly and validated first; an edit then overwrites the
post-contact velocities and re-keyframes the spheres along them, so every clip
of one seed shares a bit-identical pre-contact segment:
  --violate --violation-mode drift   legacy probe_set edit: a common drift of both
                                     balls in a random direction (VIOLATION_SCALE_RANGE)
  --violate --violation-mode split   conservation_set_v1 edit: keep the total kick
                                     a + b, move the split b/a away from m_A/m_B. This
                                     is a common drift along the contact normal, so
                                     separation speed (restitution) is untouched.
  --edit-restitution E               a ladder arm: rescale both deflections by
                                     (1 + E) / (1 + e_sim). Exact; E > 1 creates energy.
  --restitution E                    pin the simulated pair restitution.
The edit algebra lives in physics_interp/momentum/physics.py and is loaded by
file path so this script still runs in the Kubric interpreter (no torch there).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

logger = logging.getLogger("generate_scene")


def _load_physics():
  """Import physics_interp/momentum/physics.py without importing the package (which needs torch)."""
  import importlib.util
  path = Path(__file__).resolve().parents[1] / "physics_interp" / "momentum" / "physics.py"
  spec = importlib.util.spec_from_file_location("momentum_physics", path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module      # dataclasses resolve annotations through sys.modules
  spec.loader.exec_module(module)
  return module


PHYSICS = _load_physics()

# --------------------------------------------------------------------------------------
# Scene constants. Everything here is fixed across the dataset and copied into labels.json
# so a scene can be reconstructed from its labels alone.
# --------------------------------------------------------------------------------------

NUM_FRAMES = 49                 # frames 0..48 inclusive
FRAME_RATE = 8                  # fps, matches CogVideoX-2b clips
STEP_RATE = 240                 # physics steps per second (must be a multiple of FRAME_RATE)
RESOLUTION = (720, 480)         # (width, height), CogVideoX-2b native
TARGET_CONTACT_FRAME = 20
CONTACT_TOLERANCE = 1.0         # accept |actual - target| <= this, in frames

# Camera: straight down, centred on the origin. CAMERA_DISTANCE is measured from the
# plane containing the sphere centres, which is what sets the visible world extent.
# Raised from 14 to buy the room a wide speed range needs: A must travel
# speed * 2.5 s before contact and still start fully inside the frame.
CAMERA_DISTANCE = 20.0
FOCAL_LENGTH = 50.0             # mm
SENSOR_WIDTH = 36.0             # mm

# Radii are continuous, not a 3-level grid. Levels made every derived quantity
# (mass, mass ratio, momentum) land on a handful of discrete values, which a probe
# can fit as a lookup rather than a linear readout, and which makes any regression
# R^2 a statement about 6 clusters instead of a continuum.
RADIUS_RANGE = (0.30, 0.60)
MAX_RADIUS = RADIUS_RANGE[1]
CENTER_HEIGHT = MAX_RADIUS + 0.05   # every centre sits here: contact normals stay horizontal

# Two material levels: visually distinct, different densities. Density is *visible*
# via colour on purpose. Momentum is mass x velocity, so if mass were unreadable
# from the pixels then momentum would not be inferable from the video at all and a
# failed probe would say nothing about the representation.
MATERIALS = {
    "metal": {
        "density": 6.0,
        "color": (0.72, 0.74, 0.78),
        "metallic": 0.9,
        "roughness": 0.25,
        "specular": 0.9,
        "ior": 2.5,
    },
    "rubber": {
        "density": 1.5,
        "color": (0.80, 0.10, 0.10),
        "metallic": 0.0,
        "roughness": 0.8,
        "specular": 0.3,
        "ior": 1.45,
    },
}

RESTITUTION_RANGE = (0.05, 1.0)  # pair coefficient, continuous; see restitution note

# World units / second. The floor is set by legibility, not by physics: at the
# visible extent of 14.4 units across 720 px, 0.22 units/s is 1.4 px/frame, and a
# ball drifting that slowly has a velocity the video barely encodes — which would
# make a failed momentum probe uninformative about the representation. 0.35 is
# ~2.2 px/frame, about 2.7 latent cells of travel over the 20 frames before
# contact. The ceiling is set by the frame: A must travel speed * 2.5 s and still
# start fully in view.
SPEED_RANGE = (0.35, 1.55)
IMPACT_PARAM_MAX_FRAC = 0.9      # |b| <= this * (r_a + r_b); 1.0 would be a grazing miss

# Probability that B is already moving at frame 0 rather than sitting at rest.
#
# This is the single most important change for probing *total* momentum. With B
# always at rest the total momentum of the scene is exactly A's momentum, so a
# probe that finds "the momentum of the one moving ball" is indistinguishable from
# one that finds the momentum of the system. Once both bodies move, total momentum
# is a vector sum that equals neither ball's, and pointing at it requires combining
# two objects' masses and velocities.
B_MOVING_PROB = 0.7

# Contact point placement. Non-zero decorrelates *where* the collision happens from
# *when* it happens; at 0.0 every collision is at the frame centre and a probe could
# read contact location off a constant.
CONTACT_JITTER_RADIUS = 1.2

# ---- the momentum/mass confound, and the stratum that breaks it ---------------
#
# Momentum is mass x speed, so with a wide mass range and a narrow speed range the
# two are nearly the same variable: the original pilot had corr(|p|, mass_A) = 0.97,
# which means a probe "decoding momentum" could have been reading apparent size and
# colour. Widening the speed range helps but cannot fix it outright — decorrelating
# a product needs the two factors to span comparable log ranges, and speed is capped
# by how far a ball can travel and stay in frame.
#
# So a configurable fraction of scenes are drawn from a *matched-momentum* stratum:
# |p_A| comes from a fixed band that does not depend on mass at all, mass roams its
# full range, and speed takes up the slack. The rest of the dataset keeps the wide
# ranges, so overall coverage does not shrink, and `stratum` is recorded per scene so
# analysis can select on it. Measured over 400 scenes, corr(log|p_A|, log mass_A)
# falls from +0.97 in the original pilot to +0.29 inside this stratum.
#
# The band is a range rather than a single value on purpose. Pinning |p_A| exactly
# would decorrelate it from mass but leave it with no variance, so the stratum could
# only falsify a probe ("momentum is constant here, does your readout stay
# constant?") and never fit one.
#
# Its width is the knob that trades momentum variance against decorrelation. Momentum
# and mass are exactly independent over the masses for which the *whole* band is
# reachable at a legal speed — `[hi/v_max, lo/v_min]`, which is 0.81 to 2.29 units
# here — and only partially independent at the edges, where some of the band is
# unreachable and the draw gets rejected. A narrower band widens that core. At
# (0.80, 1.25) against a 4.4x speed range the core spans 2.8x in mass, which is
# enough for a mass-only readout to be clearly distinguishable from a momentum one.
CONTROL_STRATUM_FRAC = 0.35
CONTROL_MOMENTUM_RANGE = (0.80, 1.25)

# ---- momentum-conservation violations ----------------------------------------
#
# Every scene here conserves momentum exactly, which makes "is momentum conserved"
# a constant and therefore unprobeable — a classifier needs both classes. In a
# violating scene the collision is simulated normally and then the post-contact
# velocities are overwritten with ones that do *not* conserve momentum, and the
# spheres are re-keyframed to fly along those. Everything before contact, and every
# visual property, is untouched: the two classes differ only in what happens after
# the balls touch.
VIOLATION_SCALE_RANGE = (0.35, 0.75)   # |dp| / |p_pre| injected, either sign (drift mode)
# Split mode (conservation_set_v1): the split fraction b/(a+b) moves by this much, either
# sign, with a + b held. Both balls receive the same extra velocity along the contact
# normal, so the edit is a common drift along n and restitution is untouched.
SPLIT_DELTA_RANGE = (0.10, 0.35)

# Frame-0 sanity margins, in world units.
MIN_SURFACE_GAP_AT_START = 0.15   # surface-to-surface clearance between the spheres
FRAME_MARGIN = 0.10               # keep silhouettes this far inside the image border
SILHOUETTE_SAFETY = 1.1           # perspective makes a sphere project slightly wider than r

# Which frames to read pre/post collision velocities from. Both are far enough from
# TARGET_CONTACT_FRAME that a +/-1 frame timing error cannot contaminate them.
PRE_FRAME = 18
POST_FRAME = 22

MAX_ATTEMPTS = 40
SAMPLES_PER_PIXEL = 64

SEG_ID = {"A": 1, "B": 2, "floor": 0}


# --------------------------------------------------------------------------------------
# Geometry: solve for the initial condition that puts first contact on the target frame
# --------------------------------------------------------------------------------------

def half_extents():
  """Half width/height of the visible world region in the plane of the sphere centres."""
  sensor_height = SENSOR_WIDTH / RESOLUTION[0] * RESOLUTION[1]
  half_w = 0.5 * CAMERA_DISTANCE * SENSOR_WIDTH / FOCAL_LENGTH
  half_h = 0.5 * CAMERA_DISTANCE * sensor_height / FOCAL_LENGTH
  return half_w, half_h


def sphere_mass(radius: float, density: float) -> float:
  return float(density * 4.0 / 3.0 * np.pi * radius ** 3)


def _speed_for(momentum: float, mass: float) -> float | None:
  """Speed realising a target momentum, or None if it is out of range."""
  speed = momentum / mass
  return speed if SPEED_RANGE[0] <= speed <= SPEED_RANGE[1] else None


def sample_params(rng: np.random.Generator, violate: bool = False) -> tuple[dict | None, str]:
  """Draw one candidate parameter set.

  Momentum is sampled *before* speed and speed is derived as ``p / m``. Drawing
  speed directly would make momentum a deterministic increasing function of mass
  whenever mass has the wider range, which is exactly how the pilot ended up with
  momentum and mass 0.97-correlated. Sampling the target quantity first and letting
  the free factor absorb it is what puts them on separate axes.

  Returns ``(params, reason)``; ``params`` is None when the draw is infeasible, and
  the reason is recorded so the rejection log says which constraint bit.
  """
  mat_a = str(rng.choice(list(MATERIALS)))
  mat_b = str(rng.choice(list(MATERIALS)))
  r_a = float(rng.uniform(*RADIUS_RANGE))
  r_b = float(rng.uniform(*RADIUS_RANGE))
  m_a = sphere_mass(r_a, MATERIALS[mat_a]["density"])
  m_b = sphere_mass(r_b, MATERIALS[mat_b]["density"])

  control = bool(rng.uniform() < CONTROL_STRATUM_FRAC)
  if control:
    # Matched-momentum stratum: |p_A| drawn from a fixed band that does not depend
    # on mass at all, with speed absorbing whatever mass was sampled.
    lo, hi = CONTROL_MOMENTUM_RANGE
    target_p = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
  else:
    # Log-uniform over everything this mass could carry at a legal speed, so the
    # wide-range half of the dataset stays evenly covered in log momentum.
    lo, hi = m_a * SPEED_RANGE[0], m_a * SPEED_RANGE[1]
    target_p = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

  speed_a = _speed_for(target_p, m_a)
  if speed_a is None:
    return None, "momentum_unreachable_at_this_mass"

  b_moves = bool(rng.uniform() < B_MOVING_PROB)
  speed_b = 0.0
  if b_moves:
    lo, hi = m_b * SPEED_RANGE[0], m_b * SPEED_RANGE[1]
    speed_b = _speed_for(float(np.exp(rng.uniform(np.log(lo), np.log(hi)))), m_b)
    if speed_b is None:
      return None, "momentum_unreachable_at_this_mass"

  jitter_r = CONTACT_JITTER_RADIUS * np.sqrt(rng.uniform())
  jitter_theta = rng.uniform(0, 2 * np.pi)

  params = {
      "radius_a": r_a,
      "radius_b": r_b,
      "material_a": mat_a,
      "material_b": mat_b,
      "density_a": MATERIALS[mat_a]["density"],
      "density_b": MATERIALS[mat_b]["density"],
      "speed_a": float(speed_a),
      "speed_b": float(speed_b),
      "direction_a_rad": float(rng.uniform(0, 2 * np.pi)),
      "direction_b_rad": float(rng.uniform(0, 2 * np.pi)) if b_moves else 0.0,
      "b_moving": b_moves,
      "target_momentum_a": float(target_p),
      "stratum": "matched_momentum" if control else "wide",
      "impact_parameter_frac": float(rng.uniform(0, IMPACT_PARAM_MAX_FRAC)),
      "impact_parameter_sign": int(rng.choice([-1, 1])),
      "restitution": float(rng.uniform(*RESTITUTION_RANGE)),
      "contact_offset": [float(jitter_r * np.cos(jitter_theta)),
                         float(jitter_r * np.sin(jitter_theta))],
      "violates_momentum": bool(violate),
      "violation_scale": (
          float(rng.uniform(*VIOLATION_SCALE_RANGE)) * float(rng.choice([-1.0, 1.0]))
          if violate else 0.0),
      "violation_direction_rad": float(rng.uniform(0, 2 * np.pi)) if violate else 0.0,
  }
  return params, ""


def solve_geometry(p: dict) -> dict:
  """Analytic initial condition placing first contact exactly at TARGET_CONTACT_FRAME.

  Both spheres travel in straight lines at constant velocity (no gravity, no
  friction, nothing else in the scene), so their *separation* is also a straight
  line in time: ``d(t) = d_0 + v_rel * t``. First contact is the entering root of
  ``|d(t)| == r_a + r_b``. Rather than solve that quadratic, place the geometry so
  the root lands on ``t_contact`` by construction and let the simulator confirm it.

  With B at rest this reduces exactly to the single-body case — the relative
  velocity is just A's — which is why one code path covers both.
  """
  r_a, r_b = p["radius_a"], p["radius_b"]
  contact_radius = r_a + r_b
  t_contact = TARGET_CONTACT_FRAME / FRAME_RATE

  theta_a, theta_b = p["direction_a_rad"], p["direction_b_rad"]
  vel_a = p["speed_a"] * np.array([np.cos(theta_a), np.sin(theta_a), 0.0])
  vel_b = p["speed_b"] * np.array([np.cos(theta_b), np.sin(theta_b), 0.0])

  v_rel = vel_a - vel_b
  rel_speed = float(np.linalg.norm(v_rel))
  if rel_speed < 1e-6:
    # Parallel and equal: they never approach, so there is no contact to place.
    return {"infeasible": "zero_relative_velocity"}

  u = v_rel / rel_speed                                      # approach direction
  w = np.array([-u[1], u[0], 0.0])                           # left perpendicular

  b = p["impact_parameter_sign"] * p["impact_parameter_frac"] * contact_radius
  # Unit vector from A's centre to B's centre at contact. Its component along the
  # *relative* velocity is positive, which is what makes t_contact the entering root.
  along = np.sqrt(max(contact_radius ** 2 - b ** 2, 0.0))
  normal = (along * u + b * w) / contact_radius

  contact_point = np.array([p["contact_offset"][0], p["contact_offset"][1], CENTER_HEIGHT])
  pos_a_contact = contact_point - r_a * normal
  pos_b_contact = contact_point + r_b * normal

  pos_a_start = pos_a_contact - vel_a * t_contact
  pos_b_start = pos_b_contact - vel_b * t_contact

  return {
      "contact_radius": contact_radius,
      "t_contact_target_s": t_contact,
      "impact_parameter": float(b),
      "impact_parameter_abs": float(abs(b)),
      "relative_speed": rel_speed,
      "contact_normal_analytic": normal.tolist(),
      "contact_point_analytic": contact_point.tolist(),
      "pos_a_contact_analytic": pos_a_contact.tolist(),
      "pos_b_contact_analytic": pos_b_contact.tolist(),
      "pos_a_start": pos_a_start.tolist(),
      "pos_b_start": pos_b_start.tolist(),
      "vel_a_start": vel_a.tolist(),
      "vel_b_start": vel_b.tolist(),
      "initial_separation": float(np.linalg.norm(pos_a_start - pos_b_start)),
      "initial_surface_gap": float(
          np.linalg.norm(pos_a_start - pos_b_start) - contact_radius),
      "travel_before_contact": float(p["speed_a"] * t_contact),
  }


def check_frame0(p: dict, geo: dict) -> str | None:
  """Return a rejection reason, or None if the frame-0 layout is usable."""
  if "infeasible" in geo:
    return geo["infeasible"]
  if geo["initial_surface_gap"] < MIN_SURFACE_GAP_AT_START:
    return "overlap_or_too_close_at_frame0"

  half_w, half_h = half_extents()
  for tag, pos, radius in (("a", geo["pos_a_start"], p["radius_a"]),
                           ("b", geo["pos_b_start"], p["radius_b"])):
    reach = radius * SILHOUETTE_SAFETY + FRAME_MARGIN
    if abs(pos[0]) + reach > half_w or abs(pos[1]) + reach > half_h:
      return f"{tag}_out_of_frame_at_frame0"
  return None


def derive_masses(p: dict) -> dict:
  vol_a = 4.0 / 3.0 * np.pi * p["radius_a"] ** 3
  vol_b = 4.0 / 3.0 * np.pi * p["radius_b"] ** 3
  mass_a = p["density_a"] * vol_a
  mass_b = p["density_b"] * vol_b
  return {
      "volume_a": float(vol_a), "volume_b": float(vol_b),
      "mass_a": float(mass_a), "mass_b": float(mass_b),
      "mass_ratio": float(mass_a / mass_b),
  }


def _pre_state(p, geo, masses, animation, spheres):
  """The pre-contact state in the momentum module's convention (planar, n from A to B)."""
  n = np.asarray(geo["contact_normal_analytic"], dtype=np.float64)[:2]
  return PHYSICS.PreState(
      m_a=masses["mass_a"], m_b=masses["mass_b"],
      v_a=np.asarray(animation[spheres["a"]]["velocity"][PRE_FRAME], dtype=np.float64)[:2],
      v_b=np.asarray(animation[spheres["b"]]["velocity"][PRE_FRAME], dtype=np.float64)[:2],
      n=n / np.linalg.norm(n))


def _post_velocities(animation, spheres):
  return {tag: np.asarray(animation[spheres[tag]]["velocity"][POST_FRAME], dtype=np.float64)
          for tag in ("a", "b")}


def rekeyframe(spheres, animation, contact_frame, edited):
  """Re-keyframe both spheres along ``edited[tag]`` (3-vectors) from the first frame after contact.

  Anchored on the last frame before contact, so the trajectory deviates exactly at
  the collision. ``anchor + v * dt`` is not bit-identical to Bullet's integration
  (a constant offset of order 1e-2 world units can remain, §58); it is identical
  across every edited arm of a seed, which is what the protocols need.
  """
  first_moving = int(np.ceil(contact_frame))
  for tag in ("a", "b"):
    sphere = spheres[tag]
    anchor = np.array(animation[sphere]["position"][first_moving - 1], dtype=np.float64)
    for frame in range(first_moving, NUM_FRAMES):
      dt = (frame - (first_moving - 1)) / FRAME_RATE
      pos = anchor + edited[tag] * dt
      sphere.position = tuple(pos)
      sphere.keyframe_insert("position", frame)
      animation[sphere]["position"][frame] = pos
      animation[sphere]["velocity"][frame] = edited[tag]


def drift_violation_velocities(p, masses, animation, spheres):
  """Legacy probe_set edit: both balls take the same velocity change in a random direction."""
  theta = p["violation_direction_rad"]
  direction = np.array([np.cos(theta), np.sin(theta), 0.0])
  p_pre = (masses["mass_a"] * np.array(animation[spheres["a"]]["velocity"][PRE_FRAME])
           + masses["mass_b"] * np.array(animation[spheres["b"]]["velocity"][PRE_FRAME]))
  delta_p = p["violation_scale"] * float(np.linalg.norm(p_pre)) * direction
  post = _post_velocities(animation, spheres)
  shift = delta_p / (masses["mass_a"] + masses["mass_b"])
  edited = {tag: post[tag] + shift for tag in ("a", "b")}
  return edited, {"kind": "drift", "dp": delta_p.tolist(), "violation_scale": p["violation_scale"],
                  "violation_direction_rad": theta}


def split_violation_velocities(p, geo, masses, animation, spheres, seed):
  """conservation_set_v1 edit: keep a + b, move the split. Draws delta from its own rng stream."""
  rng = np.random.default_rng([int(seed), 7919])
  delta_split = float(rng.uniform(*SPLIT_DELTA_RANGE)) * float(rng.choice([-1.0, 1.0]))
  pre = _pre_state(p, geo, masses, animation, spheres)
  post = _post_velocities(animation, spheres)
  pa, pb, rec = PHYSICS.split_edit(pre, post["a"], post["b"], delta_split)
  edited = {"a": np.array([pa[0], pa[1], 0.0]), "b": np.array([pb[0], pb[1], 0.0])}
  return edited, {"kind": "split", **rec}


def restitution_arm_velocities(p, geo, masses, animation, spheres, e_sim, e_new):
  """Ladder arm: the legal outcome at e_new, by rescaling both deflections."""
  pre = _pre_state(p, geo, masses, animation, spheres)
  post = _post_velocities(animation, spheres)
  pa, pb, rec = PHYSICS.restitution_edit(pre, post["a"], post["b"], e_sim, e_new)
  edited = {"a": np.array([pa[0], pa[1], 0.0]), "b": np.array([pb[0], pb[1], 0.0])}
  return edited, {"kind": "restitution", **rec}


def exit_frames(animation, spheres):
  """First frame at which each sphere's centre leaves the visible extent (NUM_FRAMES if never)."""
  half_w, half_h = half_extents()
  out = {}
  for tag in ("a", "b"):
    pos = np.asarray(animation[spheres[tag]]["position"], dtype=np.float64)
    outside = (np.abs(pos[:, 0]) > half_w) | (np.abs(pos[:, 1]) > half_h)
    out[tag.upper()] = int(np.argmax(outside)) if outside.any() else NUM_FRAMES
  return out


def apply_violation(p, geo, masses, spheres, animation, contact_frame, mode="drift", seed=0):
  """Overwrite post-contact motion so momentum is *not* conserved (see module docstring)."""
  if mode == "split":
    edited, rec = split_violation_velocities(p, geo, masses, animation, spheres, seed)
  else:
    edited, rec = drift_violation_velocities(p, masses, animation, spheres)
  rekeyframe(spheres, animation, contact_frame, edited)
  return edited, rec


def apply_restitution_edit(p, geo, masses, spheres, animation, contact_frame, e_sim, e_new):
  edited, rec = restitution_arm_velocities(p, geo, masses, animation, spheres, e_sim, e_new)
  rekeyframe(spheres, animation, contact_frame, edited)
  return edited, rec


# --------------------------------------------------------------------------------------
# Scene construction and simulation
# --------------------------------------------------------------------------------------

def configure_gpu(renderer) -> str:
  """Point Cycles at a GPU. Returns a human-readable description of what was selected.

  MUST be called *after* the Blender renderer is constructed. Two traps here:

    1. KUBRIC_USE_GPU only flips cycles.device to "GPU". It never sets Blender's
       compute_device_type, which is NONE in a fresh bpy session, so Cycles falls
       back to CPU.
    2. Blender.__init__ calls bpy.ops.wm.read_factory_settings(), which resets
       compute_device_type back to NONE -- so configuring it beforehand, or saving it
       into userpref.blend, has no effect. Worse, the "Using the following GPU
       Device(s)" line kubric logs at that point reads stale per-device `use` flags
       and prints a GPU name even when rendering will happen on the CPU. Do not
       trust that log line; trust the render_device field in labels.json.
  """
  if os.getenv("KUBRIC_USE_GPU", "").lower() not in ("1", "true", "t", "yes"):
    renderer.blender_scene.cycles.device = "CPU"
    return "cpu (KUBRIC_USE_GPU not set)"
  import bpy
  prefs = bpy.context.preferences.addons["cycles"].preferences
  for backend in ("OPTIX", "CUDA"):
    try:
      prefs.compute_device_type = backend
      prefs.get_devices()
    except Exception:  # pylint: disable=broad-except
      continue
    matching = [d for d in prefs.devices if d.type == backend]
    if matching:
      for device in prefs.devices:
        device.use = (device.type == backend)
      renderer.blender_scene.cycles.device = "GPU"
      if backend == "OPTIX":
        # The default OpenImageDenoise runs on the CPU and costs more than the render
        # itself here: 6.0 s/frame vs 2.4 s/frame with the OptiX denoiser.
        try:
          renderer.blender_scene.cycles.denoiser = "OPTIX"
        except (TypeError, ValueError):
          pass  # older Blender without the OptiX denoiser; keep kubric's default
      return f"{backend}: {', '.join(d.name for d in matching)}"
  renderer.blender_scene.cycles.device = "CPU"
  return "cpu (no GPU device found)"


def build_and_simulate(p: dict, geo: dict, masses: dict, scratch_dir: str, render: bool):
  """Build the scene, attach views, run the simulation. Returns everything downstream needs."""
  import kubric as kb
  from kubric.simulator import PyBullet
  import pybullet as pb

  scene = kb.Scene(
      frame_start=0,
      frame_end=NUM_FRAMES - 1,
      frame_rate=FRAME_RATE,
      step_rate=STEP_RATE,
      resolution=RESOLUTION,
      gravity=(0.0, 0.0, 0.0),
      background=kb.Color(0.30, 0.30, 0.32, 1.0),
      ambient_illumination=kb.Color(0.04, 0.04, 0.04, 1.0),
  )

  # Views must exist before assets are added: kubric pushes each asset to the views
  # that are linked at the time of `scene += asset`.
  simulator = PyBullet(scene, scratch_dir)

  # Bullet only routes penetration recovery through the split-impulse pass once
  # penetration exceeds splitImpulsePenetrationThreshold (default -0.04). Shallower
  # contacts -- which is every contact here -- get an ERP velocity bias applied to the
  # real velocity instead, which *adds energy*: measured restitution came out at
  # e + 0.08 and elastic collisions gained 5% kinetic energy. Setting the threshold to
  # zero sends all penetration recovery through the split pass, where it corrects
  # position without touching velocity. With this, measured restitution matches the
  # target to 4 decimals and e=1.0 conserves kinetic energy exactly.
  pb.setPhysicsEngineParameter(splitImpulsePenetrationThreshold=0.0,
                               physicsClientId=simulator.physics_client)
  renderer = None
  if render:
    from kubric.renderer import Blender
    renderer = Blender(scene, scratch_dir, use_denoising=True,
                       adaptive_sampling=True, samples_per_pixel=SAMPLES_PER_PIXEL)
    # Kubric defaults to the Filmic view transform, which desaturates and flattens
    # contrast. Standard keeps the material colours separable, which matters more here
    # than filmic highlight rolloff on a scene with no highlights to roll off.
    renderer.blender_scene.view_settings.view_transform = "Standard"

  # Straight-down camera. Setting the quaternion directly avoids look_at(), whose
  # up-vector is degenerate when the view direction is exactly -Z.
  camera = kb.PerspectiveCamera(
      name="camera",
      focal_length=FOCAL_LENGTH,
      sensor_width=SENSOR_WIDTH,
      position=(0.0, 0.0, CENTER_HEIGHT + CAMERA_DISTANCE),
      quaternion=(1.0, 0.0, 0.0, 0.0),
  )
  scene += camera
  scene.camera = camera

  # Tilted ~20 deg off vertical: straight-down light renders the spheres as flat discs
  # with no shading gradient. The tilt costs a small shadow offset (see README).
  scene += kb.DirectionalLight(
      name="sun",
      position=(5.0, -4.0, 14.0),
      look_at=(0.0, 0.0, 0.0),
      intensity=3.0,
  )

  floor = kb.Cube(
      name="floor",
      scale=(20.0, 20.0, 0.5),
      position=(0.0, 0.0, -0.5),
      material=kb.PrincipledBSDFMaterial(color=kb.Color(0.42, 0.42, 0.44, 1.0),
                                         metallic=0.0, roughness=0.8, specular=0.2),
      static=True, friction=0.0, restitution=0.0,
      background=True, segmentation_id=SEG_ID["floor"],
  )
  scene += floor

  spheres = {}
  for tag, seg_key in (("a", "A"), ("b", "B")):
    spec = MATERIALS[p[f"material_{tag}"]]
    sphere = kb.Sphere(
        name=f"sphere_{tag}",
        scale=p[f"radius_{tag}"],
        position=geo["pos_a_start"] if tag == "a" else geo["pos_b_start"],
        velocity=geo["vel_a_start"] if tag == "a" else geo["vel_b_start"],
        mass=masses[f"mass_{tag}"],
        friction=0.0,
        # PyBullet combines restitution multiplicatively, so each object carries the
        # square root of the pair coefficient we want. Verified per scene by
        # restitution_measured in labels.json.
        restitution=float(np.sqrt(p["restitution"])),
        material=kb.PrincipledBSDFMaterial(
            color=kb.Color(*spec["color"], 1.0),
            metallic=spec["metallic"], roughness=spec["roughness"],
            specular=spec["specular"], ior=spec["ior"]),
        segmentation_id=SEG_ID[seg_key],
    )
    scene += sphere
    spheres[tag] = sphere

  # PyBullet defaults to linear/angular damping of 0.04, which would bleed momentum
  # every step: A would arrive late and the conservation check would fail outright.
  # Kubric never touches these, so zero them explicitly.
  for sphere in spheres.values():
    body_id = sphere.linked_objects[simulator]
    pb.changeDynamics(body_id, -1, linearDamping=0.0, angularDamping=0.0,
                      rollingFriction=0.0, spinningFriction=0.0,
                      physicsClientId=simulator.physics_client)

  animation, collisions = simulator.run(frame_start=0, frame_end=NUM_FRAMES - 1)
  return scene, spheres, floor, camera, simulator, renderer, animation, collisions


def analyse_collisions(collisions, spheres, floor):
  """Split kubric's contact events into sphere-sphere and everything-else."""
  sphere_set = {spheres["a"], spheres["b"]}
  pair_events, floor_events = [], []
  for event in collisions:
    instances = set(event["instances"])
    if instances == sphere_set:
      pair_events.append(event)
    elif floor in instances:
      floor_events.append(event)
  return pair_events, floor_events


def check_simulation(p, geo, masses, animation, spheres, pair_events, floor_events):
  """Physics acceptance checks. Returns (reason_or_None, measurements)."""
  anim_a = animation[spheres["a"]]
  anim_b = animation[spheres["b"]]
  vel_a = np.array(anim_a["velocity"])
  vel_b = np.array(anim_b["velocity"])

  if floor_events:
    return "floor_contact", {}
  if not pair_events:
    return "no_contact", {}

  contact_frame = float(min(e["frame"] for e in pair_events))
  discrepancy = contact_frame - TARGET_CONTACT_FRAME
  if abs(discrepancy) > CONTACT_TOLERANCE:
    return "contact_frame_miss", {"contact_frame": contact_frame, "discrepancy": discrepancy}

  # Velocities must be piecewise constant: one collision, nothing else touching.
  pre_const = np.allclose(vel_a[:PRE_FRAME + 1], vel_a[0], atol=1e-6) and \
              np.allclose(vel_b[:PRE_FRAME + 1], vel_b[0], atol=1e-6)
  post_const = np.allclose(vel_a[POST_FRAME:], vel_a[POST_FRAME], atol=1e-6) and \
               np.allclose(vel_b[POST_FRAME:], vel_b[POST_FRAME], atol=1e-6)
  if not (pre_const and post_const):
    return "velocity_not_piecewise_constant", {"contact_frame": contact_frame}

  m_a, m_b = masses["mass_a"], masses["mass_b"]
  p_pre = m_a * vel_a[PRE_FRAME] + m_b * vel_b[PRE_FRAME]
  p_post = m_a * vel_a[POST_FRAME] + m_b * vel_b[POST_FRAME]
  momentum_rel_error = float(np.linalg.norm(p_post - p_pre) / np.linalg.norm(p_pre))

  measurements = {
      "contact_frame": contact_frame,
      "discrepancy": discrepancy,
      "momentum_rel_error": momentum_rel_error,
  }
  if momentum_rel_error > 1e-3:
    return "momentum_violation", measurements
  return None, measurements


# --------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------

def project(camera, point, frame=0):
  """World point -> (pixel_x, pixel_y, in_front_of_camera)."""
  coords = camera.project_point(np.asarray(point, dtype=np.float64), frame=frame)
  return (float(coords[0]) * RESOLUTION[0], float(coords[1]) * RESOLUTION[1], float(coords[2]))


def kinetic_energy(mass, radius, velocity, angular_velocity):
  """Translational + rotational KE. The rotational term is always zero here (no friction)."""
  inertia = 0.4 * mass * radius ** 2
  return float(0.5 * mass * np.dot(velocity, velocity) +
               0.5 * inertia * np.dot(angular_velocity, angular_velocity))


def build_labels(p, geo, masses, scene, spheres, camera, animation, pair_events, measurements,
                 seed, scene_id, seg_stats, gpu_desc):
  m = {"a": masses["mass_a"], "b": masses["mass_b"]}
  radius = {"a": p["radius_a"], "b": p["radius_b"]}
  anim = {tag: animation[spheres[tag]] for tag in ("a", "b")}

  per_frame = []
  for frame in range(NUM_FRAMES):
    entry = {"frame": frame, "time_s": frame / FRAME_RATE}
    total_p = np.zeros(3)
    total_ke = 0.0
    for tag in ("a", "b"):
      pos = np.array(anim[tag]["position"][frame], dtype=np.float64)
      vel = np.array(anim[tag]["velocity"][frame], dtype=np.float64)
      ang = np.array(anim[tag]["angular_velocity"][frame], dtype=np.float64)
      momentum = m[tag] * vel
      px, py, in_front = project(camera, pos, frame=frame)
      obj = {
          "position": pos.tolist(),
          "pixel": [px, py],
          "in_frame": bool(0 <= px < RESOLUTION[0] and 0 <= py < RESOLUTION[1] and in_front > 0),
          "velocity": vel.tolist(),
          "speed": float(np.linalg.norm(vel)),
          "angular_velocity": ang.tolist(),
          "momentum": momentum.tolist(),
          "kinetic_energy": kinetic_energy(m[tag], radius[tag], vel, ang),
      }
      if seg_stats is not None:
        obj["segmentation_pixel_count"] = seg_stats[tag]["count"][frame]
        obj["segmentation_centroid_pixel"] = seg_stats[tag]["centroid"][frame]
      entry[tag.upper()] = obj
      total_p += momentum
      total_ke += obj["kinetic_energy"]
    entry["total_momentum"] = total_p.tolist()
    entry["total_kinetic_energy"] = total_ke
    per_frame.append(entry)

  pre = {tag: np.array(anim[tag]["velocity"][PRE_FRAME], dtype=np.float64) for tag in ("a", "b")}
  post = {tag: np.array(anim[tag]["velocity"][POST_FRAME], dtype=np.float64) for tag in ("a", "b")}
  p_pre = m["a"] * pre["a"] + m["b"] * pre["b"]
  p_post = m["a"] * post["a"] + m["b"] * post["b"]
  ke_pre = sum(kinetic_energy(m[t], radius[t], pre[t], np.zeros(3)) for t in ("a", "b"))
  ke_post = sum(kinetic_energy(m[t], radius[t], post[t], np.zeros(3)) for t in ("a", "b"))

  # Impulse straight from B's momentum change: exact, and independent of how the
  # solver reported its per-step normal forces.
  impulse_vec = m["b"] * (post["b"] - pre["b"])

  first_event = min(pair_events, key=lambda e: e["frame"])
  contact_pos = np.array(first_event["position"], dtype=np.float64)
  contact_normal = np.array(first_event["contact_normal"], dtype=np.float64)
  cx, cy, _ = project(camera, contact_pos, frame=int(round(measurements["contact_frame"])))

  # Restitution actually realised, along the contact normal.
  n_hat = contact_normal / max(np.linalg.norm(contact_normal), 1e-12)
  rel_pre = float(np.dot(pre["a"] - pre["b"], n_hat))
  rel_post = float(np.dot(post["a"] - post["b"], n_hat))
  restitution_measured = float(abs(rel_post) / abs(rel_pre)) if abs(rel_pre) > 1e-9 else None

  return {
      "schema_version": 1,
      "scene_id": scene_id,
      "seed": seed,
      "sampled_parameters": p,
      "derived": {**masses, **{k: geo[k] for k in (
          "contact_radius", "impact_parameter", "impact_parameter_abs",
          "initial_separation", "initial_surface_gap", "travel_before_contact",
          "t_contact_target_s")}},
      "objects": {
          "A": {"radius": p["radius_a"], "material": p["material_a"],
                "density": p["density_a"], "volume": masses["volume_a"],
                "mass": masses["mass_a"], "friction": 0.0,
                "restitution_object": float(np.sqrt(p["restitution"])),
                "segmentation_id": SEG_ID["A"], "color_rgb": MATERIALS[p["material_a"]]["color"]},
          "B": {"radius": p["radius_b"], "material": p["material_b"],
                "density": p["density_b"], "volume": masses["volume_b"],
                "mass": masses["mass_b"], "friction": 0.0,
                "restitution_object": float(np.sqrt(p["restitution"])),
                "segmentation_id": SEG_ID["B"], "color_rgb": MATERIALS[p["material_b"]]["color"]},
      },
      "mass_ratio": masses["mass_ratio"],
      "restitution": p["restitution"],
      # Probe-facing scene labels, hoisted to the top level so a probe target is a
      # short path rather than a reach into sampled_parameters.
      "stratum": p["stratum"],
      "b_moving": p["b_moving"],
      "violates_momentum": p["violates_momentum"],
      "momentum_conserved": (not p["violates_momentum"]),
      "collision": {
          "target_frame": TARGET_CONTACT_FRAME,
          "actual_frame": measurements["contact_frame"],
          "actual_frame_rounded": int(round(measurements["contact_frame"])),
          "discrepancy_frames": measurements["discrepancy"],
          "contact_point_world": contact_pos.tolist(),
          "contact_point_pixel": [cx, cy],
          "contact_normal": contact_normal.tolist(),
          "contact_normal_analytic": geo["contact_normal_analytic"],
          "impulse_vector": impulse_vec.tolist(),
          "impulse_magnitude": float(np.linalg.norm(impulse_vec)),
          "peak_normal_force": float(max(e["force"] for e in pair_events)),
          "num_contact_events": len(pair_events),
          "restitution_measured": restitution_measured,
          "pre": {
              "frame": PRE_FRAME,
              "velocity_A": pre["a"].tolist(), "velocity_B": pre["b"].tolist(),
              "momentum_A": (m["a"] * pre["a"]).tolist(),
              "momentum_B": (m["b"] * pre["b"]).tolist(),
              "total_momentum": p_pre.tolist(),
              "kinetic_energy": ke_pre,
          },
          "post": {
              "frame": POST_FRAME,
              "velocity_A": post["a"].tolist(), "velocity_B": post["b"].tolist(),
              "momentum_A": (m["a"] * post["a"]).tolist(),
              "momentum_B": (m["b"] * post["b"]).tolist(),
              "total_momentum": p_post.tolist(),
              "kinetic_energy": ke_post,
          },
          "momentum_abs_error": float(np.linalg.norm(p_post - p_pre)),
          # Recomputed from the animation rather than reused from the acceptance
          # check, because a violating scene edits the animation *after* that check
          # ran — reporting the checked value would label every violation as
          # perfectly conserved.
          "momentum_rel_error": float(
              np.linalg.norm(p_post - p_pre) / max(np.linalg.norm(p_pre), 1e-12)),
          "momentum_rel_error_simulated": measurements["momentum_rel_error"],
          "kinetic_energy_lost": float(ke_pre - ke_post),
          "kinetic_energy_ratio": float(ke_post / ke_pre) if ke_pre > 0 else None,
      },
      "per_frame": per_frame,
      "scene_constants": {
          "num_frames": NUM_FRAMES, "frame_rate": FRAME_RATE, "step_rate": STEP_RATE,
          "resolution": list(RESOLUTION), "gravity": [0.0, 0.0, 0.0],
          "center_height": CENTER_HEIGHT, "camera_distance": CAMERA_DISTANCE,
          "camera_position": [0.0, 0.0, CENTER_HEIGHT + CAMERA_DISTANCE],
          "camera_quaternion": [1.0, 0.0, 0.0, 0.0],
          "focal_length": FOCAL_LENGTH, "sensor_width": SENSOR_WIDTH,
          "visible_half_extents": list(half_extents()),
          "radius_range": list(RADIUS_RANGE),
          "restitution_range": list(RESTITUTION_RANGE),
          "speed_range": list(SPEED_RANGE), "materials": MATERIALS,
          "b_moving_prob": B_MOVING_PROB,
          "contact_jitter_radius": CONTACT_JITTER_RADIUS,
          "control_stratum_frac": CONTROL_STRATUM_FRAC,
          "control_momentum_range": list(CONTROL_MOMENTUM_RANGE),
          "pre_frame": PRE_FRAME, "post_frame": POST_FRAME,
          "samples_per_pixel": SAMPLES_PER_PIXEL,
          "render_device": gpu_desc,
      },
  }


def segmentation_stats(seg):
  """Per-frame pixel count and centroid for each sphere, straight from the mask."""
  stats = {}
  for tag, seg_id in (("a", SEG_ID["A"]), ("b", SEG_ID["B"])):
    counts, centroids = [], []
    for frame in range(seg.shape[0]):
      ys, xs = np.nonzero(seg[frame, :, :, 0] == seg_id)
      counts.append(int(xs.size))
      centroids.append([float(xs.mean()), float(ys.mean())] if xs.size else None)
    stats[tag] = {"count": counts, "centroid": centroids}
  return stats


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------

def write_outputs(out_dir: Path, rgba, seg):
  import imageio.v3 as iio

  frames_dir = out_dir / "frames"
  seg_dir = out_dir / "segmentation"
  frames_dir.mkdir(parents=True, exist_ok=True)
  seg_dir.mkdir(parents=True, exist_ok=True)

  rgb = rgba[..., :3]
  for i in range(rgb.shape[0]):
    iio.imwrite(frames_dir / f"frame_{i:04d}.png", rgb[i])
    iio.imwrite(seg_dir / f"segmentation_{i:04d}.png", seg[i, :, :, 0].astype(np.uint8))

  write_mp4(out_dir / "video.mp4", rgb, frames_dir)


def write_mp4(path: Path, rgb, frames_dir: Path):
  """h264 mp4 at FRAME_RATE. Prefers imageio-ffmpeg, falls back to a system ffmpeg.

  kubruntu ships imageio but not necessarily imageio-ffmpeg, so don't assume the
  plugin is there.
  """
  import imageio.v3 as iio
  try:
    iio.imwrite(path, rgb, fps=FRAME_RATE, codec="libx264", pixelformat="yuv420p",
                output_params=["-crf", "17"], plugin="FFMPEG")
    return
  except Exception as exc:  # pylint: disable=broad-except
    logger.warning("imageio FFMPEG plugin unavailable (%s); falling back to system ffmpeg", exc)

  ffmpeg = shutil.which("ffmpeg")
  if ffmpeg is None:
    raise RuntimeError(
        "cannot write mp4: no imageio-ffmpeg plugin and no ffmpeg on PATH. "
        "Inside the container run `pip3 install imageio-ffmpeg` or `apt-get install ffmpeg`.")
  import subprocess
  subprocess.run(
      [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(FRAME_RATE),
       "-i", str(frames_dir / "frame_%04d.png"), "-c:v", "libx264", "-crf", "17",
       "-pix_fmt", "yuv420p", str(path)],
      check=True)


# --------------------------------------------------------------------------------------

def _measured_restitution(animation, spheres, geo):
  n = np.asarray(geo["contact_normal_analytic"], dtype=np.float64)
  n /= np.linalg.norm(n)
  pre = {t: np.asarray(animation[spheres[t]]["velocity"][PRE_FRAME]) for t in ("a", "b")}
  post = {t: np.asarray(animation[spheres[t]]["velocity"][POST_FRAME]) for t in ("a", "b")}
  rel_pre = float(np.dot(pre["a"] - pre["b"], n))
  rel_post = float(np.dot(post["a"] - post["b"], n))
  return float(abs(rel_post) / abs(rel_pre))


def annotate_edit(labels, edit_record, e_sim, arm, seed_group, honest_exit, edited_exit, violation_mode):
  """The register's extra fields. ``restitution`` is made *realised* (the trap fixed at the
  source) and the simulator's value kept as ``restitution_simulated``."""
  realised = labels["collision"]["restitution_measured"]
  labels["restitution_simulated"] = float(e_sim)
  labels["restitution_realised"] = float(realised) if realised is not None else float(e_sim)
  labels["restitution"] = labels["restitution_realised"]
  labels["seed_group"] = int(seed_group)
  labels["edit"] = edit_record
  labels["violation_mode"] = violation_mode if labels["violates_momentum"] else None
  if arm is not None:
    labels["arm"] = arm
  elif labels["violates_momentum"]:
    labels["arm"] = "violating"
  elif edit_record is not None and edit_record.get("kind") == "restitution":
    labels["arm"] = f"e{edit_record['e_new']:.4g}"
  else:
    labels["arm"] = "honest"
  labels["exit_frame"] = {"honest_twin": honest_exit, "this_clip": edited_exit}
  if edit_record is not None and edit_record.get("kind") == "restitution":
    # Verify the edit landed where it was asked to, to round-off.
    labels["edit"]["restitution_realised_error"] = float(abs(labels["restitution_realised"] - edit_record["e_new"]))


def generate(seed: int, out_dir: Path, scene_id: str, render: bool = True,
             violate: bool = False, restitution: float | None = None,
             violation_mode: str = "drift", edit_restitution: float | None = None,
             arm: str | None = None, seed_group: int | None = None) -> dict:
  import kubric as kb
  import pybullet as pb

  gpu_desc = "n/a (physics only)"
  scratch_dir = tempfile.mkdtemp(prefix="kubric_scratch_")
  rejections = []
  accepted = None
  try:
    # --- Pass 1: search for a parameter set that passes every check, physics only.
    for attempt in range(MAX_ATTEMPTS):
      rng = np.random.default_rng([seed, attempt])
      p, reason = sample_params(rng, violate=violate)
      if p is None:
        rejections.append({"attempt": attempt, "reason": reason})
        continue
      if restitution is not None:
        p["restitution_sampled"] = p["restitution"]
        p["restitution"] = float(restitution)
      geo = solve_geometry(p)
      masses = derive_masses(p)

      reason = check_frame0(p, geo)
      if reason:
        rejections.append({"attempt": attempt, "reason": reason})
        continue

      (_, spheres, floor, _, simulator, _,
       animation, collisions) = build_and_simulate(p, geo, masses, scratch_dir, render=False)
      pair_events, floor_events = analyse_collisions(collisions, spheres, floor)
      reason, measurements = check_simulation(p, geo, masses, animation, spheres,
                                              pair_events, floor_events)
      pb.disconnect(physicsClientId=simulator.physics_client)

      if reason:
        rejections.append({"attempt": attempt, "reason": reason, **measurements})
        continue

      accepted = (attempt, p, geo, masses, measurements)
      break

    if accepted is None:
      raise RuntimeError(f"no valid scene after {MAX_ATTEMPTS} attempts: {rejections}")

    attempt, p, geo, masses, _ = accepted
    logger.info("accepted on attempt %d after %d rejections", attempt, len(rejections))

    # --- Pass 2: rebuild the accepted scene, this time with a renderer attached.
    (scene, spheres, floor, camera, _, renderer,
     animation, collisions) = build_and_simulate(p, geo, masses, scratch_dir, render=render)
    pair_events, floor_events = analyse_collisions(collisions, spheres, floor)
    reason, measurements = check_simulation(p, geo, masses, animation, spheres,
                                            pair_events, floor_events)
    if reason:
      raise RuntimeError(f"replay of accepted scene failed the check: {reason} {measurements}")

    # Break conservation only after the honest simulation has been validated, so a
    # violating scene is a *correct* collision with an edited aftermath rather than
    # a scene that failed physics for some unrelated reason. The same holds for a
    # restitution arm. Both are edits of the honest aftermath; everything before
    # contact is shared bit for bit across every clip of this seed.
    honest_exit = exit_frames(animation, spheres)
    e_sim = _measured_restitution(animation, spheres, geo)
    edit_record = None
    if p["violates_momentum"]:
      _, edit_record = apply_violation(p, geo, masses, spheres, animation,
                                       measurements["contact_frame"], mode=violation_mode, seed=seed)
    elif edit_restitution is not None:
      _, edit_record = apply_restitution_edit(p, geo, masses, spheres, animation,
                                              measurements["contact_frame"], e_sim, edit_restitution)
    edited_exit = exit_frames(animation, spheres)
    # The ladder's in-frame rule is relative: no arm may leave the frame earlier
    # than its honest twin (§58). The honest clip itself is never dropped here.
    if edit_record is not None and any(edited_exit[t] < honest_exit[t] for t in ("A", "B")):
      out_dir.mkdir(parents=True, exist_ok=True)
      drop = {"scene_id": scene_id, "seed": seed, "reason": "exits_frame_before_honest_twin",
              "exit_frame_honest": honest_exit, "exit_frame_edited": edited_exit, "edit": edit_record}
      (out_dir / "_DROPPED").write_text(json.dumps(drop, indent=1, default=str))
      logger.warning("scene %s dropped: %s", scene_id, drop["reason"])
      return {"dropped": drop}

    out_dir.mkdir(parents=True, exist_ok=True)
    seg_stats = None
    if render:
      gpu_desc = configure_gpu(renderer)
      logger.info("render device: %s", gpu_desc)
      frames = renderer.render(return_layers=("rgba", "segmentation"))
      # NOTE: do not run kubric's adjust_segmentation_idxs here. Its cryptomatte decoder
      # already substitutes each asset's segmentation_id, so the returned array holds
      # 0/1/2 directly. Adjusting again re-maps those values as if they were asset
      # indices (1 -> camera, 2 -> light, both "not foreground" -> 0) and silently zeroes
      # every mask.
      seg_stats = segmentation_stats(frames["segmentation"])
      write_outputs(out_dir, frames["rgba"], frames["segmentation"])

    labels = build_labels(p, geo, masses, scene, spheres, camera, animation,
                          pair_events, measurements, seed, scene_id, seg_stats, gpu_desc)
    labels["sampling_attempts"] = {"accepted_attempt": attempt, "rejections": rejections}
    labels["kubric_version"] = kb.__version__
    annotate_edit(labels, edit_record, e_sim, arm, seed_group if seed_group is not None else seed,
                  honest_exit, edited_exit, violation_mode)

    with open(out_dir / "labels.json", "w") as f:
      json.dump(labels, f, indent=1)
    (out_dir / "_COMPLETE").write_text(f"{scene_id}\n")
    return labels
  finally:
    shutil.rmtree(scratch_dir, ignore_errors=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--output-dir", type=str, required=True)
  parser.add_argument("--scene-id", type=str, default=None)
  parser.add_argument("--no-render", action="store_true",
                      help="run physics and write labels only (fast; for debugging)")
  parser.add_argument("--violate", action="store_true",
                      help="break momentum conservation after contact, for the "
                           "implausible class of a plausibility probe")
  parser.add_argument("--violation-mode", choices=("drift", "split"), default="drift",
                      help="drift: legacy random-direction common drift (probe_set); "
                           "split: keep a+b, move the split (conservation_set_v1)")
  parser.add_argument("--restitution", type=float, default=None,
                      help="pin the simulated pair restitution instead of sampling it")
  parser.add_argument("--edit-restitution", type=float, default=None,
                      help="render the legal (or e>1 impossible) outcome at this restitution "
                           "by rescaling the honest deflections; a ladder arm")
  parser.add_argument("--arm", type=str, default=None, help="arm tag written to labels (default derived)")
  parser.add_argument("--seed-group", type=int, default=None,
                      help="the scene this clip is a twin/arm of (default: the seed)")
  parser.add_argument("--logging-level", type=str, default="INFO")
  args = parser.parse_args()

  logging.basicConfig(level=args.logging_level,
                      format="%(asctime)s %(levelname)s %(name)s: %(message)s")

  out_dir = Path(args.output_dir)
  scene_id = args.scene_id or out_dir.name
  try:
    if args.violate and args.edit_restitution is not None:
      raise SystemExit("--violate and --edit-restitution are mutually exclusive")
    labels = generate(args.seed, out_dir, scene_id, render=not args.no_render,
                      violate=args.violate, restitution=args.restitution,
                      violation_mode=args.violation_mode, edit_restitution=args.edit_restitution,
                      arm=args.arm, seed_group=args.seed_group)
  except Exception:  # pylint: disable=broad-except
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_FAILED").write_text(traceback.format_exc())
    logger.error("scene %s failed:\n%s", scene_id, traceback.format_exc())
    return 1

  if "dropped" in labels:
    return 0
  collision = labels["collision"]
  logger.info("scene %s: contact frame %.3f (target %d, off by %+.3f), "
              "momentum rel error %.3e, KE ratio %.4f",
              scene_id, collision["actual_frame"], TARGET_CONTACT_FRAME,
              collision["discrepancy_frames"], collision["momentum_rel_error"],
              collision["kinetic_energy_ratio"])
  return 0


if __name__ == "__main__":
  sys.exit(main())
