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

# ---- dataset v2: making mass a genuine latent variable -------------------------
#
# In the v1 sampler mass is not latent. Density takes two values rendered as
# unmistakable materials and radius is directly visible, so
# m = rho * (4/3) * pi * r^3 is fully determined by appearance — measured on the
# 150-scene set, mass_A decodes at R2 0.934 and radius_A at 0.975. That makes the
# interesting question unfalsifiable: a probe can reach momentum by reading size
# and colour and never touch the collision at all.
#
# The v2 arms separate the Blender material from the Bullet density, which were
# always independent parameters:
#
#   latent_mass    both spheres render identically whatever their densities, so
#                  the mass ratio is recoverable ONLY from the collision, via
#                  m_A * dv_A = -m_B * dv_B. A probe that reads it has used
#                  conservation; there is no other route to it.
#   visible_mass   albedo ramps continuously with density, so the ratio is
#                  readable off a single frame. The positive control: same
#                  physics, same geometry, mass merely visible.
#
# The arms differ in exactly one thing — whether density is in the pixels — so
# the contrast isolates it. Radii are equal WITHIN a scene in both arms, which is
# what makes the mass ratio purely a density ratio and removes size as a cue.
# Radius still varies across scenes, so it stays usable as an appearance control.
ARMS = ("legacy", "latent_mass", "visible_mass")

V2_DENSITY_RANGE = (0.2, 20.0)     # 100x span: the widest ratio this can realise
V2_MAX_MASS_RATIO = 100.0          # with equal radii, mass ratio IS density ratio

# Albedo ramp for the visible_mass arm, and the single albedo latent_mass gives
# both spheres (the ramp's midpoint, so the arms sit as close in mean appearance
# as a visible cue permits).
#
# The ceiling matters beyond aesthetics: video.py separates spheres from the flat
# grey background with one luma threshold (SPHERE_LUMA_MAX 0.58 against a
# background at 0.659), so a sphere rendering brighter than that is invisible to
# the tracker. Rendered luma is not a simple function of albedo here — v1's metal
# reads 0.48 from albedo 0.72 while rubber reads 0.46 from a mean albedo of 0.33
# — so this range was *checked by rendering the extremes*, not predicted.
#
# The first attempt at (0.08, 0.60) failed that check: albedo 0.60 rendered at
# luma 0.673, brighter than the background itself, and detect_spheres lost the
# sphere and locked onto its shadow (reporting radii of 22.5 and 13.8 px for two
# spheres equal by construction). Measured on this surface, rendered luma
# ~= 0.238 + 0.726 * albedo.
#
# Re-measured at the ceiling below: albedo 0.36 -> luma mean 0.542, and 0.08 ->
# 0.296, with both spheres detected. The 4.5x albedo span is an unmistakable cue,
# and being monotone in density it is what a linear probe should find easiest.
# One residue worth knowing: at 0.36 the specular p95 reaches 0.647, so the
# tracker erodes the bright sphere's blob and stops over-reading its radius the
# way it does for a dark one. Probing never touches the tracker, but it means
# tracker radius carries a small density-dependent bias in the visible_mass arm.
V2_ALBEDO_RANGE = (0.08, 0.36)
V2_NEUTRAL_ALBEDO = 0.22
V2_SURFACE = {"metallic": 0.0, "roughness": 0.6, "specular": 0.4, "ior": 1.45}

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
VIOLATION_SCALE_RANGE = (0.35, 0.75)   # |dp| / |p_pre| injected, either sign

# Cap on the impulse asymmetry `delta` in apply_violation. At delta = 1 ball A's
# velocity change vanishes; beyond it the change reverses, which turns the
# antiparallel pair parallel and reintroduces exactly the geometric giveaway this
# mechanism exists to remove. Scenes needing more than this to hit their target
# |dp| simply realise a smaller violation, which labels.json records honestly.
MAX_VIOLATION_ASYMMETRY = 0.9

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


def albedo_for_density(density: float) -> float:
  """Albedo encoding density for the visible_mass arm, log-linear in density.

  Log-linear rather than linear because density itself is sampled log-uniformly:
  a linear ramp would put almost every scene in the dark end of the range and
  make the cue useless over most of the dataset.
  """
  lo, hi = V2_DENSITY_RANGE
  frac = (np.log(density) - np.log(lo)) / (np.log(hi) - np.log(lo))
  a_lo, a_hi = V2_ALBEDO_RANGE
  return float(a_lo + frac * (a_hi - a_lo))


def appearance_for(arm: str, density: float) -> dict:
  """Resolved Blender surface for one sphere under one arm.

  Appearance is resolved here, at sampling time, rather than looked up from
  MATERIALS at build time. That is the whole point of v2: the renderer must be
  able to draw two spheres identically while Bullet gives them different masses,
  which is impossible while the material *is* the density record.
  """
  if arm == "latent_mass":
    albedo = V2_NEUTRAL_ALBEDO
  elif arm == "visible_mass":
    albedo = albedo_for_density(density)
  else:
    raise ValueError(f"arm {arm!r} does not use a derived appearance")
  return {"color": (albedo, albedo, albedo), **V2_SURFACE}


def _sample_bodies(rng: np.random.Generator, arm: str,
                   material_pair: str = "random") -> dict:
  """Radii, densities and resolved appearances for the two spheres, per arm.

  Split out of :func:`sample_params` so the legacy arm's RNG call order survives
  verbatim. Scene ``i`` is a pure function of seed ``seed_offset + i`` and
  ``probe_set/`` is regenerated from seeds rather than backed up, so one extra
  draw on this path silently changes all 150 v1 scenes.
  """
  if arm == "legacy":
    mat_a = str(rng.choice(list(MATERIALS)))
    mat_b = str(rng.choice(list(MATERIALS)))
    # The two materials are drawn INDEPENDENTLY, so half the scenes get a matched
    # pair and their mass ratio is carried by radius alone. That is fine when the
    # question is "is mass decodable" -- it is not fine when the question is
    # "does the readout compare visible mass against the deflection split", where
    # a matched pair means the only mass cue is r^3 on two similar spheres.
    # `material_pair` forces the composition. Both draws still happen so the RNG
    # stream, and therefore every other parameter, is untouched.
    if material_pair == "mixed":
      mat_b = next(m for m in MATERIALS if m != mat_a)
    elif material_pair == "same":
      mat_b = mat_a
    r_a = float(rng.uniform(*RADIUS_RANGE))
    r_b = float(rng.uniform(*RADIUS_RANGE))
    rho_a, rho_b = MATERIALS[mat_a]["density"], MATERIALS[mat_b]["density"]
    keys = ("color", "metallic", "roughness", "specular", "ior")
    app_a = {k: MATERIALS[mat_a][k] for k in keys}
    app_b = {k: MATERIALS[mat_b][k] for k in keys}
  else:
    # Sample the RATIO first and let the absolute scale absorb what is left --
    # the same discipline the momentum draw below uses, and for the same reason:
    # the ratio is the quantity under test, so it is the one that has to come out
    # wide, log-uniform and appearance-independent. Drawing the two densities
    # independently would instead make the ratio triangular in log and pile most
    # scenes near 1:1, where the collision barely discriminates between them.
    log_max = np.log(V2_MAX_MASS_RATIO)
    log_ratio = float(rng.uniform(-log_max, log_max))
    half = 0.5 * abs(log_ratio)
    lo_g = np.log(V2_DENSITY_RANGE[0]) + half
    hi_g = np.log(V2_DENSITY_RANGE[1]) - half
    # The feasible band for the geometric mean collapses to a point at the
    # extreme ratio, by construction: V2_DENSITY_RANGE spans exactly
    # V2_MAX_MASS_RATIO, so the endpoints are reachable and nothing is rejected.
    log_rho_g = float(rng.uniform(lo_g, hi_g)) if hi_g > lo_g else 0.5 * (lo_g + hi_g)
    rho_a = float(np.exp(log_rho_g + 0.5 * log_ratio))
    rho_b = float(np.exp(log_rho_g - 0.5 * log_ratio))
    # Equal within the scene. This is what makes the mass ratio a pure density
    # ratio, so no size cue can shortcut the collision.
    r_a = r_b = float(rng.uniform(*RADIUS_RANGE))
    mat_a = mat_b = "neutral" if arm == "latent_mass" else "graded"
    app_a, app_b = appearance_for(arm, rho_a), appearance_for(arm, rho_b)

  return {
      "radius_a": r_a, "radius_b": r_b,
      "material_a": mat_a, "material_b": mat_b,
      "density_a": float(rho_a), "density_b": float(rho_b),
      "appearance_a": app_a, "appearance_b": app_b,
  }


def _pin(drawn: float, override: float | None) -> float:
  """Return the override if there is one, but only after the draw has happened."""
  return float(drawn) if override is None else float(override)


def sample_params(rng: np.random.Generator, violate: bool = False,
                  arm: str = "legacy",
                  restitution: float | None = None,
                  material_pair: str = "random") -> tuple[dict | None, str]:
  """Draw one candidate parameter set.

  Momentum is sampled *before* speed and speed is derived as ``p / m``. Drawing
  speed directly would make momentum a deterministic increasing function of mass
  whenever mass has the wider range, which is exactly how the pilot ended up with
  momentum and mass 0.97-correlated. Sampling the target quantity first and letting
  the free factor absorb it is what puts them on separate axes.

  ``arm`` selects the appearance/density coupling; see :data:`ARMS`. The two v2
  arms widen the mass range from 32x to 100x, which narrows the mass band that
  can realise the matched-momentum stratum's fixed momentum band at a legal
  speed. Those draws are rejected before any physics runs, so the cost is
  negligible and the effect is benign: the stratum simply concentrates on the
  masses that band was always about.

  Returns ``(params, reason)``; ``params`` is None when the draw is infeasible, and
  the reason is recorded so the rejection log says which constraint bit.
  """
  if arm not in ARMS:
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")

  bodies = _sample_bodies(rng, arm, material_pair)
  m_a = sphere_mass(bodies["radius_a"], bodies["density_a"])
  m_b = sphere_mass(bodies["radius_b"], bodies["density_b"])

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
      **bodies,
      "arm": arm,
      "speed_a": float(speed_a),
      "speed_b": float(speed_b),
      "direction_a_rad": float(rng.uniform(0, 2 * np.pi)),
      "direction_b_rad": float(rng.uniform(0, 2 * np.pi)) if b_moves else 0.0,
      "b_moving": b_moves,
      "target_momentum_a": float(target_p),
      "stratum": "matched_momentum" if control else "wide",
      "impact_parameter_frac": float(rng.uniform(0, IMPACT_PARAM_MAX_FRAC)),
      "impact_parameter_sign": int(rng.choice([-1, 1])),
      # `restitution` pins e instead of sampling it. The 2x2 elastic/inelastic
      # design needs e to be a controlled FACTOR, not a nuisance draw, and the
      # e-preserving violation (`apply_violation_ratio`) leaves it untouched, so
      # the twin of a pinned scene carries the same e exactly. The draw happens
      # either way and is then discarded: skipping it would shift every later
      # draw in the stream, so the SAME seed at two different e values would stop
      # sharing its geometry -- which is the whole point of pinning it.
      "restitution": _pin(float(rng.uniform(*RESTITUTION_RANGE)), restitution),
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


def apply_violation(p, geo, masses, spheres, animation, contact_frame):
  """Overwrite post-contact motion so momentum is *not* conserved, while keeping
  the two velocity changes exactly ANTIPARALLEL.

  Bullet cannot produce this: it conserves momentum by construction, which is
  precisely why every scene in the original pilot had an identical answer to "is
  momentum conserved" and the question was unprobeable. So the collision is
  simulated honestly and then the outgoing velocities are edited and the spheres
  re-keyframed to fly along the edited ones.

  Antiparallelism is the whole point of this version, and the previous one got it
  wrong. That version added the *same* velocity kick to both balls. It broke the
  momentum sum, but it also broke the geometry: in a frictionless collision the
  impulse is along the contact normal, so ``dv_A`` and ``dv_B`` are exactly
  opposed, and adding a common vector tilts them. Measured on the 300-scene set,
  ``cos(dv_A, dv_B)`` ran -1.000000 for every conserving scene against -0.997 to
  +0.994 for violating ones — so thresholding that single number separated the
  classes at **balanced accuracy 1.000**. A probe could "detect violations" by
  pure geometry, with no notion of mass or momentum anywhere. That is not the
  question we are asking.

  Here the two impulses stay along the same normal and only their *magnitudes*
  are made unequal::

      dv_A' = (1 - delta) * dv_A          dv_B' = (1 + delta) * dv_B

  Both remain scalar multiples of the original antiparallel pair, so
  ``cos(dv_A', dv_B') = -1`` exactly, while

      m_A dv_A' + m_B dv_B' = (delta_beta - delta_alpha) * J = 2 * delta * J

  is non-zero. The perturbation is split symmetrically so neither trajectory is
  individually odd — one ball rebounds slightly too weakly, the other slightly
  too strongly.

  The consequence worth understanding: with this mechanism a violation is
  **undetectable from kinematics alone**. Any antiparallel pair of deflections is
  consistent with momentum conservation under *some* mass ratio, namely
  ``m_A/m_B = |dv_B|/|dv_A|``. So the class becomes identifiable only to an
  observer that knows the masses independently of the collision. That is exactly
  the property the old mechanism lacked, and it is what makes "does the model use
  conservation" a real question rather than a geometry check.

  Returns the edited post-contact velocities keyed by tag.
  """
  # Start the edited motion at the first frame rendered after contact, anchored on
  # the last frame before it, so the trajectory deviates exactly at the collision
  # rather than a frame later — the aftermath is wrong, the approach is not.
  first_moving = int(np.ceil(contact_frame))

  vel_pre = {t: np.array(animation[spheres[t]]["velocity"][PRE_FRAME], dtype=np.float64)
             for t in ("a", "b")}
  vel_post = {t: np.array(animation[spheres[t]]["velocity"][POST_FRAME], dtype=np.float64)
              for t in ("a", "b")}
  dv = {t: vel_post[t] - vel_pre[t] for t in ("a", "b")}

  p_pre = masses["mass_a"] * vel_pre["a"] + masses["mass_b"] * vel_pre["b"]
  impulse = masses["mass_b"] * float(np.linalg.norm(dv["b"]))   # |J|, same for both

  # Choose the asymmetry that realises the requested |dp| / |p_pre|, then clamp.
  # delta must stay below 1 or ball A's deflection reverses sign, which is both
  # visually absurd and would flip the antiparallel pair to parallel.
  target = abs(p["violation_scale"]) * float(np.linalg.norm(p_pre))
  delta = target / (2.0 * impulse) if impulse > 1e-12 else 0.0
  delta = float(np.clip(delta, 0.0, MAX_VIOLATION_ASYMMETRY))
  delta *= float(np.sign(p["violation_scale"]) or 1.0)

  edited = {
      "a": vel_pre["a"] + (1.0 - delta) * dv["a"],
      "b": vel_pre["b"] + (1.0 + delta) * dv["b"],
  }

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

  return edited


def apply_violation_ratio(p, geo, masses, spheres, animation, contact_frame,
                          implied_ratio: float):
  """Break momentum conservation while leaving EVERY mass-free observable alone.

  The two deflections in a frictionless collision are antiparallel and their
  magnitudes sum to a quantity fixed by the restitution law::

      a = |dv_A|,  b = |dv_B|,   a + b = (1 + e) * u_n        <- invariant
      b / a = m_A / m_B                                       <- conservation

  So the *sum* is set by the restitution and the *split* is set by the masses.
  This edit keeps the sum and moves the split, which is the only degree of
  freedom that conservation constrains:

      a' = S / (1 + rho'),   b' = S * rho' / (1 + rho'),   S = a + b

  where ``implied_ratio`` is rho', the mass ratio the aftermath now *implies*.
  Consequences, all exact rather than approximate:

  * ``cos(dv_A', dv_B') = -1`` -- both deflections are positive rescalings, so the
    antiparallel geometry that §38a's `cos` leak exploited is untouched.
  * realised restitution is unchanged, because it depends only on ``a + b`` --
    this is the leak §38b opened, and it is closed by construction rather than
    by clamping.
  * the relative velocity is bit-identical to the honest twin, so separation
    rate, closest approach and contact geometry cannot distinguish the classes.
  * the edit is a COMMON drift ``Delta-p / M`` added to both balls, so no
    per-object trajectory looks individually odd.

  **Draw ``implied_ratio`` from the mass ratios of OTHER scenes** (a derangement
  across the set). A permutation preserves the multiset exactly, so the
  distribution of the implied ratio is identical in the two classes and no
  kinematics-only readout can separate them -- measured at AUC 0.500 per scalar
  and 0.498 for a joint 18-feature classifier under grouped CV. The violating
  clip is then a physically perfect collision *for the wrong pair of masses*,
  and only a readout that converts appearance into mass can tell.

  Returns the edited post-contact velocities keyed by tag.
  """
  first_moving = int(np.ceil(contact_frame))

  vel_pre = {t: np.array(animation[spheres[t]]["velocity"][PRE_FRAME], dtype=np.float64)
             for t in ("a", "b")}
  vel_post = {t: np.array(animation[spheres[t]]["velocity"][POST_FRAME], dtype=np.float64)
              for t in ("a", "b")}
  dv = {t: vel_post[t] - vel_pre[t] for t in ("a", "b")}
  a = float(np.linalg.norm(dv["a"]))
  b = float(np.linalg.norm(dv["b"]))
  total = a + b
  if a < 1e-12 or b < 1e-12:
    raise RuntimeError(f"degenerate deflections a={a:.3e} b={b:.3e}; cannot rescale")

  rho = float(implied_ratio)
  a_new = total / (1.0 + rho)
  b_new = total * rho / (1.0 + rho)
  edited = {
      "a": vel_pre["a"] + (a_new / a) * dv["a"],
      "b": vel_pre["b"] + (b_new / b) * dv["b"],
  }

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

  return edited


def apply_energy_edit(p, geo, masses, spheres, animation, contact_frame,
                      e_target: float):
  """Change the collision's ENERGY while leaving momentum EXACTLY conserved.

  THE MISSING CELL. `conservation_set_v1` crosses momentum against energy and one
  box comes out empty: of its 300 clips, **zero** conserve momentum while creating
  kinetic energy. Every one of its 57 energy-creating clips also breaks momentum,
  because that is how they were made -- the violation edit targets momentum and
  disturbs energy only as a side effect. So surprise on those clips cannot be
  attributed to the invented energy rather than to the missing momentum.

  And the honest half does not fill the gap either: losing kinetic energy is LEGAL
  physics. An inelastic collision converts it to heat and deformation and violates
  nothing. Testing whether a model holds energy conservation as a LAW needs clips
  that are impossible, which means energy CREATED -- `e > 1`, which no physics
  engine will produce and which therefore has to be edited into existence.

  THE EDIT, and it is the exact complement of `apply_violation_ratio`. Write the
  two deflections as magnitudes along the contact normal::

      a + b = (1 + e) * |u_n|     <- the SUM is set by the restitution
      b / a = m_A / m_B           <- the SPLIT is set by conservation

  `apply_violation_ratio` holds the sum and moves the split: momentum breaks, the
  realised restitution survives. This holds the SPLIT and moves the SUM, which is
  the orthogonal degree of freedom:

      dv_A' = lambda * dv_A,   dv_B' = lambda * dv_B,   lambda = (1+e_target)/(1+e_sim)

  A COMMON rescaling, so `m_A dv_A' + m_B dv_B' = lambda * 0 = 0` and momentum is
  conserved to the simulator's own precision -- not approximately, identically. The
  sum becomes `(1 + e_target) |u_n|`, so the realised restitution becomes exactly
  `e_target`, and the energy moves by `dKE = (1/2) mu (e_target^2 - 1) u_n^2`, which
  is POSITIVE for `e_target > 1`: energy from nothing, with the momentum books
  balanced.

  WHY THIS MAKES A CONTROLLED PAIR. Render each seed at `e_target = k` and at
  `e_target = 1/k`. Both sit the same distance from elastic in log space; one
  separates the balls `k` times too fast and the other `k` times too slow, so both
  are equally unusual as kinematics. Exactly one of them is impossible. A model that
  is more surprised by `k` than by `1/k` holds an energy prior; one that is equally
  surprised by both is detecting novelty, not physics.

  Returns the edited post-contact velocities keyed by tag.
  """
  first_moving = int(np.ceil(contact_frame))

  vel_pre = {t: np.array(animation[spheres[t]]["velocity"][PRE_FRAME], dtype=np.float64)
             for t in ("a", "b")}
  vel_post = {t: np.array(animation[spheres[t]]["velocity"][POST_FRAME], dtype=np.float64)
              for t in ("a", "b")}
  dv = {t: vel_post[t] - vel_pre[t] for t in ("a", "b")}

  # THE NORMAL COMES FROM THE DEFLECTION, not from `geo`. The collision is
  # frictionless, so the impulse -- and therefore `dv_A` -- lies along the contact
  # normal by construction, exactly. `contact_normal_analytic` is a geometric
  # reconstruction that agrees only to the accuracy of the contact-frame estimate,
  # and using it put the realised energy change up to 3.6% off the closed form
  # `dKE = 1/2 mu (e^2 - 1) u_n^2` while the restitution still looked exact --
  # because the same wrong normal was on both sides of that check. Deriving the
  # normal from `dv_A` removes the free parameter and makes the two agree.
  na = float(np.linalg.norm(dv["a"]))
  if na < 1e-12:
    raise RuntimeError(f"degenerate deflection |dv_A|={na:.3e}; cannot recover the normal")
  n_hat = dv["a"] / na
  u_pre = float(np.dot(vel_pre["a"] - vel_pre["b"], n_hat))
  u_post = float(np.dot(vel_post["a"] - vel_post["b"], n_hat))
  if abs(u_pre) < 1e-9:
    raise RuntimeError(f"degenerate approach speed u_pre={u_pre:.3e}; cannot set e")
  e_sim = abs(u_post) / abs(u_pre)

  lam = (1.0 + float(e_target)) / (1.0 + e_sim)
  edited = {t: vel_pre[t] + lam * dv[t] for t in ("a", "b")}

  p["energy_edit"] = {
      "applied": True,
      "e_target": float(e_target),
      "e_simulated": float(e_sim),
      "lambda": float(lam),
      # sign convention is irrelevant to the ratio, but record the raw scalars so a
      # validator can recompute e without trusting this function
      "u_normal_pre": u_pre,
      "u_normal_post_simulated": u_post,
  }

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

  return edited

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
    # Resolved at sampling time, not looked up from MATERIALS: under the v2 arms
    # two spheres can share an appearance while carrying different densities,
    # which is impossible while the material *is* the density record.
    spec = p[f"appearance_{tag}"]
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

  # ---- how identifiable is the mass ratio in this scene? ----------------------
  #
  # Under the v2 latent_mass arm the ratio is recoverable only from the collision,
  # through m_A*dv_A = -m_B*dv_B. That inverts to a ratio of the two velocity
  # changes, so the estimate is only as good as the SMALLER of them — and the
  # smaller one belongs to the heavier ball, shrinking as 1/(1+kappa) in the mass
  # ratio. At 100:1 the heavy ball deflects by roughly 2% of the relative
  # velocity.
  #
  # That is a fact about the physics, not about the model: past some ratio the
  # aftermath stops carrying the answer at this resolution, and a probe would
  # fail for information-theoretic reasons that say nothing about the
  # representation. Recording it per scene means the probe results can be
  # stratified by identifiability, rather than that confound being discovered
  # after the fact — which is the same mistake the v1 momentum/mass confound was.
  px_per_world = RESOLUTION[0] / (2.0 * half_extents()[0])
  post_start = int(round(measurements["contact_frame"])) + 3
  post_span_frames = max(NUM_FRAMES - 1 - post_start, 0)
  dv = {tag: float(np.linalg.norm(post[tag] - pre[tag])) for tag in ("a", "b")}
  identifiability = {
      "delta_v_A": dv["a"],
      "delta_v_B": dv["b"],
      "delta_v_A_px_per_frame": dv["a"] * px_per_world / FRAME_RATE,
      "delta_v_B_px_per_frame": dv["b"] * px_per_world / FRAME_RATE,
      # The binding constraint: the ratio needs BOTH deflections, so the weaker
      # one sets the precision. Compare against the tracker's ~0.1 px centroid
      # noise floor — this is how far the ball departs from its undeflected path
      # over the window a velocity fit actually sees.
      "min_delta_displacement_px": (
          min(dv["a"], dv["b"]) * px_per_world * post_span_frames / FRAME_RATE),
      "post_window_frames": post_span_frames,
      "px_per_world": float(px_per_world),
  }

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
                "segmentation_id": SEG_ID["A"],
                "color_rgb": list(p["appearance_a"]["color"])},
          "B": {"radius": p["radius_b"], "material": p["material_b"],
                "density": p["density_b"], "volume": masses["volume_b"],
                "mass": masses["mass_b"], "friction": 0.0,
                "restitution_object": float(np.sqrt(p["restitution"])),
                "segmentation_id": SEG_ID["B"],
                "color_rgb": list(p["appearance_b"]["color"])},
      },
      "arm": p.get("arm", "legacy"),
      "mass_ratio": masses["mass_ratio"],
      "identifiability": identifiability,
      "restitution": p["restitution"],
      # Probe-facing scene labels, hoisted to the top level so a probe target is a
      # short path rather than a reach into sampled_parameters.
      "stratum": p["stratum"],
      "b_moving": p["b_moving"],
      "violates_momentum": p["violates_momentum"],
      "momentum_conserved": (not p["violates_momentum"]),
      # The energy axis, recorded separately from the momentum axis so the 2x2 can be
      # selected on without re-deriving anything. `energy_created` is the impossible
      # class: kinetic energy appears from nowhere while the momentum books balance.
      "energy_edit": p.get("energy_edit"),
      "restitution_realised": (float(p["energy_edit"]["e_target"])
                               if p.get("energy_edit") else p["restitution"]),
      "energy_created": bool(p.get("energy_edit")
                             and p["energy_edit"]["e_target"] > 1.0),
      "energy_conserved": bool(p.get("energy_edit") is None
                               or p["energy_edit"]["e_target"] == 1.0),
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
          "arm": p.get("arm", "legacy"),
          "density_range": list(V2_DENSITY_RANGE),
          "max_mass_ratio": V2_MAX_MASS_RATIO,
          "albedo_range": list(V2_ALBEDO_RANGE),
          "neutral_albedo": V2_NEUTRAL_ALBEDO,
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
    # -movflags +faststart puts the moov atom BEFORE mdat. Without it the index sits
    # at the end of the file, and any player that cannot seek — a browser handed the
    # clip as a data: URI, for one — refuses to play it at all. Costs nothing here.
    iio.imwrite(path, rgb, fps=FRAME_RATE, codec="libx264", pixelformat="yuv420p",
                output_params=["-crf", "17", "-movflags", "+faststart"], plugin="FFMPEG")
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
       "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
      check=True)


# --------------------------------------------------------------------------------------

def generate(seed: int, out_dir: Path, scene_id: str, render: bool = True,
             violate: bool = False, arm: str = "legacy",
             restitution: float | None = None,
             implied_ratio: float | None = None,
             energy_restitution: float | None = None,
             material_pair: str = "random") -> dict:
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
      p, reason = sample_params(rng, violate=violate, arm=arm,
                                restitution=restitution,
                                material_pair=material_pair)
      if p is None:
        rejections.append({"attempt": attempt, "reason": reason})
        continue
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
    # a scene that failed physics for some unrelated reason.
    if p["violates_momentum"]:
      if implied_ratio is not None:
        p["violation_implied_ratio"] = float(implied_ratio)
        apply_violation_ratio(p, geo, masses, spheres, animation,
                              measurements["contact_frame"], implied_ratio)
      else:
        apply_violation(p, geo, masses, spheres, animation,
                        measurements["contact_frame"])

    # The ENERGY edit is mutually exclusive with the momentum edits by design: the
    # whole point of this cell is that momentum stays conserved, so combining them
    # would reintroduce exactly the confound it exists to remove.
    if energy_restitution is not None:
      if p["violates_momentum"]:
        raise ValueError("--energy-restitution conserves momentum by construction; "
                         "combining it with --violate defeats its purpose")
      apply_energy_edit(p, geo, masses, spheres, animation,
                        measurements["contact_frame"], energy_restitution)

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
  parser.add_argument("--arm", type=str, default="legacy", choices=ARMS,
                      help="appearance/density coupling. 'legacy' reproduces the "
                           "v1 probe_set exactly. 'latent_mass' renders both "
                           "spheres identically so the mass ratio is recoverable "
                           "only from the collision; 'visible_mass' ramps albedo "
                           "with density so it is readable off a single frame")
  parser.add_argument("--material-pair", type=str, default="random",
                      choices=("random", "mixed", "same"),
                      help="'random' draws each sphere's material independently "
                           "(half the scenes end up matched). 'mixed' forces one "
                           "metal and one rubber, so the mass ratio always carries "
                           "a 4x density factor on top of r^3. 'same' is the "
                           "size-only control")
  parser.add_argument("--restitution", type=float, default=None,
                      help="pin the pair restitution instead of sampling it. "
                           "The elastic/inelastic factor of the 2x2 design")
  parser.add_argument("--implied-ratio", type=float, default=None,
                      help="use the e-preserving violation: rescale the two "
                           "deflections so the aftermath implies THIS mass ratio "
                           "(m_A/m_B) instead of the true one. Draw it from another "
                           "scene's ratio so the implied-ratio distribution matches "
                           "the honest class. Requires --violate")
  parser.add_argument("--energy-restitution", type=float, default=None,
                      help="rescale BOTH deflections by (1+e)/(1+e_sim) so the "
                           "realised restitution becomes this value while momentum "
                           "stays exactly conserved. e>1 creates kinetic energy from "
                           "nothing -- the impossible class no physics engine will "
                           "produce. Mutually exclusive with --violate")
  parser.add_argument("--logging-level", type=str, default="INFO")
  args = parser.parse_args()

  logging.basicConfig(level=args.logging_level,
                      format="%(asctime)s %(levelname)s %(name)s: %(message)s")

  out_dir = Path(args.output_dir)
  scene_id = args.scene_id or out_dir.name
  try:
    labels = generate(args.seed, out_dir, scene_id, render=not args.no_render,
                      violate=args.violate, arm=args.arm,
                      restitution=args.restitution,
                      implied_ratio=args.implied_ratio,
                      energy_restitution=args.energy_restitution,
                      material_pair=args.material_pair)
  except Exception:  # pylint: disable=broad-except
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_FAILED").write_text(traceback.format_exc())
    logger.error("scene %s failed:\n%s", scene_id, traceback.format_exc())
    return 1

  collision = labels["collision"]
  logger.info("scene %s: contact frame %.3f (target %d, off by %+.3f), "
              "momentum rel error %.3e, KE ratio %.4f",
              scene_id, collision["actual_frame"], TARGET_CONTACT_FRAME,
              collision["discrepancy_frames"], collision["momentum_rel_error"],
              collision["kinetic_energy_ratio"])
  return 0


if __name__ == "__main__":
  sys.exit(main())
