#!/usr/bin/env python3
"""Generate one two-ball sliding-friction scene for friction probing.

    python3 generate_friction_scene.py --seed 0 --output-dir out/scene_00000 --arm latent_friction

The friction counterpart to `generate_scene.py`, sharing its camera, resolution,
frame count, renderer configuration and output layout so that everything above
`momentum_dataset/` — capture, pooling, probing — runs against this set
unchanged. Read `generate_scene.py` first; only what differs is explained here.

Scene design
------------
Two balls slide in separate lanes across a floor whose FRICTION VARIES BY
REGION. Left of a jittered boundary the floor is frictionless, so each ball
coasts in a straight line at constant speed. Right of it sit two patches, one
per lane, each with its own coefficient. Each ball's start is solved so that it
crosses the boundary exactly on TARGET_ENTRY_FRAME, and decelerates from there.

Three consequences worth stating, because they are what the design buys:

* **The entry frame plays the role the collision frame plays in the momentum
  set.** Frames 0-20 are strictly pre-friction and 21-48 strictly post, so the
  same phase split (and the same `labels["collision"]["actual_frame*"]` keys the
  capture code reads) applies with no changes anywhere downstream.
* **Deceleration is mass-independent.** a = mu * g regardless of the ball's mass
  or radius, so size is a *free* appearance control here: unlike the momentum
  set, where m enters the target algebraically, a probe that reads radius has
  demonstrably not read friction.
* **Friction is a property of the SURFACE, not of the ball.** The balls are
  rendered identically to each other in both arms; what carries mu is the floor.
  So "which ball is on which patch" is a genuine object-to-surface binding
  question, which is the multi-object structure this project is about.

Arms (`--arm`), the same latent/visible contrast the momentum set's v2/v3 use:

  latent_friction   every floor tile renders with one identical albedo, so the
                    boundary and the patches are invisible and mu is recoverable
                    ONLY from the motion after entry.
  visible_friction  each patch's albedo ramps log-linearly with its mu, so the
                    coefficient is readable off a single frame, before the ball
                    has reached the patch. The positive control.

Honest limits, in the spirit of CLAUDE.md §12
---------------------------------------------
mu = a / g and `a` is plainly visible in the trajectory, so `mu_A` from a
post-entry pool is an IDENTITY in exactly the sense §12 warns about — the probe
is measuring the model's kinematics plus a division by a constant. That is not a
reason to avoid the dataset; it is a reason to read it the way §13 reads the mass
ratio, against a kinematic oracle. What is genuinely new here:

* a SECOND physical domain in which to ask §3's one positive question — does a
  probe beat the best linear readout of the true factors? — which tests whether
  that result was momentum-specific.
* mu from PRE-entry pools, where the answer is not in the kinematics at all:
  in `visible_friction` it must come from surface appearance, and in
  `latent_friction` it is unrecoverable by anyone and must read zero. That pair
  is a positive control and a predicted null on the same footing as §14's.
* per-object deceleration as a new causal target for §18-style patching.

No conservation-violation mechanism is included. Every candidate for one here is
either trivially visible from the kinematics (a ball that speeds up) or
unrecoverable by any observer (mass-dependent friction under latent mass), which
are the two failure modes §12/§14 already paid for. `advertised_mu` is provided
instead: in the visible arm it renders a patch whose albedo advertises a
coefficient the floor does not have, which is the one mismatch that needs
appearance and dynamics to be bound together before it can be detected.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_scene import (  # noqa: E402  — shared, so there is one copy of each trap
    CAMERA_DISTANCE,
    FOCAL_LENGTH,
    FRAME_RATE,
    NUM_FRAMES,
    RADIUS_RANGE,
    RESOLUTION,
    SENSOR_WIDTH,
    SEG_ID,
    SILHOUETTE_SAFETY,
    STEP_RATE,
    V2_NEUTRAL_ALBEDO,
    V2_SURFACE,
    configure_gpu,
    half_extents,
    segmentation_stats,
    write_outputs,
)

logger = logging.getLogger("generate_friction_scene")

# --------------------------------------------------------------------------------------
# Scene constants
# --------------------------------------------------------------------------------------

ARMS = ("latent_friction", "visible_friction")

# Gravity magnitude, world units / s^2, pointing -z. NOT 9.81: this world is not
# metric (the visible frame is 14.4 units across and a ball is ~0.5 across), and
# at 9.81 a physically ordinary mu of 0.3 would stop a ball in a third of a
# second — three frames, with nothing to see. At g = 2.0 the useful mu range
# lands the deceleration between 0.1 and 1.6 units/s^2, which is 0.6 to 10 px per
# frame per second at this camera. mu keeps its ordinary meaning (0.05 ice, 0.8
# rubber) and `a = mu * g` is exact; only the clock is slower.
GRAVITY = 2.0

# Log-uniform: the ratio between the two patches is the relational quantity under
# test, and a linear draw would pile most scenes near 1:1 where the two balls
# decelerate indistinguishably. 16x span.
MU_RANGE = (0.05, 0.80)

# Straight-line speed before entry. Floor as in the momentum set (a ball drifting
# under ~2 px/frame barely registers in the video); the ceiling is set by keeping
# the whole 6-second trajectory inside the frame — see check_frame0 for the
# arithmetic that makes every draw feasible without rejection.
SPEED_RANGE = (0.60, 1.50)

# Heading spread about +x, degrees. Zero would make v_y identically zero, and a
# 2-vector target with a constant component inflates every multi-output R2 with a
# component that is trivially predictable — the same reason the momentum targets
# drop z. Small enough that both balls stay in their own lanes for 6 seconds.
HEADING_MAX_DEG = 8.0

# Lane centres in y, and the jitter on them. The two patches meet at y = 0, so a
# ball must stay on its own side for the whole clip; the margins below are
# checked, not assumed.
LANE_Y = 2.60
LANE_Y_JITTER = 0.35

# Boundary position jitter. At 0 every scene would put the entry at the same
# pixel column and a probe could read the entry *time* off a constant, which is
# the same failure CONTACT_JITTER_RADIUS exists to prevent in the momentum set.
BOUNDARY_X_JITTER = 0.60

TARGET_ENTRY_FRAME = 20
ENTRY_TOLERANCE = 0.5           # frames, measured minus target
PRE_FRAME = 18                  # both far enough from entry that a +/-1 frame
POST_FRAME = 22                 # timing error cannot contaminate them

# Fraction of scenes where both patches share one coefficient, so the scene has a
# single "floor friction" rather than two. Keeps a scene-level target available
# (mu is then a property of the whole floor) without giving up the per-object
# design in the rest of the set. Recorded as `stratum`.
UNIFORM_FLOOR_FRAC = 0.30

# Floor appearance. The latent arm paints every tile FLOOR_ALBEDO, so the
# boundary and the two patches are invisible; the visible arm ramps a patch's
# albedo log-linearly with its mu.
#
# The ramp goes UP from the neutral floor, never down, and that is a tracker
# constraint rather than an aesthetic one: video.py separates balls from the
# background with a single luma threshold (SPHERE_LUMA_MAX 0.58 against a
# background at 0.659 for albedo 0.42), so a patch darker than that threshold
# would be detected as one enormous ball and the pixel tracker would be useless
# on this set. Ramping up keeps every floor tile brighter than every ball.
FLOOR_ALBEDO = 0.42
FLOOR_ALBEDO_RANGE = (0.42, 0.78)
FLOOR_SURFACE = {"metallic": 0.0, "roughness": 0.8, "specular": 0.2}

# Balls are identical to each other in both arms — mu lives in the floor, so
# nothing about a ball should hint at it. Radius still varies, as the appearance
# control that a friction readout has to beat.
BALL_ALBEDO = V2_NEUTRAL_ALBEDO

# Floor tiles extend well past the visible frame so no tile edge is ever in shot.
FLOOR_EXTENT_X = 9.0
FLOOR_EXTENT_Y = 7.0
FLOOR_THICKNESS = 0.5

FRAME_MARGIN = 0.10
MIN_LANE_CLEARANCE = 0.15       # ball surface to the y = 0 seam between patches

# Physics acceptance tolerances.
DECEL_REL_TOLERANCE = 0.05      # |a_measured - mu*g| / (mu*g)
COAST_REL_TOLERANCE = 0.01      # pre-entry speed must be constant to this
MAX_VERTICAL_SPEED = 0.02       # a ball that bounces at a tile seam fails here
MAX_ANGULAR_SPEED = 1e-3        # rotation is frozen; this proves it

# Rotation is frozen by giving each ball an enormous inertia tensor, so a sphere
# SLIDES instead of rolling. Without this the friction force spins the ball up
# until it rolls without slipping at 5/7 of its entry speed and then coasts
# forever, which makes mu control a transient duration rather than the
# deceleration and destroys the clean a = mu*g label. A frozen-rotation sphere is
# a puck with a round silhouette: it keeps video.py's circle tracker usable,
# which a cube would not.
FROZEN_INERTIA = 1e6

MAX_ATTEMPTS = 20
SAMPLES_PER_PIXEL = 64

CLIP_SECONDS = (NUM_FRAMES - 1) / FRAME_RATE
ENTRY_TIME = TARGET_ENTRY_FRAME / FRAME_RATE
POST_SECONDS = CLIP_SECONDS - ENTRY_TIME


# --------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------

def albedo_for_mu(mu: float) -> float:
  """Patch albedo encoding mu for the visible arm, log-linear in mu.

  Log-linear because mu is drawn log-uniformly: a linear ramp would put nearly
  every scene at one end of the range and make the cue useless across most of
  the dataset. Same reasoning as `albedo_for_density` in the momentum set.
  """
  lo, hi = MU_RANGE
  frac = (np.log(mu) - np.log(lo)) / (np.log(hi) - np.log(lo))
  a_lo, a_hi = FLOOR_ALBEDO_RANGE
  return float(a_lo + frac * (a_hi - a_lo))


def patch_appearance(arm: str, mu: float) -> dict:
  """Resolved Blender surface for one friction patch under one arm."""
  if arm == "latent_friction":
    albedo = FLOOR_ALBEDO
  elif arm == "visible_friction":
    albedo = albedo_for_mu(mu)
  else:
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
  return {"color": (albedo, albedo, albedo), **FLOOR_SURFACE}


def sample_params(rng: np.random.Generator, arm: str = "latent_friction",
                  mismatch: bool = False) -> tuple[dict | None, str]:
  """Draw one candidate parameter set.

  Every quantity here is drawn independently of every other, and the feasibility
  arithmetic in `check_frame0` is satisfied by construction across the whole
  product of the ranges — so unlike the momentum sampler this one never rejects,
  and mu therefore cannot pick up a correlation with speed or radius through
  rejection. That is worth more than it sounds: the momentum/mass confound that
  forced the v2 rebuild entered exactly that way.
  """
  if arm not in ARMS:
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")

  log_lo, log_hi = np.log(MU_RANGE[0]), np.log(MU_RANGE[1])
  mu_a = float(np.exp(rng.uniform(log_lo, log_hi)))
  uniform_floor = bool(rng.uniform() < UNIFORM_FLOOR_FRAC)
  mu_b = mu_a if uniform_floor else float(np.exp(rng.uniform(log_lo, log_hi)))

  # What the patch LOOKS like, which is the actual coefficient unless this scene
  # is deliberately mismatched. Only the visible arm renders the difference.
  advertised = {}
  for tag, mu in (("a", mu_a), ("b", mu_b)):
    if mismatch:
      other = float(np.exp(rng.uniform(log_lo, log_hi)))
      # Force a visible gap, otherwise a "mismatch" that lands next to the true
      # value is indistinguishable from an honest scene.
      while abs(np.log(other) - np.log(mu)) < 0.5 * (log_hi - log_lo):
        other = float(np.exp(rng.uniform(log_lo, log_hi)))
      advertised[tag] = other
    else:
      advertised[tag] = mu

  params = {
      "arm": arm,
      "mu_a": mu_a,
      "mu_b": mu_b,
      "advertised_mu_a": advertised["a"],
      "advertised_mu_b": advertised["b"],
      "appearance_patch_a": patch_appearance(arm, advertised["a"]),
      "appearance_patch_b": patch_appearance(arm, advertised["b"]),
      "appearance_approach": {"color": (FLOOR_ALBEDO,) * 3, **FLOOR_SURFACE},
      "mismatched_appearance": bool(mismatch),
      "stratum": "uniform_floor" if uniform_floor else "split_floor",
      "radius_a": float(rng.uniform(*RADIUS_RANGE)),
      "radius_b": float(rng.uniform(*RADIUS_RANGE)),
      "speed_a": float(rng.uniform(*SPEED_RANGE)),
      "speed_b": float(rng.uniform(*SPEED_RANGE)),
      "heading_a_rad": float(np.deg2rad(rng.uniform(-HEADING_MAX_DEG, HEADING_MAX_DEG))),
      "heading_b_rad": float(np.deg2rad(rng.uniform(-HEADING_MAX_DEG, HEADING_MAX_DEG))),
      "boundary_x": float(rng.uniform(-BOUNDARY_X_JITTER, BOUNDARY_X_JITTER)),
      "lane_y_a": float(LANE_Y + rng.uniform(-LANE_Y_JITTER, LANE_Y_JITTER)),
      "lane_y_b": float(-LANE_Y + rng.uniform(-LANE_Y_JITTER, LANE_Y_JITTER)),
  }
  return params, ""


# --------------------------------------------------------------------------------------
# Geometry: place each ball so it crosses the boundary on the target frame
# --------------------------------------------------------------------------------------

def solve_geometry(p: dict) -> dict:
  """Initial positions putting both boundary crossings on TARGET_ENTRY_FRAME.

  Exact rather than searched: the approach region is frictionless, so the motion
  before entry is a straight line at constant velocity and the crossing time is
  linear in the start position. A ball's contact with the floor is a single point
  directly under its centre, so "the ball crosses" and "its contact patch
  crosses" are the same instant — which is why this is a sphere with rotation
  frozen rather than a box, whose contact patch would straddle the boundary for
  up to two seconds and smear the entry across a third of the clip.
  """
  geo = {}
  for tag in ("a", "b"):
    speed, heading = p[f"speed_{tag}"], p[f"heading_{tag}_rad"]
    vel = np.array([speed * np.cos(heading), speed * np.sin(heading), 0.0])
    entry = np.array([p["boundary_x"], p[f"lane_y_{tag}"], p[f"radius_{tag}"]])
    geo[f"vel_{tag}"] = vel
    geo[f"entry_{tag}"] = entry
    geo[f"pos_{tag}_start"] = entry - vel * ENTRY_TIME

    # Where it ends up, under a = mu*g opposing the motion, capped at rest.
    decel = p[f"mu_{tag}"] * GRAVITY
    t_stop = speed / decel
    t = min(POST_SECONDS, t_stop)
    travel = speed * t - 0.5 * decel * t * t
    geo[f"decel_{tag}"] = float(decel)
    geo[f"stop_time_{tag}"] = float(t_stop)
    geo[f"travel_{tag}"] = float(travel)
    geo[f"pos_{tag}_end"] = entry + vel / speed * travel
  return geo


def check_frame0(p: dict, geo: dict) -> str:
  """Frame-0 and end-of-clip sanity. Feasible for every legal draw; checked anyway."""
  half_w, half_h = half_extents()
  for tag in ("a", "b"):
    r = p[f"radius_{tag}"] * SILHOUETTE_SAFETY + FRAME_MARGIN
    for where, pos in (("start", geo[f"pos_{tag}_start"]), ("end", geo[f"pos_{tag}_end"])):
      if abs(pos[0]) + r > half_w or abs(pos[1]) + r > half_h:
        return f"{tag}_{where}_out_of_frame"
    # Each ball must stay strictly on its own patch: the two patches meet at y=0.
    lo = min(geo[f"pos_{tag}_start"][1], geo[f"pos_{tag}_end"][1])
    hi = max(geo[f"pos_{tag}_start"][1], geo[f"pos_{tag}_end"][1])
    clearance = p[f"radius_{tag}"] + MIN_LANE_CLEARANCE
    if np.sign(lo) != np.sign(hi) or min(abs(lo), abs(hi)) < clearance:
      return f"{tag}_crosses_lane_seam"
  gap = abs(geo["pos_a_start"][1] - geo["pos_b_start"][1])
  if gap < p["radius_a"] + p["radius_b"] + 0.3:
    return "lanes_too_close"
  return ""


# --------------------------------------------------------------------------------------
# Build and simulate
# --------------------------------------------------------------------------------------

_SLIDING_SPHERE_INSTALLED = False


def install_sliding_sphere_adder() -> None:
  """Re-register kubric's sphere builder to create bodies in MINIMAL coordinates.

  Kubric creates every body with ``useMaximalCoordinates=True`` — its fix for the
  sticky-walls bug (bulletphysics#3094). That flag and the frozen inertia tensor
  this scenario needs are **incompatible**: measured directly, a maximal-
  coordinates sphere with ``localInertiaDiagonal = 1e6`` produces NaN velocities
  from the first step, and kubric reports the ball coasting forever because the
  NaNs never reach an acceptance check. In minimal coordinates the same body
  slides exactly as Coulomb friction says it should — measured deceleration
  0.5999999 against mu*g = 0.6.

  So spheres are built here instead, identically except for that flag. Cubes
  still go through kubric's own path, which is what the floor tiles use; there
  are no walls in this scene for the sticky-walls fix to matter to.

  The inertia override itself must stay OUTSIDE this function, applied after the
  asset is added: kubric's ``mass`` observer fires during ``scene += ball`` and
  ``changeDynamics(mass=...)`` recomputes the inertia tensor from the shape,
  discarding anything set earlier.
  """
  global _SLIDING_SPHERE_INSTALLED
  if _SLIDING_SPHERE_INSTALLED:
    return
  import pybullet as pb
  from kubric.core import objects as core_objects
  from kubric.simulator.pybullet import (PyBullet, register_physical_object_setters,
                                         wxyz2xyzw)

  @PyBullet.add_asset.register(core_objects.Sphere)
  def _add_sliding_sphere(self, obj):  # pylint: disable=unused-variable
    radius = obj.scale[0]
    assert radius == obj.scale[1] == obj.scale[2], obj.scale
    collision_idx = self._physics_client.createCollisionShape(  # pylint: disable=protected-access
        pb.GEOM_SPHERE, radius=radius)
    body_idx = self._physics_client.createMultiBody(  # pylint: disable=protected-access
        0 if obj.static else obj.mass, collision_idx, -1, obj.position,
        wxyz2xyzw(obj.quaternion), useMaximalCoordinates=False)
    self._physics_client.changeDynamics(  # pylint: disable=protected-access
        body_idx, -1, contactProcessingThreshold=0)
    register_physical_object_setters(obj, body_idx, self._physics_client)  # pylint: disable=protected-access
    return body_idx

  _SLIDING_SPHERE_INSTALLED = True


def _slab(kb, name: str, x0: float, x1: float, y0: float, y1: float, spec: dict,
          friction: float):
  """One static floor tile, top face at z = 0."""
  return kb.Cube(
      name=name,
      scale=(0.5 * (x1 - x0), 0.5 * (y1 - y0), 0.5 * FLOOR_THICKNESS),
      position=(0.5 * (x0 + x1), 0.5 * (y0 + y1), -0.5 * FLOOR_THICKNESS),
      material=kb.PrincipledBSDFMaterial(
          color=kb.Color(*spec["color"], 1.0), metallic=spec["metallic"],
          roughness=spec["roughness"], specular=spec["specular"]),
      static=True, friction=friction, restitution=0.0,
      background=True, segmentation_id=SEG_ID["floor"],
  )


def build_and_simulate(p: dict, geo: dict, scratch_dir: str, render: bool):
  """Build the scene, attach views, run the simulation."""
  import kubric as kb
  from kubric.simulator import PyBullet
  import pybullet as pb

  install_sliding_sphere_adder()

  scene = kb.Scene(
      frame_start=0,
      frame_end=NUM_FRAMES - 1,
      frame_rate=FRAME_RATE,
      step_rate=STEP_RATE,
      resolution=RESOLUTION,
      gravity=(0.0, 0.0, -GRAVITY),
      background=kb.Color(0.30, 0.30, 0.32, 1.0),
      ambient_illumination=kb.Color(0.04, 0.04, 0.04, 1.0),
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

  # Same camera and light as the momentum set, so both datasets present the same
  # geometry to the model and px/world is identical.
  camera = kb.PerspectiveCamera(
      name="camera", focal_length=FOCAL_LENGTH, sensor_width=SENSOR_WIDTH,
      position=(0.0, 0.0, RADIUS_RANGE[1] + 0.05 + CAMERA_DISTANCE),
      quaternion=(1.0, 0.0, 0.0, 0.0),
  )
  scene += camera
  scene.camera = camera
  scene += kb.DirectionalLight(name="sun", position=(5.0, -4.0, 14.0),
                               look_at=(0.0, 0.0, 0.0), intensity=3.0)

  # The floor is three tiles that tile the plane exactly, edge to edge with no
  # overlap: overlapping coplanar faces would z-fight, and a gap would drop a
  # ball. Friction is a property of the tile, which is the whole point.
  xb = p["boundary_x"]
  scene += _slab(kb, "approach", -FLOOR_EXTENT_X, xb, -FLOOR_EXTENT_Y, FLOOR_EXTENT_Y,
                 p["appearance_approach"], 0.0)
  scene += _slab(kb, "patch_a", xb, FLOOR_EXTENT_X, 0.0, FLOOR_EXTENT_Y,
                 p["appearance_patch_a"], p["mu_a"])
  scene += _slab(kb, "patch_b", xb, FLOOR_EXTENT_X, -FLOOR_EXTENT_Y, 0.0,
                 p["appearance_patch_b"], p["mu_b"])

  balls = {}
  for tag, seg_key in (("a", "A"), ("b", "B")):
    ball = kb.Sphere(
        name=f"ball_{tag}",
        scale=p[f"radius_{tag}"],
        position=tuple(geo[f"pos_{tag}_start"]),
        velocity=tuple(geo[f"vel_{tag}"]),
        # Density is fixed and unremarkable: deceleration is mass-independent, so
        # mass is a control here rather than a factor.
        mass=float(1.0 * 4.0 / 3.0 * np.pi * p[f"radius_{tag}"] ** 3),
        # PyBullet multiplies the two lateral frictions, so a ball at 1.0 makes
        # the pair coefficient exactly the tile's. Verified per scene by
        # decel_measured in labels.json.
        friction=1.0,
        restitution=0.0,
        material=kb.PrincipledBSDFMaterial(
            color=kb.Color(BALL_ALBEDO, BALL_ALBEDO, BALL_ALBEDO, 1.0),
            **V2_SURFACE),
        segmentation_id=SEG_ID[seg_key],
    )
    scene += ball
    balls[tag] = ball

  for ball in balls.values():
    body_id = ball.linked_objects[simulator]
    pb.changeDynamics(body_id, -1, linearDamping=0.0, angularDamping=0.0,
                      rollingFriction=0.0, spinningFriction=0.0,
                      # Freeze rotation: a rolling ball would coast at 5/7 of its
                      # entry speed instead of decelerating at mu*g.
                      localInertiaDiagonal=[FROZEN_INERTIA] * 3,
                      physicsClientId=simulator.physics_client)

  animation, collisions = simulator.run(frame_start=0, frame_end=NUM_FRAMES - 1)
  return scene, balls, camera, simulator, renderer, animation, collisions


# --------------------------------------------------------------------------------------
# Physics acceptance
# --------------------------------------------------------------------------------------

def _measure_one(speed_track: np.ndarray, v0: float, decel_expected: float) -> dict:
  """Entry frame and deceleration, read back out of the simulated speed track.

  Fitted rather than read off a single pair of frames: the deceleration is the
  slope of the decaying segment, and extrapolating that line back to the coasting
  speed gives the entry time to sub-frame precision — the same trick the momentum
  set uses to locate contact between frames.
  """
  # Every frame after the crossing is pure sliding — a sphere's contact is a
  # single point, so there is no straddle to blend two coefficients — and the
  # decaying segment can be as short as two frames when a fast-stopping ball
  # meets a rough patch (mu 0.8 against speed 0.6 stops in three frames). That
  # case must NOT be rejected: rejecting it would make acceptance depend on mu
  # and speed jointly, and reintroduce through the back door exactly the kind of
  # confound the sampler is built to avoid.
  moving = speed_track > 0.02 * v0
  decaying = np.where(moving & (speed_track < v0 * (1.0 - 1e-3)))[0]
  if decaying.size < 1:
    return {"error": "no_decay_segment"}

  frames = decaying
  t = frames / FRAME_RATE
  if frames.size >= 2:
    slope, intercept = np.polyfit(t, speed_track[frames], 1)
  else:
    # One usable sample: the deceleration is constant, so two points — the
    # coasting speed at entry and this one — determine the line exactly.
    slope = (speed_track[frames[0]] - v0) / (t[0] - ENTRY_TIME)
    intercept = v0 - slope * ENTRY_TIME
  decel = float(-slope)
  if decel <= 1e-9:
    return {"error": "no_deceleration"}

  entry_time = float((v0 - intercept) / slope)   # where the fitted line meets v0
  stop_time = float(-intercept / slope)
  residual = float(np.max(np.abs(np.polyval([slope, intercept], t) - speed_track[frames])))
  stopped = np.where(~moving)[0]
  return {
      "entry_frame": entry_time * FRAME_RATE,
      "decel_measured": decel,
      "decel_expected": decel_expected,
      "decel_rel_error": abs(decel - decel_expected) / max(decel_expected, 1e-12),
      "stop_time": stop_time,
      "stop_frame": float(stop_time * FRAME_RATE),
      "stops_in_clip": bool(stopped.size > 0),
      "first_stopped_frame": int(stopped[0]) if stopped.size else None,
      "fit_frames": [int(frames[0]), int(frames[-1])],
      "fit_max_residual": residual,
  }


def check_simulation(p: dict, geo: dict, animation, balls, collisions) -> tuple[str, dict]:
  """Physics acceptance checks. Returns (reason_or_None, measurements)."""
  ball_set = set(balls.values())
  if any(set(e["instances"]) == ball_set for e in collisions):
    return "balls_collided", {}

  measurements = {}
  for tag in ("a", "b"):
    anim = animation[balls[tag]]
    vel = np.array(anim["velocity"])
    pos = np.array(anim["position"])
    ang = np.array(anim["angular_velocity"])
    speed = np.linalg.norm(vel[:, :2], axis=1)
    v0 = float(p[f"speed_{tag}"])

    if np.max(np.abs(vel[:, 2])) > MAX_VERTICAL_SPEED:
      return f"{tag}_vertical_motion", {}
    if np.max(np.abs(pos[:, 2] - p[f"radius_{tag}"])) > 0.01:
      return f"{tag}_left_the_floor", {}
    if np.max(np.abs(ang)) > MAX_ANGULAR_SPEED:
      return f"{tag}_rotated", {}
    # Frictionless approach: constant speed until the boundary.
    coast = speed[:TARGET_ENTRY_FRAME]
    if np.max(np.abs(coast - v0)) / v0 > COAST_REL_TOLERANCE:
      return f"{tag}_not_coasting", {}

    m = _measure_one(speed, v0, geo[f"decel_{tag}"])
    if "error" in m:
      return f"{tag}_{m['error']}", {}
    if abs(m["entry_frame"] - TARGET_ENTRY_FRAME) > ENTRY_TOLERANCE:
      return f"{tag}_entry_off_target", m
    if m["decel_rel_error"] > DECEL_REL_TOLERANCE:
      return f"{tag}_decel_mismatch", m
    measurements[tag] = m

  measurements["entry_frame"] = float(
      0.5 * (measurements["a"]["entry_frame"] + measurements["b"]["entry_frame"]))
  return "", measurements


# --------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------

def build_labels(p, geo, scene, balls, camera, animation, measurements, seed, scene_id,
                 seg_stats, gpu_desc) -> dict:
  half_w, _ = half_extents()
  px_per_world = RESOLUTION[0] / (2.0 * half_w)

  per_frame = []
  for frame in range(NUM_FRAMES):
    entry = {"frame": frame, "objects": {}}
    for tag, key in (("a", "A"), ("b", "B")):
      anim = animation[balls[tag]]
      vel = np.array(anim["velocity"][frame], dtype=np.float64)
      entry["objects"][key] = {
          "position": list(np.asarray(anim["position"][frame], dtype=np.float64)),
          "velocity": vel.tolist(),
          "speed": float(np.linalg.norm(vel[:2])),
      }
    per_frame.append(entry)

  def vel_at(tag, frame):
    return np.array(animation[balls[tag]]["velocity"][frame], dtype=np.float64)

  pre = {t: vel_at(t, PRE_FRAME) for t in ("a", "b")}
  post = {t: vel_at(t, POST_FRAME) for t in ("a", "b")}

  # How visible is the deceleration at all? The post-entry departure from the
  # undecelerated straight line, in pixels — the friction analogue of the
  # momentum set's min_delta_displacement_px. Below the tracker's ~0.1 px
  # centroid noise the coefficient is unrecoverable for information-theoretic
  # reasons that say nothing about any representation, so it is recorded per
  # scene and can be stratified on.
  identifiability = {}
  for tag, key in (("a", "A"), ("b", "B")):
    t = min(POST_SECONDS, geo[f"stop_time_{tag}"])
    deviation = 0.5 * geo[f"decel_{tag}"] * t * t
    identifiability[f"path_deviation_px_{key}"] = float(deviation * px_per_world)
    identifiability[f"speed_drop_px_per_frame_{key}"] = float(
        min(p[f"speed_{tag}"], geo[f"decel_{tag}"] * POST_SECONDS)
        * px_per_world / FRAME_RATE)
  identifiability["min_path_deviation_px"] = float(min(
      identifiability["path_deviation_px_A"], identifiability["path_deviation_px_B"]))

  return {
      "scene_id": scene_id,
      "seed": seed,
      "scenario": "sliding_friction",
      "arm": p["arm"],
      "stratum": p["stratum"],
      "mismatched_appearance": p["mismatched_appearance"],
      # Momentum is NOT conserved here — friction is an external force — and the
      # key is written explicitly because `momentum_conserved` defaults to True
      # when absent, which would label every scene wrongly. It is constant across
      # the set, so probe_grid drops it as a target, which is the honest outcome.
      "momentum_conserved": False,
      "objects": {
          "A": {"radius": p["radius_a"], "mass": float(4.0 / 3.0 * np.pi * p["radius_a"] ** 3),
                "speed_initial": p["speed_a"], "heading_rad": p["heading_a_rad"],
                "friction_object": 1.0, "segmentation_id": SEG_ID["A"],
                "color_rgb": [BALL_ALBEDO] * 3},
          "B": {"radius": p["radius_b"], "mass": float(4.0 / 3.0 * np.pi * p["radius_b"] ** 3),
                "speed_initial": p["speed_b"], "heading_rad": p["heading_b_rad"],
                "friction_object": 1.0, "segmentation_id": SEG_ID["B"],
                "color_rgb": [BALL_ALBEDO] * 3},
      },
      "floor": {
          "boundary_x": p["boundary_x"],
          "approach_friction": 0.0,
          "mu_A": p["mu_a"], "mu_B": p["mu_b"],
          "advertised_mu_A": p["advertised_mu_a"],
          "advertised_mu_B": p["advertised_mu_b"],
          "patch_albedo_A": p["appearance_patch_a"]["color"][0],
          "patch_albedo_B": p["appearance_patch_b"]["color"][0],
          "lane_y_A": p["lane_y_a"], "lane_y_B": p["lane_y_b"],
      },
      "friction": {
          "gravity": GRAVITY,
          "mu_A": p["mu_a"], "mu_B": p["mu_b"],
          "mu_ratio": float(p["mu_a"] / p["mu_b"]),
          "log_mu_ratio": float(np.log(p["mu_a"] / p["mu_b"])),
          "decel_A": geo["decel_a"], "decel_B": geo["decel_b"],
          "decel_A_measured": measurements["a"]["decel_measured"],
          "decel_B_measured": measurements["b"]["decel_measured"],
          "decel_rel_error_A": measurements["a"]["decel_rel_error"],
          "decel_rel_error_B": measurements["b"]["decel_rel_error"],
          "stop_frame_A": measurements["a"]["stop_frame"],
          "stop_frame_B": measurements["b"]["stop_frame"],
          "stops_in_clip_A": measurements["a"]["stops_in_clip"],
          "stops_in_clip_B": measurements["b"]["stops_in_clip"],
          "travel_A": geo["travel_a"], "travel_B": geo["travel_b"],
          "fit_max_residual_A": measurements["a"]["fit_max_residual"],
          "fit_max_residual_B": measurements["b"]["fit_max_residual"],
      },
      "identifiability": identifiability,
      # The phase-splitting event. Named `collision` and shaped like the momentum
      # set's so `runner.capture_scene` / `encode_scene` split pre/post on it with
      # no changes; here it is the frame each ball crosses onto its friction
      # patch, which both do by construction.
      "collision": {
          "event": "friction_entry",
          "target_frame": TARGET_ENTRY_FRAME,
          "actual_frame": measurements["entry_frame"],
          "actual_frame_rounded": int(round(measurements["entry_frame"])),
          "discrepancy_frames": measurements["entry_frame"] - TARGET_ENTRY_FRAME,
          "entry_frame_A": measurements["a"]["entry_frame"],
          "entry_frame_B": measurements["b"]["entry_frame"],
          "boundary_x": p["boundary_x"],
          "pre": {
              "frame": PRE_FRAME,
              "velocity_A": pre["a"].tolist(), "velocity_B": pre["b"].tolist(),
              "speed_A": float(np.linalg.norm(pre["a"][:2])),
              "speed_B": float(np.linalg.norm(pre["b"][:2])),
          },
          "post": {
              "frame": POST_FRAME,
              "velocity_A": post["a"].tolist(), "velocity_B": post["b"].tolist(),
              "speed_A": float(np.linalg.norm(post["a"][:2])),
              "speed_B": float(np.linalg.norm(post["b"][:2])),
          },
      },
      "per_frame": per_frame,
      "segmentation_stats": seg_stats,
      "scene_constants": {
          "num_frames": NUM_FRAMES, "frame_rate": FRAME_RATE, "step_rate": STEP_RATE,
          "resolution": list(RESOLUTION), "gravity": [0.0, 0.0, -GRAVITY],
          "camera_distance": CAMERA_DISTANCE,
          "camera_position": [0.0, 0.0, RADIUS_RANGE[1] + 0.05 + CAMERA_DISTANCE],
          "camera_quaternion": [1.0, 0.0, 0.0, 0.0],
          "focal_length": FOCAL_LENGTH, "sensor_width": SENSOR_WIDTH,
          "visible_half_extents": list(half_extents()),
          "px_per_world": float(px_per_world),
          "radius_range": list(RADIUS_RANGE), "speed_range": list(SPEED_RANGE),
          "mu_range": list(MU_RANGE), "heading_max_deg": HEADING_MAX_DEG,
          "lane_y": LANE_Y, "lane_y_jitter": LANE_Y_JITTER,
          "boundary_x_jitter": BOUNDARY_X_JITTER,
          "uniform_floor_frac": UNIFORM_FLOOR_FRAC,
          "floor_albedo": FLOOR_ALBEDO, "floor_albedo_range": list(FLOOR_ALBEDO_RANGE),
          "ball_albedo": BALL_ALBEDO,
          "frozen_inertia": FROZEN_INERTIA,
          "target_entry_frame": TARGET_ENTRY_FRAME,
          "pre_frame": PRE_FRAME, "post_frame": POST_FRAME,
          "samples_per_pixel": SAMPLES_PER_PIXEL,
          "arm": p["arm"],
          "render_device": gpu_desc,
      },
  }


# --------------------------------------------------------------------------------------

def generate(seed: int, out_dir: Path, scene_id: str, render: bool = True,
             arm: str = "latent_friction", mismatch: bool = False) -> dict:
  import kubric as kb
  import pybullet as pb

  gpu_desc = "n/a (physics only)"
  scratch_dir = tempfile.mkdtemp(prefix="kubric_friction_")
  rejections = []
  accepted = None
  try:
    # --- Pass 1: find a parameter set that passes every check, physics only.
    for attempt in range(MAX_ATTEMPTS):
      rng = np.random.default_rng([seed, attempt])
      p, reason = sample_params(rng, arm=arm, mismatch=mismatch)
      if p is None:
        rejections.append({"attempt": attempt, "reason": reason})
        continue
      geo = solve_geometry(p)
      reason = check_frame0(p, geo)
      if reason:
        rejections.append({"attempt": attempt, "reason": reason})
        continue

      (_, balls, _, simulator, _, animation,
       collisions) = build_and_simulate(p, geo, scratch_dir, render=False)
      reason, measurements = check_simulation(p, geo, animation, balls, collisions)
      pb.disconnect(physicsClientId=simulator.physics_client)
      if reason:
        rejections.append({"attempt": attempt, "reason": reason})
        continue

      accepted = (attempt, p, geo, measurements)
      break

    if accepted is None:
      raise RuntimeError(f"no valid scene after {MAX_ATTEMPTS} attempts: {rejections}")

    attempt, p, geo, _ = accepted
    logger.info("accepted on attempt %d after %d rejections", attempt, len(rejections))

    # --- Pass 2: rebuild the accepted scene, this time with a renderer attached.
    (scene, balls, camera, _, renderer, animation,
     collisions) = build_and_simulate(p, geo, scratch_dir, render=render)
    reason, measurements = check_simulation(p, geo, animation, balls, collisions)
    if reason:
      raise RuntimeError(f"replay of accepted scene failed the check: {reason}")

    out_dir.mkdir(parents=True, exist_ok=True)
    seg_stats = None
    if render:
      gpu_desc = configure_gpu(renderer)
      logger.info("render device: %s", gpu_desc)
      frames = renderer.render(return_layers=("rgba", "segmentation"))
      seg_stats = segmentation_stats(frames["segmentation"])
      write_outputs(out_dir, frames["rgba"], frames["segmentation"])

    labels = build_labels(p, geo, scene, balls, camera, animation, measurements,
                          seed, scene_id, seg_stats, gpu_desc)
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
  parser.add_argument("--arm", type=str, default="latent_friction", choices=ARMS,
                      help="'latent_friction' paints every floor tile identically so "
                           "mu is recoverable only from the motion; 'visible_friction' "
                           "ramps each patch's albedo with its mu")
  parser.add_argument("--violate", action="store_true",
                      help="render a patch whose albedo advertises a coefficient the "
                           "floor does not have. Only visible in the visible arm; "
                           "detecting it needs appearance and dynamics bound together")
  parser.add_argument("--logging-level", type=str, default="INFO")
  args = parser.parse_args()

  logging.basicConfig(level=args.logging_level,
                      format="%(asctime)s %(levelname)s %(name)s: %(message)s")

  out_dir = Path(args.output_dir)
  scene_id = args.scene_id or out_dir.name
  try:
    labels = generate(args.seed, out_dir, scene_id, render=not args.no_render,
                      arm=args.arm, mismatch=args.violate)
  except Exception:  # pylint: disable=broad-except
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_FAILED").write_text(traceback.format_exc())
    logger.error("scene %s failed:\n%s", scene_id, traceback.format_exc())
    return 1

  f, c = labels["friction"], labels["collision"]
  logger.info(
      "scene %s: entry frame %.2f (target %d), mu %.3f / %.3f, "
      "decel %.4f / %.4f (rel err %.1e / %.1e), min path deviation %.1f px",
      scene_id, c["actual_frame"], TARGET_ENTRY_FRAME, f["mu_A"], f["mu_B"],
      f["decel_A_measured"], f["decel_B_measured"], f["decel_rel_error_A"],
      f["decel_rel_error_B"], labels["identifiability"]["min_path_deviation_px"])
  return 0


if __name__ == "__main__":
  sys.exit(main())
