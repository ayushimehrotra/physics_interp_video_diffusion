#!/usr/bin/env python3
"""Generate one side-view sliding-friction scene: a cube on a floor, and the floor's mu.

    python3 generate_slide_scene.py --seed 0 --output-dir out/scene_00000 --arm latent_friction

The two objects here are **the cube and the floor**. That is the multi-object
structure this dataset is about: one of the pair carries the physical parameter
(the floor's coefficient of friction) and the other is the only thing that
reveals it (the cube's deceleration). Segmentation ids follow that reading, so
the existing pooling code addresses them with no changes:

    id 1 -> the cube   -> pool "A"      the object whose motion is the evidence
    id 2 -> the floor  -> pool "B"      the surface that owns the coefficient

Which makes the central probe a clean contrast rather than a single number:
`mu` from `A` asks whether the model reads the coefficient off the motion, and
`mu` from `B` asks whether it has attributed the property to the *surface*. In
the latent arm the floor is one flat albedo, so `B` should read nothing — and
that is a prediction about the dataset, not about the model.

Camera is a SIDE view: horizontal, looking across the floor plane, with a small
downward tilt so the contact between cube and floor is visible. The momentum
sets are all overhead, so nothing about this geometry is inherited — the camera
block below is the whole difference.

Physics
-------
A cube slides in a straight line and decelerates at ``a = mu * g``, exactly.

  * mass-independent, so cube size is a FREE control here: it cannot enter the
    target, unlike the momentum set where mass is a factor of the label.
  * no rotation: a sliding box does not roll, and it only tips if
    ``mu > half_width / half_height``, which is 1.0 for a cube. mu is capped
    well below that, and the simulation checks it rather than assuming it.
  * a cube rather than a sphere on purpose. A sphere with lateral friction ROLLS
    — measured, it spins up and then coasts at 5/7 of its entry speed forever,
    so mu would control a transient duration rather than the deceleration. The
    fix for a sphere is either a frozen inertia tensor (which NaNs under
    kubric's maximal-coordinates bodies) or rolling resistance (whose law is
    ``a = (5/7) mu_r g / r``, dragging radius into the target). Neither is worth
    it when a box slides cleanly.

Arms (``--arm``):

  latent_friction   the floor renders with one fixed albedo whatever its mu, so
                    the coefficient is recoverable only from the deceleration.
  visible_friction  floor albedo ramps log-linearly with mu: readable off a
                    single frame, before the cube has moved. The positive control.

Honest limit, per CLAUDE.md §12: ``mu = a / g`` and ``a`` is plainly visible, so
a probe on `mu` from the cube's motion is measuring kinematics and a division by
a constant. Read it against a kinematic oracle. What is NOT an identity is `mu`
from the FLOOR's tokens, and that contrast is the reason this dataset exists.
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

from generate_scene import (  # noqa: E402
    FRAME_RATE,
    NUM_FRAMES,
    RESOLUTION,
    STEP_RATE,
    configure_gpu,
    write_outputs,
)

logger = logging.getLogger("generate_slide_scene")

ARMS = ("latent_friction", "visible_friction")

# ---- physics ---------------------------------------------------------------
# world units / s^2. Not 9.81: this world is not metric, and at 9.81 an ordinary
# mu stops a cube in three frames with nothing to see. 1.0 rather than the two-ball
# set's 2.0 because a = mu*g sets the STOPPING DISTANCE, v^2/(2a), and at g = 2 a
# rough floor stopped the cube inside 0.2 world units — 13 px, a nearly static clip.
GRAVITY = 1.0
MU_RANGE = (0.15, 0.80)          # log-uniform; 0.80 is well under the tipping
                                 # threshold of 1.0 for a cube
# Speeds are deliberately modest. A 6-second clip at ~1.5 u/s needs 8 units of
# travel, which forces the camera far enough back that the cube renders ~35 px
# and two thirds of the frame is empty floor. Halving the speed range shrinks the
# travel to <2.5 units, lets the camera come in to 14, and puts the cube at ~64 px
# — the size the momentum set's spheres render at. At 71 px/unit even 0.5 u/s is
# 4.4 px/frame, well above the ~1.4 px/frame legibility floor.
SPEED_RANGE = (1.00, 1.60)       # world units / s at frame 0
HALF_EXTENT_RANGE = (0.25, 0.45) # cube half-size; a free control (a = mu*g is
                                 # independent of both mass and size)
DENSITY = 1.0

# ---- camera: SIDE view ------------------------------------------------------
# Horizontal-ish, off to the -y side, with a small downward tilt so the cube's
# contact with the floor reads clearly instead of sitting on the horizon line.
CAMERA_DISTANCE = 14.0           # along -y from the motion line
# Low, so the view is nearly eye-level with the cube: at 2.6 the horizon sat in
# the top sixth of the frame and the cube rode along it with dead floor beneath.
CAMERA_HEIGHT = 1.1
LOOK_AT_Z = 0.35
FOCAL_LENGTH = 50.0
SENSOR_WIDTH = 36.0

# Visible half-width in the plane of the motion, used for the frame checks.
HALF_WIDTH = 0.5 * CAMERA_DISTANCE * SENSOR_WIDTH / FOCAL_LENGTH

START_X = -0.70 * HALF_WIDTH     # launch point; the checks below keep the whole
FRAME_MARGIN = 0.6               # trajectory inside the frame with this margin

# ---- appearance -------------------------------------------------------------
# The ramp goes UP from the neutral floor so every floor tone stays brighter than
# the cube, which keeps a luma-threshold tracker usable on this set.
FLOOR_ALBEDO = 0.42
FLOOR_ALBEDO_RANGE = (0.42, 0.78)
FLOOR_SURFACE = {"metallic": 0.0, "roughness": 0.8, "specular": 0.2}
CUBE_ALBEDO = 0.20
CUBE_SURFACE = {"metallic": 0.0, "roughness": 0.6, "specular": 0.4, "ior": 1.45}

FLOOR_EXTENT = 60.0              # runs past the horizon; no floor edge in shot
FLOOR_THICKNESS = 1.0

# The floor is built as BANDS across the direction of travel, alternating between
# two albedos. Not decoration: with a single flat floor, a flat background and a
# featureless horizon, the only thing that moves in the whole frame is the cube,
# so a translating cube under a static camera is geometrically indistinguishable
# from a static cube under a panning camera — no parallax, no static reference,
# and no apparent-size change because the motion is parallel to the image plane.
# The bands give the frame a fixed reference the cube visibly passes.
#
# The pattern is IDENTICAL in every scene — same positions, same two albedos — so
# it cannot carry mu. Only the arm's ramp scales it, and that scales both bands
# together, which validate_slide.py's floor-luma check still gates.
FLOOR_BAND_WIDTH = 1.0
FLOOR_BAND_CONTRAST = 1.09       # second albedo = first * this

# The phase split. There is no collision here and no boundary — the floor is
# level and uniform — so this is simply an early/late split of the clip, kept
# under the `collision` key and at the momentum set's frame so that
# runner.capture_scene / encode_scene divide the pre/post pools unchanged.
PHASE_SPLIT_FRAME = 20
PRE_FRAME = 8
POST_FRAME = 34

SEG_ID = {"cube": 1, "floor": 2}

DECEL_REL_TOLERANCE = 0.05
MAX_VERTICAL_SPEED = 0.02
# A sliding cube must not tumble. Measured on this setup, an honest slide carries
# 0.010-0.016 rad/s of contact jitter and drifts 0.1-1.4 degrees of orientation
# over the whole clip, so a threshold near that jitter rejects every scene for a
# defect that does not exist. Both the rate and the integrated drift are checked;
# the drift is the one that would be visible.
MAX_ANGULAR_SPEED = 0.05
MAX_ORIENTATION_DRIFT_DEG = 5.0

MAX_ATTEMPTS = 20
SAMPLES_PER_PIXEL = 64

CLIP_SECONDS = (NUM_FRAMES - 1) / FRAME_RATE


# --------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------

def albedo_for_mu(mu: float) -> float:
  """Floor albedo encoding mu for the visible arm, log-linear in mu."""
  lo, hi = MU_RANGE
  frac = (np.log(mu) - np.log(lo)) / (np.log(hi) - np.log(lo))
  a_lo, a_hi = FLOOR_ALBEDO_RANGE
  return float(a_lo + frac * (a_hi - a_lo))


def sample_params(rng: np.random.Generator, arm: str = "latent_friction",
                  mismatch: bool = False) -> dict:
  """Draw one scene. Nothing is rejected, so mu cannot correlate with anything else.

  mu, speed and size are drawn independently and the frame-geometry check below
  is satisfied across the whole product of their ranges — so acceptance never
  depends on the parameters, and the confound that forced the momentum set's v2
  rebuild cannot enter here.
  """
  if arm not in ARMS:
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
  log_lo, log_hi = np.log(MU_RANGE[0]), np.log(MU_RANGE[1])
  mu = float(np.exp(rng.uniform(log_lo, log_hi)))

  advertised = mu
  if mismatch:
    advertised = float(np.exp(rng.uniform(log_lo, log_hi)))
    while abs(np.log(advertised) - np.log(mu)) < 0.5 * (log_hi - log_lo):
      advertised = float(np.exp(rng.uniform(log_lo, log_hi)))

  albedo = FLOOR_ALBEDO if arm == "latent_friction" else albedo_for_mu(advertised)
  return {
      "arm": arm,
      "mu": mu,
      "advertised_mu": advertised,
      "mismatched_appearance": bool(mismatch),
      "floor_albedo": albedo,
      "half_extent": float(rng.uniform(*HALF_EXTENT_RANGE)),
      "speed": float(rng.uniform(*SPEED_RANGE)),
      "start_x": float(START_X + rng.uniform(-0.4, 0.4)),
      "y_offset": float(rng.uniform(-0.5, 0.5)),   # depth jitter: not all scenes
                                                   # sit on the same line
  }


def derive(p: dict) -> dict:
  """Deceleration, stopping time and end point — all closed form for a sliding box."""
  decel = p["mu"] * GRAVITY
  t_stop = p["speed"] / decel
  t = min(CLIP_SECONDS, t_stop)
  travel = p["speed"] * t - 0.5 * decel * t * t
  return {
      "decel": float(decel),
      "stop_time": float(t_stop),
      "stops_in_clip": bool(t_stop <= CLIP_SECONDS),
      "travel": float(travel),
      "end_x": float(p["start_x"] + travel),
      "mass": float(DENSITY * (2 * p["half_extent"]) ** 3),
  }


def check_frame0(p: dict, g: dict) -> str:
  half = p["half_extent"]
  if abs(p["start_x"]) + half + FRAME_MARGIN > HALF_WIDTH:
    return "start_out_of_frame"
  if abs(g["end_x"]) + half + FRAME_MARGIN > HALF_WIDTH:
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
  pb.setPhysicsEngineParameter(splitImpulsePenetrationThreshold=0.0,
                               physicsClientId=simulator.physics_client)
  renderer = None
  if render:
    from kubric.renderer import Blender
    renderer = Blender(scene, scratch_dir, use_denoising=True,
                       adaptive_sampling=True, samples_per_pixel=SAMPLES_PER_PIXEL)
    renderer.blender_scene.view_settings.view_transform = "Standard"

  # THE side view. look_at is safe here (unlike the overhead cameras, whose
  # straight-down direction makes the up-vector degenerate).
  camera = kb.PerspectiveCamera(
      name="camera", focal_length=FOCAL_LENGTH, sensor_width=SENSOR_WIDTH,
      position=(0.0, -CAMERA_DISTANCE, CAMERA_HEIGHT),
      look_at=(0.0, 0.0, LOOK_AT_Z),
  )
  scene += camera
  scene.camera = camera

  # Key light from over the camera's shoulder, so the cube is lit on the face
  # that points at us and its shadow falls away across the floor.
  scene += kb.DirectionalLight(name="sun", position=(-6.0, -10.0, 12.0),
                               look_at=(0.0, 0.0, 0.0), intensity=3.2)
  scene += kb.DirectionalLight(name="fill", position=(8.0, -6.0, 6.0),
                               look_at=(0.0, 0.0, 0.0), intensity=0.8)

  # The floor IS the second object: it owns mu, and it gets its own segmentation
  # id so a probe can be pointed at the surface rather than at the cube. It is
  # built from bands across the travel direction (see FLOOR_BAND_WIDTH) so the
  # frame contains a static reference; every band shares one mu and one
  # segmentation id, so downstream nothing knows the difference.
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
  cube = kb.Cube(
      name="cube", scale=(h, h, h),
      position=(p["start_x"], p["y_offset"], h),
      velocity=(p["speed"], 0.0, 0.0),
      mass=g["mass"],
      # PyBullet multiplies the two lateral frictions, so a cube at 1.0 makes the
      # pair coefficient exactly the floor's. Checked per scene by decel_measured.
      friction=1.0, restitution=0.0,
      material=kb.PrincipledBSDFMaterial(
          color=kb.Color(CUBE_ALBEDO, CUBE_ALBEDO, CUBE_ALBEDO, 1.0), **CUBE_SURFACE),
      segmentation_id=SEG_ID["cube"],
  )
  scene += cube

  pb.changeDynamics(cube.linked_objects[simulator], -1,
                    linearDamping=0.0, angularDamping=0.0,
                    rollingFriction=0.0, spinningFriction=0.0,
                    physicsClientId=simulator.physics_client)

  animation, collisions = simulator.run(frame_start=0, frame_end=NUM_FRAMES - 1)
  return scene, cube, floor, camera, simulator, renderer, animation


# --------------------------------------------------------------------------------------
# Physics acceptance
# --------------------------------------------------------------------------------------

def check_simulation(p: dict, g: dict, animation, cube) -> tuple[str, dict]:
  anim = animation[cube]
  vel = np.array(anim["velocity"])
  pos = np.array(anim["position"])
  ang = np.array(anim["angular_velocity"])
  speed = np.abs(vel[:, 0])
  v0 = p["speed"]

  if np.max(np.abs(vel[:, 2])) > MAX_VERTICAL_SPEED:
    return "vertical_motion", {}
  if np.max(np.abs(pos[:, 2] - p["half_extent"])) > 0.02:
    return "left_the_floor", {}
  if np.max(np.abs(ang)) > MAX_ANGULAR_SPEED:
    return "tumbled", {}
  quat = np.array(anim["quaternion"])
  drift = float(np.degrees(2 * np.arccos(np.clip(np.abs(quat[:, 0]), 0, 1))).max())
  if drift > MAX_ORIENTATION_DRIFT_DEG:
    return "orientation_drift", {}
  if np.max(np.abs(vel[:, 1])) > 0.02:
    return "veered_sideways", {}

  moving = speed > 0.02 * v0
  idx = np.where(moving)[0]
  if idx.size < 2:
    return "stopped_immediately", {}
  fit = idx[1:] if idx.size > 2 else idx        # frame 0 carries the launch impulse
  t = fit / FRAME_RATE
  slope, intercept = np.polyfit(t, speed[fit], 1)
  decel = float(-slope)
  if decel <= 1e-9:
    return "no_deceleration", {}
  rel = abs(decel - g["decel"]) / g["decel"]
  if rel > DECEL_REL_TOLERANCE:
    return "decel_mismatch", {"decel_measured": decel, "decel_rel_error": rel}

  stopped = np.where(~moving)[0]
  return "", {
      "decel_measured": decel,
      "decel_rel_error": float(rel),
      "stop_frame": float(-intercept / slope * FRAME_RATE),
      "stops_in_clip": bool(stopped.size > 0),
      "first_stopped_frame": int(stopped[0]) if stopped.size else None,
      "fit_max_residual": float(np.max(np.abs(np.polyval([slope, intercept], t) - speed[fit]))),
      "total_travel": float(pos[-1, 0] - pos[0, 0]),
  }


# --------------------------------------------------------------------------------------

def build_labels(p, g, cube, floor, camera, animation, meas, seed, scene_id,
                 seg_stats, gpu_desc) -> dict:
  px_per_world = RESOLUTION[0] / (2.0 * HALF_WIDTH)
  anim = animation[cube]

  per_frame = []
  for frame in range(NUM_FRAMES):
    v = np.array(anim["velocity"][frame], dtype=np.float64)
    per_frame.append({
        "frame": frame,
        "position": list(np.asarray(anim["position"][frame], dtype=np.float64)),
        "velocity": v.tolist(),
        "speed": float(abs(v[0])),
    })

  vel_at = lambda f: np.array(anim["velocity"][f], dtype=np.float64)  # noqa: E731
  pre, post = vel_at(PRE_FRAME), vel_at(POST_FRAME)

  # Departure from the undecelerated straight line, in pixels: below the tracker's
  # ~0.1 px noise the coefficient is unrecoverable by any observer, so it is
  # recorded per scene and can be stratified on.
  t = min(CLIP_SECONDS, g["stop_time"])
  identifiability = {
      "path_deviation_px": float(0.5 * g["decel"] * t * t * px_per_world),
      "speed_drop_px_per_frame": float(
          min(p["speed"], g["decel"] * CLIP_SECONDS) * px_per_world / FRAME_RATE),
      "px_per_world": float(px_per_world),
  }

  return {
      "scene_id": scene_id,
      "seed": seed,
      "scenario": "cube_sliding_side_view",
      "arm": p["arm"],
      "mismatched_appearance": p["mismatched_appearance"],
      # Friction is an external force, so momentum is not conserved. Written
      # explicitly because the derived target defaults to True when absent.
      "momentum_conserved": False,
      "objects": {
          # A is the cube, B is the floor. The floor is a genuine object here —
          # it owns the coefficient — and it carries its own segmentation id so a
          # probe can be pointed at the surface instead of at the mover.
          "A": {"kind": "cube", "half_extent": p["half_extent"],
                "size": 2 * p["half_extent"], "mass": g["mass"],
                "speed_initial": p["speed"], "friction_object": 1.0,
                "segmentation_id": SEG_ID["cube"], "color_rgb": [CUBE_ALBEDO] * 3},
          "B": {"kind": "floor", "mu": p["mu"], "advertised_mu": p["advertised_mu"],
                "albedo": p["floor_albedo"], "friction_object": p["mu"],
                "segmentation_id": SEG_ID["floor"],
                "color_rgb": [p["floor_albedo"]] * 3},
      },
      "friction": {
          "gravity": GRAVITY,
          "mu": p["mu"],
          "log_mu": float(np.log(p["mu"])),
          "advertised_mu": p["advertised_mu"],
          "floor_albedo": p["floor_albedo"],
          "decel": g["decel"],
          "decel_measured": meas["decel_measured"],
          "decel_rel_error": meas["decel_rel_error"],
          "stop_frame": meas["stop_frame"],
          "stops_in_clip": meas["stops_in_clip"],
          "travel": meas["total_travel"],
          "fit_max_residual": meas["fit_max_residual"],
      },
      "identifiability": identifiability,
      # No collision and no boundary: the floor is level and uniform, so this is
      # an early/late split of the clip. Kept under this key, and at the momentum
      # set's frame, so the capture code's phase split needs no changes.
      "collision": {
          "event": "nominal_phase_split",
          "target_frame": PHASE_SPLIT_FRAME,
          "actual_frame": float(PHASE_SPLIT_FRAME),
          "actual_frame_rounded": PHASE_SPLIT_FRAME,
          "discrepancy_frames": 0.0,
          "pre": {"frame": PRE_FRAME, "velocity_A": pre.tolist(),
                  "speed_A": float(abs(pre[0]))},
          "post": {"frame": POST_FRAME, "velocity_A": post.tolist(),
                   "speed_A": float(abs(post[0]))},
      },
      "per_frame": per_frame,
      "segmentation_stats": seg_stats,
      "scene_constants": {
          "num_frames": NUM_FRAMES, "frame_rate": FRAME_RATE, "step_rate": STEP_RATE,
          "resolution": list(RESOLUTION), "gravity": [0.0, 0.0, -GRAVITY],
          "view": "side",
          "camera_position": [0.0, -CAMERA_DISTANCE, CAMERA_HEIGHT],
          "camera_look_at": [0.0, 0.0, LOOK_AT_Z],
          "camera_distance": CAMERA_DISTANCE, "camera_height": CAMERA_HEIGHT,
          "focal_length": FOCAL_LENGTH, "sensor_width": SENSOR_WIDTH,
          "visible_half_width": HALF_WIDTH, "px_per_world": px_per_world,
          "mu_range": list(MU_RANGE), "speed_range": list(SPEED_RANGE),
          "half_extent_range": list(HALF_EXTENT_RANGE),
          "floor_albedo": FLOOR_ALBEDO, "floor_albedo_range": list(FLOOR_ALBEDO_RANGE),
          "cube_albedo": CUBE_ALBEDO,
          "phase_split_frame": PHASE_SPLIT_FRAME,
          "pre_frame": PRE_FRAME, "post_frame": POST_FRAME,
          "samples_per_pixel": SAMPLES_PER_PIXEL,
          "arm": p["arm"], "render_device": gpu_desc,
      },
  }


def segmentation_stats(seg):
  """Per-frame pixel count and centroid for the cube, and coverage for the floor."""
  out = {}
  for key, seg_id in (("A", SEG_ID["cube"]), ("B", SEG_ID["floor"])):
    counts, centroids = [], []
    for frame in range(seg.shape[0]):
      ys, xs = np.nonzero(seg[frame, :, :, 0] == seg_id)
      counts.append(int(xs.size))
      centroids.append([float(xs.mean()), float(ys.mean())] if xs.size else None)
    out[key] = {"count": counts, "centroid": centroids}
  return out


def _trial(p: dict, g: dict, scratch_dir: str) -> tuple[str, dict]:
  """One physics-only trial, with every reference to its simulator dropped on exit.

  Deliberately NOT `pb.disconnect(...)`: kubric's `_BulletClient` disconnects in
  its own destructor, and PyBullet hands out client id 0 again as soon as the id
  is free. Disconnecting by hand therefore sets up a trap — the next attempt
  connects as client 0, the previous client object is then garbage collected, and
  its destructor disconnects the *live* client out from under the running scene.
  The failure surfaces one frame later as `pybullet.error: Not connected to
  physics server` from inside `simulator.run()`, which points at the wrong place
  entirely. Scoping the trial in a function and collecting before the next one
  starts is what makes the ordering safe.
  """
  scene, cube, floor, camera, simulator, renderer, animation = build_and_simulate(
      p, g, scratch_dir, render=False)
  return check_simulation(p, g, animation, cube)


def generate(seed: int, out_dir: Path, scene_id: str, render: bool = True,
             arm: str = "latent_friction", mismatch: bool = False) -> dict:
  import gc

  import kubric as kb

  gpu_desc = "n/a (physics only)"
  scratch_dir = tempfile.mkdtemp(prefix="kubric_slide_")
  rejections, accepted = [], None
  try:
    for attempt in range(MAX_ATTEMPTS):
      rng = np.random.default_rng([seed, attempt])
      p = sample_params(rng, arm=arm, mismatch=mismatch)
      g = derive(p)
      reason = check_frame0(p, g)
      if reason:
        rejections.append({"attempt": attempt, "reason": reason})
        continue
      reason, meas = _trial(p, g, scratch_dir)
      gc.collect()          # disconnect the trial's client BEFORE the next connect
      if reason:
        rejections.append({"attempt": attempt, "reason": reason, **meas})
        continue
      accepted = (attempt, p, g)
      break

    if accepted is None:
      raise RuntimeError(f"no valid scene after {MAX_ATTEMPTS} attempts: {rejections}")
    attempt, p, g = accepted
    logger.info("accepted on attempt %d after %d rejections", attempt, len(rejections))

    (scene, cube, floor, camera, _, renderer, animation) = build_and_simulate(
        p, g, scratch_dir, render=render)
    reason, meas = check_simulation(p, g, animation, cube)
    if reason:
      raise RuntimeError(f"replay of the accepted scene failed the check: {reason}")

    out_dir.mkdir(parents=True, exist_ok=True)
    seg_stats = None
    if render:
      gpu_desc = configure_gpu(renderer)
      logger.info("render device: %s", gpu_desc)
      frames = renderer.render(return_layers=("rgba", "segmentation"))
      seg_stats = segmentation_stats(frames["segmentation"])
      write_outputs(out_dir, frames["rgba"], frames["segmentation"])

    labels = build_labels(p, g, cube, floor, camera, animation, meas, seed, scene_id,
                          seg_stats, gpu_desc)
    labels["sampling_attempts"] = {"accepted_attempt": attempt, "rejections": rejections}
    labels["kubric_version"] = kb.__version__
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=1))
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
  parser.add_argument("--no-render", action="store_true")
  parser.add_argument("--arm", type=str, default="latent_friction", choices=ARMS)
  parser.add_argument("--violate", action="store_true",
                      help="render a floor whose albedo advertises a coefficient it "
                           "does not have; only visible in the visible arm")
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
  f = labels["friction"]
  logger.info("scene %s: mu %.3f -> a %.4f (measured %.4f, rel err %.1e), "
              "cube %.2f, v0 %.2f, travel %.2f, %s, deviation %.0f px",
              scene_id, f["mu"], f["decel"], f["decel_measured"], f["decel_rel_error"],
              labels["objects"]["A"]["size"], labels["objects"]["A"]["speed_initial"],
              f["travel"], "stops" if f["stops_in_clip"] else "still moving",
              labels["identifiability"]["path_deviation_px"])
  return 0


if __name__ == "__main__":
  sys.exit(main())
