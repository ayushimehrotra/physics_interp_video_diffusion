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
CAMERA_DISTANCE = 14.0
FOCAL_LENGTH = 50.0             # mm
SENSOR_WIDTH = 36.0             # mm

# Radii levels and the common height of both sphere centres above the floor.
RADII = (0.30, 0.45, 0.60)
CENTER_HEIGHT = max(RADII) + 0.05   # every centre sits here, so contact normals are horizontal

# Two material levels: visually distinct, different densities.
MATERIALS = {
    "metal": {
        "density": 8.0,
        "color": (0.72, 0.74, 0.78),
        "metallic": 0.9,
        "roughness": 0.25,
        "specular": 0.9,
        "ior": 2.5,
    },
    "rubber": {
        "density": 1.2,
        "color": (0.80, 0.10, 0.10),
        "metallic": 0.0,
        "roughness": 0.8,
        "specular": 0.3,
        "ior": 1.45,
    },
}

RESTITUTIONS = (0.2, 0.6, 1.0)  # pair coefficient; see restitution note in README
SPEED_RANGE = (0.45, 0.90)      # world units / second
IMPACT_PARAM_MAX_FRAC = 0.9     # |b| <= this * (r_a + r_b); 1.0 would be a grazing miss

# Contact point placement. 0.0 puts every collision at the exact frame centre, which
# maximises how long both spheres stay in frame. Raise it to decorrelate collision
# location from collision timing (see README).
CONTACT_JITTER_RADIUS = 0.0

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


def sample_params(rng: np.random.Generator) -> dict:
  """Draw one candidate parameter set. No validity checks here."""
  mat_a = str(rng.choice(list(MATERIALS)))
  mat_b = str(rng.choice(list(MATERIALS)))
  r_a = float(rng.choice(RADII))
  r_b = float(rng.choice(RADII))
  jitter_r = CONTACT_JITTER_RADIUS * np.sqrt(rng.uniform())
  jitter_theta = rng.uniform(0, 2 * np.pi)
  return {
      "radius_a": r_a,
      "radius_b": r_b,
      "material_a": mat_a,
      "material_b": mat_b,
      "density_a": MATERIALS[mat_a]["density"],
      "density_b": MATERIALS[mat_b]["density"],
      "speed_a": float(rng.uniform(*SPEED_RANGE)),
      "direction_a_rad": float(rng.uniform(0, 2 * np.pi)),
      "impact_parameter_frac": float(rng.uniform(0, IMPACT_PARAM_MAX_FRAC)),
      "impact_parameter_sign": int(rng.choice([-1, 1])),
      "restitution": float(rng.choice(RESTITUTIONS)),
      "contact_offset": [float(jitter_r * np.cos(jitter_theta)),
                         float(jitter_r * np.sin(jitter_theta))],
  }


def solve_geometry(p: dict) -> dict:
  """Analytic initial condition placing first contact exactly at TARGET_CONTACT_FRAME.

  A travels in a straight line at constant velocity (no gravity, no friction, nothing
  else in its path), so first contact is when |p_a(t) - p_b| == r_a + r_b. Rather than
  solving that quadratic for t, we place the geometry so that t == t_contact by
  construction and let the simulator confirm it.
  """
  r_a, r_b = p["radius_a"], p["radius_b"]
  contact_radius = r_a + r_b
  t_contact = TARGET_CONTACT_FRAME / FRAME_RATE

  theta = p["direction_a_rad"]
  u = np.array([np.cos(theta), np.sin(theta), 0.0])          # direction of travel
  w = np.array([-np.sin(theta), np.cos(theta), 0.0])         # left perpendicular

  b = p["impact_parameter_sign"] * p["impact_parameter_frac"] * contact_radius
  # Unit vector from A's centre to B's centre at the moment of contact. Its component
  # along u is positive, which is what makes t_contact the *entering* root.
  along = np.sqrt(max(contact_radius ** 2 - b ** 2, 0.0))
  normal = (along * u + b * w) / contact_radius

  contact_point = np.array([p["contact_offset"][0], p["contact_offset"][1], CENTER_HEIGHT])
  pos_a_contact = contact_point - r_a * normal
  pos_b = contact_point + r_b * normal

  travel = p["speed_a"] * t_contact
  pos_a_start = pos_a_contact - travel * u
  vel_a = p["speed_a"] * u

  return {
      "contact_radius": contact_radius,
      "t_contact_target_s": t_contact,
      "impact_parameter": float(b),
      "impact_parameter_abs": float(abs(b)),
      "contact_normal_analytic": normal.tolist(),
      "contact_point_analytic": contact_point.tolist(),
      "pos_a_contact_analytic": pos_a_contact.tolist(),
      "pos_a_start": pos_a_start.tolist(),
      "pos_b_start": pos_b.tolist(),
      "vel_a_start": vel_a.tolist(),
      "initial_separation": float(np.linalg.norm(pos_a_start - pos_b)),
      "initial_surface_gap": float(np.linalg.norm(pos_a_start - pos_b) - contact_radius),
      "travel_before_contact": float(travel),
  }


def check_frame0(p: dict, geo: dict) -> str | None:
  """Return a rejection reason, or None if the frame-0 layout is usable."""
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
        velocity=geo["vel_a_start"] if tag == "a" else (0.0, 0.0, 0.0),
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
          "momentum_rel_error": measurements["momentum_rel_error"],
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
          "radii_levels": list(RADII), "restitution_levels": list(RESTITUTIONS),
          "speed_range": list(SPEED_RANGE), "materials": MATERIALS,
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

def generate(seed: int, out_dir: Path, scene_id: str, render: bool = True) -> dict:
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
      p = sample_params(rng)
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
  parser.add_argument("--logging-level", type=str, default="INFO")
  args = parser.parse_args()

  logging.basicConfig(level=args.logging_level,
                      format="%(asctime)s %(levelname)s %(name)s: %(message)s")

  out_dir = Path(args.output_dir)
  scene_id = args.scene_id or out_dir.name
  try:
    labels = generate(args.seed, out_dir, scene_id, render=not args.no_render)
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
