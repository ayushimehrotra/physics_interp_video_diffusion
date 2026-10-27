#!/usr/bin/env python3
"""Render one clip of `ke_factor_set` from fully specified parameters.

    python generate_ke_scene.py --plan ke_factor_set/plan.json --clip S0007_k+1 --output-dir ke_factor_set/S0007_k+1

`generate_scene.py` SAMPLES a scene from a seed; this set needs the parameters
CONTROLLED (one factor stepped, everything else held bit-identical across a
group), so `plan_ke_set.py` writes every parameter into the plan and this script
replays it. All physics, rendering and labelling is generate_scene.py's own code,
imported, not copied -- the clips are physically identical in kind to every other
set, only the parameter source differs. No momentum or energy edit is ever
applied: every clip is an honest, simulated, elastic collision.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_scene as gs  # noqa: E402

logger = logging.getLogger("generate_ke_scene")


def simulate(p: dict, scratch_dir: str, render: bool):
  """Geometry, frame-0 check, simulation and physics check. Raises on any failure."""
  geo = gs.solve_geometry(p)
  reason = gs.check_frame0(p, geo)
  if reason:
    raise RuntimeError(f"frame0: {reason}")
  masses = gs.derive_masses(p)
  sim = gs.build_and_simulate(p, geo, masses, scratch_dir, render=render)
  scene, spheres, floor, camera, simulator, renderer, animation, collisions = sim
  pair_events, floor_events = gs.analyse_collisions(collisions, spheres, floor)
  reason, measurements = gs.check_simulation(p, geo, masses, animation, spheres,
                                             pair_events, floor_events)
  if reason:
    raise RuntimeError(f"simulation: {reason} {measurements}")
  return geo, masses, sim, pair_events, measurements


def restitution(v: dict) -> float:
  """Realised e along the TRUE normal, taken from dv_A (frictionless, so dv_A is along n).

  gs._measured_restitution uses the ANALYTIC normal, which is off by the solver's
  contact-point error and reads e = 0.997 on a collision that conserves KE to 1e-15
  (the trap recorded in CLAUDE.md §46a).
  """
  import numpy as np
  pre = {t: np.asarray(v[t][0], dtype=float) for t in ("a", "b")}
  post = {t: np.asarray(v[t][1], dtype=float) for t in ("a", "b")}
  n = post["a"] - pre["a"]
  n /= np.linalg.norm(n)
  return float(abs(np.dot(post["a"] - post["b"], n)) / abs(np.dot(pre["a"] - pre["b"], n)))


def physics_check(p: dict) -> dict:
  """Physics-only replay for the planner: realised e, KE, momentum error, exit frames."""
  import gc
  # No pb.disconnect(): kubric's _BulletClient disconnects in its own destructor and
  # PyBullet reissues client id 0 at once, so a manual disconnect kills the NEXT
  # check's live connection ("Not connected to physics server"). Let the scene go
  # out of scope and collect it instead (CLAUDE.md §25).
  gc.collect()
  scratch = tempfile.mkdtemp(prefix="ke_check_")
  try:
    geo, masses, sim, _, meas = simulate(p, scratch, render=False)
    _, spheres, _, _, _, _, animation, _ = sim
    v = {t: [animation[spheres[t]]["velocity"][f] for f in (gs.PRE_FRAME, gs.POST_FRAME)]
         for t in ("a", "b")}
    exits = gs.exit_frames(animation, spheres)
    del sim, spheres
    ke = [sum(0.5 * masses[f"mass_{t}"] * float(sum(c * c for c in v[t][i])) for t in ("a", "b"))
          for i in (0, 1)]
    return {"ok": True, "e": restitution(v),
            "ke_pre": ke[0], "ke_post": ke[1], "momentum_rel_error": meas["momentum_rel_error"],
            "contact_frame": meas["contact_frame"], "exit_frames": exits}
  except Exception as exc:  # pylint: disable=broad-except
    return {"ok": False, "reason": str(exc)[:300]}
  finally:
    gc.collect()
    shutil.rmtree(scratch, ignore_errors=True)


def generate(clip: dict, out_dir: Path, render: bool) -> dict:
  import kubric as kb
  p = dict(clip["params"])
  scratch = tempfile.mkdtemp(prefix="ke_scene_")
  try:
    geo, masses, sim, pair_events, measurements = simulate(p, scratch, render=render)
    scene, spheres, _, camera, _, renderer, animation, _ = sim
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_stats, gpu_desc = None, "n/a (physics only)"
    if render:
      gpu_desc = gs.configure_gpu(renderer)
      frames = renderer.render(return_layers=("rgba", "segmentation"))
      seg_stats = gs.segmentation_stats(frames["segmentation"])   # see generate_scene: no adjust_segmentation_idxs
      gs.write_outputs(out_dir, frames["rgba"], frames["segmentation"])
    labels = gs.build_labels(p, geo, masses, scene, spheres, camera, animation, pair_events,
                             measurements, clip["seed"], clip["name"], seg_stats, gpu_desc)
    labels["kubric_version"] = kb.__version__
    labels["exit_frames"] = gs.exit_frames(animation, spheres)
    labels["ke_design"] = {k: clip[k] for k in ("family", "group", "level", "ke_factor",
                                                "speed_scale", "mass_scale", "fixed_level")}
    with open(out_dir / "labels.json", "w") as f:
      json.dump(labels, f, indent=1)
    (out_dir / "_COMPLETE").write_text(f"{clip['name']}\n")
    return labels
  finally:
    shutil.rmtree(scratch, ignore_errors=True)


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--plan", required=True)
  ap.add_argument("--clip", required=True)
  ap.add_argument("--output-dir", required=True)
  ap.add_argument("--no-render", action="store_true")
  args = ap.parse_args()
  logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
  clip = next(c for c in json.load(open(args.plan))["clips"] if c["name"] == args.clip)
  out_dir = Path(args.output_dir)
  try:
    generate(clip, out_dir, render=not args.no_render)
  except Exception:  # pylint: disable=broad-except
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_FAILED").write_text(traceback.format_exc())
    logger.error("clip %s failed:\n%s", args.clip, traceback.format_exc())
    return 1
  return 0


if __name__ == "__main__":
  sys.exit(main())
