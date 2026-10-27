#!/usr/bin/env python3
"""Validate a rendered sliding-friction set, before anyone probes it.

    python3 validate_friction.py --dataset-dir friction_set_v1

Six checks, each of which has caught a real defect in some version of this
project's datasets. Every one is cheap; the last one needs the pixels and is the
only reason this is not a pure label script.

  1. completeness      49 frames and 49 masks per scene, labels parse, no _FAILED
  2. physics           measured deceleration equals mu*g, entry lands on frame 20,
                       the approach really is frictionless
  3. coverage          mu spans its range log-uniformly; strata as requested
  4. confounds         mu is uncorrelated with speed and with radius. The momentum
                       set had to be rebuilt twice over exactly this, so it is
                       checked rather than argued: here the sampler rejects
                       nothing, which is what makes independence structural
  5. identifiability   the post-entry path deviation, in pixels, against the
                       tracker's ~0.1 px centroid noise. A scene below that floor
                       cannot be read by ANY observer and would be a null that
                       says nothing about a model (CLAUDE.md §14)
  6. appearance leak   THE ONE THAT MATTERS FOR THE ARMS. In `latent_friction`
                       every floor tile is painted one albedo, so the boundary
                       and the patches must be invisible in the pixels. This
                       measures the rendered luma difference across the boundary
                       and its correlation with mu. In `visible_friction` the
                       same statistic is the positive control and must be large

Exit code is non-zero if any check fails, so this can gate a render.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FRAME_COUNT = 49
TRACKER_NOISE_PX = 0.1
LEAK_LUMA_TOLERANCE = 1.0      # 0-255 units, mean |patch - approach| in the latent arm
LEAK_CORR_TOLERANCE = 0.20     # |corr(luma step, log mu)| in the latent arm


def load_set(root: Path) -> list[dict]:
  labels = []
  for d in sorted(root.glob("scene_*")):
    path = d / "labels.json"
    if not path.exists():
      continue
    lab = json.loads(path.read_text())
    lab["_dir"] = str(d)
    labels.append(lab)
  return labels


def check_completeness(root: Path, labels: list[dict]) -> tuple[bool, list[str]]:
  msgs, ok = [], True
  failed = sorted(root.glob("scene_*/_FAILED"))
  if failed:
    ok = False
    msgs.append(f"  {len(failed)} scene(s) marked _FAILED: "
                f"{', '.join(p.parent.name for p in failed[:5])}")
  bad = []
  for lab in labels:
    d = Path(lab["_dir"])
    n_frames = len(list((d / "frames").glob("frame_*.png")))
    n_masks = len(list((d / "segmentation").glob("segmentation_*.png")))
    if not (d / "_COMPLETE").exists() or n_frames != FRAME_COUNT or n_masks != FRAME_COUNT:
      bad.append(f"{d.name}({n_frames}f/{n_masks}m)")
  if bad:
    ok = False
    msgs.append(f"  {len(bad)} incomplete: {', '.join(bad[:5])}")
  msgs.insert(0, f"  {len(labels)} scene(s) with labels, "
                 f"{len(labels) - len(bad)} complete with {FRAME_COUNT} frames + masks")
  return ok, msgs


def check_physics(labels: list[dict]) -> tuple[bool, list[str]]:
  rel = np.array([[l["friction"]["decel_rel_error_A"], l["friction"]["decel_rel_error_B"]]
                  for l in labels]).ravel()
  entry = np.array([l["collision"]["actual_frame"] for l in labels])
  target = labels[0]["scene_constants"]["target_entry_frame"]
  # The pre-entry phase must be a straight line at constant speed, which is what
  # makes the phase split meaningful: |v| at frame 18 must equal the sampled speed.
  coast = np.array([abs(l["collision"]["pre"]["speed_A"] - l["objects"]["A"]["speed_initial"])
                    / l["objects"]["A"]["speed_initial"] for l in labels])
  ok = bool(rel.max() < 0.05 and np.abs(entry - target).max() < 1.0 and coast.max() < 0.01)
  return ok, [
      f"  deceleration vs mu*g: median rel err {np.median(rel):.2e}, max {rel.max():.2e}",
      f"  entry frame: mean {entry.mean():.2f}, max |off target| "
      f"{np.abs(entry - target).max():.2f} frames (target {target})",
      f"  frictionless approach: max speed drift before entry {coast.max():.2e}",
  ]


def check_coverage(labels: list[dict]) -> tuple[bool, list[str]]:
  mu = np.array([[l["friction"]["mu_A"], l["friction"]["mu_B"]] for l in labels])
  lo, hi = labels[0]["scene_constants"]["mu_range"]
  strata = {}
  for l in labels:
    strata[l["stratum"]] = strata.get(l["stratum"], 0) + 1
  q = np.quantile(np.log(mu.ravel()), [0, 0.25, 0.5, 0.75, 1])
  # Log-uniform means the log quartiles sit at the range's own quartiles.
  want = np.log(lo) + np.array([0, .25, .5, .75, 1]) * (np.log(hi) - np.log(lo))
  ok = bool(mu.min() >= lo * 0.99 and mu.max() <= hi * 1.01
            and np.abs(q - want).max() < 0.35)
  return ok, [
      f"  mu: {mu.min():.3f} to {mu.max():.3f} (range {lo}-{hi}), "
      f"log quartiles off ideal by <= {np.abs(q - want).max():.2f}",
      f"  strata: {strata}",
      f"  stops within the clip: A {np.mean([l['friction']['stops_in_clip_A'] for l in labels]):.0%}, "
      f"B {np.mean([l['friction']['stops_in_clip_B'] for l in labels]):.0%}",
  ]


def check_confounds(labels: list[dict]) -> tuple[bool, list[str]]:
  mu = np.array([[l["friction"]["mu_A"], l["friction"]["mu_B"]] for l in labels]).ravel()
  speed = np.array([[l["objects"]["A"]["speed_initial"], l["objects"]["B"]["speed_initial"]]
                    for l in labels]).ravel()
  radius = np.array([[l["objects"]["A"]["radius"], l["objects"]["B"]["radius"]]
                     for l in labels]).ravel()
  c_speed = float(np.corrcoef(np.log(mu), speed)[0, 1])
  c_radius = float(np.corrcoef(np.log(mu), radius)[0, 1])
  split = [l for l in labels if l["stratum"] == "split_floor"]
  c_pair = float(np.corrcoef(
      np.log([l["friction"]["mu_A"] for l in split]),
      np.log([l["friction"]["mu_B"] for l in split]))[0, 1]) if len(split) > 8 else float("nan")
  # The threshold has to scale with n or the check is meaningless at both ends:
  # a sample correlation has sd ~ 1/sqrt(n) under H0, so a fixed 0.15 bar fails a
  # perfectly clean 6-scene pilot (where noise alone reaches 0.3-0.6) and is far
  # too loose to be worth passing at 300. Three sigma, floored at the effect size
  # actually worth caring about.
  tol = max(0.15, 3.0 / np.sqrt(len(mu)))
  tol_pair = max(0.20, 3.0 / np.sqrt(max(len(split), 1)))
  ok = bool(abs(c_speed) < tol and abs(c_radius) < tol
            and (np.isnan(c_pair) or abs(c_pair) < tol_pair))
  return ok, [
      f"  corr(log mu, speed)  {c_speed:+.3f}   corr(log mu, radius) {c_radius:+.3f}"
      f"   (|corr| bar at this n: {tol:.2f})",
      f"  corr(log mu_A, log mu_B) within split_floor {c_pair:+.3f} "
      f"(bar {tol_pair:.2f}; the uniform_floor stratum is 1.0 by construction "
      f"and is excluded)",
  ]


def check_identifiability(labels: list[dict]) -> tuple[bool, list[str]]:
  dev = np.array([l["identifiability"]["min_path_deviation_px"] for l in labels])
  under = int((dev < 1.0).sum())
  ok = bool(dev.min() > TRACKER_NOISE_PX * 10)
  return ok, [
      f"  post-entry path deviation of the WEAKER ball: min {dev.min():.1f} px, "
      f"p10 {np.quantile(dev, 0.1):.1f}, median {np.median(dev):.1f}",
      f"  scenes under 1 px (unreadable by anyone): {under}/{len(dev)}",
  ]


def _floor_luma_step(lab: dict) -> tuple[float, float]:
  """Mean rendered luma on the patch side minus the approach side, per lane.

  Sampled from frame 0, where both balls are still far from the boundary, and
  restricted to floor pixels via the segmentation mask, so no ball or shadow edge
  enters the statistic. If the latent arm is doing its job this is zero to within
  render noise, in both lanes, whatever mu is.
  """
  import imageio.v3 as iio

  d = Path(lab["_dir"])
  rgb = np.asarray(iio.imread(d / "frames" / "frame_0000.png"), dtype=np.float64)
  seg = np.asarray(iio.imread(d / "segmentation" / "segmentation_0000.png"))
  if seg.ndim == 3:
    seg = seg[..., 0]
  luma = rgb[..., :3] @ np.array([0.2126, 0.7152, 0.0722])

  sc = lab["scene_constants"]
  cam_z = sc["camera_position"][2]
  # Pixels per world unit in the FLOOR plane (z = 0), not the ball plane: the
  # camera is a perspective one, so the two scales differ by ~1.5% and using the
  # wrong one would walk the sample window off the boundary.
  px = sc["resolution"][0] / (cam_z * sc["sensor_width"] / sc["focal_length"])
  to_px = lambda wx, wy: (sc["resolution"][0] / 2 + wx * px,  # noqa: E731
                          sc["resolution"][1] / 2 - wy * px)

  xb = lab["floor"]["boundary_x"]
  out = []
  for lane_sign in (+1, -1):
    y0, y1 = sorted([lane_sign * 0.7, lane_sign * 3.6])
    sides = []
    for x0, x1 in ((xb - 2.0, xb - 0.35), (xb + 0.35, xb + 2.0)):
      (px0, py1), (px1, py0) = to_px(x0, y0), to_px(x1, y1)
      sl = (slice(int(py0), int(py1)), slice(int(px0), int(px1)))
      window, mask = luma[sl], seg[sl]
      sides.append(float(window[mask == 0].mean()))
    out.append(sides[1] - sides[0])
  return out[0], out[1]


def check_appearance_leak(labels: list[dict]) -> tuple[bool, list[str]]:
  steps = np.array([_floor_luma_step(l) for l in labels])          # [n, 2] A, B
  mu = np.array([[l["friction"]["mu_A"], l["friction"]["mu_B"]] for l in labels])
  arm = labels[0]["arm"]
  corr = [float(np.corrcoef(steps[:, k], np.log(mu[:, k]))[0, 1]) for k in (0, 1)]
  mean_abs = float(np.abs(steps).mean())

  # Correlation alone is the wrong statistic here and will cry wolf on a clean
  # set: when the step is 0.1 of 255 luma units, its correlation with anything is
  # a coin flip, and a 6-scene pilot duly produced -0.61. What a probe could
  # actually exploit is the SIZE of the mu-driven part, so the gate is the fitted
  # luma swing across the whole mu range, with the correlation kept as a
  # diagnostic beside it.
  span = np.log(mu.max()) - np.log(mu.min())
  swing = [abs(float(np.polyfit(np.log(mu[:, k]), steps[:, k], 1)[0]) * span) for k in (0, 1)]

  msgs = [
      f"  arm: {arm}",
      f"  luma step across the boundary (0-255): mean |step| {mean_abs:.2f}, "
      f"max |step| {np.abs(steps).max():.2f}",
      f"  mu-driven swing across the full mu range: lane A {swing[0]:.2f} luma, "
      f"lane B {swing[1]:.2f} luma",
      f"  corr(luma step, log mu), diagnostic only: A {corr[0]:+.3f}, B {corr[1]:+.3f}",
  ]
  if arm == "latent_friction":
    ok = bool(mean_abs < LEAK_LUMA_TOLERANCE and max(swing) < LEAK_LUMA_TOLERANCE)
    msgs.append("  latent arm: mu never reaches the renderer by construction "
                "(`patch_appearance` ignores it), so this measures the tile SEAM "
                "and any shadow spill — both must be invisible")
  else:
    ok = bool(mean_abs > 2.0 and max(abs(c) for c in corr) > 0.5)
    msgs.append("  visible arm: this is the positive control — the cue must be LARGE")
  return ok, msgs


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--dataset-dir", required=True)
  ap.add_argument("--skip-pixels", action="store_true",
                  help="skip check 6, which is the only one that reads frames")
  ap.add_argument("--out", default=None, help="write the report as JSON here")
  args = ap.parse_args(argv)

  root = Path(args.dataset_dir)
  labels = load_set(root)
  if not labels:
    print(f"no scenes with labels under {root}")
    return 1

  checks = [
      ("completeness", lambda: check_completeness(root, labels)),
      ("physics", lambda: check_physics(labels)),
      ("coverage", lambda: check_coverage(labels)),
      ("confounds", lambda: check_confounds(labels)),
      ("identifiability", lambda: check_identifiability(labels)),
  ]
  if not args.skip_pixels:
    checks.append(("appearance leak", lambda: check_appearance_leak(labels)))

  report, all_ok = {}, True
  print(f"validating {root} — {len(labels)} scene(s), arm {labels[0]['arm']}\n")
  for name, fn in checks:
    ok, msgs = fn()
    all_ok &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    for m in msgs:
      print(m)
    print()
    report[name] = {"pass": bool(ok), "detail": msgs}

  print("ALL CHECKS PASS" if all_ok else "SOME CHECKS FAILED")
  if args.out:
    Path(args.out).write_text(json.dumps(
        {"dataset_dir": str(root), "num_scenes": len(labels),
         "arm": labels[0]["arm"], "all_pass": bool(all_ok), "checks": report}, indent=1))
    print(f"-> {args.out}")
  return 0 if all_ok else 1


if __name__ == "__main__":
  sys.exit(main())
