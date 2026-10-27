#!/usr/bin/env python3
"""Validate a generated pilot set: correctness checks + parameter coverage.

    python3 validate.py --dataset-dir pilot --output-dir pilot/_validation

Reads only labels.json and the segmentation PNGs, so it runs outside the kubric
container. Writes validation_report.json, validation_plots.png, and prints a
PASS/FAIL summary. Exit code is non-zero if any hard check fails.

The hard checks:
  contact_frame     every scene within +/-1 frame of the target
  momentum          |P_post - P_pre| / |P_pre| at solver precision. With zero friction
                    and nothing else touching the spheres this is the check that says
                    the simulation is actually doing what we think it is.
  energy            kinetic energy never increases across the collision
  frame0            both spheres fully inside the frame and not overlapping at frame 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Thresholds for the hard checks.
CONTACT_TOL = 1.0
MOMENTUM_TOL = 1e-9      # float64 solver; observed values are ~1e-16
ENERGY_TOL = 1e-9        # KE ratio must not exceed 1 by more than this

# dataviz reference palette, categorical slots 1-3 (documented as passing all-pairs
# validation in both modes) plus the fixed status colours.
C_SERIES1 = "#2a78d6"
C_SERIES2 = "#eb6834"
C_SERIES3 = "#1baf7a"
C_CRITICAL = "#d03b3b"
C_GOOD = "#0ca30c"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dcdcd8"


def load_scenes(dataset_dir: Path):
  scenes = []
  for labels_path in sorted(dataset_dir.glob("scene_*/labels.json")):
    with open(labels_path) as f:
      labels = json.load(f)
    labels["_dir"] = labels_path.parent
    scenes.append(labels)
  return scenes


def segmentation_agreement(scene, frames=(0, 20, 48)):
  """Compare the projected 3D centre against the segmentation-mask centroid.

  This is the independent check on the 2D pixel labels: the labels come from the
  camera projection of the simulator's 3D position, the mask comes from the renderer.
  They are not expected to agree exactly (see README on silhouette vs centre under a
  perspective camera) -- the point is to measure the gap, not assume it is zero.
  """
  out = []
  for frame in frames:
    entry = scene["per_frame"][frame]
    for tag in ("A", "B"):
      centroid = entry[tag].get("segmentation_centroid_pixel")
      if centroid is None or not entry[tag]["in_frame"]:
        continue
      dx = centroid[0] - entry[tag]["pixel"][0]
      dy = centroid[1] - entry[tag]["pixel"][1]
      out.append({"frame": frame, "object": tag, "dx": dx, "dy": dy,
                  "distance": float(np.hypot(dx, dy))})
  return out


def analyse(scenes):
  rows = []
  for scene in scenes:
    collision = scene["collision"]
    frame0 = scene["per_frame"][0]
    half_w, half_h = scene["scene_constants"]["visible_half_extents"]
    resolution = scene["scene_constants"]["resolution"]

    # Frame-0 geometry, in pixels, from the labels themselves.
    pos_a = np.array(frame0["A"]["position"])
    pos_b = np.array(frame0["B"]["position"])
    surface_gap = float(np.linalg.norm(pos_a - pos_b)
                        - scene["objects"]["A"]["radius"] - scene["objects"]["B"]["radius"])
    px_a, px_b = np.array(frame0["A"]["pixel"]), np.array(frame0["B"]["pixel"])
    px_per_unit = resolution[0] / (2 * half_w)
    radius_px = {t: scene["objects"][t]["radius"] * px_per_unit for t in ("A", "B")}
    fully_in_frame = all(
        radius_px[t] <= p[0] <= resolution[0] - radius_px[t] and
        radius_px[t] <= p[1] <= resolution[1] - radius_px[t]
        for t, p in (("A", px_a), ("B", px_b)))
    seg_counts_frame0 = {t: frame0[t].get("segmentation_pixel_count") for t in ("A", "B")}

    in_frame_fraction = {
        t: float(np.mean([f[t]["in_frame"] for f in scene["per_frame"]])) for t in ("A", "B")}

    params = scene["sampled_parameters"]
    rows.append({
        "scene_id": scene["scene_id"],
        "seed": scene["seed"],
        "contact_frame": collision["actual_frame"],
        "contact_discrepancy": collision["discrepancy_frames"],
        "momentum_rel_error": collision["momentum_rel_error"],
        "momentum_abs_error": collision["momentum_abs_error"],
        "ke_ratio": collision["kinetic_energy_ratio"],
        "ke_lost": collision["kinetic_energy_lost"],
        "restitution": scene["restitution"],
        "restitution_measured": collision["restitution_measured"],
        "impulse_magnitude": collision["impulse_magnitude"],
        "mass_ratio": scene["mass_ratio"],
        "mass_a": scene["objects"]["A"]["mass"], "mass_b": scene["objects"]["B"]["mass"],
        "radius_a": params["radius_a"], "radius_b": params["radius_b"],
        "material_a": params["material_a"], "material_b": params["material_b"],
        "speed_a": params["speed_a"], "speed_b": params.get("speed_b", 0.0),
        "b_moving": bool(params.get("b_moving", False)),
        "stratum": scene.get("stratum", "wide"),
        "violates_momentum": bool(scene.get("violates_momentum", False)),
        "momentum_total_pre": float(np.linalg.norm(collision["pre"]["total_momentum"])),
        "momentum_total_post": float(np.linalg.norm(collision["post"]["total_momentum"])),
        "momentum_a_pre": float(np.linalg.norm(collision["pre"]["momentum_A"])),
        "direction_a_rad": params["direction_a_rad"],
        "impact_parameter_frac": params["impact_parameter_frac"],
        "frame0_surface_gap": surface_gap,
        "frame0_fully_in_frame": fully_in_frame,
        "frame0_seg_pixels_a": seg_counts_frame0["A"],
        "frame0_seg_pixels_b": seg_counts_frame0["B"],
        "in_frame_fraction_a": in_frame_fraction["A"],
        "in_frame_fraction_b": in_frame_fraction["B"],
        "num_rejections": len(scene.get("sampling_attempts", {}).get("rejections", [])),
        "seg_agreement": segmentation_agreement(scene),
    })
  return rows


def confound_report(rows):
  """Correlations between candidate probe targets, per stratum.

  This is the check that decides whether a probing result will *mean* anything.
  Momentum is mass x speed, so if the two factors are strongly correlated across
  the dataset then a probe scoring well on momentum cannot be distinguished from
  one reading apparent size and colour, and no amount of held-out accuracy fixes
  that. Correlations are taken in log space because every one of these quantities
  is sampled log-uniformly and spans more than a decade.
  """
  keys = ["momentum_total_pre", "momentum_a_pre", "mass_a", "speed_a", "radius_a"]
  out = {}
  for name, subset in (
      ("all", rows),
      ("matched_momentum", [r for r in rows if r["stratum"] == "matched_momentum"]),
      ("wide", [r for r in rows if r["stratum"] == "wide"]),
  ):
    if len(subset) < 3:
      continue
    mat = np.array([[np.log(max(r[k], 1e-12)) for k in keys] for r in subset])
    corr = np.corrcoef(mat.T)
    out[name] = {
        "n": len(subset),
        "pairs": {
            f"{keys[i]}~{keys[j]}": round(float(corr[i, j]), 3)
            for i in range(len(keys)) for j in range(i + 1, len(keys))
        },
    }
  return out


def summarise(rows):
  arr = lambda key: np.array([r[key] for r in rows], dtype=float)  # noqa: E731

  contact = arr("contact_frame")
  ke_ratio = arr("ke_ratio")
  gaps = arr("frame0_surface_gap")
  seg_distances = np.array([d["distance"] for r in rows for d in r["seg_agreement"]])
  restitution_err = np.array([abs(r["restitution_measured"] - r["restitution"]) for r in rows])

  # Violating scenes break conservation on purpose, so they are held to the
  # opposite standard: the conserving ones must conserve, and the violating ones
  # must actually violate. Pooling them would let a silently-inert violation
  # (or a broken simulation) pass as success.
  violates = np.array([r["violates_momentum"] for r in rows], dtype=bool)
  honest = np.array([r["momentum_rel_error"] for r in rows], dtype=float)[~violates]
  broken = np.array([r["momentum_rel_error"] for r in rows], dtype=float)[violates]

  checks = {
      "contact_frame": {
          "pass": bool(np.all(np.abs(contact - 20) <= CONTACT_TOL)),
          "detail": f"mean {contact.mean():.3f}, std {contact.std():.3f}, "
                    f"range [{contact.min():.3f}, {contact.max():.3f}], "
                    f"{int(np.sum(np.abs(contact - 20) <= CONTACT_TOL))}/{len(contact)} within +/-1",
      },
      "momentum_conservation": {
          "pass": bool(honest.size and np.all(honest <= MOMENTUM_TOL)),
          "detail": f"{honest.size} conserving scene(s): relative error median "
                    f"{np.median(honest) if honest.size else float('nan'):.2e}, "
                    f"max {honest.max() if honest.size else float('nan'):.2e} "
                    f"(tolerance {MOMENTUM_TOL:.0e})",
      },
      "violations_actually_violate": {
          "pass": bool(broken.size == 0 or np.all(broken > 0.05)),
          "detail": (f"{broken.size} violating scene(s): |dp|/|p| median "
                     f"{np.median(broken):.3f}, min {broken.min():.3f}"
                     if broken.size else "no violating scenes in this set"),
      },
      "energy_never_increases": {
          # Only meaningful for honest collisions; an injected impulse does work.
          "pass": bool(np.all(ke_ratio[~violates] <= 1.0 + ENERGY_TOL)),
          "detail": f"conserving scenes: KE_post/KE_pre max "
                    f"{ke_ratio[~violates].max() if (~violates).any() else float('nan'):.6f}, "
                    f"min {ke_ratio[~violates].min() if (~violates).any() else float('nan'):.4f}, "
                    f"{int(np.sum(ke_ratio[~violates] > 1.0 + ENERGY_TOL))} with an increase",
      },
      "frame0_visible_and_separate": {
          # Mask pixel counts only exist for rendered scenes; a --no-render batch
          # (used to check coverage and confounds before committing to a render)
          # still validates its geometry, just without the mask half.
          "pass": bool(all(r["frame0_fully_in_frame"] for r in rows) and np.all(gaps > 0) and
                       all(r["frame0_seg_pixels_a"] is None or
                           (r["frame0_seg_pixels_a"] and r["frame0_seg_pixels_b"])
                           for r in rows)),
          "detail": f"all fully in frame: {all(r['frame0_fully_in_frame'] for r in rows)}; "
                    f"min surface gap {gaps.min():.3f} world units; "
                    + (f"min mask pixels {min(min(r['frame0_seg_pixels_a'], r['frame0_seg_pixels_b']) for r in rows)}"
                       if all(r["frame0_seg_pixels_a"] is not None for r in rows)
                       else "no masks (physics-only batch)"),
      },
      "restitution_realised": {
          "pass": bool(np.all(restitution_err < 1e-3)),
          "detail": f"|measured - target| max {restitution_err.max():.2e}",
      },
  }

  extra = {
      "num_scenes": len(rows),
      "segmentation_vs_projection_px": {
          "median": float(np.median(seg_distances)) if seg_distances.size else None,
          "p95": float(np.percentile(seg_distances, 95)) if seg_distances.size else None,
          "max": float(seg_distances.max()) if seg_distances.size else None,
          "n_samples": int(seg_distances.size),
      },
      "in_frame_fraction": {
          "A_mean": float(arr("in_frame_fraction_a").mean()),
          "B_mean": float(arr("in_frame_fraction_b").mean()),
          "scenes_with_B_leaving": int(np.sum(arr("in_frame_fraction_b") < 1.0)),
          "scenes_with_A_leaving": int(np.sum(arr("in_frame_fraction_a") < 1.0)),
      },
      "rejections_per_scene": {
          "mean": float(arr("num_rejections").mean()), "max": int(arr("num_rejections").max()),
      },
      "ke_ratio": {"min": float(ke_ratio.min()), "median": float(np.median(ke_ratio)),
                   "max": float(ke_ratio.max())},
      "class_balance": {
          "violating": int(violates.sum()),
          "conserving": int((~violates).sum()),
          "b_moving": int(sum(r["b_moving"] for r in rows)),
          "matched_momentum_stratum": int(
              sum(r["stratum"] == "matched_momentum" for r in rows)),
      },
      "label_ranges": {
          key: {"min": float(arr(key).min()), "max": float(arr(key).max()),
                "span_x": float(arr(key).max() / max(arr(key).min(), 1e-12))}
          for key in ("mass_a", "speed_a", "momentum_total_pre", "mass_ratio")
      },
      "confounds": confound_report(rows),
  }
  return checks, extra


def style_axis(ax, title, xlabel=None, ylabel=None):
  ax.set_title(title, fontsize=9.5, color=INK, pad=7, loc="left")
  if xlabel:
    ax.set_xlabel(xlabel, fontsize=8, color=INK_MUTED)
  if ylabel:
    ax.set_ylabel(ylabel, fontsize=8, color=INK_MUTED)
  ax.tick_params(labelsize=7.5, colors=INK_MUTED, length=3, width=0.6)
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  for side in ("left", "bottom"):
    ax.spines[side].set_color(GRID)
    ax.spines[side].set_linewidth(0.8)
  ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
  ax.set_axisbelow(True)


def make_plots(rows, scenes, out_path: Path):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  arr = lambda key: np.array([r[key] for r in rows], dtype=float)  # noqa: E731
  n = len(rows)

  fig = plt.figure(figsize=(15, 15.5), facecolor=SURFACE)
  gs = fig.add_gridspec(4, 3, hspace=0.42, wspace=0.26,
                        left=0.055, right=0.975, top=0.925, bottom=0.05)
  fig.suptitle(f"Two-body collision pilot — {n} scenes", x=0.055, y=0.972,
               ha="left", fontsize=14, color=INK)
  fig.text(0.055, 0.951, "correctness checks (row 1) and parameter coverage (rows 2-4)",
           ha="left", fontsize=9.5, color=INK_MUTED)

  # --- Row 1: correctness -----------------------------------------------------------
  ax = fig.add_subplot(gs[0, 0])
  contact = arr("contact_frame")
  ax.hist(contact, bins=np.arange(19.0, 21.05, 0.05), color=C_SERIES1, edgecolor="none")
  ax.axvline(20, color=INK_MUTED, linewidth=1.2, linestyle="--")
  ax.axvspan(19, 21, color=C_GOOD, alpha=0.07)
  ax.annotate(f"all {n} within ±1\nspread {contact.min():.2f}–{contact.max():.2f}",
              xy=(0.97, 0.93), xycoords="axes fraction", ha="right", va="top",
              fontsize=8, color=INK_MUTED)
  style_axis(ax, "Contact frame (target 20)", "frame index of first contact", "scenes")

  ax = fig.add_subplot(gs[0, 1])
  momentum = arr("momentum_rel_error")
  floor = 1e-18
  ax.hist(np.log10(np.maximum(momentum, floor)), bins=24, color=C_SERIES1, edgecolor="none")
  ax.axvline(np.log10(MOMENTUM_TOL), color=C_CRITICAL, linewidth=1.2, linestyle="--")
  ax.annotate(f"tolerance 1e-9\nmax {momentum.max():.1e}\nexact zeros clamped to left edge",
              xy=(0.97, 0.93),
              xycoords="axes fraction", ha="right", va="top", fontsize=8, color=INK_MUTED)
  style_axis(ax, "Total momentum: relative error across collision",
             "log₁₀ |ΔP| / |P_pre|", "scenes")

  ax = fig.add_subplot(gs[0, 2])
  ke = arr("ke_ratio")
  restitution = arr("restitution")
  for level, color in zip(sorted(set(restitution)), (C_SERIES1, C_SERIES2, C_SERIES3)):
    mask = restitution == level
    ax.scatter(arr("mass_ratio")[mask], ke[mask], s=26, color=color, alpha=0.85,
               edgecolor=SURFACE, linewidth=0.7, label=f"e = {level:g}")
  ax.axhline(1.0, color=C_CRITICAL, linewidth=1.0, linestyle="--")
  ax.set_xscale("log")
  ax.legend(fontsize=7.5, frameon=False, labelcolor=INK_MUTED, loc="lower left")
  style_axis(ax, "Kinetic energy ratio (must never exceed 1)",
             "mass ratio m_A / m_B (log)", "KE_post / KE_pre")

  # --- Row 2: continuous sampled parameters -----------------------------------------
  ax = fig.add_subplot(gs[1, 0])
  ax.hist(arr("speed_a"), bins=12, color=C_SERIES1, edgecolor="none")
  style_axis(ax, "Coverage: speed of A", "world units / s", "scenes")

  ax = fig.add_subplot(gs[1, 1], projection="polar")
  theta = arr("direction_a_rad")
  counts, edges = np.histogram(theta, bins=12, range=(0, 2 * np.pi))
  ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
         color=C_SERIES1, edgecolor=SURFACE, linewidth=0.8)
  ax.set_title("Coverage: direction of A (0–360°)", fontsize=9.5, color=INK, pad=12, loc="left")
  ax.tick_params(labelsize=7, colors=INK_MUTED)
  ax.grid(color=GRID, linewidth=0.6)

  ax = fig.add_subplot(gs[1, 2])
  ax.hist(arr("impact_parameter_frac"), bins=12, range=(0, 0.9),
          color=C_SERIES1, edgecolor="none")
  style_axis(ax, "Coverage: impact parameter (0 = head-on)", "|b| / (r_A + r_B)", "scenes")

  # --- Row 3: sampled parameter coverage --------------------------------------------
  ax = fig.add_subplot(gs[2, 0])
  # Radii are continuous now, so this is a density rather than a level count.
  bins = np.linspace(min(arr("radius_a").min(), arr("radius_b").min()),
                     max(arr("radius_a").max(), arr("radius_b").max()), 14)
  for key, color, label in (("radius_a", C_SERIES1, "sphere A"),
                            ("radius_b", C_SERIES2, "sphere B")):
    ax.hist(arr(key), bins=bins, color=color, edgecolor="none", alpha=0.65, label=label)
  ax.legend(fontsize=7.5, frameon=False, labelcolor=INK_MUTED)
  style_axis(ax, "Coverage: radius", "radius (world units)", "scenes")

  ax = fig.add_subplot(gs[2, 1])
  combos = {}
  for r in rows:
    combos[f"{r['material_a']}→{r['material_b']}"] = combos.get(
        f"{r['material_a']}→{r['material_b']}", 0) + 1
  keys = sorted(combos)
  bars = ax.bar(range(len(keys)), [combos[k] for k in keys], 0.62,
                color=C_SERIES1, edgecolor="none")
  ax.bar_label(bars, fontsize=7, color=INK_MUTED, padding=2)
  ax.set_xticks(range(len(keys)), keys, fontsize=7.5)
  style_axis(ax, "Coverage: material pairing (A→B)", None, "scenes")

  ax = fig.add_subplot(gs[2, 2])
  # Restitution is sampled continuously, so this is a density and not a level count.
  ax.hist(restitution, bins=16, color=C_SERIES1, edgecolor="none")
  style_axis(ax, "Coverage: restitution", "pair restitution e", "scenes")

  # --- Row 4: derived quantities and label diagnostics ------------------------------
  ax = fig.add_subplot(gs[3, 0])
  ax.hist(np.log10(arr("mass_ratio")), bins=14, color=C_SERIES1, edgecolor="none")
  style_axis(ax, "Coverage: mass ratio", "log₁₀ (m_A / m_B)", "scenes")

  ax = fig.add_subplot(gs[3, 1])
  distances = np.array([d["distance"] for r in rows for d in r["seg_agreement"]])
  if distances.size:
    ax.hist(distances, bins=20, color=C_SERIES3, edgecolor="none")
    ax.annotate(
        f"median {np.median(distances):.2f} px\np95 {np.percentile(distances, 95):.2f} px",
        xy=(0.97, 0.93), xycoords="axes fraction", ha="right", va="top",
        fontsize=8, color=INK_MUTED)
    style_axis(ax, "2D label check: projected centre vs mask centroid",
               "pixel distance", "object-frames")
  else:
    # Physics-only batch: no masks to compare against, so spend the panel on the
    # thing such a batch is actually run to answer.
    for stratum, color in (("wide", C_SERIES1), ("matched_momentum", C_SERIES2)):
      sel = [r for r in rows if r["stratum"] == stratum]
      if sel:
        ax.scatter([np.log10(r["mass_a"]) for r in sel],
                   [np.log10(r["momentum_total_pre"]) for r in sel],
                   s=7, alpha=0.6, color=color, edgecolor="none", label=stratum)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK_MUTED)
    style_axis(ax, "Confound check: momentum vs mass",
               "log₁₀ mass_A", "log₁₀ |p_total| pre")

  ax = fig.add_subplot(gs[3, 2])
  num_frames = len(scenes[0]["per_frame"])
  frames = np.arange(num_frames)
  # B is dashed so the two stay distinguishable where they coincide at 1.0.
  for key, color, label, style in (("A", C_SERIES1, "sphere A", "-"),
                                   ("B", C_SERIES2, "sphere B", "--")):
    fraction = [np.mean([1.0 if s["per_frame"][f][key]["in_frame"] else 0.0 for s in scenes])
                for f in range(num_frames)]
    ax.plot(frames, fraction, color=color, linewidth=2.0, linestyle=style, label=label)
  ax.axvline(20, color=INK_MUTED, linewidth=1.0, linestyle="--")
  ax.set_ylim(0, 1.04)
  ax.legend(fontsize=7.5, frameon=False, labelcolor=INK_MUTED, loc="lower left")
  style_axis(ax, "Fraction of scenes with the object in frame", "frame index", "fraction")

  fig.savefig(out_path, dpi=110, facecolor=SURFACE)
  plt.close(fig)


SCENES = []  # populated in main()


def main():
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--dataset-dir", type=str, required=True)
  parser.add_argument("--output-dir", type=str, default=None)
  args = parser.parse_args()

  dataset_dir = Path(args.dataset_dir)
  out_dir = Path(args.output_dir) if args.output_dir else dataset_dir / "_validation"
  out_dir.mkdir(parents=True, exist_ok=True)

  global SCENES  # pylint: disable=global-statement
  SCENES = load_scenes(dataset_dir)
  if not SCENES:
    print(f"no scenes found in {dataset_dir}", file=sys.stderr)
    return 2

  rows = analyse(SCENES)
  checks, extra = summarise(rows)
  make_plots(rows, SCENES, out_dir / "validation_plots.png")

  report = {"checks": checks, "summary": extra, "per_scene": rows}
  (out_dir / "validation_report.json").write_text(json.dumps(report, indent=1, default=str))

  print(f"\n{len(rows)} scenes from {dataset_dir}\n")
  width = max(len(k) for k in checks)
  for name, result in checks.items():
    print(f"  {'PASS' if result['pass'] else 'FAIL'}  {name:<{width}}  {result['detail']}")
  seg = extra["segmentation_vs_projection_px"]
  if seg["median"] is not None:
    print(f"\n  2D labels: projected centre vs segmentation centroid — "
          f"median {seg['median']:.2f} px, p95 {seg['p95']:.2f} px, max {seg['max']:.2f} px")
  else:
    print("\n  2D labels: not checked (physics-only batch, no masks rendered)")
  print(f"  in-frame: A {extra['in_frame_fraction']['A_mean']*100:.1f}% of frames, "
        f"B {extra['in_frame_fraction']['B_mean']*100:.1f}%; "
        f"B leaves frame in {extra['in_frame_fraction']['scenes_with_B_leaving']} scenes, "
        f"A in {extra['in_frame_fraction']['scenes_with_A_leaving']}")
  print(f"  resampling: {extra['rejections_per_scene']['mean']:.2f} rejected draws per "
        f"accepted scene (max {extra['rejections_per_scene']['max']})")

  bal = extra["class_balance"]
  print(f"\n  composition: {bal['conserving']} conserving / {bal['violating']} violating, "
        f"{bal['b_moving']} with B already moving, "
        f"{bal['matched_momentum_stratum']} in the matched-momentum stratum")
  print("  label span: " + ", ".join(
      f"{k} {v['span_x']:.0f}x" for k, v in extra["label_ranges"].items()))

  # The confound table is the one that decides whether a probing result on this
  # dataset can be interpreted, so it is printed rather than buried in the JSON.
  print("\n  confounds (log-space correlation; momentum~mass is the one that matters):")
  for stratum, block in extra["confounds"].items():
    pairs = block["pairs"]
    print(f"    {stratum:<18} n={block['n']:<4} "
          f"|p_tot|~mass_A {pairs['momentum_total_pre~mass_a']:+.2f}   "
          f"|p_A|~mass_A {pairs['momentum_a_pre~mass_a']:+.2f}   "
          f"|p_tot|~speed_A {pairs['momentum_total_pre~speed_a']:+.2f}")
  print("    a momentum probe is only identifiable where |p|~mass is small; fit and")
  print("    report on the matched-momentum stratum, use the rest for coverage.")

  print(f"\n  wrote {out_dir/'validation_report.json'} and {out_dir/'validation_plots.png'}\n")

  return 0 if all(c["pass"] for c in checks.values()) else 1


if __name__ == "__main__":
  sys.exit(main())
