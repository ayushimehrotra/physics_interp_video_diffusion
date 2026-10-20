#!/usr/bin/env python3
"""Validate a rendered sliding-cube set, before anyone probes it.

    python3 validate_slide.py --dataset-dir slide_set_v1

Six checks. Exit code is non-zero if any fails, so this can gate a render.

  1. completeness    49 frames and 49 masks per scene, labels parse, no _FAILED
  2. physics         measured deceleration equals mu*g; the cube does not tumble,
                     leave the floor, or veer sideways
  3. coverage        mu spans its range log-uniformly; the mix of scenes that come
                     to rest and scenes still moving at frame 48
  4. confounds       mu is uncorrelated with launch speed and with cube size.
                     Deceleration is mass- and size-independent here, so size is a
                     free control — but only if the sampler kept it independent
  5. identifiability the post-launch departure from a constant-velocity path, in
                     pixels, against the tracker's ~0.1 px centroid noise
  6. appearance leak THE ONE THAT GATES THE ARM. In `latent_friction` the floor is
                     one albedo whatever its mu, so the rendered floor must carry
                     NO trace of the coefficient. Measured as the correlation
                     between floor luma and mu across scenes, plus the luma swing
                     across the full mu range. In `visible_friction` the same
                     statistic is the positive control and must be large.

Check 6 is the only one that reads pixels; everything else is labels only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FRAME_COUNT = 49
TRACKER_NOISE_PX = 0.1
LEAK_LUMA_TOLERANCE = 1.0      # 0-255 units, mu-driven swing in the latent arm


def load_set(root: Path) -> list[dict]:
    out = []
    for d in sorted(root.glob("scene_*")):
        p = d / "labels.json"
        if p.exists():
            lab = json.loads(p.read_text())
            lab["_dir"] = str(d)
            out.append(lab)
    return out


def check_completeness(root: Path, labels: list[dict]):
    msgs, ok = [], True
    failed = sorted(root.glob("scene_*/_FAILED"))
    if failed:
        ok = False
        msgs.append(f"  {len(failed)} scene(s) marked _FAILED: "
                    f"{', '.join(p.parent.name for p in failed[:5])}")
    bad = []
    for lab in labels:
        d = Path(lab["_dir"])
        nf = len(list((d / "frames").glob("frame_*.png")))
        nm = len(list((d / "segmentation").glob("segmentation_*.png")))
        if not (d / "_COMPLETE").exists() or nf != FRAME_COUNT or nm != FRAME_COUNT:
            bad.append(f"{d.name}({nf}f/{nm}m)")
    if bad:
        ok = False
        msgs.append(f"  {len(bad)} incomplete: {', '.join(bad[:5])}")
    msgs.insert(0, f"  {len(labels)} scene(s) with labels, {len(labels)-len(bad)} complete")
    return ok, msgs


def check_physics(labels: list[dict]):
    rel = np.array([l["friction"]["decel_rel_error"] for l in labels])
    res = np.array([l["friction"]["fit_max_residual"] for l in labels])
    ok = bool(rel.max() < 0.05)
    return ok, [
        f"  deceleration vs mu*g: median rel err {np.median(rel):.2e}, max {rel.max():.2e}",
        f"  straight-line fit residual on the speed track: max {res.max():.2e}",
        "  (tumbling, leaving the floor and sideways drift are rejected at generation "
        "time, so a scene present here already passed them)",
    ]


def check_coverage(labels: list[dict]):
    mu = np.array([l["friction"]["mu"] for l in labels])
    lo, hi = labels[0]["scene_constants"]["mu_range"]
    q = np.quantile(np.log(mu), [0, .25, .5, .75, 1])
    want = np.log(lo) + np.array([0, .25, .5, .75, 1]) * (np.log(hi) - np.log(lo))
    stops = np.mean([l["friction"]["stops_in_clip"] for l in labels])
    # Order statistics wander from the ideal quantiles as ~1/sqrt(n), so a fixed
    # bar flags a perfectly uniform 8-scene pilot while being slack at 300. Same
    # treatment as the confound correlations below.
    span = np.log(hi) - np.log(lo)
    tol = max(0.35, 1.2 * span / np.sqrt(len(labels)))
    ok = bool(mu.min() >= lo * 0.99 and mu.max() <= hi * 1.01
              and np.abs(q - want).max() < tol and 0.05 < stops < 0.98)
    return ok, [
        f"  mu: {mu.min():.3f} to {mu.max():.3f} (range {lo}-{hi}), "
        f"log quartiles off ideal by <= {np.abs(q-want).max():.2f} (bar at this n: {tol:.2f})",
        f"  comes to rest within the clip: {stops:.0%} of scenes",
    ]


def check_confounds(labels: list[dict]):
    mu = np.log([l["friction"]["mu"] for l in labels])
    v = np.array([l["objects"]["A"]["speed_initial"] for l in labels])
    size = np.array([l["objects"]["A"]["size"] for l in labels])
    cs, cz = float(np.corrcoef(mu, v)[0, 1]), float(np.corrcoef(mu, size)[0, 1])
    # Scales with n: a sample correlation has sd ~ 1/sqrt(n) under H0, so a fixed
    # bar either fails a clean pilot or is too loose to be worth passing at 300.
    tol = max(0.15, 3.0 / np.sqrt(len(labels)))
    ok = bool(abs(cs) < tol and abs(cz) < tol)
    return ok, [
        f"  corr(log mu, launch speed) {cs:+.3f}   corr(log mu, cube size) {cz:+.3f}"
        f"   (bar at this n: {tol:.2f})",
        "  nothing is rejected by the sampler, so these cannot drift through "
        "acceptance — the check is that the draw itself stayed independent",
    ]


def check_identifiability(labels: list[dict]):
    dev = np.array([l["identifiability"]["path_deviation_px"] for l in labels])
    ok = bool(dev.min() > TRACKER_NOISE_PX * 10)
    return ok, [
        f"  departure from a constant-velocity path: min {dev.min():.0f} px, "
        f"p10 {np.quantile(dev,0.1):.0f}, median {np.median(dev):.0f}",
        f"  scenes under 1 px (unreadable by anyone): {int((dev < 1).sum())}/{len(dev)}",
    ]


def _floor_luma(lab: dict) -> float:
    """Mean rendered luma of the floor at frame 0, cube and shadow excluded.

    Sampled from the floor's own segmentation id rather than a fixed rectangle, so
    the window follows the geometry instead of assuming it. The bottom rows are
    dropped: the cube's shadow reaches into the near foreground and would import a
    mu-independent but position-dependent term into the statistic.
    """
    import imageio.v3 as iio

    d = Path(lab["_dir"])
    rgb = np.asarray(iio.imread(d / "frames" / "frame_0000.png"), dtype=np.float64)
    seg = np.asarray(iio.imread(d / "segmentation" / "segmentation_0000.png"))
    if seg.ndim == 3:
        seg = seg[..., 0]
    luma = rgb[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
    floor_id = lab["objects"]["B"]["segmentation_id"]
    h = seg.shape[0]
    mask = (seg == floor_id)
    mask[int(0.75 * h):, :] = False            # drop the near foreground
    return float(luma[mask].mean())


def check_appearance_leak(labels: list[dict]):
    luma = np.array([_floor_luma(l) for l in labels])
    mu = np.log([l["friction"]["mu"] for l in labels])
    arm = labels[0]["arm"]
    corr = float(np.corrcoef(luma, mu)[0, 1])
    slope = float(np.polyfit(mu, luma, 1)[0])
    swing = abs(slope * (mu.max() - mu.min()))
    msgs = [
        f"  arm: {arm}",
        f"  floor luma (0-255): mean {luma.mean():.2f}, sd {luma.std():.3f}",
        f"  mu-driven swing across the full mu range: {swing:.3f} luma",
        f"  corr(floor luma, log mu): {corr:+.3f}",
    ]
    if arm == "latent_friction":
        ok = bool(swing < LEAK_LUMA_TOLERANCE)
        msgs.append("  latent arm: mu never reaches the renderer by construction "
                    "(the albedo is a constant), so a non-zero swing here means "
                    "something else is carrying it — lighting, shadow, or geometry")
    else:
        ok = bool(swing > 2.0 and abs(corr) > 0.5)
        msgs.append("  visible arm: the positive control — the cue must be LARGE")
    return ok, msgs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--skip-pixels", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    root = Path(args.dataset_dir)
    labels = load_set(root)
    if not labels:
        print(f"no scenes with labels under {root}")
        return 1

    checks = [("completeness", lambda: check_completeness(root, labels)),
              ("physics", lambda: check_physics(labels)),
              ("coverage", lambda: check_coverage(labels)),
              ("confounds", lambda: check_confounds(labels)),
              ("identifiability", lambda: check_identifiability(labels))]
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
            {"dataset_dir": str(root), "num_scenes": len(labels), "arm": labels[0]["arm"],
             "all_pass": bool(all_ok), "checks": report}, indent=1))
        print(f"-> {args.out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
