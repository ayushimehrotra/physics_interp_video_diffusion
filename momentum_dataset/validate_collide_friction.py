#!/usr/bin/env python3
"""Validate a rendered two-cube collision-on-friction set, before anyone probes it.

    python3 validate_collide_friction.py --dataset-dir collide_set_v1

Six checks, exit code non-zero if any fails, so this can gate a render. It is
`validate_slide.py` (§25) with the checks that a SECOND object and an INTERACTION
make necessary:

  1. completeness     49 frames and 49 masks per scene, labels parse, no _FAILED
  2. physics          measured deceleration equals mu*g for BOTH cubes in BOTH
                      phases (four numbers per scene, not one), and contact lands
                      on the target frame
  3. coverage         mu spans its range log-uniformly; both cubes still have
                      enough moving frames after contact to fit a slope through
  4. confounds        mu uncorrelated with either speed, with size, and with
                      restitution. Nothing is rejected by the sampler, so these
                      cannot drift in through acceptance — the check is that the
                      draw itself stayed independent
  5. identifiability  the two cubes are separately visible: both segmentation ids
                      present in every frame, and neither is swallowed by the
                      other at contact. Without this the object-local pools are
                      meaningless
  6. appearance leak  THE ONE THAT GATES THE ARM. In `latent_friction` the floor
                      is one albedo whatever its mu, so neither the floor NOR the
                      cubes may carry a trace of the coefficient

Checks 5 and 6 read pixels; everything else is labels only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FRAME_COUNT = 49
LEAK_LUMA_TOLERANCE = 1.0        # 0-255 units, mu-driven swing in the latent arm
DECEL_TOLERANCE = 0.08
CONTACT_TOLERANCE = 1.0
MIN_MOVING_FRAMES = 5


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


def _rel_errors(labels):
    """Four deceleration errors per scene: {a,b} x {pre,post}."""
    keys = [f"decel_measured_{k}_{p}" for k in "ab" for p in ("pre", "post")]
    rows = []
    for l in labels:
        f = l["friction"]
        rows.append([abs(f[k] - f["decel"]) / f["decel"] if f.get(k) else np.nan
                     for k in keys])
    return np.array(rows), keys


def check_physics(labels: list[dict]):
    rel, keys = _rel_errors(labels)
    cf = np.array([l["collision"]["actual_frame"] for l in labels])
    target = labels[0]["collision"]["target_frame"]
    miss = np.abs(cf - target)
    ok = bool(np.nanmax(rel) < DECEL_TOLERANCE and miss.max() <= CONTACT_TOLERANCE)
    per = "  ".join(f"{k.replace('decel_measured_','')}:{np.nanmedian(rel[:, i]):.3f}"
                    for i, k in enumerate(keys))
    return ok, [
        f"  deceleration vs mu*g, median rel err per cube/phase — {per}",
        f"  worst single value {np.nanmax(rel):.3f} (bar {DECEL_TOLERANCE})",
        f"  contact frame: {cf.min():.2f} to {cf.max():.2f}, target {target}, "
        f"max miss {miss.max():.2f}",
        "  (tumbling, leaving the floor and sideways drift are rejected at "
        "generation time, so a scene present here already passed them)",
    ]


def check_coverage(labels: list[dict]):
    mu = np.array([l["friction"]["mu"] for l in labels])
    lo, hi = labels[0]["scene_constants"]["mu_range"]
    q = np.quantile(np.log(mu), [0, .25, .5, .75, 1])
    want = np.log(lo) + np.array([0, .25, .5, .75, 1]) * (np.log(hi) - np.log(lo))
    span = np.log(hi) - np.log(lo)
    # Scales with n: order statistics wander as ~1/sqrt(n), so a bar tuned for 300
    # scenes fails a clean 6-scene pilot. Same treatment as §25.
    tol = max(0.35, 1.2 * span / np.sqrt(len(labels)))
    post = np.array([[l["friction"][f"n_moving_{k}_post"] or 0 for k in "ab"]
                     for l in labels])
    ok = bool(mu.min() >= lo * 0.99 and mu.max() <= hi * 1.01
              and np.abs(q - want).max() < tol and post.min() >= MIN_MOVING_FRAMES)
    return ok, [
        f"  mu: {mu.min():.3f} to {mu.max():.3f} (range {lo}-{hi}), "
        f"log quartiles off ideal by <= {np.abs(q-want).max():.2f} (bar at this n: {tol:.2f})",
        f"  post-contact moving frames: min {post.min()}, median {np.median(post):.0f} "
        f"(bar {MIN_MOVING_FRAMES}) — both cubes need a slope fit on BOTH sides",
    ]


def check_confounds(labels: list[dict]):
    mu = np.log([l["friction"]["mu"] for l in labels])
    cols = {
        "speed_A (rear)": [l["objects"]["A"]["speed_initial"] for l in labels],
        "speed_B (front)": [l["objects"]["B"]["speed_initial"] for l in labels],
        "cube size": [l["objects"]["A"]["size"] for l in labels],
        "restitution": [l["restitution"] for l in labels],
    }
    tol = max(0.15, 3.0 / np.sqrt(len(labels)))
    cs = {k: float(np.corrcoef(mu, v)[0, 1]) for k, v in cols.items()}
    nrej = np.array([len(l["sampling_attempts"]["rejections"]) for l in labels])
    rej_corr = (float(np.corrcoef(mu, nrej)[0, 1]) if nrej.std() > 0 else 0.0)
    ok = bool(max(abs(v) for v in cs.values()) < tol and abs(rej_corr) < tol)
    return ok, [
        "  " + "   ".join(f"corr(log mu, {k}) {v:+.3f}" for k, v in cs.items())
        + f"   (bar at this n: {tol:.2f})",
        f"  rejections: {nrej.sum()} total, corr(log mu, n_rejections) {rej_corr:+.3f}",
        "  the sampler is designed to reject nothing, so a non-zero total is the "
        "signal to re-check the ranges — acceptance must not depend on mu",
    ]


def check_identifiability(root: Path, labels: list[dict], max_scenes: int = 40):
    """Both cubes separately visible in every frame — the object-local pools depend
    on it, and a rear-end collision is exactly the geometry that can merge them."""
    import imageio.v3 as iio
    mins_a, mins_b, worst = [], [], []
    for lab in labels[:max_scenes]:
        d = Path(lab["_dir"])
        paths = sorted((d / "segmentation").glob("segmentation_*.png"))
        if not paths:
            return False, [f"  {d.name}: no segmentation/"]
        areas = []
        for p in paths:
            m = np.asarray(iio.imread(p))
            if m.ndim == 3:
                m = m[..., 0]
            areas.append([(m == 1).sum(), (m == 2).sum()])
        a = np.array(areas)
        mins_a.append(a[:, 0].min())
        mins_b.append(a[:, 1].min())
        if a.min() == 0:
            worst.append(d.name)
    ok = bool(min(mins_a) > 0 and min(mins_b) > 0)
    msgs = [
        f"  smallest per-frame pixel area, cube A: {min(mins_a)}, cube B: {min(mins_b)} "
        f"(over {len(mins_a)} scene(s))",
        "  both ids must be present in EVERY frame or the A/B pools are empty "
        "somewhere and probe_grid silently drops those scenes",
    ]
    if worst:
        msgs.append(f"  scenes where a cube vanishes: {', '.join(worst[:5])}")
    return ok, msgs


def check_appearance_leak(root: Path, labels: list[dict], max_scenes: int = 60):
    """In the latent arm neither the floor NOR the cubes may encode mu.

    The cubes are rendered identically to each other and to every other scene, so
    this is a second, independent way for the arm to fail — and unlike the floor
    it is not something §25 had to worry about, because there was only one cube.
    """
    import imageio.v3 as iio
    mu, floor_l, cube_l = [], [], []
    for lab in labels[:max_scenes]:
        d = Path(lab["_dir"])
        fr = sorted((d / "frames").glob("frame_*.png"))
        sg = sorted((d / "segmentation").glob("segmentation_*.png"))
        if not fr or not sg:
            return False, [f"  {d.name}: missing pixels"]
        img = np.asarray(iio.imread(fr[0])).astype(np.float64)
        m = np.asarray(iio.imread(sg[0]))
        if m.ndim == 3:
            m = m[..., 0]
        luma = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        if (m == 3).sum() == 0 or ((m == 1) | (m == 2)).sum() == 0:
            return False, [f"  {d.name}: floor or cube missing from segmentation"]
        mu.append(lab["friction"]["mu"])
        floor_l.append(luma[m == 3].mean())
        cube_l.append(luma[(m == 1) | (m == 2)].mean())

    mu = np.array(mu); floor_l = np.array(floor_l); cube_l = np.array(cube_l)
    arm = labels[0]["arm"]
    out = []
    ok = True
    for name, vals in (("floor", floor_l), ("cubes", cube_l)):
        if len(mu) > 2 and np.std(np.log(mu)) > 0 and vals.std() > 0:
            c = float(np.corrcoef(np.log(mu), vals)[0, 1])
            slope = np.polyfit(np.log(mu), vals, 1)[0]
            swing = abs(slope) * (np.log(mu).max() - np.log(mu).min())
        else:
            c, swing = 0.0, 0.0
        out.append(f"  {name}: luma mean {vals.mean():.2f}, sd {vals.std():.3f}, "
                   f"mu-driven swing {swing:.3f}, corr(luma, log mu) {c:+.3f}")
        if arm == "latent_friction" and swing > LEAK_LUMA_TOLERANCE:
            ok = False
    out.insert(0, f"  arm: {arm}")
    if arm == "latent_friction":
        out.append("  latent arm: mu never reaches the renderer by construction, so a "
                   "non-zero swing means something else carries it")
    return ok, out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--skip-pixels", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.dataset_dir)
    labels = load_set(root)
    if not labels:
        print(f"no scenes with labels under {root}")
        return 1
    print(f"validating {root} — {len(labels)} scene(s), arm {labels[0]['arm']}\n")

    checks = [
        ("completeness", lambda: check_completeness(root, labels)),
        ("physics", lambda: check_physics(labels)),
        ("coverage", lambda: check_coverage(labels)),
        ("confounds", lambda: check_confounds(labels)),
    ]
    if not args.skip_pixels:
        checks += [
            ("identifiability", lambda: check_identifiability(root, labels)),
            ("appearance leak", lambda: check_appearance_leak(root, labels)),
        ]

    report, all_ok = {}, True
    for name, fn in checks:
        ok, msgs = fn()
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        for m in msgs:
            print(m)
        print()
        report[name] = {"ok": bool(ok), "messages": msgs}

    print("ALL CHECKS PASS" if all_ok else "SOME CHECKS FAILED")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"dataset_dir": str(root), "n_scenes": len(labels),
             "all_ok": bool(all_ok), "checks": report}, indent=2))
        print(f"-> {args.out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
