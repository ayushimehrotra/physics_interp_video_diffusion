#!/usr/bin/env python3
"""Validate a rendered inclined-plane set, before anyone probes it.

    python3 validate_incline.py --dataset-dir incline_set_v1

Eight checks; exit code gates a render. Five are the §25 family adapted to a
slope. Three are specific to this set:

  **DISSOCIATION** — mu must NOT be a proxy for acceleration. On a level floor
  a = mu*g, so corr(mu, a) = 0.9998 and every "friction" probe is an acceleration
  probe. Here a = g*(sin th - mu*cos th) with theta drawn independently, so the
  correlation must come out near -0.65. If it drifts toward -1 the set has failed
  at its one job and nothing probed on it can be called friction.

  **SINGLE-FRAME LEAK** — mu must not be readable from any ONE frame. v1 checked
  frame 0 only, and passed: the start was pinned, so frame 0 was clean. But a
  pinned start makes s(k) - s0 = 0.5*a*t^2 exactly, so every LATER frame handed
  over `a`, and theta is visible in the same frame. Measured on the v1 render, mu
  came out of the cube's segmentation centroid alone at R2 = 0.877 (frame 21) and
  0.988 with the mask area — against the 0.459 the set exists to impose. The check
  now sweeps EVERY frame, and there is a pixel-space version of it, because the
  world coordinate is not what the model sees.

  **MASK LEAK** — the same attack run on the actual segmentation masks: cube
  centroid and mask area, per frame, in pixels. This is §20's mask-only floor
  applied before capture rather than after.

Also note the appearance-leak check is subtler here than on the flat sets. The
slope angle CHANGES THE SHADING of the floor, and it is supposed to — theta is
meant to be visible. So the leak test is the PARTIAL correlation of floor luma
with log mu controlling for theta, not the raw one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FRAME_COUNT = 49
ACCEL_TOLERANCE = 0.08
LEAK_LUMA_TOLERANCE = 1.0
MAX_ABS_CORR_MU_A = 0.90     # must stay well clear of the flat floor's 0.9998


def load_set(root: Path) -> list[dict]:
    out = []
    for d in sorted(root.glob("scene_*")):
        p = d / "labels.json"
        if p.exists():
            lab = json.loads(p.read_text())
            lab["_dir"] = str(d)
            out.append(lab)
    return out


def check_completeness(root: Path, labels):
    msgs, ok = [], True
    failed = sorted(root.glob("scene_*/_FAILED"))
    if failed:
        ok = False
        msgs.append(f"  {len(failed)} scene(s) _FAILED: "
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


def check_physics(labels):
    rel = np.array([l["friction"]["accel_rel_error"] for l in labels])
    mu = np.array([l["friction"]["mu"] for l in labels])
    mu_k = np.array([l["friction"]["mu_from_kinematics"] for l in labels])
    err = np.abs(mu_k - mu) / mu
    ok = bool(rel.max() < ACCEL_TOLERANCE and err.max() < 0.15)
    return ok, [
        f"  along-slope accel vs g(sin th - mu cos th): median rel err "
        f"{np.median(rel):.2e}, max {rel.max():.2e} (bar {ACCEL_TOLERANCE})",
        f"  mu recovered from (a, theta) vs sampled mu: median err "
        f"{np.median(err):.2e}, max {err.max():.2e}",
        "  (tumbling, leaving the slope and lateral drift are rejected at "
        "generation time, so a scene present here already passed them)",
    ]


def check_coverage(labels):
    mu = np.array([l["friction"]["mu"] for l in labels])
    th = np.array([l["friction"]["theta_deg"] for l in labels])
    tv = np.array([l["friction"]["travel_measured"] for l in labels])
    c = labels[0]["scene_constants"]
    mlo, mhi = c["mu_range"]; tlo, thi = c["theta_range_deg"]
    q = np.quantile(np.log(mu), [0, .25, .5, .75, 1])
    want = np.log(mlo) + np.array([0, .25, .5, .75, 1]) * (np.log(mhi) - np.log(mlo))
    span = np.log(mhi) - np.log(mlo)
    tol = max(0.35, 1.2 * span / np.sqrt(len(labels)))
    ok = bool(mu.min() >= mlo * 0.99 and mu.max() <= mhi * 1.01
              and th.min() >= tlo - 0.5 and th.max() <= thi + 0.5
              and np.abs(q - want).max() < tol and tv.min() > 0.8)
    return ok, [
        f"  mu {mu.min():.3f}-{mu.max():.3f} (range {mlo}-{mhi}), log quartiles off "
        f"ideal by <= {np.abs(q-want).max():.2f} (bar at this n: {tol:.2f})",
        f"  theta {th.min():.1f}-{th.max():.1f} deg (range {tlo}-{thi})",
        f"  travel {tv.min():.2f}-{tv.max():.2f} world units",
    ]


def check_dissociation(labels):
    """THE CHECK THIS SET EXISTS FOR."""
    mu = np.array([l["friction"]["mu"] for l in labels])
    a = np.array([l["friction"]["accel_measured"] for l in labels])
    th = np.radians([l["friction"]["theta_deg"] for l in labels])
    c_ma = float(np.corrcoef(mu, a)[0, 1])
    c_mt = float(np.corrcoef(mu, th)[0, 1])
    tol = max(0.20, 3.0 / np.sqrt(len(labels)))
    ok = bool(abs(c_ma) < MAX_ABS_CORR_MU_A and abs(c_mt) < tol)
    return ok, [
        f"  corr(mu, measured accel) = {c_ma:+.3f}   (flat floor: +0.9998; bar |r| < {MAX_ABS_CORR_MU_A})",
        f"  => a linear readout of acceleration ALONE tops out at R2 = {c_ma**2:.3f} on mu",
        f"  corr(mu, theta) = {c_mt:+.3f}  (sampled independently; bar at this n {tol:.2f})",
        "  if corr(mu, a) approaches -1 the set has failed at its one job and no "
        "probe on it may be called friction rather than acceleration",
    ]


SPLITS = 20          # repeated 60/40 splits; one split at n=150 is too noisy to gate on
LEAK_BAR = 0.25      # held-out R2 on mu above which a static cue is doing the work


def _held_out_r2(X, y, splits: int = SPLITS, train_frac: float = 0.6):
    """Mean held-out R2 over repeated random splits, and the worst single split.

    The project's probes use one 60/40 split, but a single split at n = 150 has
    enough variance that a gate built on it fires on noise. Gate on the mean;
    report the max so a borderline set is visible rather than silently passed.
    """
    X = np.asarray(X, float).reshape(len(y), -1)
    y = np.asarray(y, float)
    n_tr = int(round(train_frac * len(y)))
    out = []
    for s in range(splits):
        idx = np.random.RandomState(s).permutation(len(y))
        tr, te = idx[:n_tr], idx[n_tr:]
        A = np.c_[np.ones(len(tr)), X[tr]]
        w = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        pred = np.c_[np.ones(len(te)), X[te]] @ w
        out.append(1 - ((y[te] - pred) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum())
    return float(np.mean(out)), float(np.max(out))


def check_single_frame_leak(labels):
    """Can mu be read off ONE frame, with no motion perceived at all?

    THIS IS THE CHECK v1 GOT WRONG, and the mistake is worth stating precisely.
    It tested frame 0 and only frame 0, because the first incline render gave the
    answer away there: centring each path on the origin made the start position a
    function of travel, hence of `a`, hence of mu at R2 = 1.0000 from frame 0.

    Pinning the start fixed frame 0 and broke every other frame. With s0 a constant
    that every scene shares, s(k) - s0 = 0.5*a*t_k^2 exactly, so the cube's position
    in ANY frame is `a` in disguise; theta is visible in the same frame; and
    mu = (sin th - a/g)/cos th. Measured on incline_set_v1: R2 = 0.997 from a single
    frame's world position for every k >= 6.

    A set passes only if NO single frame gives mu away. Frames are in world
    coordinates here, which is cheap and runs without pixels; check_mask_leak does
    the same attack on what the model actually sees.
    """
    mu = np.array([l["friction"]["mu"] for l in labels])
    th = np.radians([l["friction"]["theta_deg"] for l in labels])
    S = np.array([[f["s"] for f in l["per_frame"]] for l in labels])

    per_frame = [(k, *_held_out_r2(np.c_[S[:, k], th], mu)) for k in range(S.shape[1])]
    k_worst, r2_worst, r2_worst_max = max(per_frame, key=lambda r: r[1])
    r2_f0 = per_frame[0][1]

    # what the set is FOR: three frames must still be enough, or there is no
    # acceleration signal to decode and the set is merely unreadable.
    tri = [0, S.shape[1] // 2, S.shape[1] - 1]
    r2_tri, _ = _held_out_r2(np.c_[S[:, tri], th], mu)

    ok = bool(r2_worst < LEAK_BAR and r2_tri > 0.8)
    return ok, [
        f"  worst SINGLE frame: f{k_worst} gives mu at R2 = {r2_worst:.3f} "
        f"(worst split {r2_worst_max:.3f}; bar <{LEAK_BAR})",
        f"  frame 0 alone: R2 = {r2_f0:.3f}   (v1 checked only this one, and it passed)",
        f"  three frames f{tri}: R2 = {r2_tri:.3f}  (must stay HIGH — this is the "
        f"acceleration the set is about)",
        "  a high single-frame value means a probe can answer from a static "
        "configuration and no result on the set is evidence about dynamics",
    ]


def check_mask_leak(root: Path, labels, stride: int = 6, max_scenes: int = 150):
    """check_single_frame_leak, run on the pixels the model is actually given.

    §20's mask-only floor, applied BEFORE capture instead of after. The cube's
    segmentation centroid and area are exactly what a pooled activation inherits,
    so if mu regresses on them the probe never had to represent anything.
    """
    from PIL import Image

    labs = labels[:max_scenes]
    mu = np.array([l["friction"]["mu"] for l in labs])
    seg_id = labs[0]["scene_constants"]["seg_id"]["cube"]
    frames = list(range(0, FRAME_COUNT, stride))

    cen = np.zeros((len(labs), len(frames), 3))
    for i, lab in enumerate(labs):
        for j, k in enumerate(frames):
            f = Path(lab["_dir"]) / "segmentation" / f"segmentation_{k:04d}.png"
            if not f.exists():
                return True, [f"  no segmentation at {f}; skipped"]
            m = np.array(Image.open(f)) == seg_id
            if not m.any():
                return False, [f"  cube absent from the mask at {lab['_dir']} frame {k}"]
            ys, xs = np.nonzero(m)
            cen[i, j] = (xs.mean(), ys.mean(), m.sum())

    rows, worst = [], (-1, -1.0)
    for j, k in enumerate(frames):
        r2_c, _ = _held_out_r2(cen[:, j, :2], mu)
        r2_a, mx = _held_out_r2(cen[:, j, :], mu)
        if r2_a > worst[1]:
            worst = (k, r2_a)
        rows.append((k, r2_c, r2_a, mx))

    ok = bool(worst[1] < LEAK_BAR)
    msgs = [f"  worst frame: f{worst[0]}, mu from the cube's mask alone R2 = {worst[1]:.3f} "
            f"(bar <{LEAK_BAR})",
            "  frame : centroid  centroid+area"]
    msgs += [f"    f{k:<4d} {c:+.3f}      {a:+.3f}" for k, c, a, _ in rows]
    msgs.append("  incline_set_v1 read +0.877 / +0.988 here at f21 — that is the failure "
                "this check exists to stop")
    return ok, msgs


def check_confounds(labels):
    mu = np.log([l["friction"]["mu"] for l in labels])
    size = np.array([l["objects"]["A"]["size"] for l in labels])
    nrej = np.array([len(l["sampling_attempts"]["rejections"]) for l in labels])
    tol = max(0.15, 3.0 / np.sqrt(len(labels)))
    cs = float(np.corrcoef(mu, size)[0, 1])
    rj = float(np.corrcoef(mu, nrej)[0, 1]) if nrej.std() > 0 else 0.0
    ok = bool(abs(cs) < tol and abs(rj) < tol)
    msgs = [
        f"  corr(log mu, cube size) {cs:+.3f}   (bar at this n: {tol:.2f})",
        f"  rejections: {nrej.sum()} total, corr(log mu, n_rejections) {rj:+.3f}",
    ]
    # The two nuisance draws exist to hide `a`; if either picked up a correlation
    # with mu they would be hiding it selectively, which is worse than not at all.
    for key, name in (("start_s", "start position"), ("launch_speed", "launch speed")):
        if key not in labels[0]["friction"]:
            continue
        v = np.array([l["friction"][key] for l in labels])
        c = float(np.corrcoef(mu, v)[0, 1]) if v.std() > 1e-9 else 0.0
        ok = ok and bool(abs(c) < tol)
        msgs.append(f"  corr(log mu, {name}) {c:+.3f}   (drawn independently; bar {tol:.2f})")
    return ok, msgs


def check_visibility(root: Path, labels, max_scenes: int = 40):
    """THE CHECK THAT WAS MISSING, and it let a black render pass six others.

    The first incline pilot rendered the slope as an unlit wedge at luma 0.6 of
    255 — the camera sat exactly in the slope plane, so every surface point was
    edge-on. Every other check passed, because the appearance-leak test measures
    the mu-driven SWING of floor luma (correctly ~0 on a black floor) and never
    asks whether the floor is visible at all.

    Three things have to be true of a usable render, and none of them were:
      - the slope occupies a real fraction of the frame
      - it is LIT (luma well above black)
      - the bands are resolvable, so the frame carries the static reference §25
        added; without them a translating cube and a panning camera are identical
      - the cube is present in EVERY frame, or its pool is empty somewhere
    """
    import imageio.v3 as iio
    cover, lum, band, cube_min = [], [], [], []
    for lab in labels[:max_scenes]:
        d = Path(lab["_dir"])
        fr = sorted((d / "frames").glob("frame_*.png"))
        sg = sorted((d / "segmentation").glob("segmentation_*.png"))
        if not fr or not sg:
            return False, [f"  {d.name}: missing pixels"]
        areas = []
        for p_ in sg:
            m = np.asarray(iio.imread(p_))
            if m.ndim == 3:
                m = m[..., 0]
            areas.append((m == 1).sum())
        cube_min.append(min(areas))
        img = np.asarray(iio.imread(fr[len(fr) // 2])).astype(np.float64)
        m = np.asarray(iio.imread(sg[len(sg) // 2]))
        if m.ndim == 3:
            m = m[..., 0]
        y = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        floor = (m == 2)
        cover.append(floor.mean())
        lum.append(y[floor].mean() if floor.any() else 0.0)
        # band contrast: spread of floor luma WITHIN a frame, which the stripes
        # create and a flat/unlit floor does not
        band.append(y[floor].std() if floor.any() else 0.0)
    cover, lum, band = np.array(cover), np.array(lum), np.array(band)
    ok = bool(cover.min() > 0.15 and lum.min() > 20.0 and band.min() > 1.0
              and min(cube_min) > 0)
    return ok, [
        f"  slope covers {100*cover.min():.1f}-{100*cover.max():.1f}% of frame (bar >15%)",
        f"  slope luma {lum.min():.1f}-{lum.max():.1f} (bar >20; a black wedge reads 0.6)",
        f"  within-frame luma sd {band.min():.2f}-{band.max():.2f} (bar >1.0; the bands)",
        f"  smallest per-frame cube area {min(cube_min)} px (must be >0 in every frame)",
    ]


def check_appearance_leak(root: Path, labels, max_scenes: int = 60):
    """Latent arm: the floor may encode THETA (it must — theta is visible) but not MU.

    So the statistic is the PARTIAL correlation of floor luma with log mu after
    regressing out theta. A raw correlation would fail every scene set by design.
    """
    import imageio.v3 as iio
    mu, th, lum = [], [], []
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
        if (m == 2).sum() == 0:
            return False, [f"  {d.name}: no floor pixels in segmentation"]
        y = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        mu.append(np.log(lab["friction"]["mu"]))
        th.append(lab["friction"]["theta"])
        lum.append(y[m == 2].mean())
    mu, th, lum = np.array(mu), np.array(th), np.array(lum)

    def resid(v):
        A = np.c_[np.ones_like(th), th]
        return v - A @ np.linalg.lstsq(A, v, rcond=None)[0]

    rl, rm = resid(lum), resid(mu)
    pc = float(np.corrcoef(rl, rm)[0, 1]) if rl.std() > 1e-9 and rm.std() > 1e-9 else 0.0
    slope = np.polyfit(rm, rl, 1)[0] if rm.std() > 1e-9 else 0.0
    swing = abs(slope) * (mu.max() - mu.min())
    raw = float(np.corrcoef(lum, th)[0, 1]) if lum.std() > 1e-9 else 0.0
    arm = labels[0]["arm"]
    ok = not (arm == "latent_friction" and swing > LEAK_LUMA_TOLERANCE)
    return ok, [
        f"  arm: {arm}",
        f"  floor luma mean {lum.mean():.2f}, sd {lum.std():.3f}",
        f"  corr(luma, theta) {raw:+.3f}  <- EXPECTED and fine: the slope changes shading",
        f"  PARTIAL corr(luma, log mu | theta) {pc:+.3f}, mu-driven swing {swing:.3f} luma "
        f"(bar {LEAK_LUMA_TOLERANCE})",
    ]


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
        ("DISSOCIATION (mu vs acceleration)", lambda: check_dissociation(labels)),
        ("SINGLE-FRAME leak", lambda: check_single_frame_leak(labels)),
        ("confounds", lambda: check_confounds(labels)),
    ]
    if not args.skip_pixels:
        checks.append(("MASK leak (pixels)", lambda: check_mask_leak(root, labels)))
        checks.append(("visibility", lambda: check_visibility(root, labels)))
        checks.append(("appearance leak", lambda: check_appearance_leak(root, labels)))

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
