"""Output-level evaluation (spec §6): do GENERATED continuations obey the laws?

Deviation: the spec's tracker is SAM 2 + CoTracker (not installed). F1 scenes are two spheres on a
uniform grey floor with a fixed top-down camera, so a model-agnostic colour tracker is used instead,
initialised from the REAL context frames (shared by the generated clip), and validated on the real
renders (``--real``): its error there is the tracker floor that the scores subtract.

Tracker (``track``):
  * appearance per ball from the context frames and ground-truth masks: mean RGB and radius (px)
  * per frame: floor colour = median RGB; foreground = colour distance to the floor > ``fg_thresh``;
    connected components >= ``min_area`` px; blobs are labelled by nearest appearance colour
  * identity: segments of consecutive two-blob frames, linked by nearest position inside a segment;
    each segment labelled once by appearance (colour + radius) summed over it, which a collision
    cannot change; the first segment is anchored by the known start positions
  * pixel -> world: per-ball affine map fitted on the label's (pixel, position) pairs over the clip
    (exact for this pinhole camera at the ball's constant height), after undoing the frame fit.
Readout (``readout``):
  * contact = first frame where the balls merge or come within 1.25 (r_A + r_B) in pixels
  * merged blobs (touching balls + shadows) larger than 1.5x one ball are split by 2-means on position
  * post velocities = least-squares slope of world position over ``post_window`` frames starting
    ``post_gap`` frames after contact; both balls must be tracked in >= ``min_post`` of them
  * quality flags: no_collision, lost, contact_at_edge, bad_fit (rms > ``max_rms`` world units)
Scores (spec §5.0.4 Outputs): the unified laws' dimensionless statistics (physlaw.eval.laws), scale-free (no true
g, fps or metric speed enters any statistic), floor = the same readout on the real clips:
  collisions   epsilon (laws.epsilon along the LINE OF CENTRES at contact; never the impulse direction, which hides
               sideways kicks and biases eps upward), eta (laws.momentum.stats, label masses), antiparallel and
               normal alignment (mass-free); track A: true pre state, masses and line of centres from the labels;
               track B: the clip's own measured states and line of centres (:func:`_contact_normal`)
  F7           bounce: epsilon from the fall / rise parabolas at contact (and sqrt of the rebound height ratio);
               projectile / drop: laws.gravity g_ratio (parabolas before vs after), axis_ratio, mass_ratio
  F8           Galileo's ramp: epsilon = sqrt(apex height / release height) above the pad level, measured in the
               video (and the pad's speed ratio)
  F10          laws.rolling order and slip = |w| r / |v| per cylinder, from the centre track and the marker wedge's
               angle (:func:`readout_rolling`, :func:`rolling_stats`)
  F6           glide: laws.momentum (Newton I) inertia_speed, inertia_turn; slide / pair: laws.friction
               decel_const (after vs before the branch), stop_hold (stop window), mass_ratio (:func:`readout_friction`)
  F9           pair: laws.oscillation period_ratio, mass_period_ratio from io.zero_crossings of the bob offsets; swing:
               epsilon from apex heights above the lowest tracked point (:func:`readout_pendulum`)
  F11          laws.statics tip_consistency (block rotation over the tip window, rad) and alpha_ratio with g_clip
               measured in the clip and the half-length in pixels (:func:`readout_support`)
  F12          laws.angular_momentum L_ratio from the centre tracks and the bar's marker-wedge heading
               (:func:`readout_spin`)
Energy: the share of measurable clips with epsilon <= 1 + tol, generated against the real-clip readout distribution
(per rung, and the real legal rungs pooled; :func:`energy_shares`).
Feasibility gate (configs/eval/outputs.yaml track_a.gate): measurable fraction of seed-0 generations of the first
``n_clips`` groups must reach ``min_pass_rate`` before the other seeds are generated.
Stage drivers: track A (§6.2) ``gate`` / ``track_a_rows`` (stage 8) / ``track_a_summary`` (stage 9); track B (§6.3)
``READOUT_SELF`` per family (stage 8) / ``track_b_summary`` (stage 9).
"""
from __future__ import annotations

import numpy as np

from physlaw.models.base import fit_frames


def ball_signatures(frames: np.ndarray, masks: np.ndarray) -> dict:
    """frames [T, h, w, 3] uint8 and instance masks [T, h, w] (1 = A, 2 = B) of the CONTEXT frames."""
    sig = {}
    for o, sid in (("A", 1), ("B", 2)):
        m = masks == sid
        sig[o] = {"rgb": frames[m].reshape(-1, 3).astype(float).mean(0) if m.any() else np.full(3, np.nan),
                  "radius_px": float(np.sqrt(m.sum(axis=(1, 2)).clip(1).mean() / np.pi))}
    return sig


# Split / size / count limits, calibrated 2026-11-06 on the 64-group F1 subset (128 real legal clips each at the
# CogVideoX-2b and 1.5-5B-I2V geometries). Real merged (touching) blobs: elongation 1.45-2.9 (p1 1.50), area
# <= 1.59 x the two balls' mask area. Real generated-part ball area / context: <= 1.20. Real frames with > 2
# ball-sized blobs: 0. False positives they remove (CogVideoX-2b, minimal prompt): a single zoomed-in ball after
# a cut, split into two "balls" (elongation 1.17-1.24), then read as a measurable static pair.
SPLIT_MIN_ELONG = 1.35
SPLIT_MAX_AREA = 2.0
SIZE_RATIO_MAX = 1.5
EXTRA_FRAMES_MAX = 2
MERGE_FRAC = 0.8          # blobs closer than this x (rA + rB) are one ball (+ its detached shadow)


def _blobs(frame: np.ndarray, sig: dict, fg_thresh: float, min_area: int, bg: np.ndarray | None = None) -> list[dict]:
    import cv2
    img = frame.astype(np.float32)
    # background: a known image (textured floors) or, by default, the frame's median colour
    floor = bg.astype(np.float32) if bg is not None else np.median(img.reshape(-1, 3), 0)
    fg = (np.linalg.norm(img - floor, axis=2) > fg_thresh).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(fg, connectivity=8)
    out = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < min_area:
            continue
        pix = img[lab == i]
        out.append({"c": cents[i][::-1].copy(), "area": float(a), "rgb": pix.mean(0),
                    "r": float(np.sqrt(a / np.pi)), "ids": [i]})               # c = (row, col)
    out.sort(key=lambda b: -b["area"])
    # a ball and its own shadow can come out as two blobs when the ball's rim is floor-coloured (Wan's grey
    # metal ball: rim rgb ~105 on a ~165 floor). Two DISTINCT balls are never closer than rA + rB centre to
    # centre (overhead view), so a blob nearer than MERGE_FRAC * (rA + rB) to a bigger one is the same ball:
    # fold it in (area-weighted), as if the two were connected.
    near = MERGE_FRAC * (sig["A"]["radius_px"] + sig["B"]["radius_px"])
    merged = []
    for b in out:                                                # largest first
        host = next((m for m in merged if np.linalg.norm(m["c"] - b["c"]) < near), None)
        if host is None:
            merged.append(dict(b, ids=list(b["ids"])))
            continue
        a0, a1 = host["area"], b["area"]
        host["c"] = (host["c"] * a0 + b["c"] * a1) / (a0 + a1)
        host["rgb"] = (host["rgb"] * a0 + b["rgb"] * a1) / (a0 + a1)
        host["area"] = a0 + a1
        host["r"] = float(np.sqrt(host["area"] / np.pi))
        host["ids"] += b["ids"]
    if len(merged) < len(out):
        out = sorted(merged, key=lambda b: -b["area"])
    # two touching balls (and their shadows) merge into one blob: split it when it is clearly larger
    # than one ball, by 2-means on pixel positions seeded at the ends of the blob's long axis
    one = max(np.pi * sig["A"]["radius_px"] ** 2, np.pi * sig["B"]["radius_px"] ** 2) * 1.3
    two = np.pi * (sig["A"]["radius_px"] ** 2 + sig["B"]["radius_px"] ** 2)
    if out and (len(out) == 1 or out[1]["area"] < 0.15 * out[0]["area"]) and out[0]["area"] > 1.5 * one \
            and out[0]["area"] <= SPLIT_MAX_AREA * two:
        big = out[0]
        ys, xs = np.nonzero(np.isin(lab, big["ids"]))
        P = np.stack([ys, xs], 1).astype(float)
        mu = P.mean(0)
        _, sv, vt = np.linalg.svd(P - mu, full_matrices=False)
        elong = float(sv[0] / max(sv[1], 1e-9))                  # 1 = round; two touching discs ~ >= 1.5
        if elong < SPLIT_MIN_ELONG:                              # one round blob (e.g. a single zoomed-in
            return out[:4]                                       # ball after a cut): never split it in two
        proj = (P - mu) @ vt[0]
        C = np.stack([P[proj <= np.quantile(proj, 0.25)].mean(0), P[proj >= np.quantile(proj, 0.75)].mean(0)])
        for _ in range(10):
            asg = np.linalg.norm(P[:, None] - C[None], axis=2).argmin(1)
            C = np.stack([P[asg == j].mean(0) if (asg == j).any() else C[j] for j in range(2)])
        parts = []
        for j in range(2):
            sel = asg == j
            pix = img[ys[sel], xs[sel]]
            parts.append({"c": C[j], "area": float(sel.sum()), "rgb": pix.mean(0), "r": float(np.sqrt(sel.sum() / np.pi)),
                          "split": True, "parent_area": big["area"], "parent_elong": elong})
        out = parts + out[1:]
    return out[:4]


def track(frames: np.ndarray, sig: dict, start: dict, *, fg_thresh: float = 30.0, min_area: int = 40,
          bg: np.ndarray | None = None) -> dict:
    """Pixel centres [T, 2 (A, B), 2 (row, col)], nan when lost / merged. ``start`` = {o: (row, col)}
    of each ball in frame 0 (from the labels).

    Identity: the clip is cut into SEGMENTS of consecutive frames with two blobs. Inside a segment
    blobs are linked frame to frame by nearest position (no velocity prediction: a bounce reverses
    it). Each segment's labelling (as-is or swapped) is then chosen once, by appearance summed over
    the whole segment (colour distance to each ball's signature + radius difference); the segment
    holding frame 0 is anchored by the known start positions instead."""
    T = len(frames)
    sigv = np.stack([sig["A"]["rgb"], sig["B"]["rgb"]])
    sigr = np.array([sig["A"]["radius_px"], sig["B"]["radius_px"]])
    cand = [None] * T
    fg_area, n_ball = np.zeros(T), np.zeros(T, int)
    one_min = min(np.pi * sig["A"]["radius_px"] ** 2, np.pi * sig["B"]["radius_px"] ** 2)
    for t in range(T):
        bl = _blobs(frames[t], sig, fg_thresh, min_area, bg)
        fg_area[t] = sum(b["area"] for b in bl)
        n_ball[t] = sum(b["area"] >= 0.5 * one_min for b in bl)
        if len(bl) >= 2:
            cand[t] = bl[:2]
    pos = np.full((T, 2, 2), np.nan)
    t = 0
    while t < T:
        if cand[t] is None:
            t += 1
            continue
        seg, prev = [], None
        while t < T and cand[t] is not None:                     # link inside the segment
            C = np.stack([b["c"] for b in cand[t]])
            o = [0, 1]
            if prev is not None and np.linalg.norm(C[::-1] - prev, axis=1).sum() < np.linalg.norm(C - prev, axis=1).sum():
                o = [1, 0]
            seg.append((t, [cand[t][i] for i in o]))
            prev = C[o]
            t += 1
        if seg[0][0] == 0:                                       # anchored by the start positions
            C0 = np.stack([b["c"] for b in seg[0][1]])
            S = np.array([start["A"], start["B"]], float)
            swap = np.linalg.norm(C0[::-1] - S, axis=1).sum() < np.linalg.norm(C0 - S, axis=1).sum()
        else:                                                    # appearance over the whole segment
            def cost(order):
                tot = 0.0
                for _, bl in seg:
                    rgb = np.stack([bl[i]["rgb"] for i in order]); rr = np.array([bl[i]["r"] for i in order])
                    tot += np.linalg.norm(rgb - sigv, axis=1).sum() / 30.0 + np.abs(rr - sigr).sum() / 3.0
                return tot
            swap = cost([1, 0]) < cost([0, 1])
        for tt, bl in seg:
            o = [1, 0] if swap else [0, 1]
            pos[tt] = np.stack([bl[i]["c"] for i in o])
    return {"pixel": pos, "fg_area": fg_area, "n_ball": n_ball}


def _world_maps(labels: dict, frames_idx: list[int], H: int, W: int, fit: str) -> dict:
    """Per-ball affine (row, col in MODEL pixels) -> world xy, from the label's pixel/position pairs."""
    sc = labels["scene_constants"]
    W0, H0 = sc["resolution"] if "resolution" in sc else labels["physlaw"]["master_format"]["resolution"]
    if fit == "squash":
        (sy, oy), (sx, ox) = (H / H0, 0.0), (W / W0, 0.0)
    elif fit == "letterbox":                     # pad to aspect, then resize (models.base.fit_frames)
        s = min(H / H0, W / W0)
        (sy, oy), (sx, ox) = (s, (H - int(round(H0 * s))) // 2), (s, (W - int(round(W0 * s))) // 2)
    else:
        raise ValueError(f"unknown frame fit {fit!r} (squash | letterbox)")
    maps = {}
    pf = labels["per_frame"]
    for o in ("A", "B"):
        P = np.array([[pf[f][o]["pixel"][1] * sy + oy, pf[f][o]["pixel"][0] * sx + ox, 1.0] for f in frames_idx])
        Wd = np.array([pf[f][o]["position"][:2] for f in frames_idx])
        maps[o] = np.linalg.lstsq(P, Wd, rcond=None)[0]                # [3, 2]
    return maps


def readout(pix: np.ndarray, labels: dict, frames_idx: list[int], fps: float, H: int, W: int, fit: str = "squash",
            *, post_gap: int = 3, post_window: int = 8, min_post: int = 4, max_rms: float = 0.05, sep_mult: float = 1.05,
            context_end: int | None = None) -> dict:
    """Contact, post velocities (world units / s) and quality flags from tracked pixel centres."""
    maps = _world_maps(labels, frames_idx, H, W, fit)
    pl = labels["physlaw"]
    rA, rB = pl["objects"]["A"]["radius"], pl["objects"]["B"]["radius"]
    world = np.full(pix.shape, np.nan)
    for i, o in enumerate("AB"):
        P = np.concatenate([pix[:, i], np.ones((len(pix), 1))], 1)
        world[:, i] = P @ maps[o]
    d = np.linalg.norm(world[:, 0] - world[:, 1], axis=1)
    ok = np.isfinite(d)
    start = context_end or 0
    T = len(pix)
    res = {"frames_both_tracked": int(ok.sum())}
    near = np.where((~ok | (d <= 1.25 * (rA + rB))) & (np.arange(T) >= max(start - 3, 0)))[0]
    if ok.sum() < 6:
        return {**res, "measurable": False, "reason": "lost"}
    if context_end is not None and not ok[context_end:].any():
        return {**res, "measurable": False, "reason": "lost_after_context"}
    if not len(near):
        return {**res, "measurable": False, "reason": "no_collision"}
    c = int(near[0])
    # post window: a fixed gap after contact (low-restitution balls may never visibly separate)
    s0 = c + post_gap
    win = [t for t in range(s0, min(T, s0 + post_window)) if ok[t]]
    res.update(contact=c, post_frames=win)
    if len(win) < min_post:
        return {**res, "measurable": False, "reason": "contact_at_edge"}
    tt = np.array(win, float) / fps
    v, x, rms = np.zeros((2, 2)), np.zeros((2, 2)), 0.0
    for i in range(2):
        A = np.stack([tt, np.ones_like(tt)], 1)
        coef, *_ = np.linalg.lstsq(A, world[win, i], rcond=None)
        v[i], x[i] = coef[0], coef[0] * c / fps + coef[1]          # x: position at the contact frame
        rms = max(rms, float(np.sqrt(((A @ coef - world[win, i]) ** 2).mean())))
    res.update(v_post=v.tolist(), x_post=x.tolist(), fit_rms=rms)
    if rms > max_rms:
        return {**res, "measurable": False, "reason": "bad_fit"}
    return {**res, "measurable": True}


def clip_inputs(clip, model_frames: list[int], H: int, W: int, fit: str, context_end: int):
    """Model-resolution real frames, signatures from the context frames, frame-0 ball pixels."""
    frames = fit_frames(clip.frames[model_frames], (H, W), fit)
    masks = fit_frames(np.repeat(clip.masks[model_frames][..., None], 3, -1), (H, W), fit)[..., 0]
    masks = np.where(np.isin(masks, (1, 2)), masks, 0)
    sig = ball_signatures(frames[:context_end], masks[:context_end])
    pf = clip.labels["per_frame"]
    W0, H0 = clip.labels["physlaw"]["master_format"]["resolution"]
    start = {o: (pf[model_frames[0]][o]["pixel"][1] * H / H0, pf[model_frames[0]][o]["pixel"][0] * W / W0) for o in "AB"}
    return frames, sig, start


# --------------------------------------------------------------------------- #
# stage drivers (scripts/07, scripts/09)
# --------------------------------------------------------------------------- #

LEGAL = ("conserve_e1", "conserve_e015")


def _read_mp4(path):
    import imageio.v3 as iio
    return iio.imread(path, plugin="FFMPEG")


def cut_score(frames: np.ndarray, k: int, horizon: int = 12, thresh: float = 30.0) -> float:
    """Largest fraction of pixels, over the first ``horizon`` generated frames, that differ from the
    CONTEXT background (per-pixel median of the context frames) by more than ``thresh``. A continuation
    only changes the balls' pixels (a few %); a generator that starts a new shot changes most of the
    frame, even if the new shot fades in over several frames (naturalness test: CogVideoX-2b cut to a
    different table / camera)."""
    bg = np.median(frames[:k], 0).astype(np.float32)
    seg = frames[k:k + horizon].astype(np.float32)
    return float((np.abs(seg - bg).max(-1) > thresh).mean(axis=(1, 2)).max())


CUT_MAX = 0.25


def measure_clip(gen_frames, clip, model_frames, k, fps, H, W, fit="squash", checks: bool = True) -> dict:
    """Track + read out one video (generated, or the real clip if gen_frames is None). A generated clip
    whose first generated frame jumps away from the context (``cut_score`` > CUT_MAX) is dropped as a
    scene cut: whatever is tracked afterwards is not a continuation of this scene.

    ``checks`` (added 2026-11-06; applied to real and generated clips alike): the cut score sees only the
    background, so a cut to a close-up of the same grey floor passed it. Also reject when the balls' total
    blob area after the context exceeds SIZE_RATIO_MAX x its context median ("size_change": zoom, growth),
    or more than two ball-sized blobs appear in > EXTRA_FRAMES_MAX frames ("extra_objects")."""
    real, sig, start = clip_inputs(clip, model_frames, H, W, fit, k)
    frames = real if gen_frames is None else gen_frames
    cs = cut_score(frames, k)
    if gen_frames is not None and cs > CUT_MAX:
        return {"frames_both_tracked": 0, "cut_score": cs, "measurable": False, "reason": "scene_cut"}
    tr = track(frames, sig, start)
    out = {**readout(tr["pixel"], clip.labels, model_frames, fps, H, W, fit, context_end=k), "cut_score": cs}
    ref = float(np.median(tr["fg_area"][:k])) if k else float("nan")
    size_ratio = float(np.max(tr["fg_area"][k:]) / ref) if ref > 0 and len(frames) > k else float("nan")
    n_extra = int((tr["n_ball"][k:] > 2).sum())
    out.update(size_ratio=size_ratio, extra_object_frames=n_extra)
    if checks and out.get("measurable"):
        if np.isfinite(size_ratio) and size_ratio > SIZE_RATIO_MAX:
            out.update(measurable=False, reason="size_change")
        elif n_extra > EXTRA_FRAMES_MAX:
            out.update(measurable=False, reason="extra_objects")
    return out


def _track_a_measure(group_dir, adapter_geom):
    """(legal clips, legal twin names, measure(video or None for the real clip, clip) -> readout) for one group:
    F1's collision readout, or READOUT_SELF's trajectory readout on the model's frames for F7 / F8."""
    from physlaw.data.io import load_clip
    legal = legal_twins(group_dir)
    clips = {t: load_clip(group_dir / t) for t in legal}
    fr, k = adapter_geom(clips[legal[0]].labels)
    fam = clips[legal[0]].pl["family"]
    if READOUT_SELF.get(fam) is readout_self:              # two-ball collision families
        return clips, legal, lambda v, c: measure_clip(v, c, fr, k, *adapter_geom.fps_hw)
    fps, H, W = adapter_geom.fps_hw
    return clips, legal, lambda v, c: READOUT_SELF[fam](fit_frames(c.frames[fr], (H, W)) if v is None else v, c, fps,
                                                        frames_idx=fr)


def gate(cache_dir, data_dir, adapter_geom, n_clips: int, min_pass_rate: float, log=print) -> dict:
    """Unsteered feasibility gate on the seed-0 generations of the first n_clips groups."""
    gens = sorted(cache_dir.glob("*/gen_seed0.mp4"))[:n_clips]
    rows = []
    for g in gens:
        clips, legal, measure = _track_a_measure(data_dir / g.parent.name, adapter_geom)
        r = measure(_read_mp4(g), clips[legal[0]])
        rows.append({"group": g.parent.name, **{kk: v for kk, v in r.items() if kk != "post_frames"}})
    rate = float(np.mean([r["measurable"] for r in rows])) if rows else 0.0
    from collections import Counter
    out = {"n": len(rows), "pass_rate": rate, "min_pass_rate": min_pass_rate, "passed": rate >= min_pass_rate,
           "reasons": dict(Counter(r.get("reason", "ok") for r in rows)), "clips": rows}
    log(f"[gate] {len(rows)} clips, measurable {rate:.0%} (need {min_pass_rate:.0%}): {out['reasons']}")
    return out


def _row(r: dict) -> dict:
    return {k: v for k, v in r.items() if k not in ("post_frames", "_pix")}


def track_a_rows(gen_dir, data_dir, adapter_geom, log=print) -> tuple[list[dict], list[dict]]:
    """Track A readouts: every generated video (one per group and seed; scored later against each legal twin)
    and the real legal twins (the tracker floor). Rows carry the readout, not the law scores (stage 09)."""
    gen_rows, real_rows = [], []
    for g in sorted({p.parent.name for p in gen_dir.glob("*/gen_seed*.mp4")}):
        clips, legal, measure = _track_a_measure(data_dir / g, adapter_geom)
        for t, c in clips.items():
            real_rows.append({"group": g, "twin": t, **_row(measure(None, c))})
        for mp4 in sorted((gen_dir / g).glob("gen_seed*.mp4")):
            r = measure(_read_mp4(mp4), clips[legal[0]])
            gen_rows.append({"group": g, "seed": int(mp4.stem.split("seed")[1]), **_row(r)})
        log(f"[trackA] {g}: {sum(r['group'] == g and r['measurable'] for r in gen_rows)} measurable generated")
    return gen_rows, real_rows


def legal_twins(group_dir) -> list[str]:
    """A group's legal twins: F1's names, else every twin whose labels say ``legal``."""
    import json
    from pathlib import Path
    twins = json.loads((Path(group_dir) / "_GROUP_COMPLETE").read_text())["twins"]
    if all(t in twins for t in LEGAL):
        return list(LEGAL)
    return [t for t in twins if json.loads((Path(group_dir) / t / "labels.json").read_text())["physlaw"].get("legal")]


STAT_KEYS = ("epsilon", "epsilon_height", "epsilon_speed", "eta", "antiparallel", "normal_A", "normal_B",
             "g_ratio", "axis_ratio", "mass_ratio", "order", "slip_s", "slip_h", "inertia_speed", "inertia_turn",
             "decel_const", "stop_hold", "period_ratio", "mass_period_ratio", "tip_consistency", "alpha_ratio", "L_ratio")
ENERGY_DEFAULTS = {"tol": 0.1}


def collision_stats(v0, v1, n, m=None) -> dict:
    """laws.epsilon (collision, along n) and laws.momentum.stats of one measured two-body event (v0, v1 [2, 2])."""
    from physlaw.eval import laws
    ctx = {"event": "collision", "n": np.asarray(n, float), **({"m": np.asarray(m, float)} if m is not None else {})}
    pre, post = {"v": np.asarray(v0, float)}, {"v": np.asarray(v1, float)}
    return {"epsilon": float(laws.epsilon(pre, post, ctx)), **laws.momentum.stats(pre, post, ctx)}


def row_stats(r: dict, labels: dict) -> dict:
    """Law statistics of one measurable readout row: track A collision rows (``v_post``: true pre velocities, masses
    and line of centres from the labels), others as the readout computed them (track B collisions, F6-F12)."""
    if not r.get("measurable"):
        return {}
    if "v_post" in r and "velocity_post" not in r:
        from physlaw.eval import laws
        pre, _, ctx = laws.from_labels(labels)
        return collision_stats(pre["v"], r["v_post"], ctx["n"], ctx["m"])
    return {k: r[k] for k in STAT_KEYS if k in r}


def true_epsilon(labels: dict) -> float:
    """The twin's own eps: the labelled ladder dose (F2, F7 bounce, F8, F9 swing), else laws.epsilon of its labelled
    event, NaN if it has no exchange event."""
    from physlaw.eval import laws
    from physlaw.eval.l0_states import event_inputs
    if np.isfinite(labels["physlaw"].get("epsilon", np.nan)):
        return float(labels["physlaw"]["epsilon"])
    ev = event_inputs(labels)
    return float(laws.epsilon({"v": ev[0]}, {"v": ev[1]}, ev[2])) if ev else float("nan")


def _boot_ci(vals_by_group: dict, stat, n_boot: int, seed: int = 0):
    gs = sorted(vals_by_group)
    if not gs:
        return None
    rng = np.random.default_rng(seed)
    bo = [stat([v for g in rng.choice(gs, len(gs)) for v in vals_by_group[g]]) for _ in range(n_boot)]
    return [float(np.nanquantile(bo, 0.025)), float(np.nanquantile(bo, 0.975))]


def _share(rows, tol, n_boot=0) -> dict:
    e = np.array([r["epsilon"] for r in rows if np.isfinite(r.get("epsilon", np.nan))], float)
    d = {"n": int(len(e)), "share_le1": float((e <= 1 + tol).mean()) if len(e) else float("nan"),
         "median": float(np.median(e)) if len(e) else float("nan")}
    if n_boot and len(e):
        by = {}
        for r in rows:
            if np.isfinite(r.get("epsilon", np.nan)):
                by.setdefault(r["group"], []).append(r["epsilon"])
        d["share_le1_ci95"] = _boot_ci(by, lambda x: float((np.asarray(x) <= 1 + tol).mean()), n_boot)
    return d


def energy_shares(gen: list[dict], real: list[dict], tol: float = 0.1, n_boot: int = 0) -> dict:
    """Share of measurable rows with epsilon <= 1 + tol: generated (group-bootstrap CI) against the real-clip readout
    distribution, per real twin (with its true eps) and pooled over the real legal (eps_true <= 1) and creating
    (eps_true > 1) rungs: the readout's own specificity / sensitivity at this tol."""
    out = {"tol": tol, "generated": _share(gen, tol, n_boot)}
    if real:
        tw = sorted({r["twin"] for r in real})
        out["real"] = {t: {**_share([r for r in real if r["twin"] == t], tol),
                           "eps_true": next((r["eps_true"] for r in real if r["twin"] == t and "eps_true" in r), None)}
                       for t in tw}
        et = lambda r: r.get("eps_true", np.nan)
        out["real_legal"] = _share([r for r in real if et(r) <= 1 + 1e-9], tol)
        out["real_creating"] = _share([r for r in real if et(r) > 1 + 1e-9], tol)
    return out


def _stats(rows: list[dict], keys=STAT_KEYS) -> dict:
    from collections import Counter
    m = [r for r in rows if r["measurable"]]
    d = {"n": len(rows), "measurable": len(m),
         "drop_reasons": dict(Counter(str(r.get("reason")) for r in rows if not r["measurable"]))}
    for k in keys:
        vals = [r[k] for r in m if isinstance(r.get(k), (int, float)) and np.isfinite(r[k])]
        if vals:
            d[f"median_{k}"], d[f"mean_{k}"] = float(np.median(vals)), float(np.mean(vals))
    viol = Counter(v for r in m for v in r.get("violations") or [])
    if viol:
        d["violations"] = dict(viol)
        d["measurable_without_violation"] = sum(not r.get("violations") for r in m)
    return d


def _labels_fn(data_dir):
    import json
    lab = {}

    def labels(g, t):
        if (g, t) not in lab:
            lab[g, t] = json.loads((data_dir / g / t / "labels.json").read_text())
        return lab[g, t]
    return labels


def track_a_summary(gen_rows, real_rows, data_dir, n_boot: int = 2000, mom_cfg: dict | None = None,
                    energy_cfg: dict | None = None) -> dict:
    """Generated rows (one per group and seed) and real legal twins (the floor): drop rates, medians of every law
    statistic (``generated``; ``real`` per twin), the energy share (:func:`energy_shares`) and, for collisions, the
    S_mom package per twin and pooled (:func:`track_a_momentum`)."""
    labels = _labels_fn(data_dir)
    first = {}
    for r in real_rows:
        first.setdefault(r["group"], r["twin"])
    G = [{**r, **row_stats(r, labels(r["group"], first[r["group"]]))} if r["group"] in first else r for r in gen_rows]
    R = [{**r, **row_stats(r, labels(r["group"], r["twin"])), "eps_true": true_epsilon(labels(r["group"], r["twin"]))}
         for r in real_rows]
    mcfg = {**MOM_DEFAULTS, **(mom_cfg or {})}
    tol = float({**ENERGY_DEFAULTS, **(energy_cfg or {})}["tol"])
    twins = sorted({r["twin"] for r in R}, key=lambda x: (LEGAL.index(x) if x in LEGAL else len(LEGAL), x))
    out = {"n_groups": len({r["group"] for r in gen_rows}), "momentum_config": mcfg,
           "generated": _stats(G), "real": {t: _stats([r for r in R if r["twin"] == t]) for t in twins},
           "energy": energy_shares([r for r in G if r["measurable"]], [r for r in R if r["measurable"]], tol, n_boot)}
    mom = track_a_momentum(gen_rows, real_rows, labels, twins, mcfg, n_boot)
    if mom:
        out["momentum"] = mom
    out["rows_generated"], out["rows_real"] = G, R
    return out


# --------------------------------------------------------------------------- #
# Momentum package on generated outcomes (2026-11-11; replaces r_p / S_outcome as the momentum test)
# --------------------------------------------------------------------------- #
# Track A (true masses, true contact normal): L1's S_mom on the generated outcome.
#   q_i = m_i dv_i, D = |q_A| + |q_B|, eta = |q_A + q_B| / D (laws.momentum_exchange).
#   interaction gate: D >= kappa * d_min, d_min = 2 mu u_n (the weakest legal collision). interaction_rate = share
#   of measurable generated clips that pass it (did the model make the balls exchange momentum at all?).
#   S_mom = 1 - (eta_gen - eta_real) / (eta_blind - eta_real), each eta pooled as a ratio of sums over clips that pass
#   the gate (generated AND the real readout of the same twin) with mass ratio >= min_mass_ratio. eta_real = the same
#   readout on the real twin (the readout's own floor), eta_blind = the mass-blind rule on the real outcome
#   (laws.mass_blind_outcome, eta = |m_A - m_B| / (m_A + m_B)). Undefined (NaN) when eta_real >= eta_blind.
#   Group-bootstrap CI.
# Track B (masses unknown): mass-free checks (laws.mass_free_momentum) on the clip's own measured states.
#   contact normal n = the line of centres at contact, from the generated contact POSITIONS (readout_self,
#   :func:`_contact_normal`; real F1 at the Wan geometry: error vs the true normal median 2.3 deg, p90 10).
#   A miss puts n perpendicular to the relative velocity, so u_n = 0 and epsilon is undefined there.
#   gate: |dv_A - dv_B| >= kappa * |u_n|, u_n = (v_A - v_B) . n (the weakest legal collision changes the relative
#   velocity by u_n; |v_A - v_B| if u_n = 0).
#   antiparallel = -cos(dv_A, dv_B) (1 for every legal outcome, any masses), normal alignment = mean of |dv . n| / |dv|,
#   epsilon and the share epsilon <= 1 + e_tol (no energy creation along n, mass-free), eta with the label masses,
#   implied mass ratio |dv_B| / |dv_A| and its spread (SD of the log) across seeds of one group (one scene, one ratio).
#   Real-clip floors: the same on the real twins (§6.1 validation readouts); ratio spread across the legal twins.
MOM_DEFAULTS = {"kappa": 0.25, "min_mass_ratio": 1.25, "e_tol": 0.1}   # e_tol: real e = 1 share 0.91, e = 1.5: 0.05
_A_KEYS = ("gen_num", "gen_den", "real_num", "real_den", "bl_num", "bl_den")


def mom_clip_a(v_gen, v_real, labels: dict) -> dict:
    """S_mom ingredients of one generated outcome (track A) against one legal twin. ``v_real``: that twin's real-clip
    readout (None if not measurable)."""
    from physlaw.eval import laws as Lw
    pre, post, ctx = Lw.from_labels(labels)
    p = pre["p"]
    g = Lw.momentum_exchange(pre, {"v": np.asarray(v_gen, float), "p": p}, ctx)
    b = Lw.momentum_exchange(pre, Lw.mass_blind_outcome(pre, post, ctx), ctx)
    m = np.asarray(ctx["m"], float)
    out = {"gen_num": g["num"], "gen_den": g["den"], "bl_num": b["num"], "bl_den": b["den"], "d_min": g["d_min"],
           "mass_ratio": float(m.max() / m.min()),
           "gen_anti": Lw.mass_free_momentum(pre, {"v": np.asarray(v_gen, float)}, ctx)["antiparallel"],
           "real_num": np.nan, "real_den": np.nan, "real_anti": np.nan}
    if v_real is not None:
        r = Lw.momentum_exchange(pre, {"v": np.asarray(v_real, float), "p": p}, ctx)
        out.update(real_num=r["num"], real_den=r["den"],
                   real_anti=Lw.mass_free_momentum(pre, {"v": np.asarray(v_real, float)}, ctx)["antiparallel"])
    return out


def _eta_S3(t):
    """t [..., 6] pooled sums in _A_KEYS order -> (eta [..., 3] gen / real / blind,
    S_mom = 1 - (eta_gen - eta_real) / (eta_blind - eta_real) clipped to [-1, 1] [...], NaN unless eta_real < eta_blind)."""
    t = np.asarray(t, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(t[..., 1::2] > 0, t[..., 0::2] / t[..., 1::2], np.nan)
        den = eta[..., 2] - eta[..., 1]
        S = np.where(den > 0, np.clip(1 - (eta[..., 0] - eta[..., 1]) / np.where(den > 0, den, 1.0), -1, 1), np.nan)
    return eta, S


def track_a_momentum(gen_rows, real_rows, labels, twins, cfg: dict | None = None, n_boot: int = 2000,
                     seed: int = 0) -> dict:
    """{twin | "legal": S_mom package} for the measurable generated collision outcomes (rows with ``v_post``).
    ``labels(group, twin)`` -> labels.json dict. "legal" pools the twins (each generated clip once per twin)."""
    cfg = {**MOM_DEFAULTS, **(cfg or {})}
    kappa, rmin = float(cfg["kappa"]), float(cfg["min_mass_ratio"])
    real = {(r["group"], r["twin"]): r for r in real_rows}
    gen = [r for r in gen_rows if r.get("measurable") and "v_post" in r]
    per = {}
    for t in twins:
        cl = []
        for r in gen:
            rr = real.get((r["group"], t))
            vr = rr["v_post"] if rr and rr.get("measurable") and "v_post" in rr else None
            cl.append({"group": r["group"], **mom_clip_a(r["v_post"], vr, labels(r["group"], t))})
        fl = []                                            # every measurable real clip of this twin (floor over all)
        for rr in real_rows:
            if rr["twin"] == t and rr.get("measurable") and "v_post" in rr:
                fl.append(mom_clip_a(rr["v_post"], rr["v_post"], labels(rr["group"], t)))
        per[t] = (cl, fl)
    if len(twins) > 1:
        per["legal"] = ([c for t in twins for c in per[t][0]], [c for t in twins for c in per[t][1]])
    out = {}
    for t, (cl, fl) in per.items():
        if not cl and not fl:
            continue
        gate_g = [c["gen_den"] >= kappa * c["d_min"] for c in cl]
        gate_r = [np.isfinite(c["real_den"]) and c["real_den"] >= kappa * c["d_min"] for c in cl]
        use = [c for c, a, b in zip(cl, gate_g, gate_r) if a and b and c["mass_ratio"] >= rmin]
        groups = sorted({c["group"] for c in use})
        G = np.zeros((len(groups), len(_A_KEYS)))
        for c in use:
            G[groups.index(c["group"])] += [c[k] for k in _A_KEYS]
        eta, S = _eta_S3(G.sum(0)) if len(use) else (np.full(3, np.nan), np.nan)
        flu = [c for c in fl if c["real_den"] >= kappa * c["d_min"] and c["mass_ratio"] >= rmin]
        eta_all = (sum(c["real_num"] for c in flu) / sum(c["real_den"] for c in flu)) if flu else np.nan
        bl_all = (sum(c["bl_num"] for c in flu) / sum(c["bl_den"] for c in flu)) if flu else np.nan
        gated_any = [c for c, a, b in zip(cl, gate_g, gate_r) if a]
        nm = lambda v: float(np.nanmean(v)) if len(v) and np.isfinite(v).any() else float("nan")
        d = {"kappa": kappa, "min_mass_ratio": rmin, "n_measurable_generated": len(cl),
             "n_interacting": int(sum(gate_g)),
             "interaction_rate": float(np.mean(gate_g)) if cl else float("nan"),
             "interaction_rate_real": float(np.mean([c["real_den"] >= kappa * c["d_min"] for c in fl])) if fl else float("nan"),
             "n_scored_clips": len(use), "n_scored_groups": len(groups),
             "eta_generated": float(eta[0]), "eta_real": float(eta[1]), "eta_blind": float(eta[2]),
             "eta_real_all": float(eta_all), "eta_blind_all": float(bl_all), "n_real_all": len(flu),
             "S_mom": float(S), "S_mom_defined": bool(np.isfinite(eta[1]) and np.isfinite(eta[2]) and eta[1] < eta[2]),
             "S_mom_ci95": None,
             "antiparallel_generated": nm([c["gen_anti"] for c in gated_any]),
             "antiparallel_real": nm([c["real_anti"] for c in fl if c["real_den"] >= kappa * c["d_min"]]),
             "group_sums": {g: G[i].tolist() for i, g in enumerate(groups)}}
        if n_boot and len(groups):
            rng = np.random.default_rng(seed)
            _, Sb = _eta_S3(G[rng.integers(0, len(G), (n_boot, len(G)))].sum(1))
            if np.isfinite(Sb).any():
                d["S_mom_ci95"] = [float(np.nanquantile(Sb, 0.025)), float(np.nanquantile(Sb, 0.975))]
        out[t] = d
    return out


def _contact_normal(x: np.ndarray, v: np.ndarray, rsum: float) -> tuple[np.ndarray, bool]:
    """Contact normal (A -> B) from positions x [2, 2] and velocities v [2, 2] at the contact frame: run the straight
    pre-contact lines to the first time the centres are ``rsum`` apart (backwards if they overlap there); closest
    approach if they never are (second value True: the lines miss)."""
    d0, w = x[1] - x[0], v[1] - v[0]
    a, b, c = float(w @ w), float(2 * d0 @ w), float(d0 @ d0 - rsum ** 2)
    miss = False
    if a < 1e-18:
        s = 0.0
    else:
        disc = b * b - 4 * a * c
        if disc >= 0:
            s = (-b - np.sqrt(disc)) / (2 * a)
        else:
            s, miss = -b / (2 * a), True
    dn = d0 + w * s
    nn = float(np.linalg.norm(dn))
    return (dn / nn if nn > 0 else np.full(2, np.nan)), miss


def line_of_centres(x0, v0, x1, v1, rsum: float) -> tuple[np.ndarray, bool]:
    """Contact normal (A -> B) = the line of centres at contact: the pre lines (positions x0 [2, 2] at the contact
    frame, velocities v0) run forward and the post lines (x1, v1) run back to r_A + r_B apart, averaged (halves the
    real p99 error). Second value: both line pairs miss (closest approach used)."""
    x0, v0, x1, v1 = (np.asarray(a, float) for a in (x0, v0, x1, v1))
    n, miss = _contact_normal(x0, v0, rsum)
    n1, miss1 = _contact_normal(x1, -v1, rsum)
    if np.isfinite(n1).all() and np.isfinite(n).all() and np.linalg.norm(n + n1) > 1e-9:
        n, miss = (n + n1) / np.linalg.norm(n + n1), miss and miss1
    return n, miss


def mass_free_clip(r: dict, labels: dict, cfg: dict | None = None) -> dict:
    """Track B momentum gate fields (``mf_*``) of one measurable readout_self row (velocity_pre / _post and its
    line-of-centres contact_normal); {} otherwise. The statistics themselves (epsilon, eta, antiparallel, normal_A/B)
    are on the row (readout_self)."""
    if not r.get("measurable") or "contact_normal" not in r or "velocity_post" not in r:
        return {}
    from physlaw.eval.laws import mass_free_momentum
    cfg = {**MOM_DEFAULTS, **(cfg or {})}
    v0, v1, n = (np.asarray(r[k], float) for k in ("velocity_pre", "velocity_post", "contact_normal"))
    dv = v1 - v0
    u_n = float((v0[0] - v0[1]) @ n)
    d_rel = float(np.linalg.norm(dv[0] - dv[1]))
    d_min = abs(u_n) if abs(u_n) > 1e-9 else float(np.linalg.norm(v0[0] - v0[1]))
    ratio = mass_free_momentum({"v": v0}, {"v": v1}, {"n": n})["implied_ratio"]
    out = {"mf_u_n": u_n, "mf_dv_rel": d_rel, "mf_d_min": d_min, "mf_interacts": bool(d_rel >= float(cfg["kappa"]) * d_min),
           "mf_contact_miss": bool(r.get("contact_miss", False)), "mf_implied_ratio": ratio,
           "mf_normal": float(np.nanmean([r.get("normal_A", np.nan), r.get("normal_B", np.nan)]))
           if np.isfinite([r.get("normal_A", np.nan), r.get("normal_B", np.nan)]).any() else float("nan")}
    pl = labels["physlaw"]
    m = [pl["objects"][o]["mass"] for o in "AB"]
    if np.isfinite(ratio) and ratio > 0:
        out["mf_log_ratio_vs_label"] = float(np.log(ratio) - np.log(m[0] / m[1]))
    if "contact_normal" in pl:                                              # validation of the positional normal
        nt = np.asarray(pl["contact_normal"], float)[:2]
        out["mf_normal_err_deg"] = float(np.degrees(np.arccos(np.clip(abs(n @ nt) / np.linalg.norm(nt), -1, 1))))
    return out


def mass_free_stats(rows: list[dict], cfg: dict | None = None, n_boot: int = 2000, seed: int = 0) -> dict:
    """Pooled track B momentum package over rows carrying ``mf_*`` fields (see :func:`mass_free_clip`). Consistency:
    rows sharing a ``group`` (seeds of one prompt / frame 0, or the legal twins of a real scene) should imply one
    mass ratio: ratio_log_sd = pooled within-group SD of log |dv_B| / |dv_A| over groups with >= 2 gated clips."""
    cfg = {**MOM_DEFAULTS, **(cfg or {})}
    m = [r for r in rows if "mf_interacts" in r]
    gt = [r for r in m if r["mf_interacts"]]
    fin = lambda k: np.array([r[k] for r in gt if np.isfinite(r.get(k, np.nan))], float)
    d = {"n_measurable": len(m), "n_interacting": len(gt),
         "interaction_rate": len(gt) / len(m) if m else float("nan"), "n_contact_miss": int(sum(r["mf_contact_miss"] for r in gt))}
    if not gt:
        return d
    tol = float(cfg["e_tol"])
    for k, name in (("antiparallel", "antiparallel"), ("mf_normal", "normal_alignment"), ("epsilon", "epsilon"),
                    ("eta", "eta")):
        v = fin(k)
        if len(v):
            d[f"{name}_mean"], d[f"{name}_median"] = float(v.mean()), float(np.median(v))
    e = fin("epsilon")
    if len(e):
        d["e_le1_share"] = float((e <= 1 + tol).mean())
        d["e_in_range_share"] = float(((e >= -tol) & (e <= 1 + tol)).mean())     # also not still approaching
        d["e_tol"] = tol
    by_g = {}
    for r in gt:
        by_g.setdefault(r["group"], []).append(r)
    stat = lambda key, f: _boot_ci({g: [x[key] for x in v if np.isfinite(x.get(key, np.nan))] for g, v in by_g.items()},
                                   lambda xs: f(np.asarray(xs, float)) if len(xs) else np.nan, n_boot, seed)
    if n_boot:
        d["antiparallel_mean_ci95"] = stat("antiparallel", np.mean)
        if len(e):
            d["e_le1_share_ci95"] = stat("epsilon", lambda x: (x <= 1 + tol).mean())
    lr = fin("mf_implied_ratio")
    if len(lr):
        d["implied_ratio_median"] = float(np.median(lr))
    if any("mf_log_ratio_vs_label" in r for r in gt):
        d["log_ratio_vs_label_median_abs"] = float(np.median(np.abs(fin("mf_log_ratio_vs_label"))))
    if any("mf_normal_err_deg" in r for r in gt):
        d["normal_err_deg_median"] = float(np.median(fin("mf_normal_err_deg")))
    ss, dof, ng = 0.0, 0, 0
    for g, v in by_g.items():
        x = np.log([r["mf_implied_ratio"] for r in v if np.isfinite(r["mf_implied_ratio"]) and r["mf_implied_ratio"] > 0])
        if len(x) >= 2:
            ss, dof, ng = ss + float(((x - x.mean()) ** 2).sum()), dof + len(x) - 1, ng + 1
    d["ratio_consistency_groups"] = ng
    d["ratio_log_sd"] = float(np.sqrt(ss / dof)) if dof else float("nan")
    return d


class Geometry:
    """Model frame sampling for the tracker, without loading the model."""

    def __init__(self, model_cfg):
        from physlaw.models.base import sample_frames
        off = model_cfg["official"]
        self.fps, self.n = int(off["fps"]), int(off["num_frames"])
        self.H, self.W = int(off["height"]), int(off["width"])
        self.center = bool(model_cfg.get("input", {}).get("center_on_contact", False))
        tg = model_cfg["token_grid"]
        self.vt = int(tg["vae_temporal"]) * int(tg.get("patch_t") or 1)   # input frames per token-frame
        self._sf = sample_frames
        self.fps_hw = (self.fps, self.H, self.W)

    def __call__(self, labels):
        sc, pl = labels["scene_constants"], labels["physlaw"]
        fr = self._sf(master_fps=sc["frame_rate"], n_master=sc["num_frames"], fps=self.fps, n_frames=self.n,
                      contact_frame=pl["first_post_contact_frame"], center_on_contact=self.center)
        k = sum(f < pl["first_post_contact_frame"] for f in fr)
        k = 1 + ((k - 1) // self.vt) * self.vt          # whole causal latent frames, as the adapter's context
        return fr, k


# --------------------------------------------------------------------------- #
# Output track B: self-consistency (spec §6b, 2026-11-06)
# --------------------------------------------------------------------------- #
# Ported from the collision readout validated on 48 real energy-ladder renders in the predecessor project
# (scripts/elasticity/collide_gate.readout, median |e error| 0.010): fast-ball-aware contact, short windows on both
# sides of the merged stretch, contact normal from the velocity changes, whole-path straightness, jitter gate.
# Thresholds in model pixels (the predecessor's 0.02 / 0.04 world units were 1 / 2 px at 50 px per unit).
TB_WINDOW_S = 0.75
TB_MAX_RESID_PX = 1.0
TB_MAX_LATERAL_PX = 2.0
TB_MIN_IMPULSE_COS = 0.95
# LOOSE gate (2026-11-06, user decision): only MEASUREMENT failures drop a clip (size change, extra objects, lost tracks,
# no collision / no separation, too few frames, tracker jitter). Law breaks the strict gate also dropped (impulses
# not opposite, still approaching after contact, curved paths, a ball accelerating on its own) are KEPT and listed in
# `violations`. Jitter vs smooth self-acceleration: each window gets a quadratic fit; its residual (<= TB_MAX_JITTER_PX)
# is the jitter test, and the velocity is the fit's value AT CONTACT (the instantaneous velocity the laws refer to).
TB_MAX_JITTER_PX = 1.0           # straight line vs quadratic switch (real tracker noise: p99 0.74, max 0.96 px)
TB_QUAD_GAIN = 0.6               # quadratic only when it cuts the line's residual to <= 0.6x (real curvature)
TB_MAX_VEL_SE = 0.07              # jitter DROP: velocity standard error > this x the scene's top speed. Wan's frames
                                  # stutter by 1-2 px (uneven per-frame steps, in the pixels, not the tracker), which a
                                  # 12-frame fit averages to ~0.2 px/frame on 2-6 px/frame speeds: still measurable.
                                  # Calibrated by adding Gaussian position noise to 25 real conserve_e1 clips:
                                  # 1 px -> 19/25 kept, |e-1| p90 0.16; 2 px -> 9/25, p90 0.24; 4 px -> 0/25
TB_MIN_WINDOW_LOOSE = 5
TB_MAX_SELF_ACCEL = 0.25          # "self_acceleration": a window where the straight line misses (> TB_MAX_RESID_PX), a
                                  # quadratic fits, and the velocity changes by > this x the scene's top speed
TB_LOOSE_MIN_E = -0.25            # "still_approaching" below this; real e = 0.15 twins whose balls stay in shadow-merge
                                  # distance measure down to -0.17 (6/318 real clips), so -0.05 would flag real physics


def _line_at(arr: np.ndarray, win: list[int], frame: float, fps: float) -> np.ndarray:
    """Straight-line least-squares fit of arr[win] over time, evaluated at ``frame``."""
    t = np.array(win, float) / fps
    coef, *_ = np.linalg.lstsq(np.stack([t, np.ones_like(t)], 1), arr[win], rcond=None)
    return coef[0] * frame / fps + coef[1]


def _lateral_rms(pts: np.ndarray) -> float:
    pts = pts[np.isfinite(pts).all(1)]
    if len(pts) < 5:
        return float("nan")
    _, sv, _ = np.linalg.svd(pts - pts.mean(0), full_matrices=False)
    return float(sv[-1] / np.sqrt(len(pts)))


def readout_self(gen_frames: np.ndarray, clip, fps: float, checks: bool = True, gate: str = "strict") -> dict:
    """Track B readout of a clip generated from frame 0 (gen_frames[0] ~ the real frame 0, same size as the
    master). Velocities and the line of centres are the clip's own; masses are the true ones (visible in F1: size +
    material). Law statistics on the row: :func:`collision_stats`.
    gate="strict": any failed check drops the clip. gate="loose": see TB_MAX_JITTER_PX above."""
    assert gate in ("strict", "loose")
    loose = gate == "loose"
    labels = clip.labels
    H, W = gen_frames.shape[1:3]
    T = len(gen_frames)
    _, sig, start = clip_inputs(clip, [0], H, W, "squash", 1)
    tr = track(gen_frames, sig, start)
    pix = tr["pixel"]                                                   # [T, 2, 2] (row, col)
    n_master = labels["scene_constants"]["num_frames"]
    maps = _world_maps(labels, list(range(0, n_master, 4)), H, W, "squash")
    world = np.full(pix.shape, np.nan)
    for i, o in enumerate("AB"):
        world[:, i] = np.concatenate([pix[:, i], np.ones((T, 1))], 1) @ maps[o]
    pl = labels["physlaw"]
    m = np.array([pl["objects"][o]["mass"] for o in "AB"], float)
    rA, rB = (sig[o]["radius_px"] for o in "AB")
    r_sum_px = rA + rB
    dpx = np.linalg.norm(pix[:, 0] - pix[:, 1], axis=1)
    ok = np.isfinite(dpx)
    margin = 1.2 * max(rA, rB)                                          # centre this close to the border: partly out
    inside = np.all((pix[..., 0] > margin) & (pix[..., 0] < H - margin) & (pix[..., 1] > margin)
                    & (pix[..., 1] < W - margin), axis=1)
    full = ok & inside
    ref = float(tr["fg_area"][0])
    res = {"frames_both_tracked": int(ok.sum()), "size_ratio": float(np.max(tr["fg_area"][1:]) / ref) if ref > 0 else float("nan"),
           "extra_object_frames": int((tr["n_ball"][1:] > 2).sum())}
    if checks and res["size_ratio"] > SIZE_RATIO_MAX:
        return {**res, "measurable": False, "reason": "size_change"}
    if checks and res["extra_object_frames"] > EXTRA_FRAMES_MAX:
        return {**res, "measurable": False, "reason": "extra_objects"}
    if ok.sum() < 10:
        return {**res, "measurable": False, "reason": "lost"}
    c0 = c1 = None
    for g in np.where(ok[:-1] & ~ok[1:])[0]:
        nxt = np.where(ok[g + 1:])[0]
        end = g + 1 + int(nxt[0]) if len(nxt) else T
        near = dpx[g] <= 1.5 * r_sum_px
        hist = [i for i in range(max(0, g - 3), g + 1) if ok[i]]
        if not near and len(hist) >= 2:
            rel = pix[hist, 1] - pix[hist, 0]
            vrel = (rel[-1] - rel[0]) / (hist[-1] - hist[0])
            steps = np.arange(1, min(end, T) - g + 1)[:, None]
            near = bool(np.min(np.linalg.norm(rel[-1] + steps * vrel, axis=1)) <= 1.25 * r_sum_px)
        if near:
            c0, c1 = g + 1, end
            break
    if c0 is None and np.nanmin(dpx) <= 1.25 * r_sum_px:
        c0 = c1 = int(np.nanargmin(dpx))
    res.update(collided=c0 is not None, min_dist_over_rsum=float(np.nanmin(dpx) / r_sum_px))
    if c0 is None:
        return {**res, "measurable": False, "reason": "no_collision"}
    res.update(contact_frame=int(c0), separate_frame=int(c1), merged_frames=int(c1 - c0))
    if c1 >= T:
        return {**res, "measurable": False, "reason": "never_separate"}
    Wn = max(4, int(round(TB_WINDOW_S * fps)))
    gap = 1 if c1 > c0 else 3          # contact from closest approach (no merge): stay 3 frames clear of it
    prew = [i for i in range(max(1, c0 - gap - Wn), c0 - gap) if full[i]]
    postw = [i for i in range(c1 + gap, min(T, c1 + gap + Wn)) if full[i]]
    res.update(n_pre=len(prew), n_post=len(postw))
    if len(prew) < (TB_MIN_WINDOW_LOOSE if loose else 4) or len(postw) < (TB_MIN_WINDOW_LOOSE if loose else 4):
        return {**res, "measurable": False, "reason": "short_window"}

    def fit(arr, win):
        t = np.array(win, float) / fps
        A = np.stack([t, np.ones_like(t)], 1)
        coef, *_ = np.linalg.lstsq(A, arr[win], rcond=None)
        return coef[0], float(np.sqrt(((A @ coef - arr[win]) ** 2).sum(1).mean()))
    def fit2(win, t_ref, k):
        # ball k in one window. Straight line unless it misses by > TB_MAX_RESID_PX and a quadratic does not (a ball
        # speeding up / slowing / bending on its own): then the quadratic's velocity AT the contact-side edge.
        # Returns (world velocity, px residual of the chosen model, |velocity change across the window| world, quad?)
        t = (np.array(win, float) - t_ref) / fps
        A1 = np.stack([t, np.ones_like(t)], 1)
        A2 = np.stack([t ** 2, t, np.ones_like(t)], 1)
        out = []
        for A, vi in ((A1, 0), (A2, 1)):
            cp, *_ = np.linalg.lstsq(A, pix[win, k], rcond=None)
            cw, *_ = np.linalg.lstsq(A, world[win, k], rcond=None)
            rr = A @ cp - pix[win, k]
            sig2 = float((rr ** 2).sum() / max(2 * (len(win) - A.shape[1]), 1))        # per-axis noise variance
            se_px = float(np.sqrt(2 * sig2 * np.linalg.inv(A.T @ A)[vi, vi])) / fps    # px / frame
            out.append((cw, float(np.sqrt((rr ** 2).sum(1).mean())), se_px, float(np.linalg.norm(cp[vi])) / fps))
        (c1w, r1, se1, sp1), (c2w, r2, se2, sp2) = out
        span = float(t.max() - t.min())
        dv = float(np.linalg.norm(2 * c2w[0] * span))
        if r1 <= TB_MAX_RESID_PX or r2 > TB_QUAD_GAIN * r1:          # noise alone: the quadratic barely helps
            return c1w[0], r1, dv, False, se1, sp1
        return c2w[1], r2, dv, True, se2, sp2
    if loose:
        tp, tq = c0 - gap, c1 + gap                                     # window edges nearest the contact
        F = {(k, w): fit2(win, tref, k) for k in (0, 1) for w, win, tref in (("pre", prew, tp), ("post", postw, tq))}
        v0 = np.stack([F[(k, "pre")][0] for k in (0, 1)])
        v1 = np.stack([F[(k, "post")][0] for k in (0, 1)])
        resid_px = max(f[1] for f in F.values())
        vscale = max(np.linalg.norm(v0[0]), np.linalg.norm(v0[1]), np.linalg.norm(v1[0]), np.linalg.norm(v1[1]), 1e-12)
        res["self_accel"] = max(f[2] for f in F.values()) / vscale      # vs the scene's speed, not the ball's own
        res["quadratic_windows"] = [f"{'AB'[k]}_{w}" for (k, w), f in F.items() if f[3]]
        res["vel_se_rel"] = max(f[4] for f in F.values()) / max(max(f[5] for f in F.values()), 1e-12)
    else:
        v0 = np.stack([fit(world[:, k], prew)[0] for k in (0, 1)])
        v1 = np.stack([fit(world[:, k], postw)[0] for k in (0, 1)])
        resid_px = max(fit(pix[:, k], w)[1] for k in (0, 1) for w in (prew, postw))
    lat = [_lateral_rms(pix[[i for i in range(1, c0 - 1) if full[i]], k]) for k in (0, 1)] + \
          [_lateral_rms(pix[[i for i in range(c1 + 1, T) if full[i]], k]) for k in (0, 1)]
    # positions at the contact frame from straight-line fits of each window (F1 add-on laws: r x v is constant
    # along a straight path, so angular momentum does not depend on this choice of time)
    x0 = np.stack([_line_at(world[:, k], prew, c0, fps) for k in (0, 1)])
    x1 = np.stack([_line_at(world[:, k], postw, c0, fps) for k in (0, 1)])
    res.update(position_pre=x0.tolist(), position_post=x1.tolist(), mass=m.tolist())
    dA, dB = v1[0] - v0[0], v1[1] - v0[1]
    rel0 = v0[0] - v0[1]
    res.update(velocity_pre=v0.tolist(), velocity_post=v1.tolist(), fit_resid_px=resid_px,
               path_lateral_px=dict(zip(["A_pre", "B_pre", "A_post", "B_post"], lat)),
               path_lateral_max_px=float(np.nanmax(lat)) if np.isfinite(lat).any() else float("nan"))
    if np.linalg.norm(dB - dA) < 0.05 * max(np.linalg.norm(rel0), 1e-12):
        return {**res, "measurable": False, "reason": "no_impulse"}
    # contact normal = the line of centres at contact (never the impulse direction: it hides sideways kicks and
    # biases e upward): pre lines run forward and post lines run back to r_A + r_B apart, averaged
    n, miss = line_of_centres(x0, v0, x1, v1, float(pl["objects"]["A"]["radius"] + pl["objects"]["B"]["radius"]))
    if not np.isfinite(n).all():
        return {**res, "measurable": False, "reason": "no_e"}
    cos = float(-dA @ dB / max(np.linalg.norm(dA) * np.linalg.norm(dB), 1e-12))
    st = collision_stats(v0, v1, n, m)
    e = st["epsilon"]
    res.update(contact_normal=n.tolist(), contact_miss=bool(miss), impulse_cos=cos, approach_speed=float(rel0 @ n), **st)
    # a ball whose velocity barely changes (|dv| < 25% of the other's) has a noise-dominated dv direction:
    # the opposite-impulse check is only applied when both changes are substantial (real F1, 2026-11-06)
    small = min(np.linalg.norm(dA), np.linalg.norm(dB)) < 0.25 * max(np.linalg.norm(dA), np.linalg.norm(dB))
    res["impulse_check"] = not small
    if loose:
        if not np.isfinite(e):
            return {**res, "measurable": False, "reason": "no_e"}
        if res["vel_se_rel"] > TB_MAX_VEL_SE:
            return {**res, "measurable": False, "reason": "jitter"}
        viol = [k for k, good in [("impulse_not_opposite", small or cos >= TB_MIN_IMPULSE_COS),
                                  ("still_approaching", e >= TB_LOOSE_MIN_E),
                                  ("curved_path", not np.isfinite(res["path_lateral_max_px"])
                                   or res["path_lateral_max_px"] <= TB_MAX_LATERAL_PX),
                                  ("self_acceleration", not res["quadratic_windows"]
                                   or res["self_accel"] <= TB_MAX_SELF_ACCEL)] if not good]
        return {**res, "measurable": True, "reason": None, "gate": "loose", "violations": viol}
    bad = [k for k, good in [("no_e", np.isfinite(e)), ("fit_residual", resid_px <= TB_MAX_RESID_PX),
                             ("impulse_not_opposite", small or cos >= TB_MIN_IMPULSE_COS),
                             ("still_approaching", not np.isfinite(e) or e >= -0.05),
                             ("curved_path", not np.isfinite(res["path_lateral_max_px"])
                              or res["path_lateral_max_px"] <= TB_MAX_LATERAL_PX)] if not good]
    return {**res, "measurable": not bad, "reason": ",".join(bad) or None}


# --------------------------------------------------------------------------- #
# Output track B stage drivers (scripts/08, scripts/09)
# --------------------------------------------------------------------------- #

READOUT_SELF = {f: readout_self for f in ("f1", "f2", "f3", "f4")}   # family -> track B readout; F6-F12 below
GATES = ("strict", "loose")


def readout_file(gate: str, real: bool = False) -> str:
    """Readout file names in a track B directory (the 2026-11-06 names, so old readouts load unchanged)."""
    return "_readout" + ("_real" if real else "") + ("" if gate == "strict" else f"_{gate}") + ".jsonl"


def real_video(clip, fps: float, hw: tuple[int, int] | None = None) -> np.ndarray:
    """A real master clip resampled to ``fps`` (exact stride) and, if given, squashed to ``hw``: the §6.1
    readout validation input (frame 0 = master frame 0, as in a frame-0 generation)."""
    sc = clip.labels["scene_constants"]
    step = int(sc["frame_rate"]) // int(fps)
    if step * int(fps) != int(sc["frame_rate"]):
        raise ValueError(f"fps {fps} does not divide the master rate {sc['frame_rate']}")
    v = clip.frames[::step]
    return v if hw is None or tuple(v.shape[1:3]) == tuple(hw) else fit_frames(v, hw, "squash")


def track_b_summary(rows: dict, real_rows: dict, data_dir, mom_cfg: dict | None = None, n_boot: int = 2000,
                    energy_cfg: dict | None = None) -> dict:
    """rows / real_rows: {gate: [readout rows]}. Per gate: counts, drop reasons, medians of every law statistic,
    loose-gate violations, the energy share against the real-clip distribution (:func:`energy_shares`) and, for
    collisions, the mass-free momentum package ``momentum`` (:func:`mass_free_stats`; per-row ``mf_*`` gate fields).
    Real clips (§6.1 validation, if given): the same per twin, ``real.legal`` over every legal clip (the ratio laws'
    floor) and ``momentum.real_legal`` pooling the legal twins."""
    labels = _labels_fn(data_dir)
    first = {}
    for g_rows in real_rows.values():
        for r in g_rows:
            first.setdefault(r["group"], r["twin"])
    lab = lambda r: labels(r["group"], r.get("twin") or first.get(r["group"]) or "conserve_e1")
    mcfg = {**MOM_DEFAULTS, **(mom_cfg or {})}
    tol = float({**ENERGY_DEFAULTS, **(energy_cfg or {})}["tol"])
    out = {"momentum_config": mcfg, "energy_tol": tol, "gates": {}}
    for gate in [g for g in GATES if g in rows or g in real_rows]:
        G = [{**r, **mass_free_clip(r, lab(r), mcfg)} for r in rows.get(gate, [])]
        R = [{**r, **mass_free_clip(r, lab(r), mcfg), "eps_true": true_epsilon(lab(r))} for r in real_rows.get(gate, [])]
        d = {"generated": _stats(G), "measurable_clips": [f"{r['group']}/seed{r['seed']}" for r in G if r["measurable"]],
             "rows": G, "energy": energy_shares([r for r in G if r["measurable"]], [r for r in R if r["measurable"]], tol, n_boot)}
        if any("mf_interacts" in r for r in G + R):
            d["momentum"] = {"generated": mass_free_stats(G, mcfg, n_boot)}
        if R:
            tw = sorted({r["twin"] for r in R})
            d["real"] = {t: _stats([r for r in R if r["twin"] == t]) for t in tw}
            d["real"]["all"] = _stats(R, ())
            d["real"]["legal"] = _stats([r for r in R if lab(r)["physlaw"].get("legal")])   # per clip (F11: per group)
            if "momentum" in d:
                d["momentum"]["real"] = {t: mass_free_stats([r for r in R if r["twin"] == t], mcfg, n_boot) for t in tw}
                legal = [t for t in tw if t in LEGAL] or [t for t in tw if labels(R[0]["group"], t)["physlaw"].get("legal")]
                if legal:
                    d["momentum"]["real_legal"] = mass_free_stats([r for r in R if r["twin"] in legal], mcfg, n_boot)
        out["gates"][gate] = d
    return out


# --------------------------------------------------------------------------- #
# F7 / F8 readout: one or two objects moving in a vertical plane, side camera (spec §6.1 per-family readouts)
# --------------------------------------------------------------------------- #
# Pixels -> world by back-projecting through the labelled pinhole camera onto the plane at the object's mean camera
# depth (exact for motion in a plane parallel to the image, as in the side views of F7 / F8). Per-object tracking:
# foreground = colour distance to the clip's per-pixel median (static camera) > TR_FG; each object takes the blob of
# ball size nearest in colour + position to its last position. Then :func:`traj_stats` per sub-family (scale-free:
# every statistic is a ratio of one quantity measured twice, so the length unit and the time unit drop out):
#   projectile / drop  parabola z(t), line x(t) on a window before and after the branch -> laws.gravity g_ratio,
#                      axis_ratio, mass_ratio (drop: heavier / lighter)
#   bounce             parabolas on the fall before and the rise after the first floor contact -> laws.epsilon
#                      (bounce) of the velocities at contact; epsilon_height = sqrt(rebound height / drop height)
#   ramp (F8, Galileo) epsilon = sqrt((apex - pad level) / (release - pad level)), heights from the track;
#                      epsilon_speed = laws.epsilon (pad) of the run-out speeds before the pad and after it
TR_FG = 30.0
TR_WINDOW_S = 0.5
TR_MIN_FRAMES = 5
TR_MAX_RESID_PX = 1.5            # strict gate: rms px residual of every fit


def master_fps(labels) -> int:
    sc = labels.get("scene_constants") or {}
    return int(sc.get("frame_rate") or labels["physlaw"]["constants"]["fps"])


def backproject(pix_rc: np.ndarray, labels: dict, H: int, W: int, depth: float) -> np.ndarray:
    """Model-pixel (row, col) [..., 2] -> world xyz [..., 3] on the plane at camera depth ``depth``."""
    cam = labels["physlaw"]["camera"]
    K, E = np.asarray(cam["K"], float), np.asarray(cam["world_to_camera"], float)
    W0, H0 = cam["resolution"]
    uv1 = np.stack([pix_rc[..., 1] * W0 / W, pix_rc[..., 0] * H0 / H, np.ones(pix_rc.shape[:-1])], -1)
    xc = depth * (uv1 @ np.linalg.inv(K).T)
    return (xc - E[:3, 3]) @ E[:3, :3]                                     # R^T (x_cam - t), row-wise


def _track_world(gen_frames: np.ndarray, clip, fps: float, frames_idx: list | None = None, objs: list | None = None,
                 world: bool = True) -> dict:
    """Shared set-up of the per-family readouts: frame i of ``gen_frames`` shows master frame frames_idx[i] (default:
    stride from frame 0); the REAL clip's frames / masks on the same frames give the signatures and the background;
    ``objs`` default: every non-``aux`` object (rods, props). Returns pl, labels, objs, frames_idx, pix [T, n, 2]
    (row, col; NaN = lost), ok [T, n], world [T, n, 3] (back-projected, if ``world``), bg, t [T] (clip time of each
    frame: the master frame index / master rate, so the time unit is the clip's own), te (event time), ke (first
    frame after the event), end (one past the last frame <= clean_until_s), wn (TR_WINDOW_S in frames)."""
    labels, pl = clip.labels, clip.labels["physlaw"]
    H, W = gen_frames.shape[1:3]
    mf = master_fps(labels)
    frames_idx = list(frames_idx or range(0, (mf // int(fps)) * len(gen_frames), mf // int(fps)))[: len(gen_frames)]
    objs = objs or [o for o in sorted(pl["objects"]) if not pl["objects"][o].get("aux")]
    ref = fit_frames(clip.frames[frames_idx], (H, W), "squash")
    rmask = fit_frames(np.repeat(clip.masks[frames_idx][..., None], 3, -1), (H, W), "squash")[..., 0]
    bg = background(gen_frames, ref, rmask)
    pix = track_objects(gen_frames, labels, objs, frames_idx, ref, rmask, bg)
    end = int(np.searchsorted(frames_idx, float(pl.get("clean_until_s", np.inf)) * mf, side="right"))
    out = {"pl": pl, "labels": labels, "objs": objs, "frames_idx": frames_idx, "pix": pix, "ok": np.isfinite(pix).all(-1),
           "bg": bg, "ref": ref, "rmask": rmask, "t": np.asarray(frames_idx, float) / mf,
           "te": float(pl.get("event_time_s", (pl["first_post_contact_frame"] - 0.5) / mf)),
           "ke": int(np.searchsorted(frames_idx, pl["first_post_contact_frame"])), "end": min(end, len(gen_frames)),
           "wn": max(TR_MIN_FRAMES, int(round(TR_WINDOW_S * fps)))}
    if world:
        out["world"] = np.stack([backproject(pix[:, k], labels, H, W, _object_depth(labels, o))
                                 for k, o in enumerate(objs)], 1)
    return out


def _object_depth(labels, o) -> float:
    E = np.asarray(labels["physlaw"]["camera"]["world_to_camera"], float)
    P = np.array([f[o]["position"] for f in labels["per_frame"]], float)
    return float(np.median(P @ E[2, :3] + E[2, 3]))


def background(frames: np.ndarray, ref_frames: np.ndarray, ref_masks: np.ndarray) -> np.ndarray:
    """Per-pixel median of the REAL reference frames with the objects masked out (an object that comes to rest would
    otherwise become background); pixels the objects always cover (an object that never moves, e.g. an F11 block that
    stays) take the median background colour of the other pixels, or, with no usable reference, the clip median.
    float32 [H, W, 3]."""
    import warnings
    sub = slice(None, None, max(1, len(ref_frames) // 24))                 # ~24 frames are enough (memory)
    if ref_frames.shape[1:3] != frames.shape[1:3]:
        return np.median(frames[sub], 0).astype(np.float32)
    ref = np.where((ref_masks[sub] > 0)[..., None], np.nan, ref_frames[sub].astype(np.float32))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bg = np.nanmedian(ref, 0)
    fill = np.median(bg[np.isfinite(bg).all(-1)], 0) if np.isfinite(bg).all(-1).any() else np.median(frames[sub], 0)
    return np.where(np.isfinite(bg), bg, fill).astype(np.float32)


def _foreground(img: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """uint8 [H, W]: colour distance to the background > TR_FG, opened 3x3."""
    import cv2
    fg = (np.linalg.norm(img.astype(np.float32) - bg, axis=2) > TR_FG).astype(np.uint8)
    return cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def track_objects(frames: np.ndarray, labels: dict, objs: list, frames_idx: list, ref_frames: np.ndarray,
                  ref_masks: np.ndarray, bg: np.ndarray | None = None) -> np.ndarray:
    """Pixel centres [T, n_obj, 2 (row, col)] (NaN = not found). Signatures (colour, radius) from the REAL reference
    frames / instance masks (the frame where each object shows most: an object entering the view is cut by the edge
    at first); start = its labelled pixel in the first frame where it is visible. ``bg``: :func:`background` if
    already computed."""
    import cv2
    T, H, W = frames.shape[:3]
    sid = {o: labels["physlaw"]["objects"][o]["segmentation_id"] for o in objs}
    W0, H0 = labels["physlaw"]["camera"]["resolution"]
    sig, prev = {}, {}
    for o in objs:
        area = (ref_masks == sid[o]).sum((1, 2))
        t0 = int(np.argmax(area > 0)) if area.any() else 0
        m = ref_masks[int(np.argmax(area))] == sid[o]
        ref0 = ref_frames[int(np.argmax(area))]
        sig[o] = (ref0[m].reshape(-1, 3).astype(float).mean(0) if m.any() else np.full(3, np.nan),
                  float(np.sqrt(max(m.sum(), 1) / np.pi)))
        px = labels["per_frame"][frames_idx[t0]][o]["pixel"]
        prev[o] = np.array([px[1] * H / H0, px[0] * W / W0])
    bg = background(frames, ref_frames, ref_masks) if bg is None else bg
    pos = np.full((T, len(objs), 2), np.nan)
    for t in range(T):
        img = frames[t].astype(np.float32)
        n, lab, stats, cents = cv2.connectedComponentsWithStats(_foreground(img, bg), connectivity=8)
        blobs = [(cents[i][::-1], img[lab == i].mean(0), stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)]
        taken = set()
        for k, o in enumerate(objs):
            rgb, r = sig[o]
            area = np.pi * r * r
            cand = [(np.linalg.norm(c - prev[o]) / r + np.linalg.norm(col - rgb) / 30.0, i)
                    for i, (c, col, a) in enumerate(blobs) if 0.3 * area <= a <= 3.0 * area and i not in taken]
            if cand:
                cost, i = min(cand)
                if cost < 6.0:
                    taken.add(i)
                    pos[t, k] = prev[o] = blobs[i][0]
    return pos


def _fit(t, y, deg):
    """Polynomial fit; returns (coef highest first, rms residual)."""
    c = np.polyfit(t, y, deg)
    return c, float(np.sqrt(((np.polyval(c, t) - y) ** 2).mean()))


def traj_stats(sub: str, t: np.ndarray, P: np.ndarray, ok: np.ndarray, ke: int, te: float, wn: int,
               masses=None, t_release: float | None = None, end: int | None = None) -> dict:
    """Law statistics of tracked trajectories. t [T] (any time unit), P [T, n_obj, 2] (horizontal, up; any length
    unit), ok [T, n_obj] tracked, ke = first frame after the event, te = event time, wn = window (frames). F7: only
    frames in free flight enter the fits: pre frames at or after ``t_release`` (drop / bounce balls are held at rest
    before it), post frames before ``end`` (one past the last frame of the clean window: no floor contact).
    Returns the statistics plus ``windows`` {object index: [frame lists]} for the caller's residual gate, or
    {"reason": ...} when not measurable."""
    from physlaw.eval import laws
    g = np.array([0.0, -1.0])                                             # the direction of gravity only
    T, n_obj = P.shape[:2]
    if sub in ("projectile", "drop", "bounce"):
        F, win = [], {}
        for k in range(n_obj):
            gap = 1 if sub == "bounce" else 0                             # skip the frames touching the floor
            pre = [i for i in range(max(0, ke - gap - wn), ke - gap)
                   if ok[i, k] and (t_release is None or t[i] >= t_release - 1e-9)]
            post = [i for i in range(ke + gap, min(T if end is None else end, ke + gap + wn)) if ok[i, k]]
            if sub == "bounce" and len(post) > 2:                         # rise only: stop at the next floor contact
                zc = P[post[0], k, 1]
                back = [j for j, i in enumerate(post[2:], 2) if P[i, k, 1] <= zc]
                post = post[: back[0]] if back else post
            if len(pre) < TR_MIN_FRAMES or len(post) < TR_MIN_FRAMES:
                return {"reason": "short_window"}
            cz0, cz1 = np.polyfit(t[pre], P[pre, k, 1], 2), np.polyfit(t[post], P[post, k, 1], 2)
            cx0, cx1 = np.polyfit(t[pre], P[pre, k, 0], 1), np.polyfit(t[post], P[post, k, 0], 1)
            F.append({"v0": [cx0[0], np.polyval(np.polyder(cz0), te)], "v1": [cx1[0], np.polyval(np.polyder(cz1), te)],
                      "a0": [0.0, 2 * cz0[0]], "a1": [0.0, 2 * cz1[0]], "z_c": float(np.polyval(cz0, te)),
                      "z_top": float(np.nanmax(np.where(ok[:ke, k], P[:ke, k, 1], np.nan))),
                      "apex": float(cz1[2] - cz1[1] ** 2 / (4 * cz1[0])) if cz1[0] < 0 else float("nan")})
            win[k] = [pre, post]
        pre = {"v": np.array([f["v0"] for f in F]), "a": np.array([f["a0"] for f in F])}
        post = {"v": np.array([f["v1"] for f in F]), "a": np.array([f["a1"] for f in F])}
        if sub == "bounce":
            f0 = F[0]
            h0, h1 = f0["z_top"] - f0["z_c"], f0["apex"] - f0["z_c"]
            return {"epsilon": float(laws.epsilon({"v": pre["v"][:1]}, {"v": post["v"][:1]}, {"event": "bounce", "g": g})),
                    "epsilon_height": float(np.sqrt(h1 / h0)) if h0 > 0 and h1 >= 0 else float("nan"), "windows": win}
        st = laws.gravity.stats(pre, post, {"g": g, **({"m": np.asarray(masses, float)} if masses is not None else {})})
        v0 = pre["v"][0]
        if abs(v0[0]) <= 0.1 * np.linalg.norm(v0):                     # no horizontal motion to compare
            st["axis_ratio"] = float("nan")
        return {**{k_: float(v) for k_, v in st.items()}, "windows": win}
    # F8 Galileo's ramp (object 0)
    z, fin = P[:, 0, 1], np.where(ok[:, 0])[0]
    before, after = fin[fin < ke], fin[fin >= ke]
    if len(before) < 2 or len(after) < 2:
        return {"reason": "lost"}
    near = fin[np.abs(fin - ke) <= max(1, wn // 4)]
    z_pad = float(np.median(z[near]))                                     # the pad / run-out level
    z_rel, z_apex = float(z[before].max()), float(z[after].max())
    h0, h1 = z_rel - z_pad, z_apex - z_pad
    if h0 <= 0:
        return {"reason": "no_descent"}
    out = {"epsilon": float(np.sqrt(h1 / h0)) if h1 >= 0 else float("nan"), "release_height_rel": 1.0,
           "apex_height_rel": float(h1 / h0)}
    flat = np.abs(z - z_pad) <= 0.02 * h0
    w0 = [i for i in before if flat[i]][-wn:]                             # run-in on the flat before the pad
    rest = [i for i in after if flat[i]]
    w1 = rest[-wn:] if len(rest) >= 2 * TR_MIN_FRAMES else []             # the last flat frames before the climb
    win = {0: [w for w in (w0, w1) if len(w) >= TR_MIN_FRAMES]}

    def speed(w):
        d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P[w, 0], axis=0), axis=1))]
        return float(np.polyfit(t[w], d, 1)[0])
    if len(w0) >= TR_MIN_FRAMES and len(w1) >= TR_MIN_FRAMES:
        out["epsilon_speed"] = float(laws.epsilon({"v": [[speed(w0), 0.0]]}, {"v": [[speed(w1), 0.0]]}, {"event": "pad"}))
    return {**out, "windows": win}


def rolling_stats(t: np.ndarray, S: np.ndarray, r, k, ang: np.ndarray | None = None) -> dict:
    """laws.rolling statistics of tracked cylinders after release: t [T] (any time unit), S [T, n_obj] distance
    travelled along the incline (any length unit; NaN = not tracked), r [n_obj] radii in the SAME length unit (e.g.
    pixels), k [n_obj] inertia factors, ang [T, n_obj] unwrapped rotation angle (rad) of each body's marker, or None.
    order: from the accelerations of quadratic fits of S. slip = |w| r / |v| with w and v the window-mean rates
    (|dtheta| / |dS| over the same window, endpoints from quadratic fits): exact for any constant slip, and a
    projection that makes the angle advance unevenly within a turn averages out. Scale-free (rad per time unit on
    both sides). Also returns accel and omega per object."""
    from physlaw.eval import laws
    n = S.shape[1]
    a, v, w = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n)
    for i in range(n):
        ok = np.isfinite(S[:, i]) & (np.isfinite(ang[:, i]) if ang is not None else True)
        if ok.sum() < 3:
            continue
        ti, t0, t1 = t[ok], t[ok][0], t[ok][-1]
        c = np.polyfit(ti, S[ok, i], 2)
        a[i], v[i] = 2 * c[0], (np.polyval(c, t1) - np.polyval(c, t0)) / (t1 - t0)
        if ang is not None:
            ca = np.polyfit(ti, ang[ok, i], 2)
            w[i] = (np.polyval(ca, t1) - np.polyval(ca, t0)) / (t1 - t0)
    post = {"v": v[:, None], "a": a[:, None], **({"w": w[:, None]} if ang is not None else {})}
    st = laws.rolling.stats({"v": np.zeros((n, 1))}, post, {"k": np.asarray(k, float), "r": np.asarray(r, float), "dt": 0.0})
    return {**{k_: float(x) for k_, x in st.items()}, "accel": a.tolist(), "omega": np.abs(w).tolist()}


# F10 readout (track B): two cylinders on one incline, oblique camera. Per object and frame: the blob = the foreground
# component under its tracked centre; the marker wedge (generate.marker: a near-black or near-white 90-degree sector
# about the rolling axis, on the end faces and the curved surface) = blob pixels whose luminance departs from the
# blob median by half the way to black / white (whichever set is larger); its angle = atan2 of the wedge centroid's
# offset from the blob centroid, unwrapped. S = displacement along the track's principal image direction; r_px = the
# blob's half-chord through its centroid along that direction. The motion lies in the end-face plane, so the
# displacement and this chord share one foreshortening and |dtheta| r / |dS| needs no camera model; approximation: the
# curved side widens the silhouette along the projected rolling axis, which lengthens the chord where that axis is not
# perpendicular to the motion on screen (validated per family on the real clips, stage 8 --real).
TR_WEDGE_FRAC = 0.03             # wedge needs >= this share of the blob's pixels
TR_MAX_SPIN = 0.8 * np.pi        # |dtheta| per frame above this: the rotation can alias ("spin_aliased")


def _wedge_points(img: np.ndarray, blob: np.ndarray):
    """(blob centroid, marker-wedge centroid) in pixels (row, col), or None when no wedge is found. The blob's
    luminances are split in two by Otsu's threshold; the wedge is the class whose mean is farther from mid-grey (the
    marker is near-black on light bodies, near-white on dark ones), whatever share of the blob it covers."""
    rr, cc = np.nonzero(blob)
    L = img[rr, cc].astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    h = np.histogram(L, 64, (0.0, 256.0))[0].astype(float)
    mid = (np.arange(64) + 0.5) * 4.0
    w0, s0 = np.cumsum(h)[:-1], np.cumsum(h * mid)[:-1]
    w1, s1 = h.sum() - w0, (h * mid).sum() - s0
    with np.errstate(divide="ignore", invalid="ignore"):
        between = np.where((w0 > 0) & (w1 > 0), w0 * w1 * (s0 / w0 - s1 / w1) ** 2, -1.0)
    if between.max() <= 0:
        return None
    lo = L < 4.0 * (int(np.argmax(between)) + 1)
    sel = lo if abs(L[lo].mean() - 127.5) >= abs(L[~lo].mean() - 127.5) else ~lo
    if sel.sum() < max(4, TR_WEDGE_FRAC * len(L)) or (~sel).sum() < max(4, TR_WEDGE_FRAC * len(L)):
        return None
    return np.array([rr.mean(), cc.mean()]), np.array([rr[sel].mean(), cc[sel].mean()])


def _wedge_angle(img: np.ndarray, blob: np.ndarray) -> float:
    pts = _wedge_points(img, blob)
    return float("nan") if pts is None else float(np.arctan2(*(pts[1] - pts[0])))


def _half_chord(blob: np.ndarray, u: np.ndarray) -> float:
    """Mean of the two half-chords of ``blob`` through its centroid along the unit image direction u (row, col)."""
    rr, cc = np.nonzero(blob)
    c, H, W = np.array([rr.mean(), cc.mean()]), *blob.shape
    out = []
    for sgn in (1.0, -1.0):
        s = np.arange(0.0, float(max(H, W)), 0.25)
        q = np.rint(c[None] + sgn * s[:, None] * u[None]).astype(int)
        inside = (q[:, 0] >= 0) & (q[:, 0] < H) & (q[:, 1] >= 0) & (q[:, 1] < W)
        hit = np.zeros(len(s), bool)
        hit[inside] = blob[q[inside, 0], q[inside, 1]]
        out.append(s[np.argmin(hit)] if not hit.all() else s[-1])
    return float(np.mean(out))


def readout_rolling(gen_frames: np.ndarray, clip, fps: float, checks: bool = True, gate: str = "strict",
                    frames_idx: list | None = None) -> dict:
    """F10 readout of a video whose frame i shows master frame frames_idx[i] (default: stride from frame 0): centre
    tracks, marker angles, :func:`rolling_stats` over the post-release window (to clean_until_s), and the strict gate
    on the px residual of the quadratic centre fits. Row: order, slip_s, slip_h, accel, omega, r_px, fit_resid."""
    import cv2
    tk = _track_world(gen_frames, clip, fps, frames_idx, world=False)
    pl, objs, pix, bg, ke, end = (tk[k] for k in ("pl", "objs", "pix", "bg", "ke", "end"))
    H, W = gen_frames.shape[1:3]
    win = np.arange(ke, end)
    res = {"frames_tracked": np.isfinite(pix).all(-1).sum(0).tolist(), "event_index": ke}
    if len(win) < TR_MIN_FRAMES:
        return {**res, "measurable": False, "reason": "short_window"}
    ang = np.full((len(win), len(objs)), np.nan)
    blobs = {}
    for j, t in enumerate(win):
        _, lab = cv2.connectedComponents(_foreground(gen_frames[t], bg), connectivity=8)
        for i in range(len(objs)):
            if not np.isfinite(pix[t, i]).all():
                continue
            r0, c0 = np.clip(np.rint(pix[t, i]).astype(int), 0, [H - 1, W - 1])
            if lab[r0, c0]:
                blobs[j, i] = lab == lab[r0, c0]
                ang[j, i] = _wedge_angle(gen_frames[t], blobs[j, i])
    tt = win / float(fps)
    S, rpx, resid = np.full((len(win), len(objs)), np.nan), [], 0.0
    for i in range(len(objs)):
        ok = np.isfinite(pix[win, i]).all(-1) & np.isfinite(ang[:, i])
        if ok.sum() < TR_MIN_FRAMES:
            return {**res, "measurable": False, "reason": "short_window"}
        a = np.unwrap(ang[ok, i])
        if checks and np.abs(np.diff(a)).max() > TR_MAX_SPIN:
            return {**res, "measurable": False, "reason": "spin_aliased"}
        ang[ok, i], ang[~ok, i] = a, np.nan
        P = pix[win[ok], i]
        d = P - P[0]
        u = np.linalg.svd(d - d.mean(0), full_matrices=False)[2][0]
        u = u if d[-1] @ u >= 0 else -u
        S[ok, i] = d @ u
        rpx.append(float(np.median([_half_chord(blobs[j, i], u) for j in np.where(ok)[0]])))
        resid = max([resid] + [_fit(tt[ok], P[:, c], 2)[1] for c in (0, 1)])
    k = [pl["objects"][o]["inertia"] / (pl["objects"][o]["mass"] * pl["objects"][o].get("size", pl["objects"][o].get("radius")) ** 2)
         for o in objs]
    res.update(rolling_stats(tt, S, rpx, k, ang), r_px=rpx, fit_resid=float(resid))
    if gate == "strict" and checks and resid > TR_MAX_RESID_PX:
        return {**res, "measurable": False, "reason": "fit_residual"}
    return {**res, "measurable": True, "reason": None}


def readout_traj(gen_frames: np.ndarray, clip, fps: float, checks: bool = True, gate: str = "strict",
                 frames_idx: list | None = None) -> dict:
    """F7 / F8 readout of a video whose frame i shows master frame frames_idx[i] (default: stride from frame 0):
    tracking, back-projection, :func:`traj_stats`, and the strict gate on the px residual of every fitted window."""
    tk = _track_world(gen_frames, clip, fps, frames_idx)
    pl, labels, objs, pix, world, ke, t, te, wn, ok = (tk[k] for k in ("pl", "labels", "objs", "pix", "world", "ke",
                                                                         "t", "te", "wn", "ok"))
    sub = pl.get("subfamily")
    res = {"subfamily": sub, "frames_tracked": ok.sum(0).tolist(), "event_index": ke}
    horiz = int(np.argmax(np.ptp(np.nan_to_num(world[..., :2]), axis=0).max(0)))   # world x or y: the moving one
    t_rel = pl.get("constants", {}).get("release_time_s")
    st = traj_stats("ramp" if pl.get("family") == "f8" else sub, t, world[..., [horiz, 2]], ok, ke, te, wn,
                    [pl["objects"][o]["mass"] for o in objs], t_rel if t_rel is not None and t_rel < te else None,
                    tk["end"])
    if "reason" in st:
        return {**res, "measurable": False, "reason": st["reason"]}
    win = st.pop("windows")
    res.update(st)
    deg = 1 if pl.get("family") == "f8" else 2
    resid = max([0.0] + [_fit(t[w], pix[w, k, a], deg)[1] for k, ws in win.items() for w in ws for a in (0, 1)])
    res["fit_resid"] = float(resid)
    if gate == "strict" and checks and resid > TR_MAX_RESID_PX:
        return {**res, "measurable": False, "reason": "fit_residual"}
    return {**res, "measurable": True, "reason": None}


def _gated(res: dict, resid: float, gate: str, checks: bool) -> dict:
    """The strict gate on the px residual of the fitted windows (TR_MAX_RESID_PX)."""
    res["fit_resid"] = float(resid)
    if gate == "strict" and checks and resid > TR_MAX_RESID_PX:
        return {**res, "measurable": False, "reason": "fit_residual"}
    return {**res, "measurable": True, "reason": None}


def _resid(t, pix, windows: dict, deg: int) -> float:
    return max([0.0] + [_fit(t[w], pix[w, k, a], deg)[1] for k, ws in windows.items() for w in ws if len(w) > deg + 1
                        for a in (0, 1)])


# --------------------------------------------------------------------------- #
# F6 / F9 / F11 / F12 readouts (track A / B), scale-free like F7 / F8 / F10
# --------------------------------------------------------------------------- #
#   F6 glide (momentum, Newton I)   straight-line fits of the puck before / after the branch -> laws.momentum
#                                   (force_free) inertia_speed, inertia_turn
#   F6 slide / pair (friction)      quadratic fits before the branch and while moving after it (io.friction_windows:
#                                   up to the labelled legal stop - margin), a line over the stop window (legal stop
#                                   + margin .. clean end) -> laws.friction decel_const, stop_hold (pair: mass_ratio,
#                                   label masses)
#   F9 pair (oscillation)           T = 2 x the interval between zero crossings (io.zero_crossings) of each bob's
#                                   horizontal offset from its pivot: the last interval before the branch, the first
#                                   after -> period_ratio, mass_period_ratio
#   F9 swing (energy)               apex heights above the lowest tracked point (local quadratic peaks): the release
#                                   apex and the first apex after the damper -> laws.epsilon (swing; no g)
#   F11 (statics)                   block / prop pixels by nearest signature colour, in master pixels (isotropic
#                                   after a squash); block angle = principal axis, half-length = sqrt(3 x major
#                                   variance); rotation over laws.incline.tip_window -> tipped; alpha from
#                                   phi(s) = alpha s^2 / 2 + q s^4 / 24 on that window up to TR_TIP_FIT_RAD; g_clip =
#                                   the block's vertical parabola after its labelled separation (inside the clean
#                                   window), else the prop's top-edge parabola / constants.prop_acc_over_g ->
#                                   laws.statics tip_consistency, alpha_ratio
#   F12 (angular_momentum)          puck / bar centres back-projected onto the table, the bar's marker-wedge heading
#                                   (its offset back-projected too, so the sense of rotation is the world's); lines
#                                   before / after the contact extrapolated to it -> laws.angular_momentum L_ratio
#                                   (label masses and inertias)
TR_TIP_FIT_RAD = 0.5         # F11: alpha is fitted on the tip window's frames until the block has turned this far
TR_COLOUR_MAX = 60.0
TR_MERGE_AREA = 1.5          # F12: a bar blob larger than this x its median area is merged with the puck


def glide_stats(t: np.ndarray, X: np.ndarray, ok: np.ndarray, ke: int, end: int, wn: int) -> dict:
    """Newton I on a tracked glide: X [T, 1, d] (any length unit), t [T] (any time unit)."""
    from physlaw.eval import laws
    pre = [i for i in range(max(0, ke - wn), ke) if ok[i, 0]]
    post = [i for i in range(ke, min(end, ke + wn)) if ok[i, 0]]
    if len(pre) < TR_MIN_FRAMES or len(post) < TR_MIN_FRAMES:
        return {"reason": "short_window"}
    v = [np.polyfit(t[w], X[w, 0], 1)[0] for w in (pre, post)]
    st = laws.momentum.stats({"v": v[0][None]}, {"v": v[1][None]}, {"force_free": True})
    return {"inertia_speed": float(st["inertia_speed"]), "inertia_turn": float(st["inertia_turn"]),
            "windows": {0: [pre, post]}}


def friction_stats(t: np.ndarray, X: np.ndarray, ok: np.ndarray, te: float, windows: dict, masses=None) -> dict:
    """laws.friction on tracked slides: X [T, n, d] (any units), t [T] clip time. ``windows`` (io.friction_windows):
    quadratic fits over the in-frame frames before the branch (a_pre, v at te) and over [te, move_end] (a_post), a
    line over [stop_start, end] (v_stop, the mean velocity over the stop window); every object tracked in a frame."""
    from physlaw.eval import laws
    n = X.shape[1]
    good = ok.all(1)
    pre = np.nonzero(good & (t < te))[0]
    mv = np.nonzero(good & (t >= te) & (t <= windows["move_end"] + 1e-9))[0]
    stp = np.nonzero(good & (t >= windows["stop_start"] - 1e-9) & (t <= windows["end"] + 1e-9))[0]
    if len(pre) < 2 * TR_MIN_FRAMES or len(mv) < 2 * TR_MIN_FRAMES:
        return {"reason": "short_window"}

    def quad(w, tc):
        c = np.polyfit(t[w] - tc, X[w].reshape(len(w), -1), 2).reshape(3, n, -1)
        return {"v": c[1], "a": 2.0 * c[0]}
    post = quad(mv, t[mv].mean())
    if len(stp) >= TR_MIN_FRAMES:
        post["v_stop"] = np.polyfit(t[stp], X[stp].reshape(len(stp), -1), 1)[0].reshape(n, -1)
    ctx = {"m": np.asarray(masses, float)} if masses is not None else {}
    st = laws.friction.stats(quad(pre, te), post, ctx)
    # accel: each object's acceleration along its motion while moving after the branch (track units; L4's
    # log_decel_A row)
    return {**{k: float(v) for k, v in st.items()}, "accel": laws.freefall.along_motion(post).tolist(),
            "windows": {k: [list(pre), list(mv)] + ([list(stp)] if len(stp) else []) for k in range(n)}}


def pendulum_stats(t: np.ndarray, X: np.ndarray, ok: np.ndarray, pivot_x, te: float, end: int, masses=None) -> dict:
    """laws.oscillation on tracked bobs: X [T, n, 2] (horizontal, up; any length unit), pivot_x [n] in the same
    unit; periods from the zero crossings of x - pivot_x before / after te."""
    from physlaw.data.io import zero_crossings
    from physlaw.eval import laws
    n = X.shape[1]
    T0, T1 = np.full(n, np.nan), np.full(n, np.nan)
    for k in range(n):
        f = np.nonzero(ok[:end, k])[0]
        if len(f) < 3:
            continue
        x = X[f, k, 0] - pivot_x[k]
        c = zero_crossings(t[f], x, np.gradient(x, t[f]))
        before = [(a, b) for a, b in zip(c[:-1], c[1:]) if b <= te]
        after = [(a, b) for a, b in zip(c[:-1], c[1:]) if a >= te]
        T0[k] = 2.0 * (before[-1][1] - before[-1][0]) if before else np.nan
        T1[k] = 2.0 * (after[0][1] - after[0][0]) if after else np.nan
    if not (np.isfinite(T0) & np.isfinite(T1)).any():
        return {"reason": "no_period"}
    st = laws.oscillation.stats({"T": T0}, {"T": T1}, {"m": np.asarray(masses, float)} if masses is not None else {})
    return {**{k: float(v) for k, v in st.items()}, "period_pre": T0.tolist(), "period_post": T1.tolist()}


def _peak(t: np.ndarray, z: np.ndarray, idx: np.ndarray, sign: float) -> float:
    """The extremum (sign +1 max, -1 min) of z over frames idx, refined by a quadratic through the nearest frames
    when its vertex falls inside them."""
    j = int(idx[np.argmax(sign * z[idx])])
    nb = [i for i in range(j - 2, j + 3) if i in set(idx.tolist())]
    if len(nb) >= 3:
        c = np.polyfit(t[nb], z[nb], 2)
        if sign * c[0] < 0 and t[nb[0]] <= -c[1] / (2 * c[0]) <= t[nb[-1]]:
            return float(c[2] - c[1] ** 2 / (4 * c[0]))
    return float(z[j])


def swing_stats(t: np.ndarray, Z: np.ndarray, ok: np.ndarray, ke: int, end: int) -> dict:
    """laws.epsilon (swing) on a tracked bob: Z [T] height (up, any unit): release apex before the event, first apex
    after it (frames < end), lowest point over both."""
    from physlaw.eval import laws
    f = np.nonzero(ok[:end])[0]
    pre, post = f[f < ke], f[f >= ke]
    if len(pre) < 3 or len(post) < 3:
        return {"reason": "short_window"}
    z0, z1, zb = _peak(t, Z, pre, 1.0), _peak(t, Z, post, 1.0), _peak(t, Z, f, -1.0)
    at = lambda z: {"p": [[0.0, z]], "v": [[0.0, 0.0]]}                  # apexes: at rest
    eps = laws.epsilon(at(z0), at(z1), {"event": "swing", "p_bottom": [0.0, zb], "g": [0.0, -1.0]})
    return {"epsilon": float(eps), "apex_height_rel": float((z1 - zb) / (z0 - zb)) if z0 > zb else float("nan")}


def support_stats(t: np.ndarray, phi: np.ndarray, z_block: np.ndarray, z_prop: np.ndarray, te: float, t_end: float,
                  a_len: float, x: float, beta: float, t_sep: float | None = None,
                  prop_ratio: float | None = None) -> dict:
    """laws.statics on a tracked block: t [T] (any time unit), phi [T] its axis angle (rad; NaN = lost), z_block /
    z_prop [T] the block's centroid height and the prop's top-edge height (up; any length unit, NaN = lost), a_len its
    half-length in that unit; te the release, t_end the clean window's end, t_sep the labelled separation."""
    from physlaw.eval import laws
    from physlaw.eval.laws.incline import tip_window
    nan = float("nan")
    okp = np.isfinite(phi)
    if (okp & (t < te)).sum() < 2:
        return {"reason": "short_window"}

    def accel(sel, z):
        m = sel & np.isfinite(z)
        return abs(2.0 * float(np.polyfit(t[m], z[m], 2)[0])) if m.sum() >= 4 else nan
    g = accel((t > t_sep) & (t <= t_end + 1e-9), z_block) if t_sep is not None and t_sep < t_end else nan
    if not np.isfinite(g) and prop_ratio:
        g = accel((t > te) & (t <= te + TR_WINDOW_S), z_prop) / float(prop_ratio)
    ph = np.full(len(phi), np.nan)
    ph[okp] = np.unwrap(2.0 * phi[okp]) / 2.0                            # an axis: period pi
    ph0 = float(np.median(ph[okp & (t < te)]))
    w_tip = tip_window(g, a_len, None if t_sep is None else t_sep - te, t_end - te)   # clip time units
    win = np.nonzero(okp & (t > te) & (t <= min(te + w_tip, t_end) + 1e-9))[0]
    if not len(win):
        return {"reason": "short_window"}
    rot = abs(float(ph[win[-1]]) - ph0)
    fit = np.zeros(len(t), bool)
    fit[win] = np.abs(ph[win] - ph0) <= TR_TIP_FIT_RAD
    alpha = nan
    if fit.sum() >= 2:                     # phi(s) = alpha s^2 / 2 + q s^4 / 24 (the cubic term vanishes from rest)
        s_ = t[fit] - te
        A = np.stack([s_ ** 2 / 2, s_ ** 4 / 24], 1)[:, : 2 if fit.sum() >= 3 else 1]
        alpha = abs(float(np.linalg.lstsq(A, ph[fit] - ph0, rcond=None)[0][0]))
    st = laws.statics.stats({"v": np.zeros((1, 2))}, {"v": np.zeros((1, 2)), "alpha": [[alpha]], "phi": [rot]},
                            {"overhang_x": x, "aspect_ba": beta, "g_clip": g, "half_length": a_len})
    return {**{k: float(v) for k, v in st.items()}, "rotation_window": rot, "alpha": alpha, "g_clip": g,
            "half_length_px": float(a_len)}


def spin_stats(t: np.ndarray, X: np.ndarray, ok: np.ndarray, heading: np.ndarray, ke: int, end: int, wn: int,
               masses, inertias) -> dict:
    """laws.angular_momentum on tracked bodies: X [T, n, 2] centres (any length unit), heading [T, n] marker heading
    (rad, NaN = not seen; a body never seen keeps spin 0), straight lines over wn frames before the contact and from
    the frame after it (ke + 1) to end, extrapolated to the contact time (midway between frames ke - 1 and ke).
    ``inertias`` [n] in mass x (length unit)^2."""
    from physlaw.eval import laws
    n = X.shape[1]
    pre = [i for i in range(max(0, ke - 1 - wn), ke - 1) if ok[i].all()]
    post = [i for i in range(ke + 1, min(end, ke + 1 + wn)) if ok[i].all()]
    if len(pre) < TR_MIN_FRAMES or len(post) < TR_MIN_FRAMES:
        return {"reason": "short_window"}
    tc = 0.5 * (t[ke - 1] + t[ke])

    def state(w):
        c = np.polyfit(t[w] - tc, X[w].reshape(len(w), -1), 1).reshape(2, n, -1)
        om = np.zeros((n, 1))
        for k in range(n):
            h = heading[w, k]
            m = np.isfinite(h)
            if m.sum() >= 3:
                om[k, 0] = np.polyfit(t[w][m], np.unwrap(h[m]), 1)[0]
        return {"p": c[1], "v": c[0], "w": om}
    p0, p1 = state(pre), state(post)
    if not np.isfinite(heading[post, -1]).sum() >= 3:
        return {"reason": "no_marker"}
    st = laws.angular_momentum.stats(p0, p1, {"m": np.asarray(masses, float), "I": np.asarray(inertias, float)})
    return {"L_ratio": float(st["L_ratio"]), "omega_post": p1["w"][:, 0].tolist(), "windows": {k: [pre, post] for k in range(n)}}


def readout_friction(gen_frames: np.ndarray, clip, fps: float, checks: bool = True, gate: str = "strict",
                     frames_idx: list | None = None) -> dict:
    """F6 readout: glide -> :func:`glide_stats`, slide / pair -> :func:`friction_stats`, on the
    back-projected tracks; strict gate on the px residual of the fitted windows."""
    tk = _track_world(gen_frames, clip, fps, frames_idx)
    pl, objs, pix, world, ok, t = (tk[k] for k in ("pl", "objs", "pix", "world", "ok", "t"))
    sub = pl.get("subfamily")
    res = {"subfamily": sub, "frames_tracked": ok.sum(0).tolist(), "event_index": tk["ke"]}
    if sub == "glide":
        st, deg = glide_stats(t, world, ok, tk["ke"], tk["end"], tk["wn"]), 1
    else:
        from physlaw.data.io import friction_windows
        st, deg = friction_stats(t, world, ok, tk["te"], friction_windows(tk["labels"]),
                                 [pl["objects"][o]["mass"] for o in objs]), 2
    if "reason" in st:
        return {**res, "measurable": False, "reason": st["reason"]}
    win = st.pop("windows")
    return _gated({**res, **st}, _resid(t, pix, win, deg), gate, checks)


def readout_pendulum(gen_frames: np.ndarray, clip, fps: float, checks: bool = True, gate: str = "strict",
                     frames_idx: list | None = None) -> dict:
    """F9 readout (side view, back-projected bob tracks; the rods are aux and not tracked, but a rod joined to its
    bob keeps the blob centroid on the pendulum line, so crossings and apex-height ratios are unchanged): pair ->
    :func:`pendulum_stats`, swing -> :func:`swing_stats`."""
    tk = _track_world(gen_frames, clip, fps, frames_idx)
    pl, objs, world, ok, t = (tk[k] for k in ("pl", "objs", "world", "ok", "t"))
    sub = pl.get("subfamily")
    res = {"subfamily": sub, "frames_tracked": ok.sum(0).tolist(), "event_index": tk["ke"]}
    if sub == "swing":
        st = swing_stats(t, world[:, 0, 2], ok[:, 0], tk["ke"], tk["end"])
    else:
        piv = pl["constants"]["pivots"]
        st = pendulum_stats(t, world[..., [0, 2]], ok, [piv[o][0] for o in objs], tk["te"], tk["end"],
                            [pl["objects"][o]["mass"] for o in objs])
    if "reason" in st:
        return {**res, "measurable": False, "reason": st["reason"]}
    return {**res, **st, "measurable": True, "reason": None}


def readout_support(gen_frames: np.ndarray, clip, fps: float, checks: bool = True, gate: str = "strict",
                    frames_idx: list | None = None) -> dict:
    """F11 readout (side view): per frame the block's and the prop's pixels (foreground pixels nearest to each
    signature colour, largest component), in master-pixel units; :func:`support_stats`."""
    import cv2
    from physlaw.eval.laws.incline import TIP_ANGLE
    pl = clip.labels["physlaw"]
    objs = sorted(pl["objects"], key=lambda o: bool(pl["objects"][o].get("aux")))          # block first, then prop
    tk = _track_world(gen_frames, clip, fps, frames_idx, objs=objs, world=False)
    H, W = gen_frames.shape[1:3]
    W0, H0 = pl["camera"]["resolution"]
    sid = [pl["objects"][o]["segmentation_id"] for o in objs]
    sig = []
    for s_ in sid:
        vis = [i for i in range(len(tk["rmask"])) if (tk["rmask"][i] == s_).any()]
        sig.append(tk["ref"][vis[0]][tk["rmask"][vis[0]] == s_].reshape(-1, 3).astype(np.float32).mean(0)
                   if vis else np.full(3, np.inf, np.float32))
    T = len(gen_frames)
    phi, zb, zp, half = np.full(T, np.nan), np.full(T, np.nan), np.full(T, np.nan), []
    for i in range(T):
        img = gen_frames[i].astype(np.float32)
        fg = _foreground(gen_frames[i], tk["bg"]).astype(bool)
        d = np.stack([np.linalg.norm(img - c, axis=2) for c in sig])
        near = d.argmin(0)
        for k in range(len(objs)):
            m = (fg & (near == k) & (d[k] < TR_COLOUR_MAX)).astype(np.uint8)
            n_, lab, st_, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            if n_ < 2:
                continue
            j = 1 + int(np.argmax(st_[1:, cv2.CC_STAT_AREA]))
            if st_[j, cv2.CC_STAT_AREA] < 20:
                continue
            rr, cc = np.nonzero(lab == j)
            y, xx = -rr * (H0 / H), cc * (W0 / W)                             # master pixels, up
            if k == 0:
                ev, V = np.linalg.eigh(np.cov(np.stack([y, xx])))
                phi[i] = float(np.arctan2(V[0, 1], V[1, 1]))
                zb[i] = float(y.mean())
                half.append(float(np.sqrt(3.0 * ev[1])))
            else:
                zp[i] = float(y.max())
    res = {"frames_tracked": [int(np.isfinite(phi).sum()), int(np.isfinite(zp).sum())], "event_index": tk["ke"]}
    if not half:
        return {**res, "measurable": False, "reason": "lost"}
    t_end = float(tk["t"][tk["end"] - 1]) if tk["end"] > 0 else float("nan")
    c = pl["constants"]
    st = support_stats(tk["t"], phi, zb, zp, tk["te"], t_end, float(np.median(half)), float(c["overhang_x"]),
                       float(c["aspect_ba"]), (pl.get("edit") or {}).get("t_separate_s"), c.get("prop_acc_over_g"))
    if "reason" in st:
        return {**res, "measurable": False, "reason": st["reason"]}
    return {**res, **st, "tipped": bool(st["rotation_window"] > TIP_ANGLE), "measurable": True, "reason": None}


def readout_spin(gen_frames: np.ndarray, clip, fps: float, checks: bool = True, gate: str = "strict",
                 frames_idx: list | None = None) -> dict:
    """F12 readout (top-down): puck and bar centres back-projected onto the table; each body's marker-wedge heading
    from the component under its tracked centre (skipped where the bar's blob is merged with the puck), the wedge
    offset back-projected so the heading is a world angle; :func:`spin_stats`; strict gate on the px residual."""
    import cv2
    tk = _track_world(gen_frames, clip, fps, frames_idx)
    pl, labels, objs, pix, world, ok, t = (tk[k] for k in ("pl", "labels", "objs", "pix", "world", "ok", "t"))
    H, W = gen_frames.shape[1:3]
    T, n = pix.shape[:2]
    head, area = np.full((T, n), np.nan), np.full((T, n), np.nan)
    blobs = {}
    for i in range(T):
        _, lab = cv2.connectedComponents(_foreground(gen_frames[i], tk["bg"]), connectivity=8)
        for k in range(n):
            if ok[i, k]:
                r0, c0 = np.clip(np.rint(pix[i, k]).astype(int), 0, [H - 1, W - 1])
                if lab[r0, c0]:
                    blobs[i, k] = lab == lab[r0, c0]
                    area[i, k] = blobs[i, k].sum()
    for (i, k), b in blobs.items():
        if area[i, k] > TR_MERGE_AREA * np.nanmedian(area[:, k]):
            continue                                                      # merged with the other body
        pts = _wedge_points(gen_frames[i], b)
        if pts is not None:
            depth = _object_depth(labels, objs[k])
            w0, w1 = backproject(np.stack(pts), labels, H, W, depth)[:, :2]
            head[i, k] = float(np.arctan2(*(w1 - w0)[::-1]))
    res = {"frames_tracked": ok.sum(0).tolist(), "event_index": tk["ke"]}
    obj = pl["objects"]
    st = spin_stats(t, world[..., :2], ok, head, tk["ke"], tk["end"], tk["wn"], [obj[o]["mass"] for o in objs],
                    [obj[o]["inertia"] for o in objs])
    if "reason" in st:
        return {**res, "measurable": False, "reason": st["reason"]}
    win = st.pop("windows")
    return _gated({**res, **st}, _resid(t, pix, win, 1), gate, checks)


READOUT_SELF.update(f6=readout_friction, f7=readout_traj, f8=readout_traj, f9=readout_pendulum, f10=readout_rolling,
                    f11=readout_support, f12=readout_spin)
