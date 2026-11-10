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
Scores: the L1 residuals (r_p, r_E, outcome E) with TRUE masses and pre velocities
(physlaw.eval.l1_forecast_laws.residuals), floor = the same readout on the real clip.
Feasibility gate (configs/eval/outputs.yaml): measurable fraction of seed-0 generations of the first
``n_clips`` groups must reach ``min_pass_rate`` before the full 4-seed evaluation is run.
"""
from __future__ import annotations

import numpy as np

from physlaw.eval.l1_forecast_laws import residuals
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
    sy, sx = H / H0, W / W0
    if fit != "squash":
        raise NotImplementedError(fit)
    maps = {}
    pf = labels["per_frame"]
    for o in ("A", "B"):
        P = np.array([[pf[f][o]["pixel"][1] * sy, pf[f][o]["pixel"][0] * sx, 1.0] for f in frames_idx])
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
    v, rms = np.zeros((2, 2)), 0.0
    for i in range(2):
        A = np.stack([tt, np.ones_like(tt)], 1)
        coef, *_ = np.linalg.lstsq(A, world[win, i], rcond=None)
        v[i] = coef[0]
        rms = max(rms, float(np.sqrt(((A @ coef - world[win, i]) ** 2).mean())))
    res.update(v_post=v.tolist(), fit_rms=rms)
    if rms > max_rms:
        return {**res, "measurable": False, "reason": "bad_fit"}
    return {**res, "measurable": True}


def score(v_post: np.ndarray, labels: dict) -> dict:
    pl = labels["physlaw"]
    m = np.array([pl["objects"][o]["mass"] for o in "AB"])
    vpre = np.array([pl["pre_velocity"][o][:2] for o in "AB"], float)
    vpost = np.array([pl["post_velocity"][o][:2] for o in "AB"], float)
    return residuals(np.asarray(v_post, float), m, vpre, vpost)


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


def gate(cache_dir, data_dir, adapter_geom, n_clips: int, min_pass_rate: float, log=print) -> dict:
    """Unsteered feasibility gate on the seed-0 generations of the first n_clips groups."""
    from physlaw.data.io import load_clip
    gens = sorted(cache_dir.glob("*/gen_seed0.mp4"))[:n_clips]
    rows = []
    for g in gens:
        c = load_clip(data_dir / g.parent.name / "conserve_e1")
        fr, k = adapter_geom(c.labels)
        r = measure_clip(_read_mp4(g), c, fr, k, *adapter_geom.fps_hw)
        rows.append({"group": g.parent.name, **{kk: v for kk, v in r.items() if kk != "post_frames"}})
    rate = float(np.mean([r["measurable"] for r in rows])) if rows else 0.0
    from collections import Counter
    out = {"n": len(rows), "pass_rate": rate, "min_pass_rate": min_pass_rate, "passed": rate >= min_pass_rate,
           "reasons": dict(Counter(r.get("reason", "ok") for r in rows)), "clips": rows}
    log(f"[gate] {len(rows)} clips, measurable {rate:.0%} (need {min_pass_rate:.0%}): {out['reasons']}")
    return out


def evaluate(cache_dir, data_dir, adapter_geom, n_boot: int = 2000, log=print) -> dict:
    """All generated seeds x legal twins: residuals, drop rate, floor from the real clips, outcome S."""
    from collections import Counter
    from physlaw.data.io import load_clip
    from physlaw.eval.l1_forecast_laws import _score
    groups = sorted({p.parent.name for p in cache_dir.glob("*/gen_seed*.mp4")})
    gen_rows, real_rows = [], []
    for g in groups:
        clips = {t: load_clip(data_dir / g / t) for t in LEGAL}
        fr, k = adapter_geom(clips[LEGAL[0]].labels)
        for t, c in clips.items():
            r = measure_clip(None, c, fr, k, *adapter_geom.fps_hw)
            if r["measurable"]:
                r.update(score(r["v_post"], c.labels))
            real_rows.append({"group": g, "twin": t, **{kk: v for kk, v in r.items() if kk != "post_frames"}})
        for mp4 in sorted((cache_dir / g).glob("gen_seed*.mp4")):
            frames = _read_mp4(mp4)
            r = measure_clip(frames, clips[LEGAL[0]], fr, k, *adapter_geom.fps_hw)
            for t, c in clips.items():            # one generated outcome, scored against each legal truth
                row = {"group": g, "twin": t, "seed": int(mp4.stem.split("seed")[1]),
                       **{kk: v for kk, v in r.items() if kk != "post_frames"}}
                if r["measurable"]:
                    row.update(score(r["v_post"], c.labels))
                gen_rows.append(row)
    out = {"n_groups": len(groups), "twins": {}}
    rng = np.random.default_rng(0)
    for t in LEGAL:
        G = [r for r in gen_rows if r["twin"] == t]
        R = [r for r in real_rows if r["twin"] == t]
        Gm, Rm = [r for r in G if r["measurable"]], [r for r in R if r["measurable"]]
        d = {"n_generated": len(G), "measurable_generated": len(Gm), "drop_rate_generated": 1 - len(Gm) / max(len(G), 1),
             "drop_reasons_generated": dict(Counter(r.get("reason", "ok") for r in G if not r["measurable"])),
             "n_real": len(R), "measurable_real": len(Rm), "drop_rate_real": 1 - len(Rm) / max(len(R), 1)}
        if Gm and Rm:
            mean = lambda rows, kk: float(np.mean([r[kk] for r in rows]))
            for kk in ("r_p", "r_E", "E_out"):
                d[f"{kk}_generated"], d[f"{kk}_floor"] = mean(Gm, kk), mean(Rm, kk)
            d["S_outcome"] = _score(d["E_out_generated"], d["E_out_floor"], 1.0)
            gs = sorted({r["group"] for r in Gm})
            bo = []
            for _ in range(n_boot):
                pick = rng.choice(gs, len(gs))
                gm = [r for gg in pick for r in Gm if r["group"] == gg]
                rm = [r for gg in pick for r in Rm if r["group"] == gg] or Rm
                bo.append(_score(mean(gm, "E_out"), mean(rm, "E_out"), 1.0))
            d["S_outcome_ci95"] = [float(np.nanquantile(bo, 0.025)), float(np.nanquantile(bo, 0.975))]
        out["twins"][t] = d
        log(f"[outputs] {t}: generated measurable {len(Gm)}/{len(G)}, real {len(Rm)}/{len(R)}")
    out["rows_generated"], out["rows_real"] = gen_rows, real_rows
    return out


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


def _lateral_rms(pts: np.ndarray) -> float:
    pts = pts[np.isfinite(pts).all(1)]
    if len(pts) < 5:
        return float("nan")
    _, sv, _ = np.linalg.svd(pts - pts.mean(0), full_matrices=False)
    return float(sv[-1] / np.sqrt(len(pts)))


def readout_self(gen_frames: np.ndarray, clip, fps: float, checks: bool = True, gate: str = "strict") -> dict:
    """Track B readout of a clip generated from frame 0 (gen_frames[0] ~ the real frame 0, same size as the
    master). Velocities are the clip's own; masses are the true ones (visible in F1: size + material).
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
    dA, dB = v1[0] - v0[0], v1[1] - v0[1]
    rel0 = v0[0] - v0[1]
    p0, p1 = m @ v0, m @ v1
    ke0, ke1 = 0.5 * (m * (v0 ** 2).sum(1)).sum(), 0.5 * (m * (v1 ** 2).sum(1)).sum()
    res.update(velocity_pre=v0.tolist(), velocity_post=v1.tolist(), fit_resid_px=resid_px,
               path_lateral_px=dict(zip(["A_pre", "B_pre", "A_post", "B_post"], lat)),
               path_lateral_max_px=float(np.nanmax(lat)) if np.isfinite(lat).any() else float("nan"),
               r_momentum=float(np.linalg.norm(p1 - p0) / max(np.linalg.norm(p0), 1e-12)),
               ke_ratio=float(ke1 / max(ke0, 1e-12)))
    if np.linalg.norm(dB - dA) < 0.05 * max(np.linalg.norm(rel0), 1e-12):
        return {**res, "measurable": False, "reason": "no_impulse"}
    n = (dB - dA) / np.linalg.norm(dB - dA)
    cos = float(-dA @ dB / max(np.linalg.norm(dA) * np.linalg.norm(dB), 1e-12))
    appr = float(rel0 @ n)
    e = float(-(v1[0] - v1[1]) @ n / appr) if abs(appr) > 1e-12 else float("nan")
    res.update(contact_normal=n.tolist(), impulse_cos=cos, approach_speed=appr, e=e)
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
