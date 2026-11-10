#!/usr/bin/env python3
"""§70 step 1: can CogVideoX1.5-5B-I2V generate a measurable single-ball bounce? + the e readout.

Modes
  embed   encode the prompts once with T5 (GPU), cache to <root>/embeds.pt
  gen     worker: generate unsteered clips for its shard of (seed, variant) jobs; several workers
          can share one GPU (--rank/--world); results -> <root>/results_r<rank>.jsonl
  measure measure e from pixels on rendered reference clips (tracker validation)

e readout (pixels only). Track the ball's colour blob; fit a parabola to the fall arc and to the
first rebound arc (contact frames excluded); contact time = where the two parabolas meet;
e_v = |rebound slope| / |fall slope| at contact (restitution by definition, valid for any initial
velocity); e_h = sqrt(rebound apex height / fall apex height) above the contact height (assumes
release from rest). A clip is measurable when the ball is intact, both arcs curve downward with a
consistent gravity (curvatures within 35%), the fits are tight, and e_v and e_h agree within 0.1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

# Official settings per model (model cards + CogVideo inference/cli_demo.py): 50 steps, guidance 6,
# CogVideoXDPMScheduler with trailing spacing, use_dynamic_cfg=True.
PROFILES = {
    "cog15": dict(model="/dev/shm/models/CogVideoX1.5-5B-I2V", H=768, W=1360, frames=81, fps=16),
    "cog5": dict(model="/dev/shm/models/CogVideoX-5b-I2V", H=480, W=720, frames=49, fps=8),
}
PROFILE = os.environ.get("I2V_PROFILE", "cog5")
_P = PROFILES[PROFILE]
MODEL = os.environ.get("I2V_MODEL", _P["model"])
H, W, FRAMES, FPS = _P["H"], _P["W"], _P["frames"], _P["fps"]
STEPS = 50
SCENE = ("A single continuous shot from a fixed camera: a purple sphere with a fine matte rubber-grain texture "
         "hangs motionless in mid-air above a plain warm honey-toned floor, in front of a plain off-white wall, "
         "seen from the side.")
PROMPTS = {
    "drop": SCENE + " The ball is released, falls straight down, hits the floor and bounces back up.",
    "bouncy": SCENE + (" The ball is released, falls straight down and bounces on the floor several times, "
                       "each bounce lower than the last, until it comes to rest."),
}
NEG = "camera motion, zoom, cut, two balls, duplicated ball, deformed ball, blurry, distorted, text"
VARIANTS = {"drop": ("drop", ""), "bouncy_neg": ("bouncy", NEG)}


# ------------------------------------------------------------------ measurement ------------------- #

def track(frames: np.ndarray, ref: np.ndarray, area0: float, thr: float = 60.0):
    """Largest blob near the ball colour per frame -> (x, y, area) or None."""
    from scipy import ndimage
    floor = np.median(frames[0].reshape(-1, 3).astype(np.float32), 0)
    thr = min(thr, 0.6 * float(np.linalg.norm(ref - floor)))
    out = []
    for f in frames:
        m = np.linalg.norm(f.astype(np.float32) - ref, axis=-1) < thr
        lab, n = ndimage.label(m)
        if n == 0:
            out.append(None)
            continue
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        k = int(np.argmax(sizes))
        if sizes[k] < 0.15 * area0:
            out.append(None)
            continue
        ys, xs = np.nonzero(lab == k + 1)
        second = float(np.sort(sizes)[-2]) if n > 1 else 0.0
        out.append((float(xs.mean()), float(ys.mean()), float(sizes[k]), second))
    return out


def _fit(t, y):
    A = np.c_[t * t, t, np.ones_like(t)]
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return c, float(np.sqrt(np.mean((A @ c - y) ** 2)))


def first_bounce_e(y: np.ndarray) -> dict:
    """Primary readout, robust at 8 fps: e1 = sqrt(first rebound height / fall height).

    floor = lowest ball position in the clip (max image-y); top = highest point before the first
    approach to the floor (handles a ball that first rises); contact = first sample within 15% of the
    drop of the floor; rebound apex = first local minimum of y after the first local maximum at/after
    contact. Needs one sample near the apex, not a fitted arc (a rebound at e = 0.3 lasts ~4 frames).
    """
    n = len(y)
    floor = float(y.max())
    near = np.nonzero(y >= floor - 0.15 * (floor - y.min()))[0]
    c = int(near[0])
    top = float(y[: c + 1].min())
    h_f = floor - top
    if h_f < 60 or c >= n - 2:
        return {"e1": float("nan"), "e1_reason": "no_fall" if h_f < 60 else "late_contact"}
    i = c
    while i + 1 < n and y[i + 1] >= y[i]:          # ride down to the contact sample (local max)
        i += 1
    j = i
    while j + 1 < n and y[j + 1] <= y[j]:          # ride up to the rebound apex (local min)
        j += 1
    if j == i:
        return {"e1": 0.0, "e1_contact": i, "e1_apex": j, "e1_reason": "no_rebound", "floor_px": floor, "top_px": top}
    h_r = floor - float(y[j])
    return {"e1": float(np.sqrt(max(h_r, 0.0) / h_f)), "e1_contact": i, "e1_apex": j, "floor_px": floor,
            "top_px": top, "e1_reason": "apex_at_end" if j == n - 1 else None}


def measure_bounce(tr, area0: float) -> dict:
    n = len(tr)
    det = np.array([p is not None for p in tr])
    res = {"det_frac": float(det.mean())}
    if det.mean() < 0.8:
        return {**res, "ok": False, "reason": "lost"}
    area = np.array([p[2] if p else np.nan for p in tr]) / area0
    second = np.array([p[3] if p else 0 for p in tr]) / area0
    res["area_ok_frac"] = float(np.nanmean((area > 0.5) & (area < 2.0)))
    # size consistency against the clip's own start (the conditioning ball), frames with the ball fully in view
    a_ref = float(np.nanmedian(area[:3]))
    rel = area / a_ref
    res["size_ratio_min"], res["size_ratio_max"] = float(np.nanmin(rel)), float(np.nanmax(rel))
    res["size_ok"] = bool(np.nanmin(rel) > 1 / 1.5 and np.nanmax(rel) < 1.5)
    res["second_blob_frac"] = float(np.mean(second > 0.3))
    t = np.arange(n, dtype=float)
    x = np.array([p[0] if p else np.nan for p in tr])
    y = np.array([p[1] if p else np.nan for p in tr])
    good = ~np.isnan(y)
    y_i = np.interp(t, t[good], y[good])
    res["y"] = [round(v, 1) for v in y_i]
    res["x_drift"] = float(np.nanmax(x) - np.nanmin(x))
    drop = float(y_i.max() - y_i[0])
    res["drop_px"] = drop
    if drop < 60:
        return {**res, "ok": False, "reason": "no_fall"}
    res.update(first_bounce_e(y_i))
    # contacts: prominent local maxima of image-y (the ball lowest), at least 70% of the drop below the
    # start. Prominence, not a global threshold: the generated ball drifts in depth, so later landings sit a
    # few px lower than the first. Each contact is the FIRST frame of its plateau (a resting ball is flat).
    from scipy.signal import find_peaks
    pad = np.r_[y_i, y_i[-1] - 1e-3]                 # a peak at the last frame still counts
    pk, _ = find_peaks(pad, prominence=max(4.0, 0.02 * drop), plateau_size=1)   # 8 fps: low-e rebounds are ~4 frames
    cand = []
    for p in pk:
        if p >= n or y_i[p] < y_i[0] + 0.7 * drop:
            continue
        q = p
        while q > 0 and y_i[q - 1] >= y_i[p] - 0.02 * drop:
            q -= 1
        cand.append(int(q))
    if not cand:
        return {**res, "ok": False, "reason": "no_contact"}
    c1 = cand[0]
    nxt = [i for i in cand if i > c1 + 2]
    c2 = nxt[0] if nxt else n - 1
    res["contacts"] = cand
    # fall arc: from release (last frame within 2 px of the start before c1) to c1-1
    rel = 0
    for i in range(c1):
        if abs(y_i[i] - y_i[0]) <= 2.0:
            rel = i
    fa = np.arange(max(rel, c1 - 14), c1)          # last <= 14 frames before contact
    ra = np.arange(c1 + 1, c2)                      # rebound arc
    # dead landing: after first contact the ball never rises more than 3% of the drop
    if c1 < n - 4 and (y_i[c1] - y_i[c1:].min()) < 0.03 * drop:
        res.update(c1=c1, e_v=0.0, e_h=0.0, landed=True)
        bad = [r for r, c in (("appearance", res["area_ok_frac"] < 0.8), ("second_ball", res["second_blob_frac"] > 0.2)) if c]
        return {**res, "ok": not bad, "reason": ",".join(bad) or "dead_landing_e0"}
    if len(fa) < 3 or len(ra) < 3:
        return {**res, "ok": False, "reason": "short_arc", "fall_n": len(fa), "reb_n": len(ra)}
    cf, rf = _fit(fa, y_i[fa])
    cr, rr = _fit(ra, y_i[ra])
    res.update(release=rel, c1=c1, c2=c2, fit_rms=[rf, rr], curv=[float(cf[0]), float(cr[0])])
    # contact time: root of (fall - rebound) nearest c1
    d = cf - cr
    roots = np.roots(d) if abs(d[0]) > 1e-9 else np.array([-d[2] / d[1]])
    roots = roots[np.isreal(roots)].real
    if len(roots) == 0:
        return {**res, "ok": False, "reason": "no_intersection"}
    tc = float(roots[np.argmin(np.abs(roots - c1))])
    yc = float(np.polyval(cf, tc))
    vf = float(np.polyval(np.polyder(cf), tc))     # +ve: moving down
    vr = float(np.polyval(np.polyder(cr), tc))     # -ve: moving up
    e_v = -vr / vf if vf > 0 else np.nan
    apex_r = cr[2] - cr[1] ** 2 / (4 * cr[0]) if cr[0] > 0 else np.nan
    h_r, h_f = yc - apex_r, yc - float(y_i[0])
    e_h = float(np.sqrt(h_r / h_f)) if (h_r > 0 and h_f > 0) else np.nan
    res.update(tc=tc, y_contact=yc, v_fall=vf, v_reb=vr, e_v=float(e_v), e_h=e_h)
    reasons = []
    if res["area_ok_frac"] < 0.8:
        reasons.append("appearance")
    if res["second_blob_frac"] > 0.2:
        reasons.append("second_ball")
    if not (cf[0] > 0 and cr[0] > 0):
        reasons.append("not_ballistic")
    elif abs(cr[0] / cf[0] - 1) > 0.35:
        reasons.append("gravity_mismatch")
    if max(rf, rr) > max(3.0, 0.03 * drop):
        reasons.append("fit_rms")
    if not (np.isfinite(e_v) and 0 < e_v < 1.2):
        reasons.append("e_range")
    elif np.isfinite(e_h) and abs(e_v - e_h) > 0.1:
        reasons.append("e_disagree")
    res["ok"] = not reasons
    res["reason"] = ",".join(reasons) or None
    return res


def load_cond(rest_dir: Path):
    from PIL import Image
    img = Image.open(rest_dir / "frames" / "frame_0000.png").convert("RGB")
    f0 = np.asarray(img)
    m0 = np.load(rest_dir / "ball_mask_f0.npy")
    return img, f0, m0, f0[m0].reshape(-1, 3).astype(np.float32).mean(0), float(m0.sum())


# ------------------------------------------------------------------ modes ------------------------- #

def mode_embed(root: Path):
    import torch
    from transformers import AutoTokenizer, T5EncoderModel
    from diffusers import CogVideoXImageToVideoPipeline
    tok = AutoTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
    te = T5EncoderModel.from_pretrained(MODEL, subfolder="text_encoder", torch_dtype=torch.bfloat16).cuda()
    pipe = CogVideoXImageToVideoPipeline(tokenizer=tok, text_encoder=te, vae=None, transformer=None, scheduler=None)
    out = {}
    with torch.no_grad():
        for p in list(PROMPTS.values()) + ["", NEG]:
            e, _ = pipe.encode_prompt(prompt=p, negative_prompt=None, do_classifier_free_guidance=False,
                                      num_videos_per_prompt=1, device=torch.device("cuda"), dtype=torch.bfloat16)
            out[p] = e.cpu()
    torch.save(out, root / "embeds.pt")
    print("embeds", len(out))


def load_pipe(root: Path):
    import torch
    from diffusers import CogVideoXImageToVideoPipeline
    from diffusers import CogVideoXDPMScheduler
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, text_encoder=None,
                                                         tokenizer=None).to("cuda")
    # official CogVideo cli_demo: DPM scheduler, trailing spacing (with use_dynamic_cfg=True at call time)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    emb = {k: v.cuda() for k, v in torch.load(root / "embeds.pt").items()}
    return pipe, emb


def save_mp4(u8, path):
    import imageio.v3 as iio
    iio.imwrite(path, u8, fps=FPS, codec="libx264", pixelformat="yuv420p",
                output_params=["-crf", "18", "-movflags", "+faststart"], plugin="FFMPEG")


def mode_gen(root: Path, rest: Path, seeds, variants, rank: int, world: int):
    import torch
    img, f0, m0, ref, area0 = load_cond(rest)
    pipe, emb = load_pipe(root)
    out_dir = root / "gen"
    out_dir.mkdir(parents=True, exist_ok=True)
    res_f = root / f"results_r{rank}.jsonl"
    done = set()
    for f in root.glob("results_r*.jsonl"):
        done |= {json.loads(l)["id"] for l in f.read_text().splitlines() if l.strip()}
    jobs = [(s, v) for v in variants for s in seeds]
    jobs = [j for j in jobs if zlib.crc32(f"{j[0]}|{j[1]}".encode()) % world == rank]
    for s, v in jobs:
        jid = f"{s}|{v}"
        if jid in done:
            continue
        pk, nk = VARIANTS[v]
        t0 = time.time()
        out = pipe(image=img, prompt_embeds=emb[PROMPTS[pk]], negative_prompt_embeds=emb[nk], height=H, width=W,
                   num_frames=FRAMES, num_inference_steps=STEPS, guidance_scale=6.0, use_dynamic_cfg=True, output_type="np",
                   generator=torch.Generator("cuda").manual_seed(s)).frames[0]
        u8 = (np.clip(out, 0, 1) * 255).round().astype(np.uint8)
        fn = out_dir / f"{s}_{v}.mp4"
        save_mp4(u8, fn)
        m = measure_bounce(track(u8, ref, area0), area0)
        rec = {"id": jid, "seed": s, "variant": v, "file": str(fn), "seconds": round(time.time() - t0, 1), **m}
        with open(res_f, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(jid, rec["seconds"], m.get("ok"), m.get("reason"), m.get("e_v"), m.get("e_h"), flush=True)


def mode_remeasure(root: Path, rest: Path):
    """Re-run measure_bounce on every generated mp4 (after a readout change); -> <root>/remeasured.jsonl."""
    import imageio.v3 as iio
    _, f0, m0, ref, area0 = load_cond(rest)
    out = []
    for f in sorted((root / "gen").glob("*.mp4")):
        u8 = iio.imread(f, plugin="FFMPEG")
        m = measure_bounce(track(u8, ref, area0), area0)
        m.pop("y", None)
        out.append({"id": f.stem, **m})
        print(f.stem, "e1 %s" % m.get("e1"), m.get("e1_reason"), m.get("ok"), m.get("reason"), "e_v %.3f e_h %.3f" % (m.get("e_v", np.nan), m.get("e_h", np.nan)),
              "contacts", m.get("contacts"), flush=True)
    (root / "remeasured.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))


def mode_measure(dirs, rest: Path):
    from PIL import Image
    _, f0, m0, ref, area0 = load_cond(rest)
    for d in dirs:
        d = Path(d)
        fr = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in sorted((d / "frames").glob("*.png"))])
        lab = json.loads((d / "labels.json").read_text())
        m = measure_bounce(track(fr, ref, area0), area0)
        print(d.name, "label e", lab["restitution"], "e1 %.3f" % m.get("e1", np.nan), "e_v %.3f e_h %.3f" % (m.get("e_v", np.nan), m.get("e_h", np.nan)),
              "ok", m["ok"], m["reason"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["embed", "gen", "measure", "remeasure"])
    ap.add_argument("--root", type=Path, default=Path(f"/root/bounce/{PROFILE}/gate"))
    ap.add_argument("--rest", type=Path, default=Path(f"/root/bounce/{PROFILE}/rest"))
    ap.add_argument("--seeds", default="0-11")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("dirs", nargs="*")
    a = ap.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    lo, hi = (int(v) for v in a.seeds.split("-"))
    if a.mode == "embed":
        mode_embed(a.root)
    elif a.mode == "gen":
        mode_gen(a.root, a.rest, list(range(lo, hi + 1)), a.variants, a.rank, a.world)
    elif a.mode == "remeasure":
        mode_remeasure(a.root, a.rest)
    else:
        mode_measure(a.dirs, a.rest)


if __name__ == "__main__":
    main()
