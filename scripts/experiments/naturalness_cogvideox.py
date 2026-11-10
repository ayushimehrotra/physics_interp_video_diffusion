#!/usr/bin/env python3
"""Does CogVideoX keep the balls if the (already rendered) scenes look more natural? No re-rendering:
the rendered master frames are transformed, identically for every twin of a group, then continued by the
adapter exactly as in the benchmark (official settings, context-latent replacement, seed 0).

    HF_HOME=/root/hf_cache HF_HUB_OFFLINE=1 /root/envs/physics_venv/bin/python \\
        scripts/experiments/naturalness_cogvideox.py --model configs/models/cogvideox_2b.yaml --groups 4

Variants
  orig            as rendered (fixed top-down camera, uniform grey floor, balls ~30-60 px)
  zoom            360x240 master window centred on the contact point, upscaled 2x (balls ~2x larger)
  zoom_wood       zoom + the grey floor replaced by a procedural wood texture; shading is kept by
                  multiplying the texture by each pixel's luminance relative to the floor (shadows
                  survive), ball pixels are left untouched
  zoom_wood_long  zoom_wood + a long, literal per-scene prompt describing a wooden table (CogVideoX
                  model cards recommend long prompts)
Scores (pixel space, tracker of physlaw/eval/outputs.py with a per-pixel temporal-median background):
  tracked_after_ctx  fraction of post-context frames with both balls found (real clip ~1.0)
  measurable         a contact and >= 4 tracked frames in the post window
  E_out              ||v_gen - v_real|| / ||v_real - v_pre||  (post velocities in px/frame; 1 = no better
                     than "no collision", 0 = exact)
Writes results/<model>/<family>/naturalness/{summary.json, summary.md, <group>_<variant>.png/.mp4}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from physlaw.config import load_config, resolve_path  # noqa: E402
from physlaw.data.io import list_clips, load_clip  # noqa: E402
from physlaw.eval import outputs as O  # noqa: E402
from physlaw.models import cogvideox as CX  # noqa: E402
from physlaw.models.base import fit_frames  # noqa: E402
from physlaw.models.registry import load_adapter  # noqa: E402

CROP_W, CROP_H = 360, 240


def wood(H: int, W: int, seed: int = 0) -> np.ndarray:
    """Procedural light oak planks (uint8 [H, W, 3]); fixed seed, the same for every clip."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    warp = gaussian_filter(rng.normal(size=(H, W)), 25) * 60
    plank = (yy // 96).astype(int)
    offs = rng.uniform(0, 400, plank.max() + 1)[plank]
    rings = np.sin(2 * np.pi * (xx + offs + warp) / 38.0 + 0.4 * np.sin(yy / 17.0))
    grain = gaussian_filter(rng.normal(size=(H, W)), (0.6, 6)) * 2.5
    t = np.clip(0.55 + 0.25 * rings + 0.12 * grain, 0, 1)
    light, dark = np.array([214, 170, 118.]), np.array([158, 108, 64.])
    img = dark + (light - dark) * t[..., None]
    img *= (1 - 0.18 * ((yy % 96) < 2))[..., None]                 # plank seams
    tone = rng.uniform(0.93, 1.05, plank.max() + 1)[plank]          # plank-to-plank tone
    return np.clip(img * tone[..., None], 0, 255).astype(np.uint8)


def retexture(frames: np.ndarray, masks: np.ndarray, tex: np.ndarray) -> np.ndarray:
    import cv2
    out = np.empty_like(frames)
    first_floor = frames[0][masks[0] == 0].reshape(-1, 3).astype(float)
    floor_l = np.median(first_floor.mean(1))
    for t in range(len(frames)):
        f = frames[t].astype(float)
        ratio = np.clip(f.mean(2) / floor_l, 0, 1.1)[..., None]
        o = tex * ratio
        ball = cv2.dilate((masks[t] > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        o[ball] = f[ball]
        out[t] = np.clip(o, 0, 255).astype(np.uint8)
    return out


def crop_box(labels, fpc):
    pf = labels["per_frame"][fpc]
    cx = (pf["A"]["pixel"][0] + pf["B"]["pixel"][0]) / 2
    cy = (pf["A"]["pixel"][1] + pf["B"]["pixel"][1]) / 2
    W0, H0 = labels["physlaw"]["master_format"]["resolution"]
    x0 = int(np.clip(round(cx - CROP_W / 2), 0, W0 - CROP_W))
    y0 = int(np.clip(round(cy - CROP_H / 2), 0, H0 - CROP_H))
    return x0, y0


def long_wood_prompt(labels):
    o = labels["objects"]
    small, large = sorted(("A", "B"), key=lambda k: o[k]["radius"])
    look = {"rubber": "matte red rubber ball", "metal": "polished silver steel ball"}
    name = {small: f"smaller {look[o[small]['material']]}", large: f"larger {look[o[large]['material']]}"}
    motion = (f"the {name['A']} and the {name['B']} roll toward each other" if labels.get("b_moving") else
              f"the {name['A']} rolls toward the {name['B']}, which is resting on the table")
    return (f"A close-up, top-down view of a light oak wooden table with visible wood grain and plank seams, softly "
            f"lit by daylight. On the table there are exactly two balls: a {name[small]} and a {name[large]}, each "
            f"casting a soft shadow on the wood. Slowly and steadily, {motion}. They collide once with a clean impact "
            f"and then roll apart in straight lines across the table, staying fully visible, keeping their size and "
            f"shape. The camera does not move at all. Realistic, natural motion, sharp focus, photorealistic.")


def pixel_readout(pos, k, r_sum_px, post_gap=3, post_window=8):
    """pos [T, 2, 2] (row, col); velocities in px / input frame."""
    T = len(pos)
    ok = np.isfinite(pos).all((1, 2))
    d = np.linalg.norm(pos[:, 0] - pos[:, 1], axis=1)
    res = {"tracked_after_ctx": float(ok[k:].mean())}
    pre = [t for t in range(max(0, k - 8), k) if ok[t]]
    if len(pre) >= 3:
        tt = np.array(pre, float)
        res["v_pre"] = np.stack([np.polyfit(tt, pos[pre, i], 1)[0] for i in range(2)]).tolist()
    near = [t for t in range(max(k - 3, 0), T) if (not ok[t]) or d[t] <= 1.25 * r_sum_px]
    if not near or not ok[k:].any():
        return {**res, "measurable": False, "reason": "no_contact" if ok[k:].any() else "lost_after_context"}
    c = near[0]
    win = [t for t in range(c + post_gap, min(T, c + post_gap + post_window)) if ok[t]]
    if len(win) < 4:
        return {**res, "contact": c, "measurable": False, "reason": "lost_after_contact"}
    tt = np.array(win, float)
    v = np.stack([np.polyfit(tt, pos[win, i], 1)[0] for i in range(2)])
    return {**res, "contact": c, "v_post": v.tolist(), "measurable": True}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="configs/models/cogvideox_2b.yaml")
    ap.add_argument("--data", default="data/f1")
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--variants", nargs="*", default=["orig", "zoom", "zoom_wood", "zoom_wood_long"])
    a = ap.parse_args()
    import imageio.v3 as iio
    import torch
    mcfg = load_config(a.model)
    data = resolve_path(a.data)
    out = REPO / "results" / mcfg.name / data.name / "naturalness"
    out.mkdir(parents=True, exist_ok=True)
    m = load_adapter(mcfg)
    set_prompt = m.set_prompt                            # caches embeddings per prompt text

    default_prompt = mcfg.get("prompt", CX.DEFAULT_PROMPT)
    groups = sorted({p.parent.name for p in list_clips(data)})[: a.groups]
    tex = wood(480, 832)
    rows = []
    for g in groups:
        c = load_clip(data / g / "conserve_e1")
        fr, k = m.input_frames(c.labels), m.context_frames(c.labels)
        fpc = c.pl["first_post_contact_frame"]
        x0, y0 = crop_box(c.labels, fpc)
        for var in a.variants:
            frames, masks = c.frames, c.masks
            if "wood" in var:
                frames = retexture(frames, masks, tex)
            if var != "orig":
                frames = frames[:, y0:y0 + CROP_H, x0:x0 + CROP_W]
                masks = masks[:, y0:y0 + CROP_H, x0:x0 + CROP_W]
            set_prompt(long_wood_prompt(c.labels) if var.endswith("_long") else default_prompt)
            real = fit_frames(frames[fr], (m.H, m.W), "squash")
            mk = fit_frames(np.repeat(masks[fr][..., None], 3, -1), (m.H, m.W), "squash")[..., 0]
            mk = np.where(np.isin(mk, (1, 2)), mk, 0)
            x = m.to_tensor(frames[fr])
            m._lat_cache = m._ctx_cache = None
            _, gen = m.forecast(x, k, seed=0, return_video=True)
            sig = O.ball_signatures(real[:k], mk[:k])
            Hs, Ws = frames.shape[1], frames.shape[2]
            pf = c.labels["per_frame"][fr[0]]
            sx, sy = m.W / Ws, m.H / Hs
            ox, oy = (0, 0) if var == "orig" else (x0, y0)
            start = {o: ((pf[o]["pixel"][1] - oy) * sy, (pf[o]["pixel"][0] - ox) * sx) for o in "AB"}
            r_sum = sig["A"]["radius_px"] + sig["B"]["radius_px"]
            res = {}
            for name, vid in (("real", real), ("gen", gen)):
                bg = np.median(vid, 0)
                pos = O.track(vid, sig, start, fg_thresh=30.0 if "wood" not in var else 40.0, bg=bg)["pixel"]
                res[name] = pixel_readout(pos, k, r_sum)
            rr, gg = res["real"], res["gen"]
            row = {"group": g, "variant": var, "ball_diam_px": float(2 * np.mean([sig[o]["radius_px"] for o in "AB"])),
                   "real": {kk: v for kk, v in rr.items() if kk in ("tracked_after_ctx", "measurable", "reason")},
                   "gen": {kk: v for kk, v in gg.items() if kk in ("tracked_after_ctx", "measurable", "reason")}}
            if rr.get("measurable") and gg.get("measurable") and "v_pre" in rr:
                vr, vgn, vp = np.array(rr["v_post"]), np.array(gg["v_post"]), np.array(rr["v_pre"])
                row["E_out"] = float(np.linalg.norm(vgn - vr) / max(np.linalg.norm(vr - vp), 1e-9))
            rows.append(row)
            pick = [k - 9, k - 1, k + 3, k + 7, k + 12, k + 19, len(fr) - 1]
            iio.imwrite(out / f"{g}_{var}.png", np.concatenate(
                [np.concatenate([real[i] for i in pick], 1), np.concatenate([gen[i] for i in pick], 1)], 0)[::2, ::2])
            iio.imwrite(out / f"{g}_{var}.mp4", np.concatenate([real, gen], 1), fps=m.fps, codec="libx264", quality=8)
            print(json.dumps(row), flush=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=1))
    lines = ["| variant | ball diam (px) | gen: both balls tracked after context | gen measurable | E_out (gen vs real) | real tracked / measurable |",
             "|---|---|---|---|---|---|"]
    for var in a.variants:
        R = [r for r in rows if r["variant"] == var]
        E = [r["E_out"] for r in R if "E_out" in r]
        lines.append(f"| {var} | {np.mean([r['ball_diam_px'] for r in R]):.0f} | "
                     f"{np.mean([r['gen']['tracked_after_ctx'] for r in R]):.2f} | "
                     f"{sum(r['gen']['measurable'] for r in R)}/{len(R)} | "
                     f"{'n/a' if not E else f'{np.median(E):.2f} (median of {len(E)})'} | "
                     f"{np.mean([r['real']['tracked_after_ctx'] for r in R]):.2f} / {sum(r['real']['measurable'] for r in R)}/{len(R)} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
