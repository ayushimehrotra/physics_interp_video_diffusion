#!/usr/bin/env python3
"""Two-ball collision gate at OFFICIAL model settings (§65/§67 used non-official ones).

CogVideoX-5b-I2V: 720x480, 49 frames, 8 fps, 50 steps, guidance 6, DPM trailing, use_dynamic_cfg -- the native
format of the existing collision sets (energy_ladder_v1 is 720x480 x 49). Plain I2V from frame 0 with a per-scene
prompt (wan_feasibility.scene_prompt), no negative prompt (the official default). No held/pasted frames.

Readout, from pixels only and without assuming the label's contact frame (the model invents the velocities):
track both spheres (video.track_spheres), contact = frame of minimum centre distance, normal = line of centres
just before contact, pre/post straight-line fits clear of contact (video.measure_momentum), then
e = -(v_A' - v_B').n / (v_A - v_B).n.  'collided' = min distance <= 1.25 x (r_A + r_B).
Run with --real first: the rendered clips must read back at their label e.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
os.environ.setdefault("I2V_PROFILE", "cog5")


def scene_prompt(labels):
    from wan_feasibility import scene_prompt as sp
    return sp(labels)


def readout(frames: np.ndarray, labels: dict, max_residual: float = 0.02) -> dict:
    from physics_interp.video import measure_momentum, track_spheres
    from physics_interp.momentum import physics
    pre = physics.pre_state_from_labels(labels)
    tr = track_spheres(frames)
    pos = tr["position"]                               # [T, 2, 2] world units, nan when lost/merged
    rad = np.nanmedian(tr["radius_px"], axis=0) if "radius_px" in tr else None
    d = np.linalg.norm(pos[:, 0] - pos[:, 1], axis=-1)
    ok = np.isfinite(d)
    res = {"frames_both_tracked": int(ok.sum())}
    if ok.sum() < 10:
        return {**res, "measurable": False, "reason": "lost"}
    r_sum = float(labels["objects"]["A"]["radius"] + labels["objects"]["B"]["radius"])
    # contact: tracking gap (merged blob) or minimum distance
    gaps = np.where(~ok[1:] & ok[:-1])[0]
    c = int(gaps[0] + 1) if len(gaps) else int(np.nanargmin(d))
    res.update(contact_frame=c, min_dist_over_rsum=float(np.nanmin(d) / r_sum) if np.isfinite(np.nanmin(d)) else None,
               collided=bool(len(gaps) or np.nanmin(d) <= 1.25 * r_sum))
    if not res["collided"]:
        return {**res, "measurable": False, "reason": "no_collision"}
    if c < 6 or c > len(frames) - 8:
        return {**res, "measurable": False, "reason": "contact_at_edge"}
    pre_ok = [i for i in range(max(0, c - 12), c - 1) if ok[i]]
    if not pre_ok:
        return {**res, "measurable": False, "reason": "no_pre"}
    # contact normal: extrapolate each ball's pre-contact straight line (least squares) to the first time the
    # centre distance equals r_A + r_B; the line of centres at that instant is the normal (exact for spheres)
    tt = np.array(pre_ok, float)
    fa, fb = (np.polyfit(tt, pos[pre_ok, k], 1) for k in (0, 1))      # each [2 (slope, icpt), 2 (x, y)]
    dp0, dv = fb[1] - fa[1], fb[0] - fa[0]                             # relative position at t=0 and velocity
    A_, B_, C_ = dv @ dv, 2 * dp0 @ dv, dp0 @ dp0 - r_sum ** 2
    disc = B_ * B_ - 4 * A_ * C_
    if A_ > 1e-12 and disc >= 0:
        tc = (-B_ - np.sqrt(disc)) / (2 * A_)
        n = dp0 + dv * tc
    else:                                                              # never touch on the fitted lines
        i0 = pre_ok[-1]
        n = pos[i0, 1] - pos[i0, 0]
    n = n / np.linalg.norm(n)
    m = measure_momentum(frames, [pre.m_a, pre.m_b], contact_frame=c, post_window=range(c + 3, len(frames)))
    v0, v1 = np.asarray(m["velocity_pre"]), np.asarray(m["velocity_post"])
    appr = float(np.dot(v0[0] - v0[1], n))
    e = float(-np.dot(v1[0] - v1[1], n) / appr) if abs(appr) > 1e-6 else float("nan")
    meas = bool(np.isfinite(e) and np.isfinite(m["fit_residual"]) and m["fit_residual"] <= max_residual)
    b_moves = float(np.linalg.norm(v1[1])) > 0.2 * max(float(np.linalg.norm(v0[0])), 1e-6)
    return {**res, "e_pixels": e, "fit_residual": float(m["fit_residual"]), "measurable": meas,
            "struck_ball_moves_after": b_moves, "velocity_pre": v0.tolist(), "velocity_post": v1.tolist(),
            "reason": None if meas else "fit_residual"}


def load_scene(d: Path):
    from PIL import Image
    fr = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in sorted((d / "frames").glob("*.png"))])
    return fr, json.loads((d / "labels.json").read_text())


def load_seg(d: Path, n: int = 49) -> np.ndarray:
    from PIL import Image
    return np.stack([np.asarray(Image.open(p)) for p in sorted((d / "segmentation").glob("*.png"))[:n]])


def fidelity(orig: np.ndarray, rec: np.ndarray, seg: np.ndarray) -> dict:
    """Mean abs pixel error (0-255) over the whole frame and over the ball pixels only."""
    err = np.abs(orig.astype(np.float32) - rec.astype(np.float32)).mean(-1)
    return {"mae_all": float(err.mean()), "mae_ball": float(err[seg > 0].mean()),
            "mae_ball_f0": float(err[0][seg[0] > 0].mean())}


def diag(a, scenes, root: Path):
    """Mentor's checklist: is the VAE or the denoiser what breaks a two-ball collision?

    vae_cog / vae_wan  encode -> decode the RENDERED clip with each model's VAE, read e back from the pixels.
    sdedit             CogVideoX-5b-I2V: forward-noise the rendered clip's latent to t_start, denoise from there
                       with the official loop (frame-0 image latent, scene prompt, 50-step DPM trailing schedule),
                       at dynamic CFG (diffusers' formula) and at fixed CFG 6. t_start 999 = plain generation.
    """
    import torch
    from PIL import Image
    dev = "cuda"
    out_f = root / f"diag_{a.diag}.jsonl"
    done = {json.loads(l)["id"] for l in out_f.read_text().splitlines() if l.strip()} if out_f.exists() else set()

    def emit(rec):
        with open(out_f, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()
                          if k not in ("velocity_pre", "velocity_post")}), flush=True)

    def to_x(fr):
        return torch.from_numpy(fr.astype(np.float32) / 127.5 - 1).permute(3, 0, 1, 2)[None]

    def to_u8(x):
        return ((x[0].permute(1, 2, 3, 0).float().clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()

    if a.diag == "real":
        for d in scenes:
            fr, lab = load_scene(d)
            emit({"id": d.name, "label_e": lab.get("restitution_realised", lab["restitution"]),
                  "ball_px": [int((load_seg(d, 1)[0] == k).sum()) for k in (1, 2)], **readout(fr[:49], lab)})
        return

    if a.diag == "vae_cog":
        from diffusers import AutoencoderKLCogVideoX
        from bounce_gate import MODEL
        vae = AutoencoderKLCogVideoX.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.bfloat16).to(dev)
        vae.enable_tiling()
        vae.enable_slicing()
        for d in scenes:
            if d.name in done:
                continue
            fr, lab = load_scene(d)
            fr, seg = fr[:49], load_seg(d)
            with torch.no_grad():
                rec = to_u8(vae.decode(vae.encode(to_x(fr).to(dev, torch.bfloat16)).latent_dist.mode()).sample)
            save_mp4_(rec, root / "diag" / f"vae_cog_{d.name}.mp4", 8)
            emit({"id": d.name, **fidelity(fr, rec, seg), **readout(rec, lab)})
        return

    if a.diag == "vae_wan":
        from diffusers import AutoencoderKLWan
        from collide_gate_wan import MODEL as WMODEL, to_cond, to_native
        vae = AutoencoderKLWan.from_pretrained(WMODEL, subfolder="vae", torch_dtype=torch.float32).to(dev)
        for d in scenes:
            if d.name in done:
                continue
            fr, lab = load_scene(d)
            fr, seg = fr[:49], load_seg(d)
            pad = np.stack([np.asarray(to_cond(f)) for f in fr])           # the gate's 1280x704 canvas
            base = to_native(pad)                                          # resize there and back, no VAE
            with torch.no_grad():
                rec = to_native(to_u8(vae.decode(vae.encode(to_x(pad).to(dev)).latent_dist.mode()).sample))
            save_mp4_(rec, root / "diag" / f"vae_wan_{d.name}.mp4", 8)
            emit({"id": d.name, **fidelity(fr, rec, seg), "resize_only_mae_ball": fidelity(fr, base, seg)["mae_ball"],
                  **readout(rec, lab)})
        return

    if a.diag == "sdedit":
        import math
        from diffusers import CogVideoXImageToVideoPipeline, CogVideoXDPMScheduler
        from bounce_gate import FRAMES, H, MODEL, STEPS, W
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(dev)
        pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        sch, tr, sf = pipe.scheduler, pipe.transformer, pipe.vae.config.scaling_factor
        sch.set_timesteps(STEPS, device=dev)
        full = sch.timesteps.clone()
        for d in scenes:
            fr, lab = load_scene(d)
            fr, seg = fr[:49], load_seg(d)
            with torch.no_grad():
                pe, ne = pipe.encode_prompt(scene_prompt(lab), negative_prompt="", do_classifier_free_guidance=True,
                                            device=dev, dtype=torch.bfloat16)
                pe = torch.cat([ne, pe])
                z0 = pipe.vae.encode(to_x(fr).to(dev, torch.bfloat16)).latent_dist.mode().permute(0, 2, 1, 3, 4) * sf
                img = pipe.video_processor.preprocess(Image.fromarray(fr[0]), height=H, width=W).to(dev, torch.bfloat16)
                _, im = pipe.prepare_latents(img, 1, tr.config.in_channels // 2, FRAMES, H, W, torch.bfloat16, dev,
                                             torch.Generator(dev).manual_seed(0), None)
                rope = pipe._prepare_rotary_positional_embeddings(H, W, z0.shape[1], dev)
            for t_start in (int(x) for x in a.t_starts.split(",")):
                for cfg in a.cfg_modes.split(","):
                    rid = f"{d.name}|t{t_start}|{cfg}"
                    if rid in done:
                        continue
                    t0 = time.time()
                    ts = full[full <= t_start]
                    eps = torch.randn(z0.shape, generator=torch.Generator(dev).manual_seed(0), device=dev, dtype=z0.dtype)
                    lat = sch.add_noise(z0, eps, ts[:1]).to(torch.bfloat16)
                    old = None
                    with torch.no_grad():
                        for i, t in enumerate(ts):
                            x = torch.cat([torch.cat([lat] * 2), torch.cat([im] * 2)], dim=2)
                            npred = tr(hidden_states=x, encoder_hidden_states=pe, timestep=t.expand(2), ofs=None,
                                       image_rotary_emb=rope, return_dict=False)[0].float()
                            if cfg == "dynamic":       # diffusers' formula, verbatim: raw timestep where a step index belongs
                                g = 1 + 6.0 * ((1 - math.cos(math.pi * ((STEPS - t.item()) / STEPS) ** 5.0)) / 2)
                            elif cfg == "ramp":        # same formula on the remaining-step fraction (N - i)/N ~ t/1000:
                                g = 1 + 6.0 * ((1 - math.cos(math.pi * (t.item() / 1000) ** 5.0)) / 2)   # 7 -> 1, smooth
                            else:
                                g = 6.0
                            u, c = npred.chunk(2)
                            npred = u + g * (c - u)
                            lat, old = sch.step(npred, old, t, ts[i - 1] if i > 0 else None, lat, return_dict=False,
                                                generator=torch.Generator(dev).manual_seed(i))
                            lat = lat.to(torch.bfloat16)
                        vid = pipe.decode_latents(lat)
                    u8 = (np.clip(pipe.video_processor.postprocess_video(vid, output_type="np")[0], 0, 1) * 255
                          ).round().astype(np.uint8)
                    save_mp4_(u8, root / "diag" / f"sdedit_{d.name}_t{t_start}_{cfg}.mp4", 8)
                    emit({"id": rid, "scene": d.name, "t_start": t_start, "cfg": cfg, "n_steps": len(ts),
                          "seconds": round(time.time() - t0, 1), **fidelity(fr, u8, seg), **readout(u8, lab)})
        return
    raise ValueError(a.diag)


def save_mp4_(u8, path, fps):
    import imageio.v3 as iio
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, u8, fps=fps, codec="libx264", pixelformat="yuv420p", output_params=["-crf", "18"], plugin="FFMPEG")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", default="", help="real | vae_cog | vae_wan | sdedit  (see diag())")
    ap.add_argument("--t-starts", default="999,699,399")
    ap.add_argument("--cfg-modes", default="dynamic,fixed")
    ap.add_argument("--dataset-dir", default="datasets/energy_ladder_v1")
    ap.add_argument("--seeds", default="9500-9509")
    ap.add_argument("--arm", default="honest")
    ap.add_argument("--gen-seeds", default="0")
    ap.add_argument("--root", default="/root/collide/cog5")
    ap.add_argument("--real", action="store_true", help="readout on the rendered clips only (validation)")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    scenes = [Path(a.dataset_dir) / f"s{s:05d}_{a.arm}" for s in range(lo, hi + 1)]
    scenes = [d for d in scenes if (d / "_COMPLETE").exists()]
    root = Path(a.root)
    (root / "gen").mkdir(parents=True, exist_ok=True)
    if a.diag:
        return diag(a, scenes, root)
    if a.real:
        for d in scenes:
            fr, lab = load_scene(d)
            r = readout(fr[:49], lab)
            print(d.name, "label e", round(lab["collision"].get("restitution_realised", lab.get("restitution", float("nan"))), 3),
                  "pixels e", None if r.get("e_pixels") is None else round(r["e_pixels"], 3), "res", r.get("fit_residual"),
                  r.get("reason"), "contact", r.get("contact_frame"), flush=True)
        return

    import torch
    from transformers import AutoTokenizer, T5EncoderModel
    from diffusers import CogVideoXImageToVideoPipeline, CogVideoXDPMScheduler
    from bounce_gate import FRAMES, H, MODEL, STEPS, W, save_mp4
    from PIL import Image
    jobs = [(d, g) for d in scenes for g in (int(x) for x in a.gen_seeds.split(","))]
    jobs = [j for j in jobs if zlib.crc32(f"{j[0].name}|{j[1]}".encode()) % a.world == a.rank]
    res_f = root / f"results_r{a.rank}.jsonl"
    done = {json.loads(l)["id"] for f in root.glob("results_r*.jsonl") for l in f.read_text().splitlines() if l.strip()}
    jobs = [j for j in jobs if f"{j[0].name}|{j[1]}" not in done]
    if not jobs:
        return
    # prompt embeddings (T5 once, then freed)
    prompts = {d.name: scene_prompt(json.loads((d / "labels.json").read_text())) for d, _ in jobs}
    tok = AutoTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
    te = T5EncoderModel.from_pretrained(MODEL, subfolder="text_encoder", torch_dtype=torch.bfloat16).cuda()
    enc = CogVideoXImageToVideoPipeline(tokenizer=tok, text_encoder=te, vae=None, transformer=None, scheduler=None)
    emb = {}
    with torch.no_grad():
        for p in set(prompts.values()) | {""}:
            emb[p], _ = enc.encode_prompt(prompt=p, negative_prompt=None, do_classifier_free_guidance=False,
                                          num_videos_per_prompt=1, device=torch.device("cuda"), dtype=torch.bfloat16)
    del te, enc
    torch.cuda.empty_cache()
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, text_encoder=None,
                                                         tokenizer=None).to("cuda")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    for d, g in jobs:
        fr, lab = load_scene(d)
        img = Image.fromarray(fr[0])
        t0 = time.time()
        out = pipe(image=img, prompt_embeds=emb[prompts[d.name]], negative_prompt_embeds=emb[""], height=H, width=W,
                   num_frames=FRAMES, num_inference_steps=STEPS, guidance_scale=6.0, use_dynamic_cfg=True,
                   output_type="np", generator=torch.Generator("cuda").manual_seed(g)).frames[0]
        u8 = (np.clip(out, 0, 1) * 255).round().astype(np.uint8)
        fn = root / "gen" / f"{d.name}_g{g}.mp4"
        save_mp4(u8, fn)
        r = readout(u8, lab)
        rec = {"id": f"{d.name}|{g}", "scene": d.name, "gen_seed": g, "prompt": prompts[d.name], "file": str(fn),
               "seconds": round(time.time() - t0, 1), **r}
        with open(res_f, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(rec["id"], rec["seconds"], "collided", r.get("collided"), "measurable", r.get("measurable"),
              "e", r.get("e_pixels"), r.get("reason"), flush=True)


if __name__ == "__main__":
    main()
