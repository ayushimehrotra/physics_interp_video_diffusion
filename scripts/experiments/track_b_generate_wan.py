#!/usr/bin/env python3
"""Output track B (self-consistency, spec §6b, added 2026-11-06): generate from FRAME 0 of each F1 group.

The model sees one frame and the minimal prompt and invents the whole clip (Physics-IQ's image-to-video
protocol); track B then checks whether the collision it generated obeys the laws given ITS OWN pre-collision
motion. Official settings from configs/models/wan22_i2v_a14b.yaml. Frame 0 is identical across a group's twins,
so one video per (group, seed). Resumable; writes cache/<model>/<family>_trackB/<group>/gen_seed<s>.mp4,
sheet_seed<s>.png and _gen.jsonl.

    WAN_A14B_PATH=/root/wan14b_local /root/envs/physics_venv/bin/python scripts/experiments/track_b_generate_wan.py \\
        --model configs/models/wan22_i2v_a14b.yaml --data data/f1 --n 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))
from physlaw.config import load_config, resolve_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=20, help="first n groups (sorted): the spec §6 gate uses 20")
    ap.add_argument("--seeds", default="0", help="comma list, or a range lo-hi (inclusive)")
    ap.add_argument("--groups", default=None, help="comma list of group ids (overrides --n)")
    a = ap.parse_args()
    import imageio.v3 as iio
    import numpy as np
    import torch
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from PIL import Image
    from minimal_prompt_cogvideox import sheet
    from physlaw.data.io import load_clip
    from physlaw.models.cogvideox import minimal_prompt
    cfg = load_config(a.model)
    off = cfg["official"]
    data = resolve_path(a.data)
    out = REPO / "cache" / cfg.name / f"{data.name}_trackB"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "_gen.jsonl"
    if "-" in a.seeds:
        lo, hi = (int(x) for x in a.seeds.split("-"))
        seeds = list(range(lo, hi + 1))
    else:
        seeds = [int(s) for s in a.seeds.split(",")]
    groups = (a.groups.split(",") if a.groups else
              sorted(p.name for p in data.iterdir() if (p / "_GROUP_COMPLETE").exists())[: a.n])
    jobs = [(g, s) for s in seeds for g in groups if not (out / g / f"gen_seed{s}.mp4").exists()]
    if not jobs:
        return
    path = os.environ.get("WAN_A14B_PATH") or cfg["local_path"]
    t0 = time.time()
    vae = AutoencoderKLWan.from_pretrained(path, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(path, vae=vae, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    mod = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    print(f"[trackB] {cfg.name} loaded from {path} in {time.time() - t0:.0f}s; {len(jobs)} clips", flush=True)
    for gid, seed in jobs:
        c = load_clip(data / gid / "conserve_e1")
        f0 = c.frames[0]
        ar = f0.shape[0] / f0.shape[1]
        H = round(np.sqrt(off["max_area"] * ar)) // mod * mod
        W = round(np.sqrt(off["max_area"] / ar)) // mod * mod
        prompt = minimal_prompt(c.labels)
        t1 = time.time()
        img = Image.fromarray(f0)
        if img.size != (W, H):
            img = img.resize((W, H), Image.BICUBIC)
        vid = pipe(image=img, prompt=prompt, negative_prompt=off["negative_prompt"], height=H, width=W,
                   num_frames=off["num_frames"], num_inference_steps=off["num_inference_steps"],
                   guidance_scale=off["guidance_scale"], output_type="np",
                   generator=torch.Generator("cuda").manual_seed(seed)).frames[0]
        u8 = (np.clip(vid, 0, 1) * 255).round().astype(np.uint8)
        (out / gid).mkdir(exist_ok=True)
        iio.imwrite(out / gid / f"gen_seed{seed}.mp4", u8, fps=off["fps"], codec="libx264", quality=9)
        sheet(u8, 1, out / gid / f"sheet_seed{seed}.png", f"{cfg.name} track B {gid} seed{seed} | f0 real, rest generated")
        rec = {"group": gid, "seed": seed, "prompt": prompt, "size": [H, W], "seconds": round(time.time() - t1, 1),
               "model_path": path, "revision": cfg.get("revision")}
        with open(log, "a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[trackB] {gid} seed{seed} {rec['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
