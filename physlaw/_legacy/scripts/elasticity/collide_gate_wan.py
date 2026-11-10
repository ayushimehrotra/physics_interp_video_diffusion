#!/usr/bin/env python3
"""Two-ball collision gate on Wan2.2-TI2V-5B at OFFICIAL settings (§67 used 49 frames at 8 fps + pasted frames).

Model card: 1280x704 (landscape) , 121 frames, 24 fps, 50 steps, guidance 5.0; diffusers WanImageToVideoPipeline with
expand_timesteps (Wan2.2 TI2V first-frame conditioning), shipped UniPC scheduler, VAE in fp32.
Frame 0 of the 720x480 render is scaled to 1056x704 and padded with its own floor colour to 1280 (no distortion);
output frames are cropped back to the 1056 centre and resized to 720x480 for the tracker. Readout as collide_gate.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from collide_gate import load_scene, readout, scene_prompt  # noqa: E402

MODEL = "/dev/shm/models/Wan2.2-TI2V-5B-Diffusers"
H, W, FRAMES, FPS, STEPS, CFG = 704, 1280, 121, 24, 50, 5.0
INNER = 1056                                           # 720x480 scaled to height 704


def save_mp4(u8, path, fps):
    import imageio.v3 as iio
    iio.imwrite(path, u8, fps=fps, codec="libx264", pixelformat="yuv420p",
                output_params=["-crf", "18", "-movflags", "+faststart"], plugin="FFMPEG")


def to_cond(frame0: np.ndarray):
    from PIL import Image
    img = Image.fromarray(frame0).resize((INNER, H), Image.BICUBIC)
    floor = tuple(int(x) for x in np.median(frame0.reshape(-1, 3), 0))
    canvas = Image.new("RGB", (W, H), floor)
    canvas.paste(img, ((W - INNER) // 2, 0))
    return canvas


def to_native(frames: np.ndarray) -> np.ndarray:
    from PIL import Image
    x0 = (W - INNER) // 2
    return np.stack([np.asarray(Image.fromarray(f[:, x0:x0 + INNER]).resize((720, 480), Image.BICUBIC)) for f in frames])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="datasets/energy_ladder_v1")
    ap.add_argument("--seeds", default="9500-9509")
    ap.add_argument("--arm", default="honest")
    ap.add_argument("--gen-seed", type=int, default=0)
    ap.add_argument("--root", default="/root/collide/wan")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    a = ap.parse_args()
    import torch
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    lo, hi = (int(x) for x in a.seeds.split("-"))
    scenes = [Path(a.dataset_dir) / f"s{s:05d}_{a.arm}" for s in range(lo, hi + 1)]
    scenes = [d for d in scenes if (d / "_COMPLETE").exists()
              and zlib.crc32(d.name.encode()) % a.world == a.rank]
    root = Path(a.root)
    (root / "gen").mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for f in root.glob("results_r*.jsonl") for l in f.read_text().splitlines() if l.strip()}
    scenes = [d for d in scenes if f"{d.name}|{a.gen_seed}" not in done]
    if not scenes:
        return
    vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(MODEL, vae=vae, torch_dtype=torch.bfloat16, expand_timesteps=True)
    pipe.to("cuda")
    for d in scenes:
        fr, lab = load_scene(d)
        prompt = scene_prompt(lab)
        t0 = time.time()
        out = pipe(image=to_cond(fr[0]), prompt=prompt, height=H, width=W, num_frames=FRAMES,
                   num_inference_steps=STEPS, guidance_scale=CFG, output_type="np",
                   generator=torch.Generator("cuda").manual_seed(a.gen_seed)).frames[0]
        u8 = (np.clip(out, 0, 1) * 255).round().astype(np.uint8)
        fn = root / "gen" / f"{d.name}_g{a.gen_seed}.mp4"
        save_mp4(u8, fn, FPS)
        nat = to_native(u8)
        r = readout(nat, lab)
        rec = {"id": f"{d.name}|{a.gen_seed}", "scene": d.name, "prompt": prompt, "file": str(fn),
               "seconds": round(time.time() - t0, 1), **r}
        with open(root / f"results_r{a.rank}.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(rec["id"], rec["seconds"], "collided", r.get("collided"), "measurable", r.get("measurable"),
              "e", r.get("e_pixels"), r.get("reason"), flush=True)


if __name__ == "__main__":
    main()
