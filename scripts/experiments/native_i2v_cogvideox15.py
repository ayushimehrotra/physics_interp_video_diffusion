#!/usr/bin/env python3
"""Native image-to-video with the OFFICIAL CogVideoX1.5-5B-I2V pipeline (no latent replacement):
does the model continue our scenes coherently from a single pre-contact frame?

    HF_HOME=/root/hf_cache HF_HUB_OFFLINE=1 /root/envs/physics_venv/bin/python \\
        scripts/experiments/native_i2v_cogvideox15.py --groups 2

Settings exactly as the official CogVideo repo's inference/cli_demo.py for 5B I2V models:
CogVideoXDPMScheduler (trailing), use_dynamic_cfg=True, 50 steps, guidance 6, 81 frames, 1360x768,
bf16, VAE tiling + slicing as in the model-card example. Conditioning image = the last input frame
(16 fps sampling) before contact; prompt = the long per-scene description (wood variant: the wood one).
Variants: orig (as rendered) and zoom_wood (360x240 window around the contact point, wood floor),
from scripts/experiments/naturalness_cogvideox.py. Writes results/cogvideox1.5-5b-i2v/f1/native_i2v/.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))
from naturalness_cogvideox import CROP_H, CROP_W, crop_box, long_wood_prompt, retexture, wood  # noqa: E402
from physlaw.config import load_config  # noqa: E402
from physlaw.data.io import list_clips, load_clip  # noqa: E402
from physlaw.models import cogvideox as CX  # noqa: E402
from physlaw.models.base import fit_frames, sample_frames  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="configs/models/cogvideox15_5b_i2v.yaml")
    ap.add_argument("--groups", type=int, default=2)
    ap.add_argument("--variants", nargs="*", default=["orig", "zoom_wood"])
    a = ap.parse_args()
    import imageio.v3 as iio
    import torch
    from diffusers import CogVideoXDPMScheduler, CogVideoXImageToVideoPipeline
    from PIL import Image
    cfg = load_config(a.model)
    off = cfg["official"]
    out = REPO / "results" / cfg.name / "f1" / "native_i2v"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(cfg["local_path"], torch_dtype=torch.bfloat16).to("cuda")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.vae.enable_tiling(); pipe.vae.enable_slicing()
    print(f"loaded in {time.time() - t0:.0f}s; scheduler {pipe.scheduler.__class__.__name__}", flush=True)
    tex = wood(480, 832)
    for g in sorted({p.parent.name for p in list_clips(REPO / "data/f1")})[: a.groups]:
        c = load_clip(REPO / "data/f1" / g / "conserve_e1")
        fpc = c.pl["first_post_contact_frame"]
        fr = sample_frames(master_fps=48, n_master=289, fps=int(off["fps"]), n_frames=int(off["num_frames"]),
                           contact_frame=fpc)
        last_pre = max(f for f in fr if f < fpc)
        x0, y0 = crop_box(c.labels, fpc)
        for var in a.variants:
            frames = c.frames
            if "wood" in var:
                frames = retexture(frames, c.masks, tex)[:, y0:y0 + CROP_H, x0:x0 + CROP_W]
            img = fit_frames(frames[[last_pre]], (int(off["height"]), int(off["width"])), "squash")[0]
            prompt = long_wood_prompt(c.labels) if "wood" in var else CX.scene_prompt(c.labels)
            t1 = time.time()
            v = pipe(image=Image.fromarray(img), prompt=prompt, num_videos_per_prompt=1,
                     num_inference_steps=int(off["num_inference_steps"]), num_frames=int(off["num_frames"]),
                     guidance_scale=float(off["guidance_scale"]), use_dynamic_cfg=bool(off["use_dynamic_cfg"]),
                     height=int(off["height"]), width=int(off["width"]),
                     generator=torch.Generator(device="cuda").manual_seed(0), output_type="np").frames[0]
            v = (np.clip(v, 0, 1) * 255).astype(np.uint8)
            pick = np.linspace(0, len(v) - 1, 7).astype(int)
            iio.imwrite(out / f"{g}_{var}.png", np.concatenate([v[i] for i in pick], 1)[::4, ::4])
            iio.imwrite(out / f"{g}_{var}.mp4", v, fps=16, codec="libx264", quality=8)
            print(f"{g} {var}: {time.time() - t1:.0f}s, temporal change {np.abs(np.diff(v.astype(float), axis=0)).mean():.2f}",
                  flush=True)


if __name__ == "__main__":
    main()
