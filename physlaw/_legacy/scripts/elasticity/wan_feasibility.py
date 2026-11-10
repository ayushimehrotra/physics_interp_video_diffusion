#!/usr/bin/env python3
"""§67 — can Wan2.2-TI2V-5B produce a measurable two-ball collision? (The §65 gate, on Wan.)

    python scripts/elasticity/wan_feasibility.py --dataset-dir datasets/energy_ladder_v1 \
        --seeds 9500-9502 --variants free hold hold_strict -o outputs/wan_feasibility

Same scenes, readout and gate as i2v_feasibility.py (§65): realised e from tracked pixels,
measurable iff all four straight-line fits have residual <= 0.02.

How Wan2.2-TI2V conditions, and why "hold" is native here: the 5B TI2V model feeds the
image's latent in *clean* at latent frame 0 and gives those tokens timestep 0 (per-token
timesteps, ``expand_timesteps``), denoising only the rest. ``hold`` extends that mask to
latent frames 0-5 — pixel frames 0-20, strictly pre-contact (the VAE is causal, frame 0
alone then groups of 4, so latent 5 ends at frame 20) — so the approach is clean context
and the aftermath is the model's. No re-noised pasting as in runner.inpaint.

Clips are 49 frames at 8 fps (the model was trained at 24 fps, so motion per frame is 3x
faster than it usually sees), upsampled 720x480 -> 1056x704 (exactly 1056/720 = 1.467x;
both sides divisible by the 32-pixel token). 13 latent frames x 22 x 33 = 9438 tokens.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))  # noqa: E401,E702
import _momentum_path  # noqa: F401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from i2v_feasibility import STRICT_PROMPT, PROMPT, parse_seeds, pixel_restitution, to_native
from physics_interp.momentum import physics
from physics_interp.video import load_frames, save_video

# Wan's own default negative prompt (the one it was tuned with), plus the §65 extras for strict.
WAN_NEGATIVE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
                "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
                "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
STRICT_NEGATIVE = WAN_NEGATIVE + ", extra balls, third ball, duplicate objects, new objects appearing, " \
                  "objects disappearing, color change, camera motion, zoom"
SCENE = "<scene>"   # placeholder: prompt built per scene from labels.json (scene_prompt)
# name: (held latent frames, prompt, negative, how they are held). "mask" = Wan's own clean
# context (timestep 0 on those tokens); "paste" = frame 0 by mask, frames 1.. re-noised to
# each step's level and re-imposed (flow-matching Euler, as runner.inpaint does on CogVideoX).
# hold_strict's fixed "one red, one metal" prompt is wrong for scenes with two balls of one
# material (s09501 rubber/rubber, s09502 metal/metal) — kept only to reproduce the first run.
VARIANTS = {
    "free": (1, PROMPT, WAN_NEGATIVE, "mask"),
    "hold": (6, PROMPT, WAN_NEGATIVE, "mask"),
    "hold_strict": (6, STRICT_PROMPT, STRICT_NEGATIVE, "mask"),
    "free_scene": (1, SCENE, STRICT_NEGATIVE, "mask"),
    "hold_scene": (6, SCENE, STRICT_NEGATIVE, "mask"),
    "paste_scene": (6, SCENE, STRICT_NEGATIVE, "paste"),
}
LOOKS = {"rubber": "matte red rubber ball", "metal": "shiny grey metal ball"}


def scene_prompt(labels: dict) -> str:
    """Describe exactly the two balls this scene has, and which one moves."""
    o = labels["objects"]
    small, large = sorted(("A", "B"), key=lambda k: o[k]["radius"])
    name = {small: f"smaller {LOOKS[o[small]['material']]}", large: f"larger {LOOKS[o[large]['material']]}"}
    motion = (f"The {name['A']} and the {name['B']} roll toward each other" if labels.get("b_moving")
              else f"The {name['A']} rolls into the {name['B']}, which is at rest")
    return (f"Top-down view of exactly two balls on a plain flat grey floor: a {name[small]} and a "
            f"{name[large]}. {motion}; they collide once and then both roll apart in straight lines at "
            f"constant speed. Only these two balls are ever visible. Fixed camera, no zoom, no cuts.")
HW = (704, 1056)
# UniPC's corrector solves a tiny linear system on the sample's device; cuSOLVER fails to
# create a handle on the RTX 5090 pod (CUSOLVER_STATUS_INTERNAL_ERROR). MAGMA does not.
torch.backends.cuda.preferred_linalg_library("magma")


class Wan:
    def __init__(self, repo: str, device="cuda"):
        from diffusers import AutoencoderKLWan, WanPipeline
        self.device = device
        self.pipe = WanPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)
        self.pipe.vae.to(device, torch.float32)        # the Wan VAE is run in fp32 upstream
        self.pipe.transformer.to(device)
        self.pipe.text_encoder.to(device)
        self.vae = self.pipe.vae
        c = self.vae.config
        self.mean = torch.tensor(c.latents_mean, device=device).view(1, c.z_dim, 1, 1, 1)
        self.inv_std = 1.0 / torch.tensor(c.latents_std, device=device).view(1, c.z_dim, 1, 1, 1)
        self._emb = {}

    @torch.no_grad()
    def embed(self, text: str) -> torch.Tensor:
        if text not in self._emb:
            e, _ = self.pipe.encode_prompt(prompt=text, negative_prompt=None,
                                           do_classifier_free_guidance=False, device=self.device)
            self._emb[text] = e.to(torch.bfloat16)
        return self._emb[text]

    def _vae_only(self):
        """The untiled fp32 VAE at 49x704x1056 needs most of the card: run it with the
        transformer on CPU. (Tiling is not an option — diffusers 0.35.1's tiled path skips the
        Wan2.2 VAE's pixel patchify: 'expected input ... to have 12 channels, but got 3'.)"""
        self.pipe.transformer.to("cpu")
        torch.cuda.empty_cache()

    def _transformer_on(self):
        self.pipe.transformer.to(self.device)

    def offload_text_encoder(self):
        self.pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()

    @torch.no_grad()
    def encode(self, frames: np.ndarray) -> torch.Tensor:
        """uint8 ``[T, H, W, 3]`` -> normalised latent ``[1, C, F, h, w]``."""
        self._vae_only()
        x = torch.from_numpy(frames).float().div(255).permute(0, 3, 1, 2)
        x = F.interpolate(x, size=HW, mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        x = (x * 2 - 1).permute(1, 0, 2, 3).unsqueeze(0).to(self.device)
        z = self.vae.encode(x).latent_dist.mode()
        return (z - self.mean) * self.inv_std

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        self._vae_only()
        v = self.vae.decode((z / self.inv_std + self.mean).float(), return_dict=False)[0]
        return ((v[0] + 1) / 2).clamp(0, 1).permute(1, 2, 3, 0)          # [T, H, W, 3]

    @torch.no_grad()
    def generate(self, cond: torch.Tensor, n_held: int, prompt: str, negative: str,
                 seed: int, steps: int, guidance: float, how: str = "mask") -> torch.Tensor:
        """WanImageToVideoPipeline's expand_timesteps loop. ``how="mask"``: the first ``n_held``
        latent frames are clean context at timestep 0. ``how="paste"``: only frame 0 is, and
        frames 1..n_held-1 are re-noised to each step's sigma and re-imposed (Euler steps)."""
        sched = self.pipe.scheduler
        sched.set_timesteps(steps, device=self.device)
        g = torch.Generator("cpu").manual_seed(seed)
        lat = torch.randn(cond.shape, generator=g).to(self.device)
        eps = torch.randn(cond.shape, generator=g).to(self.device)
        mask = torch.ones((1, 1) + tuple(cond.shape[2:]), device=self.device)
        mask[:, :, : (n_held if how == "mask" else 1)] = 0
        paste = slice(1, n_held) if how == "paste" else None
        sigmas = sched.sigmas.tolist()
        if paste:
            lat[:, :, paste] = (1 - sigmas[0]) * cond[:, :, paste] + sigmas[0] * eps[:, :, paste]
        pos, neg = self.embed(prompt), self.embed(negative)
        self._transformer_on()
        p = self.pipe.transformer.config.patch_size                       # (1, 2, 2)
        for i, t in enumerate(sched.timesteps):
            x = ((1 - mask) * cond + mask * lat).to(torch.bfloat16)
            ts = (mask[0, 0][:, ::p[1], ::p[2]] * t).flatten()[None]
            out = self.pipe.transformer(hidden_states=x, timestep=ts, encoder_hidden_states=pos,
                                        return_dict=False)[0]
            if guidance > 1:
                un = self.pipe.transformer(hidden_states=x, timestep=ts, encoder_hidden_states=neg,
                                           return_dict=False)[0]
                out = un + guidance * (out - un)
            if paste is None:
                lat = sched.step(out.float(), t, lat, return_dict=False)[0]
            else:
                # flow_prediction: v = eps - x0, x_t = (1 - s) x0 + s eps
                lat = lat + (sigmas[i + 1] - sigmas[i]) * out.float()
                s1 = sigmas[i + 1]
                lat[:, :, paste] = (1 - s1) * cond[:, :, paste] + s1 * eps[:, :, paste]
        return self.decode((1 - mask) * cond + mask * lat)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--repo", default="/workspace/models/Wan2.2-TI2V-5B-Diffusers")
    p.add_argument("--seeds", default="9500-9502")
    p.add_argument("--arm", default="honest")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--max-residual", type=float, default=0.02)
    p.add_argument("--skip-roundtrip", action="store_true")
    p.add_argument("--result-name", default="feasibility.json")
    p.add_argument("-o", "--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    wan = Wan(args.repo)
    seeds = parse_seeds(args.seeds)
    for v in args.variants:
        wan.embed(VARIANTS[v][2])
        for seed in seeds:
            lab = json.loads((args.dataset_dir / f"s{seed:05d}_{args.arm}" / "labels.json").read_text())
            wan.embed(scene_prompt(lab) if VARIANTS[v][1] == SCENE else VARIANTS[v][1])
    wan.offload_text_encoder()

    rows = []
    for seed in seeds:
        scene = args.dataset_dir / f"s{seed:05d}_{args.arm}"
        labels = json.loads((scene / "labels.json").read_text())
        real = load_frames(scene, 49)
        row = {"scene": scene.name, "e_label": physics.clip_restitution(labels),
               "real": pixel_restitution(real, labels, args.max_residual)}
        cond = wan.encode(real)
        if seed == seeds[0] and not args.skip_roundtrip:
            rt = to_native(wan.decode(cond))
            row["vae_roundtrip"] = pixel_restitution(rt, labels, args.max_residual)
        for v in args.variants:
            n_held, prompt, negative, how = VARIANTS[v]
            if prompt == SCENE:
                prompt = row["prompt"] = scene_prompt(labels)
            t = time.time()
            out = to_native(wan.generate(cond, n_held, prompt, negative, seed, args.steps, args.guidance, how))
            save_video([Image.fromarray(f) for f in out], args.out / f"{scene.name}_{v}.mp4", fps=8)
            row[v] = {**pixel_restitution(out, labels, args.max_residual), "seconds": time.time() - t}
        rows.append(row)
        print(json.dumps({"scene": row["scene"], **{
            m: {k: (round(x, 3) if isinstance(x, float) else x) for k, x in row[m].items()
                if k in ("e_pixels", "fit_residual", "measurable", "seconds")}
            for m in ["real", "vae_roundtrip"] + args.variants if m in row}}), flush=True)
        (args.out / args.result_name).write_text(json.dumps(
            {"protocol": "§67", "model": args.repo, "hw": HW, "steps": args.steps, "guidance": args.guidance,
             "variants": {v: VARIANTS[v] for v in args.variants}, "max_residual": args.max_residual,
             "rows": rows}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
