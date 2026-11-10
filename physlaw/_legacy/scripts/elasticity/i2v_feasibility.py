#!/usr/bin/env python3
"""§65 step 1 — can CogVideoX1.5-5B-I2V produce a measurable two-ball collision at all?

    python scripts/elasticity/i2v_feasibility.py --dataset-dir datasets/energy_ladder_v1 \
        --seeds 9500-9509 --arm honest -o outputs/i2v_feasibility

Per scene, three clips, each read back with the pixel tracker (video.measure_momentum):

* ``real``  — the rendered clip itself (frames 0-44). The tracker's reference: realised e
  from pixels against restitution_realised.
* ``free``  — I2V from frame 0 alone. The model invents the motion; does a collision
  happen, at roughly frame 20, with both balls trackable?
* ``hold``  — latent frames 0-5 (pixel 0-20, strictly pre-contact) held to the real clip,
  6-11 regenerated (runner.inpaint). The approach is given; the aftermath, and so e, is
  the model's. This is the setting the steering step uses.

Realised e from pixels uses the tracked pre and post velocities and the scene's contact
normal: e = -(v_A' - v_B').n / (v_A - v_B).n. A clip counts as measurable when all four
straight-line fits have residual <= --max-residual (world units; rendered clips sit ~0.002).
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))  # noqa: E401,E702
import _momentum_path  # noqa: F401

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from physics_interp.config import GenerationConfig, InversionConfig, ModelConfig
from physics_interp.models import load_model
from physics_interp.momentum import physics
from physics_interp.runner import VideoDiffusionRunner
from PIL import Image

from physics_interp.video import load_frames, measure_momentum, save_video

PROMPT = ("Top-down view of two balls on a flat grey floor under soft studio light. One ball "
          "rolls across the floor and collides with the other ball, and after the collision both "
          "balls roll apart in straight lines. Static camera, realistic physics.")
NUM_FRAMES = 45
HOLD_LATENTS = range(0, 6)     # pixel frames 0-20
STRICT_PROMPT = ("Top-down view of exactly two balls, one red and one grey metal, on a plain flat grey "
                 "floor. The red ball and the grey ball collide once and then roll apart in straight "
                 "lines. Only two balls are ever visible. Fixed camera, no zoom, no camera motion.")
NEGATIVE = ("extra balls, third ball, duplicate objects, new objects appearing, objects disappearing, "
            "color change, camera motion, zoom, blur, distortion")
# --variants: (name, prompt, negative prompt, guidance), all in hold mode.
VARIANTS = {
    "default_g6": (PROMPT, "", 6.0),
    "strict_g6": (STRICT_PROMPT, NEGATIVE, 6.0),
    "strict_g3": (STRICT_PROMPT, NEGATIVE, 3.0),
    "empty_g1": ("", "", 1.0),
}


def to_native(video: torch.Tensor, hw=(480, 720)) -> np.ndarray:
    """Decoded ``[T, h, w, 3]`` in [0, 1] -> uint8 at the renderer's resolution."""
    x = F.interpolate(video.permute(0, 3, 1, 2).float(), size=hw, mode="area")
    return (x.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)


def pixel_restitution(frames: np.ndarray, labels: dict, max_residual: float) -> dict:
    pre = physics.pre_state_from_labels(labels)
    contact = int(labels["collision"]["actual_frame_rounded"])
    m = measure_momentum(frames, [pre.m_a, pre.m_b], contact_frame=contact,
                         post_window=range(contact + 3, len(frames)))
    v0, v1 = m["velocity_pre"], m["velocity_post"]
    approach = float(np.dot(v0[0] - v0[1], pre.n))
    e = float(-np.dot(v1[0] - v1[1], pre.n) / approach) if abs(approach) > 1e-6 else float("nan")
    ok = bool(np.isfinite(e) and np.isfinite(m["fit_residual"]) and m["fit_residual"] <= max_residual)
    return {"e_pixels": e, "fit_residual": m["fit_residual"], "measurable": ok,
            "frames_tracked": m["frames_tracked"], "approach_speed_pixels": approach,
            "approach_speed_label": pre.approach_speed,
            "velocity_pre": v0.tolist(), "velocity_post": v1.tolist()}


def parse_seeds(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--seeds", default="9500-9509")
    p.add_argument("--arm", default="honest")
    p.add_argument("--modes", nargs="+", default=["real", "free", "hold"])
    p.add_argument("--free-limit", type=int, default=5, help="free I2V on only the first N scenes")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=6.0)
    p.add_argument("--max-residual", type=float, default=0.02)
    p.add_argument("--variants", nargs="+", default=[], choices=sorted(VARIANTS),
                   help="hold mode under each named prompt/guidance variant")
    p.add_argument("-o", "--out", type=Path, required=True)
    p.add_argument("--result-name", default="feasibility.json")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    adapter = load_model(ModelConfig(id="cogvideox1.5-5b-i2v"), "cuda")
    gen = GenerationConfig(num_frames=NUM_FRAMES, height=768, width=1152,
                           num_inference_steps=args.steps, guidance_scale=args.guidance)
    runner = VideoDiffusionRunner(adapter, gen)
    inv = InversionConfig(num_steps=args.steps, prompt=PROMPT)

    rows = []
    for k, seed in enumerate(parse_seeds(args.seeds)):
        scene = args.dataset_dir / f"s{seed:05d}_{args.arm}"
        labels = json.loads((scene / "labels.json").read_text())
        real = load_frames(scene, NUM_FRAMES, prefix=True)
        video = torch.from_numpy(real).float() / 255.0
        row = {"scene": scene.name, "e_label": physics.clip_restitution(labels)}
        z0 = adapter.encode_video(video)                      # also sets the image conditioning

        if "real" in args.modes:
            row["real"] = pixel_restitution(real, labels, args.max_residual)
        if "free" in args.modes and k < args.free_limit:
            t = time.time()
            # Latent 0 is frame 0 itself, which the image conditioning already fixes.
            out = to_native(runner.inpaint(z0, [0], inversion=inv, seed=seed, prompt=PROMPT,
                                           guidance_scale=args.guidance))
            save_video([Image.fromarray(f) for f in out], args.out / f"{scene.name}_free.mp4", fps=8)
            row["free"] = {**pixel_restitution(out, labels, args.max_residual), "seconds": time.time() - t}
        if "hold" in args.modes:
            t = time.time()
            out = to_native(runner.inpaint(z0, list(HOLD_LATENTS), inversion=inv, seed=seed,
                                           prompt=PROMPT, guidance_scale=args.guidance))
            save_video([Image.fromarray(f) for f in out], args.out / f"{scene.name}_hold.mp4", fps=8)
            row["hold"] = {**pixel_restitution(out, labels, args.max_residual), "seconds": time.time() - t}
        for name in args.variants:
            text, neg, g = VARIANTS[name]
            t = time.time()
            out = to_native(runner.inpaint(z0, list(HOLD_LATENTS), inversion=inv, seed=seed, prompt=text,
                                           negative_prompt=neg, guidance_scale=g))
            save_video([Image.fromarray(f) for f in out], args.out / f"{scene.name}_hold_{name}.mp4", fps=8)
            row[f"hold_{name}"] = {**pixel_restitution(out, labels, args.max_residual), "seconds": time.time() - t}
        rows.append(row)
        shown = ["real", "free", "hold"] + [f"hold_{n}" for n in args.variants]
        print(json.dumps({"scene": row["scene"], "e_label": round(row["e_label"], 3),
                          **{m: {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in row[m].items()
                                 if kk in ("e_pixels", "fit_residual", "measurable", "seconds")}
                             for m in shown if m in row}}), flush=True)
        (args.out / args.result_name).write_text(json.dumps(
            {"protocol": "§65 step 1", "prompt": PROMPT, "variants": {n: VARIANTS[n] for n in args.variants}, "num_frames": NUM_FRAMES,
             "steps": args.steps, "guidance": args.guidance, "max_residual": args.max_residual,
             "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
