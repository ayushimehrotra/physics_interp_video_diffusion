#!/usr/bin/env python3
"""Stage 7 - generate videos for output-level evaluation (spec §6), driven by configs/eval/outputs.yaml.

    # track A (§6.2): continue the REAL pre-event context through the adapter's generate()
    HF_HUB_OFFLINE=1 /root/envs/physics_venv/bin/python scripts/07_generate_outputs.py --track a \\
        --model configs/models/cogvideox_2b.yaml --data data/f1
    # track B (§6.3): frame 0 + the minimal prompt through the model's official diffusers I2V pipeline
    HF_HUB_OFFLINE=1 /root/envs/physics_venv/bin/python scripts/07_generate_outputs.py --track b \\
        --model configs/models/wan22_i2v_a14b.yaml --data data/f1

Track A, into cache/<model>/<family>/<group>/gen_seed<s>.mp4 (stage 4 already wrote seed 0 for DiTs):
  1. seed 0 for the first track_a.gate.n_clips groups, if missing (models without a stage-4 pass);
  2. feasibility gate on those clips (stage 8's readout); written to _gate.json;
  3. only if measurable >= track_a.gate.min_pass_rate (or --force): seeds 0..seeds_per_clip-1 for every group.
  The continuation depends only on the shared pre-event context, so one video per (group, seed) serves every twin.
Track B, into cache/<model>/<family>_trackB/<group>/gen_seed<s>.mp4 (+ _gen.jsonl): seeds track_b.seeds for the
  first track_b.n_groups groups. Frame 0 is identical across a group's twins. The pipeline class comes from the
  checkpoint's model_index.json, the call settings from the model config's ``official`` block (only the keys the
  pipeline accepts), the VAE dtype from ``vae_dtype``; the prompt is the family's minimal prompt.
Resumable: existing videos are skipped. --out writes elsewhere (test runs).
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from physlaw.config import load_config, resolve_path  # noqa: E402
from physlaw.eval import outputs as O  # noqa: E402


def write_mp4(path, frames, fps):
    import imageio.v3 as iio
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, frames, fps=fps, codec="libx264", quality=9)


def groups_of(data: Path) -> list[str]:
    return sorted(p.parent.name for p in data.glob("*/_GROUP_COMPLETE"))


def track_a(mcfg, data, out, cfg, force):
    from physlaw.data.io import load_clip
    geom = O.Geometry(mcfg)
    g = cfg["gate"]
    groups = groups_of(data)
    m = None

    def gen(gid, seeds):
        nonlocal m
        todo = [s for s in seeds if not (out / gid / f"gen_seed{s}.mp4").exists()]
        if not todo:
            return 0
        if m is None:
            from physlaw.models.registry import load_adapter
            m = load_adapter(mcfg)
        c = load_clip(data / gid / O.LEGAL[0])
        fr, k = m.input_frames(c.labels), m.context_frames(c.labels)
        if hasattr(m, "set_prompt") and cfg.get("prompt") == "minimal":
            from physlaw.models.cogvideox import minimal_prompt
            m.set_prompt(minimal_prompt(c.labels))
        x = m.to_tensor(c.frames[fr])
        for s in todo:
            write_mp4(out / gid / f"gen_seed{s}.mp4", m.generate(x, k, seed=s), m.fps)
        return len(todo)

    t0, n = time.time(), 0
    for gid in groups[: int(g["n_clips"])]:
        n += gen(gid, [0])
    res = O.gate(out, data, geom, int(g["n_clips"]), float(g["min_pass_rate"]))
    (out / "_gate.json").write_text(json.dumps(res, indent=1, default=float))
    if not res["passed"] and not force:
        print(f"[07] gate FAILED ({res['pass_rate']:.0%} < {g['min_pass_rate']:.0%}): no further seeds "
              f"(configs/eval/outputs.yaml track_a.gate). --force generates anyway.")
        return
    for gid in groups:
        k = gen(gid, range(int(cfg["seeds_per_clip"])))
        n += k
        if k:
            print(f"[07] {gid}: {k} videos ({(time.time() - t0) / max(n, 1):.0f} s/video)", flush=True)
    print(f"[07] track A: generated {n} videos in {(time.time() - t0) / 60:.1f} min")


def load_i2v_pipeline(mcfg):
    """The checkpoint's own diffusers pipeline (class from model_index.json), components straight to the GPU."""
    import diffusers
    import torch
    path = os.environ.get("PHYSLAW_I2V_PATH") or mcfg["local_path"]
    idx = json.loads((Path(path) / "model_index.json").read_text())
    kw = {}
    if mcfg.get("vae_dtype"):                       # e.g. Wan: transformers bf16, VAE fp32 (model card)
        vae_cls = getattr(diffusers, idx["vae"][1])
        kw["vae"] = vae_cls.from_pretrained(path, subfolder="vae", torch_dtype=getattr(torch, mcfg["vae_dtype"]))
    pipe = getattr(diffusers, idx["_class_name"]).from_pretrained(
        path, torch_dtype=getattr(torch, mcfg.get("dtype", "bfloat16")), **kw).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe, path


def generate_i2v(pipe, mcfg, frame0, prompt, seed):
    """One official I2V generation from frame 0 -> uint8 [T, H, W, 3]. Size: ``max_area`` at the frame's aspect,
    rounded to the VAE x patch multiple (the Wan card's rule), else the official height x width."""
    import numpy as np
    import torch
    from PIL import Image
    off = dict(mcfg["official"])
    if "max_area" in off:
        mod = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
        ar = frame0.shape[0] / frame0.shape[1]
        off["height"] = round(np.sqrt(off["max_area"] * ar)) // mod * mod
        off["width"] = round(np.sqrt(off["max_area"] / ar)) // mod * mod
    img = Image.fromarray(frame0)
    if img.size != (off["width"], off["height"]):
        img = img.resize((off["width"], off["height"]), Image.BICUBIC)
    accepted = set(inspect.signature(pipe.__call__).parameters)
    kw = {k: v for k, v in off.items() if k in accepted}
    vid = pipe(image=img, prompt=prompt, output_type="np", generator=torch.Generator("cuda").manual_seed(seed),
               **kw).frames[0]
    return (np.clip(vid, 0, 1) * 255).round().astype(np.uint8), (off["height"], off["width"])


def track_b(mcfg, data, out, cfg, seeds, n_groups):
    from physlaw.data.io import load_clip
    from physlaw.models.cogvideox import minimal_prompt
    jobs = [(g, s) for s in seeds for g in groups_of(data)[:n_groups] if not (out / g / f"gen_seed{s}.mp4").exists()]
    if not jobs:
        print("[07] track B: nothing to do")
        return
    t0 = time.time()
    pipe, path = load_i2v_pipeline(mcfg)
    print(f"[07] track B: {mcfg.name} loaded from {path} in {time.time() - t0:.0f}s; {len(jobs)} clips", flush=True)
    for gid, seed in jobs:
        c = load_clip(data / gid / O.LEGAL[0], masks=False)
        prompt = minimal_prompt(c.labels)
        t1 = time.time()
        u8, hw = generate_i2v(pipe, mcfg, c.frames[0], prompt, seed)
        write_mp4(out / gid / f"gen_seed{seed}.mp4", u8, mcfg["official"]["fps"])
        rec = {"group": gid, "seed": seed, "prompt": prompt, "size": list(hw), "seconds": round(time.time() - t1, 1),
               "model_path": path, "revision": mcfg.get("revision")}
        with open(out / "_gen.jsonl", "a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[07] {gid} seed{seed} {rec['seconds']}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--track", choices=["a", "b"], required=True)
    ap.add_argument("--eval", default="configs/eval/outputs.yaml")
    ap.add_argument("--out", default=None, help="default cache/<model>/<family>[_trackB]")
    ap.add_argument("--seeds", default=None, help="track B: comma list or lo-hi (default: track_b.seeds)")
    ap.add_argument("--n", type=int, default=None, help="track B: first n groups (default: track_b.n_groups)")
    ap.add_argument("--force", action="store_true", help="track A: generate every seed even if the gate fails")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    mcfg, ecfg = load_config(a.model), load_config(a.eval, a.set)
    data = resolve_path(a.data)
    if a.track == "a":
        out = Path(a.out) if a.out else REPO / "cache" / mcfg.name / data.name
        out.mkdir(parents=True, exist_ok=True)
        track_a(mcfg, data, out, ecfg["track_a"], a.force)
    else:
        cfg = ecfg["track_b"]
        out = Path(a.out) if a.out else REPO / "cache" / mcfg.name / f"{data.name}_trackB"
        out.mkdir(parents=True, exist_ok=True)
        if a.seeds and "-" in a.seeds:
            lo, hi = (int(x) for x in a.seeds.split("-"))
            seeds = list(range(lo, hi + 1))
        else:
            seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else list(cfg["seeds"])
        track_b(mcfg, data, out, cfg, seeds, a.n or int(cfg["n_groups"]))


if __name__ == "__main__":
    main()
