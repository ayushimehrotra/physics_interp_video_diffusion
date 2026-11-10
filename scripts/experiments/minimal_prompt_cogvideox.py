#!/usr/bin/env python3
"""Milestone 7 output gate with the minimal prompt (2026-11-05), seed 0, iterate-and-look.

Same generation as stage 4 (official settings; context latents replaced by the real pre-contact frames,
the model generates the continuation) and the same readout as the stage-7 gate (eval/outputs.measure_clip,
read back from the saved mp4), but the prompt is models.cogvideox.minimal_prompt(labels) per clip instead
of the config's fixed sentence. Writes to cache/<model>/<family>_minimal/ so it never mixes with the
fixed-prompt cache: <group>/gen_seed<s>.mp4, <group>/sheet_seed<s>.png, _gate_minimal.jsonl.

    HF_HUB_OFFLINE=1 /root/envs/physics_venv/bin/python scripts/experiments/minimal_prompt_cogvideox.py \\
        --model configs/models/cogvideox_2b.yaml --data data/f1 --n 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from physlaw.config import load_config, resolve_path  # noqa: E402


def sheet(frames, k, path, title, n=12):
    import numpy as np
    from PIL import Image, ImageDraw
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    tw = 360; th = round(tw * frames.shape[1] / frames.shape[2])
    im = Image.new("RGB", (tw * 4, th * ((n + 3) // 4) + 18))
    ImageDraw.Draw(im).text((4, 3), title[:220], fill=(255, 255, 255))
    for j, i in enumerate(idx):
        t = Image.fromarray(frames[i]).resize((tw, th))
        ImageDraw.Draw(t).text((5, 5), f"f{i}{' ctx' if i < k else ''}", fill=(255, 255, 0) if i >= k else (0, 255, 255))
        im.paste(t, ((j % 4) * tw, 18 + (j // 4) * th))
    im.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=5, help="first n groups (sorted), as the stage-7 gate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="replace", choices=["replace", "guided", "hold"])
    ap.add_argument("--guide", default="0:0", help="guided mode: comma list of w:tmin configs, e.g. 5:300,2:0")
    ap.add_argument("--hold-strength", default="1.0", help="comma list; each value is its own output dir")
    ap.add_argument("--groups", default=None, help="comma list of group ids (overrides --n)")
    a = ap.parse_args()
    import imageio.v3 as iio
    import numpy as np
    from physlaw.data.io import load_clip
    from physlaw.eval import outputs as O
    from physlaw.models.cogvideox import minimal_prompt
    from physlaw.models.registry import load_adapter
    mcfg = load_config(a.model)
    data = resolve_path(a.data)
    if a.mode == "hold":
        strengths = [float(v) for v in str(a.hold_strength).split(",")]
    elif a.mode == "guided":
        strengths = [tuple(float(x) for x in v.split(":")) for v in a.guide.split(",")]
    else:
        strengths = [1.0]
    def out_of(hs):
        tag = {"replace": "", "guided": (f"_guided_w{hs[0]:g}_t{hs[1]:g}" if a.mode == "guided" else ""),
               "hold": (f"_hold_s{hs:g}" if a.mode == "hold" else "")}[a.mode]
        return tag, REPO / "cache" / mcfg.name / f"{data.name}_minimal{tag}"
    def done_in(o):
        lg = o / "_gate_minimal.jsonl"
        return {(json.loads(l)["group"], json.loads(l)["seed"]) for l in lg.read_text().splitlines()} if lg.exists() else set()
    done = set.intersection(*[done_in(out_of(hs)[1]) for hs in strengths])
    groups = (a.groups.split(",") if a.groups else
              sorted(p.name for p in data.iterdir() if (p / "_GROUP_COMPLETE").exists())[: a.n])
    groups = [g for g in groups if (g, a.seed) not in done]
    if not groups:
        return
    t0 = time.time()
    m = load_adapter(mcfg)
    print(f"[minimal] {mcfg.name} loaded in {time.time() - t0:.0f}s; {len(groups)} groups", flush=True)
    jobs = [(gid, hs) for gid in groups for hs in strengths if (gid, a.seed) not in done_in(out_of(hs)[1])]
    for gid, hs in jobs:
        tag, out = out_of(hs)
        out.mkdir(parents=True, exist_ok=True)
        log = out / "_gate_minimal.jsonl"
        c = load_clip(data / gid / "conserve_e1")
        fr, k = m.input_frames(c.labels), m.context_frames(c.labels)
        prompt = minimal_prompt(c.labels)
        m.set_prompt(prompt)
        t1 = time.time()
        kw = ({"guide_w": hs[0], "guide_tmin": int(hs[1])} if a.mode == "guided" else
              {"hold_strength": hs} if a.mode == "hold" else {})
        vid = m.generate(m.to_tensor(c.frames[fr]), k, seed=a.seed, mode=a.mode, **kw)
        (out / gid).mkdir(exist_ok=True)
        mp4 = out / gid / f"gen_seed{a.seed}.mp4"
        iio.imwrite(mp4, vid, fps=m.fps, codec="libx264", quality=9)
        gen = O._read_mp4(mp4)
        r = O.measure_clip(gen, c, fr, k, m.fps, m.H, m.W)
        r = {kk: v for kk, v in r.items() if kk != "post_frames"}
        gl = getattr(m, "guide_log", [])
        rec = {"group": gid, "seed": a.seed, "mode": a.mode, "guide": list(hs) if a.mode == "guided" else None,
               "hold_strength": hs if a.mode == "hold" else None,
               "guide_dv_over_v_median": float(np.median([g["dv_over_v"] for g in gl])) if gl else None,
               "guide_ctx_loss_last": gl[-1]["ctx_loss"] if gl else None, "prompt": prompt, "context_frames": k, "n_frames": len(fr),
               "seconds": round(time.time() - t1, 1), **r}
        sheet(gen, k, out / gid / f"sheet_seed{a.seed}.png",
              f"{mcfg.name} {a.mode}{tag} {gid} seed{a.seed} | context f0-f{k - 1} real | measurable={r.get('measurable')} "
              f"reason={r.get('reason')}")
        with open(log, "a") as fh:
            fh.write(json.dumps(rec, default=float) + "\n")
        print(f"[minimal] {gid} {rec['seconds']}s measurable={r.get('measurable')} reason={r.get('reason')}", flush=True)


if __name__ == "__main__":
    main()
