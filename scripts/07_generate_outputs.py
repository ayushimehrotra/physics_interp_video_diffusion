#!/usr/bin/env python3
"""Stage 7 — feasibility gate, then generate() continuations from pre-contact frames (spec §6).

    HF_HOME=/root/hf_cache HF_HUB_OFFLINE=1 /root/envs/physics_venv/bin/python scripts/07_generate_outputs.py \\
        --model configs/models/cogvideox_2b.yaml --data data/f1

1. Gate (configs/eval/outputs.yaml feasibility_gate): track the seed-0 generations that stage 4 already
   saved (cache/<model>/<family>/<group>/gen_seed0.mp4) for the first n_clips groups; measurable
   fraction must reach min_pass_rate. Written to cache/<model>/<family>/_gate.json.
2. Only if the gate passes (or --force): generate seeds 1..seeds_per_clip-1 for every group
   (seed 0 exists). The continuation depends only on the shared pre-contact context, so one video per
   (group, seed) serves both legal twins. Resumable.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from physlaw.config import load_config, resolve_path  # noqa: E402
from physlaw.eval import outputs as O  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--eval", default="configs/eval/outputs.yaml")
    ap.add_argument("--force", action="store_true", help="generate all seeds even if the gate fails")
    a = ap.parse_args()
    mcfg, ecfg = load_config(a.model), load_config(a.eval)
    data = resolve_path(a.data)
    cache = REPO / "cache" / mcfg.name / data.name
    geom = O.Geometry(mcfg)
    g = ecfg["feasibility_gate"]
    res = O.gate(cache, data, geom, int(g["n_clips"]), float(g["min_pass_rate"]))
    (cache / "_gate.json").write_text(json.dumps(res, indent=1, default=float))
    if not res["passed"] and not a.force:
        print(f"[07] gate FAILED ({res['pass_rate']:.0%} < {g['min_pass_rate']:.0%}): no further seeds generated "
              f"(configs/eval/outputs.yaml). Use --force to generate anyway.")
        return
    import imageio.v3 as iio
    import numpy as np
    from physlaw.data.io import load_clip
    from physlaw.models.registry import load_adapter
    m = load_adapter(mcfg)
    seeds = range(1, int(ecfg["seeds_per_clip"]))
    groups = sorted(p.parent.name for p in cache.glob("*/gen_seed0.mp4"))
    t0, n = time.time(), 0
    for gid in groups:
        todo = [s for s in seeds if not (cache / gid / f"gen_seed{s}.mp4").exists()]
        if not todo:
            continue
        c = load_clip(data / gid / "conserve_e1")
        fr, k = m.input_frames(c.labels), m.context_frames(c.labels)
        x = m.to_tensor(c.frames[fr])
        for s in todo:
            iio.imwrite(cache / gid / f"gen_seed{s}.mp4", m.generate(x, k, seed=s), fps=m.fps, codec="libx264", quality=9)
            n += 1
        print(f"[07] {gid}: seeds {todo}  ({(time.time() - t0) / max(n, 1):.0f} s/video)", flush=True)
    print(f"[07] generated {n} videos in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
