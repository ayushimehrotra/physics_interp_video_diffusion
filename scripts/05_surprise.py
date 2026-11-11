#!/usr/bin/env python3
"""Stage 5 — pixel-level surprise baselines on a model's own frame sampling (spec §5 L2, configs/eval/l2).

    /root/envs/physics_venv/bin/python scripts/05_surprise.py --model configs/models/vjepa2_vitl.yaml --data data/f1

Model surprise itself is computed in stage 4 (it needs the loaded model; written into
cache/<model>/<family>/_index.jsonl). This stage adds the model-free baselines that L2 reports
beside every contrast, on exactly the frames and context split the model sees:

  copy_last_frame   predict every post-context input frame as the last context frame;
                    mse / l1 over all pixels and over the balls' actual masks ("obj")
  constvel          extrapolate each ball's projected centre at constant pixel velocity from the
                    last two context frames; mean centre error in pixels over in-frame post frames

Writes cache/<model>/<family>/_pixel_baselines.jsonl. CPU only. Resumable unless --force.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from physlaw.config import load_config, resolve_path  # noqa: E402
from physlaw.data.io import list_clips, load_clip  # noqa: E402
from physlaw.models.base import sample_frames  # noqa: E402


def baselines(args) -> dict:
    path, fps, n_frames, center, tubelet, first_tubelet = args
    c = load_clip(path)
    sc, pl = c.labels["scene_constants"], c.pl
    fpc = pl["first_post_contact_frame"]
    fr = sample_frames(master_fps=sc["frame_rate"], n_master=sc["num_frames"], fps=fps, n_frames=n_frames,
                       contact_frame=fpc, center_on_contact=center)
    k = sum(f < fpc for f in fr)
    # the model's context: whole token-frames before contact (causal VAEs: first token-frame = 1 frame)
    k = (1 + ((k - 1) // tubelet) * tubelet) if first_tubelet else (k - k % tubelet)
    last = c.frames[fr[k - 1]].astype(np.float32)
    fut = c.frames[fr[k:]].astype(np.float32)
    obj = c.masks[fr[k:]] > 0
    err = fut - last[None]
    s = {"copy_last_mse_all": float((err ** 2).mean()), "copy_last_l1_all": float(np.abs(err).mean()),
         "copy_last_mse_obj": float((err[obj] ** 2).mean()), "copy_last_l1_obj": float(np.abs(err[obj]).mean())}
    pf = c.labels["per_frame"]
    errs = []
    for o in ("A", "B"):
        p1, p0 = np.array(pf[fr[k - 1]][o]["pixel"]), np.array(pf[fr[k - 2]][o]["pixel"])
        v = p1 - p0
        for j in range(k, len(fr)):
            if pf[fr[j]][o]["in_frame"]:
                errs.append(np.linalg.norm(p1 + v * (j - (k - 1)) - np.array(pf[fr[j]][o]["pixel"])))
    s["constvel_px"] = float(np.mean(errs)) if errs else float("nan")
    return {"clip": f"{pl['twin_group_id']}/{pl['twin']}", "group": pl["twin_group_id"], "twin": pl["twin"],
            "split": pl["split"], "family": pl["family"], "surprise": s, "frames": fr, "context_frames": k}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cache", default=None, help="default cache/<model>/<family> (stage 4's --out)")
    a = ap.parse_args()
    cfg = load_config(a.model)
    data = resolve_path(a.data)
    out = Path(a.cache) if a.cache else REPO / "cache" / cfg.name / data.name
    out.mkdir(parents=True, exist_ok=True)
    f = out / "_pixel_baselines.jsonl"
    done = set()
    if f.exists() and not a.force:
        done = {json.loads(l)["clip"] for l in open(f)}
    elif f.exists():
        f.unlink()
    clips = [p for p in list_clips(data) if f"{p.parent.name}/{p.name}" not in done]
    print(f"[05] model={cfg.name}; "
          f"{len(clips)} clips to do ({len(done)} done) -> {f}", flush=True)
    if "official" in cfg:      # diffusion: native settings + latent token-frames
        fps, n = int(cfg.official.fps), int(cfg.official.num_frames)
        tg = cfg.token_grid
        tub, first = int(tg.vae_temporal) * int(tg.get("patch_t") or 1), True
    else:
        fps, n = int(cfg.input.fps), int(cfg.input.num_frames)
        tub, first = int(cfg.token_grid.tubelet), False
    center = bool(cfg.get("input", {}).get("center_on_contact", False))
    jobs = [(str(p), fps, n, center, tub, first) for p in clips]
    with ProcessPoolExecutor(a.workers) as ex, open(f, "a") as fh:
        for i, rec in enumerate(ex.map(baselines, jobs, chunksize=2)):
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            if (i + 1) % 40 == 0:
                print(f"[05] {i + 1}/{len(jobs)}", flush=True)
    print("[05] done", flush=True)


if __name__ == "__main__":
    main()
