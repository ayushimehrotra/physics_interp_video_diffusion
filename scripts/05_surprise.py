#!/usr/bin/env python3
"""Stage 5 — pixel-level surprise baselines on a model's own frame sampling (spec §5 L2, configs/eval/l2).

    /root/envs/physics_venv/bin/python scripts/05_surprise.py --model configs/models/vjepa2_vitl.yaml --data data/f1

Model surprise itself is computed in stage 4 (it needs the loaded model; written into
cache/<model>/<family>/_index.jsonl). This stage adds the model-free baselines that L2 reports
beside every contrast, on exactly the frames and context split the model sees:

  copy_last_frame   predict every post-context input frame as the last context frame;
                    mse / l1 over all pixels and over the group's object region ("obj"): ONE pixel region per
                    twin group, the union over its twins of the scored (non-aux) objects' masks over the clean
                    window (physlaw.tokens.project.group_object_region), so every twin is scored on the same
                    pixels and the region does not reveal a twin's outcome
  constvel          extrapolate each scored (non-aux) object's projected centre at constant pixel velocity from
                    the last two context frames; mean centre error in pixels over in-frame post frames

Each record carries the whole-segment value (``surprise``) and the per-frame values (``per_frame``: t_s = master-frame
time of each post input frame, surprise and weight per metric), so L2 can restrict to each group's clean window
(spec §5.0.2; physlaw.eval.l2_surprise.window_surprise). Caches written before per_frame existed: rerun with --force.

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
from physlaw.eval.l2_surprise import clean_until_s  # noqa: E402
from physlaw.models.base import sample_frames  # noqa: E402
from physlaw.tokens.project import group_object_region, roi_object_ids  # noqa: E402


def model_frames(labels: dict, fps, n_frames, center, tubelet, first_tubelet) -> tuple[list[int], int]:
    """(master frames the model sees, number of context frames: whole token-frames before contact)."""
    sc, fpc = labels["scene_constants"], labels["physlaw"]["first_post_contact_frame"]
    fr = sample_frames(master_fps=sc["frame_rate"], n_master=sc["num_frames"], fps=fps, n_frames=n_frames,
                       contact_frame=fpc, center_on_contact=center)
    k = sum(f < fpc for f in fr)
    # the model's context: whole token-frames before contact (causal VAEs: first token-frame = 1 frame)
    return fr, (1 + ((k - 1) // tubelet) * tubelet) if first_tubelet else (k - k % tubelet)


def group_baselines(args) -> list[dict]:
    """Records of the group's clips still to do (``todo``); the "obj" region is computed over ALL its twins."""
    paths, todo, model = args
    labels = [load_clip(p, frames=False, masks=False).labels for p in paths]
    fr, _ = model_frames(labels[0], *model)
    ends = [clean_until_s(lab) for lab in labels]
    t_end = np.inf if any(e is None for e in ends) else min(ends)
    region = group_object_region(((load_clip(p, frames=False).masks, lab) for p, lab in zip(paths, labels)),
                                 fr, None, t_end, float(labels[0]["scene_constants"]["frame_rate"]))
    return [baselines(p, model, region) for p in paths if f"{Path(p).parent.name}/{Path(p).name}" in todo]


def baselines(path, model, region: np.ndarray) -> dict:
    c = load_clip(path, masks=False)
    sc, pl = c.labels["scene_constants"], c.pl
    fr, k = model_frames(c.labels, *model)
    last = c.frames[fr[k - 1]].astype(np.float32)
    fut = c.frames[fr[k:]].astype(np.float32)
    obj = np.broadcast_to(region, (len(fut), *region.shape))
    err = fut - last[None]
    # per post input frame (the clean window of spec §5.0.2 is applied at aggregation, physlaw.eval.l2_surprise)
    n_obj = obj.reshape(len(fut), -1).sum(1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        per = {"copy_last_mse_all": (err ** 2).mean((1, 2, 3)), "copy_last_l1_all": np.abs(err).mean((1, 2, 3)),
               "copy_last_mse_obj": np.array([(e[m] ** 2).mean() if m.any() else np.nan for e, m in zip(err, obj)]),
               "copy_last_l1_obj": np.array([np.abs(e[m]).mean() if m.any() else np.nan for e, m in zip(err, obj)])}
    weight = {"copy_last_mse_obj": n_obj, "copy_last_l1_obj": n_obj}
    pf = c.labels["per_frame"]
    scored = {sid for _, sid in roi_object_ids(c.labels)}
    objs = [o for i, (o, d) in enumerate(pl["objects"].items()) if int(d.get("segmentation_id", i + 1)) in scored]
    cv, cn = [], []
    for j in range(k, len(fr)):
        errs = []
        for o in objs:
            p1, p0 = np.array(pf[fr[k - 1]][o]["pixel"]), np.array(pf[fr[k - 2]][o]["pixel"])
            if pf[fr[j]][o]["in_frame"]:
                errs.append(np.linalg.norm(p1 + (p1 - p0) * (j - (k - 1)) - np.array(pf[fr[j]][o]["pixel"])))
        cv.append(float(np.mean(errs)) if errs else float("nan"))
        cn.append(float(len(errs)))
    per["constvel_px"], weight["constvel_px"] = np.array(cv), np.array(cn)
    s = {}
    for m, v in per.items():            # whole post segment (unrestricted): weighted mean over frames
        w = weight.get(m, np.ones(len(v)))
        ok = np.isfinite(v) & (w > 0)
        s[m] = float((w[ok] * v[ok]).sum() / w[ok].sum()) if ok.any() else float("nan")
    fps_master = float(sc["frame_rate"])
    per_frame = {"t_s": [f / fps_master for f in fr[k:]],
                 "surprise": {m: [None if not np.isfinite(x) else float(x) for x in v] for m, v in per.items()},
                 "weight": {m: w.tolist() for m, w in weight.items()}}
    return {"clip": f"{pl['twin_group_id']}/{pl['twin']}", "group": pl["twin_group_id"], "twin": pl["twin"],
            "split": pl["split"], "family": pl["family"], "surprise": s, "per_frame": per_frame, "frames": fr,
            "context_frames": k}


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
    all_clips = list_clips(data)
    todo = {f"{p.parent.name}/{p.name}" for p in all_clips} - done
    print(f"[05] model={cfg.name}; "
          f"{len(todo)} clips to do ({len(done)} done) -> {f}", flush=True)
    if "official" in cfg:      # diffusion: native settings + latent token-frames
        fps, n = int(cfg.official.fps), int(cfg.official.num_frames)
        tg = cfg.token_grid
        tub, first = int(tg.vae_temporal) * int(tg.get("patch_t") or 1), True
    else:
        fps, n = int(cfg.input.fps), int(cfg.input.num_frames)
        tub, first = int(cfg.token_grid.tubelet), False
    center = bool(cfg.get("input", {}).get("center_on_contact", False))
    groups = {}
    for p in all_clips:
        groups.setdefault(p.parent.name, []).append(str(p))
    jobs = [(ps, todo, (fps, n, center, tub, first)) for g, ps in groups.items()
            if any(f"{g}/{Path(q).name}" in todo for q in ps)]
    i = 0
    with ProcessPoolExecutor(a.workers) as ex, open(f, "a") as fh:
        for recs in ex.map(group_baselines, jobs):
            for rec in recs:
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                i += 1
                if i % 40 == 0:
                    print(f"[05] {i}/{len(todo)}", flush=True)
    print("[05] done", flush=True)


if __name__ == "__main__":
    main()
