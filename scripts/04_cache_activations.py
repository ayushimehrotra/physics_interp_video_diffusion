#!/usr/bin/env python3
"""Stage 4 — cache model activations, forecasts and surprise for a family (spec §4, §5).

    HF_HOME=/root/hf_cache HF_HUB_OFFLINE=1 /root/envs/physics_venv/bin/python \\
        scripts/04_cache_activations.py --model configs/models/vjepa2_vitl.yaml --data data/f1

Writes under cache/<model>/<family>/:
  <group>/<twin>.npz     pooled encode() activations, float16 [n_pools, n_layers, D]
                         (+ names of pools and layers), and surprise() for that twin
  <group>/_forecast.npz  forecast() FINAL tokens, float16 [T_post_tok, gh, gw, D]: identical for
                         every twin of a group (shared pre-contact frames), so stored once
  _index.jsonl           one line per clip: ids, split, twin, surprise, file
Diffusion models (kind: diffusion): no _forecast.npz; instead each twin's .npz carries
  fc_pooled [P, K, D] the sampled forecast's activations (keys fc_keys, e.g. "snr0/L15") pooled with
                      that twin's weights over the forecast token-frames, and gen_seed0.mp4 is saved.

Pooled rather than full-token activations: 13 layers x 8192 tokens x 1024 x 2 B = 218 MB
per clip would be ~70 GB for 320 clips. Pools are ROI_A/ROI_B/ROI_pair/frame x pre/post/all
(physlaw/tokens). Full tokens for chosen layers: --save-tokens LAYER ... (for L4 later).
Resumable: clips with an existing .npz are skipped unless --force.
Diffusion, two passes: --no-forecast caches encode() + surprise() only (L0, L2). A later run without it
adds fc_pooled to every .npz that lacks it (one sampled forecast per group; encode/surprise and the
index are not recomputed).
Diffusion, noised-input-latent baseline (spec §5.1): each twin's .npz also carries
  pooled_latent [P, S, C'] the noised VAE latent encode() fed the model at each SNR (keys latent_keys, e.g.
                      "latent/snr+8"), one C' = patch_t*C*patch^2 vector per token, pooled like `pooled`.
  Any run adds it to an existing .npz that lacks it (VAE encode only, no transformer pass, nothing else recomputed).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from physlaw.config import config_hash, load_config, resolve_path  # noqa: E402
from physlaw.data.io import list_clips, load_clip  # noqa: E402
from physlaw.models.registry import load_adapter  # noqa: E402
from physlaw.tokens.pool import pool_weights  # noqa: E402
from physlaw.tokens.project import project_masks, token_frame_phase  # noqa: E402


def pool_latent(lat, Wt):
    """{"latent/snr+8": [T, gh, gw, C']} -> (pooled_latent [P, S, C'] float16, latent_keys)."""
    import torch
    lk = sorted(lat, key=lambda q: int(q.split("snr")[1]))
    return (torch.stack([Wt @ lat[q].reshape(-1, lat[q].shape[-1]).float() for q in lk], 1).half().cpu().numpy(),
            np.array(lk))


def append_npz(f_npz, **new):
    z = dict(np.load(f_npz))
    np.savez(f_npz, **z, **new)


def add_forecast(m, x, k, gid, out, Wt, names, phases, kt, fc_memo):
    """Diffusion: forecast = activations captured while sampling the continuation (seed 0); identical for
    every twin of the group (shared context latents), so sampled once per group. Pooled with THIS twin's
    weights over the forecast's token-frames; returns the arrays to append to the twin's .npz; the generated
    video is kept."""
    import torch
    if fc_memo.get("gid") != gid:
        acts_f, vid = m.forecast(x, k, seed=0, return_video=True)
        fc_memo.clear(); fc_memo.update(gid=gid, acts=acts_f)
        import imageio.v3 as iio
        iio.imwrite(out / gid / "gen_seed0.mp4", vid, fps=m.fps, codec="libx264", quality=9)
    fk = sorted(fc_memo["acts"])
    Wf = Wt.reshape(len(names), len(phases), -1)[:, kt:].reshape(len(names), -1)
    Wf = Wf / Wf.sum(1, keepdim=True).clamp_min(1e-12)
    fc_pooled = torch.stack([Wf @ fc_memo["acts"][q].reshape(-1, fc_memo["acts"][q].shape[-1]).float()
                             for q in fk], 1)
    return {"fc_pooled": fc_pooled.half().cpu().numpy(), "fc_keys": np.array(fk), "context_tokens": kt}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None, help="default cache/<model>/<family>")
    ap.add_argument("--limit", type=int, default=None, help="first N groups only")
    ap.add_argument("--save-tokens", type=int, nargs="*", default=[])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-forecast", action="store_true",
                    help="diffusion: skip the sampled forecast (encode + surprise only); a later run adds it")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()

    import torch
    cfg = load_config(a.model, a.set)
    data = resolve_path(a.data)
    clips = list_clips(data)
    groups = sorted({p.parent.name for p in clips})[: a.limit]
    clips = [p for p in clips if p.parent.name in groups]
    out = Path(a.out) if a.out else REPO / "cache" / cfg.name / data.name
    out.mkdir(parents=True, exist_ok=True)
    print(f"[04] model={cfg.name} clips={len(clips)} groups={len(groups)} out={out} hash={config_hash(cfg)}", flush=True)
    m = load_adapter(cfg)
    from physlaw.models.vjepa2 import FINAL
    t0, done = time.time(), 0
    diffusion = cfg.get("kind") == "diffusion"
    fc_memo: dict = {}          # diffusion: one sampled forecast per group, pooled per twin
    with open(out / "_index.jsonl", "a") as index:
        for path in clips:
            gid, twin = path.parent.name, path.name
            f_npz = out / gid / f"{twin}.npz"
            todo = None                                  # existing file: the parts to add (pass 2 / latent baseline)
            if f_npz.exists() and not a.force:
                have = np.load(f_npz).files if diffusion else ()
                todo = [part for part, need in (("latent", diffusion and "pooled_latent" not in have),
                                                ("forecast", diffusion and not a.no_forecast and "fc_pooled" not in have))
                        if need]
                if not todo:
                    continue
            f_npz.parent.mkdir(parents=True, exist_ok=True)
            c = load_clip(path)
            fr = m.input_frames(c.labels)
            k = m.context_frames(c.labels)
            x = m.to_tensor(c.frames[fr])
            obj = project_masks(c.masks, fr, m.token_grid)
            phases = token_frame_phase(fr, m.token_grid, c.pl["first_post_contact_frame"])
            W = pool_weights(obj, phases)
            names = sorted(W)
            Wt = torch.from_numpy(np.stack([W[n].reshape(-1) for n in names])).to(x.device)   # [P, N]
            kt = m.n_context_tokens(k) if hasattr(m, "n_context_tokens") else k // m.token_grid.tubelet
            if todo is not None:
                new = {}
                if "latent" in todo:
                    new.update(zip(("pooled_latent", "latent_keys"), pool_latent(m.latent_baseline(x), Wt)))
                if "forecast" in todo:
                    new.update(add_forecast(m, x, k, gid, out, Wt, names, phases, kt, fc_memo))
                append_npz(f_npz, **new)
                done += 1
                continue
            acts = m.encode(x)
            lat = {q: acts.pop(q) for q in [q for q in acts if isinstance(q, str) and q.startswith("latent/")]}
            layers = sorted(acts)
            # layers sharing the dominant width go into `pooled` [P, L, D]; others (VideoMAE's 512-d decoder
            # FINAL) are stored apart as pooled_final / pooled_<layer>
            dims = [acts[l].shape[-1] for l in layers]
            D = max(set(dims), key=dims.count)
            main = [l for l in layers if acts[l].shape[-1] == D]
            pool_l = lambda l: Wt @ acts[l].reshape(-1, acts[l].shape[-1]).float()
            pooled = torch.stack([pool_l(l) for l in main], 1)
            side = {("pooled_final" if l == FINAL else f"pooled_{l}"): pool_l(l).half().cpu().numpy()
                    for l in layers if l not in main}
            if lat:
                side.update(zip(("pooled_latent", "latent_keys"), pool_latent(lat, Wt)))
            layers = main
            roi_post = (obj["A"].roi | obj["B"].roi)[kt:]
            sur = m.surprise(x, k, roi_post)
            extra = {f"tokens_layer{l}": acts[l].half().cpu().numpy() for l in a.save_tokens if l in acts}
            if diffusion and not a.no_forecast:
                side.update(add_forecast(m, x, k, gid, out, Wt, names, phases, kt, fc_memo))
            np.savez(f_npz, pooled=pooled.half().cpu().numpy(), pools=np.array(names), layers=np.array(layers),
                     frames=np.array(fr), context_frames=k, phases=np.array(phases), **side, **extra)
            if not diffusion:
                fc = out / gid / "_forecast.npz"
                if not fc.exists() or a.force:
                    np.savez(fc, final=m.forecast(x, k)[FINAL].half().cpu().numpy(), context_frames=k)
            rec = {"clip": f"{gid}/{twin}", "group": gid, "twin": twin, "split": c.pl["split"],
                   "family": c.pl["family"], "surprise": sur, "file": str(f_npz.relative_to(out)),
                   "model": cfg.name, "config_hash": config_hash(cfg)}
            index.write(json.dumps(rec) + "\n"); index.flush()
            done += 1
            if done % 20 == 0:
                print(f"[04] {done}/{len(clips)}  {(time.time() - t0) / done:.2f} s/clip", flush=True)
    print(f"[04] finished {done} clips in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
