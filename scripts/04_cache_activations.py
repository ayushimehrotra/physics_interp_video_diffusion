#!/usr/bin/env python3
"""Stage 4 — cache model activations, forecasts and surprise for a family (spec §4, §5).

    HF_HOME=/root/hf_cache HF_HUB_OFFLINE=1 /root/envs/physics_venv/bin/python \\
        scripts/04_cache_activations.py --model configs/models/vjepa2_vitl.yaml --data data/f1

Writes under cache/<model>/<family>/:
  <group>/<twin>.npz     pooled encode() activations, float16 [n_pools, n_layers, D]
                         (+ names of pools and layers), and surprise() for that twin
  <group>/_forecast.npz  forecast() FINAL tokens, float16 [T_post_tok, gh, gw, D]: identical for
                         every twin of a group (shared pre-contact frames), so stored once
  <group>/_forecast_<pre_segment>.npz   instead, when the group's twins carry more than one physlaw.pre_segment
                         (F8 ramp geometries, F4 boost levels: different pre-event frames): one forecast per
                         segment (+ pre_segment), each twin's from its own segment. Readers resolve the file with
                         physlaw.eval.l1_forecast_laws.forecast_path; one-segment groups (F1/F2/F3) are unchanged.
  _index.jsonl           one line per clip: ids, split, twin, surprise, file, per_frame (surprise per post-context
                         token-frame + t_s = time of its last master frame: the L2 clean window, spec §5.0.2).
                         The "obj" surprise region is ONE per twin group, the same for every twin
                         (physlaw.tokens.project.group_object_region: the union over the group's twins of both
                         objects' ROIs over the clean window, held fixed over the post token-frames)
  <twin>.npz also holds  pooled_frame_t [T_tok, L, D] (frame pool per token-frame) and token_t_s [T_tok] (L2-rep);
                         caches written before these exist are read unrestricted (L2 logs the count)
Diffusion models (kind: diffusion): no _forecast.npz; instead each twin's .npz carries
  fc_pooled [P, K, D] the sampled forecast's activations (keys fc_keys, e.g. "snr0/L15") pooled with
                      that twin's weights over the forecast token-frames,
  fc_pooled_frame_t [T_post, K, D] the forecast's frame pool per token-frame (rows = token-frames context_tokens..;
                      L1 / L4 gravity and rolling), and gen_seed0.mp4 is saved. Groups with
                      several pre_segments: one forecast per segment (gen_seed0_<seg>.mp4), each twin pooled with
                      its own segment's, + fc_segment in the twin's .npz; a pass-2 run redoes fc_pooled of such
                      twins cached before per-segment forecasts (no fc_segment: they got one shared forecast)
                      or before fc_pooled_frame_t (the forecast is sampled again).

Pooled rather than full-token activations: 13 layers x 8192 tokens x 1024 x 2 B = 218 MB
per clip would be ~70 GB for 320 clips. Pools are ROI_A/ROI_B/ROI_pair/frame x pre/post/all
(physlaw/tokens). Full tokens for chosen layers: --save-tokens LAYER ... (for L4 later).
Resumable: clips with an existing .npz are skipped unless --force.
Encoder-only weights (adapter ``predicts`` False, VideoMAE-v2 base): encode() pools only; no surprise, no forecast.
Text-conditioned models get one prompt per clip, physlaw.models.dit.clip_prompt: the config's ``prompt`` on F1-F4,
the family's neutral scene description elsewhere (a model card's own prompt, e.g. PhysRVG's, is kept everywhere).
Diffusion, two passes: --no-forecast caches encode() + surprise() only (L0, L2). A later run without it
adds fc_pooled to every .npz that lacks it (one sampled forecast per group; encode/surprise and the
index are not recomputed).
Encoders, P-pred-real (spec §5.2 L1 probe option): --pred-real computes ONLY the adapter's pred_real()
(V-JEPA 2: predictor outputs for masked real post-event tokens, rest of the clip as context; VideoMAE:
decoder hidden states with the real future visible elsewhere), pooled with the twin's post-event pools,
into a sidecar <group>/<twin>.pred_real.npz {pooled [P_post, D], pools, n_masks}. Existing files are
never rewritten; nothing else is recomputed. Not defined for diffusion models.

    HF_HUB_OFFLINE=1 flock /root/meta/gpu.lock /root/envs/physics_venv/bin/python \
        scripts/04_cache_activations.py --model configs/models/vjepa2_vitl.yaml --data data/f1 --pred-real
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
from physlaw.eval.l1_forecast_laws import (check_shared_forecast, clip_segment, forecast_path,  # noqa: E402
                                           forecast_segments)
from physlaw.eval.l2_surprise import clean_until_s  # noqa: E402
from physlaw.models.dit import clip_prompt  # noqa: E402
from physlaw.models.registry import load_adapter  # noqa: E402
from physlaw.tokens.pool import pool_weights  # noqa: E402
from physlaw.tokens.project import (group_object_region, project_masks, roi_object_ids,  # noqa: E402
                                    token_frame_phase)


def post_weights(Wt, names, phases, kt):
    """The twin's pooling weights restricted to token-frames >= kt (the forecast's frames), renormalised."""
    Wf = Wt.reshape(len(names), len(phases), -1)[:, kt:].reshape(len(names), -1)
    return Wf / Wf.sum(1, keepdim=True).clamp_min(1e-12)


def add_pred_real(m, x, k, f_pr, Wt, names, phases, kt, n_masks):
    """Encoders: P-pred-real features pooled with this twin's post-event pools -> sidecar .pred_real.npz."""
    from physlaw.models.vjepa2 import FINAL
    A = m.pred_real(x, k, n_masks=n_masks)[FINAL]
    post = [i for i, n in enumerate(names) if n.endswith(".post")]
    pooled = post_weights(Wt, names, phases, kt)[post] @ A.reshape(-1, A.shape[-1]).float()
    np.savez(f_pr, pooled=pooled.half().cpu().numpy(), pools=np.array([names[i] for i in post]), n_masks=n_masks)


def pool_latent(lat, Wt):
    """{"latent/snr+8": [T, gh, gw, C']} -> (pooled_latent [P, S, C'] float16, latent_keys)."""
    import torch
    lk = sorted(lat, key=lambda q: int(q.split("snr")[1]))
    return (torch.stack([Wt @ lat[q].reshape(-1, lat[q].shape[-1]).float() for q in lk], 1).half().cpu().numpy(),
            np.array(lk))


def append_npz(f_npz, **new):
    """Add (or replace) arrays in an existing .npz."""
    z = dict(np.load(f_npz))
    np.savez(f_npz, **{**z, **new})


def add_forecast(m, x, k, gid, seg, raw, out, Wt, names, phases, kt, fc_memo):
    """Diffusion: forecast = activations captured while sampling the continuation (seed 0); identical for
    every twin of one (group, pre_segment) (shared context latents), so sampled once per segment (``seg`` None:
    once per group). Pooled with THIS twin's weights over the forecast's token-frames; returns the arrays to append
    to the twin's .npz; the generated video is kept. ``raw``: the twin's own pre_segment label (guard)."""
    import torch
    if fc_memo.get("gid") != gid:
        fc_memo.clear(); fc_memo.update(gid=gid, seg={})
    if seg not in fc_memo["seg"]:
        acts_f, vid = m.forecast(x, k, seed=0, return_video=True)
        fc_memo["seg"][seg] = (raw, acts_f)
        import imageio.v3 as iio
        iio.imwrite(out / gid / ("gen_seed0.mp4" if seg is None else f"gen_seed0_{seg}.mp4"), vid, fps=m.fps,
                    codec="libx264", quality=9)
    raw0, acts = fc_memo["seg"][seg]
    check_shared_forecast(f"{gid} (segment {seg})", [raw0, raw])
    fk = sorted(acts)
    Wf = post_weights(Wt, names, phases, kt)
    fc_pooled = torch.stack([Wf @ acts[q].reshape(-1, acts[q].shape[-1]).float() for q in fk], 1)
    return {"fc_pooled": fc_pooled.half().cpu().numpy(), "fc_keys": np.array(fk), "context_tokens": kt,
            "fc_pooled_frame_t": frame_pool_t([acts[q] for q in fk]),
            **({} if seg is None else {"fc_segment": np.array(seg)})}


def frame_pool_t(acts: list) -> np.ndarray:
    """Frame pool per token-frame: every [T, gh, gw, D] space averaged over its tokens -> float16 [T, n_spaces, D]
    (real clips: ``pooled_frame_t`` over the encode layers; diffusion forecasts: ``fc_pooled_frame_t`` over the fc
    keys, rows = the forecast's token-frames context_tokens, context_tokens + 1, ...)."""
    import torch
    return torch.stack([a.reshape(a.shape[0], -1, a.shape[-1]).float().mean(1) for a in acts], 1).half().cpu().numpy()


def write_forecast(m, x, k, f, seg, raw, force, users):
    """Encoders: forecast() FINAL tokens of one (group, pre_segment) -> ``f`` (written once unless ``force``).
    ``users``: forecast file -> pre_segment labels of the twins using it in this run (guard). Per-segment files
    also carry ``pre_segment``, checked when the file already exists; ``_forecast.npz`` is written exactly as before."""
    from physlaw.models.vjepa2 import FINAL
    users.setdefault(f, set()).add(raw)
    check_shared_forecast(str(f), users[f])
    if f.exists() and not force:
        if seg is not None:
            z = np.load(f)
            check_shared_forecast(str(f), [raw, str(z["pre_segment"]) if "pre_segment" in z.files else raw])
        return
    extra = {} if seg is None else {"pre_segment": np.array(raw)}
    np.savez(f, final=m.forecast(x, k)[FINAL].half().cpu().numpy(), context_frames=k, **extra)


def token_times(frames: list[int], grid, master_fps: float) -> list[float]:
    """Clip time (s) of the LAST master frame of each token-frame (a token-frame is inside a window only when all of
    it is)."""
    return [max(frames[i] for i in idx) / master_fps for idx in grid.token_frames(len(frames))]


def surprise_per_frame(sur: dict, t_post: list[float]) -> dict | None:
    """Pops surprise()'s ``per_frame`` block and returns the index record's {"t_s", "surprise", "weight"} (None, after
    popping, when it is absent or its lengths do not match the post token-frames)."""
    pf = sur.pop("per_frame", None)
    if not pf or any(len(v) != len(t_post) for v in pf["surprise"].values()):
        return None
    return {"t_s": list(t_post), **pf}


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
    ap.add_argument("--pred-real", action="store_true",
                    help="encoders: only add <twin>.pred_real.npz (P-pred-real probe features) next to each clip")
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
    m = load_adapter(cfg, need="internals")
    if a.pred_real and not (hasattr(m, "pred_real") and m.predicts):
        sys.exit(f"[04] {cfg.name}: P-pred-real is not defined for this adapter (spec §5.2: encoders only)")
    n_masks = int(cfg.get("pred_real", {}).get("n_masks", 4))
    from physlaw.models.vjepa2 import FINAL
    t0, done = time.time(), 0
    diffusion = cfg.get("kind") == "diffusion"
    fc_memo: dict = {}          # diffusion: one sampled forecast per (group, pre_segment), pooled per twin
    fc_users: dict = {}         # encoders: forecast file -> pre_segment labels of its twins (guard)
    # forecasts are keyed by (group, pre_segment); seg None = the group has one pre_segment (_forecast.npz)
    labels_of = {f"{p.parent.name}/{p.name}": load_clip(p, frames=False, masks=False).labels for p in clips}
    seg_raw = {c_: clip_segment(lab) for c_, lab in labels_of.items()}
    regions: dict = {}          # group -> its one "obj" surprise region [gh, gw] (every twin of the group)

    def obj_region(gid, fr, master_fps):
        if gid not in regions:
            twins_g = [p for p in clips if p.parent.name == gid]
            ends = [clean_until_s(labels_of[f"{gid}/{p.name}"]) for p in twins_g]
            t_end = np.inf if any(e is None for e in ends) else min(ends)
            regions.clear()
            regions[gid] = group_object_region(((load_clip(p, frames=False).masks, labels_of[f"{gid}/{p.name}"])
                                                for p in twins_g), fr, m.token_grid, t_end, master_fps)
        return regions[gid]
    seg_key = forecast_segments(seg_raw)
    for g in sorted({c.split("/")[0] for c, s_ in seg_key.items() if s_ is not None}):
        if not diffusion and forecast_path(out, g).exists():
            print(f"[04] {g}: ignoring stale {forecast_path(out, g).name} (one forecast for several pre_segments); "
                  "writing _forecast_<pre_segment>.npz", flush=True)
    with open(out / "_index.jsonl", "a") as index:
        for path in clips:
            gid, twin = path.parent.name, path.name
            seg, raw = seg_key[f"{gid}/{twin}"], seg_raw[f"{gid}/{twin}"]
            f_npz = out / gid / f"{twin}.npz"
            f_pr = out / gid / f"{twin}.pred_real.npz"
            if a.pred_real and f_pr.exists() and not a.force:
                continue
            todo = None                                  # existing file: the parts to add (pass 2 / latent baseline)
            if f_npz.exists() and not a.force and not a.pred_real:
                have = np.load(f_npz).files if diffusion else ()
                fc_missing = (("fc_pooled" not in have or "fc_pooled_frame_t" not in have
                               or (seg is not None and "fc_segment" not in have)) if diffusion
                              else m.predicts and seg is not None and not forecast_path(out, gid, seg).exists())
                todo = [part for part, need in (("latent", diffusion and "pooled_latent" not in have),
                                                ("forecast", not (diffusion and a.no_forecast) and fc_missing))
                        if need]
                if not todo:
                    continue
            f_npz.parent.mkdir(parents=True, exist_ok=True)
            c = load_clip(path)
            if hasattr(m, "set_prompt"):                 # text-conditioned models: the family's prompt (dit.clip_prompt)
                m.set_prompt(clip_prompt(cfg, c.labels))
            fr = m.input_frames(c.labels)
            k = m.context_frames(c.labels)
            x = m.to_tensor(c.frames[fr])
            obj = project_masks(c.masks, fr, m.token_grid, roi_object_ids(c.labels))
            phases = token_frame_phase(fr, m.token_grid, c.pl["first_post_contact_frame"])
            W = pool_weights(obj, phases)
            names = sorted(W)
            Wt = torch.from_numpy(np.stack([W[n].reshape(-1) for n in names])).to(x.device)   # [P, N]
            kt = m.n_context_tokens(k) if hasattr(m, "n_context_tokens") else k // m.token_grid.tubelet
            if todo is not None:
                new = {}
                if "latent" in todo:
                    new.update(zip(("pooled_latent", "latent_keys"), pool_latent(m.latent_baseline(x), Wt)))
                if "forecast" in todo and diffusion:
                    new.update(add_forecast(m, x, k, gid, seg, raw, out, Wt, names, phases, kt, fc_memo))
                elif "forecast" in todo:
                    write_forecast(m, x, k, forecast_path(out, gid, seg), seg, raw, a.force, fc_users)
                if new:
                    append_npz(f_npz, **new)
                done += 1
                continue
            if a.pred_real:
                add_pred_real(m, x, k, f_pr, Wt, names, phases, kt, n_masks)
                done += 1
                if done % 20 == 0:
                    print(f"[04] pred_real {done}/{len(clips)}  {(time.time() - t0) / done:.2f} s/clip", flush=True)
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
            master_fps = float(c.labels["scene_constants"]["frame_rate"])
            reg = obj_region(gid, fr, master_fps)
            roi_post = np.broadcast_to(reg, (len(phases) - kt, *reg.shape)).copy()
            sur = m.surprise(x, k, roi_post) if m.predicts else {}       # encoder-only weights: encode() alone
            tok_t = token_times(fr, m.token_grid, master_fps)
            per_frame = surprise_per_frame(sur, tok_t[kt:])
            if per_frame is None and m.predicts:
                print(f"[04] WARNING {gid}/{twin}: surprise() returned no per-frame values matching the "
                      f"{len(tok_t) - kt} post token-frames; L2 will use the clip-level surprise", flush=True)
            # frame pool per token-frame (L2-rep restricts it to the clean window), float16 [T_tok, L, D]
            side["pooled_frame_t"] = frame_pool_t([acts[l] for l in main])
            side["token_t_s"] = np.array(tok_t)
            extra = {f"tokens_layer{l}": acts[l].half().cpu().numpy() for l in a.save_tokens if l in acts}
            if diffusion and not a.no_forecast:
                side.update(add_forecast(m, x, k, gid, seg, raw, out, Wt, names, phases, kt, fc_memo))
            np.savez(f_npz, pooled=pooled.half().cpu().numpy(), pools=np.array(names), layers=np.array(layers),
                     frames=np.array(fr), context_frames=k, phases=np.array(phases), **side, **extra)
            if not diffusion and m.predicts:
                write_forecast(m, x, k, forecast_path(out, gid, seg), seg, raw, a.force, fc_users)
            rec = {"clip": f"{gid}/{twin}", "group": gid, "twin": twin, "split": c.pl["split"],
                   "family": c.pl["family"], "surprise": sur, "file": str(f_npz.relative_to(out)),
                   **({"per_frame": per_frame} if per_frame else {}),
                   "model": cfg.name, "config_hash": config_hash(cfg)}
            index.write(json.dumps(rec) + "\n"); index.flush()
            done += 1
            if done % 20 == 0:
                print(f"[04] {done}/{len(clips)}  {(time.time() - t0) / done:.2f} s/clip", flush=True)
    print(f"[04] finished {done} clips in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
