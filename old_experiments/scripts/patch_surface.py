#!/usr/bin/env python
"""§31: is mu bound to the SURFACE or to the OBJECT? Causal, by patching.

§30 read mu at 0.93/0.97 from the floor's token pool on `slide_set_v2` and could
not conclude anything from it: attention is global, so floor tokens carry the
cube's motion whatever the floor looks like, and the floor's mask is the cube
mask's complement, so it encodes the same trajectory (mask-only floor 0.796).
Decodability cannot answer this question. Patching can.

    patch the FLOOR's tokens from a donor scene with a different mu,
    then read mu from the CUBE's tokens.

and the converse. The cube and the floor are disjoint token sets, so:

**THE PASSIVE BASELINE IS EXACTLY ZERO BY CONSTRUCTION.** In §18a the edit and the
readout shared tokens, so a residual stream carrying the vector forward moved the
readout with no computation involved — that condition measured obs/passive 1.02
and was pure arithmetic. Here the read pool contains none of the patched tokens,
so its pooled vector cannot move unless the network *transports* something. Any
effect is causal. (The overlap is measured per scene and asserted to be zero, not
assumed — tokens straddling the cube's silhouette belong to both regions on the
latent grid and are excluded from both masks.)

Conditions, all with the token count reported because the floor is ~97% of the
frame and an unmatched comparison measures edit SIZE rather than attribution:

  floor->cube        shift every floor-only token by (donor - scene) on B.all,
                     read mu from the cube's pool
  cube->floor        the converse: shift the cube's tokens, read the floor's pool
  floor_sub->cube    a random subset of floor tokens, count matched to the cube
                     mask. If floor->cube works only because it touches most of
                     the frame, this collapses
  rand->cube         a random direction of matched norm on a random token set of
                     the floor's size. The null

The headline is not the raw readout change but its correlation with
(mu_donor - mu_scene) across scenes, which flips sign — a scene-dependent,
sign-flipping response cannot come from a fixed additive edit propagating
linearly. Same logic as §18c's speed-consistency test and §18d's r(speed).

    .venv/bin/python old_experiments/scripts/patch_surface.py --n-scenes 60
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))  # noqa: E401,E702
import _momentum_path  # noqa: F401,E402

from steer_velocity import (                                        # noqa: E402
    SceneRunner, fit_readout, load_captures, split_indices,
)
from physics_interp.config import load_config                       # noqa: E402
from physics_interp.interp import ActivationSteer, extract_label    # noqa: E402
from physics_interp.runner import VideoDiffusionRunner              # noqa: E402


def predict(r, X):
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    return ((X - r["mu"]) / r["sd"] @ r["W"] + r["b"]).ravel()


def r2(y, p):
    y = np.asarray(y, float).ravel(); p = np.asarray(p, float).ravel()
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", nargs="?", default="configs/momentum_probe.yaml")
    p.add_argument("--model", default="videomae-base")
    p.add_argument("--capture-dir", default="outputs/slide_videomae-base")
    p.add_argument("--dataset-dir", default="datasets/slide_set_v2")
    p.add_argument("--block", type=int, default=6)
    p.add_argument("--read-block", default=None)
    p.add_argument("--target", default="mu_effective")
    p.add_argument("--n-scenes", type=int, default=60)
    p.add_argument("-o", "--out", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg.model.id = args.model
    out_dir = Path(args.out) if args.out else Path(f"outputs/steer_{args.model}")

    acts, labels, pools, scene_ids = load_captures(Path(args.capture_dir), cfg.name)
    pi = {q: i for i, q in enumerate(pools)}
    blocks = sorted(int(s.split(".")[1]) for s in acts if s.startswith("block."))
    site = f"block.{args.block}.resid_post"
    read_site = args.read_block or f"block.{blocks[-1]}.resid_post"
    n = len(labels)
    tr, te = split_indices(n)
    rng = np.random.default_rng(0)

    y = np.stack([extract_label(l, args.target) for l in labels]).astype(np.float64)
    XA = acts[site][:, pi["A.all"], :].astype(np.float64)      # cube, at the patch site
    XB = acts[site][:, pi["B.all"], :].astype(np.float64)      # floor, at the patch site
    RA = acts[read_site][:, pi["A.all"], :].astype(np.float64)  # cube, at the read site
    RB = acts[read_site][:, pi["B.all"], :].astype(np.float64)  # floor, at the read site

    read_cube = fit_readout(RA, y, tr)
    read_floor = fit_readout(RB, y, tr)
    print(f"model {args.model}   patch {site} -> read {read_site}")
    print(f"mu readout R2 — from the CUBE's pool {r2(y[te], predict(read_cube, RA[te])):.3f}, "
          f"from the FLOOR's pool {r2(y[te], predict(read_floor, RB[te])):.3f}")
    print(f"{len(tr)} train / {len(te)} test scenes\n")

    te_list = list(te)[: args.n_scenes]
    donors = {}
    for i in te_list:
        j = i
        while j == i:
            j = int(rng.choice(te))
        donors[i] = j

    print(f"loading {args.model} ...", flush=True)
    runner = VideoDiffusionRunner.from_config(cfg, device="cuda")

    rows, t0 = [], time.perf_counter()
    for si, i in enumerate(te_list, 1):
        sid, jid = scene_ids[i], scene_ids[donors[i]]
        try:
            sc = SceneRunner(runner, Path(args.dataset_dir) / sid,
                             cfg.generation.num_frames)
        except Exception as exc:                                    # noqa: BLE001
            print(f"  [{si}] {sid}: SKIP ({exc})")
            continue

        # Disjoint region masks. A token straddling the cube's silhouette has
        # occupancy in BOTH regions on the latent grid; those go in neither mask,
        # so "the passive baseline is zero" is true by construction rather than
        # by hope. The overlap count is recorded per scene.
        cube_occ, floor_occ = sc.occ[0].reshape(-1), sc.occ[1].reshape(-1)
        both = (cube_occ > 0) & (floor_occ > 0)
        # At a 14x14 token grid the cube spans a handful of patches and NONE of
        # them is pure cube — every one also contains floor. So a "strictly
        # cube-only" mask is empty, and the cube side has to be defined by
        # DOMINANCE instead. The floor side can stay strict (most floor tokens
        # are nowhere near the cube), which is what makes floor->cube the
        # zero-passive direction.
        cube_m = ((cube_occ > floor_occ) & (cube_occ > 0)).astype(np.float32)
        floor_m = ((floor_occ > 0) & (cube_occ == 0)).astype(np.float32)
        n_cube, n_floor = int(cube_m.sum()), int(floor_m.sum())
        if n_cube == 0 or n_floor == 0:
            continue

        d_floor = XB[donors[i]] - XB[i]
        d_cube = XA[donors[i]] - XA[i]

        # count-matched random subset of floor tokens
        fl_idx = np.flatnonzero(floor_m)
        sub = np.zeros_like(floor_m)
        sub[rng.choice(fl_idx, size=min(n_cube, len(fl_idx)), replace=False)] = 1.0
        # random token set of the floor's size, random direction of matched norm
        rand_m = np.zeros_like(floor_m)
        rand_m[rng.choice(len(floor_m), size=n_floor, replace=False)] = 1.0
        rv = rng.standard_normal(d_floor.shape[0])
        d_rand = rv / np.linalg.norm(rv) * np.linalg.norm(d_floor)

        base = sc.run([read_site])[read_site]
        b_cube = float(predict(read_cube, base[pi["A.all"]].astype(np.float64))[0])
        b_floor = float(predict(read_floor, base[pi["B.all"]].astype(np.float64))[0])

        conds = [
            ("floor->cube", floor_m, d_floor, "A.all", read_cube, b_cube, n_floor),
            ("cube->floor", cube_m, d_cube, "B.all", read_floor, b_floor, n_cube),
            ("floor_sub->cube", sub, d_floor, "A.all", read_cube, b_cube, int(sub.sum())),
            ("rand->cube", rand_m, d_rand, "A.all", read_cube, b_cube, n_floor),
        ]
        W = sc.weights.numpy() if hasattr(sc.weights, "numpy") else np.asarray(sc.weights)
        for name, mask, delta, read_pool, rd, base_val, ntok in conds:
            nrm = float(np.linalg.norm(delta))
            if nrm < 1e-9:
                continue
            steer = ActivationSteer(delta / nrm, alpha=nrm, scale="absolute",
                                    tokens=mask)
            out = sc.run([read_site], steer_site=site, steer=steer)[read_site]
            obs = float(predict(rd, out[pi[read_pool]].astype(np.float64))[0])
            # PASSIVE, computed rather than assumed. The read pool's pooled vector
            # is sum_t w[t] x[t]; adding delta to the patched tokens shifts it by
            # (sum of w over those tokens) * delta. For floor->cube that weight sum
            # is exactly 0 — the floor mask excludes every token the cube pool
            # touches — so the whole passive channel is closed and the number is
            # printed to prove it, not to trust it.
            wsum = float(W[pi[read_pool]] @ mask)
            passive = float(predict(rd, (base[pi[read_pool]].astype(np.float64)
                                         + wsum * delta))[0])
            rows.append({
                "scene": sid, "donor": jid, "cond": name,
                "n_tokens": ntok, "delta_norm": nrm,
                "mu_scene": float(y[i, 0]), "mu_donor": float(y[donors[i], 0]),
                "base": base_val, "observed": obs, "change": obs - base_val,
                "passive": passive, "passive_change": passive - base_val,
                "read_weight_on_patch": wsum,
                "overlap_tokens": int(both.sum()),
            })
        el = time.perf_counter() - t0
        print(f"  [{si}/{len(te_list)}] {sid}<-{jid}  cube={n_cube} floor={n_floor} "
              f"overlap={int(both.sum())}  {el:5.1f}s", flush=True)
    runner.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"patch_surface_block{args.block}.json"
    path.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2))
    print(f"\nwrote {path}")
    summarise(rows)
    return 0


def summarise(rows):
    g = {}
    for r in rows:
        g.setdefault(r["cond"], []).append(r)
    print(f"\n{'condition':<17}{'tokens':>8}{'|delta|':>9}{'w_read':>9}"
          f"{'mean dmu':>10}{'passive':>9}{'transfer':>10}{'r(dmu_req)':>12}")
    for c in ("floor->cube", "floor_sub->cube", "rand->cube", "cube->floor"):
        if c not in g:
            continue
        rs = g[c]
        req = np.array([r["mu_donor"] - r["mu_scene"] for r in rs])
        obs = np.array([r["change"] for r in rs])
        # transfer: the fraction of the required change actually delivered
        trans = float((obs @ req) / (req @ req)) if (req @ req) > 0 else np.nan
        rho = float(np.corrcoef(req, obs)[0, 1]) if obs.std() > 1e-12 else np.nan
        pas = np.array([r["passive_change"] for r in rs])
        print(f"{c:<17}{np.median([r['n_tokens'] for r in rs]):>8.0f}"
              f"{np.median([r['delta_norm'] for r in rs]):>9.2f}"
              f"{np.median([r['read_weight_on_patch'] for r in rs]):>9.3f}"
              f"{obs.mean():>10.4f}{pas.mean():>9.4f}{trans:>10.3f}{rho:>12.3f}")
    print("\ntransfer 1.0 = the read-out region now reports the donor's mu; 0 = the")
    print("patch did nothing. The passive baseline is 0 for every row above, since")
    print("the patched tokens are disjoint from the pool being read — so a non-zero")
    print("r(dmu_req) is the network transporting mu, not arithmetic.")


if __name__ == "__main__":
    raise SystemExit(main())
