#!/usr/bin/env python
"""Activation patching: transplant one scene's A-local representation into
another, and ask what moves.

This is the intervention the steering runs could not be: the target values come
from activations the model actually produced, so the edited state is on-manifold
by construction, and there is no fitted direction whose alignment with the
readout could manufacture an effect. It also changes the whole redundant code at
once, which the INLP analysis showed is necessary — velocity_A survives removing
12 rounds of readout directions at R2 0.80, so no single-direction edit can
overwrite it.

Four variants, all shifting A-local tokens by a fixed vector so the pooled A.pre
mean lands exactly on the donor's (pool weights sum to 1, so a uniform shift of
the pooled mean is exactly the shift applied to each token):

  full        x_j - x_i                 transfer everything about A
  velocity    P_V (x_j - x_i)           transfer only the velocity subspace
  radius      P_U (x_j - x_i)           specificity control: a different real
                                        factor, same construction
  random      P_R (x_j - x_i)           a random subspace of matched dimension,
                                        the null for "any k-dim slice of a real
                                        activation difference does this"

The headline is the *transfer fraction*: project the readout change onto the
change that would be required to fully adopt the donor's velocity. 1.0 means the
model now represents A as moving at the donor's velocity; 0 means the patch did
nothing. Leak is B's velocity readout, which should not move at all — scene i's
B was never touched.

The speed check is what makes this stronger than the steering runs. Patching
predicts a specific, pair-dependent speed change (|v_j| - |v_i|) that flips sign
across pairs. A linear pass-through cannot produce that correlation; only a
network that re-derives speed from the patched velocity can.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steer_velocity import (                                        # noqa: E402
    SceneRunner, apply_readout, fit_readout, load_captures, r2_of, split_indices,
)
from physics_interp.config import load_config                       # noqa: E402
from physics_interp.interp import ActivationSteer, extract_label    # noqa: E402
from physics_interp.runner import VideoDiffusionRunner              # noqa: E402


def inlp_subspace(X, Y, tr, rounds):
    """Directions carrying `Y`, peeled off one readout at a time.

    Returns an orthonormal [dim, k] basis. Each round contributes as many
    directions as Y has columns, so `rounds` x Y.shape[1] = k.
    """
    Xw, cols = X.copy(), []
    for _ in range(rounds):
        r = fit_readout(Xw, Y, tr)
        D = r["W"] / r["sd"][:, None]
        D = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
        Q, _ = np.linalg.qr(D)
        cols.append(Q)
        Xw = Xw - (Xw @ Q) @ Q.T
    B = np.concatenate(cols, axis=1)
    Q, _ = np.linalg.qr(B)
    return Q


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", nargs="?", default="configs/momentum_probe.yaml")
    p.add_argument("--model", default="videomae-base")
    p.add_argument("--capture-dir", default="outputs_v3_videomae-base")
    p.add_argument("--dataset-dir", default="momentum_dataset/probe_set_v3_latent")
    p.add_argument("--block", type=int, default=6)
    p.add_argument("--read-block", default=None)
    p.add_argument("--rounds", type=int, default=8, help="INLP rounds -> k = 2*rounds")
    p.add_argument("--n-scenes", type=int, default=60)
    p.add_argument("-o", "--out", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg.model.id = args.model
    out_dir = Path(args.out) if args.out else Path(f"outputs_steer_{args.model}")

    acts, labels, pools, scene_ids = load_captures(Path(args.capture_dir), cfg.name)
    n = len(labels)
    pi = {q: i for i, q in enumerate(pools)}
    blocks = sorted(int(s.split(".")[1]) for s in acts if s.startswith("block."))
    site = f"block.{args.block}.resid_post"
    read_site = args.read_block or f"block.{blocks[-1]}.resid_post"
    tr, te = split_indices(n)
    idx_of = {s: i for i, s in enumerate(scene_ids)}

    Y = {t: np.stack([extract_label(l, t) for l in labels])
         for t in ("velocity_A_pre", "velocity_B_pre", "radius_A", "speed_A_pre")}
    X = acts[site][:, pi["A.pre"], :].astype(np.float64)
    vA = Y["velocity_A_pre"]

    print(f"site {site}, read {read_site}, {len(tr)} train / {len(te)} test")
    k = 2 * args.rounds
    V = inlp_subspace(X, vA, tr, args.rounds)
    U = inlp_subspace(X, Y["radius_A"], tr, k)          # 1 col/round -> k cols
    rng = np.random.default_rng(0)
    R, _ = np.linalg.qr(rng.normal(size=(X.shape[1], k)))
    print(f"subspaces: velocity {V.shape[1]}d, radius {U.shape[1]}d, random {R.shape[1]}d")

    readouts, r2s = {}, {}
    for target, pool in (("velocity_A_pre", "A.pre"), ("velocity_B_pre", "B.pre"),
                         ("radius_A", "A.pre"), ("speed_A_pre", "A.pre")):
        Xr = acts[read_site][:, pi[pool], :].astype(np.float64)
        r = fit_readout(Xr, Y[target], tr)
        r2s[f"{target}|{pool}"] = r2_of(r, Xr, Y[target], te)
        readouts[(target, pool)] = r
    print("readout R2: " + "  ".join(f"{a} {b:.3f}" for a, b in r2s.items()))

    # Donor pairing: each test scene receives from another test scene, drawn so
    # no scene is its own donor. Random rather than max-contrast, so the
    # transfer fraction is not inflated by selecting extreme pairs.
    te_list = list(te)[: args.n_scenes]
    donors = {}
    for i in te_list:
        j = i
        while j == i:
            j = int(rng.choice(te))
        donors[i] = j

    projections = {"full": None, "velocity": V, "radius": U, "random": R}

    print(f"\nloading {args.model} ...", flush=True)
    runner = VideoDiffusionRunner.from_config(cfg, device="cuda")
    rec_sites = [f"block.{b}.resid_post" for b in blocks]

    rows, t0 = [], time.perf_counter()
    for si, i in enumerate(te_list, 1):
        sid, jid = scene_ids[i], scene_ids[donors[i]]
        try:
            sc = SceneRunner(runner, Path(args.dataset_dir) / sid,
                             cfg.generation.num_frames)
        except Exception as exc:                                    # noqa: BLE001
            print(f"  [{si}] {sid}: SKIP ({exc})")
            continue
        if sc.n_steer_tokens == 0:
            continue

        base = sc.run(rec_sites)
        bp = {kk: apply_readout(r, base[read_site][pi[kk[1]]])[0]
              for kk, r in readouts.items()}
        # The donor difference is taken between *captured* pooled activations,
        # so it is exactly the shift that moves i's pooled A.pre onto j's.
        diff = X[donors[i]] - X[i]
        dv_target = vA[donors[i]] - vA[i]

        for pname, P in projections.items():
            delta = diff if P is None else (P @ (P.T @ diff))
            nrm = float(np.linalg.norm(delta))
            if nrm < 1e-9:
                continue
            # scale="absolute" so the applied shift is exactly `delta` and the
            # pooled mean lands where the donor's is (for the full variant).
            steer = ActivationSteer(delta / nrm, alpha=nrm, scale="absolute",
                                    tokens=sc.steer_mask)
            out = sc.run(rec_sites, steer_site=site, steer=steer)

            kA = ("velocity_A_pre", "A.pre")
            d_vA = apply_readout(readouts[kA], out[read_site][pi["A.pre"]])[0] - bp[kA]
            # Transfer fraction: how far along the required change did we get?
            # Both are in label-sd units, so the projection is dimensionless.
            tgt = dv_target / (readouts[kA]["y_sd"])
            denom = float(tgt @ tgt) + 1e-12
            row = {
                "scene": sid, "donor": jid, "variant": pname,
                "delta_norm": nrm,
                "transfer": float(d_vA @ tgt / denom),
                "d_vA_rms": float(np.sqrt((d_vA ** 2).mean())),
                "vA_true_x": float(vA[i, 0]), "vA_true_y": float(vA[i, 1]),
                "vJ_true_x": float(vA[donors[i], 0]),
                "vJ_true_y": float(vA[donors[i], 1]),
            }
            for kk, r in readouts.items():
                d = apply_readout(r, out[read_site][pi[kk[1]]])[0] - bp[kk]
                tag = f"{kk[0].replace('velocity_', 'v').replace('_pre', '')}|{kk[1][0]}"
                row[f"d_{tag}"] = float(np.sqrt((d ** 2).mean())) if d.size > 1 \
                    else float(d[0])
            dp = apply_readout(readouts[kA],
                               base[read_site][pi["A.pre"]] + delta)[0] - bp[kA]
            row["passive_transfer"] = float(dp @ tgt / denom)
            rows.append(row)
        el = time.perf_counter() - t0
        print(f"  [{si}/{len(te_list)}] {sid}<-{jid}  {el:6.1f}s, "
              f"~{el / (si / len(te_list)) - el:5.1f}s left", flush=True)
    runner.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"patch_block{args.block}_k{k}.json"
    path.write_text(json.dumps({"args": vars(args), "readout_r2": r2s,
                                "k": k, "rows": rows}, indent=2))
    print(f"\nwrote {path}")
    summarise(rows)
    return 0


def summarise(rows):
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(r["variant"], []).append(r)
    print(f"\n{'variant':<10} {'|delta|':>9} {'transfer':>9} {'passive':>9} "
          f"{'d_vA':>8} {'leak d_vB|B':>12} {'d_radius':>9} {'r(speed)':>9}")
    for v, rs in sorted(groups.items()):
        vi = np.array([[r["vA_true_x"], r["vA_true_y"]] for r in rs])
        vj = np.array([[r["vJ_true_x"], r["vJ_true_y"]] for r in rs])
        pred = np.linalg.norm(vj, axis=1) - np.linalg.norm(vi, axis=1)
        obs = np.array([r["d_speed_A|A"] for r in rs])
        rho = (np.corrcoef(pred, obs)[0, 1]
               if obs.std() > 1e-9 and pred.std() > 1e-9 else np.nan)
        m = lambda k: np.median([r[k] for r in rs])                 # noqa: E731
        print(f"{v:<10} {m('delta_norm'):>9.2f} {m('transfer'):>9.3f} "
              f"{m('passive_transfer'):>9.3f} {m('d_vA|A'):>8.3f} "
              f"{m('d_vB|B'):>12.3f} {m('d_radius_A|A'):>9.3f} {rho:>9.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
