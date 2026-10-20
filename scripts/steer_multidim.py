#!/usr/bin/env python
"""Multi-dimensional velocity steering — does the *generative* direction work
where the discriminative one only propagated?

`steer_velocity.py` steered along ridge probe weights and found the on-target
effect was pure passive propagation (observed/passive ~ 1.0). The INLP analysis
explains why: velocity_A survives removing 12 rounds of readout directions at
R2 0.80, so it is encoded redundantly across many axes. Adding a bump along one
discriminative axis moves that readout while the model still reads the original
velocity off every other axis.

The fix is to stop asking "which direction best *predicts* velocity" and start
asking "how does the activation actually *move* when velocity changes". Those
are different objects — the discriminative solution is Sigma^-1 a, the
generative one is a, and §5 already measured them to be nearly orthogonal on
CogVideoX despite both decoding at R2 ~0.85.

  probe    ridge weights X -> v      (discriminative; what was tried before)
  gen      OLS of X on v             (generative; moves every co-varying axis
                                      in the proportion real velocity does)

`gen` is still a rank-1 additive edit for a fixed delta-v — the multi-
dimensionality is in how it is *pointed*, not its rank. A genuine rank-k content
replacement needs a real target activation, which is activation patching and is
the next script. This one isolates whether direction choice alone is enough.

Signed per-component readouts are reported because RMS hides the question that
matters here: steering v_x should move the x readout and leave y alone.
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


def generative_map(X, Y):
    """OLS of activations on labels: how the activation moves per unit of y.

    Returns G with shape [k_labels, dim] so `dv @ G` is the activation change a
    real velocity change of `dv` produces. This is the direction the *data*
    says velocity travels along, as opposed to the direction that best reads it
    out — under d >> n those come apart badly.
    """
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    return np.linalg.lstsq(Yc, Xc, rcond=None)[0]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", nargs="?", default="configs/momentum_probe.yaml")
    p.add_argument("--model", default="videomae-base")
    p.add_argument("--capture-dir", default="outputs_v3_videomae-base")
    p.add_argument("--dataset-dir", default="momentum_dataset/probe_set_v3_latent")
    p.add_argument("--block", type=int, default=6)
    p.add_argument("--read-block", default=None)
    p.add_argument("--alphas", default="4,8")
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

    Y = {t: np.stack([extract_label(l, t) for l in labels])
         for t in ("velocity_A_pre", "velocity_B_pre", "radius_A", "speed_A_pre")}
    X = acts[site][:, pi["A.pre"], :].astype(np.float64)

    # ---- directions ------------------------------------------------------- #
    rng = np.random.default_rng(0)
    vA = Y["velocity_A_pre"]
    sd_v = vA[tr].std(0)

    G = generative_map(X[tr], vA[tr])                       # [2, dim]
    rd = fit_readout(X, vA, tr)                             # discriminative
    Wd = rd["W"] / rd["sd"][:, None]                        # [dim, 2], unstandardized

    unit = lambda v: v / (np.linalg.norm(v) + 1e-12)        # noqa: E731
    dirs = {
        "probe":  unit(Wd[:, 0]),                           # discriminative, x
        "gen_x":  unit(G[0]),                                # generative, x
        "gen_y":  unit(G[1]),                                # generative, y
        "gen_xy": unit(G[0] * sd_v[0] + G[1] * sd_v[1]),
    }
    perm = rng.permutation(len(tr))
    # NOTE: this is a contaminated control. OLS of X on shuffled labels still
    # loads on X's dominant variance directions, which velocity also loads on,
    # so it comes out non-trivially aligned with gen_x (measured cos -0.42) and
    # behaves like a scaled anti-gen_x rather than a null. Kept for the record;
    # `random` is the clean isotropic control and `gen_radius` the clean
    # specificity control (a real factor that is not velocity).
    dirs["gen_shuffled"] = unit(generative_map(X[tr], vA[tr][perm])[0])
    dirs["gen_radius"] = unit(generative_map(X[tr], Y["radius_A"][tr])[0])
    dirs["random"] = unit(rng.normal(size=X.shape[1]))

    print(f"site {site}, read {read_site}, {len(tr)} train / {len(te)} test")
    print("\ndirection geometry (cos):")
    names = list(dirs)
    print("            " + " ".join(f"{m:>13}" for m in names))
    for a in names:
        print(f"{a:>12} " + " ".join(f"{float(dirs[a] @ dirs[b]):>13.3f}" for b in names))

    # ---- readouts --------------------------------------------------------- #
    readouts, r2s = {}, {}
    for target, pool in (("velocity_A_pre", "A.pre"), ("velocity_B_pre", "B.pre"),
                         ("radius_A", "A.pre"), ("speed_A_pre", "A.pre")):
        Xr = acts[read_site][:, pi[pool], :].astype(np.float64)
        r = fit_readout(Xr, Y[target], tr)
        r2s[f"{target}|{pool}"] = r2_of(r, Xr, Y[target], te)
        readouts[(target, pool)] = r
    print("\nreadout R2: " + "  ".join(f"{k} {v:.3f}" for k, v in r2s.items()))

    # ---- GPU -------------------------------------------------------------- #
    alphas = [float(x) for x in args.alphas.split(",")]
    test_ids = [scene_ids[i] for i in te][: args.n_scenes]
    dataset = Path(args.dataset_dir)
    print(f"\nloading {args.model} ...", flush=True)
    runner = VideoDiffusionRunner.from_config(cfg, device="cuda")
    rec_sites = [f"block.{b}.resid_post" for b in blocks]

    idx_of = {s: i for i, s in enumerate(scene_ids)}
    rows = []
    t0 = time.perf_counter()
    for si, sid in enumerate(test_ids, 1):
        try:
            sc = SceneRunner(runner, dataset / sid, cfg.generation.num_frames)
        except Exception as exc:                                    # noqa: BLE001
            print(f"  [{si}] {sid}: SKIP ({exc})")
            continue
        if sc.n_steer_tokens == 0:
            continue
        base = sc.run(rec_sites)
        bp = {k: apply_readout(r, base[read_site][pi[k[1]]])[0]
              for k, r in readouts.items()}
        h0 = base[site][pi["A.pre"]].astype(np.float64)

        for dname, vec in dirs.items():
            for alpha in alphas:
                steer = ActivationSteer(vec, alpha=alpha, tokens=sc.steer_mask)
                out = sc.run(rec_sites, steer_site=site, steer=steer)
                # float64: activations are stored fp16 and V-JEPA2's reach
                # |x| ~ 264, so a sum of squares overflows fp16 and norm() returns
                # inf, silently reporting a perturbation of exactly 0.000.
                h1 = out[site][pi["A.pre"]].astype(np.float64)
                row = {"scene": sid, "direction": dname, "alpha": alpha,
                       "perturbation": float(np.linalg.norm(h1 - h0)
                                             / (np.linalg.norm(h0) + 1e-12))}
                for k, r in readouts.items():
                    d = apply_readout(r, out[read_site][pi[k[1]]])[0] - bp[k]
                    tag = f"{k[0].replace('velocity_', 'v').replace('_pre', '')}|{k[1][0]}"
                    if d.size == 2:                      # signed, per component
                        row[f"{tag}x"], row[f"{tag}y"] = float(d[0]), float(d[1])
                    else:
                        row[tag] = float(d[0])
                kA = ("velocity_A_pre", "A.pre")
                dp = apply_readout(readouts[kA],
                                   base[read_site][pi["A.pre"]] + (h1 - h0))[0] - bp[kA]
                row["passive_x"] = float(dp[0])
                # The scene's true velocity, so the speed response can be checked
                # for physical consistency post hoc. An additive edit is linear,
                # so a purely passive network gives the *same* speed-readout
                # change in every scene; a network that re-computes must give a
                # scene-dependent one, because d|v| = (v.dv)/|v| flips sign with
                # the sign of v. v_Ax is 52% positive here, so the two
                # predictions are cleanly separated.
                row["vA_true_x"], row["vA_true_y"] = float(vA[idx_of[sid], 0]), \
                    float(vA[idx_of[sid], 1])
                rows.append(row)
        el = time.perf_counter() - t0
        print(f"  [{si}/{len(test_ids)}] {sid}  {el:6.1f}s, "
              f"~{el / (si / len(test_ids)) - el:5.1f}s left", flush=True)
    runner.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"multidim_block{args.block}.json"
    path.write_text(json.dumps(
        {"args": vars(args), "readout_r2": r2s, "n_scenes": len(test_ids),
         "cos": {a: {b: float(dirs[a] @ dirs[b]) for b in names} for a in names},
         "rows": rows}, indent=2))
    print(f"\nwrote {path}")

    keys = [k for k in rows[0]
            if k not in ("scene", "direction", "alpha", "vA_true_x", "vA_true_y")]
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["alpha"], r["direction"]), []).append(r)
    print(f"\n{'alpha':>6} {'direction':<14} " + " ".join(f"{k:>12}" for k in keys))
    for (a, d), rs in sorted(groups.items()):
        print(f"{a:>6.1f} {d:<14} "
              + " ".join(f"{np.median([r[k] for r in rs]):>12.3f}" for k in keys))

    # ---- physical consistency of the speed response ----------------------- #
    # Does the speed readout move the way the model's *own* velocity readout
    # change implies? predicted d|v| = (v . dv) / |v|, which is scene-dependent
    # and sign-flipping. A linear pass-through cannot produce that; only a
    # network that re-derives speed from the edited velocity can.
    print(f"\n{'alpha':>6} {'direction':<14} {'r(pred,obs)':>12} {'sd(obs)':>10} "
          f"{'mean|obs|':>10}   (speed-response consistency)")
    for (a, d), rs in sorted(groups.items()):
        v = np.array([[r["vA_true_x"], r["vA_true_y"]] for r in rs])
        dv = np.array([[r["vA|Ax"], r["vA|Ay"]] for r in rs])
        nrm = np.linalg.norm(v, axis=1) + 1e-12
        pred = (v * dv).sum(1) / nrm
        obs = np.array([r["speed_A|A"] for r in rs])
        rho = (np.corrcoef(pred, obs)[0, 1]
               if obs.std() > 1e-9 and pred.std() > 1e-9 else np.nan)
        print(f"{a:>6.1f} {d:<14} {rho:>12.3f} {obs.std():>10.3f} "
              f"{np.abs(obs).mean():>10.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
