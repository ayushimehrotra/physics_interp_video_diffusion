#!/usr/bin/env python
"""Steer the friction coefficient, under Joseph et al.'s held-out-probe protocol
(App C.12) plus the passive-propagation control that protocol is missing.

Their protocol, reproduced here:

  1. disjoint train/test split by scene
  2. an orthogonal probe sequence (INLP) fit on TRAIN activations -> the steering
     subspace, K probes deep
  3. a separate evaluation probe fit on TEST activations ONLY, so it has never
     seen the steering probes or the activations used to build the subspace
  4. solve for the coordinates inside the subspace that make all K train probes
     read the target, leaving the orthogonal complement untouched
  5. ask the held-out probe what it reads off the steered activations, and check
     the two-sided signature: MAE-to-target falls while MAE-to-truth rises

Why that is not sufficient on its own, and what is added here. A residual stream
carries an added vector forward, so a readout at a later block moves whether or
not the model *read* the edit — and the held-out probe reads the same subspace
the vector was added to, so it is fooled by pure arithmetic exactly as the
steering probes are. 18a measured a rank-1 edit at observed/passive = 1.02, i.e.
entirely passive, that would pass their protocol as a success. So every condition
below reports:

  observed  the held-out probe on activations from a real forward pass
  passive   the same probe on (baseline pooled activation + the injected delta),
            i.e. what it would read if the network did nothing at all
  obs/passive  > 1 means the network amplified the edit; ~1 means arithmetic

The edit is a uniform shift of every video token. Pooling weights sum to 1, so a
uniform token shift moves the pooled mean by exactly that shift — and pooling
frame-wide (`all.all`) keeps the whole experiment clear of 20's mask leak, which
on this dataset reaches 0.99 for object-local pools.

    .venv/bin/python old_experiments/scripts/steer_friction.py --n-scenes 60
    .venv/bin/python old_experiments/scripts/steer_friction.py --model vjepa2-vitl \
        --capture-dir outputs/slide_vjepa2-vitl --block 8 \
        --read-block block.23.resid_post --n-scenes 40
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

POOL = "all.all"          # frame-wide: mask identical in every scene, zero leak


def predict(r, X):
    """Readout in LABEL units. (apply_readout returns sd units.)"""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    return ((X - r["mu"]) / r["sd"] @ r["W"] + r["b"]).ravel()


def probe_sequence(X, Y, tr, rounds):
    """INLP: fit, project the direction out, refit. Returns the probes expressed
    as affine maps on the ORIGINAL space, plus the R2 after each round.

    Probe k is fit on X projected orthogonal to directions 1..k-1. Projection is
    linear, so probe k is still affine in the original x: with P_k the projector,
    pred_k(x) = (W_k/sd_k) . P_k x + const. Keeping them in the original basis is
    what lets the steering solve treat all K jointly.
    """
    d = X.shape[1]
    P = np.eye(d)
    G, C, curve, dirs = [], [], [], []
    Xw = X.copy()
    for _ in range(rounds):
        r = fit_readout(Xw, Y, tr)
        te_mask = np.ones(len(X), bool)
        te_mask[tr] = False
        pred = predict(r, Xw[te_mask])
        y = Y[te_mask].ravel()
        curve.append(float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()))

        g = (r["W"][:, 0] / r["sd"])                  # gradient in current space
        G.append(P.T @ g)                             # ... expressed on original x
        C.append(float(r["b"][0] - (r["mu"] / r["sd"]) @ r["W"][:, 0]))

        q = g / (np.linalg.norm(g) + 1e-12)
        dirs.append(P.T @ q)
        Xw = Xw - np.outer(Xw @ q, q)
        P = P - np.outer(q, P.T @ q)                  # accumulate the projector
    return np.array(G), np.array(C), curve, np.array(dirs)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", nargs="?", default="configs/momentum_probe.yaml")
    p.add_argument("--model", default="videomae-base")
    p.add_argument("--capture-dir", default="outputs/slide_videomae-base")
    p.add_argument("--dataset-dir", default="datasets/slide_set_v2")
    p.add_argument("--block", type=int, default=6, help="where the edit is written")
    p.add_argument("--read-block", default=None, help="where the held-out probe reads")
    p.add_argument("--target", default="mu_effective")
    p.add_argument("--target-value", type=float, default=0.70,
                   help="single value every scene is steered toward, as in C.12")
    p.add_argument("--k-list", default="1,2,4,8,16,24")
    p.add_argument("--n-scenes", type=int, default=60)
    p.add_argument("--inlp-rounds", type=int, default=30)
    p.add_argument("-o", "--out", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg.model.id = args.model
    out_dir = Path(args.out) if args.out else Path(f"outputs/steer_{args.model}")
    ks = [int(k) for k in args.k_list.split(",")]

    acts, labels, pools, scene_ids = load_captures(Path(args.capture_dir), cfg.name)
    pi = {q: i for i, q in enumerate(pools)}
    blocks = sorted(int(s.split(".")[1]) for s in acts if s.startswith("block."))
    site = f"block.{args.block}.resid_post"
    read_site = args.read_block or f"block.{blocks[-1]}.resid_post"
    n = len(labels)
    tr, te = split_indices(n)

    Y = np.stack([extract_label(l, args.target) for l in labels]).astype(np.float64)
    Yv0 = np.stack([extract_label(l, "speed_initial") for l in labels]).astype(np.float64)
    # The consequences a genuine edit must carry with it: stop_frame = v0*fps/mu
    # and travel = v0^2/(2 mu). Both are scene-dependent functions of the steered
    # value, and their required change FLIPS SIGN depending on whether a scene's
    # own mu sits above or below the target — which an additive edit, being
    # linear, cannot produce by propagation. This is 18c's speed-consistency
    # test in the friction domain, and it is the only check here that tests
    # whether the model re-derived anything.
    Ystop = np.stack([extract_label(l, "stop_frame") for l in labels]).astype(np.float64)
    Ytrav = np.stack([extract_label(l, "travel") for l in labels]).astype(np.float64)
    X = acts[site][:, pi[POOL], :].astype(np.float64)
    Xr = acts[read_site][:, pi[POOL], :].astype(np.float64)
    y_sd = float(Y[tr].std())

    print(f"model {args.model}   steer {site} -> read {read_site}   pool {POOL}")
    print(f"{len(tr)} train / {len(te)} test scenes, target {args.target} "
          f"-> {args.target_value} (label sd {y_sd:.3f})\n")

    # --- the steering subspace, from TRAIN only ---------------------------- #
    G, C, curve, _ = probe_sequence(X, Y, tr, args.inlp_rounds)
    print("INLP on train: R2 after each orthogonalisation")
    print("  " + "  ".join(f"{i + 1}:{v:.2f}" for i, v in enumerate(curve)))
    below = [i + 1 for i, v in enumerate(curve) if v < 0.1]
    print(f"  dimensionality (first round under R2 0.1): "
          f"{below[0] if below else '>' + str(args.inlp_rounds)}\n")

    # --- the evaluation probe, from TEST only ------------------------------ #
    ev = fit_readout(Xr, Y, te)
    ev_v0 = fit_readout(Xr, Yv0, te)
    ev_stop = fit_readout(Xr, Ystop, te)
    ev_trav = fit_readout(Xr, Ytrav, te)
    ev_r2 = float(1 - ((Y[te].ravel() - predict(ev, Xr[te])) ** 2).sum()
                  / ((Y[te].ravel() - Y[te].mean()) ** 2).sum())
    print(f"held-out evaluation probe (fit on TEST activations only): R2 = {ev_r2:.3f}")
    print("  it has never seen the steering probes or the subspace they span\n")

    te_list = list(te)[: args.n_scenes]
    rng = np.random.default_rng(0)
    print(f"loading {args.model} ...", flush=True)
    runner = VideoDiffusionRunner.from_config(cfg, device="cuda")
    rec_sites = [read_site]

    rows, t0 = [], time.perf_counter()
    for si, i in enumerate(te_list, 1):
        sid = scene_ids[i]
        try:
            sc = SceneRunner(runner, Path(args.dataset_dir) / sid,
                             cfg.generation.num_frames)
        except Exception as exc:                                    # noqa: BLE001
            print(f"  [{si}] {sid}: SKIP ({exc})")
            continue

        base = sc.run(rec_sites)[read_site][pi[POOL]].astype(np.float64)
        base_read = float(predict(ev, base)[0])
        base_v0 = float(predict(ev_v0, base)[0])
        base_stop = float(predict(ev_stop, base)[0])
        base_trav = float(predict(ev_trav, base)[0])

        conds = []
        for k in ks:
            Gk, Ck = G[:k], C[:k]
            resid = args.target_value - (Gk @ X[i] + Ck)     # what each probe is off by
            delta = np.linalg.pinv(Gk) @ resid               # min-norm, inside the subspace
            conds.append((f"k={k}", delta))
        # matched-norm random direction, at the largest K's magnitude
        big = conds[-1][1]
        rnd = rng.standard_normal(X.shape[1])
        conds.append(("random", rnd / np.linalg.norm(rnd) * np.linalg.norm(big)))

        for name, delta in conds:
            nrm = float(np.linalg.norm(delta))
            if nrm < 1e-9:
                continue
            steer = ActivationSteer(delta / nrm, alpha=nrm, scale="absolute",
                                    tokens=None)          # every video token
            out = sc.run(rec_sites, steer_site=site, steer=steer)[read_site][pi[POOL]]
            obs = float(predict(ev, out.astype(np.float64))[0])
            passive = float(predict(ev, base + delta)[0])
            rows.append({
                "scene": sid, "cond": name, "k": name, "delta_norm": nrm,
                "true": float(Y[i, 0]), "target": args.target_value,
                "base_read": base_read, "observed": obs, "passive": passive,
                "v0_base": base_v0,
                "v0_observed": float(predict(ev_v0, out.astype(np.float64))[0]),
                "v0_true": float(Yv0[i, 0]),
                "stop_base": base_stop,
                "stop_observed": float(predict(ev_stop, out.astype(np.float64))[0]),
                "stop_passive": float(predict(ev_stop, base + delta)[0]),
                "trav_base": base_trav,
                "trav_observed": float(predict(ev_trav, out.astype(np.float64))[0]),
                "trav_passive": float(predict(ev_trav, base + delta)[0]),
            })
        el = time.perf_counter() - t0
        print(f"  [{si}/{len(te_list)}] {sid}  {el:6.1f}s, "
              f"~{el / (si / len(te_list)) - el:5.1f}s left", flush=True)
    runner.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    # Target in the filename: two runs at different targets are different
    # experiments, and without this the second silently overwrites the first.
    path = out_dir / f"steer_friction_block{args.block}_t{args.target_value:g}.json"
    path.write_text(json.dumps(
        {"args": vars(args), "inlp_curve": curve, "eval_probe_r2": ev_r2,
         "label_sd": y_sd, "rows": rows}, indent=2))
    print(f"\nwrote {path}")
    summarise(rows, y_sd)
    return 0


FPS = 8.0


def _consistency(rs, which):
    """Correlation between the OBSERVED change in a consequence readout and the
    change the steered value actually implies, scene by scene.

    stop_frame = v0*fps/mu and travel = v0^2/(2 mu), so steering mu -> mu* implies
    a change proportional to (1/mu* - 1/mu_true) scaled by the scene's own v0.
    That flips sign across scenes whenever the target sits inside the mu range,
    and an additive edit propagating linearly gives the SAME shift in every scene
    — so a high correlation here cannot come from pass-through.
    """
    inv = np.array([1.0 / r["target"] - 1.0 / r["true"] for r in rs])
    v0 = np.array([r["v0_true"] for r in rs])
    pred = v0 * FPS * inv if which == "stop" else 0.5 * v0 ** 2 * inv
    obs = np.array([r[f"{which}_observed"] - r[f"{which}_base"] for r in rs])
    if pred.std() < 1e-9 or obs.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(pred, obs)[0, 1])


def summarise(rows, y_sd):
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(r["cond"], []).append(r)

    b = next(iter(groups.values()))
    print(f"\nBASELINE (no steering), n={len(b)}")
    print(f"  MAE to truth  {np.mean([abs(r['base_read'] - r['true']) for r in b]):.4f}"
          f"   MAE to target {np.mean([abs(r['base_read'] - r['target']) for r in b]):.4f}")

    print(f"\n{'cond':<9} {'|delta|':>8} {'MAE->target':>12} {'MAE->truth':>11} "
          f"{'obs/passive':>12} {'v0 shift(sd)':>13} {'r(stop)':>9} {'r(travel)':>10}")
    order = sorted(groups, key=lambda c: (c == "random", len(c), c))
    for c in order:
        rs = groups[c]
        mt = np.mean([abs(r["observed"] - r["target"]) for r in rs])
        mr = np.mean([abs(r["observed"] - r["true"]) for r in rs])
        num = np.array([r["observed"] - r["base_read"] for r in rs])
        den = np.array([r["passive"] - r["base_read"] for r in rs])
        ratio = float(np.median(num / np.where(np.abs(den) < 1e-9, np.nan, den)))
        v0 = np.mean([abs(r["v0_observed"] - r["v0_base"]) for r in rs])
        print(f"{c:<9} {np.median([r['delta_norm'] for r in rs]):>8.2f} "
              f"{mt:>12.4f} {mr:>11.4f} {ratio:>12.2f} {v0:>13.3f} "
              f"{_consistency(rs, 'stop'):>9.3f} {_consistency(rs, 'trav'):>10.3f}")
    print("\nMAE-to-target should FALL and MAE-to-truth should RISE (Joseph et al.'s")
    print("two-sided check). obs/passive ~1 means the readout moved by arithmetic")
    print("alone and the network never read the edit.")


if __name__ == "__main__":
    raise SystemExit(main())
