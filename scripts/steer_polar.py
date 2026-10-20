#!/usr/bin/env python
"""Is the velocity code polar (speed x heading) or Cartesian (v_x, v_y)?

The correlational version — fit a generative speed axis inside each heading
quadrant and ask whether it stays put — said VideoMAE rotates (0.55 of its
split-half ceiling) and V-JEPA2 does not (0.86). This is the causal version, and
it is decisive because the two hypotheses predict opposite *directions* for the
induced change.

Steer along the fitted speed axis and watch where the velocity readout moves:

  polar      the model has a genuine magnitude knob, so the induced change
             rotates to align with each scene's own heading -> par ~ 1
  Cartesian  the "speed axis" is really one fixed direction in the plane, so the
             induced change points the same way in every scene. Headings here are
             spread over the full circle, so its alignment with each scene's own
             heading averages to ~0, while its concentration is ~1.

`par` and `conc` therefore separate the hypotheses cleanly, and they disagree in
opposite directions, so neither can be explained by a weak intervention.

The heading directions are the mirror test: rotating a vector without changing
its length means moving perpendicular to it, so a working heading steer should
show perp >> par under either code.

    .venv/bin/python scripts/steer_polar.py --model videomae-base \\
        --capture-dir outputs_v3_videomae-base --block 6
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
from steer_multidim import generative_map                           # noqa: E402
from physics_interp.config import load_config                       # noqa: E402
from physics_interp.interp import ActivationSteer, extract_label    # noqa: E402
from physics_interp.runner import VideoDiffusionRunner              # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", nargs="?", default="configs/momentum_probe.yaml")
    p.add_argument("--model", default="videomae-base")
    p.add_argument("--capture-dir", default="outputs_v3_videomae-base")
    p.add_argument("--dataset-dir", default="momentum_dataset/probe_set_v3_latent")
    p.add_argument("--block", type=int, default=6)
    p.add_argument("--read-block", default=None)
    p.add_argument("--alpha", type=float, default=8.0)
    p.add_argument("--n-scenes", type=int, default=40)
    p.add_argument("-o", "--out", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg.model.id = args.model
    out_dir = Path(args.out) if args.out else Path(f"outputs_steer_{args.model}")

    acts, labels, pools, scene_ids = load_captures(Path(args.capture_dir), cfg.name)
    pi = {q: i for i, q in enumerate(pools)}
    blocks = sorted(int(s.split(".")[1]) for s in acts if s.startswith("block."))
    site = f"block.{args.block}.resid_post"
    read_site = args.read_block or f"block.{blocks[-1]}.resid_post"
    tr, te = split_indices(len(labels))
    idx_of = {s: i for i, s in enumerate(scene_ids)}

    X = acts[site][:, pi["A.pre"], :].astype(np.float64)
    v = np.stack([extract_label(l, "velocity_A_pre") for l in labels])
    rA = np.stack([extract_label(l, "radius_A") for l in labels])
    spd = np.linalg.norm(v, axis=1, keepdims=True)
    hdg = v / spd

    unit = lambda a: a / (np.linalg.norm(a) + 1e-12)                # noqa: E731
    Gs = generative_map(X[tr], spd[tr])
    Gh = generative_map(X[tr], hdg[tr])
    Gv = generative_map(X[tr], v[tr])
    rng = np.random.default_rng(0)
    dirs = {
        "gen_speed": unit(Gs[0]),
        "gen_hdg_x": unit(Gh[0]),
        "gen_hdg_y": unit(Gh[1]),
        "gen_v_x": unit(Gv[0]),          # reference: a known Cartesian axis
        "random": unit(rng.normal(size=X.shape[1])),
    }

    readouts = {}
    for target, Yt in (("velocity_A_pre", v), ("radius_A", rA)):
        Xr = acts[read_site][:, pi["A.pre"], :].astype(np.float64)
        r = fit_readout(Xr, Yt, tr)
        r["r2"] = r2_of(r, Xr, Yt, te)
        readouts[target] = r
    print(f"{args.model}: steer {site}, read {read_site}, alpha {args.alpha}")
    print("readout R2: " + "  ".join(f"{k} {r['r2']:.3f}" for k, r in readouts.items()))

    test_ids = [scene_ids[i] for i in te][: args.n_scenes]
    print(f"\nloading {args.model} ...", flush=True)
    runner = VideoDiffusionRunner.from_config(cfg, device="cuda")
    rec_sites = [f"block.{b}.resid_post" for b in blocks]

    rows, t0 = [], time.perf_counter()
    for si, sid in enumerate(test_ids, 1):
        try:
            sc = SceneRunner(runner, Path(args.dataset_dir) / sid,
                             cfg.generation.num_frames)
        except Exception as exc:                                    # noqa: BLE001
            print(f"  [{si}] {sid}: SKIP ({exc})")
            continue
        if sc.n_steer_tokens == 0:
            continue
        base = sc.run(rec_sites)
        b_v = apply_readout(readouts["velocity_A_pre"], base[read_site][pi["A.pre"]])[0]
        b_r = apply_readout(readouts["radius_A"], base[read_site][pi["A.pre"]])[0]
        i = idx_of[sid]
        vhat = hdg[i]
        perp = np.array([-vhat[1], vhat[0]])

        for dname, vec in dirs.items():
            steer = ActivationSteer(vec, alpha=args.alpha, tokens=sc.steer_mask)
            out = sc.run(rec_sites, steer_site=site, steer=steer)
            dv = apply_readout(readouts["velocity_A_pre"],
                               out[read_site][pi["A.pre"]])[0] - b_v
            # Back to physical units before any angle is taken: the readout is
            # returned per-component in training-label sd, and v_x / v_y have
            # different sd, so angles in that space are sheared.
            dv = dv * readouts["velocity_A_pre"]["y_sd"]
            dr = apply_readout(readouts["radius_A"], out[read_site][pi["A.pre"]])[0] - b_r
            n = np.linalg.norm(dv) + 1e-12
            rows.append({
                "scene": sid, "direction": dname,
                "par": float(dv @ vhat / n),        # aligned with own heading
                "perp": float(dv @ perp / n),       # rotates own heading
                "dv_norm": float(n),
                "unit_x": float(dv[0] / n), "unit_y": float(dv[1] / n),
                "d_radius": float(dr[0]),
                "vhat_x": float(vhat[0]), "vhat_y": float(vhat[1]),
            })
        el = time.perf_counter() - t0
        print(f"  [{si}/{len(test_ids)}] {sid}  {el:6.1f}s, "
              f"~{el / (si / len(test_ids)) - el:5.1f}s left", flush=True)
    runner.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"polar_block{args.block}.json"
    path.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2))
    print(f"\nwrote {path}")

    print(f"\n{'direction':<12} {'par':>8} {'|perp|':>8} {'conc':>8} "
          f"{'|dv|':>8} {'d_radius':>9}   par: 1=polar, 0=Cartesian; conc: 1=fixed axis")
    for d in dirs:
        rs = [r for r in rows if r["direction"] == d]
        u = np.array([[r["unit_x"], r["unit_y"]] for r in rs])
        conc = float(np.linalg.norm(u.mean(0)))     # 1 = same direction every scene
        print(f"{d:<12} {np.mean([r['par'] for r in rs]):>8.3f} "
              f"{np.mean([abs(r['perp']) for r in rs]):>8.3f} {conc:>8.3f} "
              f"{np.median([r['dv_norm'] for r in rs]):>8.3f} "
              f"{np.median([abs(r['d_radius']) for r in rs]):>9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
