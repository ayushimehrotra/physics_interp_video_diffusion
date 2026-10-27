#!/usr/bin/env python
"""Elastic-vs-inelastic probe ACROSS DEPTH, with fold-assignment error bars.

`energy_loss_honest` on `conservation_set_v1`'s honest half -- 75 clips at e = 1.00
(KE ratio 1.000 on all 75) against 75 at e = 0.15 -- reusing `probe_energy.py`'s own
target definition, `scene_facts`, grouped folds and ridge, so the curve is the same
measurement as §44's table.

What is added is a SEED SWEEP. `probe_energy.py` reports one AUC per cell at a single
fold seed; the seed sets `grouped_folds` and therefore the alpha chosen per fold, so a
one-seed depth curve carries resampling wiggle that reads as structure. Each layer is
re-fit under several seeds and reported as mean +- sd. The data, the pairing and the
label are identical across seeds.

    .venv/bin/python scripts/elasticity/probe_elasticity_depth.py --pool all.post --seeds 12
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import sys
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))  # noqa: E401,E702
import _momentum_path  # noqa: F401,E402
from probe_energy import (POOLS, TARGETS, auc, cell_oof_scores,   # noqa: E402
                          load_captures, scene_facts)


def site_depth(site: str) -> int:
    return 0 if site.startswith("embed") else int(re.search(r"block\.(\d+)", site).group(1)) + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", nargs="+",
                    default=["outputs/cons_videomae-base", "outputs/cons_vjepa2-vitl"])
    ap.add_argument("--target", default="energy_loss_honest", choices=list(TARGETS))
    ap.add_argument("--pool", default="all.post")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("-o", "--out", default="docs/artifacts/elasticity_depth.json")
    args = ap.parse_args()
    T = TARGETS[args.target]

    report = {"target": args.target, "pool": args.pool, "n_seeds": args.seeds}
    for outdir in args.outputs:
        acts, labels, pools, _ = load_captures(outdir)   # also returns scene ids
        F = [scene_facts(L) for L in labels]
        sel = np.flatnonzero(T["select"](F))
        y = T["label"]([F[i] for i in sel])
        g = np.array([F[i]["seed"] for i in sel])
        Ys = np.asarray(y, float)[:, None]

        # the mass-free ceiling, for the record
        orc = {}
        for nm in ("e_meas", "rel_speed_post", "speed_sum_post"):
            v = np.array([F[i][nm] for i in sel])
            a = auc(y, v); orc[nm] = max(a, 1 - a)

        sites = sorted(acts, key=site_depth)
        print(f"\n{'=' * 76}\n{outdir}   {args.target}  pool={args.pool}  "
              f"n={len(sel)} ({int(np.sum(y))} inelastic)  "
              f"{len(sites)} layers x {args.seeds} seeds   ceiling {max(orc.values()):.3f}")
        layers = {}
        for site in sites:
            X = acts[site][:, pools.index(args.pool), :][sel]
            vals = []
            for s in range(args.seeds):
                oof = cell_oof_scores(X, Ys, g, s)
                a = auc(y, oof[:, 0])
                vals.append(max(a, 1 - a))
            vals = np.array(vals)
            layers[site] = {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1)),
                            "min": float(vals.min()), "max": float(vals.max()),
                            "values": [float(v) for v in vals]}
            print(f"  {site:<24} {vals.mean():.3f} +- {vals.std(ddof=1):.3f}   "
                  f"[{vals.min():.3f}, {vals.max():.3f}]", flush=True)
        best = max(layers, key=lambda s: layers[s]["mean"])
        report[outdir] = {"n": int(len(sel)), "n_pos": int(np.sum(y)),
                          "mass_free_ceiling": max(orc.values()), "oracles": orc,
                          "layers": layers, "best_site": best,
                          "best_mean": layers[best]["mean"]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
