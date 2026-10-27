#!/usr/bin/env python
"""FRICTION SWEEP: mu, acceleration and theta across depth on `incline_set_v2`.

USE v2, NEVER v1. §32b voided `incline_set_v1`: mu reads off a SINGLE FRAME's cube
position at R2 0.88-0.99, against the 0.459 the set exists to impose. v2 is the fix
(mask leak 0.988 -> 0.066). The captures in `outputs/incline_*` carry `gravity = 0.45`,
which identifies them as v2 -- v1 was rendered at 0.7.

Three targets, because §35's point is the relation between them:

    theta   the incline angle. Static scene geometry, visible in frame 0.
    a       acceleration down the slope (`accel_measured`). A kinematic quantity.
    mu      the friction coefficient. NOT independently visible -- it is recovered
            from a and theta, and over the sampled theta range mu is 0.997 linear in
            (a, theta), which is why §35b calls the direct-vs-composed tie an
            algebraic identity rather than a measurement.

ERROR BARS ARE FOLD-ASSIGNMENT VARIANCE: each layer is re-fit under several seeds,
which changes the fold split and the nested alpha but nothing about the data. Scored by
R2 out-of-fold, `probe_energy.cell_oof_regression`'s protocol.

    .venv/bin/python old_experiments/scripts/probe_incline_depth.py --pool all.all --seeds 12
"""
from __future__ import annotations
import argparse, glob, json, re
from pathlib import Path
import numpy as np
import sys
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))  # noqa: E401,E702
import _momentum_path  # noqa: F401,E402
from probe_energy import POOLS, cell_oof_regression, r2  # noqa: E402

TARGETS = {"mu": "mu", "acceleration": "accel_measured", "theta": "theta"}


def site_depth(site: str) -> int:
    return 0 if site.startswith("embed") else int(re.search(r"block\.(\d+)", site).group(1)) + 1


def load(outdir: str, dataset_dir: str):
    paths = sorted(glob.glob(f"{outdir}/momentum_probe/activations/*.npz"))
    if not paths:
        raise FileNotFoundError(outdir)
    store, fr, pools, sites = {}, [], None, None
    for p in paths:
        sid = Path(p).stem
        z = np.load(p, allow_pickle=True)
        m = json.loads(bytes(z["meta"]).decode())
        pools = m["pools"]
        fr.append(json.load(open(f"{dataset_dir}/{sid}/labels.json"))["friction"])
        if sites is None:
            sites = [k[4:] for k in z.files if k.startswith("act.")]
        for s in sites:
            store.setdefault(s, []).append(z["act." + s][0])
    return {s: np.stack(v).astype(np.float64) for s, v in store.items()}, fr, pools


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", nargs="+",
                    default=["outputs/incline_videomae-base", "outputs/incline_vjepa2-vitl"])
    ap.add_argument("--dataset-dir", default="datasets/incline_set_v2")
    ap.add_argument("--pool", default="all.all")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("-o", "--out", default="docs/artifacts/incline_depth.json")
    args = ap.parse_args()

    report = {"pool": args.pool, "n_seeds": args.seeds, "dataset": args.dataset_dir,
              "targets": list(TARGETS)}
    for outdir in args.outputs:
        acts, fr, pools = load(outdir, args.dataset_dir)
        n = len(fr)
        g = np.arange(n)                     # one scene per seed; no twins to group
        assert abs(fr[0]["gravity"] - 0.45) < 1e-9, "not incline_set_v2 (gravity != 0.45)"
        sites = sorted(acts, key=site_depth)
        report[outdir] = {"n": n, "targets": {}}
        print(f"\n{'=' * 78}\n{outdir}   pool={args.pool}  n={n}  "
              f"{len(sites)} layers x {args.seeds} seeds")
        # how correlated the three targets are, which is §35b's point
        M = {t: np.array([f[k] for f in fr]) for t, k in TARGETS.items()}
        print("   target correlations:  " + "  ".join(
            f"{a}~{b} {np.corrcoef(M[a], M[b])[0,1]:+.3f}"
            for a, b in (("mu", "acceleration"), ("mu", "theta"),
                         ("acceleration", "theta"))))
        for tname, key in TARGETS.items():
            y = np.array([f[key] for f in fr], float)[:, None]
            layers = {}
            for site in sites:
                X = acts[site][:, pools.index(args.pool), :]
                vals = [r2(y, cell_oof_regression(X, y, g, s)) for s in range(args.seeds)]
                vals = np.array(vals)
                layers[site] = {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1)),
                                "min": float(vals.min()), "max": float(vals.max())}
            best = max(layers, key=lambda s: layers[s]["mean"])
            report[outdir]["targets"][tname] = {"layers": layers, "best_site": best,
                                                "best_mean": layers[best]["mean"]}
            print(f"   {tname:13s} best {layers[best]['mean']:+.3f} "
                  f"+- {layers[best]['sd']:.3f} at {best:<22} "
                  f"embed.out {layers['embed.out']['mean']:+.3f}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
