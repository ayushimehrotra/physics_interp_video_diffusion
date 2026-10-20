#!/usr/bin/env python
"""Created-vs-conserved probe ACROSS DEPTH, with fold-assignment error bars.

`probe_energy_voe_pairs.py` reports one AUC per cell at a single fold seed. At n = 40
that number carries real variance from the grouped-fold split and the nested alpha
choice, and a depth curve drawn from one seed will show wiggles that are resampling
noise rather than structure. This re-fits every layer under several seeds and reports
mean +- sd across them.

The seed changes `grouped_folds` (which seeds land in which outer/inner fold) and
therefore the alpha selected per fold. It does NOT change the data, the pairing or the
label, so the spread is honest fold-assignment variance and nothing else.

Default pairing is `impossible_vs_anchor` -- energy CREATED (e = k) against energy
CONSERVED (e = 1) -- with momentum held to 3.9e-16 in both arms. That is the only place
in the project where energy creation is isolated from momentum violation.

    .venv/bin/python scripts/probe_energy_voe_depth.py --pool all.pre --seeds 12
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_energy import POOLS, auc, cell_oof_scores  # noqa: E402
from probe_energy_voe import load, facts               # noqa: E402

PAIRINGS = {"impossible_vs_anchor": ("impossible", "anchor"),
            "impossible_vs_legal":  ("impossible", "legal"),
            "legal_vs_anchor":      ("legal", "anchor")}


def site_depth(site: str) -> int:
    return 0 if site.startswith("embed") else int(re.search(r"block\.(\d+)", site).group(1)) + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", nargs="+",
                    default=["outputs_evoe_videomae-base", "outputs_evoe_vjepa2-vitl"])
    ap.add_argument("--pairing", default="impossible_vs_anchor", choices=list(PAIRINGS))
    ap.add_argument("--pool", default="all.pre")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("-o", "--out", default="docs/artifacts/energy_voe_depth.json")
    args = ap.parse_args()
    pos, neg = PAIRINGS[args.pairing]

    report = {"pairing": args.pairing, "pool": args.pool, "n_seeds": args.seeds}
    for outdir in args.outputs:
        acts, labels, pools = load(outdir)
        F = [facts(L) for L in labels]
        cellv = np.array([f["cell"] for f in F])
        seeds_g = np.array([f["seed"] for f in F])
        sel = (cellv == pos) | (cellv == neg)
        y = (cellv[sel] == pos)
        g = seeds_g[sel]
        Ys = y.astype(float)[:, None]                 # observed only; no permutations

        # the mass-free oracle, for the record
        orc = {}
        for nm in ("e_meas", "rel_post", "speed_sum_post"):
            v = np.array([f[nm] for f in F])[sel]
            a = auc(y, v); orc[nm] = max(a, 1 - a)

        sites = sorted(acts, key=site_depth)
        print(f"\n{'=' * 74}\n{outdir}   {args.pairing}  pool={args.pool}  "
              f"n={int(sel.sum())}  {len(sites)} layers x {args.seeds} seeds"
              f"   ceiling {max(orc.values()):.3f}")
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
        report[outdir] = {"n": int(sel.sum()), "mass_free_ceiling": max(orc.values()),
                          "oracles": orc, "layers": layers,
                          "best_site": best, "best_mean": layers[best]["mean"]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
