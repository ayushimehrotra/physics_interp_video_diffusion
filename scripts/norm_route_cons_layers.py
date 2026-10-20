#!/usr/bin/env python
"""§34a's norm route ACROSS DEPTH on `conservation_set_v1`, elastic vs inelastic.

`probe_conservation_set.py` reports one paired accuracy per cell, with the layer
chosen inside training. This holds the pool fixed (`all.all` by default) and reports
every layer separately, so the depth profile is visible rather than collapsed to its
argmax.

STATISTIC AND NULL ARE THAT SCRIPT'S, DELIBERATELY, so the curve and the published
table are the same measurement:

* **paired accuracy** -- the fraction of twin pairs whose violating clip outscores its
  own honest twin. The natural statistic here, because the twins share a bit-identical
  pre-contact segment.
* **the exact paired null** -- exchange the twins COMPLETELY (label *and* Delta-p) with
  probability 1/2 per pair. Flipping only the scoring label while stage 1 still
  regresses the true Delta-p leaves the regressor firing on the genuinely violating
  clip, and the null lands BELOW chance; that was measured at 0.37-0.47 and would have
  turned every result into a false positive.

Strata pool `mixed` and `same` material pairs together: 75 elastic pairs (e = 1.0) and
75 inelastic (e = 0.15).

    .venv/bin/python scripts/norm_route_cons_layers.py --pool all.all
"""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import probe_conservation_set as P  # noqa: E402


def site_depth(site: str) -> int:
    return 0 if site.startswith("embed") else int(re.search(r"block\.(\d+)", site).group(1)) + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", nargs="+",
                    default=["outputs_cons_videomae-base", "outputs_cons_vjepa2-vitl"])
    ap.add_argument("--pool", default="all.all")
    ap.add_argument("--n-perm", type=int, default=50)
    ap.add_argument("--stat", default="paired", choices=("paired", "auc"),
                    help="paired accuracy (twin-vs-twin) or unpaired AUC over all clips")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", default="docs/artifacts/norm_route_cons_layers.json")
    args = ap.parse_args()

    report = {"pool": args.pool, "n_perm": args.n_perm, "n_seeds": args.seeds, "stat": args.stat}
    for outdir in args.outputs:
        acts, labels, pools = P.load(outdir)
        F = [P.scene_facts(L) for L in labels]
        viol = np.array([f["viol"] for f in F])
        dp = np.stack([f["dp"] for f in F])
        groups = defaultdict(list)
        for i, f in enumerate(F):
            groups[(f["e"], f["seed"])].append(i)
        report[outdir] = {}
        sites = sorted(acts, key=site_depth)
        print(f"\n{'=' * 78}\n{outdir}   pool={args.pool}   {len(sites)} layers")

        for stratum, e_want in (("elastic", 1.0), ("inelastic", 0.15)):
            pairs_all = [tuple(v) for k, v in sorted(groups.items())
                         if len(v) == 2 and k[0] == e_want]
            sel = np.concatenate([list(p) for p in pairs_all])
            remap = {g: i for i, g in enumerate(sel)}
            pr = [(remap[i], remap[j]) for i, j in pairs_all]
            vsub, dsub = viol[sel], dp[sel]
            print(f"\n  --- {stratum}  ({len(pr)} pairs)", flush=True)

            cells = {site: acts[site][:, pools.index(args.pool), :][sel]
                     for site in sites}

            # ---- observed: every layer re-fit under several fold seeds ----------
            obs = {}
            for site in sites:
                vals = np.array([P.run({f"{site}/{args.pool}": cells[site]}, pr,
                                       vsub, dsub, "norm", np.random.default_rng(s), stat=args.stat)
                                 for s in range(args.seeds)])
                obs[site] = vals
                print(f"    {site:<24} {vals.mean():.3f} +- {vals.std(ddof=1):.3f}"
                      f"   [{vals.min():.3f}, {vals.max():.3f}]", flush=True)

            # ---- null: EVERY layer under each permutation, then the MAX ---------
            # The grid max is what corrects for choosing the best layer. Scoring a
            # per-layer null and reporting p at the argmax is a winner's curse --
            # that is what the first version of this sweep did.
            null_by_layer = {site: [] for site in sites}
            nullmax = []
            for t in range(args.n_perm):
                r = np.random.default_rng(1000 + t)
                vp, dpp = vsub.copy(), dsub.copy()
                for i, j in pr:                       # exchange twins COMPLETELY
                    if r.random() < 0.5:
                        vp[i], vp[j] = vp[j], vp[i]
                        dpp[[i, j]] = dpp[[j, i]]
                row = {site: P.run({f"{site}/{args.pool}": cells[site]}, pr, vp, dpp,
                                   "norm", np.random.default_rng(args.seed), stat=args.stat)
                       for site in sites}
                for site in sites:
                    null_by_layer[site].append(row[site])
                nullmax.append(max(row.values()))
                if (t + 1) % 10 == 0:
                    print(f"      null {t + 1}/{args.n_perm}", flush=True)
            nullmax = np.array(nullmax)

            means = {site: float(obs[site].mean()) for site in sites}
            best = max(means, key=means.get)
            p_corr = float((nullmax >= means[best]).mean())
            layers = {site: {"mean": means[site],
                             "sd": float(obs[site].std(ddof=1)),
                             "min": float(obs[site].min()),
                             "max": float(obs[site].max()),
                             "null_p95": float(np.percentile(null_by_layer[site], 95)),
                             "null_mean": float(np.mean(null_by_layer[site]))}
                      for site in sites}
            print(f"    BEST {means[best]:.3f} at {best}   "
                  f"grid-max null p95 {np.percentile(nullmax, 95):.3f}   "
                  f"p_corrected = {p_corr:.3f}", flush=True)
            report[outdir][stratum] = {
                "n_pairs": len(pr), "layers": layers, "best_site": best,
                "best": means[best], "p_corrected": p_corr,
                "gridmax_null_p95": float(np.percentile(nullmax, 95)),
                "gridmax_null_mean": float(nullmax.mean())}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
