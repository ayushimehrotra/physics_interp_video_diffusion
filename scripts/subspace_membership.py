#!/usr/bin/env python3
"""§49 — subspace membership: is the decoded outcome a physically possible collision?

    python scripts/subspace_membership.py --captures outputs/conservation_set_v1/*/activations [--fit-honest]

Decode (dv_A, dv_B); score = its distance from the scene's legal line divided
by its length. Scale-free, and no violation label enters at any point.
``--fit-honest`` trains every readout on honest clips only, so the readout has
never seen a violation. ``--segment`` clamps the nearest legal point to
e in [0, 1] instead of the whole line; ``--decoded-pre`` uses decoded masses
for the legal line instead of the labels'.
"""
import _momentum_path  # noqa: F401

import numpy as np

from physics_interp.momentum import cli
from physics_interp.momentum.detectors import choose_site, report, subspace_membership


def main():
    p = cli.base_parser(__doc__)
    p.add_argument("--fit-honest", action="store_true")
    p.add_argument("--segment", action="store_true", help="clamp to e in [0, 1]")
    p.add_argument("--decoded-pre", action="store_true")
    args = p.parse_args()
    sets = cli.load_sets(args)
    out = {"protocol": "§49", "fit_honest": args.fit_honest, "segment": args.segment, "models": {}}
    for name, cs in sets.items():
        out["models"][name] = {}
        for pool in args.pools:
            sites = cs.sites if args.site == "all" else [choose_site(cs, pool, rule=args.site, seed=args.seed)[0]]
            for site in sites:
                sc = subspace_membership(cs, site, pool, seed=args.seed, fit_honest=args.fit_honest,
                                         decoded_pre=args.decoded_pre,
                                         segment=(0.0, 1.0) if args.segment else None)
                rep = report(cs, sc, draws=args.draws, seed=args.seed)
                rep["site"] = site
                v = sc.valid
                rep["score_median_honest"] = float(np.median(sc.score[v & ~cs.violates]))
                rep["score_median_violating"] = float(np.median(sc.score[v & cs.violates]))
                out["models"][name][f"{pool} @ {site}"] = rep
                print(f"\n[{name}] {pool} @ {site}  dv R2 " + " ".join(f"{k}={v:.2f}" for k, v in sc.extra["ingredient_r2"].items()))
                print(f"  pooled {cli.fmt_auc(rep['pooled'])}  elastic {cli.fmt_auc(rep.get('elastic'))}  "
                      f"inelastic {cli.fmt_auc(rep.get('inelastic'))}   median score honest {rep['score_median_honest']:.3f} "
                      f"violating {rep['score_median_violating']:.3f}")
    cli.write_artifact(out, args.out, "subspace_membership" + ("_fit_honest" if args.fit_honest else "") + ".json")


if __name__ == "__main__":
    main()
