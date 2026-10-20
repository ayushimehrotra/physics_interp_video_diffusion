#!/usr/bin/env python3
"""§2a / §2c — the direct binary probe and the fitted norm route, every layer, both pool families.

    python scripts/probe_conservation_set_register.py --captures outputs/conservation_set_v1/*/activations

Per model, per pool, per site: a ridge straight to the violation label (score =
its prediction, signed; and folded |pred - 1/2|) and a ridge to the dp vector
scored by |prediction|. Grouped 5-fold by seed, penalty chosen inside the
training split, raw AUC with a label-permutation null, split by elasticity.
The depth curve is the output; the register's "chance through the first third,
rising after" is read off it.
"""
import _momentum_path  # noqa: F401

import numpy as np

from physics_interp.momentum import cli
from physics_interp.momentum.detectors import direct_binary, fitted_norm, report


def main():
    p = cli.base_parser(__doc__, site=False)
    args = p.parse_args()
    sets = cli.load_sets(args)
    out = {"protocol": "§2a/§2c", "statistic": "raw AUC, label-permutation null", "models": {}}
    for name, cs in sets.items():
        depth = cs.depth()
        out["models"][name] = {"sites": cs.sites, "depth": depth, "pools": {}}
        for pool in args.pools:
            rows = []
            print(f"\n[{name}] pool {pool}")
            print(f"  {'site':24s} {'direct(signed)':>22s} {'direct(folded)':>22s} {'fitted|dp|':>22s} {'dp R2':>7s}")
            for site in cs.sites:
                d_s = report(cs, direct_binary(cs, site, pool, seed=args.seed), draws=args.draws, seed=args.seed)
                d_f = report(cs, direct_binary(cs, site, pool, seed=args.seed, folded=True), draws=args.draws, seed=args.seed)
                fn = fitted_norm(cs, site, pool, seed=args.seed)
                f_r = report(cs, fn, draws=args.draws, seed=args.seed)
                rows.append({"site": site, "depth": depth[site], "direct_signed": d_s,
                             "direct_folded": d_f, "fitted_norm": f_r, "dp_r2": fn.extra["dp_r2"]})
                print(f"  {site:24s} {cli.fmt_auc(d_s['pooled']):>22s} {cli.fmt_auc(d_f['pooled']):>22s} "
                      f"{cli.fmt_auc(f_r['pooled']):>22s} {fn.extra['dp_r2']:7.3f}")
            out["models"][name]["pools"][pool] = rows
    cli.write_artifact(out, args.out, "probe_conservation_set.json")


if __name__ == "__main__":
    main()
