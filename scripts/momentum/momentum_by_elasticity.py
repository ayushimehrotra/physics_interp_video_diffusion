#!/usr/bin/env python3
"""§50 — four detectors on identical features, split by elasticity.

    python scripts/momentum/momentum_by_elasticity.py --captures outputs/conservation_set_v1/*/activations

Same site (chosen label-free on the ingredients), same folds, same features
for direct binary, fitted norm, composed norm and subspace membership, on the
object-local concat pool and the frame-wide all.pre pool. Raw AUC with a
4000-draw label-permutation null per stratum. Paired accuracy is not computed.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))  # noqa: E401,E702
import _momentum_path  # noqa: F401

from physics_interp.momentum import cli
from physics_interp.momentum.detectors import ROUTES, choose_site, report


def main():
    p = cli.base_parser(__doc__)
    args = p.parse_args()
    sets = cli.load_sets(args)
    out = {"protocol": "§50", "statistic": "raw AUC, label-permutation null", "models": {}}
    for name, cs in sets.items():
        out["models"][name] = {}
        for pool in args.pools:
            site, how = choose_site(cs, pool, rule=args.site, seed=args.seed)
            block = {"site": site, "site_choice": how, "routes": {}}
            print(f"\n[{name}] pool {pool} @ {site} ({how['rule']})")
            print(f"  {'route':16s} {'pooled':>22s} {'elastic':>22s} {'inelastic':>22s}")
            for route, fn in ROUTES.items():
                sc = fn(cs, site, pool, seed=args.seed)
                rep = report(cs, sc, draws=args.draws, seed=args.seed)
                block["routes"][route] = rep
                print(f"  {route:16s} {cli.fmt_auc(rep['pooled']):>22s} {cli.fmt_auc(rep.get('elastic')):>22s} "
                      f"{cli.fmt_auc(rep.get('inelastic')):>22s}")
            out["models"][name][pool] = block
    cli.write_artifact(out, args.out, "momentum_by_elasticity.json")


if __name__ == "__main__":
    main()
