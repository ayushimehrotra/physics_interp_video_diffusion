#!/usr/bin/env python3
"""§2g — failure localisation of the norm route, at every layer of every model.

    python scripts/momentum_failure.py --captures outputs/conservation_set_v1/*/activations --pools all.pre

Q1 attainable AUC at the measured dp fidelity (a reference, not a ceiling);
Q2 radial/tangential error geometry and cos(pred, true); Q3 AUC by violation
tercile; Q4 the composed route at the same site. Also the elastic/inelastic AUC
so the depth story can be checked to be momentum, not energy. AUC here is the
folded |pred| statistic (chance ≈ 0.52); see §61 for signed.
"""
import _momentum_path  # noqa: F401

from physics_interp.momentum import cli
from physics_interp.momentum.detectors import failure_localisation


def main():
    p = cli.base_parser(__doc__, site=False)
    p.set_defaults(pools=["all.pre"])
    args = p.parse_args()
    sets = cli.load_sets(args)
    out = {"protocol": "§2g", "models": {}}
    for name, cs in sets.items():
        depth = cs.depth()
        out["models"][name] = {"depth": depth, "pools": {}}
        for pool in args.pools:
            rows = []
            print(f"\n[{name}] pool {pool}")
            print(f"  {'site':24s} {'dp R2':>7s} {'AUC':>6s} {'Q1 exc':>7s} {'radial':>7s} {'tang':>6s} {'cos':>6s} "
                  f"{'small':>6s} {'large':>6s} {'elast':>6s} {'inel':>6s} {'Q4':>6s}")
            for site in cs.sites:
                r = failure_localisation(cs, site, pool, seed=args.seed)
                r["depth"] = depth[site]
                rows.append(r)
                st = r["auc_by_stratum"]
                print(f"  {site:24s} {r['dp_r2']:7.3f} {r['auc']:6.3f} {r['q1']['excess']:+7.3f} "
                      f"{r['q2']['radial_abs']:7.3f} {r['q2']['tangential_abs']:6.3f} {r['q2']['cos']:+6.2f} "
                      f"{r['q3']['small']['auc']:6.2f} {r['q3']['large']['auc']:6.2f} "
                      f"{st.get('elastic', float('nan')):6.2f} {st.get('inelastic', float('nan')):6.2f} "
                      f"{r['q4']['composed_auc']:6.3f}")
            out["models"][name]["pools"][pool] = rows
    cli.write_artifact(out, args.out, "momentum_failure_allpre.json")


if __name__ == "__main__":
    main()
