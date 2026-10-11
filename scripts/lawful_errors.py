#!/usr/bin/env python3
"""§51 — does the model apply conservation? Lawful vs unlawful error recovery.

    python scripts/lawful_errors.py --captures outputs/conservation_set_v1/*/activations

For each twin pair the true displacement between honest and violating outcomes
splits into an on-line (legal) and an off-line (impossible) part. The decoded
displacement is projected onto both; slopes over seeds give how much of each the
readout recovers, and their ratio off/on cancels shrinkage. Below 1 = the model
suppresses the impossible part = it applies the law. The second arm is the same
ratio per stratum.
"""
import _momentum_path  # noqa: F401

from physics_interp.momentum import cli
from physics_interp.momentum.detectors import choose_site, lawful_errors


def main():
    p = cli.base_parser(__doc__)
    p.set_defaults(pools=["A.all+B.all"])
    args = p.parse_args()
    sets = cli.load_sets(args)
    out = {"protocol": "§51", "models": {}}
    for name, cs in sets.items():
        out["models"][name] = {}
        for pool in args.pools:
            sites = cs.sites if args.site == "all" else [choose_site(cs, pool, rule=args.site, seed=args.seed)[0]]
            for site in sites:
                r = lawful_errors(cs, site, pool, seed=args.seed)
                out["models"][name][f"{pool} @ {site}"] = r
                print(f"\n[{name}] {pool} @ {site}  n_pairs={r['n_pairs']}  dv R2 "
                      + " ".join(f"{k}={v:.2f}" for k, v in r["ingredient_r2"].items()))
                print(f"  k_on {r['k_on']:.3f}  k_off {r['k_off']:.3f}  ratio off/on {r['ratio']:.3f} "
                      f"CI95 [{r['ratio_ci95'][0]:.3f}, {r['ratio_ci95'][1]:.3f}]  corr_off {r['corr_off']:.2f}")
                for s, b in r["by_stratum"].items():
                    print(f"  {s:10s} ratio {b['ratio']:.3f} CI95 [{b['ratio_ci95'][0]:.3f}, {b['ratio_ci95'][1]:.3f}] "
                          f"P(ratio<1)={b['p_below_1']:.2f}")
    cli.write_artifact(out, args.out, "lawful_errors.json")


if __name__ == "__main__":
    main()
