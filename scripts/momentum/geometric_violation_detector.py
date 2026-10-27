#!/usr/bin/env python3
"""§48 — the composed norm: nothing fitted to the label.

    python scripts/momentum/geometric_violation_detector.py --captures outputs/conservation_set_v1/*/activations

Decode log m_A, log m_B, dv_A and dv_B on their own visible targets, then do
the momentum arithmetic outside the probe: |m_A dv_A + m_B dv_B|. The site is
chosen on the ingredients' R^2, never the label. Honest clips have true dp = 0,
so whatever the detector reports on them is its own noise floor.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))  # noqa: E401,E702
import _momentum_path  # noqa: F401

from physics_interp.momentum import cli
from physics_interp.momentum.detectors import choose_site, composed_norm, fitted_norm, report


def main():
    p = cli.base_parser(__doc__)
    args = p.parse_args()
    sets = cli.load_sets(args)
    out = {"protocol": "§48", "models": {}}
    for name, cs in sets.items():
        out["models"][name] = {}
        for pool in args.pools:
            sites = cs.sites if args.site == "all" else [choose_site(cs, pool, rule=args.site, seed=args.seed)[0]]
            for site in sites:
                cn = composed_norm(cs, site, pool, seed=args.seed)
                rep = report(cs, cn, draws=args.draws, seed=args.seed)
                fn = report(cs, fitted_norm(cs, site, pool, seed=args.seed), draws=args.draws, seed=args.seed)
                rep["fitted_norm_same_features"] = fn
                rep["site"] = site
                out["models"][name][f"{pool} @ {site}"] = rep
                e = cn.extra
                print(f"\n[{name}] {pool} @ {site}  ingredients R2 "
                      + " ".join(f"{k}={v:.2f}" for k, v in e["ingredient_r2"].items()))
                print(f"  composed  pooled {cli.fmt_auc(rep['pooled'])}  elastic {cli.fmt_auc(rep.get('elastic'))}  "
                      f"inelastic {cli.fmt_auc(rep.get('inelastic'))}")
                print(f"  fitted    pooled {cli.fmt_auc(fn['pooled'])}  (same features)")
                print(f"  self-calibration: honest median |dp_hat| {e['noise_floor_honest']:.3f} vs violating {e['signal_violating']:.3f}")
    cli.write_artifact(out, args.out, "geometric_violation_detector.json")


if __name__ == "__main__":
    main()
