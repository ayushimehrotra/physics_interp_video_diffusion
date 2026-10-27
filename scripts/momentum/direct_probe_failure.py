#!/usr/bin/env python3
"""§2a diagnosis / §61 — why the direct binary probe cannot work, with the exact answer in hand.

    python scripts/momentum/direct_probe_failure.py --captures outputs/conservation_set_v1/*/activations

Hands a *linear* classifier the exact noiseless dp vector (and the exact
ingredients m_A, m_B, dv_A, dv_B) and scores it: the concentric-classes ceiling.
Then the exact |dp|, which separates perfectly. Then the probe itself, at every
layer: signed and folded AUC, AUC by violation tercile, and the sample-size
curve. A quadratic (degree-2 polynomial) readout of the exact ingredients is the
"ingredient-level oracle" ceiling to quote instead of a figure from a leaky set.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))  # noqa: E401,E702
import _momentum_path  # noqa: F401

import numpy as np

from physics_interp.momentum import cli, physics
from physics_interp.momentum.detectors import direct_binary, ingredient_targets
from physics_interp.momentum.stats import auc, cv_predict, grouped_folds, permutation_null


def linear_oracle(cs, X, seed):
    y = cs.violates.astype(float)[:, None]
    pred = cv_predict(X, y, cs.seed, seed=seed, alphas=(1e-6, 1e-3, 1e-1, 1.0))[:, 0]
    return {"signed": auc(pred, cs.violates), "folded": auc(np.abs(pred - 0.5), cs.violates)}


def quadratic_features(X):
    n, d = X.shape
    cols = [X]
    for i in range(d):
        for j in range(i, d):
            cols.append((X[:, i] * X[:, j])[:, None])
    return np.concatenate(cols, axis=1)


def main():
    p = cli.base_parser(__doc__, site=False)
    args = p.parse_args()
    sets = cli.load_sets(args)
    out = {"protocol": "§2a diagnosis / §61", "models": {}}
    for name, cs in sets.items():
        t = ingredient_targets(cs)
        dp, y = t["dp"], cs.violates
        ing = np.concatenate([t["log_m_a"], t["log_m_b"], t["dv_a"], t["dv_b"]], axis=1)
        rec = {
            "oracle_dp_vector_linear": linear_oracle(cs, dp, args.seed),
            "oracle_ingredients_linear": linear_oracle(cs, ing, args.seed),
            "oracle_ingredients_quadratic": linear_oracle(cs, quadratic_features(ing), args.seed),
            "oracle_dp_norm": {"auc": auc(np.linalg.norm(dp, axis=1), y)},
            "sites": {},
        }
        print(f"\n[{name}] exact dp vector, linear readout: signed {rec['oracle_dp_vector_linear']['signed']:.3f} "
              f"folded {rec['oracle_dp_vector_linear']['folded']:.3f}")
        print(f"[{name}] exact ingredients, linear {rec['oracle_ingredients_linear']['signed']:.3f}; "
              f"quadratic {rec['oracle_ingredients_quadratic']['signed']:.3f}; exact |dp| {rec['oracle_dp_norm']['auc']:.3f}")
        tn = np.linalg.norm(dp, axis=1)
        terc = np.quantile(tn[y], [1 / 3, 2 / 3])
        bins = np.digitize(tn, terc)
        honest = np.flatnonzero(~y)
        for pool in args.pools:
            rec["sites"][pool] = []
            print(f"  pool {pool}")
            for site in cs.sites:
                sc = direct_binary(cs, site, pool, seed=args.seed)
                v = sc.valid
                null = permutation_null(sc.score[v], y[v], draws=args.draws, seed=args.seed)
                by_size = {}
                for b, nm in enumerate(("small", "medium", "large")):
                    sel = np.r_[honest, np.flatnonzero(y & (bins == b))]
                    sel = sel[v[sel]]
                    by_size[nm] = auc(sc.score[sel], y[sel])
                # sample-size curve: refit on the first k seeds
                curve = {}
                for k in (30, 60, 100, 150):
                    seeds = np.unique(cs.seed)[:k]
                    m = np.isin(cs.seed, seeds) & v
                    if m.sum() >= 24:
                        sub = cs.subset(m)
                        s2 = direct_binary(sub, site, pool, seed=args.seed)
                        curve[int(m.sum())] = auc(s2.score[s2.valid], sub.violates[s2.valid])
                row = {"site": site, "signed": null, "folded_auc": auc(np.abs(sc.score[v] - 0.5), y[v]),
                       "by_violation_size": by_size, "sample_size_curve": curve}
                rec["sites"][pool].append(row)
                print(f"    {site:24s} signed {null['auc']:.3f} (p={null['p']:.3f})  folded {row['folded_auc']:.3f}  "
                      f"small/med/large {by_size['small']:.2f}/{by_size['medium']:.2f}/{by_size['large']:.2f}  "
                      f"n-curve {' '.join(f'{k}:{a:.2f}' for k, a in curve.items())}")
        out["models"][name] = rec
    cli.write_artifact(out, args.out, "direct_probe_failure.json")


if __name__ == "__main__":
    main()
