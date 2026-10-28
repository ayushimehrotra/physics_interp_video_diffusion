#!/usr/bin/env python3
"""§56 (+§56h) — what shape the elasticity manifold is.

    python scripts/elasticity/elasticity_manifold.py --captures outputs/energy_ladder_v1/*/activations

Per seed, the arms' vectors at one pool are analysed with nothing fitted:
Q1 shape — participation ratio, Spearman(PC1, e), corr(distance, |Δe|), the
off-line residual sqrt(1 - var_PC1) — on all arms and on the legal (e ≤ 1) arms
alone (§56h a). Q2 shared? — cross-seed |cos| of PC1 against the random-direction
reference, and its correlation with the cross-seed cosine of the scenes'
physical legal directions. Q3 boundary — the nearest impossible arm's residual
from a line through the legal arms against a held-out legal arm's. §56h b —
is the PC2 score a smooth function of e: leave-one-out R^2 of a quadratic over
the legal arms, against a permutation null. §56i — the population centroid
curve: centre each seed on its legal arms, average across seeds per e, and ask
whether the centroids survive the averaging (retention vs 1/sqrt(S)) and predict
a held-out seed's residuals (LOSO R^2, raw / line / cubic in e), against an
e-shuffled-within-seed null. Restitution is read from restitution_realised throughout.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))  # noqa: E401,E702
import _momentum_path  # noqa: F401

import numpy as np

from physics_interp.momentum import cli, physics
from physics_interp.momentum.ladders import ladders
from physics_interp.momentum.stats import (participation_ratio, pca, polyfit_curve, eval_curve,
                                           random_cosine_reference, spearman)


def shape_stats(X, e):
    scores, comps, var = pca(X)
    frac = var / var.sum()
    pc1 = scores[:, 0]
    rho = spearman(pc1, e)
    D = np.linalg.norm(X[:, None] - X[None], axis=2)
    dE = np.abs(e[:, None] - e[None])
    iu = np.triu_indices(len(e), 1)
    return {
        "participation_ratio": participation_ratio(var),
        "var_pc1": float(frac[0]), "var_pc12": float(frac[:2].sum()),
        "dims_for_90pct": int(np.searchsorted(np.cumsum(frac), 0.9) + 1),
        "spearman_pc1_e": abs(rho) if np.isfinite(rho) else float("nan"),
        "corr_distance_de": float(np.corrcoef(D[iu], dE[iu])[0, 1]),
        "offline_residual": float(np.sqrt(max(1 - frac[0], 0))),
    }, comps[0], scores


def loo_r2_quadratic(e, y):
    if len(e) < 4:
        return float("nan")
    preds = np.empty_like(y)
    for i in range(len(e)):
        m = np.arange(len(e)) != i
        coef = polyfit_curve(e[m], y[m][:, None], 2)
        preds[i] = eval_curve(coef, e[i])[0, 0]
    ss_res = ((y - preds) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def boundary(X, e):
    """Line through the legal arms minus the top legal one; residuals of that held-out
    arm and of the nearest impossible arm, in units of the legal family's extent."""
    legal = e <= 1 + 1e-9
    imp = ~legal
    if legal.sum() < 3 or imp.sum() == 0:
        return None
    li = np.flatnonzero(legal)
    held = li[-1]
    fit = li[:-1]
    coef = polyfit_curve(e[fit], X[fit], 1)
    extent = float(np.linalg.norm(X[li[-1]] - X[li[0]]))
    res_held = float(np.linalg.norm(X[held] - eval_curve(coef, e[held])[0])) / extent
    near = np.flatnonzero(imp)[0]
    res_imp = float(np.linalg.norm(X[near] - eval_curve(coef, e[near])[0])) / extent
    reach_held = abs(e[held] - e[fit].max())
    reach_imp = abs(e[near] - e[fit].max())
    return {"res_heldout_legal": res_held, "res_nearest_impossible": res_imp,
            "gap": res_imp - res_held, "gap_over_interp_residual": (res_imp - res_held) / max(res_held, 1e-12),
            "reach_heldout": float(reach_held), "reach_impossible": float(reach_imp),
            "e_heldout": float(e[held]), "e_impossible": float(e[near])}


def _population_stats(R, e):
    """R [S, E, d] seed-centred residuals on a shared e grid. Retention: |centroid| over the
    rms per-seed residual, ~1/sqrt(S) if the seeds' e-offsets point in unrelated directions.
    LOSO R^2: a held-out seed's residuals predicted by the other seeds' centroids, raw and
    through a degree-1 and degree-3 curve in e."""
    S = len(R)
    cen = R.mean(0)
    retention = float(np.mean(np.linalg.norm(cen, axis=1) / np.sqrt((R ** 2).sum(2).mean(0))))
    tot = float((R ** 2).sum())
    res = {"raw": 0.0, "line": 0.0, "cubic": 0.0}
    for s in range(S):
        c = (cen * S - R[s]) / (S - 1)
        res["raw"] += float(((R[s] - c) ** 2).sum())
        for name, deg in (("line", 1), ("cubic", 3)):
            res[name] += float(((R[s] - eval_curve(polyfit_curve(e, c, deg), e)) ** 2).sum())
    return retention, {k: 1 - v / tot for k, v in res.items()}


def population_curve(X, lads, valid, rng, draws):
    """§56i — the mentor's centroid design. Legal arms only, so every seed shares one e grid
    and the seed mean is the same mixture of arms; null shuffles e within each seed."""
    # Realised e drifts from the planned arm by up to ~7% in some seeds, so arms are matched by
    # rank within each seed and placed at the median realised e across seeds.
    legal = [L for L in lads if valid[L.idx].all()]
    n = [int((L.e <= 1 + 1e-9).sum()) for L in legal]
    if not n or max(n) < 4:
        return None
    k = max(set(n), key=n.count)
    R, E = [], []
    for L, m in zip(legal, n):
        if m != k:
            continue
        rows = L.idx[np.argsort(L.e)[:k]]
        E.append(np.sort(L.e)[:k])
        Xs = X[rows].astype(np.float64)
        R.append(Xs - Xs.mean(0))
    R = np.stack(R)
    e = np.median(np.stack(E), 0)
    common = [float(x) for x in e]
    ret, r2 = _population_stats(R, e)
    null_ret, null_r2 = [], []
    for _ in range(draws):
        Rp = np.stack([r[rng.permutation(len(e))] for r in R])
        a, b = _population_stats(Rp, e)
        null_ret.append(a)
        null_r2.append(b["cubic"])
    return {"n_seeds": len(R), "e_grid": common, "retention": ret,
            "retention_null_median": float(np.median(null_ret)), "retention_iid_reference": 1 / np.sqrt(len(R)),
            "p_retention": float((1 + np.sum(np.array(null_ret) >= ret)) / (1 + draws)),
            "loso_r2": r2, "loso_r2_cubic_null_median": float(np.median(null_r2)),
            "p_loso_cubic": float((1 + np.sum(np.array(null_r2) >= r2["cubic"])) / (1 + draws))}


def main():
    p = cli.base_parser(__doc__, site=False)
    p.set_defaults(pools=["all.post"])
    p.add_argument("--null-draws", type=int, default=200)
    args = p.parse_args()
    sets = cli.load_sets(args)
    out = {"protocol": "§56/§56h", "models": {}}
    rng = np.random.default_rng(args.seed)
    for name, cs in sets.items():
        lads = ladders(cs, min_arms=4)
        depth = cs.depth()
        dim = next(iter(cs.acts.values())).shape[-1]
        rec = {"n_seeds": len(lads), "pools": {}}
        print(f"\n[{name}] {len(lads)} ladders, dim {dim}")
        for pool in args.pools:
            ref = random_cosine_reference(dim * len(cs.resolve_pool(pool)), seed=args.seed)
            rows = []
            print(f"  pool {pool}; random |cos| reference {ref:.4f}")
            print(f"  {'site':24s} {'PR all':>6s} {'PR leg':>6s} {'rho':>5s} {'offline':>7s} {'xseed|cos|':>10s} "
                  f"{'x/null':>6s} {'phys corr':>9s} {'Q3 gap':>7s} {'PC2 LOO R2':>10s} {'>null':>5s}")
            for site in cs.sites:
                X, valid = cs.features(site, pool)
                per_all, per_leg, dirs, phys_dirs, q3s, loo, loo_null = [], [], [], [], [], [], []
                for L in lads:
                    if not valid[L.idx].all():
                        continue
                    Xs, e = X[L.idx], L.e
                    s_all, pc1, _ = shape_stats(Xs, e)
                    per_all.append(s_all)
                    dirs.append(pc1)
                    b = physics.legal_direction(L.pre)
                    phys_dirs.append(b / np.linalg.norm(b))
                    legal = e <= 1 + 1e-9
                    if legal.sum() >= 3:
                        s_leg, _, scores = shape_stats(Xs[legal], e[legal])
                        per_leg.append(s_leg)
                        if legal.sum() >= 4:
                            pc2 = scores[:, 1]
                            loo.append(loo_r2_quadratic(e[legal], pc2))
                            loo_null.append(np.mean([loo_r2_quadratic(e[legal], pc2[rng.permutation(legal.sum())])
                                                     for _ in range(max(args.null_draws // 10, 5))]))
                    q = boundary(Xs, e)
                    if q:
                        q3s.append(q)
                if not per_all:
                    continue
                med = lambda key, rows_: float(np.median([r[key] for r in rows_])) if rows_ else float("nan")
                dirs = np.stack(dirs)
                phys_dirs = np.stack(phys_dirs)
                iu = np.triu_indices(len(dirs), 1)
                cos_enc = np.abs(dirs @ dirs.T)[iu]
                cos_phys = (phys_dirs @ phys_dirs.T)[iu]
                row = {
                    "site": site, "depth": depth[site], "n_seeds": len(per_all),
                    "all_arms": {k: med(k, per_all) for k in per_all[0]},
                    "legal_arms": {k: med(k, per_leg) for k in per_leg[0]} if per_leg else None,
                    "cross_seed_abs_cos": float(cos_enc.mean()), "random_reference": ref,
                    "cross_seed_over_null": float(cos_enc.mean() / ref),
                    "corr_with_physical_direction_cos": float(np.corrcoef(cos_enc, np.abs(cos_phys))[0, 1]),
                    "q3_boundary": {k: med(k, q3s) for k in q3s[0]} if q3s else None,
                    "pc2_loo_r2_median": float(np.median(loo)) if loo else float("nan"),
                    "pc2_loo_r2_null_median": float(np.median(loo_null)) if loo_null else float("nan"),
                    "pc2_loo_frac_seeds_positive": float(np.mean(np.array(loo) > 0)) if loo else float("nan"),
                    "pc2_loo_frac_seeds_above_null": float(np.mean(np.array(loo) > np.array(loo_null))) if loo else float("nan"),
                    "population": population_curve(X, lads, valid, rng, max(args.null_draws // 2, 20)),
                }
                rows.append(row)
                pop = row["population"] or {}
                la = row["legal_arms"] or {}
                print(f"  {site:24s} {row['all_arms']['participation_ratio']:6.2f} {la.get('participation_ratio', float('nan')):6.2f} "
                      f"{row['all_arms']['spearman_pc1_e']:5.2f} {row['all_arms']['offline_residual']:7.2f} "
                      f"{row['cross_seed_abs_cos']:10.3f} {row['cross_seed_over_null']:6.1f} "
                      f"{row['corr_with_physical_direction_cos']:+9.2f} "
                      f"{(row['q3_boundary'] or {}).get('gap', float('nan')):+7.2f} "
                      f"{row['pc2_loo_r2_median']:10.2f} {row['pc2_loo_frac_seeds_above_null']:5.2f}"
                      + (f"  | pop ret {pop['retention']:.3f} (null {pop['retention_null_median']:.3f}, p {pop['p_retention']:.3f})"
                         f" LOSO R2 raw/line/cubic {pop['loso_r2']['raw']:+.3f}/{pop['loso_r2']['line']:+.3f}/"
                         f"{pop['loso_r2']['cubic']:+.3f} (null {pop['loso_r2_cubic_null_median']:+.3f}, p {pop['p_loso_cubic']:.3f})"
                         if pop else ""))
            rec["pools"][pool] = rows
        out["models"][name] = rec
    cli.write_artifact(out, args.out, "elasticity_manifold.json")


if __name__ == "__main__":
    main()
