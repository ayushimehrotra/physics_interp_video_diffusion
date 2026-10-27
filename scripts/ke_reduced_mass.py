#!/usr/bin/env python3
"""Kinetic-energy readout with the REDUCED MASS varied — `ke_factor_set`.

    python scripts/ke_reduced_mass.py --model vjepa2-vitl \
        --captures outputs_ke_vjepa2-vitl/momentum_probe/activations

The set (momentum_dataset/plan_ke_set.py) steps KE through five levels,
x {1/2, 1/sqrt2, 1, sqrt2, 2}, by one route per family:

    M  both masses x c, velocities fixed  ->  mu, M, KE, K_rel = 1/2 mu v_rel^2 all x c
                                              and the motion is the SAME video at every level
    S  both speeds x s, masses fixed      ->  KE, K_rel x s^2, mu fixed

So within an M group the reduced mass is the only thing that moves, and the
only way to follow KE there is to read mass off appearance (material and size
together; radius is decorrelated from the mass level by construction).

1. WITHIN-FAMILY. Grouped CV of log KE inside each family; the within-group
   slope of predicted on true log KE says how much of the level-to-level change
   the readout follows. Within M that is the reduced-mass-only readout.
2. CROSS-FAMILY, the calibrated test the set was built for. Fit log KE on one
   family, apply it unchanged to the other, read the within-group slope:

        train S -> apply M:  1 energy code   2 momentum code   0 speed code
        train M -> apply S:  1 energy code   0.5 momentum      0 mass code

   because within group dlog|p| = dlogKE/2 in S and = dlogKE in M. The cell is
   chosen on the SOURCE family only, so the target contributes nothing to
   selection. The slope is read only from the ROUTE-CALIBRATED readout (trained
   on group-demeaned source data, so it sees speed alone in S and mass alone in
   M), against a same-route split-half ceiling. The raw-target transfer is
   reported but is NOT the test: S groups sit at five mass levels and M groups at
   five speed levels, so the source's between-group variation already carries
   the other route, and log K_rel = log mu + 2 log v_rel - log 2 is linear in the
   factors — slope 1 follows from decodable factors alone (§34b). The composed
   row (factors transferred separately, then added) measures that identity.
3. GATES. The ingredients must cross the same gap: a mass readout trained on S
   (mass varies only BETWEEN S groups) must follow mass within M groups, and a
   speed readout trained on M must follow speed within S groups.
4. APPEARANCE. Within M the level also flips material for some groups
   (corr(#metal, level) = 0.77); the within-group slope is re-estimated with the
   material count as a covariate, so a readout keyed to "metal or rubber" cannot
   pass as graded mass. Radius is NOT a covariate: it is decorrelated from the
   level by design (corr 0.04), and with both material and radius held fixed the
   mass is determined (m = rho r^3), so the partial slope would rest on residual
   noise. The covariates' own R^2 on the target is reported beside it.

Nulls permute the levels within each group (the predictions are fixed), so a
group's own offset can never be the signal. CIs bootstrap over groups.
CPU only, reads captures.
"""
import _momentum_path  # noqa: F401

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from _fast_cv import cv_predict, fit_full
from physics_interp.momentum import cli
from physics_interp.momentum.captures import load_clipset
from physics_interp.momentum.stats import r2

POOLS = ["A.pre", "A.post", "A.all", "B.pre", "B.post", "B.all", "all.pre", "all.post", "all.all", "A.all+B.all"]
CODES = {"S->M": {"energy": 1.0, "momentum": 2.0, "speed": 0.0},
         "M->S": {"energy": 1.0, "momentum": 0.5, "mass": 0.0}}

_G: dict = {}


def xy(v):
    return np.asarray(v, dtype=np.float64)[:2]


def targets(cs) -> dict[str, np.ndarray]:
    T = {k: [] for k in ("family", "group", "level", "log_ke", "log_ke_rel", "log_mu", "log_mass", "log_vrel",
                         "log_p", "n_metal", "log_radius")}
    for l in cs.labels:
        d, c, o = l["ke_design"], l["collision"]["pre"], l["objects"]
        ma, mb = o["A"]["mass"], o["B"]["mass"]
        mu = ma * mb / (ma + mb)
        vrel = np.linalg.norm(xy(c["velocity_A"]) - xy(c["velocity_B"]))
        T["family"].append(d["family"]); T["group"].append(d["group"]); T["level"].append(d["level"])
        T["log_ke"].append(np.log(c["kinetic_energy"]))
        T["log_ke_rel"].append(np.log(0.5 * mu * vrel ** 2))
        T["log_mu"].append(np.log(mu)); T["log_mass"].append(np.log(ma + mb)); T["log_vrel"].append(np.log(vrel))
        T["log_p"].append(np.log(np.linalg.norm(xy(c["total_momentum"]))))
        T["n_metal"].append(float((o["A"]["material"] == "metal") + (o["B"]["material"] == "metal")))
        T["log_radius"].append(np.log(0.5 * (o["A"]["radius"] + o["B"]["radius"])))
    return {k: np.array(v) for k, v in T.items()}


def design_control(T) -> dict:
    """The set's promises, checked on the labels: within an M group KE moves with mu
    and nothing kinematic moves; within an S group KE moves with v_rel^2 and mu is fixed."""
    out = {}
    for fam in ("M", "S"):
        m = T["family"] == fam
        dev = {k: within(T[k][m], T["group"][m]) for k in ("log_ke", "log_ke_rel", "log_mu", "log_vrel", "log_p")}
        ke = dev["log_ke"]
        out[fam] = {
            "slope_log_mu_on_log_ke": slope(dev["log_mu"], ke),
            "slope_log_vrel_on_log_ke": slope(dev["log_vrel"], ke),
            "slope_log_p_on_log_ke": slope(dev["log_p"], ke),
            "max_abs_log_ke_rel_minus_log_ke_within": float(np.max(np.abs(dev["log_ke_rel"] - ke))),
            "max_abs_identity_log_ke_rel": float(np.max(np.abs(
                T["log_ke_rel"][m] - (T["log_mu"][m] + 2 * T["log_vrel"][m] - np.log(2.0))))),
            "corr_within_log_radius_level": corr(within(T["log_radius"][m], T["group"][m]), ke),
            "corr_within_n_metal_level": corr(within(T["n_metal"][m], T["group"][m]), ke),
        }
    return out


def corr(a, b):
    """Pearson r, NaN (quietly) when either side is constant — material never changes within an S group."""
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------------- #
# within-group statistics
# --------------------------------------------------------------------------- #


def within(x, g):
    x = np.asarray(x, dtype=np.float64).copy()
    for gi in np.unique(g):
        m = g == gi
        x[m] -= x[m].mean(0)
    return x


def slope(pred_dev, true_dev):
    return float(np.sum(pred_dev * true_dev) / np.sum(true_dev ** 2))


def within_stats(pred, true, g, *, covars=None, draws=2000, boots=2000, seed=0) -> dict:
    """Pooled within-group slope of pred on true (optionally partialling covariates),
    within-group correlation, a level-permutation null and a group bootstrap."""
    pred, true = np.ravel(pred), np.ravel(true)
    dp, dt = within(pred, g), within(true, g)
    C = None if covars is None else within(np.column_stack(covars), g)

    def fit(dp_, dt_, C_):
        if C_ is None:
            return slope(dp_, dt_)
        A = np.column_stack([dt_, C_])
        live = A.std(0) > 1e-12
        coef = np.linalg.lstsq(A[:, live], dp_, rcond=None)[0]
        return float(coef[0])

    obs = fit(dp, dt, C)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(g == gi) for gi in np.unique(g)]
    null = np.empty(draws)
    for k in range(draws):
        perm = np.empty(len(true), int)
        for idx in groups:
            perm[idx] = rng.permutation(idx)
        null[k] = fit(dp, dt[perm], None if C is None else C[perm])
    boot = np.empty(boots)
    for k in range(boots):
        pick = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        boot[k] = fit(dp[pick], dt[pick], None if C is None else C[pick])
    return {"slope": obs, "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "p_slope_gt_0": float((1 + np.sum(null >= obs)) / (draws + 1)),
            "null_sd": float(null.std()), "corr_within": float(np.corrcoef(dp, dt)[0, 1]),
            "n_groups": len(groups),
            # how much of the within-group target the covariates already explain: near 1 means
            # the partial slope rests on a sliver of residual variance and is not interpretable
            "covariate_r2_on_target": (None if C is None else
                                       float(1 - np.sum((dt - C @ np.linalg.lstsq(C, dt, rcond=None)[0]) ** 2)
                                             / np.sum(dt ** 2)))}


# --------------------------------------------------------------------------- #
# grid
# --------------------------------------------------------------------------- #


def _cell(args):
    site, pool, fam = args
    cs, T = _G["cs"], _G["T"]
    m = T["family"] == fam
    X, valid = cs.features(site, pool)
    rows = np.flatnonzero(m & valid)
    pred = cv_predict(X[rows], T["log_ke"][rows], T["group"][rows])[:, 0]
    return site, pool, fam, rows, pred


def grid(cs, T, workers):
    _G.update(cs=cs, T=T)
    jobs = [(s, p, f) for s in cs.sites for p in POOLS for f in ("S", "M")]
    with Pool(workers) as pool:
        res = pool.map(_cell, jobs, chunksize=1)
    out = {}
    for site, p, fam, rows, pred in res:
        y, g = T["log_ke"][rows], T["group"][rows]
        out[(site, p, fam)] = {"r2": r2(y[:, None], pred[:, None]),
                               "within_slope": slope(within(pred, g), within(y, g)),
                               "rows": rows, "pred": pred}
    return out


def transfer_pred(cs, T, site, pool, s_rows, d_rows, target, *, demean=False):
    """Fit ``target`` on ``s_rows`` at one cell; predict ``d_rows`` unchanged."""
    X, _ = cs.features(site, pool)
    y = T[target]
    if demean:
        # calibrate on within-group variation only: demean features and target inside each source group
        Xd = X.copy(); yd = y.astype(np.float64).copy()
        Xd[s_rows] = within(X[s_rows], T["group"][s_rows]); yd[s_rows] = within(y[s_rows], T["group"][s_rows])
        model, a = fit_full(Xd, yd, T["group"], s_rows)
    else:
        model, a = fit_full(X, y, T["group"], s_rows)
    return model.predict(X, d_rows, [a])[0][:, 0]


def family_rows(cs, T, site, pool, fam):
    _, valid = cs.features(site, pool)
    return np.flatnonzero((T["family"] == fam) & valid)


def transfer(cs, T, site, pool, src, dst, target, *, demean=False, eval_target=None, covars=None):
    """Fit ``target`` on family ``src`` at one cell, apply unchanged to ``dst``."""
    s_rows, d_rows = family_rows(cs, T, site, pool, src), family_rows(cs, T, site, pool, dst)
    pred = transfer_pred(cs, T, site, pool, s_rows, d_rows, target, demean=demean)
    true = T[eval_target or target][d_rows]
    cv = None if covars is None else [T[c][d_rows] for c in covars]
    return within_stats(pred, true, T["group"][d_rows], covars=cv)


def split_half_ceiling(cs, T, site, pool, fam, target, *, seed=0):
    """Demeaned transfer between two disjoint halves of the SAME family's groups:
    what a route-calibrated readout scores when the route does not change."""
    rows = family_rows(cs, T, site, pool, fam)
    groups = np.unique(T["group"][rows])
    rng = np.random.default_rng(seed)
    half = set(rng.permutation(groups)[: len(groups) // 2])
    a = rows[np.isin(T["group"][rows], list(half))]
    b = rows[~np.isin(T["group"][rows], list(half))]
    pa = transfer_pred(cs, T, site, pool, a, b, target, demean=True)
    pb = transfer_pred(cs, T, site, pool, b, a, target, demean=True)
    pred = np.concatenate([pa, pb]); idx = np.concatenate([b, a])
    return within_stats(pred, T[target][idx], T["group"][idx])


def _boot_one(args):
    site, pool, src, dst, b = args
    cs, T = _G["cs"], _G["T"]
    rng = np.random.default_rng(1000 + b)
    s_rows, d_rows = family_rows(cs, T, site, pool, src), family_rows(cs, T, site, pool, dst)
    groups = np.unique(T["group"][s_rows])
    pick = rng.choice(groups, len(groups), replace=True)
    # a group drawn twice becomes two groups to the folds, so its copies never straddle a split
    members = [s_rows[T["group"][s_rows] == gname] for gname in pick]
    rows = np.concatenate(members)
    X, _ = cs.features(site, pool)
    Xb = np.concatenate([X[rows], X[d_rows]])
    gb = np.concatenate([np.repeat([f"b{j}" for j in range(len(pick))], [len(m) for m in members]),
                         T["group"][d_rows]])
    y = T["log_ke"]
    yb = np.concatenate([y[rows], y[d_rows]])
    src_idx = np.arange(len(rows))
    Xd = Xb.copy(); yd = yb.astype(np.float64).copy()
    Xd[src_idx] = within(Xb[src_idx], gb[src_idx]); yd[src_idx] = within(yb[src_idx], gb[src_idx])
    model, a = fit_full(Xd, yd, gb, src_idx)
    dst_idx = np.arange(len(rows), len(rows) + len(d_rows))
    pred = model.predict(Xb, dst_idx, [a])[0][:, 0]
    return slope(within(pred, gb[dst_idx]), within(yb[dst_idx], gb[dst_idx]))


def source_bootstrap(cs, T, site, pool, src, dst, *, draws=100, workers=32):
    """Route-calibrated slope with the SOURCE readout refit on bootstrap-resampled source
    groups: the variability a target-only bootstrap cannot see."""
    with Pool(workers) as pool_:
        vals = np.array(pool_.map(_boot_one, [(site, pool, src, dst, b) for b in range(draws)]))
    return {"median": float(np.median(vals)), "ci95": [float(np.quantile(vals, 0.025)),
                                                       float(np.quantile(vals, 0.975))], "draws": draws}


def composed_transfer(cs, T, site, pool, src, dst):
    """log K_rel = log mu + 2 log v_rel - log 2 EXACTLY, so a raw-target transfer of log K_rel
    is what decoding the two factors and adding them gives, whatever the model holds (§34b).
    Transfer the factors separately and compose: if this matches the direct transfer, the
    direct slope is that identity, not an energy code."""
    s_rows, d_rows = family_rows(cs, T, site, pool, src), family_rows(cs, T, site, pool, dst)
    pred = (transfer_pred(cs, T, site, pool, s_rows, d_rows, "log_mu")
            + 2.0 * transfer_pred(cs, T, site, pool, s_rows, d_rows, "log_vrel") - np.log(2.0))
    return within_stats(pred, T["log_ke_rel"][d_rows], T["group"][d_rows])


def nearest_code(direction, s):
    codes = CODES[direction]
    return min(codes, key=lambda c: abs(codes[c] - s))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--captures", required=True, type=Path)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    cs = load_clipset(args.captures)
    T = targets(cs)
    print(f"[{args.model}] {cs.describe()}", flush=True)
    ctrl = design_control(T)
    for fam, c in ctrl.items():
        print(f"  design {fam}: dlog mu/dlogKE {c['slope_log_mu_on_log_ke']:.3f}  dlog v_rel/dlogKE "
              f"{c['slope_log_vrel_on_log_ke']:.3f}  dlog|p|/dlogKE {c['slope_log_p_on_log_ke']:.3f}  "
              f"corr(radius, level) {c['corr_within_log_radius_level']:+.3f}  corr(#metal, level) "
              f"{c['corr_within_n_metal_level']:+.3f}", flush=True)

    G = grid(cs, T, args.workers)
    out = {"protocol": "KE readout with the reduced mass varied (ke_factor_set)", "model": args.model,
           "captures": str(args.captures), "n": cs.n, "design_control": ctrl, "within_family": {},
           "transfer": {}, "gates": {}, "appearance": {}}

    best = {}
    for fam in ("S", "M"):
        cells = {k: v for k, v in G.items() if k[2] == fam}
        b = max(cells, key=lambda k: cells[k]["r2"])
        best[fam] = b
        v = cells[b]
        g = T["group"][v["rows"]]
        ws = within_stats(v["pred"], T["log_ke"][v["rows"]], g)
        out["within_family"][fam] = {
            "best_cell": f"{b[1]} @ {b[0]}", "r2": v["r2"], "within": ws,
            "embed_best_r2": max(G[(cs.sites[0], pp, fam)]["r2"] for pp in POOLS),
            "depth_curve_best_pool": {s: {"r2": G[(s, b[1], fam)]["r2"], "within_slope": G[(s, b[1], fam)]["within_slope"]}
                                      for s in cs.sites},
            "best_r2_by_pool": {pp: max(G[(s, pp, fam)]["r2"] for s in cs.sites) for pp in POOLS},
        }
        print(f"  within {fam}: best {b[1]} @ {b[0]}  R2 {v['r2']:.3f}  within-group slope {ws['slope']:.3f} "
              f"[{ws['ci95'][0]:.3f}, {ws['ci95'][1]:.3f}] corr {ws['corr_within']:.3f} p {ws['p_slope_gt_0']:.4f}", flush=True)

    # the reduced-mass-only readout, at the cell chosen on S (M never selects its own cell)
    sS = best["S"]
    X, valid = cs.features(sS[0], sS[1])
    mM = np.flatnonzero((T["family"] == "M") & valid)
    gM = T["group"][mM]
    predM = cv_predict(X[mM], T["log_ke"][mM], gM)[:, 0]
    out["reduced_mass_readout_at_S_cell"] = {
        "cell": f"{sS[1]} @ {sS[0]}",
        "within": within_stats(predM, T["log_ke"][mM], gM),
        "within_partial_appearance": within_stats(predM, T["log_ke"][mM], gM,
                                                  covars=[T["n_metal"][mM]]),
        "mu_readout_within": within_stats(cv_predict(X[mM], T["log_mu"][mM], gM)[:, 0], T["log_mu"][mM], gM),
    }
    rm = out["reduced_mass_readout_at_S_cell"]
    print(f"  reduced-mass readout (M, cell from S): slope {rm['within']['slope']:.3f} "
          f"[{rm['within']['ci95'][0]:.3f}, {rm['within']['ci95'][1]:.3f}] p {rm['within']['p_slope_gt_0']:.4f};  "
          f"partial on appearance {rm['within_partial_appearance']['slope']:.3f} "
          f"[{rm['within_partial_appearance']['ci95'][0]:.3f}, {rm['within_partial_appearance']['ci95'][1]:.3f}];  "
          f"log mu itself {rm['mu_readout_within']['slope']:.3f}", flush=True)

    # Two calibration conventions, and only one of them is a test.
    #   raw       the source family's BETWEEN-group variation already carries the other route
    #             (S groups sit at five mass levels, M groups at five speed levels), and log KE
    #             is linear in (log m, log v): slope 1 is guaranteed by the factors alone. The
    #             composed row below measures exactly that.
    #   demeaned  calibrated on the source family's own route only (speed in S, mass in M).
    #             A readout that holds KE as ONE direction follows the other route at slope 1;
    #             separate factor codes give ~0. Read against the same-route split-half ceiling.
    for direction, (src, dst) in {"S->M": ("S", "M"), "M->S": ("M", "S")}.items():
        site, pool, _ = best[src]
        rec = {"cell_chosen_on_source": f"{pool} @ {site}"}
        for conv, dm in (("raw", False), ("group_demeaned", True)):
            for tgt in ("log_ke", "log_ke_rel"):
                rec[f"{tgt}/{conv}"] = transfer(cs, T, site, pool, src, dst, tgt, demean=dm)
        rec["log_ke_rel/raw_composed_from_factors"] = composed_transfer(cs, T, site, pool, src, dst)
        rec["log_ke/group_demeaned_partial_appearance"] = transfer(
            cs, T, site, pool, src, dst, "log_ke", demean=True,
            covars=["n_metal"] if dst == "M" else None)
        rec["ceiling_same_route_split_half"] = split_half_ceiling(cs, T, site, pool, src, "log_ke")
        d, c = rec["log_ke/group_demeaned"], rec["ceiling_same_route_split_half"]
        rec["demeaned_fraction_of_ceiling"] = float(d["slope"] / c["slope"])
        # robustness: the route-calibrated slope at every block of the chosen pool, so the
        # verdict is not one cell's
        curve = {}
        for s in cs.sites:
            sr, dr = family_rows(cs, T, s, pool, src), family_rows(cs, T, s, pool, dst)
            pr = transfer_pred(cs, T, s, pool, sr, dr, "log_ke", demean=True)
            curve[s] = slope(within(pr, T["group"][dr]), within(T["log_ke"][dr], T["group"][dr]))
        rec["route_calibrated_depth_curve"] = curve
        rec["route_calibrated_max_over_sites"] = float(max(curve.values()))
        depth = cs.depth()
        deep = np.array([v for s, v in curve.items() if depth[s] >= 0.25])
        rec["route_calibrated_over_sites_depth_ge_0.25"] = {
            "median": float(np.median(deep)), "q25": float(np.quantile(deep, 0.25)),
            "q75": float(np.quantile(deep, 0.75)), "n_sites": int(len(deep))}
        rec["route_calibrated_source_bootstrap"] = source_bootstrap(cs, T, site, pool, src, dst)
        rec["nearest_code_demeaned"] = nearest_code(direction, d["slope"])
        rec["expected"] = CODES[direction]
        out["transfer"][direction] = rec
        r, k = rec["log_ke/raw"], rec["log_ke_rel/raw_composed_from_factors"]
        print(f"  transfer {direction} @ {pool} {site}:\n"
              f"      raw {r['slope']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]   "
              f"K_rel raw {rec['log_ke_rel/raw']['slope']:.3f} vs composed from mu, v_rel {k['slope']:.3f}  (identity check)\n"
              f"      ROUTE-CALIBRATED {d['slope']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}] p {d['p_slope_gt_0']:.4f}  "
              f"ceiling {c['slope']:.3f} [{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}]  -> {rec['demeaned_fraction_of_ceiling']:.2f} of ceiling, "
              f"nearest code '{rec['nearest_code_demeaned']}'  expected {CODES[direction]}\n"
              f"      source-refit bootstrap {rec['route_calibrated_source_bootstrap']['median']:.3f} "
              f"[{rec['route_calibrated_source_bootstrap']['ci95'][0]:.3f}, {rec['route_calibrated_source_bootstrap']['ci95'][1]:.3f}]   "
              f"over blocks (depth>=0.25): median {rec['route_calibrated_over_sites_depth_ge_0.25']['median']:.3f} "
              f"IQR [{rec['route_calibrated_over_sites_depth_ge_0.25']['q25']:.3f}, "
              f"{rec['route_calibrated_over_sites_depth_ge_0.25']['q75']:.3f}]  max {rec['route_calibrated_max_over_sites']:.3f}",
              flush=True)

    # gates: the ingredients must cross the same gap
    site, pool, _ = best["S"]
    out["gates"]["mass_S->M"] = transfer(cs, T, site, pool, "S", "M", "log_mass")
    site, pool, _ = best["M"]
    out["gates"]["speed_M->S"] = transfer(cs, T, site, pool, "M", "S", "log_vrel")
    for k, v in out["gates"].items():
        print(f"  gate {k}: within-group slope {v['slope']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] "
              f"p {v['p_slope_gt_0']:.4f}", flush=True)

    cli.write_artifact(out, args.out, f"ke_reduced_mass_{args.model}.json")


if __name__ == "__main__":
    main()
