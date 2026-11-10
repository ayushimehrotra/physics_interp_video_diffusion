"""L2 violation sensitivity (spec §5): zero-fit paired surprise contrasts.

For each contrast (violating twin, legal twin) and each surprise distance, over twin groups:
  * win rate      P(surprise(viol) > surprise(legal)), ties count 1/2; bootstrap 95% CI over groups
  * Wilcoxon      signed-rank two-sided p on the paired differences
  * Cliff's delta between the two surprise distributions (unpaired effect size)
  * Spearman      paired: rho(surprise difference, violation-magnitude difference) over groups
Magnitude: momentum contrasts use summary.momentum_residual; others summary.ke_ratio.

Every contrast is computed for the model AND for the pixel baselines of scripts/05_surprise.py
(copy-last-frame, constant-velocity extrapolation) on the same frames, so an asymmetry that a
copy-last-frame predictor reproduces is visible next to the model's.

Motion-adjusted contrast (``motion_adjusted``): a model can win a contrast just because the
violating twin moves more (or less) after contact. Calibrate on the legal-vs-legal pairs, where
no law is broken: per group, Δlog s_model = β · Δlog c (both orderings, so no intercept;
equivalent to a group fixed-effect fit on the two legal twins), for several motion covariate sets c
(COVARIATE_SETS: copy-last-frame pixel MSE, post-contact speed sum, post-contact KE, speed + pixel). For each violating
contrast the residual r_g = Δlog s_model − β·Δlog c asks: is the model MORE surprised
than the motion change predicts? Win rate of r > 0, Wilcoxon on r, and a bootstrap over groups
that refits β each time. The legal control is scored leave-one-group-out.
Assumption to keep in mind: the motion change of a restitution change (legal pairs) is matched
in kind to that of a violation. The β fit's R² says how much of legal-pair surprise motion explains.

mom_violate_e1 creates energy in some groups (labels physlaw.energy_physical); its contrast is
also reported on the energy-physical subset.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _index(path: Path) -> dict[str, dict]:
    recs = {}
    for line in open(path):
        r = json.loads(line)
        recs[r["clip"]] = r
    return recs


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a)[:, None], np.asarray(b)[None, :]
    return float(((a > b).sum() - (a < b).sum()) / (a.size * b.size))


def _win(d: np.ndarray) -> float:
    return float(((d > 0) + 0.5 * (d == 0)).mean())


def paired_stats(sv: np.ndarray, sl: np.ndarray, mag: np.ndarray | None, n_boot: int = 2000, seed: int = 0) -> dict:
    from scipy.stats import spearmanr, wilcoxon
    d = sv - sl
    rng = np.random.default_rng(seed)
    boots = [_win(d[rng.integers(0, len(d), len(d))]) for _ in range(n_boot)]
    out = {"n": int(len(d)), "win_rate": _win(d),
           "win_rate_ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
           "wilcoxon_p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
           "cliffs_delta": cliffs_delta(sv, sl),
           "median_diff": float(np.median(d)), "median_rel_diff": float(np.median(d / np.abs(sl).clip(1e-12)))}
    if mag is not None and np.std(mag) > 0 and np.std(d) > 0:
        r = spearmanr(d, mag)
        out["spearman_vs_magnitude"] = float(r.statistic)
        out["spearman_p"] = float(r.pvalue)
    return out


def _magnitude(labels: dict, viol: str) -> float:
    s = labels["physlaw"]["summary"]
    return float(s["momentum_residual"] if viol.startswith("mom_violate") else s["ke_ratio"])


def run(sources: dict[str, Path], data_dir: Path, cfg, log=print) -> dict:
    """sources: name -> _index.jsonl-style file whose records carry {"clip", "group", "twin", "surprise"}."""
    idx = {name: _index(p) for name, p in sources.items()}
    labels_cache: dict[str, dict] = {}

    def lab(clip):
        if clip not in labels_cache:
            labels_cache[clip] = json.loads((data_dir / clip / "labels.json").read_text())
        return labels_cache[clip]

    out = {"contrasts": {}}
    for viol, legal in cfg["contrasts"]:
        subsets = {"all": None}
        if viol == "mom_violate_e1":
            subsets["energy_physical"] = True
        for sub, flag in subsets.items():
            key = f"{viol} vs {legal}" + ("" if sub == "all" else f" [{sub}]")
            res = {}
            for src, recs in idx.items():
                groups = sorted({r["group"] for r in recs.values()})
                pairs = [(f"{g}/{viol}", f"{g}/{legal}") for g in groups
                         if f"{g}/{viol}" in recs and f"{g}/{legal}" in recs]
                if flag is not None:
                    pairs = [p for p in pairs if lab(p[0])["physlaw"]["energy_physical"] == flag]
                if not pairs:
                    continue
                mag = np.array([_magnitude(lab(v), viol) - _magnitude(lab(l), viol) for v, l in pairs])
                metrics = sorted(set(recs[pairs[0][0]]["surprise"]) & set(recs[pairs[0][1]]["surprise"]))
                res[src] = {m: paired_stats(np.array([recs[v]["surprise"][m] for v, _ in pairs]),
                                            np.array([recs[l]["surprise"][m] for _, l in pairs]), mag)
                            for m in metrics}
            out["contrasts"][key] = res
            log(f"[L2] {key}: " + "; ".join(f"{s}: {len(v)} metrics" for s, v in res.items()))
    names = list(idx)
    if "pixel_baselines" in idx and len(names) > 1:
        model = [n for n in names if n != "pixel_baselines"][0]
        ep = {c: lab(c)["physlaw"]["energy_physical"] for c in idx[model]}
        cov = {c: {**idx["pixel_baselines"][c]["surprise"], **kinematic_covariates(lab(c))}
               for c in idx[model] if c in idx["pixel_baselines"]}
        out["motion_adjusted"] = motion_adjusted(idx[model], cov, [tuple(c) for c in cfg["contrasts"]], energy_physical=ep)
    return out


# Motion covariates. Pixel ones come from scripts/05_surprise.py; kinematic ones from labels.
# "{region}" is filled with the model metric's region (obj / all).
COVARIATE_SETS = {
    "copy_last": ["copy_last_mse_{region}"],
    "post_speed": ["post_speed_sum"],
    "post_ke": ["post_ke"],
    "speed+copy_last": ["post_speed_sum", "copy_last_mse_{region}"],
}


def kinematic_covariates(labels: dict) -> dict[str, float]:
    pl = labels["physlaw"]
    v = {o: np.asarray(pl["post_velocity"][o], float)[:2] for o in ("A", "B")}
    return {"post_speed_sum": float(np.linalg.norm(v["A"]) + np.linalg.norm(v["B"])),
            "post_ke": float(pl["summary"]["ke_post"])}


def _dlog(vals: dict, pairs, key):
    return np.array([np.log(vals[v][key]) - np.log(vals[l][key]) for v, l in pairs])


def _fit(dm: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Least squares through the origin (legal pairs in both orderings => no intercept)."""
    return np.linalg.lstsq(C, dm, rcond=None)[0]


def motion_adjusted(model_recs: dict, cov: dict[str, dict], contrasts, calibration=("conserve_e1", "conserve_e015"),
                    n_boot: int = 2000, seed: int = 0, energy_physical: dict | None = None) -> dict:
    """model_recs: clip -> record with "surprise"; cov: clip -> {covariate: positive value}."""
    from scipy.stats import wilcoxon
    sur = {c: r["surprise"] for c, r in model_recs.items()}
    groups = sorted({r["group"] for r in model_recs.values()})
    have = lambda g, t: f"{g}/{t}" in sur and f"{g}/{t}" in cov
    cal_groups = [g for g in groups if have(g, calibration[0]) and have(g, calibration[1])]
    cal_pairs = [(f"{g}/{calibration[0]}", f"{g}/{calibration[1]}") for g in cal_groups]
    metrics = sorted(m for m in sur[cal_pairs[0][0]] if m.split("_")[-1] in ("obj", "all"))
    out = {"calibration": list(calibration), "covariate_sets": COVARIATE_SETS, "fit": {}, "contrasts": {}}
    for cs, keys_t in COVARIATE_SETS.items():
        rng = np.random.default_rng(seed)
        for m in metrics:
            keys = [k.format(region=m.split("_")[-1]) for k in keys_t]
            cm = _dlog(sur, cal_pairs, m)
            CC = np.stack([_dlog(cov, cal_pairs, k) for k in keys], 1)
            beta = _fit(cm, CC)
            r2 = 1 - ((cm - CC @ beta) ** 2).sum() / (cm ** 2).sum()     # uncentred (no intercept)
            out["fit"].setdefault(cs, {})[m] = {"covariates": keys, "beta": beta.tolist(), "r2_uncentred": float(r2)}
            boot = []
            for _ in range(n_boot):
                i = rng.integers(0, len(cal_groups), len(cal_groups))
                boot.append(_fit(cm[i], CC[i]))
            for viol, legal in contrasts:
                subsets = {"all": None}
                if viol == "mom_violate_e1" and energy_physical is not None:
                    subsets["energy_physical"] = True
                for sub, flag in subsets.items():
                    key = f"{viol} vs {legal}" + ("" if sub == "all" else f" [{sub}]")
                    gs = [g for g in groups if have(g, viol) and have(g, legal)
                          and (flag is None or energy_physical[f"{g}/{viol}"] == flag)]
                    pairs = [(f"{g}/{viol}", f"{g}/{legal}") for g in gs]
                    dm = _dlog(sur, pairs, m)
                    D = np.stack([_dlog(cov, pairs, k) for k in keys], 1)
                    if (viol, legal) == tuple(calibration):          # leave-one-group-out for the control
                        r = np.array([dm[j] - D[j] @ _fit(np.delete(cm, j), np.delete(CC, j, 0)) for j in range(len(gs))])
                    else:
                        r = dm - D @ beta
                    bw, bmed = [], []
                    for k in range(n_boot):
                        i = rng.integers(0, len(gs), len(gs))
                        rr = dm[i] - D[i] @ boot[k]
                        bw.append(_win(rr)); bmed.append(np.median(rr))
                    out["contrasts"].setdefault(cs, {}).setdefault(key, {})[m] = {
                        "n": len(gs), "raw_win_rate": _win(dm), "adj_win_rate": _win(r),
                        "adj_win_rate_ci95": [float(np.quantile(bw, 0.025)), float(np.quantile(bw, 0.975))],
                        "adj_median_log_ratio": float(np.median(r)),
                        "adj_median_ci95": [float(np.quantile(bmed, 0.025)), float(np.quantile(bmed, 0.975))],
                        "adj_wilcoxon_p": float(wilcoxon(r).pvalue) if np.any(r != 0) else 1.0}
    return out


def adjusted_table(adj: dict, metrics=None) -> str:
    rows = ["How well each motion covariate set explains the LEGAL pairs' surprise difference "
            "(uncentred R², fit through the origin):", "",
            "| distance | " + " | ".join(adj["fit"]) + " |", "|---|" + "---|" * len(adj["fit"])]
    ms = list(next(iter(adj["fit"].values())))
    for m in ms:
        rows.append(f"| {m} | " + " | ".join(f"{adj['fit'][cs][m]['r2_uncentred']:.2f}" for cs in adj["fit"]) + " |")
    for cs, by_c in adj["contrasts"].items():
        rows += ["", f"### covariates: {cs}", "",
                 "| contrast | distance | n | raw win | adjusted win [95% CI] | adj. median log-ratio [95% CI] | Wilcoxon p |",
                 "|---|---|---|---|---|---|---|"]
        for c, by_m in by_c.items():
            for m, s in by_m.items():
                if metrics and m not in metrics:
                    continue
                rows.append(f"| {c} | {m} | {s['n']} | {s['raw_win_rate']:.2f} | {s['adj_win_rate']:.2f} "
                            f"[{s['adj_win_rate_ci95'][0]:.2f}, {s['adj_win_rate_ci95'][1]:.2f}] | "
                            f"{s['adj_median_log_ratio']:+.3f} [{s['adj_median_ci95'][0]:+.3f}, {s['adj_median_ci95'][1]:+.3f}] | "
                            f"{s['adj_wilcoxon_p']:.2g} |")
    return "\n".join(rows)


def table(res: dict) -> str:
    rows = ["| contrast | source | distance | n | win rate [95% CI] | Wilcoxon p | Cliff's δ | median rel. diff | ρ vs magnitude |",
            "|---|---|---|---|---|---|---|---|---|"]
    for c, by_src in res["contrasts"].items():
        for src, by_m in by_src.items():
            for m, s in by_m.items():
                rho = s.get("spearman_vs_magnitude")
                rows.append(f"| {c} | {src} | {m} | {s['n']} | {s['win_rate']:.2f} "
                            f"[{s['win_rate_ci95'][0]:.2f}, {s['win_rate_ci95'][1]:.2f}] | {s['wilcoxon_p']:.2g} | "
                            f"{s['cliffs_delta']:+.2f} | {s['median_rel_diff']:+.1%} | "
                            f"{'' if rho is None else f'{rho:+.2f}'} |")
    return "\n".join(rows)
