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

Contrasts are configured per family (``contrasts.<family>``: a ``calibration`` legal pair and a ``list`` of
{viol, legal, magnitude, stratify}); the calibration pair is also scored as the legal-vs-legal control.
``stratify`` names a boolean label (``physlaw.energy_physical`` for F1 momentum contrasts): the contrast is
also reported on each stratum that is non-empty.

Added for spec v2 §5.3 (all in ``motion_adjusted``):
  * dose-response   Spearman of the adjusted residual r_g with the violation magnitude difference
  * control check   per (covariate set, distance): the legal-vs-legal control's adjusted win-rate CI must
                    contain 0.5; otherwise every contrast under it is ``inconclusive``
  * Holm            across the violating contrasts within each (covariate set, distance); a verdict per contrast
  * summary         per contrast, under ``primary_covariates``: the distance is selected on half the groups and
                    the adjusted win rate reported on the other half (probes.select.heldout_select)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from physlaw.probes.select import heldout_select, holm


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


def _magnitude(labels: dict, key: str | None) -> float:
    return float(labels["physlaw"]["summary"][key]) if key else 0.0


def family_contrasts(cfg, family: str) -> tuple[tuple[str, str], list[dict]]:
    """(calibration pair, contrast dicts) for a family. Old configs (a flat list of [viol, legal] pairs,
    the last legal-vs-legal) are read too, with the F1 magnitude / stratification rules."""
    c = cfg["contrasts"]
    if isinstance(c, dict):
        fam = c[family]
        return tuple(fam["calibration"]), [dict(x) for x in fam["list"]]
    pairs = [tuple(x) for x in c]
    cal = next(p for p in pairs if p[0].startswith("conserve") and p[1].startswith("conserve"))
    out = [{"viol": v, "legal": l, "magnitude": "momentum_residual" if v.startswith("mom_violate") else "ke_ratio",
            "stratify": "energy_physical" if v.startswith("mom_violate") else None} for v, l in pairs if (v, l) != cal]
    return cal, out


def _subsets(con: dict, lab) -> dict:
    """{key suffix: predicate on the violating clip} for a contrast: all, and each stratum of ``stratify``."""
    out = {"": lambda clip: True}
    f = con.get("stratify")
    if f:
        out[f" [{f}]"] = lambda clip: bool(lab(clip)["physlaw"][f])
        out[f" [not {f}]"] = lambda clip: not bool(lab(clip)["physlaw"][f])
    return out


def run(sources: dict[str, Path], data_dir: Path, cfg, log=print) -> dict:
    """sources: name -> _index.jsonl-style file whose records carry {"clip", "group", "twin", "surprise"}."""
    idx = {name: _index(p) for name, p in sources.items()}
    labels_cache: dict[str, dict] = {}

    def lab(clip):
        if clip not in labels_cache:
            labels_cache[clip] = json.loads((data_dir / clip / "labels.json").read_text())
        return labels_cache[clip]

    cal, cons = family_contrasts(cfg, data_dir.name)
    control = {"viol": cal[0], "legal": cal[1], "magnitude": None, "control": True}
    out = {"calibration": list(cal), "contrasts": {}}
    for con in cons + [control]:
        viol, legal = con["viol"], con["legal"]
        for suffix, keep in _subsets(con, lab).items():
            key = f"{viol} vs {legal}{suffix}"
            res = {}
            for src, recs in idx.items():
                groups = sorted({r["group"] for r in recs.values()})
                full = [(f"{g}/{viol}", f"{g}/{legal}") for g in groups if f"{g}/{viol}" in recs and f"{g}/{legal}" in recs]
                pairs = [p for p in full if keep(p[0])]
                if not pairs or (suffix and len(pairs) == len(full)):      # empty, or a stratum that is everything
                    continue
                mag = (np.array([_magnitude(lab(v), con["magnitude"]) - _magnitude(lab(l), con["magnitude"])
                                 for v, l in pairs]) if con.get("magnitude") else None)
                metrics = sorted(set(recs[pairs[0][0]]["surprise"]) & set(recs[pairs[0][1]]["surprise"]))
                res[src] = {m: paired_stats(np.array([recs[v]["surprise"][m] for v, _ in pairs]),
                                            np.array([recs[l]["surprise"][m] for _, l in pairs]), mag)
                            for m in metrics}
            if res:
                out["contrasts"][key] = res
                log(f"[L2] {key}: " + "; ".join(f"{s}: {len(v)} metrics" for s, v in res.items()))
    for src in idx:          # Holm across the violating contrasts, per source and distance
        ms = {m for c, v in out["contrasts"].items() if src in v for m in v[src]}
        for m in ms:
            ph = holm({c: v[src][m]["wilcoxon_p"] for c, v in out["contrasts"].items()
                       if src in v and m in v[src] and not c.startswith(f"{cal[0]} vs {cal[1]}")})
            for c, p in ph.items():
                out["contrasts"][c][src][m]["wilcoxon_p_holm"] = p
    names = list(idx)
    if "pixel_baselines" in idx and len(names) > 1:
        model = [n for n in names if n != "pixel_baselines"][0]
        cov = {c: {**idx["pixel_baselines"][c]["surprise"], **kinematic_covariates(lab(c))}
               for c in idx[model] if c in idx["pixel_baselines"]}
        out["motion_adjusted"] = motion_adjusted(idx[model], cov, cons, calibration=cal, labels=lab,
                                                 primary=cfg.get("primary_covariates", "speed+copy_last"),
                                                 n_splits=int(cfg.get("selection", {}).get("n_splits", 50)))
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


def motion_adjusted(model_recs: dict, cov: dict[str, dict], contrasts: list[dict], calibration=("conserve_e1", "conserve_e015"),
                    n_boot: int = 2000, seed: int = 0, labels=None, primary: str = "speed+copy_last",
                    n_splits: int = 50, alpha: float = 0.05) -> dict:
    """model_recs: clip -> record with "surprise"; cov: clip -> {covariate: positive value};
    contrasts: dicts {viol, legal, magnitude, stratify}; labels: clip -> labels.json (for strata, magnitudes)."""
    from scipy.stats import spearmanr, wilcoxon
    sur = {c: r["surprise"] for c, r in model_recs.items()}
    groups = sorted({r["group"] for r in model_recs.values()})
    have = lambda g, t: f"{g}/{t}" in sur and f"{g}/{t}" in cov
    cal_groups = [g for g in groups if have(g, calibration[0]) and have(g, calibration[1])]
    cal_pairs = [(f"{g}/{calibration[0]}", f"{g}/{calibration[1]}") for g in cal_groups]
    metrics = sorted(m for m in sur[cal_pairs[0][0]] if m.split("_")[-1] in ("obj", "all"))
    control = {"viol": calibration[0], "legal": calibration[1], "control": True}
    ctrl_key = f"{calibration[0]} vs {calibration[1]}"
    out = {"calibration": list(calibration), "covariate_sets": COVARIATE_SETS, "primary_covariates": primary,
           "fit": {}, "contrasts": {}, "summary": {}}
    resid: dict = {}          # (cs, contrast key, metric) -> {group: r_g}, for the selection rule
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
            for con in [*contrasts, control]:
                viol, legal = con["viol"], con["legal"]
                for suffix, keep in (_subsets(con, labels) if labels else {"": lambda c: True}).items():
                    key = f"{viol} vs {legal}{suffix}"
                    gfull = [g for g in groups if have(g, viol) and have(g, legal)]
                    gs = [g for g in gfull if keep(f"{g}/{viol}")]
                    if not gs or (suffix and len(gs) == len(gfull)):
                        continue
                    pairs = [(f"{g}/{viol}", f"{g}/{legal}") for g in gs]
                    dm = _dlog(sur, pairs, m)
                    D = np.stack([_dlog(cov, pairs, k) for k in keys], 1)
                    if con.get("control"):                     # leave-one-group-out for the control
                        r = np.array([dm[j] - D[j] @ _fit(np.delete(cm, j), np.delete(CC, j, 0)) for j in range(len(gs))])
                    else:
                        r = dm - D @ beta
                    resid[(cs, key, m)] = dict(zip(gs, r))
                    bw, bmed = [], []
                    for k in range(n_boot):
                        i = rng.integers(0, len(gs), len(gs))
                        rr = dm[i] - D[i] @ boot[k]
                        bw.append(_win(rr)); bmed.append(np.median(rr))
                    rec = {"n": len(gs), "raw_win_rate": _win(dm), "adj_win_rate": _win(r),
                           "adj_win_rate_ci95": [float(np.quantile(bw, 0.025)), float(np.quantile(bw, 0.975))],
                           "adj_median_log_ratio": float(np.median(r)),
                           "adj_median_ci95": [float(np.quantile(bmed, 0.025)), float(np.quantile(bmed, 0.975))],
                           "adj_wilcoxon_p": float(wilcoxon(r).pvalue) if np.any(r != 0) else 1.0}
                    if con.get("magnitude") and labels is not None:      # dose-response
                        mag = np.array([_magnitude(labels(v), con["magnitude"]) - _magnitude(labels(l), con["magnitude"])
                                        for v, l in pairs])
                        if np.std(mag) > 0 and np.std(r) > 0:
                            sp = spearmanr(r, mag)
                            rec["dose_spearman"], rec["dose_spearman_p"] = float(sp.statistic), float(sp.pvalue)
                    out["contrasts"].setdefault(cs, {}).setdefault(key, {})[m] = rec
            # control check, Holm and verdicts within this (covariate set, distance)
            by_c = out["contrasts"][cs]
            ctrl = by_c[ctrl_key][m]
            ok = ctrl["adj_win_rate_ci95"][0] <= 0.5 <= ctrl["adj_win_rate_ci95"][1]
            ctrl["control_at_chance"] = ok
            ph = holm({c: v[m]["adj_wilcoxon_p"] for c, v in by_c.items() if c != ctrl_key and m in v})
            for c, p in ph.items():
                rec = by_c[c][m]
                rec["adj_wilcoxon_p_holm"] = p
                rec["verdict"] = ("inconclusive (legal-vs-legal control not at chance)" if not ok else
                                  "no difference" if not p < alpha else
                                  "more surprised by violation" if rec["adj_win_rate"] > 0.5 else "less surprised by violation")
    # summary: under the primary covariate set, distance selected on held-out groups
    for key in out["contrasts"].get(primary, {}):
        ms = [m for m in metrics if (primary, key, m) in resid]

        def score(m, gids, key=key):
            r = resid[(primary, key, m)]
            v = np.array([r[g] for g in gids if g in r])
            return _win(v) if len(v) else float("nan")
        sel = heldout_select(score, ms, list(resid[(primary, key, ms[0])]), n_splits=n_splits)
        rec = out["contrasts"][primary][key][sel["selected"]]
        out["summary"][key] = {**sel, "covariates": primary, "full_sample_adj_win_rate": rec["adj_win_rate"],
                               "full_sample_ci95": rec["adj_win_rate_ci95"], "verdict": rec.get("verdict", "control"),
                               "control_at_chance": out["contrasts"][primary][ctrl_key][sel["selected"]]["control_at_chance"]}
    return out


def adjusted_table(adj: dict, metrics=None) -> str:
    rows = []
    if adj.get("summary"):
        s0 = next(iter(adj["summary"].values()))
        rows += [f"### Summary (covariates {adj['primary_covariates']}; distance selected on {s0['n_select_groups']} groups, "
                 f"adjusted win rate on the other {s0['n_eval_groups']}, {2 * s0['n_splits']} evaluations; "
                 f"variants searched = {s0['n_variants']} distances; zero-fit, no folds)", "",
                 "| contrast | selected distance (picks) | held-out adj. win [2.5%, 97.5%] | full-sample adj. win [95% CI] | verdict (Holm) |",
                 "|---|---|---|---|---|"]
        for c, s in adj["summary"].items():
            rows.append(f"| {c} | {s['selected']} ({s['selected_counts'][s['selected']]}/{2 * s['n_splits']}) | "
                        f"{s['score']:.2f} [{s['score_q025']:.2f}, {s['score_q975']:.2f}] | {s['full_sample_adj_win_rate']:.2f} "
                        f"[{s['full_sample_ci95'][0]:.2f}, {s['full_sample_ci95'][1]:.2f}] | {s['verdict']} |")
        rows.append("")
    rows += ["How well each motion covariate set explains the LEGAL pairs' surprise difference "
             "(uncentred R², fit through the origin):", "",
             "| distance | " + " | ".join(adj["fit"]) + " |", "|---|" + "---|" * len(adj["fit"])]
    ms = list(next(iter(adj["fit"].values())))
    for m in ms:
        rows.append(f"| {m} | " + " | ".join(f"{adj['fit'][cs][m]['r2_uncentred']:.2f}" for cs in adj["fit"]) + " |")
    for cs, by_c in adj["contrasts"].items():
        rows += ["", f"### covariates: {cs}", "",
                 "| contrast | distance | n | raw win | adjusted win [95% CI] | adj. median log-ratio [95% CI] | Wilcoxon p (Holm) | dose ρ (p) | verdict |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for c, by_m in by_c.items():
            for m, s in by_m.items():
                if metrics and m not in metrics:
                    continue
                dose = f"{s['dose_spearman']:+.2f} ({s['dose_spearman_p']:.2g})" if "dose_spearman" in s else ""
                ver = s.get("verdict") or ("control at chance" if s.get("control_at_chance") else "CONTROL NOT AT CHANCE")
                rows.append(f"| {c} | {m} | {s['n']} | {s['raw_win_rate']:.2f} | {s['adj_win_rate']:.2f} "
                            f"[{s['adj_win_rate_ci95'][0]:.2f}, {s['adj_win_rate_ci95'][1]:.2f}] | "
                            f"{s['adj_median_log_ratio']:+.3f} [{s['adj_median_ci95'][0]:+.3f}, {s['adj_median_ci95'][1]:+.3f}] | "
                            f"{s['adj_wilcoxon_p']:.2g} ({s.get('adj_wilcoxon_p_holm', float('nan')):.2g}) | {dose} | {ver} |")
    return "\n".join(rows)


def table(res: dict) -> str:
    rows = ["| contrast | source | distance | n | win rate [95% CI] | Wilcoxon p (Holm) | Cliff's δ | median rel. diff | ρ vs magnitude |",
            "|---|---|---|---|---|---|---|---|---|"]
    for c, by_src in res["contrasts"].items():
        for src, by_m in by_src.items():
            for m, s in by_m.items():
                rho = s.get("spearman_vs_magnitude")
                rows.append(f"| {c} | {src} | {m} | {s['n']} | {s['win_rate']:.2f} "
                            f"[{s['win_rate_ci95'][0]:.2f}, {s['win_rate_ci95'][1]:.2f}] | {s['wilcoxon_p']:.2g} "
                            f"({s.get('wilcoxon_p_holm', float('nan')):.2g}) | "
                            f"{s['cliffs_delta']:+.2f} | {s['median_rel_diff']:+.1%} | "
                            f"{'' if rho is None else f'{rho:+.2f}'} |")
    return "\n".join(rows)
