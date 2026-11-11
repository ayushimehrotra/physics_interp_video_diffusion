"""L2 violation sensitivity (spec §5): zero-fit paired surprise contrasts.

For each contrast (violating twin, legal twin) and each surprise distance, over twin groups:
  * win rate      P(surprise(viol) > surprise(legal)), ties count 1/2; bootstrap 95% CI over groups
  * Wilcoxon      signed-rank two-sided p on the paired differences
  * Cliff's delta between the two surprise distributions (unpaired effect size)
  * Spearman      paired: rho(surprise difference, violation magnitude) over groups
Magnitude (``magnitude`` in the contrast config): ``eta`` = the momentum-exchange imbalance η of the violating
twin (laws.momentum_exchange on its true labels; η = |m_A dv_A + m_B dv_B| / (|m_A dv_A| + |m_B dv_B|), the size
of the drift edit relative to the exchange) for the momentum contrasts; any other value is a labels
``physlaw.summary`` key, differenced violating − legal (``ke_ratio`` for energy_create). ``summary.momentum_residual``
is NOT used for momentum: |Δp| / |p_pre| mostly tracks |p_pre| (rank corr. 0.36 with the edit size on F1).

Primary contrast: the contrast flagged ``primary: true`` (F1: mom_violate_e015 vs conserve_e015) is listed first
and named in ``primary_contrast``. Reason: an observer that knows momentum but assumes the wrong restitution wins
the e1 contrast only ~53% of the time (if it assumes e = 0), while the e015 contrast stays at 0.95-1.00 for any e.

Ideal-observer reference bands (``ideal_observer``, labels only, per contrast): win rate P(|pred − o_viol| >
|pred − o_legal|) in the 4-D velocity-change space o = (dv_A, dv_B), ties 1/2, for predictors: perfect (the true
legal outcome), law with assumed e in {1, 0.5, 0}, and mass-blind (the true relative-velocity change split equally,
laws.mass_blind_outcome); overall and on unequal-mass groups (mass ratio >= 1.25). Also the share of groups whose
violating twin is slower (|v_post| norm ratio < 1).

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
CAVEAT: β is fitted on legal pairs that differ only in RELATIVE velocity and applied to momentum violations that
differ only in CENTRE-OF-MASS velocity, so it is not identified for CoM motion; read the adjusted win rate with
the pixel-baseline gap below, which is the primary motion control.

Primary motion control (``vs_pixel`` in each motion-adjusted record): model adjusted win − copy_last raw win and
model adjusted win − constant_velocity raw win (and raw − raw), with a paired group bootstrap CI (the same group
resample for model and baseline; β refitted per draw). A model whose win rate a pixel extrapolator matches has
shown nothing beyond motion.

Contrasts are configured per family (``contrasts.<family>``: a ``calibration`` legal pair and a ``list`` of
{viol, legal, magnitude, stratify}); the calibration pair is also scored as the legal-vs-legal control.
``stratify`` names a boolean label (``physlaw.energy_physical`` for F1 momentum contrasts): the contrast is
also reported on each stratum that is non-empty.

Added for spec v2 §5.3 (all in ``motion_adjusted``):
  * dose-response   Spearman of the adjusted residual r_g with the violation magnitude difference
  * control check   per (covariate set, distance): the legal-vs-legal control's adjusted win-rate CI must
                    contain 0.5; otherwise every contrast under it is ``inconclusive``
  * Holm            across the violating contrasts within each (covariate set, distance); a verdict per contrast
  * summary         per contrast, under ``primary_covariates``: among the distances whose control is at chance,
                    the distance is selected on half the groups and the adjusted win rate reported on the other
                    half (probes.select.heldout_select)

F2 group contrasts (``contrasts.f2.group_contrasts``; l2.json ``group_contrasts``, md section "F2 group contrasts"):
one twin group = the legal ladder e0.0..e1.0 + momentum violators mom_violate_e<e>_<pos|neg> (+ energy-creating
arms e1.1..e1.5). ``speed_matched``: each violator vs the ladder twin closest in post-contact KE (or total speed),
so speed cannot separate the pair (fields: matching.median_abs_log_ke_ratio, raw.<src>.<distance>.<subset>.win_rate).
``family_min``: violator more surprising than the least surprising ladder twin (chance K/(K+1)), and its rank among
ladder ∪ {violator} (rank_mean, chance 0.5). Labels-only ideal observers incl. a speed-only observer
(``ideal_observer``); motion adjustment with β fitted on ``ladder_calibration`` pairs and checked on held-out
adjacent-level controls (``motion_adjusted.controls``), pixel-baseline gap (``vs_pixel``). ``energy_kink``: pending.
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
    out = {"n": int(len(d)), "n_zero_diff": int(np.sum(d == 0)), "win_rate": _win(d),
           "win_rate_boot_sd": float(np.std(boots)),
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


def eta_of(labels: dict) -> float:
    """Momentum-exchange imbalance η of a clip, from its true labels (laws.momentum_exchange)."""
    from physlaw.eval import laws
    return float(laws.momentum_exchange(*laws.from_labels(labels))["eta"])


def contrast_magnitude(key: str | None, lab_viol: dict, lab_legal: dict) -> float:
    """Dose of one violating/legal pair: ``eta`` -> η of the violating twin; any other key -> the labels
    ``physlaw.summary`` value, violating − legal."""
    if key == "eta":
        return eta_of(lab_viol)
    return _magnitude(lab_viol, key) - _magnitude(lab_legal, key)


# --------------------------------------------------------------------------- ideal-observer reference bands

IDEAL_E = (1.0, 0.5, 0.0)        # assumed restitutions of the law-based observers
UNEQUAL_MASS_RATIO = 1.25


def _win_dist(pred: np.ndarray, o_viol: np.ndarray, o_legal: np.ndarray) -> float:
    a, b = np.linalg.norm(pred - o_viol), np.linalg.norm(pred - o_legal)
    if np.isclose(a, b, rtol=1e-12, atol=1e-12):
        return 0.5
    return 1.0 if a > b else 0.0


def ideal_observer_bands(items: list[tuple], unequal_ratio: float = UNEQUAL_MASS_RATIO) -> dict:
    """Win rates of reference observers on paired outcomes, from kinematics only.

    items: (PreState, o_viol [4], o_legal [4]) per group, o = (dv_A, dv_B) in the plane. An observer predicts
    an outcome ``pred`` and "wins" a group when |pred − o_viol| > |pred − o_legal| (ties 1/2). Observers:
    ``perfect`` (pred = o_legal), ``law_e{e}`` (the frictionless legal outcome at assumed restitution e), and
    ``mass_blind`` (o_legal's relative-velocity change split equally between the balls,
    laws.mass_blind_outcome). Reported overall and on groups with mass ratio >= ``unequal_ratio``, plus
    ``viol_slower`` = share of groups with |v_post(viol)| / |v_post(legal)| < 1 (Frobenius norm over both balls)."""
    from physlaw.data import collision_physics as cp
    from physlaw.eval import laws
    names = ["perfect", *[f"law_e{e:g}" for e in IDEAL_E], "mass_blind"]
    wins = {k: [] for k in names}
    unequal, slower = [], []
    for st, ov, ol in items:
        ov, ol = np.asarray(ov, float).reshape(4), np.asarray(ol, float).reshape(4)
        v0 = np.stack([st.v_a, st.v_b])
        mb = laws.mass_blind_outcome({"v": v0}, {"v": v0 + ol.reshape(2, 2)}, {})["v"] - v0
        preds = {"perfect": ol, **{f"law_e{e:g}": cp.legal_outcome(st, e) for e in IDEAL_E}, "mass_blind": mb.reshape(4)}
        for k, pr in preds.items():
            wins[k].append(_win_dist(pr, ov, ol))
        unequal.append(max(st.m_a, st.m_b) / min(st.m_a, st.m_b) >= unequal_ratio)
        slower.append(np.linalg.norm(v0 + ov.reshape(2, 2)) / np.linalg.norm(v0 + ol.reshape(2, 2)) < 1)
    unequal = np.array(unequal, bool)
    out = {"n": len(items), "n_unequal_mass": int(unequal.sum()), "unequal_mass_ratio": unequal_ratio,
           "win_rate": {k: float(np.mean(v)) for k, v in wins.items()},
           "win_rate_unequal_mass": {k: (float(np.mean(np.array(v)[unequal])) if unequal.any() else float("nan"))
                                     for k, v in wins.items()},
           "viol_slower": float(np.mean(slower)) if slower else float("nan")}
    return out


def ideal_observer(pairs: list[tuple[dict, dict]], **kw) -> dict:
    """``ideal_observer_bands`` from (labels of the violating twin, labels of the legal twin) per group."""
    from physlaw.data import collision_physics as cp
    from physlaw.eval import laws
    items = []
    for lv, ll in pairs:
        pv, qv, ctx = laws.from_labels(lv)
        pl, ql, _ = laws.from_labels(ll)
        st = cp.PreState(float(ctx["m"][0]), float(ctx["m"][1]), pl["v"][0], pl["v"][1], ctx["n"])
        items.append((st, (qv["v"] - pv["v"]).reshape(4), (ql["v"] - pl["v"]).reshape(4)))
    return ideal_observer_bands(items, **kw)


def ideal_line(io: dict) -> str:
    """One md line for a contrast's ideal-observer bands."""
    w, u = io["win_rate"], io["win_rate_unequal_mass"]
    law = " / ".join(f"{w[f'law_e{e:g}']:.2f}" for e in IDEAL_E)
    return (f"ideal observer (labels only, 4-D Δv distance, n={io['n']}): perfect {w['perfect']:.2f}; law e=1/0.5/0 "
            f"{law}; mass-blind {w['mass_blind']:.2f} ({u['mass_blind']:.2f} on {io['n_unequal_mass']} unequal-mass "
            f"groups, ratio >= {io['unequal_mass_ratio']:g}); violator slower in {io['viol_slower']:.0%} of groups")


def family_contrasts(cfg, family: str) -> tuple[tuple[str, str], list[dict]]:
    """(calibration pair, contrast dicts) for a family. Old configs (a flat list of [viol, legal] pairs,
    the last legal-vs-legal) are read too, with the F1 magnitude / stratification rules."""
    c = cfg["contrasts"]
    if isinstance(c, dict):
        fam = c[family]
        cons = [dict(x) for x in fam.get("list") or []]       # F2 may have only group_contrasts (group_spec)
        return tuple(fam["calibration"]), sorted(cons, key=lambda x: not x.get("primary", False))   # primary first
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
    prim = [f"{c['viol']} vs {c['legal']}" for c in cons if c.get("primary")]
    out = {"calibration": list(cal), "primary_contrast": prim[0] if prim else None, "contrast_info": {}, "contrasts": {}}
    groups_all = sorted(p.name for p in data_dir.iterdir() if p.is_dir() and (p / cal[0] / "labels.json").exists())
    for con in cons:          # labels-only references: ideal-observer bands per contrast
        key = f"{con['viol']} vs {con['legal']}"
        info = {"viol": con["viol"], "legal": con["legal"], "primary": bool(con.get("primary")),
                "magnitude": con.get("magnitude")}
        gs = [g for g in groups_all if (data_dir / g / con["viol"] / "labels.json").exists()
              and (data_dir / g / con["legal"] / "labels.json").exists()]
        try:
            info["ideal_observer"] = ideal_observer([(lab(f"{g}/{con['viol']}"), lab(f"{g}/{con['legal']}")) for g in gs])
        except (KeyError, NotImplementedError) as e:          # non-collision family
            log(f"[L2] {key}: no ideal-observer bands ({e!r})")
        out["contrast_info"][key] = info
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
                mag = (np.array([contrast_magnitude(con["magnitude"], lab(v), lab(l)) for v, l in pairs])
                       if con.get("magnitude") else None)
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
        out["motion_adjusted"]["contrast_info"] = out["contrast_info"]     # for adjusted_table (same object)
    spec = group_spec(cfg, data_dir.name)
    if spec:                 # F2: speed_matched / family_min over the whole twin group
        model = next((n for n in names if n != "pixel_baselines"), None)
        gdirs = [g for g in groups_all if any((data_dir / g / t / "labels.json").exists() for t in spec["ladder"])]
        kin = group_kinematics(lab, gdirs, lambda g: sorted(q.name for q in (data_dir / g).iterdir()
                                                             if (q / "labels.json").exists()))
        cov_g = None
        if "pixel_baselines" in idx and model is not None:
            cov_g = {c: {**idx["pixel_baselines"][c]["surprise"], **kinematic_covariates(lab(c))}
                     for c in idx[model] if c in idx["pixel_baselines"]}
        out["group_contrasts"] = group_contrasts(
            kin, {src: {c: r["surprise"] for c, r in recs.items()} for src, recs in idx.items()}, spec,
            cov=cov_g, model=model, primary=cfg.get("primary_covariates", "speed+copy_last"),
            n_boot=int(cfg.get("group_n_boot", 1000)))
        log(f"[L2] group contrasts: " + ", ".join(f"{k}: {v.get('n_rows', v.get('status'))} rows" for k, v in out["group_contrasts"]["contrasts"].items()))
    return out


# Motion covariates. Pixel ones come from scripts/05_surprise.py; kinematic ones from labels.
# "{region}" is filled with the model metric's region (obj / all).
# Pixel baselines for the primary motion control: name -> surprise key ("{region}" = the model metric's region).
PIXEL_BASELINES = {"copy_last": "copy_last_mse_{region}", "const_velocity": "constvel_px"}
BETA_CAVEAT = ("β is fitted on legal pairs that differ only in relative velocity and applied to violations that differ "
               "only in CoM velocity, so it is not identified for CoM motion")

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
    prim = [f"{c['viol']} vs {c['legal']}" for c in contrasts if c.get("primary")]
    out = {"calibration": list(calibration), "covariate_sets": COVARIATE_SETS, "primary_covariates": primary,
           "primary_contrast": prim[0] if prim else None, "beta_caveat": BETA_CAVEAT,
           "pixel_baselines": PIXEL_BASELINES, "fit": {}, "contrasts": {}, "summary": {}}
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
                    # pixel-baseline raw differences on the same pairs (primary motion control)
                    pix = {}
                    for bn, bk in PIXEL_BASELINES.items():
                        bk = bk.format(region=m.split("_")[-1])
                        if all(bk in cov[v] and bk in cov[l] for v, l in pairs):
                            pix[bn] = np.array([cov[v][bk] - cov[l][bk] for v, l in pairs])
                    bw, bmed = [], []
                    bpix = {bn: {"adj": [], "raw": []} for bn in pix}
                    for k in range(n_boot):
                        i = rng.integers(0, len(gs), len(gs))
                        rr = dm[i] - D[i] @ boot[k]
                        bw.append(_win(rr)); bmed.append(np.median(rr))
                        for bn, bd in pix.items():          # paired: same group resample for model and baseline
                            wb = _win(bd[i])
                            bpix[bn]["adj"].append(bw[-1] - wb); bpix[bn]["raw"].append(_win(dm[i]) - wb)
                    rec = {"n": len(gs), "n_zero_diff": int(np.sum(r == 0)), "raw_win_rate": _win(dm), "adj_win_rate": _win(r),
                           "adj_win_rate_boot_sd": float(np.std(bw)),
                           "adj_win_rate_ci95": [float(np.quantile(bw, 0.025)), float(np.quantile(bw, 0.975))],
                           "adj_median_log_ratio": float(np.median(r)),
                           "adj_median_ci95": [float(np.quantile(bmed, 0.025)), float(np.quantile(bmed, 0.975))],
                           "adj_wilcoxon_p": float(wilcoxon(r).pvalue) if np.any(r != 0) else 1.0}
                    q = lambda x: [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]
                    rec["vs_pixel"] = {bn: {"baseline_raw_win": _win(bd),
                                            "adj_minus_baseline": rec["adj_win_rate"] - _win(bd),
                                            "adj_minus_baseline_ci95": q(bpix[bn]["adj"]),
                                            "raw_minus_baseline": rec["raw_win_rate"] - _win(bd),
                                            "raw_minus_baseline_ci95": q(bpix[bn]["raw"])}
                                       for bn, bd in pix.items()}
                    if con.get("magnitude") and labels is not None:      # dose-response
                        mag = np.array([contrast_magnitude(con["magnitude"], labels(v), labels(l)) for v, l in pairs])
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
    # summary: under the primary covariate set, the distance is selected on held-out groups among the distances
    # whose legal-vs-legal control sits at chance (a validity filter that never looks at a violation)
    valid = [m for m in metrics if out["contrasts"][primary][ctrl_key][m]["control_at_chance"]]
    for key in out["contrasts"].get(primary, {}):
        ms = [m for m in valid if (primary, key, m) in resid]
        if not ms:
            out["summary"][key] = {"verdict": "inconclusive (no distance has its control at chance)",
                                   "n_variants": 0, "covariates": primary}
            continue

        def score(m, gids, key=key):
            r = resid[(primary, key, m)]
            v = np.array([r[g] for g in gids if g in r])
            return _win(v) if len(v) else float("nan")
        sel = heldout_select(score, ms, list(resid[(primary, key, ms[0])]), n_splits=n_splits)
        rec = out["contrasts"][primary][key][sel["selected"]]
        out["summary"][key] = {**sel, "covariates": primary, "n_distances_total": len(metrics),
                               "primary": key == out["primary_contrast"],
                               "vs_pixel": rec.get("vs_pixel", {}),
                               "full_sample_adj_win_rate": rec["adj_win_rate"], "full_sample_ci95": rec["adj_win_rate_ci95"],
                               "verdict": rec.get("verdict", "control")}
    return out


# --------------------------------------------------------------------------- F2 group contrasts
# One F2 twin group = the legal restitution ladder (e = 0.0 .. 1.0) + momentum violators mom_violate_e<e>_<pos|neg>
# (twins.f2_violators), all sharing the pre-contact frames. Two contrasts use the whole group:
#   speed_matched  violator vs the ladder twin whose post-contact KE (or total speed) is closest, so speed cannot
#                  separate the pair. Matching quality: median |log KE_viol / KE_legal|; pairs above
#                  ``max_log_gap`` are left out (counted as unmatched: e.g. pos violators at e = 1 have more KE than
#                  any legal outcome), and with ``require_bracketed`` so are violators outside the ladder's range.
#                  ``balance_faster`` weights rows so violators faster / slower than their partner (in the match
#                  scalar) count equally: an observer of that scalar alone then sits at 0.5. Wilcoxon p uses
#                  unweighted per-group means.
#   family_min     violator wins if more surprising than the LEAST surprising ladder twin (the model's best-fitting
#                  legal outcome). Chance level is NOT 0.5: an exchangeable violator beats the minimum of K legal
#                  twins with probability K / (K + 1) (``chance_win``). Rank = share of ladder twins less surprising
#                  than the violator (ties 1/2): 1.0 = most surprising, chance 0.5.
# Motion adjustment: per clip a = log s − β·log c, β fitted through the origin on the ``ladder_calibration`` pairs
# (legal-vs-legal, Δlog s = β·Δlog c); the ``heldout_controls`` (adjacent ladder levels, never in the fit) must
# have an adjusted win-rate CI containing 0.5, else the contrast is ``inconclusive``. Rows are violator clips;
# several per group, so every CI is a bootstrap over GROUPS (β refitted per draw, the same draw for model and
# pixel baselines) and the Wilcoxon test runs on per-group means.

GROUP_KINDS = ("speed_matched", "family_min")
PENDING_KINDS = ("energy_kink",)      # configured (twins in e order) but no statistic yet: reported as pending


def group_spec(cfg, family: str) -> dict | None:
    """The ``contrasts.<family>`` group-contrast block, or None."""
    c = cfg["contrasts"]
    fam = c.get(family) if isinstance(c, dict) else None
    if not fam or not fam.get("group_contrasts"):
        return None
    return {"ladder": [str(x) for x in fam["ladder"]],
            "calibration_pairs": [tuple(p) for p in fam["ladder_calibration"]],
            "controls": [tuple(p) for p in fam.get("heldout_controls", [])],
            "contrasts": [dict(x) for x in fam["group_contrasts"] if x["kind"] not in PENDING_KINDS],
            "pending": [dict(x) for x in fam["group_contrasts"] if x["kind"] in PENDING_KINDS]}


def _post(st, o) -> np.ndarray:
    return np.stack([st.v_a, st.v_b]) + np.asarray(o, float).reshape(2, 2)


def kin_scalar(st, o, by: str = "ke") -> float:
    """Post-contact KE (``ke``) or total speed |v_A| + |v_B| (``speed``) of outcome o = (dv_A, dv_B)."""
    v = _post(st, o)
    if by == "ke":
        return float(0.5 * st.m_a * v[0] @ v[0] + 0.5 * st.m_b * v[1] @ v[1])
    if by == "speed":
        return float(np.linalg.norm(v, axis=1).sum())
    raise ValueError(by)


def eta_outcome(st, o) -> float:
    q = np.array([st.m_a, st.m_b])[:, None] * np.asarray(o, float).reshape(2, 2)
    den = float(np.linalg.norm(q, axis=1).sum())
    return float(np.linalg.norm(q.sum(0)) / den) if den > 0 else float("nan")


def match_partner(st, o_viol, ladder_out: dict, by: str = "ke") -> tuple[str, float, float]:
    """(ladder twin closest to the violator in log ``by``, |log KE ratio|, |log speed ratio|)."""
    x = np.log(kin_scalar(st, o_viol, by))
    best = min(ladder_out, key=lambda k: abs(np.log(kin_scalar(st, ladder_out[k], by)) - x))
    gk = abs(np.log(kin_scalar(st, o_viol, "ke") / kin_scalar(st, ladder_out[best], "ke")))
    gs = abs(np.log(kin_scalar(st, o_viol, "speed") / kin_scalar(st, ladder_out[best], "speed")))
    return best, float(gk), float(gs)


def _segment_dist(o, d: np.ndarray, k0: float, k1: float) -> float:
    """Distance from o to the segment {k d : k in [k0, k1]} in R^4."""
    o = np.asarray(o, float).reshape(4)
    k = float(np.clip(o @ d / (d @ d), k0, k1))
    return float(np.linalg.norm(o - k * d))


def group_observers(st, e_range=(0.0, 1.0)) -> dict:
    """name -> surprise(o) for the labels-only reference observers (4-D Δv space, as ``ideal_observer_bands``):
    ``perfect`` distance to the legal segment e in e_range (every ladder twin scores 0); ``law_e{e}`` distance to
    the legal outcome at assumed e; ``mass_blind`` distance to the mass-blind segment (the relative-velocity change
    (1+e) u_n n split equally, e in e_range); ``speed_e{e}`` |total post speed − total speed of the legal outcome at
    e| (a speed-only observer: knows how fast things should go, not where); ``ke_e{e}`` the same with post KE."""
    from physlaw.data import collision_physics as cp
    u, n = st.approach_speed, st.n
    mb = np.concatenate([-n / 2, n / 2])
    obs = {"perfect": lambda o: cp.distance_to_legal_line(o, st, segment=e_range)}
    for e in IDEAL_E:
        obs[f"law_e{e:g}"] = (lambda o, ol=cp.legal_outcome(st, e): float(np.linalg.norm(np.asarray(o, float) - ol)))
    obs["mass_blind"] = lambda o: _segment_dist(o, mb, (1 + e_range[0]) * u, (1 + e_range[1]) * u)
    for e in IDEAL_E:
        obs[f"speed_e{e:g}"] = (lambda o, s0=kin_scalar(st, cp.legal_outcome(st, e), "speed"): abs(kin_scalar(st, o, "speed") - s0))
    for e in IDEAL_E:
        obs[f"ke_e{e:g}"] = (lambda o, k0=kin_scalar(st, cp.legal_outcome(st, e), "ke"): abs(kin_scalar(st, o, "ke") - k0))
    return obs


def build_rows(kin: dict, spec: dict) -> dict:
    """Label-only bookkeeping per group contrast. kin: group -> (PreState, {twin: o4}).
    Returns {contrast name: [row dicts]}; a row is one violator clip with its legal partner(s)."""
    out = {}
    for con in spec["contrasts"]:
        name, kind = con.get("name", con["kind"]), con["kind"]
        by, gap = con.get("match", "ke"), con.get("max_log_gap")
        prefix = con.get("viol_prefix", "mom_violate")
        rows = []
        for g, (st, outs) in kin.items():
            lad = {t: outs[t] for t in spec["ladder"] if t in outs}
            if not lad:
                continue
            for v in sorted(t for t in outs if t.startswith(prefix)):
                r = {"group": g, "viol": v, "sign": v.rsplit("_", 1)[-1], "eta": eta_outcome(st, outs[v]),
                     "energy_physical": kin_scalar(st, outs[v], "ke") <= kin_scalar(st, np.zeros(4), "ke") + 1e-12}
                if kind == "speed_matched":
                    partner, gk, gs = match_partner(st, outs[v], lad, by)
                    xs = [kin_scalar(st, o, by) for o in lad.values()]
                    bracketed = min(xs) <= kin_scalar(st, outs[v], by) <= max(xs)
                    r.update(legal=[partner], log_gap_ke=gk, log_gap_speed=gs, bracketed=bool(bracketed),
                             matched=(gap is None or (gk if by == "ke" else gs) <= float(gap))
                             and (bracketed or not con.get("require_bracketed", False)))
                elif kind == "family_min":
                    r.update(legal=list(lad), matched=True)
                else:
                    raise ValueError(f"unknown group contrast kind {kind!r} (have {GROUP_KINDS})")
                rows.append(r)
        if kind == "speed_matched" and con.get("balance_faster", False):
            # faster and slower violators (vs their partner, in the matching scalar) get equal total weight: an
            # observer that only knows the expected speed at a ladder edge then sits at exactly 0.5
            for r in rows:
                st, outs = kin[r["group"]]
                r["faster"] = kin_scalar(st, outs[r["viol"]], by) > kin_scalar(st, outs[r["legal"][0]], by)
            mr = [r for r in rows if r["matched"]]
            nf = sum(r["faster"] for r in mr)
            for r in mr:
                k = nf if r["faster"] else len(mr) - nf
                r["weight"] = len(mr) / (2.0 * k) if 0 < nf < len(mr) else 1.0
        out[name] = rows
    return out


def _row_subsets(rows: list[dict]) -> dict:
    """{subset name: boolean mask}: all, by drift sign, by energy_physical, by violator twin."""
    m = lambda f: np.array([bool(f(r)) for r in rows], bool)
    out = {"all": m(lambda r: True)}
    for sg in sorted({r["sign"] for r in rows}):
        out[f"sign={sg}"] = m(lambda r, sg=sg: r["sign"] == sg)
    out["energy_physical"] = m(lambda r: r["energy_physical"])
    out["not energy_physical"] = ~out["energy_physical"]
    for v in sorted({r["viol"] for r in rows}):
        out[v] = m(lambda r, v=v: r["viol"] == v)
    return {k: v for k, v in out.items() if v.any()}


def _row_index(rows: list[dict], clip_ix: dict) -> tuple[np.ndarray, np.ndarray]:
    """(iv [n], il [n, K]): clip indices of each row's violator and legal twins; -1 = missing."""
    K = max(len(r["legal"]) for r in rows)
    iv = np.array([clip_ix.get(f"{r['group']}/{r['viol']}", -1) for r in rows], int)
    il = np.full((len(rows), K), -1, int)
    for i, r in enumerate(rows):
        for j, t in enumerate(r["legal"]):
            il[i, j] = clip_ix.get(f"{r['group']}/{t}", -1)
    return iv, il


def score_rows(kind: str, a: np.ndarray, iv: np.ndarray, il: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Per row (win, rank) from a per-clip surprise array ``a`` (NaN = missing clip; ties count 1/2).
    speed_matched: win = s_viol > s_partner. family_min: win = s_viol > min over the ladder; rank = share of
    ladder twins less surprising than the violator (1.0 = most surprising). NaN where a row is not scorable."""
    ax = np.append(np.asarray(a, float), np.nan)        # index -1 -> NaN
    sv, SL = ax[iv], ax[il]
    if kind == "speed_matched":
        sl = SL[:, 0]
        with np.errstate(invalid="ignore"):
            win = np.where(np.isclose(sv, sl, rtol=1e-12, atol=1e-12), 0.5, (sv > sl).astype(float))
        win[~(np.isfinite(sv) & np.isfinite(sl))] = np.nan
        return win, None
    fin = np.isfinite(SL)
    cnt = fin.sum(1)
    mn = np.where(fin, SL, np.inf).min(1)
    with np.errstate(invalid="ignore"):
        win = np.where(np.isclose(sv, mn, rtol=1e-12, atol=1e-12), 0.5, (sv > mn).astype(float))
        rank = ((fin & (SL < sv[:, None])).sum(1) + 0.5 * (fin & (SL == sv[:, None])).sum(1)) / np.maximum(cnt, 1)
    bad = ~np.isfinite(sv) | (cnt == 0)
    win[bad] = np.nan
    rank[bad] = np.nan
    return win, rank


def _wmean(x: np.ndarray, grp: np.ndarray, wg: np.ndarray, mask: np.ndarray | None = None,
           rw: np.ndarray | None = None) -> float:
    """Mean of row values with per-group bootstrap weights (a group drawn k times counts k times) and
    optional row weights ``rw`` (speed_matched ``balance_faster``)."""
    ok = np.isfinite(x) if mask is None else (mask & np.isfinite(x))
    w = wg[grp[ok]] * (1.0 if rw is None else rw[ok])
    s = w.sum()
    return float((w * x[ok]).sum() / s) if s > 0 else float("nan")


def _avg(x: np.ndarray, mask: np.ndarray, rw: np.ndarray) -> float:
    """Row-weighted mean over ``mask`` (NaN rows skipped)."""
    return _wmean(x, np.zeros(len(x), int), np.ones(1), mask, rw)


def _rw(rows: list[dict]) -> np.ndarray:
    return np.array([r.get("weight", 1.0) for r in rows], float)


def _q(x) -> list[float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))] if len(x) else [float("nan")] * 2


def _weights(n_groups: int, n_boot: int, seed: int) -> np.ndarray:
    """[n_boot, n_groups] multiplicities of a bootstrap over groups."""
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups) for _ in range(n_boot)])


def _wilcoxon_groups(d: np.ndarray, grp: np.ndarray) -> float:
    from scipy.stats import wilcoxon
    ok = np.isfinite(d)
    gm = np.array([d[ok & (grp == g)].mean() for g in np.unique(grp[ok])])
    return float(wilcoxon(gm).pvalue) if len(gm) > 1 and np.any(gm != 0) else 1.0


def ideal_group_bands(kin: dict, rows_by: dict, spec: dict) -> dict:
    """Labels-only win rates (and family_min mean ranks) of the reference observers (``group_observers``),
    per group contrast and row subset."""
    obs = {g: group_observers(st) for g, (st, _) in kin.items()}
    clips = [(g, t) for g, (_, outs) in kin.items() for t in outs]
    clip_ix = {f"{g}/{t}": i for i, (g, t) in enumerate(clips)}
    names = list(next(iter(obs.values()))) if obs else []
    a = {ob: np.array([obs[g][ob](kin[g][1][t]) for g, t in clips]) for ob in names}
    out = {}
    for con in spec["contrasts"]:
        name, kind = con.get("name", con["kind"]), con["kind"]
        rows = [r for r in rows_by[name] if r["matched"]]
        if not rows:
            continue
        iv, il = _row_index(rows, clip_ix)
        subs, rw = _row_subsets(rows), _rw(rows)
        res = {"observers": names, "n": len(rows), "win_rate": {}, "rank_mean": {}}
        if kind == "family_min":
            K = np.array([len(r["legal"]) for r in rows])
            res["chance_win"] = float(np.mean(K / (K + 1)))
        for ob in names:
            win, rank = score_rows(kind, a[ob], iv, il)
            res["win_rate"][ob] = {k: _avg(win, m, rw) for k, m in subs.items()}
            if rank is not None:
                res["rank_mean"][ob] = {k: _avg(rank, m, rw) for k, m in subs.items()}
        if kind == "speed_matched":
            sp = np.array([kin_scalar(kin[g][0], kin[g][1][t], "speed") for g, t in clips])
            res["viol_faster_than_partner"] = float(np.mean(sp[iv] > sp[il[:, 0]]))
            res["viol_faster_than_partner_weighted"] = _avg((sp[iv] > sp[il[:, 0]]).astype(float), np.ones(len(rows), bool), rw)
        out[name] = res
    return out


def group_contrasts(kin: dict, surprise: dict, spec: dict, cov: dict | None = None, model: str | None = None,
                    primary: str = "speed+copy_last", n_boot: int = 1000, seed: int = 0) -> dict:
    """F2 speed_matched / family_min contrasts.

    kin: group -> (PreState, {twin: o4}) from labels; surprise: source -> clip -> {metric: value} (clip =
    "group/twin"); cov: clip -> {covariate: positive value} incl. the pixel-baseline surprises (motion
    adjustment and pixel-baseline gap; skipped when None); model: the source to motion-adjust."""
    from scipy.stats import spearmanr
    rows_by = build_rows(kin, spec)
    out = {"ladder": spec["ladder"], "calibration_pairs": [list(p) for p in spec["calibration_pairs"]],
           "heldout_controls": [list(p) for p in spec["controls"]], "contrasts": {},
           "chance_note": "speed_matched: chance win 0.5. family_min: chance win K/(K+1) for K ladder twins "
                          "(an exchangeable violator rarely is the least surprising), chance rank 0.5.",
           "ideal_observer": ideal_group_bands(kin, rows_by, spec)}
    for con in spec.get("pending", []):
        name = con.get("name", con["kind"])
        out["contrasts"][name] = {"kind": con["kind"], "config": con, "status": "pending (statistic not implemented)",
                                  "n_groups_complete": sum(all(t in outs for t in con.get("twins", [])) for _, outs in kin.values())}
    for con in spec["contrasts"]:
        name, kind = con.get("name", con["kind"]), con["kind"]
        all_rows = rows_by[name]
        rows = [r for r in all_rows if r["matched"]]
        rec = {"kind": kind, "config": con, "n_rows": len(rows), "n_unmatched": len(all_rows) - len(rows),
               "n_groups": len({r["group"] for r in rows}), "raw": {}}
        if kind == "speed_matched" and rows:
            gk = np.array([r["log_gap_ke"] for r in rows])
            pc: dict = {}
            for r in rows:
                pc[r["legal"][0]] = pc.get(r["legal"][0], 0) + 1
            rec["matching"] = {"by": con.get("match", "ke"), "max_log_gap": con.get("max_log_gap"),
                               "require_bracketed": bool(con.get("require_bracketed", False)),
                               "share_bracketed_all_rows": float(np.mean([r["bracketed"] for r in all_rows])),
                               "balance_faster": bool(con.get("balance_faster", False)),
                               "share_faster_in_match_scalar": float(np.mean([r.get("faster", np.nan) for r in rows])) if con.get("balance_faster") else None,
                               "median_abs_log_ke_ratio": float(np.median(gk)), "p90_abs_log_ke_ratio": float(np.quantile(gk, 0.9)),
                               "median_abs_log_speed_ratio": float(np.median([r["log_gap_speed"] for r in rows])),
                               "median_abs_log_ke_ratio_all_rows": float(np.median([r["log_gap_ke"] for r in all_rows])),
                               "partner_counts": dict(sorted(pc.items()))}
        if kind == "family_min" and rows:
            K = np.array([len(r["legal"]) for r in rows])
            rec["chance_win"] = float(np.mean(K / (K + 1)))
        out["contrasts"][name] = rec
        if not rows:
            continue
        ug = sorted({r["group"] for r in rows})
        grp = np.array([ug.index(r["group"]) for r in rows])
        W = _weights(len(ug), n_boot, seed)
        subs, rw = _row_subsets(rows), _rw(rows)
        eta = np.array([r["eta"] for r in rows])
        for src, recs in surprise.items():
            clips = sorted(recs)
            clip_ix = {c: i for i, c in enumerate(clips)}
            iv, il = _row_index(rows, clip_ix)
            for m in sorted({m for c in recs.values() for m in c}):
                a = np.array([recs[c].get(m, np.nan) for c in clips], float)
                win, rank = score_rows(kind, a, iv, il)
                d = np.append(a, np.nan)[iv] - np.append(a, np.nan)[il[:, 0]] if kind == "speed_matched" else None
                for sub, mask in subs.items():
                    ok = mask & np.isfinite(win)
                    if not ok.any():
                        continue
                    r_ = {"n": int(ok.sum()), "n_groups": int(len(np.unique(grp[ok]))), "win_rate": _avg(win, ok, rw),
                          "win_rate_ci95": _q([_wmean(win, grp, w, mask, rw) for w in W])}
                    if rank is not None:
                        r_.update(rank_mean=_avg(rank, ok, rw), rank_ci95=_q([_wmean(rank, grp, w, mask, rw) for w in W]),
                                  chance_win=rec["chance_win"])
                        dose = rank
                    else:
                        r_["wilcoxon_p_group_means"] = _wilcoxon_groups(np.where(ok, d, np.nan), grp)
                        dose = d
                    if np.std(eta[ok]) > 0 and np.std(dose[ok]) > 0:
                        sp = spearmanr(dose[ok], eta[ok])
                        r_["dose_spearman_eta"], r_["dose_spearman_p"] = float(sp.statistic), float(sp.pvalue)
                    rec["raw"].setdefault(src, {}).setdefault(m, {})[sub] = r_
    if cov is not None and model in surprise:
        out["motion_adjusted"] = _group_motion_adjusted(rows_by, surprise[model], cov, spec, primary, n_boot, seed)
    return out


def _group_motion_adjusted(rows_by: dict, sur: dict, cov: dict, spec: dict, primary: str, n_boot: int, seed: int) -> dict:
    """β fitted (through the origin) on the ladder calibration pairs; adjusted log surprise a = log s − β·log c per
    clip, scored like the raw contrasts; held-out adjacent-level controls; pixel-baseline gap (model adjusted / raw
    win − baseline raw win). CIs: bootstrap over groups, β refitted per draw, one draw shared by model and baseline."""
    clips = sorted(c for c in sur if c in cov)
    clip_ix = {c: i for i, c in enumerate(clips)}
    ug = sorted({c.split("/")[0] for c in clips})
    gpos = {g: i for i, g in enumerate(ug)}
    W = _weights(len(ug), n_boot, seed)
    metrics = sorted({m for c in clips for m in sur[c] if m.split("_")[-1] in ("obj", "all")})
    tasks = {}           # name -> (kind, rows, iv, il, grp, subsets)
    for name, rows in [*((f"control: {a} vs {b}", [{"group": g, "viol": a, "legal": [b]} for g in ug])
                         for a, b in spec["controls"]),
                       *((con.get("name", con["kind"]), [r for r in rows_by[con.get("name", con["kind"])] if r["matched"]])
                         for con in spec["contrasts"])]:
        if not rows:
            continue
        kind = "speed_matched" if name.startswith("control: ") else next(
            c["kind"] for c in spec["contrasts"] if c.get("name", c["kind"]) == name)
        iv, il = _row_index(rows, clip_ix)
        subs = {"all": np.ones(len(rows), bool)} if name.startswith("control: ") else _row_subsets(rows)
        tasks[name] = (kind, rows, iv, il, np.array([gpos.get(r["group"], 0) for r in rows]), subs, _rw(rows))
    cal = [(clip_ix.get(f"{g}/{a}", -1), clip_ix.get(f"{g}/{b}", -1), gpos[g]) for g in ug for a, b in spec["calibration_pairs"]]
    cal = np.array([x for x in cal if x[0] >= 0 and x[1] >= 0], int).reshape(-1, 3)
    out = {"covariate_sets": COVARIATE_SETS, "primary_covariates": primary, "pixel_baselines": PIXEL_BASELINES,
           "beta_caveat": BETA_CAVEAT, "fit": {}, "controls": {}, "contrasts": {}}
    for cs, keys_t in COVARIATE_SETS.items():
        for m in metrics:
            region = m.split("_")[-1]
            keys = [k.format(region=region) for k in keys_t]
            ls = np.array([np.log(sur[c][m]) if sur[c].get(m, 0) > 0 else np.nan for c in clips])
            LC = np.array([[np.log(cov[c][k]) if cov[c].get(k, 0) > 0 else np.nan for k in keys] for c in clips])
            y = ls[cal[:, 0]] - ls[cal[:, 1]]
            X = LC[cal[:, 0]] - LC[cal[:, 1]]
            fit_ok = np.isfinite(y) & np.isfinite(X).all(1)
            if fit_ok.sum() < 2:
                continue
            y, X, cg = y[fit_ok], X[fit_ok], cal[fit_ok, 2]
            beta = _fit(y, X)
            r2 = 1 - ((y - X @ beta) ** 2).sum() / (y ** 2).sum()
            out["fit"].setdefault(cs, {})[m] = {"covariates": keys, "beta": beta.tolist(), "r2_uncentred": float(r2),
                                                "n_pairs": int(len(y)), "n_groups": int(len(np.unique(cg)))}
            betas = []
            for w in W:
                sw = np.sqrt(w[cg])[:, None]
                betas.append(_fit(y * sw[:, 0], X * sw) if (w[cg] > 0).sum() > 1 else beta)
            adj = lambda b: ls - LC @ b
            A0 = adj(beta)
            A_k = [adj(b) for b in betas]
            pixel = {bn: np.array([cov[c].get(bk.format(region=region), np.nan) for c in clips], float)
                     for bn, bk in PIXEL_BASELINES.items()}
            ctrl_ok = True
            for name, (kind, rows, iv, il, grp, subs, rw) in tasks.items():
                w0, rk0 = score_rows(kind, A0, iv, il)
                wr, rkr = score_rows(kind, np.exp(ls), iv, il)
                wk = [score_rows(kind, A, iv, il) for A in A_k]
                pw = {bn: score_rows(kind, B, iv, il) for bn, B in pixel.items() if np.isfinite(B).any()}
                for sub, mask in subs.items():
                    ok = mask & np.isfinite(w0)
                    if not ok.any():
                        continue
                    bw = [_wmean(x[0], grp, w, mask, rw) for x, w in zip(wk, W)]
                    r_ = {"n": int(ok.sum()), "n_groups": int(len(np.unique(grp[ok]))), "raw_win_rate": _avg(wr, mask, rw),
                          "adj_win_rate": _avg(w0, ok, rw), "adj_win_rate_ci95": _q(bw)}
                    if name.startswith("control: "):
                        r_["control_at_chance"] = bool(r_["adj_win_rate_ci95"][0] <= 0.5 <= r_["adj_win_rate_ci95"][1])
                        ctrl_ok &= r_["control_at_chance"]
                        out["controls"].setdefault(cs, {}).setdefault(m, {})[name[len("control: "):]] = r_
                        continue
                    if rk0 is not None:
                        brk = [_wmean(x[1], grp, w, mask, rw) for x, w in zip(wk, W)]
                        K = np.array([len(r["legal"]) for r in rows])[mask]
                        r_.update(raw_rank_mean=_avg(rkr, mask, rw), adj_rank_mean=_avg(rk0, ok, rw),
                                  adj_rank_ci95=_q(brk), chance_win=float(np.mean(K / (K + 1))))
                    r_["vs_pixel"] = {}
                    for bn, (pwin, prk) in pw.items():
                        if not np.isfinite(pwin[mask]).any():
                            continue
                        pb = _avg(pwin, mask, rw)
                        braw = [_wmean(wr, grp, w, mask, rw) for w in W]
                        bpb = [_wmean(pwin, grp, w, mask, rw) for w in W]
                        g = {"baseline_raw_win": pb,
                             "adj_minus_baseline": r_["adj_win_rate"] - pb,
                             "adj_minus_baseline_ci95": _q(np.subtract(bw, bpb)),
                             "raw_minus_baseline": r_["raw_win_rate"] - pb,
                             "raw_minus_baseline_ci95": _q(np.subtract(braw, bpb))}
                        if prk is not None:
                            bpr = [_wmean(prk, grp, w, mask, rw) for w in W]
                            pr = _avg(prk, mask, rw)
                            g.update(baseline_rank_mean=pr, adj_rank_minus_baseline=r_["adj_rank_mean"] - pr,
                                     adj_rank_minus_baseline_ci95=_q(np.subtract(brk, bpr)))
                        r_["vs_pixel"][bn] = g
                    out["contrasts"].setdefault(cs, {}).setdefault(name, {}).setdefault(m, {})[sub] = r_
            for by_m in out["contrasts"].get(cs, {}).values():      # verdicts need every control of (cs, m)
                for s in by_m.get(m, {}).values():
                    chance, (lo, hi) = s.get("chance_win", 0.5), s["adj_win_rate_ci95"]
                    s["verdict"] = ("inconclusive (held-out legal control not at chance)" if not ctrl_ok else
                                    "more surprised by violation" if lo > chance else
                                    "less surprised by violation" if hi < chance else "no difference")
    return out


def group_table(gc: dict) -> str:
    """md for the F2 group contrasts: matching, ideal observers, raw (all sources), motion-adjusted (primary set)."""
    rows = ["## F2 group contrasts (violators vs the legal restitution ladder)", "", gc.get("chance_note", ""), ""]
    for name, c in gc["contrasts"].items():
        if "status" in c:
            rows += [f"### {name} ({c['kind']}): {c['status']}; {c['n_groups_complete']} groups have every twin", ""]
            continue
        rows.append(f"### {name} ({c['kind']}): {c['n_rows']} violator clips in {c['n_groups']} groups"
                    + (f", {c['n_unmatched']} unmatched (left out)" if c.get("n_unmatched") else "")
                    + (f"; chance win {c['chance_win']:.2f}" if "chance_win" in c else ""))
        rows.append("")
        if "matching" in c:
            mt = c["matching"]
            rows.append(f"Matching by {mt['by']} (max |log gap| {mt['max_log_gap']}): median |log KE_viol/KE_legal| "
                        f"{mt['median_abs_log_ke_ratio']:.3f} (p90 {mt['p90_abs_log_ke_ratio']:.3f}; all rows before the cut "
                        f"{mt['median_abs_log_ke_ratio_all_rows']:.3f}), median |log speed ratio| {mt['median_abs_log_speed_ratio']:.3f}; "
                        f"partners {mt['partner_counts']}")
            rows.append("")
        io = gc.get("ideal_observer", {}).get(name)
        if io:
            subs = list(next(iter(io["win_rate"].values())))
            rows += [f"Ideal observers (labels only, n={io['n']}): win rate" + (" / mean rank" if io["rank_mean"] else "")
                     + (f"; violator faster than its partner in {io['viol_faster_than_partner']:.0%}" if "viol_faster_than_partner" in io else ""),
                     "", "| observer | " + " | ".join(subs) + " |", "|---|" + "---|" * len(subs)]
            for ob in io["observers"]:
                cells = [f"{io['win_rate'][ob][s]:.2f}" + (f" / {io['rank_mean'][ob][s]:.2f}" if io["rank_mean"] else "") for s in subs]
                rows.append(f"| {ob} | " + " | ".join(cells) + " |")
            rows.append("")
        rk = c["kind"] == "family_min"
        rows += ["| source | distance | subset | n (groups) | win [95% CI] |" + (" mean rank [95% CI] |" if rk else " Wilcoxon p (group means) |")
                 + " dose ρ(η) |", "|---|---|---|---|---|---|---|"]
        for src, by_m in c["raw"].items():
            for m, by_s in by_m.items():
                for sub, s in by_s.items():
                    extra = (f"{s['rank_mean']:.2f} {_ci(s['rank_ci95'])}" if rk else f"{s.get('wilcoxon_p_group_means', float('nan')):.2g}")
                    dose = f"{s['dose_spearman_eta']:+.2f}" if "dose_spearman_eta" in s else ""
                    rows.append(f"| {src} | {m} | {sub} | {s['n']} ({s['n_groups']}) | {s['win_rate']:.2f} {_ci(s['win_rate_ci95'])} | {extra} | {dose} |")
        rows.append("")
    ma = gc.get("motion_adjusted")
    if ma:
        cs = ma["primary_covariates"]
        rows += [f"### Motion-adjusted (covariates {cs}; β fitted on ladder pairs {gc['calibration_pairs']})", "",
                 "Held-out legal controls (adjacent ladder levels, not in the fit; adjusted win must contain 0.5):", "",
                 "| distance | control | n | raw win | adj. win [95% CI] | at chance |", "|---|---|---|---|---|---|"]
        for m, by_c in ma["controls"].get(cs, {}).items():
            for k, s in by_c.items():
                rows.append(f"| {m} | {k} | {s['n']} | {s['raw_win_rate']:.2f} | {s['adj_win_rate']:.2f} {_ci(s['adj_win_rate_ci95'])} | {s['control_at_chance']} |")
        rows += ["", "| contrast | distance | subset | n | raw win | adj. win [95% CI] | adj. rank [95% CI] | chance | "
                 "adj − copy_last [CI] | adj − const_vel [CI] | verdict |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, by_m in ma["contrasts"].get(cs, {}).items():
            for m, by_s in by_m.items():
                for sub, s in by_s.items():
                    gap = [f"{g['adj_minus_baseline']:+.2f} {_ci(g['adj_minus_baseline_ci95'])}" if g else ""
                           for g in (s["vs_pixel"].get(bn) for bn in ("copy_last", "const_velocity"))]
                    rank = f"{s['adj_rank_mean']:.2f} {_ci(s['adj_rank_ci95'])}" if "adj_rank_mean" in s else ""
                    rows.append(f"| {name} | {m} | {sub} | {s['n']} | {s['raw_win_rate']:.2f} | {s['adj_win_rate']:.2f} "
                                f"{_ci(s['adj_win_rate_ci95'])} | {rank} | {s.get('chance_win', 0.5):.2f} | {gap[0]} | {gap[1]} | {s['verdict']} |")
    return "\n".join(rows)


def group_kinematics(lab, groups: list[str], twins_of) -> dict:
    """group -> (PreState, {twin: o4}) from labels (laws.from_labels; the twins share one pre-state)."""
    from physlaw.data import collision_physics as cp
    from physlaw.eval import laws
    kin = {}
    for g in groups:
        outs, st = {}, None
        for t in twins_of(g):
            pv, qv, ctx = laws.from_labels(lab(f"{g}/{t}"))
            if st is None:
                st = cp.PreState(float(ctx["m"][0]), float(ctx["m"][1]), pv["v"][0], pv["v"][1], ctx["n"])
            outs[t] = (qv["v"] - pv["v"]).reshape(4)
        if st is not None:
            kin[g] = (st, outs)
    return kin


def _ci(x: list) -> str:
    return f"[{x[0]:+.2f}, {x[1]:+.2f}]"


def contrast_lines(info: dict, primary: str | None) -> list[str]:
    """Per contrast: a header (primary marked), its dose magnitude and the ideal-observer bands."""
    rows = []
    for c, i in info.items():
        rows.append(f"- **{c}**{' (PRIMARY momentum contrast)' if c == primary else ''}; dose magnitude: "
                    f"{'η of the violating twin' if i.get('magnitude') == 'eta' else i.get('magnitude') or 'none'}")
        if "ideal_observer" in i:
            rows.append(f"  - {ideal_line(i['ideal_observer'])}")
    return rows


def pixel_gap_table(adj: dict, cs: str | None = None, metrics=None) -> list[str]:
    """Primary motion control: model adjusted (and raw) win − pixel-baseline raw win, paired group bootstrap CI."""
    cs = cs or adj["primary_covariates"]
    rows = [f"### Primary motion control: model − pixel-baseline win rate (model adjusted with covariates {cs})", "",
            "Positive = the model wins the contrast more often than a pixel extrapolator on the same frames "
            "(95% CI: paired bootstrap over groups, β refitted per draw). Caveat: " + adj.get("beta_caveat", BETA_CAVEAT) + ".", "",
            "| contrast | distance | n | adj. win | raw win | copy_last win | adj − copy_last [CI] | raw − copy_last [CI] | "
            "const_vel win | adj − const_vel [CI] | raw − const_vel [CI] |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, by_m in adj["contrasts"].get(cs, {}).items():
        for m, s in by_m.items():
            if metrics and m not in metrics:
                continue
            cells = []
            for bn in ("copy_last", "const_velocity"):
                g = s.get("vs_pixel", {}).get(bn)
                cells += ([f"{g['baseline_raw_win']:.2f}", f"{g['adj_minus_baseline']:+.2f} {_ci(g['adj_minus_baseline_ci95'])}",
                           f"{g['raw_minus_baseline']:+.2f} {_ci(g['raw_minus_baseline_ci95'])}"] if g else ["", "", ""])
            star = " (primary)" if c == adj.get("primary_contrast") else ""
            rows.append(f"| {c}{star} | {m} | {s['n']} | {s['adj_win_rate']:.2f} | {s['raw_win_rate']:.2f} | " + " | ".join(cells) + " |")
    return rows


def adjusted_table(adj: dict, metrics=None) -> str:
    rows = []
    if adj.get("contrast_info"):
        rows += ["### Contrasts (primary first) and labels-only ideal-observer references", "",
                 *contrast_lines(adj["contrast_info"], adj.get("primary_contrast")), ""]
    rows += [*pixel_gap_table(adj, metrics=metrics), ""]
    s0 = next((v for v in adj.get("summary", {}).values() if "selected" in v), None)
    if s0:
        rows += [f"### Summary (covariates {adj['primary_covariates']}; distance selected on {s0['n_select_groups']} groups, "
                 f"adjusted win rate on the other {s0['n_eval_groups']}, {2 * s0['n_splits']} evaluations; "
                 f"variants searched = {s0['n_variants']} of {s0['n_distances_total']} distances (those whose "
                 "legal-vs-legal control is at chance); zero-fit, no folds)", "",
                 "| contrast | selected distance (picks) | held-out adj. win [2.5%, 97.5%] | full-sample adj. win [95% CI] | verdict (Holm) |",
                 "|---|---|---|---|---|"]
        for c, s in adj["summary"].items():
            if "selected" not in s:
                rows.append(f"| {c} | | | | {s['verdict']} |")
                continue
            rows.append(f"| {c}{' (primary)' if s.get('primary') else ''} | {s['selected']} ({s['selected_counts'][s['selected']]}/{2 * s['n_splits']}) | "
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
        rows += ["", f"### covariates: {cs}", "", f"Caveat: {adj.get('beta_caveat', BETA_CAVEAT)}; the pixel-baseline gap "
                 "columns are the primary motion control. Dose ρ: Spearman of the adjusted residual with the contrast's "
                 "magnitude (η of the violating twin for momentum).", "",
                 "| contrast | distance | n | raw win | adjusted win [95% CI] | adj. median log-ratio [95% CI] | Wilcoxon p (Holm) | dose ρ (p) | adj − copy_last | adj − const_vel | verdict |",
                 "|---|---|---|---|---|---|---|---|---|---|---|"]
        for c, by_m in by_c.items():
            for m, s in by_m.items():
                if metrics and m not in metrics:
                    continue
                dose = f"{s['dose_spearman']:+.2f} ({s['dose_spearman_p']:.2g})" if "dose_spearman" in s else ""
                ver = s.get("verdict") or ("control at chance" if s.get("control_at_chance") else "CONTROL NOT AT CHANCE")
                gap = [f"{g['adj_minus_baseline']:+.2f} {_ci(g['adj_minus_baseline_ci95'])}" if g else ""
                       for g in (s.get("vs_pixel", {}).get(bn) for bn in ("copy_last", "const_velocity"))]
                rows.append(f"| {c} | {m} | {s['n']} | {s['raw_win_rate']:.2f} | {s['adj_win_rate']:.2f} "
                            f"[{s['adj_win_rate_ci95'][0]:.2f}, {s['adj_win_rate_ci95'][1]:.2f}] | "
                            f"{s['adj_median_log_ratio']:+.3f} [{s['adj_median_ci95'][0]:+.3f}, {s['adj_median_ci95'][1]:+.3f}] | "
                            f"{s['adj_wilcoxon_p']:.2g} ({s.get('adj_wilcoxon_p_holm', float('nan')):.2g}) | {dose} | "
                            f"{gap[0]} | {gap[1]} | {ver} |")
    return "\n".join(rows)


def table(res: dict) -> str:
    rows = []
    if res.get("contrast_info") and "motion_adjusted" not in res:     # otherwise adjusted_table shows them
        rows += [*contrast_lines(res["contrast_info"], res.get("primary_contrast")), ""]
    rows += ["| contrast | source | distance | n | win rate [95% CI] | Wilcoxon p (Holm) | Cliff's δ | median rel. diff | ρ vs magnitude |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c, by_src in res["contrasts"].items():
        for src, by_m in by_src.items():
            for m, s in by_m.items():
                rho = s.get("spearman_vs_magnitude")
                rows.append(f"| {c}{' (primary)' if c == res.get('primary_contrast') else ''} | {src} | {m} | {s['n']} | {s['win_rate']:.2f} "
                            f"[{s['win_rate_ci95'][0]:.2f}, {s['win_rate_ci95'][1]:.2f}] | {s['wilcoxon_p']:.2g} "
                            f"({s.get('wilcoxon_p_holm', float('nan')):.2g}) | "
                            f"{s['cliffs_delta']:+.2f} | {s['median_rel_diff']:+.1%} | "
                            f"{'' if rho is None else f'{rho:+.2f}'} |")
    if res.get("group_contrasts"):
        rows += ["", group_table(res["group_contrasts"])]
    return "\n".join(rows)
