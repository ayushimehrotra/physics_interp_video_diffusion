"""L1 forecast law consistency (spec §5): do the model's OWN predictions of the post-contact
states obey momentum and energy conservation?

Steps (spec), with the deviations needed for F1 noted:
1. Post-contact velocity probes per object, ridge on encode() FINAL-layer activations of REAL
   post-contact frames (FINAL = the predictor's target space). Deviation: with 64 groups a
   train/test split leaves ~9 test groups, so probes are CROSS-FITTED: grouped 5-fold over twin
   groups, repeated over ``fold_seeds`` seeds, trained on every twin of the training groups;
   every test-group prediction comes from a probe that never saw that group.
2. forecast(): predictor outputs for the post-contact token-frames from the pre-contact frames only
   (cached by stage 4; identical for every twin of a group).
3. Probes applied to the forecast pooled with the same pooling weights -> predicted v_A', v_B'.
4. Law residuals with TRUE masses and TRUE pre-contact velocities:
     momentum r_p = |m_A v̂_A' + m_B v̂_B' − p_pre| / |p_pre|
     energy   r_E = max(0, KE(v̂') − KE_pre) / KE_pre

Calibration:
  * floor     the same probes on encode() of the REAL post-contact frames (probe noise alone)
  * null      both balls continue unchanged (spec). For F1 this conserves p and KE EXACTLY
              (r = 0), so it cannot anchor S. It is reported but not used in S.
  * shuffle   (added) the forecast decoded for a DIFFERENT group, scored against this group's masses
              and pre velocities: a prediction that ignores this scene's physics.
  * S_law     = 1 − (r_forecast − r_floor) / (r_shuffle − r_floor), clipped to [−1, 1].
  * outcome   (added) E = ||v̂' − v'|| / ||v' − v_pre|| over both balls (4-vector): the error of
              the predicted post velocities relative to the true velocity change. The spec's null
              has E = 1 by construction, so S_outcome = 1 − (E_forecast − E_floor)/(1 − E_floor).
              The legal outcome depends on e, which the context does not show (both legal twins
              share the context), so outcome is reported per legal twin.
Pools for the forecast:
  * roi    the evaluated twin's own post-contact ROI (spec). It tells the probe WHERE the ball
           really goes, which leaks the outcome (L0 mask floor: R² up to ~0.4 for velocity).
  * frame  all post-contact tokens: no location leak, weaker probes.
  * *_mm   moment-matched: before the real-frame probe is applied, forecast features are shifted and
           scaled to the real features' per-dimension mean / sd, both estimated on the TRAINING groups
           only, without labels. Needed because predictor outputs are compressed relative to encoder
           outputs (decoded forecast spread ~0.1 of real without it).
Sensitivity check: the floor on the momentum-violating twins should show large r_p (the probes
can see a violation in real video); otherwise a low forecast residual means nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from physlaw.eval.l0_states import complete_groups, load_index
from physlaw.probes.ridge import DEFAULT_ALPHAS, _inner_alpha, _SVDRidge, grouped_kfold

LEGAL = ("conserve_e1", "conserve_e015")
VIOLATING = ("mom_violate_e1", "mom_violate_e015", "energy_create")


def _physics(labels):
    pl = labels["physlaw"]
    m = np.array([pl["objects"][o]["mass"] for o in "AB"])
    vpre = np.array([pl["pre_velocity"][o][:2] for o in "AB"], float)
    vpost = np.array([pl["post_velocity"][o][:2] for o in "AB"], float)
    return m, vpre, vpost


def residuals(v_hat: np.ndarray, m: np.ndarray, vpre: np.ndarray, vpost: np.ndarray) -> dict:
    """v_hat [2, 2] (object, xy) predicted post velocities."""
    p_pre = m @ vpre
    ke_pre = 0.5 * (m * (vpre ** 2).sum(1)).sum()
    ke_hat = 0.5 * (m * (v_hat ** 2).sum(1)).sum()
    return {"r_p": float(np.linalg.norm(m @ v_hat - p_pre) / np.linalg.norm(p_pre)),
            "r_E": float(max(0.0, ke_hat - ke_pre) / ke_pre),
            "E_out": float(np.linalg.norm(v_hat - vpost) / np.linalg.norm(vpost - vpre))}


def _score(rf, rfl, rref):
    d = rref - rfl
    return float(np.clip(1 - (rf - rfl) / d, -1, 1)) if d > 0 else float("nan")


def run(cache_dir: Path, data_dir: Path, cfg, model_cfg, log=print) -> dict:
    probe = cfg["probe"]
    alphas = tuple(float(a) for a in probe.get("alphas", DEFAULT_ALPHAS))
    k = int(probe["k_folds"])
    n_seeds = int(cfg.get("fold_seeds_l1", 4))
    n_boot = int(cfg["bootstrap"]["n"])
    recs = complete_groups(load_index(cache_dir))
    clips = [r["clip"] for r in recs]
    row = {c: i for i, c in enumerate(clips)}
    labels = {c: json.loads((data_dir / c / "labels.json").read_text()) for c in clips}
    groups = np.array([r["group"] for r in recs])
    ugroups = sorted(set(groups))

    # real post-contact pooled FINAL activations (stage 4) and pooling weights (L0's _maskfeat_v2)
    first = np.load(cache_dir / recs[0]["file"])
    pools = [str(p) for p in first["pools"]]
    PP = ("roi_A.post", "roi_B.post", "frame.post")
    zs = [np.load(cache_dir / r["file"]) for r in recs]
    mf = np.load(cache_dir / "_maskfeat_v2.npz")
    if list(mf["clips"]) != clips:
        raise RuntimeError("_maskfeat_v2.npz is for another clip list: run L0 first (it builds the weights)")
    spaces = {}     # name -> (real {pool: [n, D]}, forecast {pool: [n, D]})
    if "fc_pooled" in first.files:
        # diffusion: forecast activations already pooled per twin by stage 4, one space per (SNR, layer) key;
        # the real counterpart is encode() at the same key
        lkeys = [str(l) for l in first["layers"]]
        for q in [str(x) for x in first["fc_keys"]]:
            li, fi = lkeys.index(q), [str(x) for x in first["fc_keys"]].index(q)
            real = {p: np.stack([z["pooled"][pools.index(p), li] for z in zs]).astype(np.float64) for p in PP}
            fc = {p: np.stack([z["fc_pooled"][pools.index(p), fi] for z in zs]).astype(np.float64) for p in PP}
            spaces[q] = (real, fc)
    else:
        # the forecast space: a separate `pooled_final` (VideoMAE decoder, other width) or layer -1 of `pooled`
        if "pooled_final" in first.files:
            get = lambda z, p: z["pooled_final"][pools.index(p)]
        else:
            li = [int(l) for l in first["layers"]].index(-1)
            get = lambda z, p: z["pooled"][pools.index(p), li]
        real = {p: np.stack([get(z, p) for z in zs]).astype(np.float64) for p in PP}
        phases = {c: np.array([str(x) for x in z["phases"]]) for c, z in zip(clips, zs)}
        # forecasts pooled with each twin's weights: [n, D] per pool
        fc = {p: np.zeros_like(real["frame.post"]) for p in real}
        for g in ugroups:
            z = np.load(cache_dir / g / "_forecast.npz")
            F = z["final"].astype(np.float32)                      # [T_post, gh, gw, D]
            t0 = int(z["context_frames"]) // int(model_cfg.token_grid.tubelet)
            for c in [c for c in clips if c.startswith(g + "/")]:
                i = row[c]
                post = phases[c][t0:] == "post"
                for p in ("roi_A.post", "roi_B.post"):
                    w = mf[f"w.{p}"][i].reshape(len(phases[c]), *F.shape[1:3])[t0:] * post[:, None, None]
                    fc[p][i] = (w.reshape(-1) / w.sum()) @ F.reshape(-1, F.shape[-1])
                fc["frame.post"][i] = F[post].reshape(-1, F.shape[-1]).mean(0)
        spaces["final"] = (real, fc)

    roi = {"A": "roi_A.post", "B": "roi_B.post"}
    frame = {"A": "frame.post", "B": "frame.post"}
    base_variants = {"roi": (roi, False), "roi_mm": (roi, True), "frame": (frame, False), "frame_mm": (frame, True)}
    out = {"n_groups": len(ugroups), "fold_seeds": n_seeds, "variants": {}}
    runs = [((sp, vn), pm) for sp in spaces for vn, pm in base_variants.items()]
    for (space, vbase), (pmap, mm) in runs:
        real, fc = spaces[space]
        vname = vbase if len(spaces) == 1 else f"{space}|{vbase}"
        # cross-fitted predictions: real (floor) and forecast, averaged over seeds -> [n, 2 objects, 2]
        pr_real = np.zeros((len(clips), 2, 2)); pr_fc = np.zeros((len(clips), 2, 2))
        r2_real = []
        for oi, o in enumerate("AB"):
            Y = np.array([labels[c]["physlaw"]["post_velocity"][o][:2] for c in clips], float)
            X, Xf = real[pmap[o]], fc[pmap[o]]
            for seed in range(n_seeds):
                oof = np.zeros_like(Y)
                for te in grouped_kfold(groups, k, seed):
                    tr = np.setdiff1d(np.arange(len(clips)), te)
                    a = _inner_alpha(X[tr], Y[tr], groups[tr], alphas, k, seed, False, True)
                    mdl = _SVDRidge(X[tr], Y[tr], True)
                    oof[te] = mdl.predict(X[te], a)
                    xf = Xf[te]
                    if mm:   # moment matching: forecast features to real-feature mean/sd, train-fold stats, no labels
                        mu_r, sd_r = X[tr].mean(0), X[tr].std(0) + 1e-8
                        mu_f, sd_f = Xf[tr].mean(0), Xf[tr].std(0) + 1e-8
                        xf = (xf - mu_f) / sd_f * sd_r + mu_r
                    pr_fc[te, oi] += mdl.predict(xf, a) / n_seeds
                pr_real[:, oi] += oof / n_seeds
                r2_real.append(1 - ((Y - oof) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
        # per (group, twin) residuals
        per = {}
        for c in clips:
            g, t = c.split("/")
            m, vpre, vpost = _physics(labels[c])
            i = row[c]
            per[c] = {"forecast": residuals(pr_fc[i], m, vpre, vpost),
                      "floor": residuals(pr_real[i], m, vpre, vpost),
                      "null": residuals(vpre, m, vpre, vpost)}
        # shuffle: other groups' decoded forecast for the same twin type, against this group's physics
        for c in clips:
            g, t = c.split("/")
            m, vpre, vpost = _physics(labels[c])
            others = [residuals(pr_fc[row[f"{h}/{t}"]], m, vpre, vpost) for h in ugroups if h != g]
            per[c]["shuffle"] = {kk: float(np.mean([o[kk] for o in others])) for kk in others[0]}

        res = {"probe_r2_real_post": float(np.mean(r2_real)), "twins": {}}
        rng = np.random.default_rng(0)
        for t in LEGAL + VIOLATING:
            gs = [g for g in ugroups if f"{g}/{t}" in per]
            arr = {src: {kk: np.array([per[f"{g}/{t}"][src][kk] for g in gs]) for kk in ("r_p", "r_E", "E_out")}
                   for src in ("forecast", "floor", "null", "shuffle")}
            mean = {src: {kk: float(v.mean()) for kk, v in d.items()} for src, d in arr.items()}
            S = {"momentum": _score(mean["forecast"]["r_p"], mean["floor"]["r_p"], mean["shuffle"]["r_p"]),
                 "energy": _score(mean["forecast"]["r_E"], mean["floor"]["r_E"], mean["shuffle"]["r_E"]),
                 "outcome": _score(mean["forecast"]["E_out"], mean["floor"]["E_out"], 1.0)}
            boots = {kk: [] for kk in S}
            for _ in range(n_boot):
                i = rng.integers(0, len(gs), len(gs))
                mb = {src: {kk: v[i].mean() for kk, v in d.items()} for src, d in arr.items()}
                boots["momentum"].append(_score(mb["forecast"]["r_p"], mb["floor"]["r_p"], mb["shuffle"]["r_p"]))
                boots["energy"].append(_score(mb["forecast"]["r_E"], mb["floor"]["r_E"], mb["shuffle"]["r_E"]))
                boots["outcome"].append(_score(mb["forecast"]["E_out"], mb["floor"]["E_out"], 1.0))
            ci = {kk: [float(np.nanquantile(b, 0.025)), float(np.nanquantile(b, 0.975))] for kk, b in boots.items()}
            res["twins"][t] = {"n": len(gs), "mean": mean, "S": S, "S_ci95": ci,
                               "median": {src: {kk: float(np.median(v)) for kk, v in d.items()} for src, d in arr.items()}}
        res["diagnostics"] = _diagnostics(clips, row, labels, groups, ugroups, pr_real, pr_fc,
                                          {o: fc[pmap[o]] for o in "AB"}, alphas, k,
                                          {o: mf[f"w.{pmap[o]}"] for o in "AB"} if pmap is roi else None)
        out["variants"][vname] = res
        log(f"[L1] {vname}: probe R² on real post frames {res['probe_r2_real_post']:.3f}")
    return out


def _corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def _diagnostics(clips, row, labels, groups, ugroups, pr_real, pr_fc, Xfc, alphas, k, mask_w=None) -> dict:
    """Is a bad forecast score the predictor's fault or the probe transfer's?
    On conserve_e1 rows (one per group; the forecast is the same for every twin of a group):
      * corr(decoded forecast, true post v) and corr(decoded forecast, pre v); same for decoded real
      * spread of decoded forecasts relative to decoded real frames (shrinkage => distribution shift)
      * forecast-trained probe: ridge on FORECAST features -> post v (and -> pre v), cross-fitted by
        group. Not the spec's L1 (probe and forecast are no longer independent), but it says whether
        the outcome is linearly present in the forecast at all."""
    rows_ = np.array([row[f"{g}/conserve_e1"] for g in ugroups])
    post = np.array([[labels[clips[i]]["physlaw"]["post_velocity"][o][:2] for o in "AB"] for i in rows_], float)
    pre = np.array([[labels[clips[i]]["physlaw"]["pre_velocity"][o][:2] for o in "AB"] for i in rows_], float)
    d = {"n": len(rows_),
         "corr_forecast_vs_post": _corr(pr_fc[rows_], post), "corr_forecast_vs_pre": _corr(pr_fc[rows_], pre),
         "corr_real_vs_post": _corr(pr_real[rows_], post),
         "spread_ratio_forecast_over_real": float(pr_fc[rows_].std(0).mean() / pr_real[rows_].std(0).mean()),
         "corr_post_vs_pre_truth": _corr(post, pre)}
    g = groups[rows_]
    if mask_w is not None:   # how much of a forecast-trained ROI probe can the ROI's placement alone give?
        sc = []
        for oi, o in enumerate("AB"):
            X, y = mask_w[o][rows_], post[:, oi]
            oof = np.zeros_like(y)
            for te in grouped_kfold(g, k, 0):
                tr = np.setdiff1d(np.arange(len(y)), te)
                a = _inner_alpha(X[tr], y[tr], g[tr], alphas, k, 0, False, True)
                oof[te] = _SVDRidge(X[tr], y[tr], True).predict(X[te], a)
            sc.append(1 - ((y - oof) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())
        d["mask_floor_r2_post_same_n"] = float(np.mean(sc))
    for name, Y in (("post", post), ("pre", pre)):
        sc = []
        for oi, o in enumerate("AB"):
            X = Xfc[o][rows_]
            y = Y[:, oi]
            oof = np.zeros_like(y)
            for te in grouped_kfold(g, k, 0):
                tr = np.setdiff1d(np.arange(len(y)), te)
                a = _inner_alpha(X[tr], y[tr], g[tr], alphas, k, 0, False, True)
                oof[te] = _SVDRidge(X[tr], y[tr], True).predict(X[te], a)
            sc.append(1 - ((y - oof) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())
        d[f"forecast_trained_probe_r2_{name}"] = float(np.mean(sc))
    return d


def table(res: dict) -> str:
    rows = []
    for vname, v in res["variants"].items():
        rows += [f"### forecast pooled by: {vname}  (probe R² on real post-contact frames: {v['probe_r2_real_post']:.3f})", "",
                 "| twin | n | r_p forecast / floor / shuffle / null | S_momentum [95% CI] | r_E forecast / floor / shuffle | S_energy [95% CI] | E_out forecast / floor | S_outcome [95% CI] |",
                 "|---|---|---|---|---|---|---|---|"]
        for t, d in v["twins"].items():
            mn, S, ci = d["mean"], d["S"], d["S_ci95"]
            f = lambda kk: f"{S[kk]:+.2f} [{ci[kk][0]:+.2f}, {ci[kk][1]:+.2f}]"
            rows.append(f"| {t} | {d['n']} | {mn['forecast']['r_p']:.3f} / {mn['floor']['r_p']:.3f} / "
                        f"{mn['shuffle']['r_p']:.3f} / {mn['null']['r_p']:.3f} | {f('momentum')} | "
                        f"{mn['forecast']['r_E']:.3f} / {mn['floor']['r_E']:.3f} / {mn['shuffle']['r_E']:.3f} | {f('energy')} | "
                        f"{mn['forecast']['E_out']:.3f} / {mn['floor']['E_out']:.3f} | {f('outcome')} |")
        dg = v.get("diagnostics")
        if dg:
            rows += ["", "Diagnostics (conserve_e1, one row per group): " + ", ".join(
                f"{k_} = {val:.3f}" if isinstance(val, float) else f"{k_} = {val}" for k_, val in dg.items())]
        rows.append("")
    rows.append("S is computed for every twin for reference, but L1's law test is the LEGAL twins (conserve_*): the "
                "forecast is the same for all twins of a group, while violating twins' floors show whether the probes "
                "can see a violation in real video.")
    return "\n".join(rows)
