"""L1 forecast law consistency (spec §5.2, revised in v2): do the model's OWN predictions of the
post-event states obey the laws?

Pipeline. forecast() gives the model's prediction of the post-event token-frames from pre-event frames
only (cached by stage 4; identical for every twin of a group). A post-velocity probe decodes it into
v̂_A', v̂_B'. Probes are CROSS-FITTED over twin groups (grouped ``k_folds`` x ``fold_seeds_l1`` seeds;
5 x 4 by default, allowed at < 100 groups by spec §5): every group's decode comes from probes that never
saw that group. Laws come from ``physlaw.eval.laws`` with TRUE masses, pre velocities and contact geometry.

Probe options (spec §5.2; all reported):
  * P-real-mm    probe trained on encode() of REAL post-event frames; forecast features moment-matched to
                 the real ones (per-dim mean / sd from training groups, no labels) before decoding.
  * P-pred-real  probe trained on the model's own forecast-space outputs for MASKED REAL post-event tokens
                 with the rest of the clip as context (stage 4 --pred-real sidecars; encoders only). Same
                 space as the forecast, so applied without moment matching.
  * P-forecast   probe trained ON the forecast features (cross-fitted): an information ceiling only,
                 reported as decoding R², never used to score laws.
Pools: ``frame`` (all post-event tokens; leak-free) is PRIMARY. ``roi`` (each ball's ROI at its TRUE
post-event location) is secondary and leaks where the ball goes (its mask floor is reported).

Scores per twin (the forecast is shared by a group's twins; the LEGAL twins are the law test, the
violating twins' floors show whether the probes can see a violation in real video):
  * S_outcome (primary)  E = ||v̂' − v'|| / ||v' − v_pre|| (4-vector); the no-event null has E = 1, so
                         S = 1 − (E_fc − E_floor) / (1 − E_floor). Per legal twin (e is not visible).
  * S_family (primary)   d = laws.momentum.family_distance: distance to the legal family {o(e), e ∈ [0, 1]}
                         normalised so the §5.2 null (pass-through) has d_null = 1;
                         S = 1 − (d_fc − d_floor) / (d_null − d_floor), d_floor = the probes on the real
                         post-event frames (of this option). Also ê, the family's argmin restitution.
  * S_momentum (demoted) a forecast that ignores the collision conserves momentum exactly in F1, so this
                         cannot show understanding on its own. Reference = shuffled-group forecast (built
                         behaviour; STATUS v2 disagreement 4). CoM and angular-momentum residuals carry the
                         same caveat and are reported the same way.
  * energy               scored only when decoded forecast speeds have 80-120% of the spread of the decoded
                         real frames (otherwise "not scorable": shrunk forecasts never create energy).
Bootstrap 95% CIs over groups. ``legal`` pools both legal twins per group.
Selection rule: DiTs give one space per (SNR, layer); for each probe option and pool the space is selected on
half the groups and scored on the other half (probes.select.heldout_select), separately for S_outcome and
S_family. The number of spaces searched is reported.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from physlaw.eval import laws
from physlaw.eval.l0_states import complete_groups, load_index
from physlaw.probes.ridge import DEFAULT_ALPHAS, _inner_alpha, _SVDRidge, grouped_kfold
from physlaw.probes.select import heldout_select

LEGAL = ("conserve_e1", "conserve_e015")
VIOLATING = ("mom_violate_e1", "mom_violate_e015", "energy_create")
RES_LAWS = ("momentum", "energy", "com", "angular_momentum")
POOLS = {"frame": {"A": "frame.post", "B": "frame.post"}, "roi": {"A": "roi_A.post", "B": "roi_B.post"}}
SPREAD_OK = (0.8, 1.2)


def residuals(v_hat: np.ndarray, m: np.ndarray, vpre: np.ndarray, vpost: np.ndarray) -> dict:
    """v_hat [2, 2] predicted post velocities -> {r_p, r_E, E_out} (kept for eval/outputs.py)."""
    pre, ctx = {"v": vpre}, {"m": m}
    post = {"v": np.asarray(v_hat, float)}
    return {"r_p": laws.momentum.residual(pre, post, ctx), "r_E": laws.energy.residual(pre, post, ctx),
            "E_out": float(np.linalg.norm(v_hat - vpost) / np.linalg.norm(vpost - vpre))}


def _S(fc, fl, ref):
    """1 − (fc − fl) / (ref − fl), clipped to [−1, 1]; NaN where ref <= fl. Works on arrays."""
    fc, fl, ref = np.broadcast_arrays(*(np.asarray(a, float) for a in (fc, fl, ref)))
    d = ref - fl
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d > 0, np.clip(1 - (fc - fl) / d, -1, 1), np.nan)


# --------------------------------------------------------------------------- inputs

def _load_spaces(cache_dir, recs, clips, model_cfg, mf):
    """{space: {"real"|"fc"|"pr": {pool: [n, D]}}}; "pr" only where every clip has a .pred_real.npz."""
    first = np.load(cache_dir / recs[0]["file"])
    pools = [str(p) for p in first["pools"]]
    PP = ("roi_A.post", "roi_B.post", "frame.post")
    zs = [np.load(cache_dir / r["file"]) for r in recs]
    spaces = {}
    if "fc_pooled" in first.files:     # diffusion: one space per (SNR, layer) key, pooled per twin by stage 4
        lkeys = [str(l) for l in first["layers"]]
        fkeys = [str(x) for x in first["fc_keys"]]
        for q in fkeys:
            li, fi = lkeys.index(q), fkeys.index(q)
            spaces[q] = {"real": {p: np.stack([z["pooled"][pools.index(p), li] for z in zs]).astype(np.float64) for p in PP},
                         "fc": {p: np.stack([z["fc_pooled"][pools.index(p), fi] for z in zs]).astype(np.float64) for p in PP}}
        return spaces
    if "pooled_final" in first.files:  # forecast space: VideoMAE's decoder (own width) or layer -1 of `pooled`
        get = lambda z, p: z["pooled_final"][pools.index(p)]
    else:
        li = [int(l) for l in first["layers"]].index(-1)
        get = lambda z, p: z["pooled"][pools.index(p), li]
    real = {p: np.stack([get(z, p) for z in zs]).astype(np.float64) for p in PP}
    phases = {c: np.array([str(x) for x in z["phases"]]) for c, z in zip(clips, zs)}
    fc = {p: np.zeros_like(real["frame.post"]) for p in real}
    row = {c: i for i, c in enumerate(clips)}
    for g in sorted({c.split("/")[0] for c in clips}):
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
    spaces["final"] = {"real": real, "fc": fc}
    prf = [cache_dir / r["file"].replace(".npz", ".pred_real.npz") for r in recs]
    if all(f.exists() for f in prf):
        zp = [np.load(f) for f in prf]
        pp = [str(p) for p in zp[0]["pools"]]
        spaces["final"]["pr"] = {p: np.stack([z["pooled"][pp.index(p)] for z in zp]).astype(np.float64) for p in PP}
    return spaces


# --------------------------------------------------------------------------- decoding

def _crossfit(X_train, Y, groups, X_apply, k, n_seeds, alphas, mm_src=None):
    """Cross-fitted ridge: returns (oof decode of X_train, decode of X_apply, mean R² of the oof), both
    averaged over seeds. ``mm_src``: moment-match X_apply to X_train with training-fold stats first."""
    n = len(Y)
    oof_m, app_m, r2 = np.zeros_like(Y), np.zeros_like(Y), []
    for seed in range(n_seeds):
        oof = np.zeros_like(Y)
        for te in grouped_kfold(groups, k, seed):
            tr = np.setdiff1d(np.arange(n), te)
            a = _inner_alpha(X_train[tr], Y[tr], groups[tr], alphas, k, seed, False, True)
            mdl = _SVDRidge(X_train[tr], Y[tr], True)
            oof[te] = mdl.predict(X_train[te], a)
            xa = X_apply[te]
            if mm_src is not None:
                xa = (xa - mm_src[tr].mean(0)) / (mm_src[tr].std(0) + 1e-8) * (X_train[tr].std(0) + 1e-8) + X_train[tr].mean(0)
            app_m[te] += mdl.predict(xa, a) / n_seeds
        oof_m += oof / n_seeds
        r2.append(1 - ((Y - oof) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
    return oof_m, app_m, float(np.mean(r2))


def _ceiling(X, Y, groups, k, alphas):
    """P-forecast: ridge trained on the features themselves, cross-fitted by group (one seed). R²."""
    oof = np.zeros_like(Y)
    for te in grouped_kfold(groups, k, 0):
        tr = np.setdiff1d(np.arange(len(Y)), te)
        a = _inner_alpha(X[tr], Y[tr], groups[tr], alphas, k, 0, False, True)
        oof[te] = _SVDRidge(X[tr], Y[tr], True).predict(X[te], a)
    return float(1 - ((Y - oof) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


# --------------------------------------------------------------------------- scoring

def _per_clip(v_fc, v_fl, states, clips, row, ugroups, res_laws=RES_LAWS):
    """Per clip: forecast / floor / null / shuffle values of every score. v_* [n, 2, 2]."""
    per = {}
    for c in clips:
        pre, post, ctx = states[c]
        i = row[c]
        d = {}
        for src, v in (("forecast", v_fc[i]), ("floor", v_fl[i]), ("null", pre["v"])):
            x = {"v": v, "p": pre["p"]}
            d[src] = {ln: laws.LAWS[ln].residual(pre, x, ctx) for ln in res_laws}
            d[src]["E_out"] = float(np.linalg.norm(v - post["v"]) / np.linalg.norm(post["v"] - pre["v"]))
            d[src]["d_family"], d[src]["e_hat"] = laws.momentum.family_distance(x, pre, ctx)
        per[c] = d
    for c in clips:      # shuffle: other groups' decoded forecast (same twin type) against this group's physics
        g, t = c.split("/")
        pre, _, ctx = states[c]
        others = [v_fc[row[f"{h}/{t}"]] for h in ugroups if h != g]
        per[c]["shuffle"] = {ln: float(np.mean([laws.LAWS[ln].residual(pre, {"v": v, "p": pre["p"]}, ctx) for v in others]))
                             for ln in res_laws}
    return per


def _twin_scores(per, gs_rows, n_boot, rng, energy_ok, res_laws=RES_LAWS):
    """gs_rows: list over groups of the clip keys pooled for that group (1 per twin; 2 for ``legal``)."""
    keys = ("E_out", "d_family", "e_hat", *res_laws)
    arr = {src: {k_: np.array([np.mean([per[c][src][k_] for c in cs]) for cs in gs_rows]) for k_ in keys
                 if k_ in per[gs_rows[0][0]][src]} for src in ("forecast", "floor", "null", "shuffle")}

    def scores(ix):
        m = {src: {k_: v[ix].mean(-1) for k_, v in d.items()} for src, d in arr.items()}
        out = {"outcome": _S(m["forecast"]["E_out"], m["floor"]["E_out"], 1.0),
               "family": _S(m["forecast"]["d_family"], m["floor"]["d_family"], 1.0)}
        for ln in [l for l in res_laws if l != "energy" or energy_ok]:
            out[ln] = _S(m["forecast"][ln], m["floor"][ln], m["shuffle"][ln])
        return out
    n = len(gs_rows)
    S = {k_: float(v[0]) for k_, v in scores(np.arange(n)[None]).items()}
    B = scores(rng.integers(0, n, (n_boot, n)))
    ci = {k_: [float(np.nanquantile(b, 0.025)), float(np.nanquantile(b, 0.975))] if np.isfinite(b).any() else [np.nan, np.nan]
          for k_, b in B.items()}
    mean = {src: {k_: float(v.mean()) for k_, v in d.items()} for src, d in arr.items()}
    med = {src: {k_: float(np.median(v)) for k_, v in d.items()} for src, d in arr.items()}
    return {"n": n, "S": S, "S_ci95": ci, "mean": mean, "median": med}


def run(cache_dir: Path, data_dir: Path, cfg, model_cfg, log=print) -> dict:
    probe = cfg["probe"]
    alphas = tuple(float(a) for a in probe.get("alphas", DEFAULT_ALPHAS))
    k = int(probe["k_folds"])
    n_seeds = int(cfg.get("fold_seeds_l1", 4))
    n_boot = int(cfg["bootstrap"]["n"])
    n_splits = int(cfg.get("selection", {}).get("n_splits", 50))
    res_laws = tuple(cfg.get("laws", RES_LAWS))
    recs = complete_groups(load_index(cache_dir))
    clips = [r["clip"] for r in recs]
    row = {c: i for i, c in enumerate(clips)}
    labels = {c: json.loads((data_dir / c / "labels.json").read_text()) for c in clips}
    states = {c: laws.from_labels(labels[c]) for c in clips}
    groups = np.array([r["group"] for r in recs])
    ugroups = sorted(set(groups))
    twins = sorted({c.split("/")[1] for c in clips}, key=lambda t: (t not in LEGAL, t))
    mf = np.load(cache_dir / "_maskfeat_v2.npz")
    if list(mf["clips"]) != clips:
        raise RuntimeError("_maskfeat_v2.npz is for another clip list: run L0 first (it builds the weights)")
    spaces = _load_spaces(cache_dir, recs, clips, model_cfg, mf)
    Y = {o: np.array([states[c][1]["v"][oi] for c in clips]) for oi, o in enumerate("AB")}
    Ypre = {o: np.array([states[c][0]["v"][oi] for c in clips]) for oi, o in enumerate("AB")}
    true_speed = np.linalg.norm(np.stack([Y["A"], Y["B"]], 1), axis=-1)
    options = ["P-real-mm"] + (["P-pred-real"] if any("pr" in s for s in spaces.values()) else [])
    out = {"n_groups": len(ugroups), "n_clips": len(clips), "options": options,
           "protocol": {"k_folds": k, "fold_seeds": n_seeds, "test_groups_per_fold": len(ugroups) / k,
                        "cross_fitting_note": "5-fold x 4 seeds cross-fitting (spec §5: allowed below 100 groups)",
                        "n_spaces": len(spaces), "selection_splits": n_splits, "bootstrap": n_boot},
           "variants": {}, "ceiling": {}, "selected": {}}
    rng = np.random.default_rng(0)
    legal_rows = [[f"{g}/{t}" for t in LEGAL] for g in ugroups]
    for space, S_ in spaces.items():
        for pname, pmap in POOLS.items():
            # P-forecast ceiling (information in the forecast; not a law score)
            rows1 = np.array([row[f"{g}/{LEGAL[0]}"] for g in ugroups])
            ceil = {f"post_{o}": _ceiling(S_["fc"][pmap[o]][rows1], Y[o][rows1], groups[rows1], k, alphas) for o in "AB"}
            ceil.update({f"pre_{o}": _ceiling(S_["fc"][pmap[o]][rows1], Ypre[o][rows1], groups[rows1], k, alphas) for o in "AB"})
            if pname == "roi":   # how much a forecast-trained ROI probe gets from the ROI's placement alone
                ceil.update({f"mask_floor_post_{o}": _ceiling(mf[f"w.{pmap[o]}"][rows1].astype(np.float64), Y[o][rows1],
                                                              groups[rows1], k, alphas) for o in "AB"})
            out["ceiling"][f"{space}|{pname}"] = ceil
            for opt in options:
                src = "real" if opt == "P-real-mm" else "pr"
                if src not in S_:
                    continue
                v_fc, v_fl, r2 = np.zeros((len(clips), 2, 2)), np.zeros((len(clips), 2, 2)), []
                for oi, o in enumerate("AB"):
                    oof, app, r = _crossfit(S_[src][pmap[o]], Y[o], groups, S_["fc"][pmap[o]], k, n_seeds, alphas,
                                            mm_src=S_["fc"][pmap[o]] if opt == "P-real-mm" else None)
                    v_fl[:, oi], v_fc[:, oi] = oof, app
                    r2.append(r)
                per = _per_clip(v_fc, v_fl, states, clips, row, ugroups, res_laws)
                sp_fc = np.linalg.norm(v_fc, axis=-1)
                sp_fl = np.linalg.norm(v_fl, axis=-1)
                ratio = float(sp_fc.std() / sp_fl.std())
                energy_ok = SPREAD_OK[0] <= ratio <= SPREAD_OK[1]
                res = {"probe_r2_floor": float(np.mean(r2)), "spread_ratio_forecast_over_floor": ratio,
                       "energy_scorable": energy_ok, "speed_spread_true": float(true_speed.std()),
                       "corr_forecast_vs_post": float(np.corrcoef(v_fc.ravel(), np.stack([Y["A"], Y["B"]], 1).ravel())[0, 1]),
                       "twins": {}}
                res["twins"]["legal"] = _twin_scores(per, legal_rows, n_boot, rng, energy_ok, res_laws)
                for t in twins:
                    res["twins"][t] = _twin_scores(per, [[f"{g}/{t}"] for g in ugroups], n_boot, rng, energy_ok, res_laws)
                res["_per"] = per
                out["variants"][f"{space}|{opt}|{pname}"] = res
                log(f"[L1] {space}|{opt}|{pname}: floor R² {res['probe_r2_floor']:.3f}, S_outcome(legal) "
                    f"{res['twins']['legal']['S']['outcome']:+.2f}, S_family(legal) {res['twins']['legal']['S']['family']:+.2f}")

    # selection rule over spaces, per (option, pool), on the legal rows
    for opt in options:
        for pname in POOLS:
            vs = [v for v in out["variants"] if v.endswith(f"|{opt}|{pname}")]
            if not vs:
                continue
            sel = {}
            for score, key in (("outcome", "E_out"), ("family", "d_family")):
                def f(v, gids, key=key):
                    per = out["variants"][v]["_per"]
                    cs = [f"{g}/{t}" for g in gids for t in LEGAL]
                    fc = np.mean([per[c]["forecast"][key] for c in cs])
                    fl = np.mean([per[c]["floor"][key] for c in cs])
                    return float(_S(fc, fl, 1.0))
                sel[score] = heldout_select(f, vs, ugroups, n_splits=n_splits)
            out["selected"][f"{opt}|{pname}"] = sel
    for v in out["variants"].values():
        v.pop("_per")
    return out


def table(res: dict) -> str:
    pr = res["protocol"]
    rows = [f"n_groups = {res['n_groups']} ({pr['cross_fitting_note']}; ~{pr['test_groups_per_fold']:.0f} test groups "
            f"per fold); spaces searched = {pr['n_spaces']}; bootstrap {pr['bootstrap']} over groups. "
            "Primary: S_outcome and S_family on the frame pool (leak-free). The ROI pool sits at the TRUE post-event "
            "location and leaks it (secondary). S_momentum / CoM / angular momentum: a forecast that ignores the "
            "collision conserves them, so they are reported, not primary (reference: shuffled-group forecast).", ""]
    rows += ["## Selected space (held-out rule)", "",
             "| probe option | pool | score | selected space (picks) | held-out S [2.5%, 97.5% of splits] | variants searched | groups select / eval |",
             "|---|---|---|---|---|---|---|"]
    for key, sel in res["selected"].items():
        opt, pname = key.split("|")
        for score, s in sel.items():
            rows.append(f"| {opt} | {pname}{' (leaks location)' if pname == 'roi' else ''} | S_{score} | "
                        f"{s['selected'].split('|')[0]} ({s['selected_counts'][s['selected']]}/{2 * s['n_splits']}) | "
                        f"{s['score']:+.2f} [{s['score_q025']:+.2f}, {s['score_q975']:+.2f}] | {s['n_variants']} | "
                        f"{s['n_select_groups']} / {s['n_eval_groups']} |")
    for vname, v in res["variants"].items():
        space, opt, pname = vname.split("|")
        sel = res["selected"].get(f"{opt}|{pname}", {}).get("family", {}).get("selected")
        if pr["n_spaces"] > 1 and vname != sel:
            continue          # DiTs: only the selected space is tabled (all are in l1.json)
        f = lambda d, kk: (f"{d['S'][kk]:+.2f} [{d['S_ci95'][kk][0]:+.2f}, {d['S_ci95'][kk][1]:+.2f}]"
                           if kk in d["S"] and np.isfinite(d["S"][kk]) else "n/a")
        en = "" if v["energy_scorable"] else f" Energy NOT SCORABLE (decoded forecast speed spread {v['spread_ratio_forecast_over_floor']:.2f} of the real-frame decode, outside 0.8-1.2)."
        rows += ["", f"## {opt}, pool {pname}{' (secondary: leaks the post-event location)' if pname == 'roi' else ' (primary)'}"
                 + (f", space {space}" if pr["n_spaces"] > 1 else ""), "",
                 f"Probe R² on real post-event frames (floor decode) {v['probe_r2_floor']:.3f}; corr(decoded forecast, true post v) "
                 f"{v['corr_forecast_vs_post']:.2f}.{en}", "",
                 "| twin | n | S_outcome [95% CI] | S_family [95% CI] | d_family forecast / floor | ê forecast / floor (median) | "
                 "S_momentum (demoted) | S_com | S_angmom | S_energy |",
                 "|---|---|---|---|---|---|---|---|---|---|"]
        for t, d in v["twins"].items():
            m, md = d["mean"], d["median"]
            rows.append(f"| {t} | {d['n']} | {f(d, 'outcome')} | {f(d, 'family')} | {m['forecast']['d_family']:.3f} / "
                        f"{m['floor']['d_family']:.3f} | {md['forecast']['e_hat']:.2f} / {md['floor']['e_hat']:.2f} | "
                        f"{f(d, 'momentum')} | {f(d, 'com')} | {f(d, 'angular_momentum')} | "
                        f"{f(d, 'energy') if v['energy_scorable'] else 'not scorable'} |")
    rows += ["", "## P-forecast (information ceiling: probes trained ON forecasts; never a law score)", "",
             "| space / pool | post v R² (A, B) | pre v R² (A, B) | ROI mask floor, post v R² (A, B) |", "|---|---|---|---|"]
    for key, c in res["ceiling"].items():
        mfl = f"{c['mask_floor_post_A']:.2f}, {c['mask_floor_post_B']:.2f}" if "mask_floor_post_A" in c else ""
        rows.append(f"| {key} | {c['post_A']:.2f}, {c['post_B']:.2f} | {c['pre_A']:.2f}, {c['pre_B']:.2f} | {mfl} |")
    return "\n".join(rows)
