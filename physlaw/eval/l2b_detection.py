"""L2b trained violation detection (spec §5.4): is the violation DECODABLE from post-event activations,
whatever the model's own surprise says?

Per L2 contrast (configs/eval/l2_surprise.yaml ``contrasts.<family>``, plus the legal-vs-legal pair as a
reference: two legal outcomes that differ in e are of course distinguishable), rows = the violating and the
legal twin of every group, y = 1 for the violating one. Features: post-event pooled activations, ROI pool
(``roi_pair.post``) and frame pool (``frame.post``), every cached layer (DiTs: every SNR x layer).

Probe: L2-penalised logistic regression, solved by Newton in the SVD basis of the standardised training rows
(exactly the primal L2 problem; the penalty is rotation invariant). Penalty by inner grouped CV (AUC) on the
training groups. Grouped K-fold over twin groups x ``fold_seeds_l2b`` seeds (both twins of a group are always
in the same fold: they share their pre-event frames). Score: out-of-fold AUC.
Permutation null: within-group label swaps (each group's two twins exchange labels with p = 1/2), 500 draws,
max over the searched grid (layers x pools) -> corrected p; Holm across contrasts.
Mask floor (ROI pool): the same probe on the pooling weights alone (where the balls go differs between twins).
Selection rule: the reported layer (per pool, and overall) is selected on half the groups by held-out AUC and
scored on the other half (probes.select.heldout_select).
Encoded-but-unused index = (L2b AUC − 0.5) − (L2 adjusted win rate − 0.5), both held-out (L2 from l2.json's
motion-adjusted summary). Large and positive: decodable, but the model's own prediction ignores it. A heuristic:
it subtracts a PAIRED win rate (per group) from an UNPAIRED AUC (rows pooled over groups); the two are not on
the same scale, so read its sign and size loosely.

Kinematic baseline (``kinematic``): the same logistic protocol (grouped CV, same seeds and penalty grid) on the
TRUE post-event kinematics from labels, KINEMATIC_FEATURES = |v_A'|, |v_B'|, KE'/KE, |Δv_A|, |Δv_B|, |v_cm'|.
The F1 momentum violators are mostly slower than their legal twins (72-81% of groups), so these features alone
separate them; ``delta_auc_vs_kinematic`` = model held-out AUC − kinematic AUC (and ``delta_auc_vs_mask_floor``).
A model only shows violation detection beyond speed if ΔAUC > 0.

Norm routes (``norm_routes``): the "momentum conserved" label is a threshold on a NORM (|Δp| = 0 vs > 0), which a
linear logistic probe on the activations cannot express (predecessor §34a / kinetic_energy §46d: a linear
classifier on the exact Δp vector scored AUC 0.56, on |Δp| 1.00). So two more routes per cell, same grouped
folds / seeds, ridge penalty by inner grouped CV (physlaw.probes.ridge.fit_probe):
  * ``fitted_norm``    ridge activations -> true Δp = m_A dv_A + m_B dv_B (2-D, labels); score = |Δp̂|
  * ``composed_norm``  ridge activations -> true (dv_A, dv_B) (4-D); Δp̂ = m_A dv̂_A + m_B dv̂_B with the TRUE
                       masses; score = |Δp̂|
AUC of the score (violating = 1), cell selected on held-out groups like the logistic route; also in the
transfer test.

Null sanity: the within-group label swap IS the paired sign-flip null (swapping a group's two labels flips the
sign of its pair). ``null_diag`` reports the share of draws that differ from the identity labelling (and from
its global flip, which gives the same AUC), and the null's mean / sd for the best cell, so a degenerate
(non-moving) null is visible.

Transfer test (``transfer``, config ``l2b.transfer.<family>``): train the probe on one contrast (e.g.
mom_violate_e1 vs conserve_e1) and score it on another (mom_violate_e015 vs conserve_e015), grouped: the test
rows of a fold come from groups whose rows were NOT in training (twins share pre-event frames). The model cell
is the one selected on the training contrast (``overall.selected``); the kinematic baseline is scored the same
way. Read the model's transfer AUC against the kinematic baseline's: a pure restitution/speed cue would not
transfer, but on F1 the violators are slower in BOTH contrasts (72% / 81% of groups), so a speed shortcut CAN
transfer (kinematic transfer AUC ~0.6-0.67 on F1); only model transfer above the kinematic one is evidence.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from physlaw.eval.l0_states import _lname, complete_groups, load_index, load_pooled
from physlaw.eval.l2_surprise import family_contrasts
from physlaw.probes.ridge import _inner_alpha, _SVDRidge, auc_score, fit_probe, grouped_kfold
from physlaw.probes.select import heldout_select, holm

POOLS = {"roi": "roi_pair.post", "frame": "frame.post"}
_G: dict = {}


# --------------------------------------------------------------------------- logistic probe

def _project(X, tr):
    """Standardise on ``tr``, SVD; returns (Z_tr, map(X_rows) -> Z)."""
    mx, sx = X[tr].mean(0), X[tr].std(0)
    sx[sx == 0] = 1.0
    U, s, Vt = np.linalg.svd((X[tr] - mx) / sx, full_matrices=False)
    keep = s > 1e-8 * s[0]
    V = Vt[keep].T
    return U[:, keep] * s[keep], lambda rows: ((X[rows] - mx) / sx) @ V


def logistic_fit(Z: np.ndarray, y: np.ndarray, lam: float, iters: int = 50) -> np.ndarray:
    """min sum log(1 + exp(-y' f)) + lam/2 |w|^2, f = Z w + b (b unpenalised); Newton. Returns [w; b]."""
    A = np.hstack([Z, np.ones((len(Z), 1))])
    P = np.full(A.shape[1], float(lam)); P[-1] = 1e-8
    beta = np.zeros(A.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(A @ beta, -30, 30)))
        g = A.T @ (p - y) + P * beta
        H = (A.T * (p * (1 - p))) @ A + np.diag(P)
        step = np.linalg.solve(H, g)
        beta -= step
        if np.abs(step).max() < 1e-7:
            break
    return beta


def _decision(beta, Z):
    return Z @ beta[:-1] + beta[-1]


def _inner_lam(Xtr, ytr, gtr, lams, k, seed):
    """Penalty by inner grouped CV (AUC) on the training rows."""
    inner = np.zeros((len(lams), len(ytr)))
    for ite in grouped_kfold(gtr, k, seed):
        itr = np.setdiff1d(np.arange(len(ytr)), ite)
        Zi, pj = _project(Xtr, itr)
        Zo = pj(ite)
        for li, lam in enumerate(lams):
            inner[li, ite] = _decision(logistic_fit(Zi, ytr[itr], lam), Zo)
    sc = [auc_score(ytr, inner[li]) for li in range(len(lams))]
    return lams[int(np.nanargmax(sc))]


def cv_auc(X, y, groups, lams, k, fold_seeds, seed0=0, fixed_lam=None, ys=None):
    """Grouped CV logistic probe. Returns (per-seed AUCs, oof decision values [fold_seeds, n], chosen lams).
    ``ys`` [P, n]: extra label vectors scored with the SAME folds and ``fixed_lam`` (permutation null)."""
    n = len(y)
    oofs, aucs, chosen = [], [], []
    null_oof = None if ys is None else np.zeros((len(ys), n))
    for fs in range(fold_seeds):
        oof = np.zeros(n)
        for te in grouped_kfold(groups, k, seed0 + fs):
            tr = np.setdiff1d(np.arange(n), te)
            Ztr, proj = _project(X, tr)
            Zte = proj(te)
            if fixed_lam is None:            # inner grouped CV on the training groups, by AUC
                lam = _inner_lam(X[tr], y[tr], groups[tr], lams, k, seed0 + fs + 10_000)
            else:
                lam = fixed_lam
            chosen.append(lam)
            oof[te] = _decision(logistic_fit(Ztr, y[tr], lam), Zte)
            if ys is not None:
                for pi, yp in enumerate(ys):
                    null_oof[pi, te] = _decision(logistic_fit(Ztr, yp[tr], lam), Zte)
        aucs.append(auc_score(y, oof))
        oofs.append(oof)
    nulls = None if ys is None else np.array([auc_score(yp, o) for yp, o in zip(ys, null_oof)])
    return aucs, np.stack(oofs), chosen, nulls


def transfer_auc(X_tr, y_tr, g_tr, X_te, y_te, g_te, lams, k, fold_seeds, seed0=0) -> list[float]:
    """Train on (X_tr, y_tr), score on (X_te, y_te), grouped: folds over the union of groups; per fold the probe
    is fitted on the training-set rows of the OTHER folds' groups (penalty by inner grouped CV on them) and
    scores the test-set rows of this fold's groups. Returns the per-seed AUCs on the test set."""
    g_tr, g_te = np.asarray(g_tr), np.asarray(g_te)
    allg = np.concatenate([g_tr, g_te])
    aucs = []
    for fs in range(fold_seeds):
        dec = np.full(len(y_te), np.nan)
        for f in grouped_kfold(allg, k, seed0 + fs):
            held = np.unique(allg[f])
            tr = np.where(~np.isin(g_tr, held))[0]
            te = np.where(np.isin(g_te, held))[0]
            if not len(te) or len(np.unique(y_tr[tr])) < 2:
                continue
            lam = _inner_lam(X_tr[tr], y_tr[tr], g_tr[tr], lams, k, seed0 + fs + 10_000)
            X = np.vstack([X_tr[tr], X_te[te]])
            Ztr, proj = _project(X, np.arange(len(tr)))
            dec[te] = _decision(logistic_fit(Ztr, y_tr[tr], lam), proj(np.arange(len(tr), len(X))))
        ok = np.isfinite(dec)
        aucs.append(auc_score(y_te[ok], dec[ok]))
    return aucs


def transfer_norm_auc(X_tr, T_tr, g_tr, X_te, y_te, g_te, comb_te, alphas, k, fold_seeds, seed0=0) -> list[float]:
    """Norm-route transfer: ridge X -> T fitted on the training set's rows (other folds' groups, penalty by inner
    grouped CV), predicted on the test set's rows of this fold's groups; score = comb_te(prediction) (the norm).
    Returns per-seed AUCs on the test set."""
    g_tr, g_te = np.asarray(g_tr), np.asarray(g_te)
    allg = np.concatenate([g_tr, g_te])
    aucs = []
    for fs in range(fold_seeds):
        pred = np.full((len(y_te), T_tr.shape[1]), np.nan)
        for f in grouped_kfold(allg, k, seed0 + fs):
            held = np.unique(allg[f])
            tr = np.where(~np.isin(g_tr, held))[0]
            te = np.where(np.isin(g_te, held))[0]
            if not len(te):
                continue
            a = _inner_alpha(X_tr[tr], T_tr[tr], g_tr[tr], alphas, k, seed0 + fs, False, True)
            pred[te] = _SVDRidge(X_tr[tr], T_tr[tr]).predict(X_te[te], a)
        ok = np.isfinite(pred).all(1)
        aucs.append(auc_score(y_te[ok], comb_te(pred)[ok]))
    return aucs


def momentum_targets(labels: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(Δp [2] = m_A dv_A + m_B dv_B, (dv_A, dv_B) [4], masses [2]) of a collision clip from its labels."""
    from physlaw.eval import laws
    pre, post, ctx = laws.from_labels(labels)
    m = np.asarray(ctx["m"], float)
    dv = np.asarray(post["v"], float)[:, :2] - np.asarray(pre["v"], float)[:, :2]
    return m @ dv, dv.reshape(4), m


def null_diagnostics(perms: np.ndarray, y: np.ndarray, null: np.ndarray | None) -> dict:
    """Is the permutation null moving? Share of draws != identity (and != its global flip), and the null's spread."""
    diff_id = np.any(perms != y, 1)
    diff_flip = np.any(perms != 1 - y, 1)
    out = {"share_differ_from_identity": float(diff_id.mean()),
           "share_differ_from_identity_and_flip": float((diff_id & diff_flip).mean()),
           "n_distinct": int(len({p.tobytes() for p in perms}))}
    if null is not None:
        out.update(null_mean=float(np.mean(null)), null_sd=float(np.std(null)),
                   null_q05_q95=[float(np.quantile(null, 0.05)), float(np.quantile(null, 0.95))])
    return out


# --------------------------------------------------------------------------- kinematic baseline

KINEMATIC_FEATURES = ("speed_A_post", "speed_B_post", "ke_ratio", "dv_A", "dv_B", "v_cm_post")


def kinematic_features(labels: dict) -> np.ndarray:
    """True post-event kinematics of a collision clip: |v_A'|, |v_B'|, KE'/KE, |Δv_A|, |Δv_B|, |v_cm'|."""
    from physlaw.eval import laws
    pre, post, ctx = laws.from_labels(labels)
    m, v0, v1 = np.asarray(ctx["m"], float), np.asarray(pre["v"], float), np.asarray(post["v"], float)
    ke = lambda v: float((m * (v ** 2).sum(-1)).sum())
    sp, dv = np.linalg.norm(v1, axis=-1), np.linalg.norm(v1 - v0, axis=-1)
    return np.array([sp[0], sp[1], ke(v1) / ke(v0), dv[0], dv[1], np.linalg.norm(m @ v1) / m.sum()])


def swap_perms(groups: np.ndarray, y: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """[n, len(y)] label vectors: within each group the two twins swap labels with probability 1/2."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    out = np.empty((n, len(y)))
    for i in range(n):
        flip = dict(zip(ug, rng.random(len(ug)) < 0.5))
        out[i] = [1 - v if flip[g] else v for g, v in zip(groups, y)]
    return out


def _run_cell(job):
    key, X = job, _G["feat"](job)
    lams, k, fs = _G["lams"], _G["k"], _G["fold_seeds"]
    aucs, oof, chosen, _ = cv_auc(X, _G["y"], _G["groups"], lams, k, fs)
    out = {"auc_mean": float(np.mean(aucs)), "auc_sd": float(np.std(aucs)), "oof": oof}
    if _G.get("targets") is not None and key[0] == "act":      # norm routes (ridge -> |Δp̂|)
        for route, (T, comb) in _G["targets"].items():
            pr = fit_probe(X, T, _G["groups"], alphas=lams, k=k, fold_seeds=fs)
            sc = np.stack([comb(o) for o in pr.oof])             # [fold_seeds, n]
            out[route] = {"oof": sc, "auc_mean": float(np.mean([auc_score(_G["y"], o) for o in sc])),
                          "r2_mean": pr.score_mean}
    if _G["perms"] is not None and key[0] == "act":      # reduced protocol for the null: one fold seed, fixed lam
        lam = float(np.median(chosen))
        q, _, _, nulls = cv_auc(X, _G["y"], _G["groups"], lams, k, 1, fixed_lam=lam, ys=_G["perms"])
        out.update(quick=float(q[0]), null=nulls)
    return key, out


# --------------------------------------------------------------------------- run

def run(cache_dir: Path, data_dir: Path, cfg, l2_summary: dict | None = None, workers: int = 8, log=print) -> dict:
    probe = cfg["probe"]
    lams = tuple(float(a) for a in cfg.get("l2b", {}).get("lambdas", probe["alphas"]))
    k = int(probe["k_folds"])
    fold_seeds = int(cfg.get("l2b", {}).get("fold_seeds", 4))
    n_perm = int(cfg.get("l2b", {}).get("n_permutations", probe["n_permutations"]))
    n_splits = int(cfg.get("selection", {}).get("n_splits", 50))
    recs = complete_groups(load_index(cache_dir))
    clips = [r["clip"] for r in recs]
    acts, pools, layers, _ = load_pooled(cache_dir, recs)
    mf = np.load(cache_dir / "_maskfeat_v2.npz")
    if list(mf["clips"]) != clips:
        raise RuntimeError("_maskfeat_v2.npz is for another clip list: run L0 first (it builds the weights)")
    cal, cons = family_contrasts(cfg, data_dir.name)
    cons = cons + [{"viol": cal[0], "legal": cal[1], "control": True}]
    row = {c: i for i, c in enumerate(clips)}
    try:
        kin = np.stack([kinematic_features(json.loads((data_dir / c / "labels.json").read_text())) for c in clips])
    except (KeyError, NotImplementedError, FileNotFoundError) as e:       # non-collision family / no labels
        log(f"[L2b] no kinematic baseline ({e!r})")
        kin = None
    try:
        mt = [momentum_targets(json.loads((data_dir / c / "labels.json").read_text())) for c in clips]
        DP, DV, MS = (np.stack([t[i] for t in mt]) for i in range(3))
    except (KeyError, NotImplementedError, FileNotFoundError) as e:
        log(f"[L2b] no norm routes ({e!r})")
        DP = None
    routes = ("fitted_norm", "composed_norm") if DP is not None and cfg.get("l2b", {}).get("norm_routes", True) else ()
    keep: dict = {}          # contrast key -> (rows, y, groups, selected cell's job), for the transfer test
    ugroups = sorted({r["group"] for r in recs})
    out = {"n_groups": len(ugroups), "protocol": {
        "k_folds": k, "fold_seeds": fold_seeds, "test_groups_per_fold": len(ugroups) / k, "n_permutations": n_perm,
        "lambdas": list(lams), "selection_splits": n_splits, "n_cells": len(layers) * len(POOLS),
        "note": f"grouped {k}-fold x {fold_seeds} seeds cross-fitting (spec §5: allowed below 100 groups)"},
        "kinematic_features": list(KINEMATIC_FEATURES), "contrasts": {}}
    for con in cons:
        key = f"{con['viol']} vs {con['legal']}"
        gs = [g for g in ugroups if f"{g}/{con['viol']}" in row and f"{g}/{con['legal']}" in row]
        rows = np.array([row[f"{g}/{t}"] for g in gs for t in (con["viol"], con["legal"])])
        y = np.tile([1.0, 0.0], len(gs))
        groups = np.repeat(gs, 2)
        perms = swap_perms(groups, y, n_perm, seed=0) if n_perm else None
        feat = lambda j, rows=rows: (acts[rows, j[1], j[2]].astype(np.float64) if j[0] == "act"
                                     else mf[f"w.{j[1]}"][rows].astype(np.float64))
        tg = None
        if routes:
            tg = {"fitted_norm": (DP[rows], lambda P: np.linalg.norm(P, axis=-1)),
                  "composed_norm": (DV[rows], lambda P, ms=MS[rows]: np.linalg.norm(ms[:, :1] * P[:, :2] + ms[:, 1:] * P[:, 2:], axis=-1))}
        _G.update(feat=feat, y=y, groups=groups, lams=lams, k=k, fold_seeds=fold_seeds, perms=perms, targets=tg)
        jobs = [("act", pools.index(p), li) for p in POOLS.values() for li in range(len(layers))]
        jobs.append(("mask", POOLS["roi"]))
        res = {}
        with ProcessPoolExecutor(workers) as ex:       # fork: workers inherit _G
            for j, r in ex.map(_run_cell, jobs, chunksize=1):
                res[j] = r
        cells = {f"{_lname(layers[j[2]])}/{pools[j[1]]}": r for j, r in res.items() if j[0] == "act"}
        cell_job = {f"{_lname(layers[j[2]])}/{pools[j[1]]}": j for j in res if j[0] == "act"}

        def score(c, gids, cells=cells, groups=groups):
            m = np.isin(groups, gids)
            return float(np.mean([auc_score(y[m], o[m]) for o in cells[c]["oof"]]))
        rec = {"n_groups": len(gs), "n_rows": len(rows), "control": bool(con.get("control")),
               "auc_by_cell": {c: v["auc_mean"] for c, v in cells.items()},
               "overall": heldout_select(score, list(cells), groups, n_splits=n_splits), "pools": {}}
        for pname, pool in POOLS.items():
            pc = [c for c in cells if c.endswith("/" + pool)]
            rec["pools"][pname] = heldout_select(score, pc, groups, n_splits=n_splits)
        rec["pools"]["roi"]["mask_floor_auc"] = res[("mask", POOLS["roi"])]["auc_mean"]
        rec["norm_routes"] = {}
        for route in routes:
            def rscore(c, gids, route=route):
                m = np.isin(groups, gids)
                return float(np.mean([auc_score(y[m], o[m]) for o in cells[c][route]["oof"]]))
            sel = heldout_select(rscore, list(cells), groups, n_splits=n_splits)
            sel["full_sample_auc"] = cells[sel["selected"]][route]["auc_mean"]
            sel["r2_dp"] = cells[sel["selected"]][route]["r2_mean"]
            rec["norm_routes"][route] = sel
        rec["delta_auc_vs_mask_floor"] = rec["overall"]["score"] - rec["pools"]["roi"]["mask_floor_auc"]
        if kin is not None:
            ka, _, _, _ = cv_auc(kin[rows], y, groups, lams, k, fold_seeds)
            rec["kinematic"] = {"auc_mean": float(np.mean(ka)), "auc_sd": float(np.std(ka))}
            rec["delta_auc_vs_kinematic"] = rec["overall"]["score"] - rec["kinematic"]["auc_mean"]
        keep[key] = (rows, y, groups, cell_job[rec["overall"]["selected"]],
                     {r: cell_job[v["selected"]] for r, v in rec["norm_routes"].items()})
        if perms is not None:
            best = max(cells, key=lambda c: cells[c]["quick"])
            null_max = np.max(np.stack([v["null"] for v in cells.values()]), 0)
            rec.update(best_cell_quick=best, null_max_q95=float(np.quantile(null_max, 0.95)),
                       p_corrected=float((1 + np.sum(null_max >= cells[best]["quick"])) / (len(null_max) + 1)),
                       null_diag={**null_diagnostics(perms, y, cells[best]["null"]),
                                  "null_max_sd": float(np.std(null_max))})
        l2 = (l2_summary or {}).get(key)
        if l2 is not None and "score" in l2:
            rec["l2_adj_win_heldout"] = l2["score"]
            rec["encoded_but_unused"] = (rec["overall"]["score"] - 0.5) - (l2["score"] - 0.5)
        out["contrasts"][key] = rec
        log(f"[L2b] {key}: held-out AUC {rec['overall']['score']:.3f} ({rec['overall']['selected']}), "
            f"p_corr {rec.get('p_corrected', float('nan')):.3g}, ROI mask floor {rec['pools']['roi']['mask_floor_auc']:.3f}, "
            f"kinematic AUC {rec.get('kinematic', {}).get('auc_mean', float('nan')):.3f}"
            + "".join(f", {r} {v['score']:.3f}" for r, v in rec["norm_routes"].items())
            + (f", null sd {rec['null_diag']['null_sd']:.3f}" if "null_diag" in rec else ""))
    ph = holm({c: v.get("p_corrected", np.nan) for c, v in out["contrasts"].items() if not v["control"]})
    for c, p in ph.items():
        out["contrasts"][c]["p_holm"] = p
    out["transfer"] = {}
    tcfg = (cfg.get("l2b", {}).get("transfer") or {})
    for t in (tcfg.get(data_dir.name) or []):
        ktr, kte = f"{t['train'][0]} vs {t['train'][1]}", f"{t['test'][0]} vs {t['test'][1]}"
        if ktr not in keep or kte not in keep:
            continue
        (rtr, ytr, gtr, job, rjobs), (rte, yte, gte, _, _) = keep[ktr], keep[kte]
        feat = lambda r, j=job: acts[r, j[1], j[2]].astype(np.float64)
        ma = transfer_auc(feat(rtr), ytr, gtr, feat(rte), yte, gte, lams, k, fold_seeds)
        tr = {"train": ktr, "test": kte, "cell": out["contrasts"][ktr]["overall"]["selected"],
              "model_auc": float(np.mean(ma)), "model_auc_sd": float(np.std(ma)),
              "test_within_auc": out["contrasts"][kte]["overall"]["score"], "norm_routes": {}}
        for route, rj in rjobs.items():
            T = DP if route == "fitted_norm" else DV
            comb = ((lambda P: np.linalg.norm(P, axis=-1)) if route == "fitted_norm" else
                    (lambda P, ms=MS[rte]: np.linalg.norm(ms[:, :1] * P[:, :2] + ms[:, 1:] * P[:, 2:], axis=-1)))
            na = transfer_norm_auc(feat(rtr, rj), T[rtr], gtr, feat(rte, rj), yte, gte, comb, lams, k, fold_seeds)
            tr["norm_routes"][route] = {"cell": out["contrasts"][ktr]["norm_routes"][route]["selected"],
                                        "auc": float(np.mean(na)), "auc_sd": float(np.std(na)),
                                        "test_within_auc": out["contrasts"][kte]["norm_routes"][route]["score"]}
        if kin is not None:
            ka = transfer_auc(kin[rtr], ytr, gtr, kin[rte], yte, gte, lams, k, fold_seeds)
            tr.update(kinematic_auc=float(np.mean(ka)), kinematic_auc_sd=float(np.std(ka)))
        out["transfer"][f"{ktr} -> {kte}"] = tr
        log(f"[L2b] transfer {ktr} -> {kte}: model AUC {tr['model_auc']:.3f} ({tr['cell']}), "
            f"kinematic AUC {tr.get('kinematic_auc', float('nan')):.3f}, "
            + ", ".join(f"{r} {v['auc']:.3f}" for r, v in tr["norm_routes"].items()))
    return out


def table(res: dict) -> str:
    pr = res["protocol"]
    rows = [f"n_groups = {res['n_groups']} ({pr['note']}; ~{pr['test_groups_per_fold']:.0f} test groups per fold); "
            f"variants searched = {pr['n_cells']} (layers x 2 pools); {pr['n_permutations']} within-group label-swap "
            f"permutations, max over the grid; layer selected on half the groups, AUC on the other half "
            f"({2 * pr['selection_splits']} evaluations). L2b is NOT motion-adjusted: a probe can separate twins by how much "
            "the balls move, so read each AUC against the legal-vs-legal reference row (two legal outcomes, different "
            "motion) and the ROI mask floor; the encoded-but-unused index inherits that confound.", "",
            "Kinematic baseline: the same probe protocol on the true post-event kinematics ("
            + ", ".join(res.get("kinematic_features", KINEMATIC_FEATURES)) + "); the model detects the violation beyond "
            "speed only if ΔAUC (model − kinematic) > 0. Norm routes: ridge to the true Δp (fitted_norm) or to (dv_A, dv_B) "
            "composed with true masses (composed_norm), score |Δp̂|; 'conserved' is a threshold on a norm, which a linear "
            "logistic probe cannot express. Encoded-but-unused is a heuristic: it subtracts a paired win rate "
            "from an unpaired AUC.", "",
            "| contrast | n groups | held-out AUC, overall (selected cell) | frame pool AUC (layer) | ROI pool AUC (layer) | "
            "ROI mask floor | kinematic AUC | ΔAUC vs kinematic | ΔAUC vs mask floor | fitted_norm |Δp̂| AUC (held-out) | "
            "composed_norm AUC (held-out) | p_corr (Holm) | null sd (best cell) | L2 adj. win (held-out) | encoded-but-unused |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, v in res["contrasts"].items():
        o, fr, ro = v["overall"], v["pools"]["frame"], v["pools"]["roi"]
        p = f"{v['p_corrected']:.3g} ({v.get('p_holm', float('nan')):.3g})" if "p_corrected" in v else ""
        l2 = f"{v['l2_adj_win_heldout']:.2f}" if "l2_adj_win_heldout" in v else ""
        ebu = f"{v['encoded_but_unused']:+.2f}" if "encoded_but_unused" in v else ""
        kin = f"{v['kinematic']['auc_mean']:.3f}" if "kinematic" in v else ""
        dk = f"{v['delta_auc_vs_kinematic']:+.3f}" if "delta_auc_vs_kinematic" in v else ""
        dm = f"{v['delta_auc_vs_mask_floor']:+.3f}" if "delta_auc_vs_mask_floor" in v else ""
        nr = [f"{v['norm_routes'][r]['score']:.3f} ({v['norm_routes'][r]['selected']})" if r in v.get("norm_routes", {}) else ""
              for r in ("fitted_norm", "composed_norm")]
        nsd = f"{v['null_diag']['null_sd']:.3f}" if "null_diag" in v else ""
        rows.append(f"| {c}{' (legal-vs-legal reference)' if v['control'] else ''} | {v['n_groups']} | "
                    f"{o['score']:.3f} ({o['selected']}) | {fr['score']:.3f} ({fr['selected'].rsplit('/', 1)[0]}) | "
                    f"{ro['score']:.3f} ({ro['selected'].rsplit('/', 1)[0]}) | {ro['mask_floor_auc']:.3f} | {kin} | {dk} | {dm} | "
                    f"{nr[0]} | {nr[1]} | {p} | {nsd} | {l2} | {ebu} |")
    if res.get("transfer"):
        rows += ["", "### Transfer across restitution (train on one contrast, score held-out groups of the other)", "",
                 "Compare with the kinematic transfer AUC: F1 violators are slower in both contrasts, so a speed "
                 "shortcut can transfer too; only model transfer above the kinematic baseline is evidence.", "",
                 "| train -> test | model cell (selected on train) | model transfer AUC | kinematic transfer AUC | "
                 "test within-contrast AUC | fitted_norm transfer AUC (within) | composed_norm transfer AUC (within) |",
                 "|---|---|---|---|---|---|---|"]
        for t in res["transfer"].values():
            ka = f"{t['kinematic_auc']:.3f}" if "kinematic_auc" in t else ""
            nr = [f"{t['norm_routes'][r]['auc']:.3f} ({t['norm_routes'][r]['test_within_auc']:.3f})"
                  if r in t.get("norm_routes", {}) else "" for r in ("fitted_norm", "composed_norm")]
            rows.append(f"| {t['train']} -> {t['test']} | {t['cell']} | {t['model_auc']:.3f} ± {t['model_auc_sd']:.3f} | "
                        f"{ka} | {t['test_within_auc']:.3f} | {nr[0]} | {nr[1]} |")
    return "\n".join(rows)
