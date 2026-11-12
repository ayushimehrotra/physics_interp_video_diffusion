"""L2b trained violation detection (spec §5.4): is the violation DECODABLE from post-event activations,
whatever the model's own surprise says?

Equality laws only (``L2B_LAWS``: momentum incl. the F6 glide, gravity, friction F6 slide / pair, oscillation F9 pair,
rolling F10, angular_momentum F12; spec §5.0.3 R5): a threshold on a monotone energy dose is decodable from any correlate of the
dose, so energy is scored by the representational kink (L2-rep, l2_surprise) instead.
Per L2 pairwise contrast (configs/eval/l2_surprise.yaml ``contrasts.<family>.list`` with its ``law``, plus the
legal-vs-legal pairs as references: two legal outcomes that differ in e are of course distinguishable), rows = the
violating twin(s) and the legal twin of every group, y = 1 for the violating ones (a listed ``viol``, e.g. the glide's
speed_x0.9 + speed_x1.1, puts both violators of a group in its rows). Features: post-event pooled activations, ROI pool
(``roi_pair.post``) and frame pool (``frame.post``), every cached layer (DiTs: every SNR x layer).

Probe: L2-penalised logistic regression, solved by Newton in the SVD basis of the standardised training rows
(exactly the primal L2 problem; the penalty is rotation invariant). Penalty by inner grouped CV (AUC) on the
training groups. Grouped K-fold over twin groups x ``fold_seeds_l2b`` seeds (both twins of a group are always
in the same fold: they share their pre-event frames). Score: out-of-fold AUC.
Uncertainty: the group bootstrap of the selected cell's out-of-fold AUC (``overall.auc_ci95``) and of the paired
differences below; no permutation null (held-out selection already removes the winner's curse).
Mask floor (ROI pool): the same probe on the pooling weights alone (where the balls go differs between twins); the
ROI pool's selected cell vs the mask floor gets a paired group-bootstrap CI (``pools.roi.vs_mask_floor``).
Verdict (``verdict``, :func:`detection_verdict`): detected only if the selected cell's AUC CI is above 0.5 and beats
the kinematic baseline (paired CI > 0) and, when the selected cell is an ROI cell, also the ROI mask floor (audit M0).
Selection rule: the reported layer (per pool, and overall) is selected on half the groups by held-out AUC and
scored on the other half (probes.select.heldout_select).
Kinematic baseline (``kinematic``): the same logistic protocol (grouped CV, same seeds and penalty grid) on the
TRUE post-event kinematics from labels: collisions KINEMATIC_FEATURES = |v_A'|, |v_B'|, KE'/KE, |Δv_A|, |Δv_B|,
|v_cm'| at the contact; other families WINDOW_FEATURES per (non-aux) object over the group's clean window after the
event (mean speed, speed at the window's end, net displacement, mean spin |ω|): the violations there (a changed glide
speed, a slide that does not stop, a changed period, a swapped rolling race or slipping spin, an extra bar spin) only show over
time.
The F1 momentum violators are mostly slower than their legal twins (72-81% of groups), so these features alone
separate them; ``delta_auc_vs_kinematic`` = model held-out AUC − kinematic AUC (and ``delta_auc_vs_mask_floor``),
and ``vs_kinematic`` = the paired group-bootstrap CI of (model AUC − kinematic AUC) on the full-sample selected cell
(both out-of-fold scores resampled with the same groups). A model only shows detection beyond speed if the CI > 0.

Speed-matched stratum (``speed_matched``, tolerance ``l2b.speed_matched_tol``): the groups whose violator's KE is
within the tolerance of its legal twin's, |log ke_ratio_to_legal| < tol (labels physlaw.ke_ratio_to_legal; old
labels: summary ke_post ratio), the out-of-fold scores restricted to them: model AUC (selected cell), kinematic AUC
and the paired CI of the difference. F2 lists both drift signs (mom_violate_e<e>_pos / _neg) as separate contrasts.

Norm routes (``norm_routes``): the "momentum conserved" label is a threshold on a NORM (|Δp| = 0 vs > 0), which a
linear logistic probe on the activations cannot express (predecessor §34a / kinetic_energy §46d: a linear
classifier on the exact Δp vector scored AUC 0.56, on |Δp| 1.00). So two more routes per cell, same grouped
folds / seeds, ridge penalty by inner grouped CV (physlaw.probes.ridge.fit_probe):
  * ``fitted_norm``    ridge activations -> true Δp = m_A dv_A + m_B dv_B (2-D, labels); score = |Δp̂|
  * ``composed_norm``  ridge activations -> true (dv_A, dv_B) (4-D); Δp̂ = m_A dv̂_A + m_B dv̂_B with the TRUE
                       masses; score = |Δp̂|
AUC of the score (violating = 1), cell selected on held-out groups like the logistic route; also in the
transfer test.

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

from physlaw.eval.l0_states import _lname, load_index, load_pooled
from physlaw.eval.l2_surprise import contrast_key, control_pairs, family_contrasts, group_windows, viols_of
from physlaw.probes.ridge import _inner_alpha, _SVDRidge, auc_score, feature_scale, fit_probe, grouped_kfold
from physlaw.probes.select import heldout_select

POOLS = {"roi": "roi_pair.post", "frame": "frame.post"}
L2B_LAWS = ("momentum", "gravity", "friction", "oscillation", "rolling", "angular_momentum")   # contrast ``law`` (default
#   momentum); never energy (spec §5.0.3 R5) or statics (a kink, l2_surprise.group_kink)
_G: dict = {}


# --------------------------------------------------------------------------- logistic probe

def _project(X, tr):
    """Standardise on ``tr``, SVD; returns (Z_tr, map(X_rows) -> Z)."""
    mx, sx = X[tr].mean(0), feature_scale(X[tr])
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


def cv_auc(X, y, groups, lams, k, fold_seeds, seed0=0):
    """Grouped CV logistic probe, penalty by inner grouped CV (AUC) on each fold's training groups.
    Returns (per-seed AUCs, oof decision values [fold_seeds, n], chosen lams)."""
    n = len(y)
    oofs, aucs, chosen = [], [], []
    for fs in range(fold_seeds):
        oof = np.zeros(n)
        for te in grouped_kfold(groups, k, seed0 + fs):
            tr = np.setdiff1d(np.arange(n), te)
            Ztr, proj = _project(X, tr)
            lam = _inner_lam(X[tr], y[tr], groups[tr], lams, k, seed0 + fs + 10_000)
            chosen.append(lam)
            oof[te] = _decision(logistic_fit(Ztr, y[tr], lam), proj(te))
        aucs.append(auc_score(y, oof))
        oofs.append(oof)
    return aucs, np.stack(oofs), chosen


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
    """(Δp [2] = m_A dv_A + m_B dv_B, (dv_A, dv_B) [4], masses [2]) of a collision clip (F1-F4) from its labels."""
    from physlaw.eval import laws
    pre, post, ctx = laws.from_labels(labels)
    m = np.asarray(ctx["m"], float)
    dv = np.asarray(post["v"], float)[:, :2] - np.asarray(pre["v"], float)[:, :2]
    return m @ dv, dv.reshape(4), m


# --------------------------------------------------------------------------- kinematic baseline

KINEMATIC_FEATURES = ("speed_A_post", "speed_B_post", "ke_ratio", "dv_A", "dv_B", "v_cm_post")
WINDOW_FEATURES = ("mean_speed", "end_speed", "displacement", "mean_spin")


def _is_collision(labels: dict) -> bool:
    from physlaw.data.io import COLLISION_FAMILIES
    return labels["physlaw"].get("family", "f1") in COLLISION_FAMILIES


def kinematic_feature_names(labels: dict) -> list[str]:
    if _is_collision(labels):
        return list(KINEMATIC_FEATURES)
    objs = [n for n, o in labels["physlaw"]["objects"].items() if not o.get("aux")]
    return [f"{f}_{n}" for n in objs for f in WINDOW_FEATURES]


def kinematic_features(labels: dict, t_end: float | None = None) -> np.ndarray:
    """True post-event kinematics. Collisions (F1-F4): |v_A'|, |v_B'|, KE'/KE, |Δv_A|, |Δv_B|, |v_cm'| at the contact.
    Other families: per non-aux object, over the labelled frames from the first post-event frame to ``t_end`` (the
    group's clean window; default the clip end), WINDOW_FEATURES = mean speed, speed at the window's end, net
    displacement, mean |angular velocity|. Needs per-frame labels."""
    if _is_collision(labels):
        from physlaw.eval import laws
        pre, post, ctx = laws.from_labels(labels)
        m, v0, v1 = np.asarray(ctx["m"], float), np.asarray(pre["v"], float), np.asarray(post["v"], float)
        ke = lambda v: float((m * (v ** 2).sum(-1)).sum())
        sp, dv = np.linalg.norm(v1, axis=-1), np.linalg.norm(v1 - v0, axis=-1)
        return np.array([sp[0], sp[1], ke(v1) / ke(v0), dv[0], dv[1], np.linalg.norm(m @ v1) / m.sum()])
    from physlaw.eval.l2_surprise import event_time_s
    pl, pf = labels["physlaw"], labels.get("per_frame")
    if not pf:
        raise ValueError(f"kinematic_features: {pl.get('family')} labels without per_frame (re-run stage 1 labels)")
    objs = [n for n, o in pl["objects"].items() if not o.get("aux")]
    t0, t1 = event_time_s(labels), np.inf if t_end is None else float(t_end)
    fr = [f for f in pf if t0 - 1e-9 <= f["time_s"] <= t1 + 1e-9] or [pf[-1]]
    out = []
    for n in objs:
        v = np.array([f[n]["velocity"] for f in fr], float)
        w = np.array([f[n].get("angular_velocity", [0.0, 0.0, 0.0]) for f in fr], float)
        sp = np.linalg.norm(v, axis=-1)
        out += [sp.mean(), sp[-1], np.linalg.norm(np.subtract(fr[-1][n]["position"], fr[0][n]["position"])),
                np.linalg.norm(w, axis=-1).mean()]
    return np.array(out, float)


def _run_cell(job):
    key, X = job, _G["feat"](job)
    lams, k, fs = _G["lams"], _G["k"], _G["fold_seeds"]
    aucs, oof, _ = cv_auc(X, _G["y"], _G["groups"], lams, k, fs)
    out = {"auc_mean": float(np.mean(aucs)), "auc_sd": float(np.std(aucs)), "oof": oof}
    if _G.get("targets") is not None and key[0] == "act":      # norm routes (ridge -> |Δp̂|)
        for route, (T, comb) in _G["targets"].items():
            pr = fit_probe(X, T, _G["groups"], alphas=lams, k=k, fold_seeds=fs)
            sc = np.stack([comb(o) for o in pr.oof])             # [fold_seeds, n]
            out[route] = {"oof": sc, "auc_mean": float(np.mean([auc_score(_G["y"], o) for o in sc])),
                          "r2_mean": pr.score_mean}
    return key, out


def paired_auc_diff(y: np.ndarray, groups: np.ndarray, oof_a: np.ndarray, oof_b: np.ndarray | None = None,
                    keep_groups=None, n_boot: int = 1000, seed: int = 0) -> dict:
    """AUC of out-of-fold scores ``oof_a`` ([seeds, n]; AUC = mean over seeds) with its group-bootstrap CI, and
    with ``oof_b`` the paired difference AUC(a) − AUC(b) (the same group draw for both). ``keep_groups``: restrict
    to these groups."""
    groups = np.asarray(groups)
    ug = np.unique(groups) if keep_groups is None else np.array(sorted(set(keep_groups)))
    rows_of = [np.flatnonzero(groups == g) for g in ug]
    auc = lambda o, r: float(np.mean([auc_score(y[r], x[r]) for x in np.atleast_2d(o)]))
    allr = np.concatenate(rows_of) if rows_of else np.array([], int)
    nan2 = [float("nan")] * 2
    out = {"n_groups": int(len(ug)), "auc_a": auc(oof_a, allr) if len(allr) else float("nan")}
    if oof_b is not None:
        out["auc_b"] = auc(oof_b, allr) if len(allr) else float("nan")
        out["diff"] = out["auc_a"] - out["auc_b"]
    rng = np.random.default_rng(seed)
    ba, bd = [], []
    for _ in range(n_boot if len(ug) > 1 else 0):
        r = np.concatenate([rows_of[i] for i in rng.integers(0, len(ug), len(ug))])
        ba.append(auc(oof_a, r))
        if oof_b is not None:
            bd.append(ba[-1] - auc(oof_b, r))
    q = lambda b: ([float(np.nanquantile(b, 0.025)), float(np.nanquantile(b, 0.975))]
                   if np.isfinite(b).any() else nan2)
    out["auc_a_ci95"] = q(np.asarray(ba, float)) if ba else nan2
    if oof_b is not None:
        out["diff_ci95"] = q(np.asarray(bd, float)) if bd else nan2
    return out


def detection_verdict(rec: dict) -> str:
    """``detected`` only if the selected cell's AUC CI is above 0.5, its paired CI over the kinematic baseline is
    above 0 and, when the selected cell is an ROI cell, its paired CI over the ROI mask floor is above 0 (audit M0);
    else ``not detected: <the first failed condition>``."""
    lo = lambda d, k: (d or {}).get(k, [float("nan")])[0]
    if not lo(rec["overall"], "auc_ci95") > 0.5:
        return "not detected: AUC CI not above 0.5"
    if not lo(rec.get("vs_kinematic"), "diff_ci95") > 0:
        return "not detected beyond the kinematic baseline"
    if rec["overall"].get("vs_mask_floor") is not None and not lo(rec["overall"]["vs_mask_floor"], "diff_ci95") > 0:
        return "not detected beyond the ROI mask floor (selected ROI cell)"
    return "detected (beyond the kinematic baseline" + (" and the ROI mask floor)" if rec["overall"].get("vs_mask_floor")
                                                        is not None else "; frame pool)")


def ke_ratio_to_legal(lab_viol: dict, lab_legal: dict) -> float | None:
    """KE of the violating twin / KE of its legal twin: labels physlaw.ke_ratio_to_legal, else summary ke_post
    (collision labels); None when neither is labelled."""
    r = lab_viol["physlaw"].get("ke_ratio_to_legal")
    if r is not None:
        return float(r)
    kv, kl = ((l["physlaw"].get("summary") or {}).get("ke_post") for l in (lab_viol, lab_legal))
    return float(kv) / float(kl) if kv is not None and kl else None


# --------------------------------------------------------------------------- run

def run(cache_dir: Path, data_dir: Path, cfg, workers: int = 8, log=print) -> dict:
    probe = cfg["probe"]
    lams = tuple(float(a) for a in cfg.get("l2b", {}).get("lambdas", probe["alphas"]))
    k = int(probe["k_folds"])
    fold_seeds = int(cfg.get("l2b", {}).get("fold_seeds", 4))
    n_splits = int(cfg.get("selection", {}).get("n_splits", 50))
    recs = load_index(cache_dir)        # every clip: a contrast keeps the groups holding its twins (sub-families of
    clips = [r["clip"] for r in recs]   # one family have different twin sets, e.g. F6 glide / slide / pair, F9 pair / swing)
    acts, pools, layers, _ = load_pooled(cache_dir, recs)
    with np.load(cache_dir / "_maskfeat_v2.npz") as z:       # read eagerly: forked workers must not share the zip
        mf = {k: z[k] for k in z.files if k == "clips" or k.startswith("w.")}
    mrow = {str(c): i for i, c in enumerate(mf["clips"])}
    l2b = cfg.get("l2b", {})
    tol = float(l2b.get("speed_matched_tol", 0.05))
    n_boot = int(l2b.get("n_boot", 1000))
    from physlaw.data.io import COLLISION_FAMILIES
    fam = data_dir.name
    _, cons = family_contrasts(cfg, fam)
    cons = [c for c in cons if c.get("law", "momentum") in L2B_LAWS]
    if not cons:
        raise ValueError(f"L2b: no contrasts of the laws {L2B_LAWS} configured for {fam} (energy and statics are not "
                         "scored by L2b: spec §5.0.3 R5; see L2-rep and the L2 kinks)")
    cons += [{"viol": v, "legal": l, "control": True} for v, l in control_pairs(cfg, fam)]
    row = {c: i for i, c in enumerate(clips)}
    labs = {}

    def lab(c):
        if c not in labs:
            labs[c] = json.loads((data_dir / c / "labels.json").read_text())
        return labs[c]
    # kinematic baseline: collisions at the contact; other families over each group's clean window
    win = {} if fam in COLLISION_FAMILIES else group_windows(data_dir, lab)
    t_end = lambda c: (win.get(c.split("/")[0]) or {}).get("t_end")
    kin_cache: dict = {}

    def kin(rows):                        # [len(rows), F]; F depends on the sub-family (objects), fixed per contrast
        for r in rows:
            if r not in kin_cache:
                kin_cache[r] = kinematic_features(lab(clips[r]), t_end(clips[r]))
        return np.stack([kin_cache[r] for r in rows])
    if fam in COLLISION_FAMILIES:          # norm routes: the momentum change needs two colliding bodies
        mt = [momentum_targets(lab(c)) for c in clips]
        DP, DV, MS = (np.stack([t[i] for t in mt]) for i in range(3))
    routes = ("fitted_norm", "composed_norm") if fam in COLLISION_FAMILIES and l2b.get("norm_routes", True) else ()
    keep: dict = {}          # contrast key -> (rows, y, groups, selected cell's job), for the transfer test
    ugroups = sorted({r["group"] for r in recs})
    out = {"n_groups": len(ugroups), "protocol": {
        "k_folds": k, "fold_seeds": fold_seeds, "test_groups_per_fold": len(ugroups) / k, 
        "lambdas": list(lams), "selection_splits": n_splits, "n_cells": len(layers) * len(POOLS),
        "note": f"grouped {k}-fold x {fold_seeds} seeds cross-fitting (spec §5: allowed below 100 groups)"},
        "kinematic_features": (list(KINEMATIC_FEATURES) if fam in COLLISION_FAMILIES
                               else [f"{f} (per object, clean window)" for f in WINDOW_FEATURES]), "contrasts": {}}
    for con in cons:
        key, viols = contrast_key(con), viols_of(con)
        tw = (*viols, con["legal"])
        gs = [g for g in ugroups if all(f"{g}/{t}" in row for t in tw)]
        if len(gs) < 2 * k:
            log(f"[L2b] {key}: {len(gs)} groups with these twins (need >= {2 * k}): skipped")
            continue
        rows = np.array([row[f"{g}/{t}"] for g in gs for t in tw])
        if any(clips[r] not in mrow for r in rows):
            raise RuntimeError(f"_maskfeat_v2.npz lacks clips of {key}: run L0 first (it builds the mask weights)")
        mrows = np.array([mrow[clips[r]] for r in rows])
        y = np.tile([1.0] * len(viols) + [0.0], len(gs))
        groups = np.repeat(gs, len(tw))
        feat = lambda j, rows=rows, mrows=mrows: (acts[rows, j[1], j[2]].astype(np.float64) if j[0] == "act"
                                                  else mf[f"w.{j[1]}"][mrows].astype(np.float64))
        tg = None
        if routes:
            tg = {"fitted_norm": (DP[rows], lambda P: np.linalg.norm(P, axis=-1)),
                  "composed_norm": (DV[rows], lambda P, ms=MS[rows]: np.linalg.norm(ms[:, :1] * P[:, :2] + ms[:, 1:] * P[:, 2:], axis=-1))}
        _G.update(feat=feat, y=y, groups=groups, lams=lams, k=k, fold_seeds=fold_seeds, targets=tg)
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
               "law": None if con.get("control") else con.get("law", "momentum"),
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
        oof_m = cells[rec["overall"]["selected"]]["oof"]
        rec["overall"]["auc_ci95"] = paired_auc_diff(y, groups, oof_m, n_boot=n_boot)["auc_a_ci95"]
        # audit M0: the ROI pool sits where each twin's objects went, so an ROI cell must beat the same probe on the
        # pooling weights alone (paired group-bootstrap CI of ROI AUC − mask-floor AUC, full-sample selected cells)
        oof_mask = res[("mask", POOLS["roi"])]["oof"]
        rec["pools"]["roi"]["vs_mask_floor"] = paired_auc_diff(y, groups, cells[rec["pools"]["roi"]["selected"]]["oof"],
                                                               oof_mask, n_boot=n_boot)
        roi_sel = rec["overall"]["selected"].endswith("/" + POOLS["roi"])
        rec["overall"]["vs_mask_floor"] = (paired_auc_diff(y, groups, oof_m, oof_mask, n_boot=n_boot) if roi_sel
                                           else None)
        sm_groups = None
        if not con.get("control") and len(viols) == 1:      # KE-matched stratum (collision labels carry the KE ratio)
            kr = {g: ke_ratio_to_legal(lab(f"{g}/{viols[0]}"), lab(f"{g}/{con['legal']}")) for g in gs}
            kr = {g: r for g, r in kr.items() if r is not None and r > 0}
            if kr:
                sm_groups = [g for g in kr if abs(np.log(kr[g])) < tol]
                rec["speed_matched"] = {"tol_abs_log_ke_ratio": tol, "n_groups": len(sm_groups),
                                        "median_abs_log_ke_ratio_all": float(np.median(np.abs(np.log(list(kr.values())))))}
        ka, koof, _ = cv_auc(kin(rows), y, groups, lams, k, fold_seeds)
        rec["kinematic"] = {"auc_mean": float(np.mean(ka)), "auc_sd": float(np.std(ka)),
                            "features": kinematic_feature_names(lab(clips[rows[0]]))}
        rec["delta_auc_vs_kinematic"] = rec["overall"]["score"] - rec["kinematic"]["auc_mean"]
        rec["vs_kinematic"] = paired_auc_diff(y, groups, oof_m, koof, n_boot=n_boot)
        if sm_groups is not None and len(sm_groups) > 1:
            d = paired_auc_diff(y, groups, oof_m, koof, keep_groups=sm_groups, n_boot=n_boot)
            rec["speed_matched"].update(model_auc=d["auc_a"], kinematic_auc=d["auc_b"], diff=d["diff"],
                                        diff_ci95=d["diff_ci95"])
        rec["verdict"] = detection_verdict(rec)
        keep[key] = (rows, y, groups, cell_job[rec["overall"]["selected"]],
                     {r: cell_job[v["selected"]] for r, v in rec["norm_routes"].items()})
        out["contrasts"][key] = rec
        log(f"[L2b] {key}: held-out AUC {rec['overall']['score']:.3f} ({rec['overall']['selected']}), "
            f"CI {rec['overall']['auc_ci95']}, ROI mask floor {rec['pools']['roi']['mask_floor_auc']:.3f}, "
            f"kinematic AUC {rec.get('kinematic', {}).get('auc_mean', float('nan')):.3f}"
            + "".join(f", {r} {v['score']:.3f}" for r, v in rec["norm_routes"].items()))
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
        ka = transfer_auc(kin(rtr), ytr, gtr, kin(rte), yte, gte, lams, k, fold_seeds)
        tr.update(kinematic_auc=float(np.mean(ka)), kinematic_auc_sd=float(np.std(ka)))
        out["transfer"][f"{ktr} -> {kte}"] = tr
        log(f"[L2b] transfer {ktr} -> {kte}: model AUC {tr['model_auc']:.3f} ({tr['cell']}), "
            f"kinematic AUC {tr.get('kinematic_auc', float('nan')):.3f}, "
            + ", ".join(f"{r} {v['auc']:.3f}" for r, v in tr["norm_routes"].items()))
    return out


def table(res: dict) -> str:
    pr = res["protocol"]
    rows = [f"n_groups = {res['n_groups']} ({pr['note']}; ~{pr['test_groups_per_fold']:.0f} test groups per fold); "
            f"variants searched = {pr['n_cells']} (layers x 2 pools); layer selected on half the groups, AUC on the other half "
            f"({2 * pr['selection_splits']} evaluations); CI: group bootstrap of the selected cell's out-of-fold AUC. a probe can separate twins by how much "
            "the objects move, so read each AUC against the legal-vs-legal reference row (two legal outcomes, different "
            "motion), the ROI mask floor, the kinematic baseline and the speed-matched stratum. Energy is not scored "
            "here (spec §5.0.3 R5; see L2-rep).", "",
            "Kinematic baseline: the same probe protocol on the true post-event kinematics ("
            + ", ".join(res.get("kinematic_features", KINEMATIC_FEATURES)) + "); the model detects the violation beyond "
            "speed only if the paired group-bootstrap CI of model − kinematic AUC (selected cell) is above 0. Speed-matched: "
            "groups whose violator's KE is within the tolerance of its legal twin's. Norm routes: ridge to the true Δp (fitted_norm) or to (dv_A, dv_B) "
            "composed with true masses (composed_norm), score |Δp̂|; 'conserved' is a threshold on a norm, which a linear "
            "logistic probe cannot express.", "",
            "| contrast | n groups | held-out AUC, overall (selected cell) [CI] | frame pool AUC (layer) | ROI pool AUC (layer) | "
            "ROI mask floor | kinematic AUC | ΔAUC vs kinematic | model − kinematic, selected cell [paired CI] | "
            "speed-matched: n groups, model / kinematic AUC, diff [CI] | ΔAUC vs mask floor | ROI − mask floor [paired CI] | "
            "fitted_norm |Δp̂| AUC (held-out) | composed_norm AUC (held-out) | verdict |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, v in res["contrasts"].items():
        o, fr, ro = v["overall"], v["pools"]["frame"], v["pools"]["roi"]
        vk = v.get("vs_kinematic")
        vk = f"{vk['diff']:+.3f} [{vk['diff_ci95'][0]:+.3f}, {vk['diff_ci95'][1]:+.3f}]" if vk else ""
        sm = v.get("speed_matched", {})
        sm = (f"{sm['n_groups']}, {sm.get('model_auc', float('nan')):.3f} / {sm.get('kinematic_auc', float('nan')):.3f}"
              + (f", {sm['diff']:+.3f} [{sm['diff_ci95'][0]:+.3f}, {sm['diff_ci95'][1]:+.3f}]" if "diff" in sm else "")) if sm else ""
        kin = f"{v['kinematic']['auc_mean']:.3f}" if "kinematic" in v else ""
        dk = f"{v['delta_auc_vs_kinematic']:+.3f}" if "delta_auc_vs_kinematic" in v else ""
        dm = f"{v['delta_auc_vs_mask_floor']:+.3f}" if "delta_auc_vs_mask_floor" in v else ""
        vm = ro.get("vs_mask_floor")
        vm = f"{vm['diff']:+.3f} [{vm['diff_ci95'][0]:+.3f}, {vm['diff_ci95'][1]:+.3f}]" if vm else ""
        nr = [f"{v['norm_routes'][r]['score']:.3f} ({v['norm_routes'][r]['selected']})" if r in v.get("norm_routes", {}) else ""
              for r in ("fitted_norm", "composed_norm")]
        rows.append(f"| {c}{' (legal-vs-legal reference)' if v['control'] else ''} | {v['n_groups']} | "
                    f"{o['score']:.3f} ({o['selected']}) [{o['auc_ci95'][0]:.3f}, {o['auc_ci95'][1]:.3f}] | {fr['score']:.3f} ({fr['selected'].rsplit('/', 1)[0]}) | "
                    f"{ro['score']:.3f} ({ro['selected'].rsplit('/', 1)[0]}) | {ro['mask_floor_auc']:.3f} | {kin} | {dk} | {vk} | {sm} | {dm} | "
                    f"{vm} | {nr[0]} | {nr[1]} | {v.get('verdict', '')} |")
    if res.get("transfer"):
        rows += ["", "### Transfer (train on one contrast, score held-out groups of the other)", "",
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
