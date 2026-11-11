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
motion-adjusted summary). Large and positive: decodable, but the model's own prediction ignores it.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from physlaw.eval.l0_states import _lname, complete_groups, load_index, load_pooled
from physlaw.eval.l2_surprise import family_contrasts
from physlaw.probes.ridge import auc_score, grouped_kfold
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
                inner = np.zeros((len(lams), len(tr)))
                for ite in grouped_kfold(groups[tr], k, seed0 + fs + 10_000):
                    itr = np.setdiff1d(np.arange(len(tr)), ite)
                    Zi, pj = _project(X[tr], itr)
                    Zo = pj(ite)
                    for li, lam in enumerate(lams):
                        inner[li, ite] = _decision(logistic_fit(Zi, y[tr][itr], lam), Zo)
                sc = [auc_score(y[tr], inner[li]) for li in range(len(lams))]
                lam = lams[int(np.nanargmax(sc))]
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
    ugroups = sorted({r["group"] for r in recs})
    out = {"n_groups": len(ugroups), "protocol": {
        "k_folds": k, "fold_seeds": fold_seeds, "test_groups_per_fold": len(ugroups) / k, "n_permutations": n_perm,
        "lambdas": list(lams), "selection_splits": n_splits, "n_cells": len(layers) * len(POOLS),
        "note": f"grouped {k}-fold x {fold_seeds} seeds cross-fitting (spec §5: allowed below 100 groups)"},
        "contrasts": {}}
    for con in cons:
        key = f"{con['viol']} vs {con['legal']}"
        gs = [g for g in ugroups if f"{g}/{con['viol']}" in row and f"{g}/{con['legal']}" in row]
        rows = np.array([row[f"{g}/{t}"] for g in gs for t in (con["viol"], con["legal"])])
        y = np.tile([1.0, 0.0], len(gs))
        groups = np.repeat(gs, 2)
        perms = swap_perms(groups, y, n_perm, seed=0) if n_perm else None
        feat = lambda j, rows=rows: (acts[rows, j[1], j[2]].astype(np.float64) if j[0] == "act"
                                     else mf[f"w.{j[1]}"][rows].astype(np.float64))
        _G.update(feat=feat, y=y, groups=groups, lams=lams, k=k, fold_seeds=fold_seeds, perms=perms)
        jobs = [("act", pools.index(p), li) for p in POOLS.values() for li in range(len(layers))]
        jobs.append(("mask", POOLS["roi"]))
        res = {}
        with ProcessPoolExecutor(workers) as ex:       # fork: workers inherit _G
            for j, r in ex.map(_run_cell, jobs, chunksize=1):
                res[j] = r
        cells = {f"{_lname(layers[j[2]])}/{pools[j[1]]}": r for j, r in res.items() if j[0] == "act"}

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
        if perms is not None:
            best = max(cells, key=lambda c: cells[c]["quick"])
            null_max = np.max(np.stack([v["null"] for v in cells.values()]), 0)
            rec.update(best_cell_quick=best, null_max_q95=float(np.quantile(null_max, 0.95)),
                       p_corrected=float((1 + np.sum(null_max >= cells[best]["quick"])) / (len(null_max) + 1)))
        l2 = (l2_summary or {}).get(key)
        if l2 is not None and "score" in l2:
            rec["l2_adj_win_heldout"] = l2["score"]
            rec["encoded_but_unused"] = (rec["overall"]["score"] - 0.5) - (l2["score"] - 0.5)
        out["contrasts"][key] = rec
        log(f"[L2b] {key}: held-out AUC {rec['overall']['score']:.3f} ({rec['overall']['selected']}), "
            f"p_corr {rec.get('p_corrected', float('nan')):.3g}, ROI mask floor {rec['pools']['roi']['mask_floor_auc']:.3f}")
    ph = holm({c: v.get("p_corrected", np.nan) for c, v in out["contrasts"].items() if not v["control"]})
    for c, p in ph.items():
        out["contrasts"][c]["p_holm"] = p
    return out


def table(res: dict) -> str:
    pr = res["protocol"]
    rows = [f"n_groups = {res['n_groups']} ({pr['note']}; ~{pr['test_groups_per_fold']:.0f} test groups per fold); "
            f"variants searched = {pr['n_cells']} (layers x 2 pools); {pr['n_permutations']} within-group label-swap "
            f"permutations, max over the grid; layer selected on half the groups, AUC on the other half "
            f"({2 * pr['selection_splits']} evaluations). L2b is NOT motion-adjusted: a probe can separate twins by how much "
            "the balls move, so read each AUC against the legal-vs-legal reference row (two legal outcomes, different "
            "motion) and the ROI mask floor; the encoded-but-unused index inherits that confound.", "",
            "| contrast | n groups | held-out AUC, overall (selected cell) | frame pool AUC (layer) | ROI pool AUC (layer) | "
            "ROI mask floor | p_corr (Holm) | L2 adj. win (held-out) | encoded-but-unused |",
            "|---|---|---|---|---|---|---|---|---|"]
    for c, v in res["contrasts"].items():
        o, fr, ro = v["overall"], v["pools"]["frame"], v["pools"]["roi"]
        p = f"{v['p_corrected']:.3g} ({v.get('p_holm', float('nan')):.3g})" if "p_corrected" in v else ""
        l2 = f"{v['l2_adj_win_heldout']:.2f}" if "l2_adj_win_heldout" in v else ""
        ebu = f"{v['encoded_but_unused']:+.2f}" if "encoded_but_unused" in v else ""
        rows.append(f"| {c}{' (legal-vs-legal reference)' if v['control'] else ''} | {v['n_groups']} | "
                    f"{o['score']:.3f} ({o['selected']}) | {fr['score']:.3f} ({fr['selected'].rsplit('/', 1)[0]}) | "
                    f"{ro['score']:.3f} ({ro['selected'].rsplit('/', 1)[0]}) | {ro['mask_floor_auc']:.3f} | {p} | {l2} | {ebu} |")
    return "\n".join(rows)
