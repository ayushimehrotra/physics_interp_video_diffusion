#!/usr/bin/env python
"""Statistics for voe_run.py output. Nothing fit except the legal-only novelty curves.
    analyze.py momentum <jsonl> <out.json>
    analyze.py ladder   <jsonl> <out.json>
"""
from __future__ import annotations
import json, sys
from math import comb
import numpy as np

METS = (("l1", +1), ("mse", +1), ("cos", -1))   # sign: +1 -> larger is more surprising


def sign_p(k, n):
    return sum(comb(n, j) for j in range(k, n + 1)) / 2 ** n


def detectable(n, alpha=0.05):
    for k in range(n + 1):
        if sign_p(k, n) < alpha:
            return k
    return None


def surp(fut, tag, met, sign):
    d = fut.get(tag)
    return None if d is None else sign * d[met]   # oriented: larger = more surprising


# --------------------------------------------------------------------------- momentum
def momentum(rows):
    out = {"n_pairs": len(rows)}
    n = len(rows)
    kmin = detectable(n)
    out["detectable_at_p05"] = {"k": kmin, "n": n, "rate": kmin / n}
    # controls
    idv = np.array([r["identity"]["violating"] for r in rows])
    out["identity_cos"] = {"min": float(idv.min()), "mean": float(idv.mean())}
    out["ball_frac_mean"] = float(np.mean([r["ball_frac"] for r in rows]))
    sub = {}
    for tag in ("all", "balls"):
        for met in ("mse", "cos"):
            p = np.array([r["substrate"][tag]["pred"][met] for r in rows])
            q = np.array([r["substrate"][tag]["pers"][met] for r in rows])
            w = (p > q) if met == "cos" else (p < q)
            sub[f"{tag}.{met}"] = {"pred_mean": float(p.mean()), "pers_mean": float(q.mean()),
                                   "pred_wins": int(w.sum()), "n": n}
    out["substrate"] = sub
    out["room_cos_honest_violating"] = {
        tag: float(np.mean([r["futures"]["violating"]["room"][tag] for r in rows])) for tag in ("all", "balls")}
    # the measurement
    res = {}
    cells = ["ALL", "elastic_mixed", "elastic_same", "inelastic_mixed", "inelastic_same"]
    for tag in ("balls", "all", "own"):
        res[tag] = {}
        for met, sg in METS:
            res[tag][met] = {}
            for c in cells:
                rr = [r for r in rows if c == "ALL" or r["cell"] == c]
                v = np.array([surp(r["futures"]["violating"], tag, met, sg) for r in rr], float)
                h = np.array([surp(r["futures"]["honest"], tag, met, sg) for r in rr], float)
                ok = ~(np.isnan(v) | np.isnan(h))
                d = (v - h)[ok]
                k, m = int((d > 0).sum()), int(ok.sum())
                res[tag][met][c] = {"k_violating_worse": k, "n": m, "p_one_sided": sign_p(k, m),
                                    "p_two_sided": min(1.0, 2 * min(sign_p(k, m), sign_p(m - k, m))),
                                    "mean_diff": float(d.mean()), "median_diff": float(np.median(d))}
    out["voe"] = res
    # trend: per-pair slope of (viol - honest) per tubelet on ball tokens vs tubelet index
    tr = {}
    for met, sg in METS:
        slopes = []
        for r in rows:
            a = r["futures"]["violating"]["tub_balls"][met]
            b = r["futures"]["honest"]["tub_balls"][met]
            xs, ys = [], []
            for t, (x1, x2) in enumerate(zip(a, b)):
                if x1 is not None and x2 is not None:
                    xs.append(t); ys.append(sg * (x1 - x2))
            if len(xs) >= 3:
                slopes.append(np.polyfit(xs, ys, 1)[0])
        s = np.array(slopes)
        tr[met] = {"n": len(s), "mean_slope": float(s.mean()),
                   "t": float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))),
                   "n_positive": int((s > 0).sum()), "p_sign_positive": sign_p(int((s > 0).sum()), len(s))}
    out["trend_tubelet_slope_balls"] = tr
    return out


# --------------------------------------------------------------------------- ladder
ARMS = ["e0.2", "e0.3333", "e0.5", "e0.6667", "e0.8", "honest", "e1.25", "e1.5", "e2", "e3", "e5"]
EPLAN = dict(zip(ARMS, [0.2, 1 / 3, 0.5, 2 / 3, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]))
LEGAL = ARMS[:6]
MATCH = [("e1.25", "e0.8"), ("e1.5", "e0.6667"), ("e2", "e0.5"), ("e3", "e0.3333"), ("e5", "e0.2"),
         # matched departure from the anchor in pixel velocity (|e-1|) / in token space (room cos)
         ("e1.5", "e0.5"), ("e2", "e0.2"), ("e1.25", "e0.6667")]


def fit_eval(xl, yl, xi, kind):
    """Legal-only novelty curve in lam = (1+e)/2 (the pixel-velocity scale); predict xi."""
    xl, yl, xi = map(np.asarray, (xl, yl, xi))
    if kind == "quad":
        c = np.polyfit(xl, yl, 2)
        return np.polyval(c, xi), float(-c[1] / (2 * c[0])) if c[0] > 0 else None
    if kind == "vee":      # a + b|lam - lam0|, lam0 grid-searched, b >= 0
        best = None
        for l0 in np.linspace(0.5, 1.2, 141):
            A = np.c_[np.ones_like(xl), np.abs(xl - l0)]
            coef, *_ = np.linalg.lstsq(A, yl, rcond=None)
            if coef[1] < 0:
                continue
            sse = float(((A @ coef - yl) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, l0, coef)
        _, l0, coef = best
        return coef[0] + coef[1] * np.abs(xi - l0), float(l0)
    raise ValueError(kind)


def ladder(rows):
    out = {"n_seeds": len(rows)}
    out["arms_present"] = {a: sum(a in r["futures"] for r in rows) for a in ARMS}
    # controls
    idmin = min(min(v for v in r["identity"].values()) for r in rows)
    out["identity_cos_min_over_all_arms"] = idmin
    sub = {}
    for tag in ("all", "balls"):
        for met in ("mse", "cos"):
            p = np.array([r["substrate"][tag]["pred"][met] for r in rows])
            q = np.array([r["substrate"][tag]["pers"][met] for r in rows])
            w = (p > q) if met == "cos" else (p < q)
            sub[f"{tag}.{met}"] = {"pred_mean": float(p.mean()), "pers_mean": float(q.mean()),
                                   "pred_wins": int(w.sum()), "n": len(rows)}
    out["substrate"] = sub
    out["room_cos_vs_anchor_balls_median"] = {
        a: float(np.median([r["futures"][a]["room"]["balls"] for r in rows if a in r["futures"]]))
        for a in ARMS if a != "honest"}
    out["ball_frac_mean"] = float(np.mean([r["ball_frac"] for r in rows]))
    # realised e
    out["e_realised_median"] = {
        a: float(np.median([r["meta"][a]["restitution_realised"] for r in rows if a in r["futures"]]))
        for a in ARMS}
    res = {}
    for tag in ("balls", "all", "own"):
        res[tag] = {}
        for met, sg in METS:
            R = {}
            S = {a: {} for a in ARMS}
            for r in rows:
                for a in ARMS:
                    if a in r["futures"]:
                        S[a][r["seed"]] = surp(r["futures"][a], tag, met, sg)
            # table: median raw, median delta vs anchor, median within-seed rank (among the
            # 6 legal arms + this arm, 1 = least surprising; legal arms ranked among the 6)
            tab = {}
            for a in ARMS:
                seeds = list(S[a])
                raw = np.array([S[a][s] for s in seeds])
                dlt = np.array([S[a][s] - S["honest"][s] for s in seeds])
                ranks = []
                for s in seeds:
                    pool = [S[b][s] for b in LEGAL if b != a] + [S[a][s]]
                    ranks.append(1 + sum(v < S[a][s] for v in pool))
                tab[a] = {"e": EPLAN[a], "n": len(seeds), "median": float(np.median(raw)),
                          "median_minus_anchor": float(np.median(dlt)),
                          "n_worse_than_anchor": int((dlt > 0).sum()),
                          "median_rank_among_legal_plus_this": float(np.median(ranks))}
            # persistence error per arm (motion-amount baseline) and predictor skill = pred/pers
            if tag in ("all", "balls") and all("pers" in r["futures"][a] for r in rows for a in r["futures"]):
                for a in ARMS:
                    ss = [r for r in rows if a in r["futures"]]
                    pe = np.array([sg * r["futures"][a]["pers"][tag][met] for r in ss])
                    pa = np.array([sg * r["futures"]["honest"]["pers"][tag][met] for r in ss])
                    pr = np.array([sg * r["futures"][a][tag][met] for r in ss])
                    tab[a]["pers_median_minus_anchor"] = float(np.median(pe - pa))
                    tab[a]["pred_minus_pers_median"] = float(np.median(pr - pe))
            R["table"] = tab
            # minimum location
            argmins_all, argmins_legal = [], []
            for r in rows:
                s = r["seed"]
                av = [(S[a][s], a) for a in ARMS if s in S[a]]
                argmins_all.append(min(av)[1])
                argmins_legal.append(min((S[a][s], a) for a in LEGAL)[1])
            R["argmin_all_arms_counts"] = {a: argmins_all.count(a) for a in ARMS if argmins_all.count(a)}
            R["argmin_legal_counts"] = {a: argmins_legal.count(a) for a in LEGAL if argmins_legal.count(a)}
            med = {a: tab[a]["median_minus_anchor"] for a in ARMS}
            R["argmin_of_median_curve"] = min(med, key=med.get)
            # matched |log e| pairs
            mt = {}
            for imp, leg in MATCH:
                ss = [s for s in S[imp] if s in S[leg]]
                d = np.array([S[imp][s] - S[leg][s] for s in ss])
                k = int((d > 0).sum())
                mt[f"{imp}_vs_{leg}"] = {"k_imp_worse": k, "n": len(ss), "p_one_sided": sign_p(k, len(ss)),
                                         "p_two_sided": min(1.0, 2 * min(sign_p(k, len(ss)), sign_p(len(ss) - k, len(ss)))),
                                         "median_diff": float(np.median(d)) if len(d) else None}
            R["matched_logE"] = mt
            # legal-only novelty curves, extrapolated
            nov = {}
            for kind in ("vee", "quad"):
                resid = {a: [] for a in ARMS[6:]}
                lam0s = []
                insample_rmse = []
                for r in rows:
                    s = r["seed"]
                    xl = [(1 + EPLAN[a]) / 2 for a in LEGAL]
                    yl = [S[a][s] for a in LEGAL]
                    imps = [a for a in ARMS[6:] if s in S[a]]
                    yhat_l, _ = fit_eval(xl, yl, xl, kind)
                    insample_rmse.append(float(np.sqrt(np.mean((np.array(yl) - yhat_l) ** 2))))
                    if not imps:
                        continue
                    yhat, l0 = fit_eval(xl, yl, [(1 + EPLAN[a]) / 2 for a in imps], kind)
                    lam0s.append(l0)
                    for a, yh in zip(imps, yhat):
                        resid[a].append(S[a][s] - yh)
                allres = np.concatenate([np.array(v) for v in resid.values() if v])
                k = int((allres > 0).sum())
                nov[kind] = {
                    "per_arm": {a: {"n": len(v), "median_resid": float(np.median(v)) if v else None,
                                    "n_above": int((np.array(v) > 0).sum()),
                                    "p_one_sided": sign_p(int((np.array(v) > 0).sum()), len(v))}
                                for a, v in resid.items()},
                    "pooled": {"k_above": k, "n": len(allres), "p_one_sided": sign_p(k, len(allres)),
                               "p_two_sided": min(1.0, 2 * min(sign_p(k, len(allres)), sign_p(len(allres) - k, len(allres))))},
                    "insample_rmse_median": float(np.median(insample_rmse)),
                    "lam0_median": (float(np.median([l for l in lam0s if l is not None]))
                                    if any(l is not None for l in lam0s) else None)}
                if nov[kind]["lam0_median"] is not None:
                    nov[kind]["e_star_median"] = 2 * nov[kind]["lam0_median"] - 1
            R["novelty_fit_legal_only"] = nov
            # jump at e = 1: slope per unit lambda, last legal step vs first impossible step
            jl, jc = [], []
            for r in rows:
                s = r["seed"]
                if s in S["e1.25"]:
                    jl.append((S["honest"][s] - S["e0.8"][s]) / 0.1)
                    jc.append((S["e1.25"][s] - S["honest"][s]) / 0.125)
            jl, jc = np.array(jl), np.array(jc)
            # curvature reference: the legal step before (0.6667 -> 0.8), per unit lambda
            k = int((jc > jl).sum())
            R["jump_at_1"] = {"n": len(jl), "median_slope_0.8_to_1": float(np.median(jl)),
                              "median_slope_1_to_1.25": float(np.median(jc)),
                              "k_cross_steeper": k, "p_one_sided": sign_p(k, len(jl))}
            # second differences inside the legal range, for comparison (is the curve convex anyway?)
            sl1, sl2 = [], []
            for r in rows:
                s = r["seed"]
                sl1.append((S["e0.8"][s] - S["e0.6667"][s]) / ((0.8 - 2 / 3) / 2))
                sl2.append((S["honest"][s] - S["e0.8"][s]) / 0.1)
            sl1, sl2 = np.array(sl1), np.array(sl2)
            k2 = int((sl2 > sl1).sum())
            R["legal_convexity_ref"] = {"median_slope_0.667_to_0.8": float(np.median(sl1)),
                                        "median_slope_0.8_to_1": float(np.median(sl2)),
                                        "k_second_steeper": k2, "n": len(sl1), "p_one_sided": sign_p(k2, len(sl1))}
            res[tag][met] = R
    out["voe"] = res
    return out


if __name__ == "__main__":
    exp, src, dst = sys.argv[1:4]
    rows = [json.loads(l) for l in open(src)]
    # dedupe by gid (resumed runs)
    rows = list({r["gid"]: r for r in rows}.values())
    rep = momentum(rows) if exp == "momentum" else ladder(rows)
    json.dump(rep, open(dst, "w"), indent=1)
    print(json.dumps(rep, indent=1)[:200000])
