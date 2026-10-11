#!/usr/bin/env python3
"""§57 / §60 — the curve-aware detector: population γ (null) and per-scene curves.

Per scene (§60):

    python scripts/curve_detector.py --per-scene --captures outputs/scene_ladder_v1/*/activations

  For every seed and site, a quadratic in e through the seed's legal arms minus
  the honest one; score(x) = min over e of |x - curve(e)| divided by the family's
  extent, so a move along the family costs nothing. Honest and violating arms are
  scored against the same curve. Matched control: the held-out legal arm that
  moved from honest by the amount closest to the violating arm's, scored against
  a curve refit without it. Chord ablation: degree 1. Kinematic oracle: the same
  procedure on the labels' post-contact positions. Raw AUC honest vs violating,
  permutation null, and the seed count where violating is more off-curve than
  the matched legal move.

Population γ (§57):

    python scripts/curve_detector.py --ladder outputs/energy_ladder_v1/<model>/activations \
        --captures outputs/conservation_set_v1/<model>/activations

  γ(e) is the 20-seed mean displacement from the anchor arm on the ladder set.
  On the conservation set a ridge Â (honest clips only, grouped folds) predicts
  the deep post pool from the patch-embedding pre pool; the residual R is scored
  by its best alignment with the curve (max |cos| over 32 knots). Ablations:
  chord and random path. Also reports the averaging-cancellation diagnostic
  |mean δ| / mean |δ| against the random reference (§57 audit).
"""
import _momentum_path  # noqa: F401

import numpy as np

from physics_interp.momentum import cli
from physics_interp.momentum.captures import ClipSet, load_clipset
from physics_interp.momentum.ladders import ladders
from physics_interp.momentum.stats import (auc, cv_predict, eval_curve, nearest_on_curve, permutation_null,
                                           polyfit_curve)

E_MARGIN = 0.25


def family_score(X_knots, e_knots, x, degree, e_range):
    coef = polyfit_curve(e_knots, X_knots, degree)
    extent = float(np.linalg.norm(eval_curve(coef, e_knots.max())[0] - eval_curve(coef, e_knots.min())[0]))
    if extent == 0:
        return float("nan"), float("nan")
    d, e_at = nearest_on_curve(coef, x, *e_range)
    return d / extent, e_at


def per_scene(cs: ClipSet, pool: str, feature_fn, degree: int, seed: int, draws: int):
    lads = [L for L in ladders(cs, min_arms=3) if L.violating is not None and L.honest is not None]
    rows = {}
    for site in (cs.sites if feature_fn is None else ["labels"]):
        if feature_fn is None:
            X, valid = cs.features(site, pool)
        else:
            X = np.stack([feature_fn(l) for l in cs.labels])
            valid = np.ones(cs.n, bool)
        sh, sv, sm, e_h, skipped = [], [], [], [], 0
        for L in lads:
            knots, ek = L.knots_without_honest()
            if len(knots) < max(3, degree + 1) or not valid[knots].all() or not valid[[L.honest, L.violating]].all():
                skipped += 1
                continue
            e_range = (min(ek.min(), cs.restitution[L.honest]) - E_MARGIN, max(ek.max(), cs.restitution[L.honest]) + E_MARGIN)
            s_h, eh = family_score(X[knots], ek, X[L.honest], degree, e_range)
            s_v, _ = family_score(X[knots], ek, X[L.violating], degree, e_range)
            # matched legal: knot whose displacement from honest is closest to the violating arm's
            dv = np.linalg.norm(X[L.violating] - X[L.honest])
            dk = np.linalg.norm(X[knots] - X[L.honest], axis=1)
            j = int(np.argmin(np.abs(dk - dv)))
            rest = np.arange(len(knots)) != j
            if rest.sum() >= max(3, degree + 1):
                s_m, _ = family_score(X[knots][rest], ek[rest], X[knots[j]], degree, e_range)
            else:
                s_m = float("nan")
            sh.append(s_h); sv.append(s_v); sm.append(s_m); e_h.append(eh - cs.restitution[L.honest])
        sh, sv, sm = np.array(sh), np.array(sv), np.array(sm)
        if len(sh) == 0:
            continue
        scores = np.r_[sh, sv]
        y = np.r_[np.zeros(len(sh), bool), np.ones(len(sv), bool)]
        null = permutation_null(scores, y, draws=draws, seed=seed)
        ok = np.isfinite(sm)
        rows[site] = {
            "n_seeds": int(len(sh)), "skipped": skipped, "degree": degree,
            "auc_honest_vs_violating": null,
            "auc_matched_vs_violating": auc(np.r_[sm[ok], sv[ok]], np.r_[np.zeros(ok.sum(), bool), np.ones(ok.sum(), bool)]),
            "violating_more_off_curve_than_matched": f"{int((sv[ok] > sm[ok]).sum())}/{int(ok.sum())}",
            "mean_score": {"honest": float(sh.mean()), "matched_legal": float(np.nanmean(sm)), "violating": float(sv.mean())},
            "honest_e_extrapolation_median": float(np.median(np.abs(e_h))),
        }
    return rows


def post_positions(labels):
    fr = labels["per_frame"]
    return np.array([fr[f][t]["position"][:2] for f in range(21, len(fr)) for t in ("A", "B")]).ravel()


def population_gamma(ladder: ClipSet, cs: ClipSet, pool_pre: str, pool_post: str, seed: int, draws: int, knots: int = 32):
    out = {}
    ref_site = ladder.sites[0]
    rng = np.random.default_rng(seed)
    for site in ladder.sites:
        Xl, vl = ladder.features(site, pool_post)
        deltas, es = [], None
        for L in ladders(ladder, min_arms=3):
            if L.honest is None or not vl[L.idx].all():
                continue
            d = Xl[L.idx] - Xl[L.honest]
            deltas.append(d); es = L.e
        if not deltas:
            continue
        deltas = np.stack(deltas)                        # [seeds, arms, dim]
        gamma = deltas.mean(0)                           # [arms, dim]
        cancel = float(np.linalg.norm(gamma, axis=1).mean() / np.linalg.norm(deltas, axis=2).mean())
        rand_ref = float(np.linalg.norm(rng.normal(size=deltas.shape).mean(0), axis=1).mean()
                         / np.linalg.norm(rng.normal(size=deltas.shape), axis=2).mean())
        per_scene_align = []
        for s in range(len(deltas)):
            a, g = deltas[s].ravel(), gamma.ravel()
            per_scene_align.append(float(a @ g / (np.linalg.norm(a) * np.linalg.norm(g) + 1e-12)))
        # curve knots along e, chord, random path
        coef = polyfit_curve(es, gamma, 2)
        e_grid = np.linspace(es.min(), es.max(), knots)
        curve_knots = eval_curve(coef, e_grid)
        chord_knots = eval_curve(polyfit_curve(es, gamma, 1), e_grid)
        rand_knots = rng.normal(size=curve_knots.shape)
        # Â on the conservation set: embed.out pre -> this site's post, honest only
        Xpre, vpre = cs.features(ref_site, pool_pre)
        Xpost, vpost = cs.features(site, pool_post)
        valid = vpre & vpost
        idx = np.flatnonzero(valid)
        pred = np.full_like(Xpost, np.nan)
        pred[idx] = cv_predict(Xpre[idx], Xpost[idx], cs.seed[idx], seed=seed, fit_mask=~cs.violates[idx])
        R = Xpost - pred
        res = {}
        for nm, K in (("curve", curve_knots), ("chord", chord_knots), ("random", rand_knots)):
            Kn = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-12)
            Rn = R[idx] / (np.linalg.norm(R[idx], axis=1, keepdims=True) + 1e-12)
            align = np.abs(Rn @ Kn.T).max(1)
            score = -align   # less reachable by a legal move = more violating
            res[nm] = permutation_null(score, cs.violates[idx], draws=draws, seed=seed)
        out[site] = {"gamma_cancellation": cancel, "random_cancellation_reference": rand_ref,
                     "gamma_alignment_with_each_scene_median": float(np.median(per_scene_align)),
                     "auc": res, "curve_minus_chord": res["curve"]["auc"] - res["chord"]["auc"]}
    return out


def main():
    p = cli.base_parser(__doc__, site=False)
    p.set_defaults(pools=["all.post"])
    p.add_argument("--per-scene", action="store_true", help="§60 on a per-scene ladder set")
    p.add_argument("--ladder", default=None, help="§57: energy_ladder captures to learn γ from")
    p.add_argument("--pool-pre", default="all.pre")
    args = p.parse_args()
    sets = cli.load_sets(args)
    if args.per_scene:
        out = {"protocol": "§60", "models": {}}
        for name, cs in sets.items():
            depth = cs.depth()
            rec = {"depth": depth, "pools": {}}
            for pool in args.pools:
                block = {"curve": per_scene(cs, pool, None, 2, args.seed, args.draws),
                         "chord": per_scene(cs, pool, None, 1, args.seed, args.draws)}
                oracle = per_scene(cs, pool, post_positions, 2, args.seed, min(args.draws, 500))["labels"]
                block["kinematic_oracle"] = oracle
                rec["pools"][pool] = block
                print(f"\n[{name}] pool {pool}   kinematic oracle AUC {oracle['auc_honest_vs_violating']['auc']:.3f}, "
                      f"mean score honest {oracle['mean_score']['honest']:.3f} matched {oracle['mean_score']['matched_legal']:.3f} "
                      f"violating {oracle['mean_score']['violating']:.3f}")
                print(f"  {'site':24s} {'curve AUC':>20s} {'chord AUC':>9s} {'viol>matched':>12s} {'honest':>7s} {'matched':>7s} {'viol':>7s} {'Δe extrap':>9s}")
                for site in cs.sites:
                    c = block["curve"].get(site); ch = block["chord"].get(site)
                    if not c:
                        continue
                    ms = c["mean_score"]
                    print(f"  {site:24s} {cli.fmt_auc(c['auc_honest_vs_violating']):>20s} {ch['auc_honest_vs_violating']['auc']:9.3f} "
                          f"{c['violating_more_off_curve_than_matched']:>12s} {ms['honest']:7.3f} {ms['matched_legal']:7.3f} "
                          f"{ms['violating']:7.3f} {c['honest_e_extrapolation_median']:9.2f}")
            out["models"][name] = rec
        cli.write_artifact(out, args.out, "curve_detector_per_scene.json")
    else:
        if not args.ladder:
            p.error("pass --per-scene, or --ladder for the population-γ version")
        ladder = load_clipset(args.ladder, step=args.step)
        print(f"[ladder] {ladder.describe()}")
        out = {"protocol": "§57", "models": {}}
        for name, cs in sets.items():
            res = population_gamma(ladder, cs, args.pool_pre, args.pools[0], args.seed, args.draws)
            out["models"][name] = res
            print(f"\n[{name}] pool {args.pools[0]}")
            print(f"  {'site':24s} {'curve':>7s} {'chord':>7s} {'random':>7s} {'c-ch':>6s} {'cancel':>7s} {'rand ref':>8s} {'align/scene':>11s}")
            for site, r in res.items():
                a = r["auc"]
                print(f"  {site:24s} {a['curve']['auc']:7.3f} {a['chord']['auc']:7.3f} {a['random']['auc']:7.3f} "
                      f"{r['curve_minus_chord']:+6.2f} {r['gamma_cancellation']:7.3f} {r['random_cancellation_reference']:8.3f} "
                      f"{r['gamma_alignment_with_each_scene_median']:11.2f}")
        cli.write_artifact(out, args.out, "curve_detector_population.json")


if __name__ == "__main__":
    main()
