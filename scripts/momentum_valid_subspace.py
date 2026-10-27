#!/usr/bin/env python3
"""The subspace of valid collisions, read from PRE-COLLISION clips, and how violating ones leave it.

    python scripts/momentum_valid_subspace.py \
        --model videomaev2-giant \
        --full  outputs_cons_videomaev2-giant/momentum_probe/activations \
        --trunc outputs_construnc_videomaev2-giant/activations

PART A — THE VALID SUBSPACE, FROM BEFORE CONTACT. On every resample-truncation
rung (``k*/``; k <= 20 means the encoder saw nothing after contact) decode the
outcome ``(dv_A, dv_B)`` the physics dictates — the legal outcome at the clip's
restitution — with a readout fit on honest clips only. For a smooth frictionless
pair that outcome lies on ONE line per scene, ``J * (-n/m_A, +n/m_B)``; the
other three directions of the scene's basis (tangential impulse, common drift
along and across n) are forbidden. So the decoded outcome is decomposed in that
basis and scored by how much of it stays on the line: tangential fraction,
relative drift (its own momentum violation), implied vs true restitution.
Twins share a bit-identical pre-contact segment, so on k <= 20 an honest clip
and its violating twin must encode identically; that is checked, not assumed.

PART B — HOW VIOLATING COLLISIONS DEVIATE. On full-clip captures decode the
OBSERVED outcome, again with a readout fit on honest clips only (it has never
seen a violation), and ask:

  * membership — distance of the decoded outcome from the scene's legal line
    over its length (§49). AUC honest vs violating, and the exact paired sign
    test over twins. Group permutation is a no-op on a balanced twin design
    (§46d), so the paired test is the null, not a shuffled-group one.
  * direction — the true twin difference is a COMMON DRIFT along the contact
    normal (§42a: the edit keeps the impulse sum and moves its split). Decompose
    the DECODED twin difference in the same basis: how much is drift (read as a
    momentum change), how much a change of impulse (read as a different
    collision), how much tangential.
  * expectation gap — distance between the outcome decoded from the full clip
    and the one decoded from the k = 20 pre-collision clip, honest vs violating.
  * representation — share of the twin activation difference lying in the span
    of the honest outcome readout, against random subspaces of equal rank.

Cells are chosen by honest-clip R^2 only; no violation label enters selection.
CPU only, reads captures.
"""
import _momentum_path  # noqa: F401

import argparse
import json
import math
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from _fast_cv import cv_predict, fit_full
from physics_interp.momentum import cli, physics
from physics_interp.momentum.captures import load_clipset
from physics_interp.momentum.stats import auc, bootstrap_ci, r2

TRUNC_POOLS = ["A.all", "B.all", "all.all", "A.all+B.all"]     # resample rungs: only *.all is exact
FULL_POOLS = ["A.pre", "A.post", "A.all", "B.pre", "B.post", "B.all", "all.pre", "all.post", "all.all",
              "A.post+B.post"]

_G: dict = {}     # worker globals, filled before the fork


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #


def pre_state(labels) -> physics.PreState:
    """``physics.pre_state_from_labels`` with the SOLVER's contact normal.

    The library prefers ``contact_normal_analytic``, which on this set sits up to
    0.66 deg off the true impulse direction: the honest outcome then misses its own
    legal line by up to 2% and scores membership 0.012 instead of 0 (the §46a bug).
    ``contact_normal`` is the simulator's event normal, recorded before any edit, and
    agrees with dv_A's direction to 1e-6 deg on every honest clip.
    """
    col = labels["collision"]
    n = physics._xy(col["contact_normal"])
    n /= np.linalg.norm(n)
    pre = physics.pre_state_from_labels(labels)
    if np.dot(pre.v_a - pre.v_b, n) < 0:
        n = -n
    pre = physics.PreState(pre.m_a, pre.m_b, pre.v_a, pre.v_b, n)
    pre.check()
    return pre


def clip_physics(cs):
    pres = [pre_state(l) for l in cs.labels]
    e = cs.restitution
    legal = np.stack([physics.legal_outcome(p, ei) for p, ei in zip(pres, e)])
    observed = np.stack([physics.outcome_from_labels(l) for l in cs.labels])
    cell = np.array(["_".join(c.split("_")[:2]) for c in cs.clip_ids])
    return pres, legal, observed, cell


def twin_pairs(cs) -> list[tuple[int, int]]:
    """``(honest, violating)`` index pairs by seed. ``ClipSet.twins`` keys on ``arm``,
    which these labels set to the material arm ('legacy'), so pair on the flag."""
    out = []
    for s in np.unique(cs.seed):
        idx = np.flatnonzero(cs.seed == s)
        h = [i for i in idx if not cs.violates[i]]
        v = [i for i in idx if cs.violates[i]]
        if len(h) == 1 and len(v) == 1:
            out.append((h[0], v[0]))
    return out


def geometry_control(pres, legal, observed, violates):
    """Honest outcomes must BE the legal outcome, and decompose onto the line exactly."""
    worst_fit = float(np.max(np.abs(legal[~violates] - observed[~violates])))
    worst = 0.0
    for p, o in zip(pres, legal):
        d = physics.decompose(o, p)
        worst = max(worst, abs(d["drift_n"]), abs(d["drift_t"]), abs(d["J_tangential"]) / abs(d["J_normal"]))
    if worst > 1e-9 or worst_fit > 1e-6:
        raise SystemExit(f"geometry control failed: decomposition {worst:.2e}, honest legal-vs-observed {worst_fit:.2e}")
    return {"decomposition_worst": worst, "honest_legal_vs_observed_worst": worst_fit}


# --------------------------------------------------------------------------- #
# the grid, parallel over (site, pool)
# --------------------------------------------------------------------------- #


def _cell(args):
    key, site, pool = args
    cs, Y, fit_mask = _G[key]
    X, valid = cs.features(site, pool)
    idx = np.flatnonzero(valid)
    pred = np.full_like(Y, np.nan)
    pred[idx] = cv_predict(X[idx], Y[idx], cs.seed[idx], fit_mask=fit_mask[idx])
    return site, pool, pred


def run_grid(key, cs, Y, fit_mask, pools, workers):
    _G[key] = (cs, Y, fit_mask)
    jobs = [(key, s, p) for s in cs.sites for p in pools]
    with Pool(workers) as pool:
        out = pool.map(_cell, jobs, chunksize=1)
    del _G[key]
    return {(s, p): pred for s, p, pred in out}


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #


def sign_test(diff: np.ndarray) -> dict:
    """Exact one-sided binomial test that ``diff > 0`` more often than not (ties dropped)."""
    d = diff[np.isfinite(diff) & (diff != 0)]
    n, k = len(d), int((d > 0).sum())
    p = sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n if n else float("nan")
    return {"n": n, "positive": k, "fraction": k / n if n else float("nan"), "p_one_sided": p}


def decoded_geometry(o_hat, pres, e_true, rows):
    drift, tang, e_imp, onl = [], [], [], []
    for i in rows:
        d = physics.decompose(o_hat[i], pres[i])
        drift.append(d["relative_drift"]); tang.append(d["tangential_fraction"])
        e_imp.append(physics.implied_restitution(o_hat[i], pres[i]))
        onl.append(1.0 - physics.membership_score(o_hat[i], pres[i]) ** 2)
    e_imp = np.array(e_imp)
    return {
        "relative_drift_median": float(np.median(drift)),
        "tangential_fraction_median": float(np.median(tang)),
        "on_line_energy_fraction_median": float(np.median(onl)),
        "corr_implied_e_true_e": float(np.corrcoef(e_imp, e_true[rows])[0, 1]),
        "implied_e_median_by_true_e": {f"{v:.2f}": float(np.median(e_imp[np.isclose(e_true[rows], v)]))
                                       for v in np.unique(np.round(e_true[rows], 2))},
    }


def references(o_hat, legal, pres, e_true, rows, *, draws=200, seed=0):
    """Two brackets for Part A.

    * ``ceiling_without_e`` — restitution is latent until impact (it is pinned per
      cell, not per material), so even a perfect pre-collision observer knows the
      legal LINE but not where on it: the best it can do is the legal outcome at the
      mean e. Its R^2 is the ceiling for any k <= 20 readout.
    * ``shuffled_residuals`` — the decoded outcome's own errors, re-dealt to other
      scenes. Same error sizes, no relation to each scene's line: if the decoded
      outcome stays on its legal line more than this, the errors are STRUCTURED
      toward the valid subspace rather than isotropic.
    """
    e_bar = float(np.mean(e_true[rows]))
    o_bar = np.stack([physics.legal_outcome(pres[i], e_bar) for i in rows])
    rng = np.random.default_rng(seed)
    resid = o_hat[rows] - legal[rows]
    shuffled = []
    for _ in range(draws):
        fake = legal.copy()
        fake[rows] = legal[rows] + resid[rng.permutation(len(rows))]
        shuffled.append(decoded_geometry(fake, pres, e_true, rows))
    obs = decoded_geometry(o_hat, pres, e_true, rows)
    # direction in which "more structured than shuffled" lies for each statistic
    keys = {"relative_drift_median": -1, "tangential_fraction_median": -1, "on_line_energy_fraction_median": +1}
    sh = {}
    for k, sign in keys.items():
        vals = np.array([s[k] for s in shuffled])
        sh[k] = float(np.median(vals))
        sh[k + "_p_structured"] = float((1 + np.sum(sign * vals >= sign * obs[k])) / (draws + 1))
    return {
        "ceiling_without_e": {"e_bar": e_bar, "r2": r2(legal[rows], o_bar)},
        "shuffled_residuals": sh,
    }


def twin_difference(o_hat, observed, pres, pairs):
    """Decoded vs true twin difference, in the scene's outcome basis."""
    rows = []
    for h, v in pairs:
        dt = observed[v] - observed[h]
        dh = o_hat[v] - o_hat[h]
        B = physics.outcome_basis(pres[h])
        ct, ch = np.linalg.solve(B, dt), np.linalg.solve(B, dh)
        u = dt / np.linalg.norm(dt)
        rows.append({
            "cos": float(dh @ dt / (np.linalg.norm(dh) * np.linalg.norm(dt) + 1e-30)),
            "along": float(dh @ u / np.linalg.norm(dt)),
            "orth": float(np.linalg.norm(dh - (dh @ u) * u) / np.linalg.norm(dt)),
            "true_coef": ct, "hat_coef": ch,
            "drift_recovered": float(ch[2] / ct[2]),
            # impulse change, in units of the impulse the momentum change would need (|dp| = M |drift|)
            "impulse_change_rel": float(ch[0] / ((pres[h].m_a + pres[h].m_b) * abs(ct[2]))),
            "tangential_rel": float(ch[1] / ((pres[h].m_a + pres[h].m_b) * abs(ct[2]))),
            "drift_t_rel": float(ch[3] / abs(ct[2])),
        })
    true_pure_drift = float(max(max(abs(r["true_coef"][0]), abs(r["true_coef"][1]), abs(r["true_coef"][3]))
                                / abs(r["true_coef"][2]) for r in rows))

    def summ(key):
        vals = np.array([r[key] for r in rows])
        pt, lo, hi = bootstrap_ci(vals, np.median)
        return {"median": pt, "ci95": [lo, hi]}

    return {"n_pairs": len(rows), "true_difference_is_pure_normal_drift_worst": true_pure_drift,
            **{k: summ(k) for k in ("cos", "along", "orth", "drift_recovered", "impulse_change_rel",
                                     "tangential_rel", "drift_t_rel")}}


def readout_span_share(cs, site, pool, Y, fit_rows, pairs, seed=0, draws=200):
    """Share of ||h_v - h_h||^2 inside span(W_outcome) vs random rank-matched subspaces."""
    X, _ = cs.features(site, pool)
    model, alpha = fit_full(X, Y, cs.seed, fit_rows)
    W = model.weights(alpha)
    Q, _ = np.linalg.qr(W)
    Xs = (X - model.mu) * model.scale
    D = np.stack([Xs[v] - Xs[h] for h, v in pairs])
    tot = (D ** 2).sum(1)
    obs = float(np.median(((D @ Q) ** 2).sum(1) / tot))
    rng = np.random.default_rng(seed)
    live = np.flatnonzero(model.scale > 0)
    null = []
    for _ in range(draws):
        R = np.zeros((X.shape[1], Q.shape[1]))
        R[live] = rng.normal(size=(len(live), Q.shape[1]))
        Rq, _ = np.linalg.qr(R)
        null.append(np.median(((D @ Rq) ** 2).sum(1) / tot))
    null = np.array(null)
    return {"site": site, "pool": pool, "rank": int(Q.shape[1]), "share_median": obs,
            "null_mean": float(null.mean()), "null_p95": float(np.quantile(null, 0.95)),
            "ratio_to_null": float(obs / null.mean()), "p": float((1 + np.sum(null >= obs)) / (draws + 1))}


# --------------------------------------------------------------------------- #
# the two parts
# --------------------------------------------------------------------------- #


def part_a(trunc_root: Path, workers: int):
    rungs = sorted((d for d in trunc_root.glob("k*") if d.is_dir() and any(d.glob("*.npz"))),
                   key=lambda d: int(d.name[1:]))
    out, k20 = {}, None
    for d in rungs:
        k = int(d.name[1:])
        t0 = time.time()
        cs = load_clipset(d)
        pres, legal, observed, cell = clip_physics(cs)
        ctrl = geometry_control(pres, legal, observed, cs.violates)
        honest = ~cs.violates
        grid = run_grid("a", cs, legal, honest, TRUNC_POOLS, workers)
        r2s = {f"{p} @ {s}": r2(legal[honest], pr[honest]) for (s, p), pr in grid.items()}
        best = max(grid, key=lambda sp: r2(legal[honest], grid[sp][honest]))
        o_hat = grid[best]
        h_rows = np.flatnonzero(honest)
        pairs = twin_pairs(cs)
        X, _ = cs.features(best[0], best[1])
        twin_rel = float(np.max([np.linalg.norm(X[h] - X[v]) / np.linalg.norm(X[h]) for h, v in pairs]))
        rec = {
            "k": k, "pre_collision": k <= 20, "n": cs.n, "n_honest": int(honest.sum()), "geometry_control": ctrl,
            "best_cell": f"{best[1]} @ {best[0]}", "best_r2_honest": r2s[f"{best[1]} @ {best[0]}"],
            "r2_by_pool_best_site": {p: max(v for kk, v in r2s.items() if kk.startswith(p + " @")) for p in TRUNC_POOLS},
            "r2_depth_curve_best_pool": {s: r2(legal[honest], grid[(s, best[1])][honest]) for s in cs.sites},
            "decoded_geometry_honest": decoded_geometry(o_hat, pres, cs.restitution, h_rows),
            "references": references(o_hat, legal, pres, cs.restitution, h_rows),
            "twin_activation_max_rel_diff": twin_rel,
        }
        out[str(k)] = rec
        g, ref = rec["decoded_geometry_honest"], rec["references"]
        sh = ref["shuffled_residuals"]
        print(f"  k={k:2d} n={cs.n} best {rec['best_cell']:28s} R2 {rec['best_r2_honest']:.3f} "
              f"(ceiling w/o e {ref['ceiling_without_e']['r2']:.3f})  "
              f"on-line {g['on_line_energy_fraction_median']:.3f} [shuffled {sh['on_line_energy_fraction_median']:.3f}, "
              f"p {sh['on_line_energy_fraction_median_p_structured']:.3f}]  "
              f"tang {g['tangential_fraction_median']:.3f} [{sh['tangential_fraction_median']:.3f}]  "
              f"rel-drift {g['relative_drift_median']:.3f} [{sh['relative_drift_median']:.3f}]  "
              f"corr(e_impl,e) {g['corr_implied_e_true_e']:+.3f}  twin diff {twin_rel:.1e}  ({time.time() - t0:.0f}s)",
              flush=True)
        if k == 20:
            k20 = (cs.clip_ids, o_hat)
    return out, k20


def part_b(full_root: Path, k20, workers: int):
    cs = load_clipset(full_root)
    pres, legal, observed, cell = clip_physics(cs)
    ctrl = geometry_control(pres, legal, observed, cs.violates)
    honest = ~cs.violates
    pairs = twin_pairs(cs)
    print(f"  {cs.describe()}\n  {len(pairs)} twin pairs", flush=True)
    grid = run_grid("b", cs, observed, honest, FULL_POOLS, workers)
    best = max(grid, key=lambda sp: r2(observed[honest], grid[sp][honest]))
    o_hat = grid[best]
    y = cs.violates

    def membership(o):
        return np.array([physics.membership_score(o[i], pres[i]) for i in range(cs.n)])

    def detect(score):
        rec = {"auc": auc(score, y), "paired": sign_test(np.array([score[v] - score[h] for h, v in pairs]))}
        rec["by_cell"] = {}
        for c in np.unique(cell):
            m = cell == c
            pc = [(h, v) for h, v in pairs if cell[h] == c]
            rec["by_cell"][c] = {"auc": auc(score[m], y[m]),
                                 "paired": sign_test(np.array([score[v] - score[h] for h, v in pc]))}
        return rec

    # The legal line needs the masses. The rows above use the TRUE masses, so the readout
    # supplies only the outcome. The model-only version takes the masses from the model
    # too (decoded from A.all+B.all, the static pools; masses are never edited, so every
    # clip trains), and is the one that asks whether the representation is internally
    # consistent with conservation.
    log_m = np.log(np.array([[p.m_a, p.m_b] for p in pres]))
    mgrid = run_grid("m", cs, log_m, np.ones(cs.n, bool), ["A.all+B.all"], workers)
    msite = max(mgrid, key=lambda sp: r2(log_m, mgrid[sp]))
    m_hat = np.exp(mgrid[msite])
    pres_dec = [physics.PreState(float(m_hat[i, 0]), float(m_hat[i, 1]), p.v_a, p.v_b, p.n)
                for i, p in enumerate(pres)]

    def membership_dec(o):
        return np.array([physics.membership_score(o[i], pres_dec[i]) for i in range(cs.n)])

    s_hat = membership(o_hat)
    res = {
        "decoded_masses": {"cell": f"{msite[1]} @ {msite[0]}", "r2_log_masses": r2(log_m, mgrid[msite])},
        "membership_decoded_outcome_decoded_masses": detect(membership_dec(o_hat)),
        "membership_true_outcome_decoded_masses": detect(membership_dec(observed)),
        "n": cs.n, "n_pairs": len(pairs), "geometry_control": ctrl,
        "best_cell": f"{best[1]} @ {best[0]}", "best_r2_honest": r2(observed[honest], o_hat[honest]),
        "r2_violating_observed": r2(observed[y], o_hat[y]),
        "membership_true_outcome": detect(membership(observed)),
        "membership_decoded": detect(s_hat),
        "membership_median": {"honest": float(np.median(s_hat[honest])), "violating": float(np.median(s_hat[y]))},
        "drift_legal_cosine_median": float(np.median([physics.drift_legal_cosine(p.m_a, p.m_b) for p in pres])),
        "twin_difference": twin_difference(o_hat, observed, pres, pairs),
        "implied_e_shift_violating_minus_honest": bootstrap_ci(np.array(
            [physics.implied_restitution(o_hat[v], pres[v]) - physics.implied_restitution(o_hat[h], pres[h])
             for h, v in pairs]), np.median),
    }
    # depth curve at the chosen pool: R^2 (selection quantity) beside the detection it implies
    res["depth_curve_best_pool"] = {}
    for s in cs.sites:
        o = grid[(s, best[1])]
        sc = membership(o)
        res["depth_curve_best_pool"][s] = {"r2_honest": r2(observed[honest], o[honest]), "auc": auc(sc, y),
                                           "paired_fraction": sign_test(np.array([sc[v] - sc[h] for h, v in pairs]))["fraction"]}
    res["readout_span_share"] = readout_span_share(cs, best[0], best[1], observed, np.flatnonzero(honest), pairs)
    if k20 is not None:
        ids, o_pre = k20
        pos = {c: i for i, c in enumerate(ids)}
        o_pre_al = np.stack([o_pre[pos[c]] if c in pos else np.full(4, np.nan) for c in cs.clip_ids])
        gap = np.linalg.norm(o_hat - o_pre_al, axis=1) / np.linalg.norm(o_pre_al, axis=1)
        ok = np.isfinite(gap)
        res["expectation_gap_k20"] = {
            "auc": auc(gap[ok], y[ok]),
            "paired": sign_test(np.array([gap[v] - gap[h] for h, v in pairs if ok[h] and ok[v]])),
            "median_honest": float(np.median(gap[ok & honest])), "median_violating": float(np.median(gap[ok & y])),
        }
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--full", required=True, type=Path)
    p.add_argument("--trunc", required=True, type=Path)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    print(f"[{args.model}] PART A — valid subspace from pre-collision rungs ({args.trunc})", flush=True)
    a, k20 = part_a(args.trunc, args.workers)
    print(f"[{args.model}] PART B — violating deviation on full clips ({args.full})", flush=True)
    b = part_b(args.full, k20, args.workers)
    out = {"protocol": "momentum valid-collision subspace (pre-collision) + violating deviation",
           "model": args.model, "full": str(args.full), "trunc": str(args.trunc), "part_a": a, "part_b": b}
    cli.write_artifact(out, args.out, f"momentum_valid_subspace_{args.model}.json")
    m, t = b["membership_decoded"], b["twin_difference"]
    print(f"\n  best {b['best_cell']}  R2 honest {b['best_r2_honest']:.3f}  violating-observed {b['r2_violating_observed']:.3f}")
    print(f"  membership AUC {m['auc']:.3f}  paired {m['paired']['positive']}/{m['paired']['n']} "
          f"p={m['paired']['p_one_sided']:.2g}  (true-outcome ceiling {b['membership_true_outcome']['auc']:.3f})")
    md, mt = b["membership_decoded_outcome_decoded_masses"], b["membership_true_outcome_decoded_masses"]
    print(f"  model-only (decoded masses, R2 {b['decoded_masses']['r2_log_masses']:.3f} @ {b['decoded_masses']['cell']}): "
          f"AUC {md['auc']:.3f} paired {md['paired']['positive']}/{md['paired']['n']} p={md['paired']['p_one_sided']:.2g}; "
          f"true outcome + decoded masses AUC {mt['auc']:.3f}")
    print(f"  twin diff: cos {t['cos']['median']:+.3f}  along {t['along']['median']:+.3f}  "
          f"drift recovered {t['drift_recovered']['median']:+.3f}  impulse change {t['impulse_change_rel']['median']:+.3f}")
    if "expectation_gap_k20" in b:
        g = b["expectation_gap_k20"]
        print(f"  expectation gap vs k=20: AUC {g['auc']:.3f}  paired {g['paired']['positive']}/{g['paired']['n']} "
              f"p={g['paired']['p_one_sided']:.2g}")
    r = b["readout_span_share"]
    print(f"  twin activation diff in outcome-readout span: {r['share_median']:.2e} vs null {r['null_mean']:.2e} "
          f"(x{r['ratio_to_null']:.1f}, p={r['p']:.3f})")


if __name__ == "__main__":
    main()
