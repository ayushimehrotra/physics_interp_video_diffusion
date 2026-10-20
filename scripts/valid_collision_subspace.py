#!/usr/bin/env python3
"""§47 — the subspace of valid collisions, before and after contact.

    python scripts/valid_collision_subspace.py --captures outputs/<set>_trunc/<model>/k*/activations

Each capture dir is one truncation rung (meta["truncate"] = frames the encoder
saw). At each rung the decoded outcome (dv_A, dv_B) is decomposed in the
{normal impulse, tangential impulse, common drift} basis of the scene's true
pre-contact state. "Direct" is the drift of the decoded outcome; "structured"
is the drift of a physics-respecting readout built from the same decoded
ingredients (decoded masses and a decoded scalar impulse, laid along the legal
direction, with its momentum change measured against the true masses). The
gap direct - structured is the prediction's own momentum violation.

Geometry control first: the true outcome must decompose to drift ~0,
tangential ~0, implied e == true e, or the script refuses to continue.

The target outcome is the honest physics reconstructed from the pre-contact
state at the clip's realised restitution; by default only honest clips are
used (pass --include-violating to mirror v1's protocol).
"""
import _momentum_path  # noqa: F401

from collections import defaultdict

import numpy as np

from physics_interp.momentum import cli, physics
from physics_interp.momentum.captures import load_clipset
from physics_interp.momentum.stats import bootstrap_ci, cv_predict


def geometry_control(cs):
    worst = {"drift": 0.0, "tangential": 0.0, "e": 0.0}
    for l in cs.labels:
        pre = physics.pre_state_from_labels(l)
        o = physics.legal_outcome(pre, physics.clip_restitution(l))
        d = physics.decompose(o, pre)
        worst["drift"] = max(worst["drift"], abs(d["drift_n"]), abs(d["drift_t"]))
        worst["tangential"] = max(worst["tangential"], abs(d["J_tangential"]) / abs(d["J_normal"]))
        worst["e"] = max(worst["e"], abs(physics.implied_restitution(o, pre) - physics.clip_restitution(l)))
    if max(worst.values()) > 1e-9:
        raise SystemExit(f"geometry control failed: {worst}")
    return worst


def main():
    p = cli.base_parser(__doc__, pools=False)
    p.add_argument("--pool", default="A.all+B.all")
    p.add_argument("--include-violating", action="store_true")
    args = p.parse_args()
    by_model = defaultdict(dict)
    for c in args.captures:
        cs = load_clipset(c, step=args.step, limit=args.limit)
        k = cs.metas[0].get("truncate")
        model = cs.metas[0].get("model", "model")
        if k is None:
            k = cs.metas[0].get("n_frames_seen", "full")
        by_model[model][k] = cs
    out = {"protocol": "§47", "models": {}}
    for model, rungs in by_model.items():
        print(f"\n[{model}] rungs {sorted(rungs)}")
        rec = {"rungs": {}}
        for k in sorted(rungs, key=lambda v: (isinstance(v, str), v)):
            cs = rungs[k]
            if not args.include_violating:
                cs = cs.subset(~cs.violates)
            ctrl = geometry_control(cs)
            pres = [physics.pre_state_from_labels(l) for l in cs.labels]
            e_true = cs.restitution
            target = np.stack([physics.legal_outcome(pre, e) for pre, e in zip(pres, e_true)])
            J_true = np.array([physics.decompose(o, pre)["J_normal"] for o, pre in zip(target, pres)])[:, None]
            log_m = np.stack([[np.log(pre.m_a), np.log(pre.m_b)] for pre in pres])
            site = cs.sites[-1] if args.site in ("last", "ingredients") else (cs.sites[0] if args.site == "embed" else args.site)
            X, valid = cs.features(site, args.pool)
            idx = np.flatnonzero(valid)
            o_hat = np.full_like(target, np.nan)
            o_hat[idx] = cv_predict(X[idx], target[idx], cs.seed[idx], seed=args.seed)
            J_hat = np.full_like(J_true, np.nan)
            J_hat[idx] = cv_predict(X[idx], J_true[idx], cs.seed[idx], seed=args.seed)
            m_hat = np.full_like(log_m, np.nan)
            m_hat[idx] = cv_predict(X[idx], log_m[idx], cs.seed[idx], seed=args.seed)
            direct, structured, tang, e_imp = [], [], [], []
            for i in idx:
                pre = pres[i]
                d = physics.decompose(o_hat[i], pre)
                direct.append(d["relative_drift"]); tang.append(d["tangential_fraction"])
                e_imp.append(physics.implied_restitution(o_hat[i], pre))
                pre_hat = physics.PreState(float(np.exp(m_hat[i, 0])), float(np.exp(m_hat[i, 1])), pre.v_a, pre.v_b, pre.n)
                o_struct = J_hat[i, 0] * physics.legal_direction(pre_hat)
                structured.append(physics.decompose(o_struct, pre)["relative_drift"])
            direct, structured = np.array(direct), np.array(structured)
            gap, lo, hi = bootstrap_ci(direct - structured, np.median, seed=args.seed)
            row = {
                "n": int(len(idx)), "site": site, "geometry_control_worst": ctrl,
                "relative_drift_direct_median": float(np.median(direct)),
                "relative_drift_structured_median": float(np.median(structured)),
                "gap_median": gap, "gap_ci95": [lo, hi],
                "tangential_fraction_median": float(np.median(tang)),
                "corr_implied_e_true_e": float(np.corrcoef(e_imp, e_true[idx])[0, 1]) if np.std(e_true[idx]) > 0 else float("nan"),
            }
            rec["rungs"][str(k)] = row
            print(f"  k={str(k):>4s} n={row['n']:3d} @ {site}: drift direct {row['relative_drift_direct_median']:.3f} "
                  f"structured {row['relative_drift_structured_median']:.3f} gap {gap:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
                  f"tangential {row['tangential_fraction_median']:.2f}  corr(e_impl, e) {row['corr_implied_e_true_e']:+.3f}")
        out["models"][model] = rec
    cli.write_artifact(out, args.out, "valid_collision_subspace.json")


if __name__ == "__main__":
    main()
