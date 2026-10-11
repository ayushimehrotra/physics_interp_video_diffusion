#!/usr/bin/env python3
"""§59 — violation of expectation against the whole legal family.

    python scripts/voe_ladder.py --surprise outputs/scene_ladder_v1/vjepa2/surprise.json

``--surprise`` is written by ``scripts/capture_encoder.py --surprise`` (V-JEPA2's
own predictor; nothing fitted): one record per clip with the clip's labels and
its prediction error under each metric and token set. For every future of a
seed, r = the fraction of that seed's other legal futures it is more surprising
than. Legal futures are the null (mean 0.5 by construction).

Reported: the raw AUC of r as a momentum detector (violating vs legal, all
clips) and as an energy detector (e > 1 arms vs legal); the per-seed
surprise-vs-e trend and its residual for the violating twin against its honest
twin at the same e (the direct question); the within-seed slope across the
physical boundary 1.0 -> 1.25 against the last legal step 0.8 -> 1.0.
"""
import _momentum_path  # noqa: F401

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from physics_interp.momentum import cli, physics
from physics_interp.momentum.stats import permutation_null


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--surprise", required=True, type=Path)
    p.add_argument("--draws", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-legal", type=int, default=3)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    table = json.loads(args.surprise.read_text())
    records = table["clips"] if isinstance(table, dict) else table
    metrics = sorted({m for r in records for m in r["surprise"]})
    by_seed = defaultdict(list)
    for r in records:
        by_seed[physics.clip_seed(r["labels"])].append(r)
    out = {"protocol": "§59", "metrics": {}, "n_seeds": len(by_seed)}
    for metric in metrics:
        rank, is_viol, is_imp, strata = [], [], [], []
        twin_diff, resid_diff, slope_ratio, skipped = [], [], [], 0
        for seed, rs in by_seed.items():
            e = np.array([physics.clip_restitution(r["labels"]) for r in rs])
            viol = np.array([bool(r["labels"].get("violates_momentum", False)) for r in rs])
            s = np.array([r["surprise"][metric] for r in rs])
            legal = (~viol) & (e <= 1 + 1e-9)
            if legal.sum() < args.min_legal:
                skipped += 1
                continue
            honest = [i for i, r in enumerate(rs) if physics.clip_arm(r["labels"]) == "honest"]
            for i in range(len(rs)):
                others = legal.copy(); others[i] = False
                rank.append(float((s[others] < s[i]).mean()))
                is_viol.append(viol[i]); is_imp.append(bool(e[i] > 1 + 1e-9))
                strata.append("elastic" if e[honest[0]] > 0.5 else "inelastic")
            # per-seed trend over legal arms, residual for the violating twin vs honest twin
            coef = np.polyfit(e[legal], s[legal], 1)
            vi = np.flatnonzero(viol)
            if len(vi) == 1 and honest:
                h, v = honest[0], vi[0]
                twin_diff.append(float(s[v] - s[h]))
                resid_diff.append(float((s[v] - np.polyval(coef, e[v])) - (s[h] - np.polyval(coef, e[h]))))
            # boundary slope
            get = lambda ev: s[np.isclose(e, ev, atol=1e-6) & ~viol]
            a, b, c = get(0.8), get(1.0), get(1.25)
            if len(a) and len(b) and len(c):
                last_legal = (b[0] - a[0]) / 0.2
                across = (c[0] - b[0]) / 0.25
                slope_ratio.append(across / last_legal if last_legal != 0 else np.nan)
        rank, is_viol, is_imp, strata = map(np.array, (rank, is_viol, is_imp, strata))
        legal_mask = ~is_viol & ~is_imp
        rec = {"skipped_seeds": skipped,
               "momentum_auc": permutation_null(rank[~is_imp], is_viol[~is_imp], draws=args.draws, seed=args.seed),
               "energy_auc": permutation_null(rank[~is_viol], is_imp[~is_viol], draws=args.draws, seed=args.seed),
               "legal_rank_mean": float(rank[legal_mask].mean()),
               "impossible_rank_mean": float(rank[is_imp].mean()) if is_imp.any() else float("nan"),
               "by_stratum": {}}
        for st in ("elastic", "inelastic"):
            m = (strata == st) & ~is_imp
            if m.sum() >= 8 and is_viol[m].any():
                rec["by_stratum"][st] = permutation_null(rank[m], is_viol[m], draws=args.draws, seed=args.seed)
        td, rd, sr = np.array(twin_diff), np.array(resid_diff), np.array(slope_ratio)
        rec["violating_minus_honest_twin"] = {"mean": float(td.mean()), "positive": f"{int((td > 0).sum())}/{len(td)}"}
        rec["residualised_twin_difference"] = {"mean": float(rd.mean()), "positive": f"{int((rd > 0).sum())}/{len(rd)}"}
        if len(sr):
            sr = sr[np.isfinite(sr)]
            rec["boundary_slope_ratio"] = {"median": float(np.median(sr)), "n": int(len(sr)),
                                           "frac_above_1": float((sr > 1).mean())}
        out["metrics"][metric] = rec
        print(f"\n[{metric}] momentum AUC {cli.fmt_auc(rec['momentum_auc'])}   energy AUC {cli.fmt_auc(rec['energy_auc'])}   "
              + "  ".join(f"{k} {cli.fmt_auc(v)}" for k, v in rec["by_stratum"].items()))
        print(f"  violating - honest twin: mean {td.mean():+.4f}, positive {rec['violating_minus_honest_twin']['positive']}; "
              f"residualised {rd.mean():+.4f}, positive {rec['residualised_twin_difference']['positive']}")
        if len(sr):
            print(f"  slope across e=1 boundary / last legal step: median {np.median(sr):.2f} (n={len(sr)}, >1 in {(sr > 1).mean():.2f})")
    cli.write_artifact(out, args.out, "voe_ladder.json")


if __name__ == "__main__":
    main()
