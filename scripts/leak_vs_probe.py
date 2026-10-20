"""Is the binary probe winning THROUGH a mass-free leak, or through momentum?

A violating scene can be spotted two ways. The intended route needs both masses:
`m_A dv_A + m_B dv_B != 0`. A leak route needs none -- some purely kinematic
scalar that no conserving scene ever produces. v4's leak is restitution:
`e = |rel_post| / |rel_pre|` exceeds 1 in 12 of 75 violating scenes and in 0
conserving ones, so those 12 are certifiable with no notion of mass anywhere.

That split is the instrument. Out-of-fold predictions from the same protocol as
`joseph_intphys_probe.py`, then recall computed separately on the leaked and
unleaked violating scenes.

  recall(leaked) >> recall(unleaked)  ->  the probe found the leak
  recall(leaked) ~= recall(unleaked)  ->  whatever it found is not the leak

CONFOUND, AND IT IS NOT SMALL. The leaked scenes are the ones where the impulse
asymmetry bit hardest, so they also carry larger |dp|. High recall on them is
therefore ambiguous between "found the leak" and "big violations are easier".
`--by-magnitude` splits the UNLEAKED scenes into |dp| terciles as the control:
if recall rises with |dp| among scenes with no leak at all, magnitude explains
it and the leak split proves nothing on its own.

n is small -- 12 scenes over 5 folds is ~2-3 per held-out fold. Counts are
printed, not just rates.

    .venv/bin/python scripts/leak_vs_probe.py \
        --outputs outputs_v4_videomae-base \
        --dataset-dir momentum_dataset/probe_set_v4_legacy_fixed \
        --pool all.post
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import joseph_intphys_probe as J  # noqa: E402


def scene_facts(dataset_dir: str) -> dict:
    """Per-scene ground truth: class, the leak scalar, and the violation size."""
    out = {}
    for f in sorted(glob.glob(f"{dataset_dir}/scene_*/labels.json")):
        d = json.load(open(f))
        pf = {p["frame"]: p for p in d["per_frame"]}
        ma, mb = d["objects"]["A"]["mass"], d["objects"]["B"]["mass"]
        g = lambda t, fr: np.array(pf[fr][t]["velocity"][:2], float)  # noqa: E731
        p0 = ma * g("A", 18) + mb * g("B", 18)
        p1 = ma * g("A", 22) + mb * g("B", 22)
        dvA, dvB = g("A", 22) - g("A", 18), g("B", 22) - g("B", 18)
        nA, nB = np.linalg.norm(dvA), np.linalg.norm(dvB)
        out[os.path.basename(os.path.dirname(f))] = {
            "viol": bool(d["violates_momentum"]),
            # restitution_measured is the leak: >1 is physically impossible and
            # readable from the two trajectories alone.
            "e": float(d["collision"]["restitution_measured"]),
            "rel": float(np.linalg.norm(p1 - p0) / max(np.linalg.norm(p0), 1e-12)),
            "cos": float(dvA @ dvB / (nA * nB)) if nA > 1e-12 and nB > 1e-12 else np.nan,
        }
    return out


def oof_scores(X, y, seed, device):
    """Out-of-fold logit per scene. Mirrors `probe_layer` exactly, but keeps the
    per-scene predictions instead of collapsing them to a fold mean."""
    folds = J.stratified_folds(y, J.N_FOLDS, seed)
    scores = np.full(len(y), np.nan)
    for f in range(J.N_FOLDS):
        te = folds[f]
        tr = np.concatenate([folds[k] for k in range(J.N_FOLDS) if k != f])
        rng = np.random.default_rng(seed + f)
        perm = rng.permutation(len(tr))
        n_in = int(round(0.8 * len(tr)))
        itr, iva = tr[perm[:n_in]], tr[perm[n_in:]]
        Ztr, Zva = J.standardize(X[itr], X[itr], X[iva])
        best = (-np.inf, J.LEARNING_RATES[0], J.WEIGHT_DECAYS[0])
        for lr in J.LEARNING_RATES:
            for wd in J.WEIGHT_DECAYS:
                s = J.auc(J.train_logistic(Ztr, y[itr], Zva, lr, wd, seed, device),
                          y[iva] > 0.5)
                if np.isfinite(s) and s > best[0]:
                    best = (s, lr, wd)
        _, lr, wd = best
        Ztr2, Zte = J.standardize(X[tr], X[tr], X[te])
        scores[te] = J.train_logistic(Ztr2, y[tr], Zte, lr, wd, seed, device)
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", nargs="+", required=True)
    ap.add_argument("--dataset-dir", default="momentum_dataset/probe_set_v4_legacy_fixed")
    ap.add_argument("--pool", default="all.post")
    ap.add_argument("--site", default=None, help="default: the best-AUC layer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    import torch  # noqa: PLC0415
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    facts = scene_facts(args.dataset_dir)
    report = {}

    for outdir in args.outputs:
        acts, labels, pool_names = J.load(outdir)
        ids = [l["scene_id"] for l in labels]
        y = np.array([float(l["momentum_conserved"]) for l in labels])
        sites = sorted(acts, key=J.site_order)

        if args.site:
            site = args.site
        else:
            scored = {s: J.probe_layer(J.pool_matrix(acts[s], args.pool, pool_names),
                                       y, args.seed, device)["auc_mean"] for s in sites}
            site = max(scored, key=scored.get)

        X = J.pool_matrix(acts[site], args.pool, pool_names)
        s = oof_scores(X, y, args.seed, device)
        pred_viol = s < 0.0                     # logit is for the CONSERVING class
        truth_viol = y < 0.5

        e = np.array([facts[i]["e"] for i in ids])
        rel = np.array([facts[i]["rel"] for i in ids])
        leaked = truth_viol & (e > 1.0 + 1e-9)
        unleak = truth_viol & ~(e > 1.0 + 1e-9)

        print(f"\n{'='*76}\n{outdir}   pool={args.pool}  site={site}\n{'='*76}")
        print(f"  overall accuracy {float((pred_viol == truth_viol).mean()):.3f}   "
              f"AUC {J.auc(s, y > 0.5):.3f}")
        spec = float((~pred_viol[~truth_viol]).mean())
        print(f"  specificity (conserving called conserving) {spec:.3f}")
        for name, sel in (("LEAKED   (e>1)", leaked), ("UNLEAKED (e<=1)", unleak)):
            n = int(sel.sum())
            hit = int(pred_viol[sel].sum())
            print(f"  {name:<16} n={n:>3}  called violating {hit:>3}  "
                  f"recall {hit/max(n,1):.3f}   mean |dp|/|p| {rel[sel].mean():.3f}")

        # Control: does recall rise with |dp| among UNLEAKED scenes alone?
        idx = np.flatnonzero(unleak)
        order = idx[np.argsort(rel[idx])]
        k = max(1, len(order) // 3)
        print("  control -- unleaked scenes only, by violation magnitude:")
        for name, sel in (("small |dp|", order[:k]), ("mid |dp|", order[k:2*k]),
                          ("large |dp|", order[2*k:])):
            hit = int(pred_viol[sel].sum())
            print(f"    {name:<12} n={len(sel):>3}  recall {hit/max(len(sel),1):.3f}"
                  f"   |dp|/|p| {rel[sel].mean():.3f}")
        report[outdir] = {
            "site": site, "pool": args.pool,
            "accuracy": float((pred_viol == truth_viol).mean()),
            "auc": J.auc(s, y > 0.5),
            "recall_leaked": float(pred_viol[leaked].mean()) if leaked.any() else None,
            "recall_unleaked": float(pred_viol[unleak].mean()) if unleak.any() else None,
            "n_leaked": int(leaked.sum()), "n_unleaked": int(unleak.sum()),
        }

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
