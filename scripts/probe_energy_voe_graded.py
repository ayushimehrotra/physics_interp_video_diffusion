"""§34's INSTRUMENT applied to `energy_voe_v1`: regress signed dKE, threshold OUTSIDE.

§34a's rule: ask whether the target is a linear functional of what the model
represents. `momentum_conserved` is a threshold on a NORM and a linear readout handed
the exact Delta-p scores 0.559 -- the binary label was the wrong instrument, and
regressing the vector then taking |prediction| outside the probe recovered 0.633 ->
0.765.

`energy_created` is the MIRROR case. It is one-sided, so the exact signed dKE gives
1.000 and taking |dKE| DESTROYS it (0.634). The nonlinearity to put outside is
therefore a threshold, not a norm -- but the graded route can still beat the binary
one for the §34a reason that survives: a binary label discards the ranking, and ridge
fit against a continuous target uses every clip's distance from the boundary.

Scored against the SAME impossible-vs-legal label and the SAME exact paired null as
`probe_energy_voe.py`, so the two instruments are directly comparable.

MASS-FREE CEILING STILL APPLIES (§46c). dKE is mass-free on this set by 44c-bis, so a
graded probe that beats the binary one has still only read kinematics better.
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_energy import POOLS, auc, cell_oof_regression  # noqa: E402
from probe_energy_voe import load, facts, paired_permuted_labels  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", nargs="+", required=True)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", default="docs/artifacts/energy_voe_graded.json")
    args = ap.parse_args()

    report = {"pools": POOLS, "instrument": "regress signed dKE, rank outside"}
    for outdir in args.outputs:
        acts, labels, pools = load(outdir)
        F = [facts(L) for L in labels]
        L2 = {json.loads(bytes(np.load(p, allow_pickle=True)["meta"]).decode())["scene_id"]:
              json.loads(bytes(np.load(p, allow_pickle=True)["meta"]).decode())["labels"]
              for p in sorted(glob.glob(f"{outdir}/momentum_probe/activations/*.npz"))}
        cells = {f"{s}/{p}": acts[s][:, pools.index(p), :] for s in acts for p in POOLS}
        seeds = np.array([f["seed"] for f in F])
        cellv = np.array([f["cell"] for f in F])
        # signed dKE, the continuous target. KE_post - KE_pre from the labels.
        dke = np.array([-lab["collision"]["kinetic_energy_lost"] for lab in labels])

        sel = cellv != "anchor"
        y = (cellv[sel] == "impossible")
        g = seeds[sel]
        print(f"\n{'='*80}\n{outdir}   n={int(sel.sum())}, {len(cells)} cells")
        print(f"  ORACLE  exact signed dKE -> label   AUC "
              f"{max(auc(y, dke[sel]), 1-auc(y, dke[sel])):.3f}   (one-sided, linear)")
        print(f"  ORACLE  exact |dKE|      -> label   AUC "
              f"{max(auc(y, np.abs(dke[sel])), 1-auc(y, np.abs(dke[sel]))):.3f}   "
              f"(the norm destroys it -- mirror of §34)")

        Ys = paired_permuted_labels(y, g, args.n_perm, args.seed)
        grid_obs, grid_null = {}, {}
        for cname, X in cells.items():
            # regress the CONTINUOUS signed dKE; the label never enters the fit
            # `cell_oof_regression` takes a COLUMN target -- `RidgeSplit.predict`
            # documents `Y [n_train, P]`, and a 1-D y broadcasts to [n_train, n_train]
            # inside the eigen-solve rather than erroring on shape.
            oof = cell_oof_regression(X[sel], dke[sel].reshape(-1, 1),
                                      g, args.seed).ravel()
            a = auc(y, oof)
            grid_obs[cname] = max(a, 1 - a)
            # null: the same OOF prediction ranked against permuted labels
            grid_null[cname] = np.array([max(auc(Ys[:, c] > .5, oof),
                                             1 - auc(Ys[:, c] > .5, oof))
                                         for c in range(1, Ys.shape[1])])
        names = list(cells)
        obs = np.array([grid_obs[c] for c in names])
        null = np.stack([grid_null[c] for c in names])
        nullmax = null.max(axis=0)
        b = int(np.argmax(obs))
        p_corr = float((nullmax >= obs[b]).mean())
        print(f"  R2 route best  AUC {obs[b]:.3f}  at {names[b]}   "
              f"null p95 {np.percentile(nullmax, 95):.3f}   p_corrected {p_corr:.3f}")

        # per-pool, to sit beside probe_energy_voe.py's table
        per_pool = {}
        for p in POOLS:
            idx = [i for i, c in enumerate(names) if c.endswith("/" + p)]
            j = idx[int(np.argmax(obs[idx]))]
            per_pool[p] = {"cell": names[j], "auc": float(obs[j])}
            print(f"    {p:9s} {obs[j]:.3f}   {names[j].split('/')[0]}")
        report[outdir] = {"grid_best": {"cell": names[b], "auc": float(obs[b]),
                                        "p_corrected": p_corr},
                          "per_pool": per_pool,
                          "oracle_signed_dke": float(max(auc(y, dke[sel]),
                                                         1 - auc(y, dke[sel])))}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
