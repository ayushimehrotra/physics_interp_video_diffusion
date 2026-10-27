"""All THREE pairings of `energy_voe_pilot`, not just the two `probe_energy_voe.py` uses.

That script sets `sel = cellv != "anchor"` and so only ever scores impossible (`e=k`,
energy CREATED) against legal (`e=1/k`, energy LOST). The anchor cell -- `e=1` exactly,
a bit-exact no-op, energy CONSERVED -- is dropped. So the contrast this set makes
uniquely clean has never been probed:

    CREATED vs CONSERVED, with momentum held to 3.9e-16 in BOTH arms.

§44c's `energy_created` is the confounded version: on `conservation_set_v1` all 57
energy-creating clips break momentum too, so nothing there separates "created energy"
from "broke momentum". Here they are separated by construction.

Three pairings, same grid, same grouped folds, same exact paired null:

    impossible vs legal    created vs lost       (what §46d already reported)
    impossible vs anchor   created vs CONSERVED  (the missing cell)
    legal      vs anchor   lost vs conserved     (control: both LEGAL physics)

The mass-free ceiling is printed for each and is expected to be 1.000 in all three --
`e` separates every pairing perfectly, §46c. The number that carries information is
how the three AUCs ORDER, which says what the representation is tracking.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import sys
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))  # noqa: E401,E702
import _momentum_path  # noqa: F401,E402
from probe_energy import POOLS, auc, auc_cols, cell_oof_scores  # noqa: E402
from probe_energy_voe import load, facts, paired_permuted_labels  # noqa: E402

PAIRINGS = [("impossible", "legal",   "created vs lost      "),
            ("impossible", "anchor",  "created vs CONSERVED "),
            ("legal",      "anchor",  "lost vs conserved    ")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", nargs="+", required=True)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", default="docs/artifacts/energy_voe_pairs.json")
    args = ap.parse_args()

    report = {"pools": POOLS, "pairings": {}}
    for outdir in args.outputs:
        acts, labels, pools = load(outdir)
        F = [facts(L) for L in labels]
        cells = {f"{s}/{p}": acts[s][:, pools.index(p), :] for s in acts for p in POOLS}
        names = list(cells)
        seeds = np.array([f["seed"] for f in F])
        cellv = np.array([f["cell"] for f in F])
        print(f"\n{'=' * 82}\n{outdir}   {len(F)} clips, {len(cells)} cells")

        report["pairings"].setdefault(outdir, {})
        for pos, neg, label in PAIRINGS:
            sel = (cellv == pos) | (cellv == neg)
            y = (cellv[sel] == pos)
            g = seeds[sel]
            Ys = paired_permuted_labels(y, g, args.n_perm, args.seed)

            orc = {}
            for nm in ("e_meas", "rel_post", "speed_sum_post"):
                v = np.array([f[nm] for f in F])[sel]
                a = auc(y, v)
                orc[nm] = max(a, 1 - a)
            mf = max(orc.values())

            obs, null = [], []
            for cname in names:
                X = cells[cname]
                oof = cell_oof_scores(X[sel], Ys, g, args.seed)
                a = auc_cols(np.repeat(y[:, None], Ys.shape[1], axis=1), oof)
                obs.append(a[0]); null.append(a[1:])
            obs = np.array(obs); null = np.stack(null)
            nullmax = null.max(axis=0)
            b = int(np.argmax(obs))
            p_corr = float((nullmax >= obs[b]).mean())

            print(f"  {label} n={int(sel.sum()):3d}  "
                  f"grid best {obs[b]:.3f}  p_corr {p_corr:.3f}  "
                  f"null p95 {np.percentile(nullmax, 95):.3f}  "
                  f"ceiling {mf:.3f}   {names[b]}")
            report["pairings"][outdir][f"{pos}_vs_{neg}"] = {
                "n": int(sel.sum()), "auc": float(obs[b]), "cell": names[b],
                "p_corrected": p_corr, "mass_free_ceiling": mf, "oracles": orc,
                "null_p95": float(np.percentile(nullmax, 95)),
                # per-cell, so a depth curve can be drawn without refitting
                "full_grid": {c: float(v) for c, v in zip(names, obs)},
                "null_p95_per_cell": {c: float(np.percentile(n_, 95))
                                      for c, n_ in zip(names, null)}}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
