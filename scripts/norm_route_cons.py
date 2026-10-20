#!/usr/bin/env python
"""§34a's norm route on `conservation_set_v1`, split ELASTIC vs INELASTIC.

    stage 1  ridge:  activations -> (dpx, dpy)     linear
    stage 2  numpy:  score = |prediction|          the norm, applied OUTSIDE
    stage 3  AUC of that score against honest / violating

WHY THIS SET rather than v1 or v4. `probe_set` (v1) hands the classes away through
`cos(dv_A,dv_B)` at AUC 1.000 (§38); `probe_set_v4_legacy_fixed` closes that and opens a
smaller one through super-elastic `e` (12/75, §38b). `conservation_set_v1` is the
leak-free 2x2 -- all nine §42 gates pass, worst mass-free scalar 0.561 against a 0.053
null sd -- and it is the only momentum set that carries an elastic / inelastic split.

TWO PROTOCOL CHANGES FROM `norm_route.py`, both required here:

* **Folds are GROUPED BY SEED.** Honest and violating twins share a bit-identical
  pre-contact segment, so a random split lets the probe recognise a scene's own twin.
  §42b measured that trap at 0.787 discrimination on leak-free data. `norm_route.py`
  uses a plain 60/40 split, which is safe on v1/v4 (no twins) and is NOT safe here.
* **The null is over the FULL grid**, all pools x all sites, not sites within one pool.
  §34 corrected over the whole cell grid; `norm_route.py` corrects within a pool and is
  therefore less conservative. Numbers here are directly comparable to §34's.

All nine pools are swept and the best is reported PER MODEL PER STRATUM, with the
grid-max null covering that same selection.

    .venv/bin/python scripts/norm_route_cons.py
"""
from __future__ import annotations
import argparse, glob, json, re
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probe_energy import POOLS, ALPHAS, RidgeSplit, grouped_folds, auc, r2  # noqa: E402


def load(outdir: str):
    acts, labels, pools = {}, [], None
    for f in sorted(glob.glob(f"{outdir}/momentum_probe/activations/*.npz")):
        z = np.load(f, allow_pickle=True)
        m = json.loads(bytes(z["meta"]).decode())
        pools = m["pools"]
        labels.append(m["labels"])
        for k in z.files:
            if k.startswith("act."):
                acts.setdefault(k[4:], []).append(np.asarray(z[k][0], np.float64))
    return {s: np.stack(v) for s, v in acts.items()}, labels, pools


def physics(dataset_dir: str) -> dict:
    """scene_id -> (Delta-p 2-vector, violating flag, elastic flag, seed)."""
    out = {}
    for f in sorted(glob.glob(f"{dataset_dir}/*_seed*/labels.json")):
        d = json.load(open(f))
        name = Path(f).parent.name
        pf = {p["frame"]: p for p in d["per_frame"]}
        g = lambda t, fr: np.array(pf[fr][t]["velocity"][:2], float)   # noqa: E731
        ma, mb = d["objects"]["A"]["mass"], d["objects"]["B"]["mass"]
        out[name] = dict(
            dp=ma * (g("A", 22) - g("A", 18)) + mb * (g("B", 22) - g("B", 18)),
            viol=bool(d["violates_momentum"]),
            elastic=name.startswith("elastic"),
            seed=int(re.search(r"seed(\d+)$", name).group(1)))
    return out


def site_depth(site: str) -> int:
    return 0 if site.startswith("embed") else int(re.search(r"block\.(\d+)", site).group(1)) + 1


def cell_scores(X, Ydp, groups, seed, n_out=5, n_in=3):
    """|predicted Delta-p| out-of-fold, for every permutation column at once.

    `Ydp` is `[n, P, 2]`: column 0 is the observed pairing, the rest are permuted.
    RidgeSplit eigendecomposes the training Gram ONCE -- it does not depend on the
    target -- so every alpha and every permutation is a matmul after that.
    """
    n, P, _ = Ydp.shape
    flat = Ydp.reshape(n, P * 2)
    rng = np.random.default_rng(seed)
    outer = grouped_folds(groups, n_out, rng)
    oof = np.zeros((n, P))
    for k in range(n_out):
        te = outer[k]
        tr = np.concatenate([outer[j] for j in range(n_out) if j != k])
        inner = grouped_folds(groups[tr], n_in, np.random.default_rng(seed + 1000 + k))
        val = {a: np.zeros(P) for a in ALPHAS}
        for f in range(n_in):
            iva = tr[inner[f]]
            itr = np.concatenate([tr[inner[j]] for j in range(n_in) if j != f])
            preds = RidgeSplit(X[itr], X[iva]).predict(flat[itr])
            truth = flat[iva].reshape(len(iva), P, 2)
            for a, Pv in preds.items():
                pv = Pv.reshape(len(iva), P, 2)
                ss_res = ((truth - pv) ** 2).sum(axis=(0, 2))
                ss_tot = ((truth - truth.mean(0)) ** 2).sum(axis=(0, 2))
                val[a] += np.where(ss_tot > 0, 1 - ss_res / np.maximum(ss_tot, 1e-30), 0.0)
        best = np.argmax(np.stack([val[a] for a in ALPHAS]), axis=0)          # [P]
        preds = RidgeSplit(X[tr], X[te]).predict(flat[tr])
        stack = np.stack([preds[a].reshape(len(te), P, 2) for a in ALPHAS])   # [A,nte,P,2]
        chosen = stack[best, :, np.arange(P)]                                 # [P,nte,2]
        oof[te] = np.linalg.norm(chosen, axis=2).T                            # stage 2
    return oof


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", nargs="+",
                    default=["outputs_cons_videomae-base", "outputs_cons_vjepa2-vitl"])
    ap.add_argument("--dataset-dir", default="momentum_dataset/conservation_set_v1")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", default="docs/artifacts/norm_route_cons.json")
    args = ap.parse_args()

    phys = physics(args.dataset_dir)
    report = {}
    for outdir in args.outputs:
        acts, labels, pool_names = load(outdir)
        ids = [l["scene_id"] for l in labels]
        keep = [i for i, s in enumerate(ids) if s in phys]
        info = [phys[ids[i]] for i in keep]
        report[outdir] = {}
        print(f"\n{'#' * 78}\n# {outdir}   {len(keep)} clips, "
              f"{len(acts)} sites x {len(POOLS)} pools\n{'#' * 78}")

        for stratum in ("elastic", "inelastic"):
            m = np.array([f["elastic"] == (stratum == "elastic") for f in info])
            idx = np.flatnonzero(m)
            Ydp0 = np.stack([info[i]["dp"] for i in idx])
            y = np.array([info[i]["viol"] for i in idx])          # violating = positive
            g = np.array([info[i]["seed"] for i in idx])

            rng = np.random.default_rng(args.seed)
            perms = [np.arange(len(idx))] + [rng.permutation(len(idx))
                                             for _ in range(args.n_perm)]
            Ydp = np.stack([Ydp0[p] for p in perms], axis=1)      # [n, P, 2]
            Ylab = np.stack([y[p] for p in perms], axis=1)        # [n, P]

            cells, obs, null = [], [], []
            for site in sorted(acts, key=site_depth):
                for pool in POOLS:
                    X = acts[site][:, pool_names.index(pool), :][idx]
                    sc = cell_scores(X, Ydp, g, args.seed)
                    a = np.array([auc(Ylab[:, c], sc[:, c]) for c in range(sc.shape[1])])
                    cells.append(f"{site}/{pool}"); obs.append(a[0]); null.append(a[1:])
            obs = np.array(obs); null = np.stack(null)
            nullmax = null.max(axis=0)
            b = int(np.argmax(obs))
            p = float((nullmax >= obs[b]).mean())
            best_pool = cells[b].split("/")[1]
            print(f"  {stratum:10s} n={len(idx):3d} ({int(y.sum())} violating)  "
                  f"BEST {obs[b]:.3f} at {cells[b]}   "
                  f"full-grid null p95 {np.percentile(nullmax, 95):.3f}  p = {p:.3f}")
            report[outdir][stratum] = {
                "n": int(len(idx)), "best_cell": cells[b], "best_pool": best_pool,
                "auc": float(obs[b]), "p_corrected": p,
                "null_p95": float(np.percentile(nullmax, 95)),
                "null_mean": float(nullmax.mean()),
                "cells": {c: float(v) for c, v in zip(cells, obs)}}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
