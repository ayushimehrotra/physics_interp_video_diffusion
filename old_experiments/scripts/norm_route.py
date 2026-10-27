"""§34a's two-stage readout: regress the Delta-p VECTOR, take the norm outside.

WHY THIS EXISTS. `momentum_conserved` is three steps from what a linear probe
computes. A linear map emits weighted sums; the label needs

    Delta-p  = m_A dv_A + m_B dv_B     <- PRODUCTS of the ingredients
    |Delta-p|= sqrt(dpx^2 + dpy^2)     <- a NORM
    label    = |Delta-p| > 0           <- a threshold

Measured on v4: a linear classifier holding the EXACT masses and EXACT velocity
changes scores AUC 0.490 -- chance -- because it cannot multiply them. Handed the
exact Delta-p vector it still only reaches 0.533, because it cannot take a norm.
Allowed quadratic terms it reaches 0.893.

So the fix is not a bigger encoder; it is to stop asking the probe for the label.

    stage 1  ridge:  activations -> (dpx, dpy)      linear, and known to work
                                                    (R2 0.517, p=0.000 in §34)
    stage 2  numpy:  score = |prediction|           the nonlinearity, applied AFTER
    stage 3  AUC of that score against the label

Conserving scenes sit at Delta-p = (0,0); violating scenes lie on a ring around
it. No line separates a point from the ring enclosing it, but distance from the
origin does -- and that is all the norm is.

NULLS. Reported per cell and as a grid max. Taking the best of ~13-25 layers
inflates: `--n-perm` refits the whole pipeline on permuted labels and takes the
grid max each time, so the null is over the same selection the observed number
enjoyed. §34 used 500; the default here is 200 for cost.

    .venv/bin/python old_experiments/scripts/norm_route.py \
        --outputs outputs/v4_videomae-base outputs/v4_vjepa2-vitl \
        --dataset-dir datasets/probe_set_v4_legacy_fixed --pool all.post
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))  # noqa: E401,E702
import _momentum_path  # noqa: F401,E402

import joseph_intphys_probe as J  # noqa: E402
from physics_interp.interp import ridge_fit, _r2  # noqa: E402

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]


def delta_p_table(dataset_dir: str) -> dict:
    out = {}
    for f in sorted(glob.glob(f"{dataset_dir}/scene_*/labels.json")):
        d = json.load(open(f))
        pf = {p["frame"]: p for p in d["per_frame"]}
        g = lambda t, fr: np.array(pf[fr][t]["velocity"][:2], float)  # noqa: E731
        ma, mb = d["objects"]["A"]["mass"], d["objects"]["B"]["mass"]
        dp = ma * (g("A", 22) - g("A", 18)) + mb * (g("B", 22) - g("B", 18))
        out[os.path.basename(os.path.dirname(f))] = {
            "dp": dp, "conserved": float(d["momentum_conserved"])}
    return out


def norm_route(X, Ydp, y_bin, seed=0, train_fraction=0.6):
    """One cell. Returns (auc, r2_of_the_vector_regression).

    The split, the inner alpha selection and the standardisation are `fit_probe`'s,
    so the R2 reported here is comparable to every other ridge number in the repo.
    """
    n = len(X)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    tr, te = order[:int(round(train_fraction * n))], order[int(round(train_fraction * n)):]
    n_in = max(4, int(round(0.75 * len(tr))))
    itr, iva = tr[:n_in], tr[n_in:]
    if len(iva) < 2:
        itr, iva = tr, tr

    mu, sd = X[itr].mean(0), X[itr].std(0) + 1e-6
    scores = []
    for a in ALPHAS:
        W, b = ridge_fit((X[itr] - mu) / sd, Ydp[itr], a)
        scores.append(_r2(Ydp[iva], (X[iva] - mu) / sd @ W + b))
    alpha = float(ALPHAS[int(np.nanargmax(scores))])

    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    W, b = ridge_fit((X[tr] - mu) / sd, Ydp[tr], alpha)
    pred = (X[te] - mu) / sd @ W + b          # stage 1: the VECTOR
    score = np.linalg.norm(pred, axis=1)      # stage 2: the norm, outside the probe
    # violating is the positive class for detection
    return J.auc(score, y_bin[te] < 0.5), _r2(Ydp[te], pred)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", nargs="+", required=True)
    ap.add_argument("--dataset-dir", default="datasets/probe_set_v4_legacy_fixed")
    ap.add_argument("--pools", default="all.post,all.pre,all.all")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    table = delta_p_table(args.dataset_dir)
    pools = [p.strip() for p in args.pools.split(",") if p.strip()]
    report = {}

    for outdir in args.outputs:
        acts, labels, pool_names = J.load(outdir)
        ids = [l["scene_id"] for l in labels]
        Ydp = np.stack([table[i]["dp"] for i in ids])
        y = np.array([table[i]["conserved"] for i in ids])
        sites = sorted(acts, key=J.site_order)
        report[outdir] = {}
        print(f"\n{'#'*80}\n# {outdir}   n={len(y)}\n{'#'*80}")

        for pool in pools:
            print(f"\n  pool={pool}")
            print(f"  {'layer':<22}{'norm-route AUC':>16}{'Delta-p R2':>14}")
            rows = {}
            for s in sites:
                X = J.pool_matrix(acts[s], pool, pool_names).astype(np.float64)
                auc, r2 = norm_route(X, Ydp, y, args.seed)
                rows[s] = {"auc": auc, "r2": r2}
                print(f"  {s:<22}{auc:>16.3f}{r2:>14.3f}")
            best = max(rows, key=lambda k: rows[k]["auc"])
            obs = rows[best]["auc"]

            # grid-max null over the SAME layer selection the observed value got
            nulls = []
            for p in range(args.n_perm):
                perm = np.random.default_rng(1000 + p).permutation(len(y))
                nulls.append(max(
                    norm_route(J.pool_matrix(acts[s2], pool, pool_names).astype(np.float64),
                               Ydp[perm], y[perm], args.seed)[0] for s2 in sites))
            nulls = np.array(nulls)
            pval = float((nulls >= obs).mean())
            print(f"    best AUC {obs:.3f} at {best}   "
                  f"(Delta-p R2 {rows[best]['r2']:.3f})")
            print(f"    grid-max null over {args.n_perm} perms: mean {nulls.mean():.3f}, "
                  f"95th {np.percentile(nulls,95):.3f}  ->  p = {pval:.3f}")
            report[outdir][pool] = {"best_site": best, "auc": obs,
                                    "delta_p_r2": rows[best]["r2"],
                                    "null_mean": float(nulls.mean()),
                                    "null_p95": float(np.percentile(nulls, 95)),
                                    "p": pval, "layers": rows}

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=1, default=float)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
