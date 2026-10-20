#!/usr/bin/env python
"""TABLE: raw mu probe vs mu composed from decoded (a, theta).

Three routes to mu from the same activations, all out-of-fold under the identical
protocol (`cell_oof_regression`, grouped folds, nested alpha, N fold seeds):

  DIRECT      activations -> mu                       one ridge
  COMPOSED    activations -> a, activations -> theta,
              then (a_hat, theta_hat) -> mu           a LEARNED linear second stage
  STRUCTURED  activations -> a, activations -> theta,
              then mu = tan(theta_hat) - a_hat / (g cos theta_hat)
                                                      the closed form, IMPOSED

Every stage is out-of-fold: the second stage is fit on out-of-fold first-stage
predictions, so no scene's own a_hat or theta_hat is used to fit the map that scores it.

Reference oracles on TRUE labels, same protocol (docs/artifacts/incline_mu_oracles.json):
  mu from a alone       0.402      mu from (a, theta)   0.997

§35b's caveat applies to the COMPOSED column and is the reason STRUCTURED is worth
reporting separately: mu is 0.997 linear in (a, theta) over the sampled theta range, so
a learned second stage has an almost exact linear target and direct-vs-composed becomes
an algebraic identity rather than a measurement. The closed form imposes the
nonlinearity instead of learning it -- §43e's rule.

    .venv/bin/python scripts/incline_composed_mu.py
"""
from __future__ import annotations
import argparse, glob, json, re
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_energy import cell_oof_regression, r2  # noqa: E402

MODELS = {"outputs_incline_videomae-base": ("VideoMAE-base", "A.pre"),
          "outputs_incline_vjepa2-vitl":   ("V-JEPA2-ViT-L", "all.pre")}


def site_depth(s): return 0 if s.startswith("embed") else int(re.search(r"block\.(\d+)", s).group(1)) + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="momentum_dataset/incline_set_v2")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("-o", "--out", default="docs/artifacts/incline_composed_mu.json")
    args = ap.parse_args()

    report = {"n_seeds": args.seeds}
    for outdir, (label, pool) in MODELS.items():
        paths = sorted(glob.glob(f"{outdir}/momentum_probe/activations/*.npz"))
        store, fr, pools, sites = {}, [], None, None
        for p in paths:
            z = np.load(p, allow_pickle=True)
            m = json.loads(bytes(z["meta"]).decode()); pools = m["pools"]
            fr.append(json.load(open(f"{args.dataset_dir}/{Path(p).stem}/labels.json"))["friction"])
            if sites is None:
                sites = [k[4:] for k in z.files if k.startswith("act.")]
            for s in sites:
                store.setdefault(s, []).append(z["act." + s][0])
        acts = {s: np.stack(v).astype(np.float64) for s, v in store.items()}
        n = len(fr); g = np.arange(n)
        gr = fr[0]["gravity"]
        mu = np.array([f["mu"] for f in fr])[:, None]
        a  = np.array([f["accel_measured"] for f in fr])[:, None]
        th = np.array([f["theta"] for f in fr])[:, None]

        print(f"\n{'=' * 74}\n{label}   pool={pool}   n={n}", flush=True)
        rows = {}
        for site in sorted(acts, key=site_depth):
            X = acts[site][:, pools.index(pool), :]
            d, c, st = [], [], []
            for s in range(args.seeds):
                d.append(r2(mu, cell_oof_regression(X, mu, g, s)))
                ah = cell_oof_regression(X, a,  g, s)
                th_h = cell_oof_regression(X, th, g, s)
                c.append(r2(mu, cell_oof_regression(np.hstack([ah, th_h]), mu, g, s)))
                st.append(r2(mu, np.tan(th_h) - ah / (gr * np.cos(th_h))))
            rows[site] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
                          for k, v in (("direct", d), ("composed", c), ("structured", st))}
            print(f"  {site:<24} direct {np.mean(d):+.3f}  composed {np.mean(c):+.3f}"
                  f"  structured {np.mean(st):+.3f}", flush=True)
        best = {k: max(rows, key=lambda s: rows[s][k]["mean"])
                for k in ("direct", "composed", "structured")}
        report[outdir] = {"label": label, "pool": pool, "n": n, "layers": rows,
                          "best": {k: {"site": best[k], **rows[best[k]][k]}
                                   for k in best}}
        for k in ("direct", "composed", "structured"):
            b = report[outdir]["best"][k]
            print(f"    BEST {k:11s} {b['mean']:+.3f} +- {b['sd']:.3f} at {b['site']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
