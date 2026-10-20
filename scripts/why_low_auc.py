"""Why is the conservation AUC ~0.6 and not Joseph et al.'s 0.85-0.95?

Four candidate explanations. This separates them, because they imply different
fixes and only one of them is fixed by a bigger encoder.

  A. THE PIPELINE / n IS THE LIMIT.  Test: run the identical protocol on a
     trivially VISIBLE binary (is B moving; is radius_A above median). If those
     do not reach ~0.9 at n=150, nothing about this setup can reach 0.9 and the
     conservation number is not a statement about conservation.

  B. THE DATA IS THE LIMIT.  Test: learning curve. Subsample the training half
     and see whether AUC is still climbing at n=150. If it is, the fix is scenes,
     not parameters.

  C. THE LABEL SHAPE IS THE LIMIT (§34a).  Test: hand a linear classifier the
     EXACT, noiseless quantities and see what it can do. If perfect information
     scores ~0.56, no encoder can beat that under this framing.

  D. THE ENCODER IS THE LIMIT.  This is the only one a larger model fixes, and
     it is only worth testing after A-C are ruled out.

Run:
    .venv/bin/python scripts/why_low_auc.py \
        --outputs outputs_v4_videomae-base outputs_v4_vjepa2-vitl \
        --dataset-dir momentum_dataset/probe_set_v4_legacy_fixed --pool all.post
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


def scene_table(dataset_dir: str) -> tuple[list[str], dict]:
    ids, rec = [], {}
    for f in sorted(glob.glob(f"{dataset_dir}/scene_*/labels.json")):
        d = json.load(open(f))
        pf = {p["frame"]: p for p in d["per_frame"]}
        g = lambda t, fr: np.array(pf[fr][t]["velocity"][:2], float)  # noqa: E731
        ma, mb = d["objects"]["A"]["mass"], d["objects"]["B"]["mass"]
        dvA, dvB = g("A", 22) - g("A", 18), g("B", 22) - g("B", 18)
        dp = ma * dvA + mb * dvB
        sid = os.path.basename(os.path.dirname(f))
        ids.append(sid)
        rec[sid] = {
            "conserved": float(d["momentum_conserved"]),
            "b_moving": float(d["b_moving"]),
            "radius_A": d["objects"]["A"]["radius"],
            "mass_A": ma, "mass_B": mb,
            "dp": dp, "dvA": dvA, "dvB": dvB,
            "abs_dp": float(np.linalg.norm(dp)),
        }
    return ids, rec


def bin_above_median(v: np.ndarray) -> np.ndarray:
    return (v > np.median(v)).astype(float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", nargs="+", required=True)
    ap.add_argument("--dataset-dir", default="momentum_dataset/probe_set_v4_legacy_fixed")
    ap.add_argument("--pool", default="all.post")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    import torch  # noqa: PLC0415
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ids, rec = scene_table(args.dataset_dir)
    report = {}

    # ---- C: perfect-information ceilings ---------------------------------- #
    # Same classifier, same folds, exact noiseless inputs. This bounds every
    # activation number below, and it needs no model at all.
    y = np.array([rec[i]["conserved"] for i in ids])
    ceilings = {
        "exact Delta-p vector (2d)":      np.stack([rec[i]["dp"] for i in ids]),
        "exact |Delta-p| (1d)":           np.array([[rec[i]["abs_dp"]] for i in ids]),
        "exact Delta-p + quadratic (5d)": np.stack([
            np.concatenate([rec[i]["dp"], rec[i]["dp"] ** 2,
                            [rec[i]["dp"][0] * rec[i]["dp"][1]]]) for i in ids]),
        "exact visible ingredients (6d)": np.stack([
            np.concatenate([[rec[i]["mass_A"], rec[i]["mass_B"]],
                            rec[i]["dvA"], rec[i]["dvB"]]) for i in ids]),
    }
    print(f"\n{'='*74}\nC. PERFECT-INFORMATION CEILINGS  (linear classifier, exact inputs)\n{'='*74}")
    report["ceilings"] = {}
    for name, X in ceilings.items():
        r = J.probe_layer(X.astype(np.float64), y, args.seed, dev)
        report["ceilings"][name] = r["auc_mean"]
        print(f"  {name:<34} acc {r['accuracy_mean']:.3f}   AUC {r['auc_mean']:.3f}")
    print("  -> a norm is not in the linear span; the 6d row is what a probe could")
    print("     reach if it read every visible ingredient perfectly.")

    # ---- A + B: positive controls and learning curve ---------------------- #
    for outdir in args.outputs:
        acts, labels, pool_names = J.load(outdir)
        sids = [l["scene_id"] for l in labels]
        sites = sorted(acts, key=J.site_order)
        targets = {
            "momentum_conserved (the target)": np.array([rec[i]["conserved"] for i in sids]),
            "b_moving (visible)":               np.array([rec[i]["b_moving"] for i in sids]),
            "radius_A > median (visible)":      bin_above_median(
                np.array([rec[i]["radius_A"] for i in sids])),
            "mass_A > median (visible)":        bin_above_median(
                np.array([rec[i]["mass_A"] for i in sids])),
        }
        print(f"\n{'='*74}\nA. POSITIVE CONTROLS — {outdir}  pool={args.pool}\n{'='*74}")
        report[outdir] = {}
        best_site = {}
        for tname, ty in targets.items():
            best = (-np.inf, None)
            for s in sites:
                X = J.pool_matrix(acts[s], args.pool, pool_names)
                r = J.probe_layer(X, ty, args.seed, dev)
                if r["auc_mean"] > best[0]:
                    best = (r["auc_mean"], s, r["accuracy_mean"])
            best_site[tname] = best[1]
            report[outdir][tname] = {"auc": best[0], "site": best[1]}
            print(f"  {tname:<34} best AUC {best[0]:.3f}  acc {best[2]:.3f}  @ {best[1]}")

        print(f"\n{'-'*74}\nB. LEARNING CURVE at each target's own best cell\n{'-'*74}")
        print(f"  {'target':<34}" + "".join(f"{f'n={n}':>10}" for n in (40, 70, 100, 150)))
        report[outdir]["learning_curve"] = {}
        for tname, ty in targets.items():
            X = J.pool_matrix(acts[best_site[tname]], args.pool, pool_names)
            row, curve = f"  {tname:<34}", {}
            for n in (40, 70, 100, 150):
                rng = np.random.default_rng(args.seed)
                # stratified subsample so every point keeps a 50/50 base rate
                keep = np.concatenate([
                    rng.permutation(np.flatnonzero(ty == c))[: n // 2] for c in (0.0, 1.0)])
                r = J.probe_layer(X[keep], ty[keep], args.seed, dev)
                curve[n] = r["auc_mean"]
                row += f"{r['auc_mean']:>10.3f}"
            report[outdir]["learning_curve"][tname] = curve
            print(row)
        print("  -> still climbing at n=150 means the fix is scenes, not parameters.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=1, default=float)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
