"""Validation gates for `conservation_set_v1`.

Nine checks. The first four are physics; the rest are the LEAK AUDIT, promoted from
a post-hoc analysis to a gate, because every false positive in this project so far
came from a channel nobody checked before rendering.

Why these particular scalars. Under the e-preserving edit the twins differ ONLY in
how the fixed deflection sum a + b is split between the two balls -- verified: the
relative velocity is bit-identical (2.2e-16) and the edit is a common drift Dp/M.
So the whole leak surface is the implied mass ratio b/a, and the construction
matches its distribution exactly by permuting donors. These checks confirm that
survived contact with the renderer.

CHECK 8 IS THE ONE THAT BITES. A joint classifier over the full kinematic
observation must be scored with GROUPED folds. Ungrouped, a scene's honest twin
lands in train while its violating twin is in test, the classifier memorises the
honest deflection split per scene, and it reads AUC 0.213 -- a leak the data does
not contain. Grouped, the same classifier reads 0.498.

Grouping by seed is not enough on its own: a violating clip borrows its implied
ratio from a DONOR seed's honest twin, so the same value sits as "honest" in one
seed and "violating" in another. With the donor in train and the recipient in test
the classifier learns the inverted label, and the AUC goes systematically BELOW 0.5
(conservation_set_1000, inelastic/same: 0.384 on a set with no signal). So training
also drops every seed donor-linked to the test fold; that reads 0.497.

    python momentum_dataset/validate_conservation_set.py \
        --dataset-dir momentum_dataset/conservation_set_v1
"""
from __future__ import annotations
import argparse, glob, json
from collections import defaultdict
from pathlib import Path
import numpy as np

LEAK_TOL = 0.58          # a mass-free scalar above this fails; null sd ~0.06 at n=60
JOINT_TOL = 0.60


def auc(v, c):
    """AUC with machine-noise treated as ties.

    Several of the audited scalars are invariant BY CONSTRUCTION -- realised
    restitution, the deflection sum -- and differ only in the last bits of a
    float64. Ranking those bits produces an arbitrary AUC from a quantity the
    design pins exactly, which reads as a leak. Quantise to 1e-9 of the pooled
    span first, and report an exactly-constant scalar as 0.5 rather than noise.
    """
    v, c = np.asarray(v, float), np.asarray(c, float)
    if len(v) == 0 or len(c) == 0:
        return np.nan
    allv = np.concatenate([v, c])
    span = float(allv.max() - allv.min())
    scale = max(abs(float(allv.mean())), 1.0)
    if span <= 1e-9 * scale:
        return 0.5                      # constant to machine precision: no information
    q = span * 1e-9
    v, c = np.round(v / q), np.round(c / q)
    return float(((v[:, None] > c[None, :]).sum()
                  + 0.5 * (v[:, None] == c[None, :]).sum()) / (len(v) * len(c)))


def read(d):
    L = json.load(open(d / "labels.json"))
    pf = {p["frame"]: p for p in L["per_frame"]}
    v = lambda t, fr: np.array(pf[fr][t]["velocity"][:2], float)
    x = lambda t, fr: np.array(pf[fr][t]["position"][:2], float)
    n = np.array(L["collision"]["contact_normal"][:2], float); n /= np.linalg.norm(n)
    uA, uB = v("A", 18), v("B", 18)
    un = (uA - uB) @ n
    if un < 0:
        n, un = -n, -un                      # orient along the approach; else logs go NaN
    dvA, dvB = v("A", 22) - uA, v("B", 22) - uB
    a, b = np.linalg.norm(dvA), np.linalg.norm(dvB)
    vA, vB = v("A", 22), v("B", 22)
    mA, mB = L["objects"]["A"]["mass"], L["objects"]["B"]["mass"]
    half = np.array(L["scene_constants"]["visible_half_extents"])
    dt = (48 - 20) / L["scene_constants"]["frame_rate"]
    nrm = np.linalg.norm
    return dict(
        name=d.name, seed=L["seed"], e=L["restitution"], viol=L["momentum_conserved"] < 0.5,
        mats=(L["objects"]["A"]["material"], L["objects"]["B"]["material"]),
        rel=L["collision"]["momentum_rel_error"], ke=L["collision"]["kinetic_energy_ratio"],
        cos=float(dvA @ dvB / (a * b)),
        e_real=float(-((vA - vB) @ n) / un),
        a=a, b=b, S=a + b, implied=b / a, true_rho=mA / mB,
        speedA=float(nrm(vA)), speedB=float(nrm(vB)),
        dspeedA=float(nrm(vA) - nrm(uA)), dspeedB=float(nrm(vB) - nrm(uB)),
        relspeed=float(nrm(vA - vB)),
        centroid=float(nrm((vA + vB) / 2 - (uA + uB) / 2)),
        both_faster=float(nrm(vA) > nrm(uA) and nrm(vB) > nrm(uB)),
        exits=float((np.abs(x("A", 20) + vA * dt) > half).any()
                    or (np.abs(x("B", 20) + vB * dt) > half).any()),
        feats=[*uA, *uB, *n, un, a, b, np.log(a), np.log(b), np.log(b / a),
               nrm(vA), nrm(vB), nrm(vA) - nrm(uA), nrm(vB) - nrm(uB), nrm(vA - vB)],
    )


def grouped_joint_auc(rows, donor, seed=0):
    """Linear logistic on the full kinematic observation, 5-fold GROUPED by seed, with
    seeds donor-linked to the test fold purged from training. `donor`: seed -> donor seed."""
    X = np.array([r["feats"] for r in rows]); y = np.array([r["viol"] for r in rows], float)
    groups = np.array([r["seed"] for r in rows])
    uniq = np.unique(groups); rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(uniq), 5)
    oof = np.zeros(len(y))
    for g in folds:
        fs = set(g.tolist())
        linked = fs | {donor[s] for s in fs} | {s for s in uniq if donor[s] in fs}
        te = np.isin(groups, g); tr = ~np.isin(groups, list(linked))
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
        w = np.zeros(Z.shape[1]); R = 1e-4 * np.eye(Z.shape[1])
        for _ in range(60):
            p = 1 / (1 + np.exp(-Z[tr] @ w)); W = np.clip(p * (1 - p), 1e-6, None)
            w = w + np.linalg.solve(Z[tr].T @ (Z[tr] * W[:, None]) + R,
                                    Z[tr].T @ (y[tr] - p) - R @ w)
        oof[te] = Z[te] @ w
    return auc(oof[y > 0.5], oof[y < 0.5])


MASS_FREE = ["a", "b", "implied", "S", "e_real", "speedA", "speedB",
             "dspeedA", "dspeedB", "relspeed", "centroid", "both_faster", "exits"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", default="momentum_dataset/conservation_set_v1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.dataset_dir)
    dirs = sorted(d for d in root.glob("*_seed*") if (d / "_COMPLETE").exists())
    rows = [read(d) for d in dirs]
    print(f"{len(rows)} completed clips in {root}\n")
    fails, report = [], {}

    def gate(ok, label, detail):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:44s} {detail}")
        report[label] = {"pass": bool(ok), "detail": detail}
        if not ok:
            fails.append(label)

    # 1-4: physics
    plan = json.load(open(root / "plan.json"))
    seen = {r["name"] for r in rows}
    gate(len(rows) == plan["n_clips"], "1 every planned clip rendered",
         f"{len(rows)}/{plan['n_clips']}")
    bys = defaultdict(list)
    for r in rows:
        bys[(r["seed"], r["e"])].append(r)
    orphan = [k for k, v in bys.items() if len(v) != 2]
    gate(not orphan, "2 both twins present for every seed", f"{len(orphan)} orphaned")
    gate(max(abs(r["cos"] + 1) for r in rows) < 1e-9, "3 deflections exactly antiparallel",
         f"max |cos+1| = {max(abs(r['cos'] + 1) for r in rows):.2e}")
    gate(max(abs(r["e_real"] - r["e"]) for r in rows) < 1e-6,
         "4 realised restitution == pinned e",
         f"max dev = {max(abs(r['e_real'] - r['e']) for r in rows):.2e}")
    h = [r for r in rows if not r["viol"]]; v = [r for r in rows if r["viol"]]
    gate(max(r["rel"] for r in h) < 1e-9 and min(r["rel"] for r in v) > 0.05,
         "5 honest conserve, violating do not",
         f"honest max {max(r['rel'] for r in h):.1e}, violating min {min(r['rel'] for r in v):.3f}")

    # 6-8: the leak audit, per cell
    worst = 0.0; worst_name = ""; worst_n = (0, 0)
    for cell in sorted({(r["e"], r["mats"][0] != r["mats"][1]) for r in rows}):
        e, mixed = cell
        sub = [r for r in rows if r["e"] == e and (r["mats"][0] != r["mats"][1]) == mixed]
        hv = [r for r in sub if r["viol"]]; hh = [r for r in sub if not r["viol"]]
        if not hv or not hh:
            continue
        tag = f"{'elastic' if e > 0.5 else 'inelastic'}/{'mixed' if mixed else 'same'}"
        for k in MASS_FREE:
            a_ = auc([r[k] for r in hv], [r[k] for r in hh]); a_ = max(a_, 1 - a_)
            if a_ > worst:
                worst, worst_name, worst_n = a_, f"{k} in {tag}", (len(hv), len(hh))
        j = grouped_joint_auc(sub, {c["seed"]: c["donor_seed"] for c in plan["clips"]
                                    if c["violate"] and c["restitution"] == e})
        gate(abs(j - 0.5) <= JOINT_TOL - 0.5, f"6 joint mass-free AUC, {tag}", f"{j:.3f}")
    nv, nh = worst_n
    null_sd = np.sqrt((nv + nh + 1) / (12 * max(nv * nh, 1)))
    gate(worst <= LEAK_TOL, "7 worst single mass-free scalar",
         f"{worst:.3f} ({worst_name}); null sd at n={nv}/{nh} is {null_sd:.3f}")
    ke_a = auc([r["ke"] for r in v], [r["ke"] for r in h])
    gate(abs(ke_a - 0.5) < 0.12, "8 KE ratio (competing law) near chance",
         f"AUC {max(ke_a, 1 - ke_a):.3f}")
    ex = auc([r["exits"] for r in v], [r["exits"] for r in h])
    gate(abs(ex - 0.5) < 0.05, "9 frame-exit balanced across classes",
         f"AUC {max(ex, 1 - ex):.3f}")

    print(f"\n{'ALL GATES PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
