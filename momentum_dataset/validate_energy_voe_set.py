"""Check `energy_voe_v1` is the set it claims to be, before anything is probed on it.

Six checks, and the first two are the ones the whole design rests on:

  1. MOMENTUM IS CONSERVED IN EVERY CLIP. If it is not, the set has reintroduced the
     exact confound it exists to remove and every result on it is uninterpretable.
  2. ENERGY IS CREATED ONLY IN THE `impossible` CELL, and by the amount the closed
     form predicts: `dKE = 1/2 mu (e^2 - 1) u_n^2`.
  3. Realised restitution equals its target, measured from the rendered velocities
     rather than trusted from the plan.
  4. Twins of a seed share their pre-contact segment exactly, which is what makes the
     k / 1/k comparison matched rather than between-scene.
  5. `|log e|` is equal within each pair, so the two members are equidistant from
     elastic and equally unusual as kinematics.
  6. Both balls stay in frame for the whole clip.

It also REPORTS, rather than fails on, the mass-free separability of the two classes:
`e` is visible, so a probe reads the label off separation speed and learns nothing --
that is §44 and it has not gone away. This set is for violation-of-expectation. The
number is printed so nobody mistakes the set for a probe set.

    python momentum_dataset/validate_energy_voe_set.py
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PRE_FRAME = 18


def auc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); sv = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="momentum_dataset/energy_voe_v1")
    args = ap.parse_args()

    paths = sorted(Path(args.root).glob("*/labels.json"))
    if not paths:
        raise SystemExit(f"no labels.json under {args.root}")
    L = [json.load(open(p)) for p in paths]
    print(f"{len(L)} clips under {args.root}\n")

    fails = []

    # ---- 1. momentum, the defining property -------------------------------------
    mom = np.array([x["collision"]["momentum_rel_error"] for x in L])
    print(f"1. momentum relative error   max {mom.max():.3e}   "
          f"{'OK' if mom.max() < 1e-9 else 'FAIL'}")
    if mom.max() >= 1e-9:
        fails.append(f"momentum not conserved in {(mom >= 1e-9).sum()} clips")

    # ---- 2/3. energy and restitution ---------------------------------------------
    rows = []
    for x in L:
        c = x["collision"]
        e_t = x["energy_edit"]["e_target"]
        mA, mB = x["objects"]["A"]["mass"], x["objects"]["B"]["mass"]
        vA = np.array(c["pre"]["velocity_A"], float); vB = np.array(c["pre"]["velocity_B"], float)
        dvA = np.array(c["post"]["velocity_A"], float) - vA
        n = dvA / max(np.linalg.norm(dvA), 1e-12)
        u = float((vA - vB) @ n)
        mu = mA * mB / (mA + mB)
        rows.append(dict(seed=x["seed"], e_t=e_t, cell=("impossible" if e_t > 1 else
                                                        "anchor" if e_t == 1 else "legal"),
                         e_meas=c["restitution_measured"],
                         dke=c["post"]["kinetic_energy"] - c["pre"]["kinetic_energy"],
                         dke_pred=0.5 * mu * (e_t ** 2 - 1) * u * u,
                         kr=c["kinetic_energy_ratio"], created=x["energy_created"]))
    e_err = max(abs(r["e_meas"] - r["e_t"]) for r in rows)
    print(f"2. realised e vs target      max err {e_err:.3e}   "
          f"{'OK' if e_err < 1e-6 else 'FAIL'}")
    if e_err >= 1e-6:
        fails.append("realised restitution does not match its target")

    dke_err = max(abs(r["dke"] - r["dke_pred"]) / max(abs(r["dke_pred"]), 1e-12)
                  for r in rows if r["e_t"] != 1.0)
    print(f"3. dKE vs ½·mu·(e²−1)·u²    max rel err {dke_err:.3e}   "
          f"{'OK' if dke_err < 1e-6 else 'FAIL'}")
    if dke_err >= 1e-6:
        fails.append("energy change does not follow the closed form")

    by_cell = defaultdict(list)
    for r in rows:
        by_cell[r["cell"]].append(r)
    print("\n   cell         n   KE ratio range        energy created?")
    for cell in ("anchor", "legal", "impossible"):
        v = by_cell.get(cell, [])
        if not v:
            continue
        kr = [x["kr"] for x in v]
        ncre = sum(x["kr"] > 1 + 1e-9 for x in v)
        print(f"   {cell:11s}{len(v):4d}   [{min(kr):.3f}, {max(kr):.3f}]   {ncre}/{len(v)}")
        if cell == "impossible" and ncre != len(v):
            fails.append(f"{len(v) - ncre} `impossible` clips do not create energy")
        if cell in ("anchor", "legal") and ncre:
            fails.append(f"{ncre} `{cell}` clips create energy")

    # ---- 4. matched twins ---------------------------------------------------------
    seeds = defaultdict(list)
    for x, p in zip(L, paths):
        seeds[x["seed"]].append(x)
    worst = 0.0
    for s, group in seeds.items():
        ref = group[0]["per_frame"]
        for x in group[1:]:
            for ra, rb in zip(ref, x["per_frame"]):
                if ra["frame"] > PRE_FRAME:
                    break
                for t in ("A", "B"):
                    worst = max(worst, float(np.max(np.abs(
                        np.array(ra[t]["position"]) - np.array(rb[t]["position"])))))
    print(f"\n4. twins share pre-contact   max position delta {worst:.3e}   "
          f"{'OK' if worst < 1e-9 else 'FAIL'}")
    if worst >= 1e-9:
        fails.append("twins diverge before contact")

    # ---- 5. |log e| matched within pairs ------------------------------------------
    bad = []
    for s, group in seeds.items():
        ks = defaultdict(list)
        for x in group:
            e_t = x["energy_edit"]["e_target"]
            if e_t == 1.0:
                continue
            ks[round(max(e_t, 1 / e_t), 6)].append(e_t)
        for k, es in ks.items():
            if len(es) != 2 or abs(abs(np.log(es[0])) - abs(np.log(es[1]))) > 1e-9:
                bad.append((s, k, es))
    print(f"5. |log e| matched in pairs  {len(bad)} bad   {'OK' if not bad else 'FAIL'}")
    if bad:
        fails.append(f"{len(bad)} pairs are not equidistant from elastic")

    # ---- 6. in frame ---------------------------------------------------------------
    out = sum(1 for x in L for row in x["per_frame"]
              for t in ("A", "B") if not row[t]["in_frame"])
    print(f"6. all frames in shot        {out} out-of-frame readings   "
          f"{'OK' if out == 0 else 'FAIL'}")
    if out:
        fails.append(f"{out} out-of-frame readings")

    # ---- reported, not enforced ----------------------------------------------------
    imp = [r for r in rows if r["cell"] == "impossible"]
    leg = [r for r in rows if r["cell"] == "legal"]
    if imp and leg:
        y = np.array([True] * len(imp) + [False] * len(leg))
        a_e = auc(y, np.array([r["e_meas"] for r in imp + leg]))
        a_kr = auc(y, np.array([r["kr"] for r in imp + leg]))
        print(f"\n   REPORTED, NOT A FAILURE — mass-free separability of "
              f"impossible vs legal:")
        print(f"     realised e   AUC {a_e:.3f}      KE ratio   AUC {a_kr:.3f}")
        print("     Both near 1.000 BY DESIGN. `e` is visible and mass-free, so a probe")
        print("     reads the label off separation speed and demonstrates nothing (§44).")
        print("     This set is for VIOLATION OF EXPECTATION: the violation has to be")
        print("     visible for the model to react at all. The question is whether the")
        print("     model treats IMPOSSIBLE differently from merely UNUSUAL, which is")
        print("     what the k / 1-over-k pairing isolates. Do not probe this set.")

    print()
    if fails:
        print("VALIDATION FAILED")
        for f in fails:
            print(f"  - {f}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
