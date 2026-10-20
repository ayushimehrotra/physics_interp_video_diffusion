#!/usr/bin/env python3
"""Validate a twin or ladder set (conservation_set_v1, energy_voe_450, energy_ladder_v1, scene_ladder_v1).

    python3 validate_sets.py --dataset-dir conservation_set_v1 [--disjoint-from energy_ladder_v1 ...]

Reads only labels.json. The checks (register §58 "all 7 checks pass", plus the
pipeline audit's items):

  momentum            every non-violating clip conserves momentum to MOMENTUM_TOL
  restitution         restitution_realised equals the arm's target to RESTITUTION_TOL,
                      and labels['restitution'] agrees with it (the label trap)
  energy_closed_form  dKE on every arm equals -(1/2) mu (1 - e^2) (v_rel.n)^2
  pre_contact_shared  every clip of a seed group has identical pre-contact velocities
                      and positions (frames 0..20) to 0.0
  violations_violate  every violating clip breaks momentum, and keeps its twin's
                      separation speed (the split edit holds a + b)
  exit_rule           no edited arm leaves the frame before its honest twin
  one_phase_split     contact lands after frame 20 in every clip (one phase split per set)
  seeds_disjoint      seed groups do not overlap any --disjoint-from set

Exit code is non-zero if any check fails. Writes _validation/validation_report.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

MOMENTUM_TOL = 1e-9
RESTITUTION_TOL = 1e-9
ENERGY_TOL = 1e-9
CONTACT_FRAME = 20


def load(root: Path) -> list[dict]:
    out = []
    for lp in sorted(root.glob("*/labels.json")):
        l = json.loads(lp.read_text())
        l["_dir"] = lp.parent.name
        out.append(l)
    return out


def _xy(v):
    return np.asarray(v, dtype=np.float64)[:2]


def pre_state(l):
    col = l["collision"]
    n = _xy(col.get("contact_normal_analytic") or col["contact_normal"])
    n /= np.linalg.norm(n)
    return (l["objects"]["A"]["mass"], l["objects"]["B"]["mass"], _xy(col["pre"]["velocity_A"]),
            _xy(col["pre"]["velocity_B"]), n)


def realised_e(l):
    if l.get("restitution_realised") is not None:
        return float(l["restitution_realised"])
    return float(l["collision"]["restitution_measured"])


def check_all(clips: list[dict], disjoint_sets: list[list[dict]]) -> dict:
    checks = {}
    groups = defaultdict(list)
    for l in clips:
        groups[int(l.get("seed_group", l["seed"]))].append(l)
    honest = [l for l in clips if not l.get("violates_momentum", False)]
    viol = [l for l in clips if l.get("violates_momentum", False)]

    # momentum
    errs = [l["collision"]["momentum_rel_error"] for l in honest]
    checks["momentum"] = {"pass": bool(errs) and max(errs) <= MOMENTUM_TOL,
                          "detail": f"{len(honest)} non-violating clip(s): rel error max {max(errs) if errs else float('nan'):.2e}"}

    # restitution: realised vs arm target and vs the top-level label
    worst, trap = 0.0, 0
    for l in clips:
        target = (l.get("edit") or {}).get("e_new", l.get("restitution_simulated", l["restitution"]))
        worst = max(worst, abs(realised_e(l) - float(target)))
        if abs(float(l["restitution"]) - realised_e(l)) > 1e-6:
            trap += 1
    checks["restitution"] = {"pass": worst <= RESTITUTION_TOL and trap == 0,
                             "detail": f"|realised - target| max {worst:.2e}; labels['restitution'] disagrees with realised in {trap} clip(s)"}

    # energy closed form, on non-violating clips
    worst = 0.0
    for l in honest:
        m_a, m_b, v_a, v_b, n = pre_state(l)
        mu = m_a * m_b / (m_a + m_b)
        u = float(np.dot(v_a - v_b, n))
        e = realised_e(l)
        dke = l["collision"]["post"]["kinetic_energy"] - l["collision"]["pre"]["kinetic_energy"]
        worst = max(worst, abs(dke - (-0.5 * mu * (1 - e ** 2) * u ** 2)))
    checks["energy_closed_form"] = {"pass": worst <= 1e-9 * max(1.0, 1.0),
                                    "detail": f"|dKE - closed form| max {worst:.2e}"}

    # pre-contact shared within seed group
    worst, n_groups = 0.0, 0
    for g, ls in groups.items():
        if len(ls) < 2:
            continue
        n_groups += 1
        ref = ls[0]
        for l in ls[1:]:
            for tag in ("A", "B"):
                worst = max(worst, float(np.abs(np.asarray(l["collision"]["pre"][f"velocity_{tag}"])
                                                - np.asarray(ref["collision"]["pre"][f"velocity_{tag}"])).max()))
                for f in range(CONTACT_FRAME + 1):
                    worst = max(worst, float(np.abs(np.asarray(l["per_frame"][f][tag]["position"])
                                                    - np.asarray(ref["per_frame"][f][tag]["position"])).max()))
    checks["pre_contact_shared"] = {"pass": worst == 0.0, "detail": f"{n_groups} group(s); max pre-contact difference {worst:.2e}"}

    # violations violate, and keep separation speed
    if viol:
        min_viol = min(l["collision"]["momentum_rel_error"] for l in viol)
        sep = 0.0
        for l in viol:
            twins = [h for h in groups[int(l.get("seed_group", l["seed"]))] if h.get("arm", "honest") == "honest"]
            if twins:
                sep = max(sep, abs(realised_e(l) - realised_e(twins[0])))
        checks["violations_violate"] = {"pass": min_viol > 1e-3 and sep <= 1e-9,
                                        "detail": f"{len(viol)} violating: |dp|/|p| min {min_viol:.3f}; "
                                                  f"restitution differs from honest twin by at most {sep:.2e}"}
    else:
        checks["violations_violate"] = {"pass": True, "detail": "no violating clips in this set"}

    # exit rule
    bad = 0
    for l in clips:
        ex = l.get("exit_frame")
        if ex and any(ex["this_clip"][t] < ex["honest_twin"][t] for t in ("A", "B")):
            bad += 1
    checks["exit_rule"] = {"pass": bad == 0, "detail": f"{bad} clip(s) leave the frame before their honest twin"}

    # one phase split
    cf = [l["collision"]["actual_frame"] for l in clips]
    checks["one_phase_split"] = {"pass": min(cf) > CONTACT_FRAME and max(cf) < CONTACT_FRAME + 1,
                                 "detail": f"contact frame range [{min(cf):.3f}, {max(cf):.3f}]"}

    # seeds disjoint
    mine = set(groups)
    overlap = set()
    for other in disjoint_sets:
        overlap |= mine & {int(l.get("seed_group", l["seed"])) for l in other}
    checks["seeds_disjoint"] = {"pass": not overlap, "detail": f"{len(overlap)} overlapping seed(s)" if overlap else "disjoint"}

    arms = defaultdict(int)
    for l in clips:
        arms[l.get("arm", "violating" if l.get("violates_momentum") else "honest")] += 1
    es = sorted({round(realised_e(l), 4) for l in honest})
    summary = {"num_clips": len(clips), "num_seed_groups": len(groups), "arms": dict(arms),
               "restitutions_realised": es,
               "strata": {"elastic": sum(realised_e(l) > 0.5 for l in clips), "inelastic": sum(realised_e(l) <= 0.5 for l in clips)},
               "clips_with_a_ball_leaving_frame": sum(any(ex["this_clip"][t] < len(l["per_frame"]) for t in ("A", "B"))
                                                       for l in clips for ex in [l.get("exit_frame")] if ex)}
    return {"checks": checks, "summary": summary}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", required=True, type=Path)
    p.add_argument("--disjoint-from", type=Path, nargs="*", default=[])
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()
    clips = load(args.dataset_dir)
    if not clips:
        sys.exit(f"no labels.json under {args.dataset_dir}")
    report = check_all(clips, [load(d) for d in args.disjoint_from])
    out = args.output_dir or (args.dataset_dir / "_validation")
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation_report.json").write_text(json.dumps(report, indent=1, default=str))
    ok = True
    for name, c in report["checks"].items():
        ok &= bool(c["pass"])
        print(f"  {'PASS' if c['pass'] else 'FAIL'}  {name:20s} {c['detail']}")
    print(json.dumps(report["summary"], indent=1))
    print(f"\n{'ALL CHECKS PASS' if ok else 'SOME CHECKS FAILED'} -> {out / 'validation_report.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
