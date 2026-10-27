#!/usr/bin/env python3
"""Validate `ke_factor_set` -- physics, the exact KE ladder, and the leak audit.

    python validate_ke_set.py --dataset-dir ke_factor_set [--plan-only] [--out _validation.json]

Reads each clip's labels.json (or, with --plan-only, the planner's simulated physics,
so the audit runs before a single pixel is rendered). numpy only. Exit code 1 if any
gate fails. Gates:

  complete        every planned clip has _COMPLETE, none _FAILED       (rendered only)
  physics         momentum to 1e-9, KE_post/KE_pre and realised e to 1e-6, contact on frame 20 +- 1
  ladder          KE / KE(level 0) == 2^(k/2) within every group, to 1e-6
  one_factor      S groups: masses, radii, materials constant; M groups: every pre- AND
                  post-contact velocity constant. Directions and geometry constant in both
  same_population no clip-level variable tells the families apart: |AUC - 0.5| < 0.10
                  for log KE, log masses, log speeds, radii, material, |p|, geometry
  nuisance_flat   within-group, no nuisance tracks the level: |corr| < 0.10 for radius in
                  family M and for mass, radius, material in family S

Reported, not gated, because they are the factor itself rather than a leak:
material vs level in family M (it IS the visible mass), frame-0 separation and exit
frame vs level in family S (they ARE speed), and the momentum slope that tells a KE
readout from a |p| readout (see plan_ke_set.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _v(x):
  return np.asarray(x, dtype=float)[:2]


def records(root: Path, plan: dict, plan_only: bool):
  rows, missing, failed = [], [], []
  for c in plan["clips"]:
    p, d = c["params"], root / c["name"]
    rec = {"name": c["name"], "group": c["group"], "family": c["family"], "level": c["level"]}
    if plan_only:
      ph = c["physics"]
      m_a = p["density_a"] * 4 / 3 * np.pi * p["radius_a"] ** 3
      m_b = p["density_b"] * 4 / 3 * np.pi * p["radius_b"] ** 3
      rec.update(ke_pre=ph["ke_pre"], ke_post=ph["ke_post"], e=ph["e"],
                 mom_err=ph["momentum_rel_error"], contact=ph["contact_frame"],
                 exit_a=ph["exit_frames"]["A"], exit_b=ph["exit_frames"]["B"], v_post=None)
    else:
      if (d / "_FAILED").exists():
        failed.append(c["name"])
      if not (d / "_COMPLETE").exists():
        missing.append(c["name"])
        continue
      L = json.load(open(d / "labels.json"))
      col = L["collision"]
      m_a, m_b = L["derived"]["mass_a"], L["derived"]["mass_b"]
      pre, post = col["pre"], col["post"]
      dva = _v(post["velocity_A"]) - _v(pre["velocity_A"])
      n = dva / np.linalg.norm(dva)
      rel = lambda s: abs(np.dot(_v(s["velocity_A"]) - _v(s["velocity_B"]), n))
      rec.update(ke_pre=pre["kinetic_energy"], ke_post=post["kinetic_energy"],
                 e=rel(post) / rel(pre), mom_err=col["momentum_rel_error"],
                 contact=col["actual_frame"], exit_a=L["exit_frames"]["A"], exit_b=L["exit_frames"]["B"],
                 v_post=np.r_[_v(post["velocity_A"]), _v(post["velocity_B"])])
    va = p["speed_a"] * np.array([np.cos(p["direction_a_rad"]), np.sin(p["direction_a_rad"])])
    vb = p["speed_b"] * np.array([np.cos(p["direction_b_rad"]), np.sin(p["direction_b_rad"])])
    rec.update(m_a=m_a, m_b=m_b, r_a=p["radius_a"], r_b=p["radius_b"],
               metal=float(p["material_a"] == "metal"), speed_a=p["speed_a"], speed_b=p["speed_b"],
               v_pre=np.r_[va, vb], p_tot=float(np.linalg.norm(m_a * va + m_b * vb)),
               b_frac=p["impact_parameter_frac"], b_moving=float(p["b_moving"]),
               dirs=(p["direction_a_rad"], p["direction_b_rad"], p["impact_parameter_sign"],
                     *p["contact_offset"]),
               sep0=float(np.linalg.norm(va - vb)) * 2.5)  # |v_rel| * t_contact: frame-0 separation term
    rows.append(rec)
  return rows, missing, failed


def auc(x, y):
  """P(x_S > x_M) with ties at 1/2 -- rank AUC of one scalar for family identity."""
  x, y = np.asarray(x, float), np.asarray(y, bool)
  order = np.argsort(x, kind="mergesort")
  ranks = np.empty(len(x))
  xs = x[order]
  i = 0
  while i < len(x):                       # average ranks over ties
    j = i
    while j + 1 < len(x) and xs[j + 1] == xs[i]:
      j += 1
    ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
    i = j + 1
  n1, n0 = y.sum(), (~y).sum()
  return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def within_corr(rows, key, fam):
  """corr(level, x) after removing each group's mean of x: only within-group variation counts."""
  R = [r for r in rows if r["family"] == fam]
  x = np.array([r[key] for r in R], float)
  k = np.array([r["level"] for r in R], float)
  g = np.array([r["group"] for r in R])
  xd = x.copy()
  for u in np.unique(g):
    xd[g == u] -= x[g == u].mean()
  if xd.std() < 1e-12:
    return 0.0
  return float(np.corrcoef(xd, k)[0, 1])


def within_slope(rows, key, fam):
  """Slope of log(key) on log KE using only within-group variation."""
  R = [r for r in rows if r["family"] == fam]
  g = np.array([r["group"] for r in R])
  x, y = np.log([r["ke_pre"] for r in R]), np.log([r[key] for r in R])
  for u in np.unique(g):
    x[g == u] -= x[g == u].mean()
    y[g == u] -= y[g == u].mean()
  return float(np.dot(x, y) / np.dot(x, x))


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--dataset-dir", default="momentum_dataset/ke_factor_set")
  ap.add_argument("--plan-only", action="store_true")
  ap.add_argument("--out", default=None)
  args = ap.parse_args()
  root = Path(args.dataset_dir)
  plan = json.load(open(root / "plan.json"))
  rows, missing, failed = records(root, plan, args.plan_only)
  gates, report = {}, {}

  if not args.plan_only:
    gates["complete"] = not missing and not failed
    report["complete"] = {"rendered": len(rows), "planned": len(plan["clips"]),
                          "missing": missing[:20], "failed": failed}
  if not rows:
    print("no clips to validate"); return 1

  ke_r = np.array([abs(r["ke_post"] / r["ke_pre"] - 1) for r in rows])
  e_d = np.array([abs(r["e"] - 1) for r in rows])
  mom = np.array([r["mom_err"] for r in rows])
  cf = np.array([abs(r["contact"] - 20) for r in rows])
  gates["physics"] = bool(mom.max() < 1e-9 and ke_r.max() < 1e-6 and e_d.max() < 1e-6 and cf.max() <= 1)
  report["physics"] = {"momentum_rel_err_max": float(mom.max()), "ke_ratio_dev_max": float(ke_r.max()),
                       "e_dev_max": float(e_d.max()), "contact_offset_max": float(cf.max())}

  groups = {}
  for r in rows:
    groups.setdefault(r["group"], []).append(r)
  lad, one = [], []
  for gname, R in groups.items():
    R = sorted(R, key=lambda r: r["level"])
    base = [r for r in R if r["level"] == 0]
    if base:
      for r in R:
        lad.append(abs(r["ke_pre"] / base[0]["ke_pre"] / 2 ** (r["level"] / 2) - 1))
    fixed = (("m_a", "m_b", "r_a", "r_b", "metal") if R[0]["family"] == "S" else ("v_pre",)) + ("dirs", "b_frac", "b_moving")
    for key in fixed:
      vals = np.array([np.atleast_1d(np.asarray(r[key], float)) for r in R])
      one.append(float(np.abs(vals - vals[0]).max()))
    if R[0]["family"] == "M" and R[0]["v_post"] is not None:
      vp = np.array([r["v_post"] for r in R])
      one.append(float(np.abs(vp - vp[0]).max()) / 1e3)   # solver-level, tolerance 1e-6 after scaling
  gates["ladder"] = bool(max(lad) < 1e-6)
  gates["one_factor"] = bool(max(one) < 1e-9)
  report["ladder"] = {"max_rel_dev": float(max(lad)), "groups": len(groups)}
  report["one_factor"] = {"max_abs_dev": float(max(one))}

  fam = np.array([r["family"] == "S" for r in rows])
  pop = {}
  for key, f in (("log_ke", lambda r: np.log(r["ke_pre"])), ("log_m_a", lambda r: np.log(r["m_a"])),
                 ("log_m_b", lambda r: np.log(r["m_b"])), ("log_speed_a", lambda r: np.log(r["speed_a"])),
                 ("speed_b", lambda r: r["speed_b"]), ("r_a", lambda r: r["r_a"]), ("r_b", lambda r: r["r_b"]),
                 ("metal", lambda r: r["metal"]), ("log_p", lambda r: np.log(r["p_tot"])),
                 ("b_frac", lambda r: r["b_frac"]), ("b_moving", lambda r: r["b_moving"]),
                 ("sep0", lambda r: r["sep0"])):
    pop[key] = auc([f(r) for r in rows], fam)
  worst = max(pop, key=lambda k: abs(pop[k] - 0.5))
  gates["same_population"] = bool(abs(pop[worst] - 0.5) < 0.10)
  report["same_population"] = {"auc": pop, "worst": worst}

  for r in rows:
    r["log_m_a"], r["log_r_a"], r["log_speed_a"] = np.log(r["m_a"]), np.log(r["r_a"]), np.log(r["speed_a"])
    r["exit"] = min(r["exit_a"], r["exit_b"])
  nuis = {"M": {k: within_corr(rows, k, "M") for k in ("log_r_a", "log_speed_a", "b_frac", "exit")},
          "S": {k: within_corr(rows, k, "S") for k in ("log_m_a", "log_r_a", "metal")}}
  gates["nuisance_flat"] = bool(abs(nuis["M"]["log_r_a"]) < 0.10 and all(abs(v) < 0.10 for v in nuis["S"].values()))
  report["nuisance_flat"] = nuis
  report["reported_not_gated"] = {
      "M: corr(material, level) -- the visible mass cue": within_corr(rows, "metal", "M"),
      "S: corr(frame-0 separation, level) -- speed": within_corr(rows, "sep0", "S"),
      "S: corr(first exit frame, level) -- speed": within_corr(rows, "exit", "S"),
      "within-group momentum slope dlog|p|/dlogKE (S 0.5, M 1.0 by construction)": {
          f: within_slope(rows, "p_tot", f) for f in "SM"},
  }

  for g, ok in gates.items():
    print(f"  {'PASS' if ok else 'FAIL'}  {g}")
  print(json.dumps(report, indent=1, default=float))
  out = Path(args.out) if args.out else root / ("_validation_plan.json" if args.plan_only else "_validation.json")
  out.write_text(json.dumps({"gates": gates, "report": report}, indent=1, default=float))
  return 0 if all(gates.values()) else 1


if __name__ == "__main__":
  sys.exit(main())
