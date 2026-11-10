"""Leaderboard (spec §8) from results/<model>/<family>/<level>.json: one row per (model, family), blank where a level
has not been run.

Columns and where each comes from (first match wins):
  L0           mean R² over l0.json targets
  L1           S per law on the legal twins: l1.json ``S_family`` blocks (law-family score, if present), else the
               legacy outcome / momentum S of the leak-free ``frame_mm`` variant (DiTs: median over (SNR, layer) spaces,
               no selection)
  L2           motion-adjusted win rate per contrast (covariates speed + copy-last, first available distance of
               PRIMARY_DISTANCES, fixed a priori: object-region MSE of the model's own prediction / clean-context loss);
               l2.json ``primary`` if the file names one
  L2b          AUC per contrast (l2b.json); encoded-but-unused = (AUC - 0.5) - (L2 adjusted win - 0.5)
  L3           Galilean invariance error per target and boost level (f4), mass-independence slope of the forecast
               (f6/f7), transfer gaps
  L4           r(predicted, observed) change of v_B' along n for the mass direction (frame_mm, else frame)
  outputs      track A S_outcome per legal twin + measurable fraction; track B measurable (strict, loose) and the
               median law residuals
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PRIMARY_DISTANCES = ("mse_obj", "clean_mse_obj", "fut_mse_obj", "l1_obj", "mse_all", "clean_mse_all", "fut_mse_all")
COVARIATES = ("speed+copy_last", "post_speed", "copy_last")
LEGAL = ("conserve_e1", "conserve_e015")


def _num(x):
    return float(x) if isinstance(x, (int, float)) and np.isfinite(x) else None


def _l0(d):
    v = [t["r2"] for t in d.get("targets", {}).values() if _num(t.get("r2")) is not None]
    return {"mean_r2": float(np.mean(v)), "n_targets": len(v)} if v else None


def _l1(d):
    out = {}
    for blk in (d.get("S_family"), d.get("law_family", {}).get("S_family") if isinstance(d.get("law_family"), dict) else None):
        if isinstance(blk, dict):
            out.update({f"S_family.{k}": _num(v if not isinstance(v, dict) else v.get("S")) for k, v in blk.items()})
    vs = d.get("variants", {})
    keys = [k for k in vs if k == "frame_mm" or k.endswith("|frame_mm")]
    for t in LEGAL:
        for law in ("outcome", "momentum"):
            vals = [vs[k]["twins"][t]["S"][law] for k in keys if t in vs[k].get("twins", {})]
            vals = [v for v in vals if _num(v) is not None]
            if vals:
                out[f"S_{law}.{t}"] = float(np.median(vals))
    if keys:
        out["variant"] = "frame_mm" if keys == ["frame_mm"] else f"median over {len(keys)} spaces (frame_mm)"
    return out or None


def _l2(d):
    if isinstance(d.get("primary"), dict):
        return {c: _num(v) for c, v in d["primary"].items()}
    ma = d.get("motion_adjusted", {}).get("contrasts", {})
    cs = next((c for c in COVARIATES if c in ma), None)
    if cs is None:
        return None
    out = {}
    for c, by_m in ma[cs].items():
        m = next((m for m in PRIMARY_DISTANCES if m in by_m), None)
        if m:
            out[c] = by_m[m]["adj_win_rate"]
    out["_distance"], out["_covariates"] = m, cs
    return out


def _l2b(d):
    src = d.get("contrasts", d)
    out = {}
    for c, v in src.items():
        a = v.get("auc") if isinstance(v, dict) else None
        if a is None and isinstance(v, dict):
            a = v.get("best", {}).get("auc") if isinstance(v.get("best"), dict) else None
        if _num(a) is not None:
            out[c] = float(a)
    return out or None


def _l3(d):
    out = {}
    for t, v in d.get("galilean", {}).get("targets", {}).items():
        out[f"within_error.{t}"] = v["within_error"]
        for lv, w in v["by_level"].items():
            out[f"invariance_error.{t}.{lv}"] = w["invariance_error"]
    for sp, v in d.get("mass_independence", {}).get("spaces", {}).items():
        if "forecast" in v:
            out[f"mass_slope.{sp}"] = v["forecast"]["slope_rel"]
    for t, v in d.get("transfer", {}).get("targets", {}).items():
        for k, w in v.items():
            out[f"transfer_gap.{t}.{k}"] = w["gap"]
    if d.get("stand_in"):
        out["stand_in"] = True
    return out or None


def _l4(d):
    vs = d.get("variants", {})
    k = next((k for k in vs if k.endswith("frame_mm")), next((k for k in vs if k.endswith("frame")), None))
    if k is None or "mass" not in vs[k]:
        return None
    out = {"r_B": vs[k]["mass"]["r_B"], "variant": k}
    for c, v in vs[k].items():
        if c != "mass":
            out[f"r_B.{c}"] = v["r_B"]
    if d.get("steering"):
        out["steering_sign_agrees"] = f"{d['steering']['n_sign_agrees']}/{d['steering']['n_scored']}"
    if d.get("stand_in"):
        out["stand_in"] = True
    return out


def _out_a(d):
    out = {}
    for t, v in d.get("twins", {}).items():
        out[f"measurable.{t}"] = f"{v['measurable_generated']}/{v['n_generated']}"
        if _num(v.get("S_outcome")) is not None:
            out[f"S_outcome.{t}"] = v["S_outcome"]
    if d.get("gate"):
        out["gate_passed"] = d["gate"].get("passed")
    return out or None


def _out_b(d):
    out = {}
    for g, v in d.get("gates", {}).items():
        s = v["generated"]
        out[f"measurable.{g}"] = f"{s['measurable']}/{s['n']}"
        for k, x in s.items():
            if k.startswith("median_res_"):
                out[f"{k[7:]}.{g}"] = x
        if "measurable_without_violation" in s:
            out[f"no_violation.{g}"] = f"{s['measurable_without_violation']}/{s['n']}"
    return out or None


READERS = {"l0": _l0, "l1": _l1, "l2": _l2, "l2b": _l2b, "l3": _l3, "l4": _l4,
           "outputs_trackA": _out_a, "outputs_trackB": _out_b}
COLUMNS = ("l0", "l1", "l2", "l2b", "encoded_unused", "l3", "l4", "outputs_trackA", "outputs_trackB")


def collect(results: Path) -> list[dict]:
    rows = []
    for fam_dir in sorted(p for p in results.glob("*/*") if p.is_dir() and p.parent.name != "figures"):
        row = {"model": fam_dir.parent.name, "family": fam_dir.name}
        for name, fn in READERS.items():
            f = fam_dir / f"{name}.json"
            if f.exists():
                v = fn(json.loads(f.read_text()))
                if v:
                    row[name] = v
        if row.get("l2b") and row.get("l2"):
            row["encoded_unused"] = {c: (a - 0.5) - (row["l2"][c] - 0.5) for c, a in row["l2b"].items()
                                     if _num(row["l2"].get(c)) is not None}
        if len(row) > 2:
            rows.append(row)
    return rows


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:+.2f}" if abs(v) < 10 else f"{v:.3g}"
    return str(v)


def markdown(rows: list[dict]) -> str:
    """One table per family; each level a column, its sub-scores listed inside the cell."""
    out = ["# Leaderboard (spec §8)", "",
           "Blank = level not run. Cells list sub-scores as key value; see physlaw/analysis/aggregate.py for sources.", ""]
    for fam in sorted({r["family"] for r in rows}):
        out += [f"## {fam}", "", "| model | " + " | ".join(COLUMNS) + " |", "|---|" + "---|" * len(COLUMNS)]
        for r in [r for r in rows if r["family"] == fam]:
            cells = []
            for c in COLUMNS:
                v = r.get(c)
                if v is None:
                    cells.append("")
                elif isinstance(v, dict):
                    cells.append("<br>".join(f"{k} {_fmt(x)}" for k, x in v.items() if not k.startswith("_")))
                else:
                    cells.append(_fmt(v))
            out.append(f"| {r['model']} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def run(results: Path) -> dict:
    rows = collect(results)
    (results / "leaderboard.json").write_text(json.dumps({"columns": list(COLUMNS), "rows": rows}, indent=1, default=float))
    (results / "leaderboard.md").write_text(markdown(rows))
    return {"rows": rows}
