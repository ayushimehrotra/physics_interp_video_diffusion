#!/usr/bin/env python3
"""Stage 9 - output-level law scores from stage 8's readouts (spec §5.0.4 Outputs, §6), driven by configs/eval/outputs.yaml.

    /root/envs/physics_venv/bin/python scripts/09_eval_outputs.py --track a --model configs/models/cogvideox_2b.yaml --data data/f1
    /root/envs/physics_venv/bin/python scripts/09_eval_outputs.py --track b --model configs/models/wan22_i2v_a14b.yaml --data data/f1

Every readout feeds the unified laws (physlaw.eval.laws; scale-free: no true g, fps or metric speed): collisions
epsilon (line of centres), eta, antiparallel, normal alignment; F7 bounce epsilon, projectile / drop g_ratio,
axis_ratio, mass_ratio; F8 epsilon (Galileo's ramp heights) and epsilon_speed.
Track A -> results/<model>/<family>/outputs_trackA.{json,md}: drop rates, medians of every statistic (generated vs
  the real legal twins), the energy share eps <= 1 + tol (config ``energy``) against the real readout distribution,
  and for collisions the S_mom package per twin and pooled (config ``momentum``); the gate (_gate.json, stage 7).
Track B -> results/<model>/<family>/outputs_trackB.{json,md}: the same per gate (strict, loose), loose-gate
  violations, the mass-free momentum package, and the real-clip validation / floors if stage 8 --real wrote it.
--readouts: the stage 8 output directory (default: the generation directory in cache/).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from physlaw.config import config_hash, load_config, resolve_path  # noqa: E402
from physlaw.eval import outputs as O  # noqa: E402


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def _f(x, fmt="{:.3f}"):
    return fmt.format(x) if isinstance(x, (int, float)) and x == x else "-"


def _ci(x):
    return f" [{x[0]:+.2f}, {x[1]:+.2f}]" if x else ""


def md_stats(sets: list, keys) -> list[str]:
    """Medians of the law statistics for [(name, _stats dict)]; only the keys some set has."""
    keys = [k for k in keys if any(f"median_{k}" in st for _, st in sets)]
    rows = ["| clips | n | measurable | drop reasons | " + " | ".join(f"median {k}" for k in keys) + " | violations |",
            "|---|---|---|---|" + "---|" * len(keys) + "---|"]
    for name, st in sets:
        rows.append(f"| {name} | {st['n']} | {st['measurable']} | {st['drop_reasons']} | "
                    + " | ".join(_f(st.get(f"median_{k}")) for k in keys) + f" | {st.get('violations', '')} |")
    return rows + [""]


def md_energy(e) -> list[str]:
    if not e or not e["generated"]["n"] and "real" not in e:
        return []
    rows = [f"### Energy: share with epsilon <= 1 + {e['tol']}", "",
            "| clips | n | share eps <= 1 + tol [95% CI] | median eps | true eps |", "|---|---|---|---|---|"]
    g = e["generated"]
    rows.append(f"| generated | {g['n']} | {_f(g['share_le1'], '{:.2f}')}{_ci(g.get('share_le1_ci95'))} | {_f(g['median'])} | - |")
    for name in ("real_legal", "real_creating"):
        if name in e:
            rows.append(f"| {name.replace('_', ' ')} (pooled) | {e[name]['n']} | {_f(e[name]['share_le1'], '{:.2f}')} | "
                        f"{_f(e[name]['median'])} | - |")
    for t, d in e.get("real", {}).items():
        rows.append(f"| real {t} | {d['n']} | {_f(d['share_le1'], '{:.2f}')} | {_f(d['median'])} | {_f(d.get('eps_true'))} |")
    return rows + [""]


def md_track_a(res) -> list[str]:
    rows = [f"Gate: {res['gate']}", ""]
    rows += md_stats([("generated", res["generated"])] + [(f"real {t}", s) for t, s in res["real"].items()], O.STAT_KEYS)
    rows += md_energy(res.get("energy"))
    mom = res.get("momentum")
    if mom:
        mc = res.get("momentum_config", {})
        rows += [f"### Momentum (S_mom package, kappa {mc.get('kappa')}, mass ratio >= {mc.get('min_mass_ratio')})", "",
                 "Interaction rate = measurable generated clips whose momentum exchange D >= kappa x the weakest legal "
                 "collision (real clips in brackets); eta = |sum m dv| / sum |m dv| pooled over gated clips (generated / "
                 "real readout floor / mass-blind rule); S_mom = 1 - (eta_gen - eta_real) / (eta_blind - eta_real).", "",
                 "| legal twin | interaction rate (real) | scored clips / groups | eta gen / real / blind | "
                 "eta real / blind, all real clips | S_mom [95% CI] |", "|---|---|---|---|---|---|"]
        for t, m in mom.items():
            S = _f(m.get("S_mom"), "{:+.2f}") + _ci(m.get("S_mom_ci95"))
            if not m.get("S_mom_defined", True) and m.get("n_scored_clips"):
                S = "undefined (eta_real >= eta_blind)"
            rows.append(f"| {t} | {_f(m.get('interaction_rate'), '{:.2f}')} ({_f(m.get('interaction_rate_real'), '{:.2f}')}) | "
                        f"{m.get('n_scored_clips', '-')} / {m.get('n_scored_groups', '-')} | "
                        f"{_f(m.get('eta_generated'))} / {_f(m.get('eta_real'))} / {_f(m.get('eta_blind'))} | "
                        f"{_f(m.get('eta_real_all'))} / {_f(m.get('eta_blind_all'))} ({m.get('n_real_all', 0)}) | {S} |")
    return rows


def md_momentum_b(d, e_tol) -> list[str]:
    """Track B mass-free momentum table for one gate."""
    mo = d.get("momentum")
    if not mo:
        return []
    rows = ["### Momentum (mass-free)", "",
            "Gated on |dv_A - dv_B| >= kappa |u_n| (n = the line of centres at the generated contact). antiparallel "
            "= -cos(dv_A, dv_B) (1 = opposite pushes, any masses); normal alignment = |dv . n| / |dv|; "
            f"eps <= 1 share counts eps <= 1 + {e_tol} (in brackets: -{e_tol} <= eps <= 1 + {e_tol}); ratio consistency = SD of log |dv_B| / |dv_A| across "
            "seeds of one group (real: across the legal twins of one scene).", "",
            "| clips | measurable | interaction rate | antiparallel mean [95% CI] | normal alignment mean | median eps | "
            "eps <= 1 share (in [0, 1]) | median eta | implied ratio median | ratio log-SD (groups) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    sets = [("generated", mo["generated"])] + ([("real legal (pooled)", mo["real_legal"])] if "real_legal" in mo else []) \
        + [(f"real {t}", s) for t, s in mo.get("real", {}).items()]
    for name, s in sets:
        rows.append(f"| {name} | {s['n_measurable']} | {_f(s.get('interaction_rate'), '{:.2f}')} | "
                    f"{_f(s.get('antiparallel_mean'))}{_ci(s.get('antiparallel_mean_ci95'))} | {_f(s.get('normal_alignment_mean'))} | "
                    f"{_f(s.get('epsilon_median'))} | {_f(s.get('e_le1_share'), '{:.2f}')} ({_f(s.get('e_in_range_share'), '{:.2f}')}) | {_f(s.get('eta_median'))} | {_f(s.get('implied_ratio_median'))} | "
                    f"{_f(s.get('ratio_log_sd'))} ({s.get('ratio_consistency_groups', 0)}) |")
    return rows + [""]


def md_track_b(res) -> list[str]:
    rows = []
    for gate, d in res["gates"].items():
        rows += [f"## Gate: {gate}", ""]
        rows += md_stats([("generated", d["generated"])] + [(f"real {t}", st) for t, st in d.get("real", {}).items()
                                                            if t != "all"], O.STAT_KEYS)
        if "real" in d:
            a = d["real"]["all"]
            rows += [f"Real-clip validation (§6.1): {a['measurable']}/{a['n']} measurable.", ""]
        rows += md_energy(d.get("energy")) + md_momentum_b(d, res.get("momentum_config", {}).get("e_tol"))
        rows += [f"Measurable generated clips: {', '.join(d['measurable_clips']) or 'none'}", ""]
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--track", choices=["a", "b"], required=True)
    ap.add_argument("--eval", default="configs/eval/outputs.yaml")
    ap.add_argument("--readouts", default=None)
    ap.add_argument("--out", default=None, help="default results/<model>/<family>")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    mcfg, ecfg = load_config(a.model), load_config(a.eval, a.set)
    data = resolve_path(a.data)
    rd = Path(a.readouts) if a.readouts else REPO / "cache" / mcfg.name / (data.name if a.track == "a" else f"{data.name}_trackB")
    out_dir = Path(a.out) if a.out else REPO / "results" / mcfg.name / data.name
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(model=mcfg.name, family=data.name, track=a.track.upper(), readouts=str(rd),
                model_config_hash=config_hash(mcfg), eval_config_hash=config_hash(ecfg))
    if a.track == "a":
        gen_rows, real_rows = _load(rd / "_readout_trackA.jsonl"), _load(rd / "_readout_trackA_real.jsonl")
        if not gen_rows:
            sys.exit(f"[09] no readouts in {rd}: run scripts/08_track_outputs.py --track a first")
        res = O.track_a_summary(gen_rows, real_rows, data, int(ecfg["bootstrap"]["n"]), ecfg.get("momentum"),
                                ecfg.get("energy"))
        gate = rd / "_gate.json"
        res["gate"] = {k: v for k, v in json.loads(gate.read_text()).items() if k != "clips"} if gate.exists() else None
        md = md_track_a(res)
    else:
        rows = {g: _load(rd / O.readout_file(g)) for g in O.GATES}
        real = {g: _load(rd / O.readout_file(g, real=True)) for g in O.GATES}
        rows, real = {g: r for g, r in rows.items() if r}, {g: r for g, r in real.items() if r}
        if not rows and not real:
            sys.exit(f"[09] no readouts in {rd}: run scripts/08_track_outputs.py --track b first")
        res = O.track_b_summary(rows, real, data, ecfg.get("momentum"), int(ecfg["bootstrap"]["n"]), ecfg.get("energy"))
        md = md_track_b(res)
    res.update(meta)
    name = f"outputs_track{a.track.upper()}"
    (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1, default=float))
    text = "\n".join([f"# Output track {a.track.upper()} - {mcfg.name} on {data.name}", "",
                      f"Readouts: `{rd}`", ""] + md) + "\n"
    (out_dir / f"{name}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
