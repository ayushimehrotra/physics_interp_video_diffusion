#!/usr/bin/env python3
"""Stage 9 - output-level law scores from stage 8's readouts (spec §6), driven by configs/eval/outputs.yaml.

    /root/envs/physics_venv/bin/python scripts/09_eval_outputs.py --track a --model configs/models/cogvideox_2b.yaml --data data/f1
    /root/envs/physics_venv/bin/python scripts/09_eval_outputs.py --track b --model configs/models/wan22_i2v_a14b.yaml --data data/f1

Laws per family: ``laws`` in the config (physlaw.eval.laws residuals).
Track A -> results/<model>/<family>/outputs_trackA.{json,md}: per legal twin, drop rates (generated, real), mean
  residuals of the generated outcome vs the real-clip floor (true pre state), S_outcome with a group-bootstrap CI,
  and the gate (_gate.json from stage 7).
Track B -> results/<model>/<family>/outputs_trackB.{json,md}: per gate (strict, loose), measurable counts, drop
  reasons, medians of e / momentum residual / KE ratio / every law residual given the clip's OWN pre-event state,
  loose-gate violations, and the real-clip validation if stage 8 --real wrote it to the same directory.
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
    return fmt.format(x) if isinstance(x, (int, float)) else "-"


def md_track_a(res) -> list[str]:
    laws = res["laws"]
    rows = [f"Gate: {res['gate']}", "",
            "| legal twin | generated measurable | drop reasons | real measurable | r_p gen / floor | r_E gen / floor | "
            "E_out gen / floor | " + " | ".join(f"{l} gen / floor" for l in laws) + " | S_outcome [95% CI] |",
            "|---|---|---|---|---|---|---|" + "---|" * len(laws) + "---|"]
    for t, d in res["twins"].items():
        f = lambda kk: f"{_f(d.get(kk + '_generated'))} / {_f(d.get(kk + '_floor'))}"
        ci = d.get("S_outcome_ci95")
        rows.append(f"| {t} | {d['measurable_generated']}/{d['n_generated']} | {d['drop_reasons_generated']} | "
                    f"{d['measurable_real']}/{d['n_real']} | {f('r_p')} | {f('r_E')} | {f('E_out')} | "
                    + " | ".join(f(f"res_{l}") for l in laws) + " | "
                    + _f(d.get("S_outcome"), "{:+.2f}") + (f" [{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "") + " |")
    return rows


def md_track_b(res) -> list[str]:
    keys = ["e", "r_momentum", "ke_ratio", *O.TRAJ_KEYS, *(f"res_{l}" for l in res["laws"])]
    rows = []
    for gate, d in res["gates"].items():
        rows += [f"## Gate: {gate}", "",
                 "| clips | n | measurable | drop reasons | " + " | ".join(f"median {k}" for k in keys) + " | violations |",
                 "|---|---|---|---|" + "---|" * len(keys) + "---|"]
        sets = [("generated", d["generated"])] + [(f"real {t}", s) for t, s in d.get("real", {}).items() if t != "all"]
        for name, s in sets:
            rows.append(f"| {name} | {s['n']} | {s['measurable']} | {s['drop_reasons']} | "
                        + " | ".join(_f(s.get(f"median_{k}")) for k in keys) + f" | {s.get('violations', '')} |")
        if "real" in d:
            a = d["real"]["all"]
            rows.append(f"\nReal-clip validation (§6.1): {a['measurable']}/{a['n']} measurable.")
        rows += ["", f"Measurable generated clips: {', '.join(d['measurable_clips']) or 'none'}", ""]
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
    fam = data.name.split("_")[0]
    laws = ecfg["laws"].get(fam, [])                   # a list, or {subfamily: list}
    rd = Path(a.readouts) if a.readouts else REPO / "cache" / mcfg.name / (data.name if a.track == "a" else f"{data.name}_trackB")
    out_dir = Path(a.out) if a.out else REPO / "results" / mcfg.name / data.name
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(model=mcfg.name, family=data.name, track=a.track.upper(), readouts=str(rd),
                model_config_hash=config_hash(mcfg), eval_config_hash=config_hash(ecfg))
    if a.track == "a":
        gen_rows, real_rows = _load(rd / "_readout_trackA.jsonl"), _load(rd / "_readout_trackA_real.jsonl")
        if not gen_rows:
            sys.exit(f"[09] no readouts in {rd}: run scripts/08_track_outputs.py --track a first")
        res = O.track_a_summary(gen_rows, real_rows, data, laws, int(ecfg["bootstrap"]["n"]))
        gate = rd / "_gate.json"
        res["gate"] = {k: v for k, v in json.loads(gate.read_text()).items() if k != "clips"} if gate.exists() else None
        md = md_track_a(res)
    else:
        rows = {g: _load(rd / O.readout_file(g)) for g in O.GATES}
        real = {g: _load(rd / O.readout_file(g, real=True)) for g in O.GATES}
        rows, real = {g: r for g, r in rows.items() if r}, {g: r for g, r in real.items() if r}
        if not rows and not real:
            sys.exit(f"[09] no readouts in {rd}: run scripts/08_track_outputs.py --track b first")
        res = O.track_b_summary(rows, real, data, laws)
        md = md_track_b(res)
    res.update(meta)
    name = f"outputs_track{a.track.upper()}"
    (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1, default=float))
    text = "\n".join([f"# Output track {a.track.upper()} - {mcfg.name} on {data.name}", "",
                      f"Readouts: `{rd}`; laws: {laws or 'none'}", ""] + md) + "\n"
    (out_dir / f"{name}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
