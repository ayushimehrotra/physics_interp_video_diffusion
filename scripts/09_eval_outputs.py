#!/usr/bin/env python3
"""Stage 9 — output-level law checks on generated continuations (spec §6). Tracking (spec stage 8) is
done here, by physlaw/eval/outputs.py's colour tracker (see its docstring for the deviation from SAM 2 +
CoTracker and its validation on real clips).

    /root/envs/physics_venv/bin/python scripts/09_eval_outputs.py --model configs/models/cogvideox_2b.yaml --data data/f1

Writes results/<model>/<family>/outputs.{json,md}: per legal twin, drop rates (generated and real),
residuals r_p / r_E / outcome E for generated vs the real-clip floor, S_outcome with bootstrap CI,
and the gate result.
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    mcfg = load_config(a.model)
    data = resolve_path(a.data)
    cache = REPO / "cache" / mcfg.name / data.name
    out_dir = REPO / "results" / mcfg.name / data.name
    out_dir.mkdir(parents=True, exist_ok=True)
    res = O.evaluate(cache, data, O.Geometry(mcfg))
    gate = cache / "_gate.json"
    res["gate"] = {k: v for k, v in json.loads(gate.read_text()).items() if k != "clips"} if gate.exists() else None
    res.update(model=mcfg.name, family=data.name, model_config_hash=config_hash(mcfg))
    (out_dir / "outputs.json").write_text(json.dumps(res, indent=1, default=float))
    rows = [f"# Output-level evaluation — {mcfg.name} on {data.name}", "",
            f"Gate: {res['gate']}", "",
            "| legal twin | generated measurable | drop reasons | real measurable | r_p gen / floor | r_E gen / floor | E_out gen / floor | S_outcome [95% CI] |",
            "|---|---|---|---|---|---|---|---|"]
    for t, d in res["twins"].items():
        f = lambda kk: f"{d.get(kk + '_generated', float('nan')):.3f} / {d.get(kk + '_floor', float('nan')):.3f}"
        ci = d.get("S_outcome_ci95")
        rows.append(f"| {t} | {d['measurable_generated']}/{d['n_generated']} | {d['drop_reasons_generated']} | "
                    f"{d['measurable_real']}/{d['n_real']} | {f('r_p')} | {f('r_E')} | {f('E_out')} | "
                    f"{d.get('S_outcome', float('nan')):+.2f}" + (f" [{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "") + " |")
    (out_dir / "outputs.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
