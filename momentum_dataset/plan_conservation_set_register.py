#!/usr/bin/env python3
"""Plan conservation_set_v1: honest/violating twins at pinned restitution (register Part 1, §55–§58).

    python3 plan_conservation_set_register.py --output-dir conservation_set_v1 --num-seeds 150 --seed-offset 101
    python3 run_plan.py conservation_set_v1/plan.json --workers 3

150 seeds from 101, alternating e = 1.00 (elastic: honest clips conserve energy
exactly) and e = 0.15 (inelastic: both classes lose the same energy), each
rendered twice with the same seed: honest, and the split-changing violation
(--violate --violation-mode split). The two twins share a bit-identical
pre-contact segment. Nothing else about the probe_set sampler changes.

Writes plan.json (one job per clip) and nothing else; run_plan.py executes it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def clip_id(seed: int, arm: str) -> str:
    return f"s{seed:05d}_{arm}"


def jobs_for_seed(seed: int, e: float) -> list[dict]:
    common = ["--restitution", f"{e}", "--seed-group", str(seed)]
    return [
        {"clip_id": clip_id(seed, "honest"), "seed": seed, "arm": "honest", "restitution": e,
         "args": common + ["--arm-tag", "honest"]},
        {"clip_id": clip_id(seed, "violating"), "seed": seed, "arm": "violating", "restitution": e,
         "args": common + ["--violate", "--violation-mode", "split", "--arm-tag", "violating"]},
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-seeds", type=int, default=150)
    p.add_argument("--seed-offset", type=int, default=101)
    p.add_argument("--restitutions", type=float, nargs="+", default=[1.0, 0.15],
                   help="pinned restitutions, assigned to seeds round-robin")
    args = p.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i in range(args.num_seeds):
        seed = args.seed_offset + i
        jobs += jobs_for_seed(seed, args.restitutions[i % len(args.restitutions)])
    plan = {"set": root.name, "kind": "conservation", "seed_offset": args.seed_offset,
            "num_seeds": args.num_seeds, "restitutions": args.restitutions, "violation_mode": "split",
            "jobs": jobs}
    (root / "plan.json").write_text(json.dumps(plan, indent=1))
    print(f"{len(jobs)} jobs for {args.num_seeds} seeds -> {root / 'plan.json'}")


if __name__ == "__main__":
    main()
