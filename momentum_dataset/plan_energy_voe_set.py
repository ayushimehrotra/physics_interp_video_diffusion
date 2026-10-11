#!/usr/bin/env python3
"""Plan the restitution-arm sets: energy_voe_450, energy_ladder_v1, scene_ladder_v1 (§54b, §56, §58).

    # energy_voe_450: 150 seeds x {sub, anchor, super}, anchor e sampled, sub = e - 0.2, super = e + 0.3
    python3 plan_energy_voe_set.py --mode three-arm --output-dir energy_voe_450 --num-seeds 150 --seed-offset 101

    # energy_ladder_v1: 20 seeds from 9500 (disjoint from the conservation set), anchor e = 1, 11 arms
    python3 plan_energy_voe_set.py --mode ladder --output-dir energy_ladder_v1 --num-seeds 20 --seed-offset 9500

    # scene_ladder_v1: every conservation seed's own ladder, plus its violating twin re-rendered in the same batch
    python3 plan_energy_voe_set.py --from-conservation-plan conservation_set_v1/plan.json --output-dir scene_ladder_v1

Every arm of a seed runs the same seed with the same pinned restitution, so the
simulated collision — and the whole pre-contact segment — is identical; the arm
is an edit of the aftermath (--edit-restitution E). The honest clip is the arm
whose E equals the pinned e and carries no edit. The generator drops an arm that
leaves the frame earlier than its honest twin (the relative in-frame rule).

The three-arm chord position is fixed by physics at (e_a - e_sub)/(e_sup - e_sub);
with the defaults that is 0.2/0.5 = 0.400 (§54b).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LEGAL = (0.15, 0.2, 1 / 3, 0.5, 2 / 3, 0.8, 1.0)
IMPOSSIBLE = (1.25, 1.5, 2.0, 3.0, 5.0)
ENERGY_LADDER = (0.2, 1 / 3, 0.5, 2 / 3, 0.8, 1.0) + IMPOSSIBLE      # 11 arms, anchor e = 1


def tag(e: float) -> str:
    return f"e{e:.4g}"


def arm_job(seed: int, e_sim: float, e_arm: float, arm: str) -> dict:
    args = ["--restitution", f"{e_sim}", "--seed-group", str(seed), "--arm", arm]
    if abs(e_arm - e_sim) > 1e-9:
        args += ["--edit-restitution", f"{e_arm}"]
    return {"clip_id": f"s{seed:05d}_{arm}", "seed": seed, "arm": arm, "restitution": e_arm,
            "restitution_simulated": e_sim, "args": args}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mode", choices=("three-arm", "ladder"), default=None)
    p.add_argument("--from-conservation-plan", type=Path, default=None)
    p.add_argument("--num-seeds", type=int, default=150)
    p.add_argument("--seed-offset", type=int, default=101)
    p.add_argument("--sub", type=float, default=-0.2, help="three-arm: e offset of the sub arm")
    p.add_argument("--super", dest="sup", type=float, default=0.3, help="three-arm: e offset of the super arm")
    p.add_argument("--anchor-range", type=float, nargs=2, default=(0.25, 0.7),
                   help="three-arm: anchor e is drawn uniformly here (keeps sub >= 0.05)")
    p.add_argument("--anchor", type=float, default=1.0, help="ladder: the pinned simulated e")
    p.add_argument("--arms", type=float, nargs="+", default=None, help="restitution arms (default per mode)")
    p.add_argument("--rng-seed", type=int, default=0)
    args = p.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    jobs = []
    if args.from_conservation_plan:
        src = json.loads(args.from_conservation_plan.read_text())
        arms = args.arms or (LEGAL + IMPOSSIBLE)
        for job in src["jobs"]:
            seed, e_sim = job["seed"], job["restitution"]
            if job["arm"] == "violating":
                jobs.append({**job, "clip_id": f"s{seed:05d}_violating"})   # identical args: same render
                continue
            for e in arms:
                arm = "honest" if abs(e - e_sim) < 1e-9 else tag(e)
                jobs.append(arm_job(seed, e_sim, e, arm))
            if not any(abs(e - e_sim) < 1e-9 for e in arms):
                raise SystemExit(f"seed {seed}: pinned e={e_sim} is not one of the arms; the honest clip must be an arm")
        kind = "scene-ladder"
    elif args.mode == "three-arm":
        rng = np.random.default_rng(args.rng_seed)
        for i in range(args.num_seeds):
            seed = args.seed_offset + i
            e = float(rng.uniform(*args.anchor_range))
            jobs += [arm_job(seed, e, e + args.sub, "sub"), arm_job(seed, e, e, "anchor"),
                     arm_job(seed, e, e + args.sup, "super")]
        kind = "three-arm"
    elif args.mode == "ladder":
        arms = args.arms or ENERGY_LADDER
        for i in range(args.num_seeds):
            seed = args.seed_offset + i
            for e in arms:
                jobs.append(arm_job(seed, args.anchor, e, "honest" if abs(e - args.anchor) < 1e-9 else tag(e)))
        kind = "ladder"
    else:
        p.error("pass --mode or --from-conservation-plan")
    plan = {"set": root.name, "kind": kind, "seed_offset": args.seed_offset, "jobs": jobs,
            "chord_position_physics": (-args.sub) / (args.sup - args.sub) if kind == "three-arm" else None}
    (root / "plan.json").write_text(json.dumps(plan, indent=1))
    seeds = {j["seed"] for j in jobs}
    print(f"{len(jobs)} jobs for {len(seeds)} seeds ({kind}) -> {root / 'plan.json'}")


if __name__ == "__main__":
    main()
