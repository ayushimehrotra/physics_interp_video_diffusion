#!/usr/bin/env python
"""Run one experiment config. This is the only script an experiment needs.

Single GPU:

    .venv/bin/python scripts/run_experiment.py configs/red_ball_smoke.yaml

All GPUs on the box (the wrapper picks the count):

    ./scripts/run_experiment.sh configs/red_ball_smoke.yaml

Every run writes its own directory under ``output_dir/<experiment>/`` and a
``DONE`` marker when complete, so re-running a config only executes what is
missing. That makes an interrupted sweep resumable and makes it safe to hand a
collaborator a config and say "just run this until it finishes".
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

# Allow running from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from physics_interp import (  # noqa: E402
    VideoDiffusionRunner,
    expand_grid,
    get_dist_info,
    load_config,
    shard,
)
from physics_interp.video import write_json  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run a video diffusion experiment grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("config", type=Path, help="path to an experiment YAML")
    p.add_argument(
        "-o", "--output-dir", default=None,
        help="override output_dir from the config",
    )
    p.add_argument(
        "--device", default=None,
        help="override device (default: cuda:LOCAL_RANK)",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="run at most this many specs on this rank (smoke tests)",
    )
    p.add_argument(
        "--no-resume", action="store_true",
        help="re-run specs that already have a DONE marker",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="print the run plan and exit without loading the model",
    )
    p.add_argument(
        "--list-sites", action="store_true",
        help="load the model, print every hookable site name, and exit",
    )
    p.add_argument(
        "--keep-going", action="store_true",
        help="continue past a failing run instead of aborting the rank",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.no_resume:
        cfg.resume = False

    info = get_dist_info(args.device)
    specs = expand_grid(cfg)
    mine = shard(specs, info)

    if cfg.resume:
        todo = [s for s in mine if not s.is_done(cfg.output_dir)]
        skipped = len(mine) - len(todo)
    else:
        todo, skipped = mine, 0
    if args.limit is not None:
        todo = todo[: args.limit]

    if info.is_main:
        print(
            f"experiment '{cfg.name}': {len(cfg.prompts)} prompt(s) x "
            f"{len(cfg.seeds)} seed(s) = {len(specs)} run(s) across "
            f"{info.world_size} process(es) -> {cfg.output_dir}",
            flush=True,
        )
    info.log(f"assigned {len(mine)}, skipping {skipped} done, running {len(todo)}")

    if args.dry_run:
        for s in todo:
            info.log(f"  would run {s.name}: {s.prompt.text[:70]!r}")
        return 0
    if not todo and not args.list_sites:
        info.log("nothing to do")
        return 0

    info.log(f"loading {cfg.model.id} ...")
    runner = VideoDiffusionRunner.from_config(cfg, device=info.device)
    info.log(f"loaded: {runner.describe()}")

    if args.list_sites:
        if info.is_main:
            for name in runner.sites:
                print(name)
        return 0

    failures: list[tuple[str, str]] = []
    try:
        for i, spec in enumerate(todo, 1):
            info.log(f"[{i}/{len(todo)}] {spec.name} :: {spec.prompt.text[:60]!r}")
            try:
                out = runner.run(spec, cfg.output_dir)
                info.log(f"    -> {out}")
            except Exception:
                failures.append((spec.name, traceback.format_exc()))
                info.log(f"    FAILED {spec.name}:\n{traceback.format_exc()}")
                if not args.keep_going:
                    raise
    finally:
        runner.close()

    if failures:
        # A failure report per rank, so a sweep's damage is auditable after the
        # fact rather than lost in interleaved stdout.
        report = Path(cfg.output_dir) / cfg.name / f"failures_rank{info.rank}.json"
        write_json([{"run": n, "traceback": tb} for n, tb in failures], report)
        info.log(f"{len(failures)} run(s) failed; report written to {report}")
        return 1

    info.log(f"done: {len(todo)} run(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
