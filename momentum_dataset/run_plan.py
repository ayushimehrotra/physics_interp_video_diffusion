#!/usr/bin/env python3
"""Execute a plan.json from plan_conservation_set.py / plan_energy_voe_set.py, in parallel, resumably.

    /workspace/kubric/.venv/bin/python run_plan.py conservation_set_v1/plan.json --workers 3

A clip is done when it has _COMPLETE (rendered) or _DROPPED (the generator's
relative in-frame rule). Re-running regenerates only what is missing. Use
--workers 3, not 5 (see CLAUDE.md). Writes manifest.json beside the plan.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger("run_plan")
HERE = Path(__file__).resolve().parent


def is_done(d: Path) -> bool:
    return ((d / "_COMPLETE").exists() and (d / "labels.json").exists()) or (d / "_DROPPED").exists()


def run_one(job: dict, root: Path, timeout: int, python: str, no_render: bool) -> dict:
    d = root / job["clip_id"]
    cmd = [python, str(HERE / "generate_scene.py"), "--seed", str(job["seed"]),
           "--output-dir", str(d), "--scene-id", job["clip_id"], *job["args"]]
    if no_render:
        cmd.append("--no-render")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {**job, "status": "timeout", "seconds": time.time() - t0}
    status = "ok" if proc.returncode == 0 and (d / "_COMPLETE").exists() else \
        ("dropped" if (d / "_DROPPED").exists() else "failed")
    if status == "failed":
        d.mkdir(parents=True, exist_ok=True)
        (d / "_STDERR").write_text(proc.stderr[-20000:])
    return {**job, "status": status, "seconds": time.time() - t0, "returncode": proc.returncode,
            "stderr_tail": proc.stderr.strip()[-600:] if status == "failed" else ""}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("plan", type=Path)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s: %(message)s")
    plan = json.loads(args.plan.read_text())
    root = args.plan.parent
    todo = [j for j in plan["jobs"] if not is_done(root / j["clip_id"])]
    if args.limit:
        todo = todo[: args.limit]
    logger.info("%s: %d jobs, %d to run (workers=%d)", plan["set"], len(plan["jobs"]), len(todo), args.workers)
    results, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, r in enumerate(pool.map(lambda j: run_one(j, root, args.timeout, args.python, args.no_render), todo), 1):
            results.append(r)
            rate = (time.time() - t0) / n
            logger.info("[%d/%d] %s %s in %.0fs (~%.0f min left)", n, len(todo), r["clip_id"], r["status"],
                        r["seconds"], rate * (len(todo) - n) / 60)
    done = [j["clip_id"] for j in plan["jobs"] if (root / j["clip_id"] / "_COMPLETE").exists()]
    dropped = [j["clip_id"] for j in plan["jobs"] if (root / j["clip_id"] / "_DROPPED").exists()]
    failures = [r for r in results if r["status"] in ("failed", "timeout")]
    manifest = {"set": plan["set"], "kind": plan["kind"], "num_jobs": len(plan["jobs"]), "complete": done,
                "dropped": dropped, "failed": failures, "wall_seconds": time.time() - t0}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    logger.info("done: %d complete, %d dropped, %d failed", len(done), len(dropped), len(failures))
    for f in failures:
        logger.error("%s failed (rc=%s): %s", f["clip_id"], f.get("returncode"), f.get("stderr_tail", "")[-300:])
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
