#!/usr/bin/env python3
"""Generate N collision scenes in parallel, resuming whatever is already done.

    python3 run_batch.py --num-scenes 50 --output-dir pilot --workers 5

Each scene is an independent subprocess running generate_scene.py, so a crashed or
killed scene costs only itself. Re-running the same command regenerates only what is
missing: a scene counts as done when it has a `_COMPLETE` marker.

Scene i always uses seed (--seed-offset + i), so the dataset is reproducible and can be
extended later without renumbering.
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

logger = logging.getLogger("run_batch")

HERE = Path(__file__).resolve().parent


def scene_dir(root: Path, index: int) -> Path:
  return root / f"scene_{index:05d}"


def is_complete(directory: Path) -> bool:
  return (directory / "_COMPLETE").exists() and (directory / "labels.json").exists()


def is_violation(index: int, violate_frac: float) -> bool:
  """Whether scene ``index`` breaks momentum conservation.

  Deterministic in the index and interleaved rather than blocked, so any prefix
  of the dataset is already close to the requested class balance — a run killed
  half way through still gives a usable two-class set instead of one class.
  """
  if violate_frac <= 0:
    return False
  return int((index + 1) * violate_frac) > int(index * violate_frac)


def run_one(index: int, root: Path, seed_offset: int, timeout: int, python: str,
            no_render: bool = False, violate_frac: float = 0.0) -> dict:
  directory = scene_dir(root, index)
  seed = seed_offset + index
  started = time.time()
  violate = is_violation(index, violate_frac)
  cmd = [python, str(HERE / "generate_scene.py"),
         "--seed", str(seed),
         "--output-dir", str(directory),
         "--scene-id", directory.name]
  if no_render:
    cmd.append("--no-render")
  if violate:
    cmd.append("--violate")
  try:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
  except subprocess.TimeoutExpired:
    return {"index": index, "seed": seed, "status": "timeout", "seconds": time.time() - started}

  elapsed = time.time() - started
  if proc.returncode == 0 and is_complete(directory):
    return {"index": index, "seed": seed, "status": "ok", "seconds": elapsed}
  (directory).mkdir(parents=True, exist_ok=True)
  (directory / "_STDERR").write_text(proc.stderr[-20000:])
  return {"index": index, "seed": seed, "status": "failed", "seconds": elapsed,
          "returncode": proc.returncode, "stderr_tail": proc.stderr.strip()[-800:]}


def main():
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--num-scenes", type=int, required=True)
  parser.add_argument("--output-dir", type=str, required=True)
  parser.add_argument("--workers", type=int, default=4,
                      help="concurrent scenes; each holds ~0.5GB of VRAM (default: 4)")
  parser.add_argument("--seed-offset", type=int, default=0)
  parser.add_argument("--timeout", type=int, default=3600, help="per-scene seconds")
  parser.add_argument("--python", type=str, default=sys.executable)
  parser.add_argument("--force", action="store_true", help="regenerate scenes already complete")
  parser.add_argument("--no-render", action="store_true",
                      help="physics and labels only, no images. Seconds per scene "
                           "instead of minutes — use it to check label coverage "
                           "and confounds before committing to a full render")
  parser.add_argument("--violate-frac", type=float, default=0.0,
                      help="fraction of scenes whose post-contact motion breaks "
                           "momentum conservation, interleaved. 0.5 gives the "
                           "balanced two-class set a plausibility probe needs")
  args = parser.parse_args()

  logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s: %(message)s")
  root = Path(args.output_dir)
  root.mkdir(parents=True, exist_ok=True)

  todo, done = [], []
  for i in range(args.num_scenes):
    (done if (is_complete(scene_dir(root, i)) and not args.force) else todo).append(i)
  logger.info("%d scenes requested: %d already complete, %d to generate (workers=%d)",
              args.num_scenes, len(done), len(todo), args.workers)

  results = []
  started = time.time()
  if todo:
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
      futures = [pool.submit(run_one, i, root, args.seed_offset, args.timeout,
                             args.python, args.no_render, args.violate_frac)
                 for i in todo]
      for n, future in enumerate(futures, start=1):
        result = future.result()
        results.append(result)
        rate = (time.time() - started) / n
        logger.info("[%d/%d] scene_%05d %s in %.0fs (avg %.0fs/scene, ~%.0f min left)",
                    n, len(todo), result["index"], result["status"], result["seconds"],
                    rate, rate * (len(todo) - n) / 60)

  failures = [r for r in results if r["status"] != "ok"]
  manifest = {
      "output_dir": str(root),
      "num_scenes": args.num_scenes,
      "seed_offset": args.seed_offset,
      "already_complete": len(done),
      "generated": len(results) - len(failures),
      "failed": failures,
      "wall_seconds": time.time() - started,
      "scenes": [scene_dir(root, i).name for i in range(args.num_scenes)
                 if is_complete(scene_dir(root, i))],
  }
  (root / "manifest.json").write_text(json.dumps(manifest, indent=1))

  logger.info("done: %d/%d scenes complete in %.1f min",
              len(manifest["scenes"]), args.num_scenes, manifest["wall_seconds"] / 60)
  for failure in failures:
    logger.error("scene_%05d failed (rc=%s): %s", failure["index"],
                 failure.get("returncode"), failure.get("stderr_tail", "")[-300:])
  return 1 if failures else 0


if __name__ == "__main__":
  sys.exit(main())
