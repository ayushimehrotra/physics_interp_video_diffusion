"""Render `energy_voe_v1` from the manifest written by plan_energy_voe_set.py.

Same contract as run_conservation_set.py: resumable (a clip with `_COMPLETE` is
skipped), failures retried once serially, workers capped because concurrent Blender
processes exhaust GPU memory.

    python momentum_dataset/run_energy_voe_set.py --workers 4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEN = HERE / "generate_scene.py"


def render(clip, root: Path, python: str) -> tuple[str, bool, float]:
    out = root / clip["name"]
    if (out / "_COMPLETE").exists():
        return clip["name"], True, 0.0
    if out.exists():
        subprocess.run(["rm", "-rf", str(out)], check=False)
    cmd = [python, str(GEN), "--seed", str(clip["seed"]),
           "--output-dir", str(out), "--scene-id", clip["name"],
           "--arm", clip["arm"], "--material-pair", clip["material_pair"],
           "--restitution", repr(clip["restitution"]),
           "--energy-restitution", repr(clip["energy_restitution"])]
    t = time.time()
    subprocess.run(cmd, capture_output=True, text=True)
    return clip["name"], (out / "_COMPLETE").exists(), time.time() - t


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default="momentum_dataset/energy_voe_v1/plan.json")
    ap.add_argument("--workers", type=int, default=4,
                    help="8 concurrent Blender processes exhausted a 32 GB GPU on the "
                         "conservation set; 4 is the tested ceiling")
    ap.add_argument("--python", default="/workspace/kubric/.venv/bin/python")
    args = ap.parse_args()

    plan = json.load(open(args.plan))
    root = Path(args.plan).parent
    clips = plan["clips"]
    todo = [c for c in clips if not (root / c["name"] / "_COMPLETE").exists()]
    print(f"{len(clips)} clips, {len(clips) - len(todo)} already done, {len(todo)} to render")

    t0 = time.time()
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (name, ok, dt) in enumerate(
                pool.map(lambda c: render(c, root, args.python), todo), 1):
            if not ok:
                failed.append(name)
            print(f"  [{i}/{len(todo)}] {name} {'ok' if ok else 'FAILED'} {dt:.0f}s",
                  flush=True)

    if failed:
        print(f"\nretrying {len(failed)} failures serially")
        still = []
        for name in failed:
            clip = next(c for c in clips if c["name"] == name)
            _, ok, dt = render(clip, root, args.python)
            print(f"  {name} {'ok' if ok else 'FAILED AGAIN'} {dt:.0f}s", flush=True)
            if not ok:
                still.append(name)
        failed = still

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min; {len(failed)} unrecovered")
    for name in failed:
        print(f"  FAILED {name}")


if __name__ == "__main__":
    main()
