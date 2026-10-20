"""Render `conservation_set_v1` from the manifest written by plan_conservation_set.py.

Resumable: a clip with `_COMPLETE` is skipped, so re-running finishes a partial set.
Failures are retried once SERIALLY -- the observed failure mode on this box is a
transient OptiX allocation spike during BVH build when several Blender processes
start at once, which succeeds on its own.

    python momentum_dataset/run_conservation_set.py --workers 4
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
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
    cmd = [python, str(GEN), "--seed", str(clip["seed"]), "--output-dir", str(out),
           "--scene-id", clip["name"], "--arm", "legacy",
           "--material-pair", clip["material_pair"],
           "--restitution", repr(clip["restitution"])]
    if clip["violate"]:
        cmd += ["--violate", "--implied-ratio", repr(clip["implied_ratio"])]
    t = time.time()
    subprocess.run(cmd, capture_output=True, text=True)
    return clip["name"], (out / "_COMPLETE").exists(), time.time() - t


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default="momentum_dataset/conservation_set_v1/plan.json")
    ap.add_argument("--workers", type=int, default=4,
                    help="8 concurrent Blender processes exhausted a 32 GB GPU here; "
                         "4 is the tested ceiling")
    ap.add_argument("--python", default="/workspace/kubric/.venv/bin/python")
    args = ap.parse_args()

    plan = json.load(open(args.plan))
    root = Path(args.plan).parent
    clips = plan["clips"]
    todo = [c for c in clips if not (root / c["name"] / "_COMPLETE").exists()]
    print(f"{len(clips)} clips, {len(clips) - len(todo)} already done, {len(todo)} to render",
          flush=True)

    done, failed, t0, times = 0, [], time.time(), []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for name, ok, dt in pool.map(lambda c: render(c, root, args.python), todo):
            done += 1
            if ok:
                times.append(dt)
            else:
                failed.append(name)
            rate = (time.time() - t0) / done
            print(f"  [{done}/{len(todo)}] {'ok ' if ok else 'FAIL'} {name:52s} "
                  f"{dt:6.0f}s   eta {(len(todo) - done) * rate / 60:5.0f} min", flush=True)

    for name in list(failed):                       # serial retry
        clip = next(c for c in clips if c["name"] == name)
        print(f"  retrying {name} serially", flush=True)
        _, ok, dt = render(clip, root, args.python)
        if ok:
            failed.remove(name)
    n_ok = sum(1 for c in clips if (root / c["name"] / "_COMPLETE").exists())
    print(f"\n{n_ok}/{len(clips)} complete in {(time.time() - t0) / 60:.0f} min", flush=True)
    if failed:
        print("STILL FAILED:", failed, flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
