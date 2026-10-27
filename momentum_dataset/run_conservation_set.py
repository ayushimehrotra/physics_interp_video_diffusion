"""Render `conservation_set_v1` from the manifest written by plan_conservation_set.py.

Resumable: a clip with `_COMPLETE` is skipped, so re-running finishes a partial set.
Failures are retried once -- the observed failure mode on this box is a
transient OptiX allocation spike during BVH build when several Blender processes
start at once, which succeeds on its own.

Clips render into local disk (STAGE) and are copied to the dataset directory by a
separate thread pool, `_COMPLETE` last. Writing the 98 PNGs straight to /workspace
(geesefs) stalled each Blender process on FUSE for minutes once the volume's latency
rose: 49 s of rendering became ~400 s per clip, with the GPU idle.

    python momentum_dataset/run_conservation_set.py --workers 4

Also renders `ke_factor_set` (plan_ke_set.py): its clips carry full `params` and are
replayed through generate_ke_scene.py instead of being sampled from a seed.
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEN = HERE / "generate_scene.py"
STAGE = Path("/root/render_stage")


def stage_dir(root: Path, name: str) -> Path:
    return STAGE / root.name / name


def publish(root: Path, name: str) -> None:
    """Copy a finished clip from local staging into the dataset, `_COMPLETE` last."""
    src, dst = stage_dir(root, name), root / name
    if dst.exists():
        shutil.rmtree(dst)
    # copyfile, not copy2/copytree: geesefs refuses chmod, so copystat raises
    # "Operation not permitted" after the bytes are already copied
    for f in sorted(src.rglob("*")):
        if f.is_file() and f.name != "_COMPLETE":
            (dst / f.relative_to(src)).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dst / f.relative_to(src))
    shutil.copyfile(src / "_COMPLETE", dst / "_COMPLETE")
    shutil.rmtree(src)


def render(clip, root: Path, python: str) -> tuple[str, bool, float]:
    out = stage_dir(root, clip["name"])
    if (root / clip["name"] / "_COMPLETE").exists():
        return clip["name"], True, 0.0
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if "params" in clip:            # ke_factor_set: fully specified, replayed rather than sampled
        cmd = [python, str(HERE / "generate_ke_scene.py"), "--plan", str((root / "plan.json").resolve()),
               "--clip", clip["name"], "--output-dir", str(out)]
    else:
        cmd = [python, str(GEN), "--seed", str(clip["seed"]), "--output-dir", str(out),
               "--scene-id", clip["name"], "--arm", "legacy",
               "--material-pair", clip["material_pair"],
               "--restitution", repr(clip["restitution"])]
        if clip["violate"]:
            cmd += ["--violate", "--implied-ratio", repr(clip["implied_ratio"])]
    t = time.time()
    subprocess.run(cmd, capture_output=True, text=True, cwd=out.parent)
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
    copier = ThreadPoolExecutor(max_workers=8)
    copies = []
    for c in clips:                                  # staged by an earlier run, never copied
        if (stage_dir(root, c["name"]) / "_COMPLETE").exists() and \
                not (root / c["name"] / "_COMPLETE").exists():
            copies.append(copier.submit(publish, root, c["name"]))
    pending = {c["name"] for c in clips if (stage_dir(root, c["name"]) / "_COMPLETE").exists()}
    todo = [c for c in clips
            if not (root / c["name"] / "_COMPLETE").exists() and c["name"] not in pending]
    print(f"{len(clips)} clips, {len(clips) - len(todo)} already done, {len(todo)} to render",
          flush=True)

    done, failed, t0 = 0, [], time.time()
    for attempt in ("", "retry "):
        batch = todo if not attempt else [c for c in clips if c["name"] in failed]
        if attempt and batch:
            print(f"  retrying {len(batch)} failed clips", flush=True)
        failed, done, t0 = [], 0, time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for name, ok, dt in pool.map(lambda c: render(c, root, args.python), batch):
                done += 1
                if ok:
                    if not (root / name / "_COMPLETE").exists():
                        copies.append(copier.submit(publish, root, name))
                else:
                    failed.append(name)
                rate = (time.time() - t0) / done
                print(f"  [{attempt}{done}/{len(batch)}] {'ok ' if ok else 'FAIL'} {name:52s} "
                      f"{dt:6.0f}s   eta {(len(batch) - done) * rate / 60:5.0f} min", flush=True)
    print(f"  waiting for {sum(not f.done() for f in copies)} copies to /workspace", flush=True)
    for f in copies:
        f.result()
    copier.shutdown()
    n_ok = sum(1 for c in clips if (root / c["name"] / "_COMPLETE").exists())
    print(f"\n{n_ok}/{len(clips)} complete in {(time.time() - t0) / 60:.0f} min", flush=True)
    if failed:
        print("STILL FAILED:", failed, flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
